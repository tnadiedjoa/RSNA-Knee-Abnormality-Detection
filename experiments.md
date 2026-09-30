# Experiment log

Every change tested on the project gets an entry here, kept or not: what changed, the commit,
the result before and after (same conditions), and the conclusion. Scores are cross-validated
on our own folds unless marked **LB** (Kaggle public leaderboard):

- **silver AUC** — macro AUC against the confident report labels (grade absent vs moderate/severe),
  out-of-fold over all ~4,400 training studies;
- **gold AUC** — macro AUC against the 58 expert-labelled studies, out-of-fold (each gold study is
  scored by the fold that did not train on it). Only 58 studies: ±0.03 is noise.

## Summary

| ID | Date | Change | Result | Kept |
|----|------|--------|--------|------|
| E00 | 09-30 | Remote-zip cache, canonical frames, study-level laterality, JPEG cache, Kaggle chain | train/test parity byte-identical | ✓ |
| E01 | 09-30 | Local LLM report labels (Qwen3.6-27B), synovitis-from-effusion prior | gold agreement 0.873 | ✓ |
| E02 | 09-30 | ResNet34 2.5D, 5 series, 224 px, 8 windows | silver 0.894, gold 0.806; **LB 0.879** | ✓ (reference) |
| E03 | 09-30 | Soft bootstrapping with E02 OOF (α=0.5) | silver 0.894→0.917, gold 0.806→0.865 (possible leak, see E05); **LB 0.882** | ✓ |
| E04 | 09-30 | 256 px, 10 windows | folds 0–1: silver 0.897→0.896 | ✗ |
| E05 | 09-30 | Leak-free nested check of E03 | fold 0 silver 0.895→0.908 (E03 showed 0.922) | ✓ gain real, ~half |

## Phase 1 — new pipeline (from 2026-09-30)

The legacy pipeline below could not be submitted as such: the test set has **no reports**, so the
text model (and the 0.628 text+image blend) has no input at test time, and the image model had
only 58 labelled studies. Its code is kept in `legacy/`.

### E00 — Infrastructure (2026-09-30, night)

- **Data access without storage.** The competition archive is 265 GB zipped (570 GB of DICOMs) and
  the home quota is 100 GB. `knee/remote_zip.py` reads the zip's central directory over HTTP range
  requests (signed Kaggle URL) and fetches one series per request; `scripts/build_cache.py` turns
  every series into the cache directly (3 CPU jobs, ~40 min for 24,386 series, 0 errors).
- **Canonical frames** (`knee/dicom_io.py`): slices sorted by physical position, in-plane axes
  re-oriented to the radiological display of the plane, 150 mm centre crop, 256 px, p1–p99.5
  window. Every knee is mirrored into a right knee (lateral on the image left / first sagittal
  slices), checked visually on posterior coronal slices (fibular head always bottom-left).
- **Laterality per study**: median knee x-position over the study's series (agrees with the DICOM
  tag on 98.8% of 11,665 tagged series), then tags; 175 series in 55 studies had been mirrored
  inconsistently series-by-series and were re-fetched. Test time decides the side the same way
  from headers first.
- **Cache format**: slices as JPEG q95 (mean error 0.9 grey level / 255), 16 GB instead of 34 GB
  as zlib arrays — needed to fit the quota. The test pipeline applies the same round trip:
  the Kaggle pipeline and the training cache were checked **byte-identical** on the 15 visible
  test series.
- **Kaggle chain** validated with a random-weight model (private dataset + private notebook,
  offline wheels for JPEG/JPEG 2000 decoders, which the Kaggle image lacks): 0.7 min for the
  3 visible studies.

### E01 — Report labels with a local LLM (2026-09-30)

Qwen3.6-27B (Intel int4 AutoRound, vLLM, one RTX 3090) grades each finding absent / equivocal /
mild / moderate / severe / not mentioned (`scripts/llm_labels.py`); grades map to fixed soft
labels (`knee/labels.py`). Reports never leave the cluster (rule 4.b). Agreement with the 58 gold
studies (macro AUC of the soft label vs gold):

| Label set | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Baker | Cont | Frac | **macro** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-27B v1 (silence prior 0.4 for synovitis) | .988 | .981 | .953 | .880 | .908 | .832 | .871 | .782 | .652 | .942 | .787 | .799 | 0.865 |
| same, silence prior 0.1 | | | | | | | | | .711 | | | | 0.869 |
| same + unmentioned synovitis from effusion | | | | | | | | | .751 | | | | **0.873** |
| *reference: pilkwang v2 (hosted API — not used)* | | | | | | | | | | | | | 0.866 |
| *reference: stevenleehans v2 (provenance unknown — not used)* | | | | | | | | | | | | | 0.887 |

Weak spots are intrinsic to the reports: "small effusion" is split 13/23 between gold negatives
and positives; the annotators mark synovitis on half the studies while reports rarely mention it;
impaction injuries are "fracture" for the LLM and sometimes "contusion" for the annotators.
**Kept**: synovitis-from-effusion prior. Public tables made with hosted APIs are only a reference.

### E02 — First image model: ResNet34 2.5D, 5 series (2026-09-30)

Run `2026-09-30_0257_e02_r34_base` (3 folds on P100, 2 on the RTX 3090). Five series per study
(sagittal / coronal / axial fat-suppressed fluid-sensitive, sagittal and coronal other), 8 windows
of 3 adjacent slices per series (12 at evaluation), 224 px, ImageNet ResNet34 shared by all
windows, 2-layer transformer over the 40 tokens, one attention query per finding + max over
windows; soft BCE, unmentioned findings weighted 0.5, gold studies weight 3; 12 epochs, AdamW
2e-4 (head 1e-3), cosine. No flips (knees are canonical right knees).

| | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Baker | Cont | Frac | **macro** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| silver (OOF, 4,407 studies) | .853 | .829 | .858 | .777 | .941 | .879 | .871 | .984 | .975 | .959 | .920 | .882 | **0.894** |
| gold (OOF, 58 studies) | .737 | .789 | .875 | .611 | .969 | .799 | .743 | .940 | .739 | .920 | .810 | .744 | **0.806** |

Silver AUC per fold plateaus from epoch ~8 (0.876–0.901 at epoch 12). Weakest: the fine structures
(lateral meniscus, MCL, ACL, medial meniscus). Epoch time: 8.4 min on P100, 4.3 min on the 3090
(data-bound with 4 CPUs; an unpacked copy of the cache on local disk now makes it GPU-bound).
Packaged and run offline on Kaggle (private notebook `knee-submit` v3, 5 folds); the API token
cannot create submissions (403), so the first **LB** submission is to be made from the web page.

### E03 — Soft bootstrapping: labels = ½ report + ½ out-of-fold prediction of E02 (2026-09-30)

Same model as E02; the report soft labels of the non-gold studies are averaged with E02's
out-of-fold predictions (`--bootstrap-run`, α = 0.5), which pulls the labels towards what the
images show (the gold labels are image reads, the reports are not). Folds 0–2 on P100, then
3–4 (3090) — same folds as E02:

| folds 0–2 | silver | gold (34 studies) |
|---|---|---|
| E02 (before) | 0.897 | 0.811 |
| **E03 bootstrap α=0.5** | **0.918** | **0.864** |

All 5 folds (OOF, same folds as E02): **silver 0.894 → 0.917, gold 0.806 → 0.865** (gold ACL
0.737→0.837, lateral meniscus 0.611→0.739, contusion 0.810→0.907, fracture 0.744→0.858; medial
meniscus 0.875→0.846). A rank blend E02+E03 adds nothing (silver 0.917, gold 0.850).
**Caveat — possible leak:** the E02 model that produced the pseudo-labels of fold-j studies had
been trained on fold-0 studies, so fold-0 validation labels reach fold-0 training targets
indirectly. E05 measures the gain without that path. Kept provisionally.

**Public LB: 0.882** (notebook `knee-submit` v4, 5 folds, ~33 min on Kaggle for the hidden test set)
vs **0.879** for E02 (v3): +0.003, within LB noise — far less than the out-of-fold gains
(silver +0.023, gold +0.059, leak-free silver +0.013). Neither the report labels (silver) nor the
58 gold studies track the LB closely: E02 gold 0.806 → LB 0.879, E03 gold 0.865 → LB 0.882.

### E04 — 256 px and 10 windows per series (2026-09-30)

Same as E02 but `--img-size 256 --windows 10 --windows-eval 14` (native cache resolution),
folds 0–1 on the RTX 3090: silver 0.896 vs 0.897 (E02, same folds), gold 0.845 vs 0.809 on 23
studies (noise level). Fracture and ACL up, MCL and contusion down; 1.6× the compute.
**Not kept** for now; worth retrying on top of better labels.

### E05 — Leak-free check of E03 (2026-09-30)

Nested CV around fold 0: four teachers trained without fold 0 (`--exclude-folds 0`, run
`2026-09-30_0639_e05_inner_excl0`, each validated on one of folds 1–4) give out-of-fold labels
for folds 1–4 that never saw fold 0; the student (`2026-09-30_0639_e05_student_f0`, α = 0.5) is
evaluated on fold 0.

| fold 0 | silver (882 studies) | gold (11 studies) |
|---|---|---|
| E02, report labels | 0.8946 | 0.844 |
| **E05 student, leak-free bootstrap** | **0.9078** | 0.826 |
| E03, bootstrap with the leak | 0.9223 | 0.874 |

The gain is real but about **half** of what E03 showed (+0.013 instead of +0.028 silver); E03's
out-of-fold scores are optimistic. Gold on 11 studies is noise. **Kept** (the leak only inflates
the CV, not the submitted model); future comparisons use fold 0 with the E05 teachers' labels,
which is leak-free.
