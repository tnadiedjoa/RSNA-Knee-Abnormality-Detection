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
| E06 | 09-30 | Backbone / slice density / span, fold 0 | ConvNeXt-tiny 0.908→0.931; span, density, CoAtNet-0 no gain | ✓ ConvNeXt |
| E07 | 09-30 | ConvNeXt-tiny, 5 folds, E03 labels | silver 0.917→0.943, gold 0.865→0.888 (vs E03, same leak); **LB 0.882→0.918** | ✓ |
| E08 | 09-30 | On ConvNeXt, fold 0: 130 mm FOV / 256 px / ConvNeXt-small / lr 1e-4 | FOV +0.012, others ≤ +0.006 | ✓ FOV (not confirmed by E09) |
| E09 | 09-30 | ConvNeXt + 130 mm FOV, 5 folds | silver 0.9415 (E07 0.9429); blend E07+E09 0.9474, **LB 0.919** | ✓ as ensemble member |
| E10 | 09-30 | Fold 0: 120 mm FOV, ConvNeXt-small + 130 mm; seed reruns | no stacking; seed reruns cancelled at epoch 6 | ✗ |
| E11 | 09-30 | ConvNeXt-tiny 256 px, 5 folds | silver 0.9435, gold 0.895; blend E07+E09+E11+E12 0.9504 | ✓ as ensemble member |
| E12 | 09-30 | E07 with seed 1, 5 folds (seed noise, ensemble member) | silver 0.9422 (E07 0.9429): 5-fold seed noise ~0.001; blend E07+E12 0.9479 | ✓ as ensemble member |
| E13 | 09-30 | E07 recipe at 384 px on a 384 px cache (0.39 mm/px) | silver **0.9459** (224/256 px: 0.9415–0.9435), gold 0.901; blend E07+E09+E11+E12+E13 **0.9517**, **LB 0.922** | ✓ |
| E14 | 10-01 | DINOv2 ViT-S/14 backbone (another family), E07 recipe, 5 folds on P100 | epoch 3: silver 0.79 / 0.70 (ConvNeXt ~0.91); cancelled | ✗ |
| E15 | 10-01 | EfficientNetV2-S (IN-21k, another CNN family), E07 recipe, 5 folds on P100 | silver 0.9334, gold 0.872; lowers every blend it joins | ✗ |
| E16 | 10-01 | E13 with seed 1 (384 px) | silver **0.9461**, gold 0.906 (E13 0.9459 / 0.901): 384 px holds across seeds; 6-model blend 0.9525 | ✓ |
| E17 | 10-01 | Inference settings (more windows, wider span), no retraining | 24 windows: +0.0005 to +0.0015 on every model; 6-model blend 0.9533 (+0.0016 [+0.0011, +0.0021] vs Version 8) | ✓ |
| E18 | 10-01 | E13 with seed 2 + weight EMA 0.998 | silver 0.9440 (EMA 0.9440), gold 0.904; 7-model blend 0.9529 at 12 windows | ✓ member, EMA ✗ |
| E19 | 10-01 | 384 px, 12 training windows (batch 6), 24 at evaluation, seed 3, EMA | 5 folds at 24 eval windows: silver 0.9480, gold 0.908 (E13 0.9471 / 0.904, E16 0.9477 / 0.905); V9 set + E19: 0.9542 / 0.914 (V9 0.9536 / 0.913) | ✗ recipe, + member |
| E20 | 10-01 | ResNet-50 pretrained on RadImageNet (medical images), E07 recipe, P100 | folds 0–3 silver 0.9437 / 0.9425 / 0.9439 / 0.9463 at epoch 12; fold 4 running | pending |
| E21a | 10-01 | CoAtNet-0 with the public recipe's backbone lr 3e-5 (E06 fold-0 protocol) | silver 0.9269 (E06b 0.8860, E06a ConvNeXt 0.9311) | low lr closes most of the gap → E22 |
| E22 | 10-01 | CoAtNet-0, lr 3e-5, 5 folds, E03 labels | fold 0 silver 0.9322; folds 1–4 stopped for E23 | paused |
| E23 | 10-01 | E07 recipe with round-2 labels: 0.3 report + 0.7 mean OOF of 7 runs (labels' gold agreement 0.927 vs 0.897) | silver 0.9516, gold 0.901 (E07: 0.9429 / 0.888); partly leaky, see E23 | Version 10 → LB |
| E24 | 10-01 | E13 (384 px) with the round-2 labels of E23 | silver **0.9551**, gold **0.922** (E13: 0.9459 / 0.901) — alone above the whole V9 ensemble (0.9536 / 0.913) | ✓ (LB pending) |

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

### E06 — Backbone, slice density and slice span (2026-09-30)

Leak-free protocol: fold 0 only, labels bootstrapped from the E05 teachers (α = 0.5), so every
variant compares with the E05 student (ResNet34, 224 px, 8 windows over 8–92%). One change each,
same optimiser settings (backbone lr 2e-4, 12 epochs). All on RTX 3090s (`slurm/multi_gpu.sbatch`,
cache unpacked to the node's local disk).

| Variant | Change | silver fold 0 | gold (11) | Epoch time (3090) |
|---|---|---|---|---|
| E05 student (reference) | ResNet34, 8 windows over 8–92% | 0.9078 | 0.826 | 2.1 min |
| **E06a** | **ConvNeXt-tiny (IN-22k), activation recomputation** | **0.9311** | 0.856 | 4.2 min |
| E06b | CoAtNet-0, activation recomputation | 0.8860 | 0.798 | 5.5 min |
| E06c | ResNet34, 16 windows (24 at eval) over 2–98% | 0.9081 | 0.830 | 3.4 min |
| E06d | ResNet34, 8 windows over 2–98% (span alone) | 0.9029 | 0.785 | 2.1 min |

ConvNeXt-tiny gains on every finding (MCL 0.831→0.908, medial meniscus 0.878→0.929, ACL
0.889→0.905). Neither a wider slice span nor twice the windows helps the ResNet34. CoAtNet-0 learns
slowly with the ResNet learning rate and ends below the reference; not retried for now.
**Kept**: ConvNeXt-tiny. Without activation recomputation, ConvNeXt-tiny and CoAtNet-0 do not fit
a 16 GB P100 with 8 studies × 40 windows per batch.

### E07 — ConvNeXt-tiny on all 5 folds (2026-09-30)

E06a's backbone with E03's label recipe (bootstrap from E02, α = 0.5), so the LB difference with
E03 (0.882) measures the backbone alone. Run `2026-09-30_1213_e07_convnext_5f`, 4.0–4.6 min per
epoch on RTX 3090s. Out-of-fold (same leaky recipe as E03, so comparable with it): **silver
0.917 → 0.943, gold 0.865 → 0.888**. A rank blend with E02 and E03 is worse (silver 0.933).
Packaged as notebook `knee-submit` v5 (0.8 min on the 3 visible test studies).

**Public LB: 0.918** (E03: 0.882, +0.036). The silver gain (+0.026, 95% CI ±0.003) carried over
to the LB — silver is a reliable guide for model-side changes (labels unchanged), unlike the
label-side change of E03 (silver +0.023 in-leak / +0.013 leak-free, LB +0.003).

### E08 — On top of ConvNeXt-tiny: field of view, resolution, size, learning rate (2026-09-30)

Same leak-free fold-0 protocol as E06; reference E06a (ConvNeXt-tiny, silver 0.9311). Paired
bootstrap on fold 0 (882 studies): differences below ~0.008 are noise.

| Variant | Change |
|---|---|
| E08a | field of view 150 → 130 mm (`--zoom 1.154`, 0.58 mm/px instead of 0.67 at 224 px) |
| E08b | 256 px (the cache's native 0.59 mm/px over 150 mm) |
| E08c | ConvNeXt-small (IN-22k) |
| E08d | backbone learning rate 2e-4 → 1e-4 |

Results vs E06a (0.9311), paired bootstrap 95% CI: E08a **0.9430, +0.012 [+0.007, +0.018]**;
E08b 0.9344, +0.003 [−0.002, +0.009]; E08c 0.9368 (+0.006, 1.7× the compute); E08d 0.9337,
+0.003 [−0.002, +0.008]. Kept: 130 mm field of view (but see E09).

### E09 — ConvNeXt-tiny + 130 mm field of view on 5 folds (2026-09-30)

Run `2026-09-30_1656_e09_convnext_zoom130_5f`, E03 label recipe (same as E07). Out-of-fold silver
**0.9415 vs 0.9429 for E07** (−0.001, CI [−0.003, +0.001]): the field of view gain seen on fold 0
does not hold (per fold: +0.0025, +0.001, +0.001, −0.004, −0.006). The paired bootstrap covers the
choice of studies, not training randomness: **a single-run gain of ~0.01 on fold 0 can be seed
noise** — from now on, confirm fold-0 gains with a second seed or more folds before a 5-fold run.

The two models are complementary: **rank blend E07 + E09 = 0.9474, +0.0045 [+0.003, +0.005]**,
better on every fold; gold 0.897. Submitted as notebook `knee-submit` v7 (10 networks, 1.2 min on
the 3 visible test studies); v6 (E09 alone) not submitted.

**Public LB: 0.919** (v5 / E07 alone: 0.918): +0.001, within LB noise (~±0.01). The +0.0045 silver
gain of the blend is too small to show on the public LB.

### E10 — Do the fold-0 gains stack? And how noisy is one training run? (2026-09-30)

Fold 0, leak-free labels (E05 teachers), ConvNeXt family:

| Run | Change | silver fold 0 |
|---|---|---|
| E06a (reference) | ConvNeXt-tiny, 150 mm | 0.9311 |
| E08a | + 130 mm field of view | 0.9430 |
| **E10a** | + 120 mm field of view | **0.9423** |
| E08c | ConvNeXt-small, 150 mm | 0.9368 |
| **E10b** | ConvNeXt-small + 130 mm | **0.9373** |

120 mm ≈ 130 mm; ConvNeXt-small does not stack with the tighter field of view. Given E09 (the
130 mm gain did not hold over 5 folds), E10c/E10d reran E06a and E08a with seed 1 (P100) to
measure run-to-run noise. **Cancelled at epoch 6 of 12** (21:59) to free a Slurm job slot for the
384 px cache (E13); the 5-fold seed comparison E12 vs E07 measures the noise that matters.
Partial curves in `logs/e10cd_partial_summary.txt` (epoch 6: E10c 0.9314, E10d 0.9350).

### E11 / E12 — More ensemble members (2026-09-30, running overnight)

E11: ConvNeXt-tiny at 256 px, 5 folds (4 on P100, fold 2 on the 3090: silver 0.9415 vs 0.9419 for
E07 and 0.9428 for E09 on fold 2). E12: exactly E07 with seed 1, 5 folds in sequence on the 3090.
A Slurm job (`logs/chain_e11.sh`) packages E07 + E09 + E11 and pushes notebook v8 if the 3-model
rank blend beats E07 + E09 by more than 0.001 silver.

21:55: the interactive job holding the 3090 ended (its `srun` session closed); E12 kept running
outside any allocation. After fold 2 it was stopped and folds 3–4 relaunched in the new
interactive job (`logs/handoff_e12.sh`, `logs/e12b_local.log`), same run folder.

**E12 result** (23:45): silver **0.9422**, gold 0.895 — E07 (seed 42) 0.9429, E09 0.9415. Over 5
folds, one training run moves silver by ~0.001, so a 5-fold gain above ~0.003 is real. The E07+E09
blend gain was seed averaging, not the field of view: E07+E12 (same recipe, two seeds) = **0.9479**,
E07+E09 = 0.9474, E07+E09+E12 = **0.9493** (gold 0.904). On the LB, +0.0045 silver gave +0.001.

### E13 — 384 px cache (2026-09-30, building)

The best public single model reads 384 px at ~0.35 mm/px; our cache stores 256 px over 150 mm
(0.59 mm/px). New cache `data/cache384`: same frames, 384 px (0.39 mm/px), only the 20,792 series
the model reads (`scripts/build_cache.py --size 384 --sides-from-meta --slots-only`, study sides
taken from the existing meta, so `data/series_meta.parquet` is untouched). Check on 20 series:
downsized to 256 px it matches the 256 cache within 1.3 grey levels (a mirrored side would differ
by ~30–60). Built by `slurm/build_cache384.sbatch` (3 shards, ~7 min each at 15–20 series/s):
**20,792 series, 0 errors, 25 GB** (home: 76 of 100 GB used).

Training (E13): exactly the E07 recipe with `--img-size 384` on this cache, run
`2026-09-30_2230_e13_convnext_384_5f` — folds 3–4 in a 2-GPU 3090 Slurm job (`slurm/runs/e13_34.txt`,
`RAW_GB=100` so the job unpacks the 384 cache to local disk), fold 2 in a 1-GPU job after it
(`slurm/runs/e13_2.txt`), folds 0–1 on the interactive 3090 after E12 (`logs/chain_e13.sh`,
`logs/launch_e13_local.sh`). Run configs now record `cache_dir`
and `cache_size`.

Inference: `knee/infer.py` decodes each test series once and writes one cache per `cache_size`
the models need. Checked on a fake test folder (the 3 visible test studies + 3 train studies,
fetched from the archive): with E07 alone the predictions equal the previous code's exactly (max
difference 0.0); with a 256 px and a 384 px model, the test caches of the train studies are
byte-identical to `data/cache` and `data/cache384` (15/15 series each).

**E13 folds 3–4** (01:23): per-fold silver 0.9491 / 0.9371 (E07 0.9495 / 0.9404, E12 0.9421 / 0.9363).
On the 1,762 studies of folds 3–4: E13 0.9434, E07 0.9455, E09 0.9402, E12 0.9395 — 384 px is not a
clear single-model gain (better on the menisci, 0.944 / 0.905 vs ~0.93 / 0.86–0.90, worse on the MCL).
As a blend member it helps a little more than another 224 px seed: E07+E13 **0.9493** vs E07+E12
0.9475; E07+E09+E12+E13 0.9501. The E13 fold-2 Slurm job (pending behind other users' 3090 jobs) was
cancelled; fold 2 runs on the interactive 3090 after folds 0–1 (`logs/chain_e13_f2.sh`).

### E14 — DINOv2 ViT-S/14, another backbone family (2026-10-01, cancelled)

The public 0.943 ensembles gain most from mixing families (CoAtNet, DINOv2, RadImageNet), while two
seeds of one recipe give +0.005 silver and ~+0.001 LB. E14: `vit_small_patch14_dinov2.lvd142m`
(22 M parameters, self-supervised LVD-142M pretraining, positions resampled to 224 px via timm's
`dynamic_img_size`, enabled for `vit_*` backbones in `knee/model.py`), E07 recipe otherwise, backbone
lr 5e-5. On P100s (free, and ViTs are matmul-bound rather than depthwise-conv-bound): folds 0–1 job
1016103, folds 2–3 1016104 and fold 4 1016105 queued behind the per-user job limit. Decision rule:
keep it if the blend with E07 beats E07+E12 (0.9479) by more than ~0.002.

**E14 result**: 23.5 min/epoch on P100; silver after epochs 1–3: fold 0 0.739 → 0.762 → 0.787,
fold 1 0.663 → 0.683 → 0.698 (ConvNeXt-tiny: ~0.84 → 0.89 → 0.91; loss 0.542 vs 0.478 at epoch 3).
Cancelled at 02:46 with all its jobs: a model that far behind does not help a rank blend. Like
CoAtNet-0 in E06, the attention backbone learns slowly in this setup (3 adjacent MRI slices as RGB,
~40 windows per study, 12 epochs); making it work would need its own tuning (warm-up, layer-wise lr
decay, longer schedule), not a night run.

### E15 — EfficientNetV2-S, another CNN family (2026-10-01, running)

Same goal as E14 with a CNN, since CNNs (ResNet, ConvNeXt) learn well in this setup:
`tf_efficientnetv2_s.in21k_ft_in1k` (20 M parameters, ImageNet-21k like our ConvNeXt), E07 recipe
unchanged (backbone lr 2e-4), 224 px, on P100s (`slurm/runs/e15_*.txt`). Stop if epoch 3 is far
below ConvNeXt (~0.91); keep if the blend with E07 beats E07+E12 (0.9479) by more than ~0.002.
Epochs 1–2 (14–16 min each on P100): fold 0 0.870 → 0.901, fold 1 0.851 → 0.893 — as fast as ConvNeXt.

**E11 result** (fold runs ended ~03:45): silver **0.9435**, gold 0.895 (E07 0.9429 / 0.888). Rank blends
over all 5 folds: E07+E11 0.9481, E07+E09+E11 0.9494, **E07+E09+E11+E12 0.9504** (gold 0.905).
The `chain_e11` packaging job never started — the queued E15 jobs took the two freed job slots —
and was cancelled at 05:30: a version with E13 and E15 replaces v8. `logs/push_version.sh` now
does package → wait for the dataset → set `MODELS` → push → check the visible-test run.

**Inference for larger ensembles** (05:30, test pending): the hidden test is ~1,300 studies and the
visible-test log shows ~0.8 s of DICOM decoding per series on Kaggle (~1.7 h for the test), so ~30
fold networks could approach the 9 h limit. `knee/infer.py` now (1) reads each batch once and runs
all fold networks of a model on it (5× less JPEG decoding) and (2) takes `time_budget_s`: models are
processed in order and one is skipped when, at the measured cost per pixel, it would end past the
budget, so a submission is always written. To be checked against the previous code on the fake test
folder once the 3090 is free (a first attempt ran out of GPU memory next to E13; E13 unaffected).
**Checked at 07:00**: predictions identical to the previous code (max difference 0.0, alone and with a
384 px model), 16 s → 7 s for E07 on the 6 studies; with a 1 s budget the second model is skipped and
a complete submission is still written.

### Results of the night (2026-10-01, 07:00) — all 5 folds, 4,407 studies

| Model | silver | gold |
|---|---|---|
| E07 ConvNeXt-tiny 224 px | 0.9429 | 0.888 |
| E09 same, 130 mm field of view | 0.9415 | 0.892 |
| E11 same, 256 px | 0.9435 | 0.895 |
| E12 E07 with seed 1 | 0.9422 | 0.895 |
| **E13 384 px** | **0.9459** | **0.901** |
| E15 EfficientNetV2-S | 0.9334 | 0.872 |
| E07+E09 (v7, **LB 0.919**) | 0.9474 | 0.897 |
| E07+E09+E11+E12 | 0.9504 | 0.905 |
| **E07+E09+E11+E12+E13** | **0.9517** | 0.908 |
| all six | 0.9513 | 0.906 |

Over 5 folds, 384 px is the best single model (+0.003 over the 224/256 px runs, three times the
seed spread) and carries the best blend. EfficientNetV2-S is 0.01 below and lowers every blend it
joins: not kept. Exhaustive search over subsets: the best blend of each size always contains E13.

**Notebook `knee-submit` version 8** (v8 was never pushed by `chain_e11`, so Kaggle numbers this one 8):
E13, E11, E07, E12, E09 (25 fold networks) in that order of priority, with an 8 h time budget
(`logs/push_version.sh`, output in `logs/push_1001_0703`). Visible-test run complete at 07:40: all
five models used, 1.8 min for the 3 studies. **E16**: E13 with seed 1, folds
1–2 and 3–4 on nodemm05 (jobs 1016204 / 1016205), fold 0 on the interactive 3090.

### E17 — Inference settings, no retraining (2026-10-01)

`scripts/eval_variants.py` re-predicts every fold's validation studies with the fold network under
other inference settings (the first variant, the training settings, must reproduce the stored OOF);
whatever wins transfers to the notebook as is. On E13: 12 windows (reference), 16, 24, and 16 over
2–98% of the slices. Batch size scales down with the window count (4 studies × 12 windows, as in
`knee/infer.py`): a first attempt at 8 studies × 24 windows of 384 px ran out of memory (job 1016220,
with a smoke test of mine sharing its GPU), no fold lost elsewhere.

**Weight EMA** (`--ema <decay>` in `scripts/train.py`): an exponential moving average of the weights,
updated after each step. It does not change training, so each epoch evaluates both the raw and the
EMA weights on the same run (`silver_auc_ema` in the history); EMA weights and OOF are saved apart
(`checkpoints/ema_fold*.pt`, `oof_ema_fold*.csv`) so packaging and blends are unaffected unless chosen.

**E17 result** (job 1016222, ~9 min per fold for the four settings):

| E13 inference | silver | gold |
|---|---|---|
| stored OOF | 0.94594 | 0.9007 |
| 12 windows (training setting) | 0.94593 | 0.9007 |
| 16 windows | 0.94671 | 0.9022 |
| 24 windows | **0.94715** | 0.9040 |
| 16 windows over 2–98% | 0.94613 | 0.9045 |

The reference reproduces the stored OOF (difference 1e-5). Denser windows help steadily: 24 vs 12,
paired bootstrap over studies **+0.0012 [+0.0003, +0.0020]** — real but small, for twice the
inference time of that model. A wider span adds nothing on silver.

### E16 / E18 — More 384 px seeds (2026-10-01)

**E16** (seed 1, jobs 1016204/1016205 + interactive fold 0, done 09:35): silver **0.9461**, gold 0.906 —
the 384 px gain reproduces (E13 0.9459). E13+E16 0.9508; all six (E07, E09, E11, E12, E13, E16)
**0.9525** (gold 0.911), i.e. +0.0008 [+0.0005, +0.0011] over Version 8's five (which is itself +0.0044
[+0.0036, +0.0052] over v7). Too small to spend a submission on alone; it waits for E18 and for
Version 8's runtime (to decide on denser inference windows).

**E18** (seed 2, weight EMA 0.998, launched 09:35 by `logs/chain_e18.sh` after an EMA smoke test:
`checkpoints/ema_fold0.pt` written, both AUCs printed): folds 1–2 / 3–4 jobs 1016280 / 1016281, fold
0 on the interactive 3090.

**Version 8 on the public LB: 0.922** (v7: 0.919). Submitted 07:50, scored by 09:46: the hidden-test
run with 25 fold networks (five of them at 384 px) took under 2 h, far from the 9 h limit — room for
more models and denser inference windows. Silver +0.0044 gave LB +0.003 here, a better transfer than
the v7 blend (+0.0045 → +0.001): consistent with a model-side change (resolution).

**Scheduling (11:50)**: the 2-GPU E18 jobs waited on Resources (Slurm kept the one free 3090 for the
older 2-GPU job), so E18 folds 1–4 became 1-GPU jobs with `--time 3:30` (backfill). The 24-window
check of the other members (`logs/e17b_w24.sh`) was stuck behind the 4-job limit and moved to the
interactive 3090 after E18 fold 0. E18 at epochs 8–10: EMA ≥ raw by 0.000–0.0004 so far.

### E19 — Denser training windows at 384 px (2026-10-01, running)

Since 24 windows at inference beat 12 (E17), train on 12 windows per series instead of 8 (batch 6
studies instead of 8 to fit 24 GB), evaluate on 24; seed 3, EMA 0.998. Folds 1–2 / 3–4 as 2-GPU jobs
1016412 / 1016413, fold 0 on the interactive 3090 after the 24-window check (`logs/chain_e19.sh`).
Compare with the mean of E13/E16/E18 evaluated at 24 windows.

**24 windows on the other members** (`logs/e17b_w24.sh`, interactive 3090, 12:12–12:52), silver
stored → 24 windows: E16 0.9461 → 0.9477, E11 0.9435 → 0.9445, E07 0.9429 → 0.9434, E12 0.9422 →
0.9432, E09 0.9415 → 0.9427 — every model gains. Six-model blend (E07, E09, E11, E12, E13, E16):
12 windows 0.9525, **24 windows 0.9533**, +0.0016 [+0.0011, +0.0021] over Version 8; doubling the
weight of the 384 px models gives 0.9537 (not used: one more knob tuned on silver for +0.0004).

**EMA (E18, folds 0–3)**: raw 0.9457, EMA 0.9457 (gold 0.893 / 0.893). With the cosine schedule the
learning rate is already near zero over the last epochs, so the weight average adds nothing. Dropped.
E13 on the same four folds: 0.9481 (seed spread). `knee/infer.py` takes `windows` (batch size
48 // windows, so 4 studies × 12 windows as before).

**E19 first attempt failed** (13:05, all five folds): out of memory in the end-of-epoch validation —
6 studies × 5 series × 24 windows = 720 images of 384 px in one batch (training at 12 windows fitted).
My watcher missed it for ~2 h (its `pgrep -f` pattern matched its own command line), so the
interactive 3090 and two Slurm GPUs sat idle 13:05–14:55. Fix: the validation batch now scales down
when `windows_eval` > 12 (`bs * 12 // windows_eval`; unchanged for every earlier run). Relaunched
14:57: fold 0 on the interactive 3090, folds 1–4 as 1-GPU jobs 1016802–1016805.

**E18 complete** (15:14): silver **0.9440**, EMA weights 0.9440 (gold 0.904 / 0.904). The three 384 px
seeds span 0.9440–0.9461 (mean 0.9454 vs 0.9425 for the four 224/256 px runs): +0.003 on average,
with a seed spread of 0.002. E13+E16+E18 0.9521 (gold 0.915); all seven at 12 windows 0.9529
(six: 0.9525). Version 9 = these seven at 24 windows, pushed by `logs/chain_v9.sh`.

**Notebook `knee-submit` version 9** (15:55): E13, E16, E18, E11, E07, E12, E09 (35 fold networks,
384 px models first), 24 windows per series, 8 h budget. Visible-test run complete, all seven models
used, 2.3 min for the 3 studies (Version 8: 1.8 min) — the hidden test should take ~2–3 h.
Expected silver ≈ 0.9535 (six models at 24 windows 0.9533; E18 adds ~+0.0004 at 12 windows).

## Public weights: provenance audit (2026-10-01, decision 1)

Read-only audit of the public weight families behind the 0.94 notebooks (Kaggle CLI: dataset
descriptions, small files, training notebooks; no weights downloaded; working files in the session
scratchpad). Rule 4.b as we read it forbids sending report text to a hosted LLM.

| Family (datasets) | Labels used | Origin of the labels | Verdict |
|---|---|---|---|
| pilkwang DINOv2-small ×20 (`rsna-knee-weights`) | `report_labels_v2.csv` (`rsna-knee-llm-labels`) | `api_labeler.py`: `anthropic.Anthropic`, default `claude-opus-5`, report text in the message | tainted (probable: no training log) |
| RadImageNet heads: antoinegg1 e9/e11, prvsiyan V52, sofiaanjenje E11/E13 | mean of pilkwang + stevenleehans + lixin73 ("GPT-5.6-Sol") tables | includes the Anthropic table (`make_targets` requires it) | tainted |
| dreaddevelopment CoAtNet-rmlp-2 384 (`raptor-knee-*`) | `labels_llm_soft` (`rsna-knee-labels`) | "an LLM", never named | unknown |
| mattiaangeli CoAtNets and DINOv3 | unpublished `targets_dualsharp_soft.csv` | not documented | unknown |
| marwanmath RadImageNet ResNet-50 (backbone only) | none from the competition | RadImageNet pretraining (upstream Lab-Rasool/RadImageNet, MIT) | clean |

**Correction (2026-10-01 21:00): the premise is wrong.** The competition host (Po-Hao "Howard" Chen)
posted "Use of Commercially Hosted LLMs" two months ago: hosted LLMs and external inference
services are **permitted**; sending report text to an external LLM/API (e.g. to extract labels) is
not, by itself, prohibited private sharing. Conditions (Section 2.6.b, External Data and Tools):
the service/data/models must be reasonably accessible to all participants at minimal cost, and
comply with their terms of use. The forum was unreadable from here (Kaggle API), so the audit
rested on a strict reading of the data-security rule; it should have been checked with the user.
So the public weights are **not** tainted by their labels. Licences (Kaggle metadata): CC0-1.0 for
pilkwang weights + labels, dreaddevelopment raptor-knee-*, mattiaangeli CoAtNets; CC-BY-NC-SA-4.0
for antoinegg1 e11 heads; "other" for prvsiyan v52. Original (superseded) conclusion follows.

Conclusion (superseded): no competition-trained public weights are verifiably clean; using them would contradict
the rule reading behind our own local-LLM labels. What is clean and reusable: generic or medical
pretrained backbones (RadImageNet ResNet-50 → E20) and the public training recipes with our labels.
The public CoAtNet trainer (`coatnet_rmlp_2_rw_384`, 384 px, 12 training / 24 eval windows,
16 epochs, **backbone lr 3e-5**, head lr 1e-3, average of the 3 best epochs) trains its attention
backbone at a learning rate ~7× below ours — a likely reason why CoAtNet-0 (E06b) and DINOv2 (E14)
learned so slowly here.

### E20 — RadImageNet ResNet-50 (2026-10-01, queued)

`make_backbone("radimagenet_resnet50")` in `knee/model.py`: timm ResNet-50 loaded with the RadImageNet
weights (pinned Hugging Face revision; the 318 tensors map one-to-one, strict load). E07 recipe,
224 px, on P100s (jobs 1017022–1017024, waiting for a slot behind E19).

**E19 folds 0–2** (done 19:03; folds 3–4 still running): silver at epoch 12, 24 eval windows: 0.9482 /
0.9474 / 0.9519. On those 2,645 studies, all at 24 eval windows: E19 **0.9486**, E13 0.9495, E16 0.9481
(gold 0.911 / 0.901 / 0.905). Training on 12 windows instead of 8 brings nothing beyond seed noise;
only the denser *inference* helps (E17). EMA again equal to raw weights.

### E21a — CoAtNet-0 at the public recipe's learning rate (2026-10-01)

E06b (CoAtNet-0, backbone lr 2e-4) ended at 0.886 on the leak-free fold-0 protocol, far below
ConvNeXt-tiny (0.931). The public CoAtNet trainer uses 3e-5. E21a reruns E06b with `--lr 3e-5`
(interactive 3090, ~1 h): if it closes the gap, a CoAtNet-rmlp-2 at 384 px (the public best family)
trained on our labels becomes worth its cost.

E21a result (fold 0, 12 epochs, 5.6 min/epoch): silver **0.9269** (gold 0.840 on 11 studies), against
0.8860 for E06b at lr 2e-4: the learning rate was the problem, +0.041. It stays 0.004 below
ConvNeXt-tiny (E06a 0.9311), so CoAtNet-0 is not a better single model but becomes a credible
non-ConvNeXt member for the blend. Silver still rose until epoch 8, then flat.

### E22 — CoAtNet-0 at lr 3e-5, 5 folds (2026-10-01 20:35, running)

E21a's settings with the E03 label recipe (bootstrap from E02, α = 0.5) on all 5 folds, 224 px,
on the interactive 3090 (~70 min per fold → ~02:30). Goal: blend diversity (attention backbone).

## Public pipeline + our ensemble (2026-10-01 evening, user's go on decision 1)

The user gave the go ("oui intègre les") after the host's clarification. Best reproducible public
notebook: `yamadan96/rsna-knee-d4-public0946` — **0.943** on the public LB (its author submitted it
unchanged; the "0946" of the name comes from the parent `ryokucha/rsna-knee-d4-blend-0946-ours10`,
which reaches 0.946 by blending its author's private ConvNeXt ensemble at 10%). Pipeline: DINOv2 ×20,
A5 attention pooling ×5, RadImageNet heads, 4 Raptor views, 4 CoAtNet readers; 2×T4, pinned Kaggle
image (timm 1.0.22), all sources public.

`scripts/build_blend_nb.py` → `kaggle/blend/` (private kernel `theophilenadiedjoa/knee-blend`): our
Version 9 ensemble runs first in a subprocess (4.5 h budget, decoders installed with `--target`
so the public environment is untouched), then the public cells verbatim, then a per-label rank
average (public 0.85 + ours 0.15; public alone if our part fails). Code review of the public cells:
no network code (internet is off anyway); `exec`/`compile` only patch the CoAtNet inference modules
shipped in the attached datasets, pinned by SHA-256; subprocesses are its own inference workers.
The push from here was refused by my permission guard (third-party code under the user's
account): the user pushes it.
Pinned public image: Python 3.12 (cpython-312 caches in its outputs) → our cp312 decoder wheels install there.

### Official label criteria vs our soft labels (2026-10-01 ~22:00)

The user shared the challenge's clinical overview (label criteria: ACL/MCL high-grade only, effusion
and Baker's moderate/large, OA > 50 % cartilage loss over ≥ 1 cm, meniscus signal reaching the
surface on ≥ 2 images, "on the fence" → negative; reports are multilingual; ground truth = two MSK
radiologists + adjudication, from the images). Read literally, report-"mild" findings should be
negatives. The 58 gold studies say otherwise — P(gold positive | Qwen grade), pooled over labels:
absent 0.04 (7/180), not mentioned 0.13 (23/176), equivocal 0.19 (14/72), **mild 0.57 (50/87)**,
moderate 0.70 (69/98), severe 0.93 (77/83). Effusion "mild" (small) is gold-positive 22/34 times:
reports under-grade relative to the image annotators. Our a-priori mapping (0.02 / 0.10 / 0.25 /
0.6 / 0.85 / 0.97) is already close to these rates, so no remapping. Fracture: 4/29 gold positives
are not mentioned in the report (image-only signal). The model's five slots already put the three
fluid-sensitive fat-suppressed planes first, as the overview recommends.

**22:00 SSH drop**: the interactive job (1015856) ended with the SSH session and killed E22 during
fold 1 (fold 0 done: silver 0.9322, gold(fold) 0.879). Relaunched 22:40: folds 1–2 on the new
interactive 3090 (job 1017468), folds 3–4 as a batch job (`slurm/runs/e22_34.txt`, submitted by
`logs/chain_e22.sh` when E19 fold 4 frees a slot) so they survive further drops. Slurm jobs were
unaffected: E20 folds 0–1 done (silver 0.9437 / 0.9425 at epoch 12), folds 2–4 and E19 fold 4 running.

### Forum: single models at 0.950–0.954, labels are the lever (2026-10-01 22:45)

Threads shown by the user: a 5-fold ResNet at 224 px scores **0.954** LB (13th), a single-fold
CoAtNet at 224 px 0.950 (43rd, Archit Konde), a Qwen 3.5 2B VLM with LoRA 0.950. So the gap between our
0.922 ensemble and the top is not architecture or resolution. Archit Konde's recipe: his OOF on the
58 gold studies beat the extracted labels, so he mixed OOF predictions into the labels (soft, never
0/1; replacing was worse than mixing; masking noisy cells did nothing), with more weight on the model
than "felt sane", which mattered most for silent cells (synovitis). Out-of-fold predictions are
essential (in-fold ones parrot the labels). His single-model OOF on the 58: 0.930 (synovitis 0.797,
lateral OA 0.816). One participant saw 50/50 bootstrapping lift CV but not the LB.

Our E03 labels were exactly this, but with a weak teacher (E02 ResNet-34, gold 0.806) and α = 0.5.
Gold agreement of candidate training labels (macro AUC on the 58, soft label vs gold):

| Labels | Gold |
|---|---|
| report soft (Qwen) | 0.870 |
| E03 = ½ report + ½ E02 OOF (current training labels) | 0.897 |
| teacher = mean OOF of E13, E16, E18, E11, E07, E12, E09 | 0.914 |
| (1 − α) report + α teacher, α = 0.25 / 0.5 / 0.6 / **0.7** / 0.8 / 0.9 | 0.913 / 0.922 / 0.926 / **0.927** / 0.926 / 0.923 |
| same α on silent cells only (mentioned cells at 0.5), α = 0.7 | 0.924 |

Uniform α = 0.7 (Archit's "heavier than sane" zone; 0.6–0.8 is a plateau, the 58 cannot separate
them). The per-finding gain is largest for effusion, contusion, fracture, Baker's and medial OA.
Teacher stored as a pseudo-run, `runs/2026-10-01_2300_teacher7_oof_mean/` (oof_fold*.csv = mean of the
7 runs' OOF probabilities, `teacher.json` lists them), so `--bootstrap-run` takes it unchanged.

### E23 — round-2 labels (2026-10-01 22:47, running)

E07's exact recipe (ConvNeXt-tiny 224, 8/12 windows, seed 42, silence weight 0.5, gold weight 3) with
`--bootstrap-run 2026-10-01_2300_teacher7_oof_mean --bootstrap-alpha 0.7`: only the labels change,
so E23 vs E07 (gold 0.888, LB 0.918) measures the label effect, and E23 alone goes to the LB.
Folds 0–1 on the interactive 3090; folds 2–4 as a batch job (`logs/chain_e23.sh`, after E19 fold 4).
E22 (CoAtNet on the old labels) is stopped after fold 0 to free the GPU: if round-2 labels win, all
old-label members lose value. Silver is not comparable across label sets (its targets come from the
report only, but the student is pulled towards the teacher); judge on gold and the LB.
E23 fold 0 epoch 1: silver 0.8675, gold(fold) 0.770, 4.7 min/epoch (validation passed).

**E19 complete** (23:03): 5 folds at 24 eval windows silver 0.9480, gold 0.908 — level with E13/E16
(0.9471 / 0.9477). Added to the Version 9 set (all at 24 windows except E18 at 12): silver 0.9536 →
0.9542, gold 0.913 → 0.914. Its slot went to E23 folds 2–4 (job 1017537, nodemm05).

**E20 folds 2–3** done (23:10): silver 0.9439 / 0.9463 at epoch 12 (fold 4 running, P100).

### E24 — 384 px with round-2 labels (2026-10-01 23:12, running)

Speculative follow-up to E23 on the slot E20 freed: E13's recipe (ConvNeXt-tiny 384 px, seed 42) with
the round-2 labels (same teacher and α = 0.7 as E23), 2-GPU job 1017572 on node40 (folds 0–2 / 3–4,
~8 h). If E23 shows the label gain on the LB, E24 is the strong version ready in the morning.
E24 first epochs passed (23:30): 12.9 min/epoch (folds 0–2 ~07:00, folds 3–4 ~04:30).

**E23 interim** (23:30) vs E07 at the same point: fold 0 epoch 10 silver **0.9524** / gold(fold) 0.927
(E07: 0.9407 / 0.879); fold 2 epoch 5 0.9407 / 0.917 (E07: 0.9312 / 0.881). Caution: the student's
evaluation is leaky in the E05 sense — its training labels come from teacher models that were trained
on the student's validation fold (report labels and, at weight 3, its gold labels), so part of this
gain is distilled from the validation fold itself (E05 found about half of E03's apparent gain real).
The gold agreement of the labels themselves (0.927, table above) has no first-order leak (teacher OOF on a
gold study never saw that study), but a second-order one remains: the teacher runs were trained on E03
labels, which mix in E02 OOF from models that did see it. The LB decides.

**Version 9 runtime** (23:40): still running after ~4 h on the hidden test. My 2–3 h estimate was
wrong: it scaled the 3-study visible run, where fixed overhead dominates. Compute vs Version 8:
35 vs 25 networks, three 384 px models instead of one, 24 vs 12 windows → ~3.6× → ~5–6 h expected;
the 8 h budget skips the last models if needed, so it cannot hit Kaggle's 9 h limit.

**knee-blend version 1 pushed by the user** (23:40): public 0.85 + ours (Version 9 set) 0.15; visible-test run in progress, watched by `logs/watch_blend.sh`.
knee-blend v1 visible-test run COMPLETE (23:52): decoders ok, all 7 of our models (2.7 min), public pipeline COMPLETE, blend written (public 0.85 + ours 0.15), 7.0 min in total. Log: logs/blend_1001_2238/. Ready to submit.

**E23 folds 0–2 done** (00:30): silver 0.9532 / 0.9513 / 0.9532, gold(fold) 0.922 / 0.888 / 0.916 (E07 fold
0: 0.9414 / 0.881, fold 2: 0.9419). Folds 3–4 in job 1017537.

### E25 — RadImageNet ResNet-50 with round-2 labels (2026-10-02 00:33, running)

The interactive 3090 freed by E23 folds 0–1: E20's recipe (RadImageNet ResNet-50, 224 px; old labels
~0.944 per fold, level with ConvNeXt-tiny 224) with the round-2 labels, 5 folds in sequence. A
non-ConvNeXt member for the round-2 ensemble. Vulnerable to SSH drops (interactive job).
Kaggle: Version 9 (submitted 20:30) and knee-blend v1 (23:56) are being scored.
E25 fold 0 epoch 1: silver 0.8999, gold(fold) 0.890, 3.9 min/epoch on the 3090 (5 folds ~04:30); validation passed.

**External knee-MRI data** (discussed 00:40): allowed by the host when access is a click-through
(MRNet, fastMRI/fastMRI+, OAI, SKM-TEA). Deferred with the user's agreement: labels first; if time
allows, MRNet as a cheap pretraining test; OAI only if its access really is one click. Registrations
are in the user's name.

**E20 complete** (01:15): silver 0.9420, gold 0.883 (E07: 0.9429 / 0.888) — RadImageNet pretraining is
level with ImageNet-22k ConvNeXt-tiny at 224 px. Added to the 8 old-label ConvNeXt runs: silver 0.9542
→ 0.9548, gold 0.914 → 0.914. Its slot goes to **E26**: CoAtNet-0 at lr 3e-5 (E22's recipe) with the
round-2 labels, 5 folds in one 1-GPU 3090 job (1017787, ~07:30).

**Version 9 on the public LB: 0.924** (Version 8: 0.922). Submitted 20:30, scored by 01:35 (~5 h on the
hidden test, as re-estimated). Silver +0.0018 (0.9517 → 0.9535) gave LB +0.002: 24 inference windows and
two more 384 px seeds transfer.

**E23 complete** (01:40): silver **0.9516**, gold **0.901** (E07, same recipe on the old labels: 0.9429 /
0.888; the 7-run old-label teacher: 0.9537 / 0.914). One 224 px model on round-2 labels nearly matches
the 7-model ensemble that taught it. Gold per finding vs E07: medial meniscus 0.89 → 0.94, lateral
meniscus 0.80 → 0.85, PF OA 0.79 → 0.83, contusion 0.89 → 0.95; ACL 0.94 → 0.92 and MCL 0.93 → 0.89
(single-digit positives, noise). Leak caveat of E23 applies. → Version 10 = E23 alone (24 windows;
E07 alone scored 0.918 at 12 windows, the window change is worth ~+0.001).
**Version 10** (knee-submit, E23 alone, 24 windows) pushed and checked 02:20: visible-test run complete, 1.0 min. To submit.

**E24 folds 3–4 done** (04:00, fold 4 silver 0.9473); folds 0–1 done, fold 2 running (~06:10). **E26** started
02:55 on nodemm05.

**E25 complete** (04:18): RadImageNet ResNet-50 on round-2 labels: silver **0.9455**, gold **0.899** (same
model on old labels, E20: 0.9420 / 0.883). The label gain holds on a second architecture (+0.0035 silver,
+0.016 gold; ConvNeXt: +0.0087 / +0.013). E23 + E25 rank blend: 0.9536 / 0.906.
Note for round 3: the round-2 students' OOF is leaky on gold too (their labels distil teacher runs that
were trained on the student's validation fold, gold labels included at weight 3), so gold agreement can
no longer rank round-3 label mixtures honestly — wait for the LB (Version 10) before iterating.

### E27 — ConvNeXt 256 px, seed 1, round-2 labels (2026-10-02 04:25, running)

E11's recipe with seed 1 on the round-2 labels, interactive 3090 (freed by E25), ~4.5 h: a member for
the round-2 ensemble.
E27 fold 0 epoch 1: silver 0.8591, 6.0 min/epoch (5 folds ~10:20); validation passed.

**E24 complete** (06:10): ConvNeXt 384 px on round-2 labels: silver **0.9551**, gold **0.922** (E13, same
recipe on old labels: 0.9459 / 0.901). One model beats the whole Version 9 ensemble (0.9536 / 0.913).
Round-2 blends (rank average, common studies): E24+E23 0.9562 / 0.917; E24+E23+E25 0.9564 / 0.916;
2·E24+E23+E25 0.9569 / 0.919. → Version 11 = E24, E23, E25 at 24 windows (equal weights), pushed now
so it is ready once Version 10 confirms the label gain on the LB.

### E28 — E24 with seed 1 (2026-10-02 06:15, running)

A second seed of the best single model (384 px, round-2 labels): seed ensembles of the 384 px model gave
the largest blend gains in round 1. node40 was idle with no one waiting: folds 0–1 / 2–3 in a 2-GPU job
(1017917) and fold 4 in a 1-GPU job (1017918), ~5 h.
E28 first epochs passed (06:34): 12.7–13.5 min/epoch on the three GPUs (~2.7 h per fold; folds 0–3 ~11:45, fold 4 ~09:15).
**Version 11** (knee-submit: E24, E23, E25 at 24 windows) pushed and checked 06:54: all three models used, 1.6 min for the 3 visible studies. To submit after Version 10.

**E26 complete** (08:25): CoAtNet-0 (lr 3e-5) on round-2 labels: silver 0.9430, gold 0.899 — the weakest
round-2 member, and it lowers every round-2 blend (E24+E23+E25 0.9564 → 0.9555 with E26). Not used.

### E29 — E24 with seed 2 (2026-10-02 08:30, running)

Third seed of the best model; 5 GPUs were free on the 3090 nodes with nobody waiting. One 2-GPU job (1017959),
folds 0–2 / 3–4, ~8 h. With E24 and E28: a three-seed 384 px ensemble on round-2 labels.
E28 fold 4 done (08:38): silver 0.9448 at epoch 12 (E24 fold 4: 0.9473); 11.4 min/epoch, faster than estimated.
E29 first epochs passed (08:47): 12.7–12.9 min/epoch (folds 3–4 ~14:00, folds 0–2 ~16:30).

**knee-blend v1 submission** (submitted 23:56): status COMPLETE at ~09:00 but no public score through the API (checked until 09:22). Ran ~9 h — at Kaggle's limit; possible timeout or scoring error (asked the user what the page shows). Suspect: our part used most of its 4.5 h budget and the public pipeline's own time budget starts from its own T0, unaware of the time already spent.
Versions 10 and 11 submitted by the user (~09:35).

**knee-blend v1 = Notebook Timeout** (Kaggle page, confirmed by the user). Our part runs first under a
4.5 h budget that includes DICOM decoding (`knee/infer.run` starts its clock before `prepare_cache`), so the
public pipeline got ≤ 4.5 h while its own 8 h budget counts from its own start; on the ~1,300 hidden
studies (> 48, so its CoAt readers run one at a time) it needs more than that. Fix in
`scripts/build_blend_nb.py` (v2): public cells first (their budget is then right), ours after them with the
time left before 8.6 h (skipped below 0.75 h), our own budget = time left − 20 min, a hard timeout that
kills the process group 10 min before the end, on the GPU with the most free memory, 12 windows; the
public submission goes alone if ours is missing. Default models: round-2 E24, E23, E25. Weight set after
the Version 11 score.

**E27 complete** (09:45): ConvNeXt 256 px seed 1 on round-2 labels: silver 0.9530, gold 0.907. Round-2
blends: E24+E23+E25 0.9564 / 0.916 → + E27 0.9571 / 0.914.

### E30 — round-3 labels (2026-10-02 09:55, running, speculative)

Teacher = mean OOF of the round-2 runs E24, E23, E25, E27 (`runs/2026-10-02_0950_teacher_r2_oof_mean/`);
labels = 0.3 report + 0.7 teacher; student = E07/E23 recipe (ConvNeXt-tiny 224) on the interactive 3090
(~4.5 h). Its silver/gold are leakier still (two distillation hops through the validation fold), so only
the LB can judge it against E23 (Version 10). Started before the Version 10 score to keep the GPU busy.
E30 fold 0 epoch 1: silver 0.8493, 4.8 min/epoch (5 folds ~14:45); validation passed.

**Version 10 (E23 alone) on the public LB: 0.921** (scored 10:15; the first of the two 09:30 submissions,
the single 224 px model being the fastest). E07 alone (Version 5, same recipe on old labels, 12 windows):
0.918. Net of the 24-window gain (~+0.001, E17) the round-2 labels bring **~+0.002 on the LB**, against
+0.0087 silver and +0.013 gold in validation: most of the apparent gain was the distillation leak (as
E03 → LB +0.003 for gold +0.06, and as reported on the forum). Lesson: label recipes need a leak-free
check (E05-style nested teacher) before they are trusted; silver and gold of bootstrapped students are
not evidence. E30 (round 3) stays running but is deprioritised.

**E28 complete** (11:10): E24 with seed 1: silver 0.9542, gold 0.920. E24+E28 0.9571 / 0.926; the five
round-2 runs (E24, E28, E23, E25, E27) 0.9577 / 0.917.

### Public report-label tables (2026-10-02 11:05)

Three public label tables are CC0 (Kaggle metadata): pilkwang v2 (Claude), stevenleehans v2, lixin73
(GPT-5.6-Sol). Agreement of the raw report labels with the 58 gold studies (no trained model involved,
so no distillation leak): Qwen 0.870, pilkwang 0.867, **steven 0.887**, lixin 0.835; mean of all four
0.884 (rank 0.886); best pair steven + lixin 0.893. Through the teacher mix the base matters less:
α = 0.5 0.9224 (Qwen base) vs 0.9263 (4-table base); α = 0.7 0.9273 vs 0.9285.
`scripts/train.py --mix-public <names>` averages the tables with ours before bootstrapping (missing
cells keep ours).

### E31 — 4-table base labels (2026-10-02 11:12, running)

E23 with `--mix-public pilkwang_v2 steven_v2 lixin_sol56` and α = 0.5 (the base weighs more), 1-GPU job
1018179 (~16:30). Compare with E23 on the LB (0.921).
E31 fold 0 epoch 1: silver 0.8527, 4.7 min/epoch (~16:00); validation passed.

**Version 11 on the public LB: 0.929** (scored 11:30) — E24, E23, E25 (round-2 labels) at 24 windows, three
models: +0.005 over Version 9 (0.924, seven old-label models). Silver 0.9536 → 0.9564 (+0.003) transferred
in full here, unlike E23 alone at 224 px (+0.002 LB): the round-2 labels pay off at 384 px and in a
mixed-architecture blend. Next: Version 12 = the five round-2 runs (E24, E28, E23, E25, E27; silver
0.9577), and knee-blend v2 with these as our part.
**Version 12** (knee-submit: E24, E28, E23, E25, E27 at 24 windows) pushed and checked 12:10: all five used, 1.8 min for the 3 visible studies. knee-models dataset now holds only the five round-2 runs. knee-blend v2 built (weight 0.3, same five runs at 12 windows), to be pushed by the user.
Version 12 submitted by the user (12:17); knee-blend v2 (weight 0.3) pushed by the user, rehearsal running.
knee-blend v2 rehearsal COMPLETE (12:30): public pipeline first (0.07 h), 14.7 GB free on both T4s after it, ours on GPU 1 with all five round-2 runs, blend public 0.70 + ours 0.30 written, 0.09 h in total (logs/blend_1002_1211/). Ready to submit.
knee-blend v2 submitted by the user (12:35).

**E30 complete** (14:05): round-3 labels, ConvNeXt 224: silver 0.9525, gold 0.922 (E23, round 2: 0.9516 /
0.901). The gold jump is the kind the leak produces (two distillation hops); only the LB can tell →
Version 13 = E30 alone, to compare with Version 10 (E23 alone, 0.921).

### E32 — 384 px with round-3 labels (2026-10-02 14:15, folds 0–1 running)

The 3090 nodes are busy (another user has 10+ jobs waiting), so only the interactive GPU: E24's recipe
with the round-3 teacher, folds 0–1 (~5 h). Folds 2–4 only if Version 13 confirms round 3.
E32 fold 0 epoch 1: silver 0.8414, 13.4 min/epoch (folds 0–1 ~19:40); validation passed.
**Version 13** (knee-submit: E30 alone, round-3 labels, 24 windows) pushed and checked 14:48: 1.0 min for the 3 visible studies. Compare with Version 10 (E23 alone, 0.921).
Version 13 submitted by the user (14:57); today's 5 submissions used (V10, V11, V12, knee-blend v2, V13).

**E31 complete** (15:20): 4-table base labels (Qwen + 3 public CC0 tables), teacher7 at α = 0.5: silver
0.9498, gold 0.906 (E23: 0.9516 / 0.901). Silver is lower by construction (its targets are Qwen's alone
and E31 learns a different base); gold is level within noise. E23 + E31 rank blend 0.9545 / 0.911 — the
two label sets give diverse members. LB test of E31 tomorrow. All 3090s are allocated (others' jobs),
so the free slot stays unused for now.

**E29 complete** (15:25): E24 with seed 2: silver 0.9549, gold 0.929. Three 384 px seeds (E24+E28+E29)
0.9579 / 0.929; with E23, E25, E27 0.9582 / 0.919; + E31 0.9585 / 0.921; E24+E28+E29+E25+E31 0.9585 / 0.925.
Candidate Version 14 (after the Version 12/13 scores): the three seeds + E25 + E31 (+ E23, E27).

**Version 12 on the public LB: 0.929** (five round-2 runs; = Version 11): E28 and E27 add nothing visible.
**Version 13 (E30 alone, round 3): 0.927** vs Version 10 (E23 alone, round 2): 0.921 → **+0.006 on the
LB for one more labelling round** (round 1 → 2 gave ~+0.002 net at 224 px). One 224 px model on round-3
labels nearly matches the three-model round-2 ensemble. Iterating the labels is the lever after all;
the leak inflates the validation numbers but the direction holds.
→ E32 folds 2–3 (job 1018661, running) and fold 4 (1018662, pending), E33 = RadImageNet ResNet-50 on
round-3 labels (1018663, pending). Next: round-3 ensemble (E32, E30, E33) and a round-4 teacher.
E32 folds 2–3 job: fold 2 epoch 1 silver 0.8591, 12.7 min/epoch (folds 2–3 ~21:30); validation passed.
E32 fold 4 (job 1018662, started 17:02): epoch 1 silver 0.8535, 12.8 min/epoch (~19:45). E33 (job 1018663, started
17:11): fold 0 epoch 1 silver 0.8991, 3.8 min/epoch (~21:00). Both passed validation.
E32 folds 0–2 done (18:55), last epoch silver / gold(fold) against E30 (same round-3 labels, 224 px):
fold 0 0.9567 / 0.932 vs 0.9547 / 0.935, fold 1 0.9570 / 0.927 vs 0.9519 / 0.894, fold 2 0.9569 / 0.941 vs
0.9520 / 0.918 → 384 px adds +0.002 to +0.005 silver on round 3 as on round 2.

E32 complete (20:30), all 5 folds OOF (`scripts.eval_oof`):

| run | labels | px | silver | gold |
|---|---|---|---|---|
| E32 | round 3 | 384 | **0.9563** | **0.9268** |
| E30 | round 3 | 224 | 0.9525 | 0.9221 |
| E24 | round 2 | 384 | 0.9551 | 0.9215 |
| E32 + E30 + E24 rank blend | | | 0.9579 | 0.9284 |

E32 vs E30 is the clean comparison (same labels): 384 px +0.004 silver, +0.005 gold. E32 vs E24 crosses
label rounds, where the leak inflates the newer one (see the round-2 lesson): the LB decides.

### E34 — 256 px seed 1 with round-3 labels (2026-10-02 19:00)

E27 recipe (ConvNeXt-tiny 256 px, seed 1) on the round-3 labels: folds 0–2 on the interactive GPU freed
by E32 (`logs/launch_e34_local.sh`, ~22:15), folds 3–4 as job 1018988 (`slurm/runs/e34_34.txt`, pending
until a slot frees, ~1 h per fold). It pended on Resources for 1.5 h with GPUs free: `multi_gpu.sbatch` asks 96 GB by
default and no node had that much unallocated; `scontrol update JobId=1018988 MinMemoryNode=32768` → it
started at 20:38 on nodemm05 (~22:50). 1-GPU jobs now pass `--mem=32G` (usage line of the script). Purpose: the round-4 teacher then mirrors the round-2 one (384 px,
224 px, RadImageNet R50, 256 px seed 1 → E32, E30, E33, E34) and the round-3 ensemble gets a 4th member.

E33 complete (20:45): silver 0.9467, gold 0.9085 (E25, same recipe on round-2 labels: 0.9455 / 0.8989).
Round-3 ensemble E32 + E30 + E33 (rank blend): 0.9572 / 0.9253; E32 + E30 alone 0.9571 / 0.9259; the
round-2 analogue E24 + E23 + E25 (Version 11, LB 0.929): 0.9564 / 0.9157.

**knee-blend v2 scored (21:00): public LB 0.942** (public pipeline 0.7 + our five round-2 runs 0.3, rank
average), against **0.943** for the public pipeline alone. It finished inside the 9 h limit this time
(public part first, hard timeout on ours). The scoring run's log is hidden, so it is not certain that our
part ran rather than timing out. If it had timed out, the public submission would have been kept alone, and a
0.942 (not 0.943) suggests our part did run. So our 0.929 ensemble, at weight 0.3, adds nothing to a 0.943
pipeline and may cost ~0.001. Our models are weaker and probably carry little that the public ensemble
does not already have. Next try only with a clearly stronger ensemble of ours (round 3/4 at 384 px, if it
beats 0.929 alone on the LB), at a lower weight (0.15–0.2), since each try costs a submission and ~9 h of
Kaggle GPU.

### Round 4: E35 (224 px) and E36 (384 px) (2026-10-02 20:48)

Teacher = mean OOF of the round-3 runs E32, E30, E33 (`runs/2026-10-02_2050_teacher_r3_oof_mean/`, built
with the new `scripts.make_teacher`, which rebuilds the round-2 teacher bit for bit). Three runs, not four:
E34 would only be done at 22:50 and two GPUs were free at 20:50; E34 stays a round-3 ensemble member.
Teacher OOF: silver 0.9576, gold 0.926 (round-2 teacher 0.9576 / 0.914; leak-inflated, direction only).
Labels = 0.3 report + 0.7 teacher, as every round.
- **E35** = E30 recipe (ConvNeXt-tiny 224 px) on round-4 labels, job 1019036 (2 GPUs: folds 0–2 / 3–4,
  ~23:45) → its LB against E30 alone (Version 13, 0.927) measures round 4 against round 3.
- **E36** = E32 recipe (384 px) on round-4 labels: folds 0–1 job 1019037 (~01:45), folds 2 and 3 job
  1019038 (2 GPUs, pending a slot), fold 4 on the interactive GPU after E34 folds 0–2
  (`logs/chain_e36_f4.sh`, ~00:40).

**Night of 2026-10-02/03.** E34 (256 px seed 1, round 3) and E35 (224 px, round 4) complete, E36 folds 0–3
complete. The SSH session dropped ~22:40 and took the interactive job down with E36 fold 4 at epoch 1
(the memory rule again: long runs only in batch jobs, the chain to the interactive GPU was a mistake);
fold 4 rerun as job 1019403 at 03:25 (~06:05).

| run | labels | px | silver | gold |
|---|---|---|---|---|
| E35 | round 4 | 224 | 0.9530 | 0.9207 |
| E30 | round 3 | 224 | 0.9525 | 0.9221 |
| E34 | round 3 | 256 seed 1 | 0.9540 | 0.9217 |
| E27 | round 2 | 256 seed 1 | 0.9530 | 0.9065 |
| E32 + E30 + E33 + E34 | round 3 | | 0.9580 | 0.9250 |
| E32 + E30 + E34 | round 3 | | 0.9579 | 0.9274 |

Round 4 is flat against round 3 on our own validation (E35 vs E30: +0.0005 silver, −0.001 gold), where
round 3 had moved +0.015 gold over round 2 (E34 vs E27). With the leak, flat here may mean flat or worse
on the LB: Version 15 (E35 alone) against Version 13 (E30 alone, 0.927) decides whether a round 5 is worth it.

knee-submit **Version 14** pushed (04:00): `['*e32*', '*e30*', '*e34*', '*e33*']` (round-3 ensemble), all four
runs ran on the visible test in 1.8 min. The dataset version holds E32, E30, E33, E34, E35 so that
Version 15 (E35 alone) only changes MODELS (`SKIP_PACKAGE=1 bash logs/push_version.sh …`). Later dataset
versions stay supersets, so older notebook versions find their runs whether or not Kaggle pins the
dataset version.

**E36 complete** (05:45, fold 4 rerun as a batch job): round 4, 384 px: silver 0.9559, gold 0.9287 (E32, round 3,
384 px: 0.9563 / 0.9268) — flat, like E35 vs E30 at 224 px. Rank blends: E36 + E35 0.9570 / 0.9278; the six
round-3/4 runs (E36, E32, E35, E34, E30, E33) **0.9585** / 0.9283; round-3 four (Version 14) 0.9580 / 0.9250.

knee-submit versions ready for 2026-10-03 (all checked on the visible test; the dataset version of 06:00
holds E31, E32, E30, E33, E34, E35, E36, a superset of what Versions 14–15 need):

| version | MODELS | question it answers |
|---|---|---|
| 14 | E32, E30, E34, E33 | round-3 ensemble vs Version 11 (round 2, 0.929) |
| 15 | E35 | round 4 vs round 3 (Version 13, E30 alone, 0.927) |
| 16 | E36, E32, E35, E34, E30, E33 | best ensemble on our validation (2.3 min on 3 studies) |
| 17 | E31 | public-table base labels vs ours (Version 10, E23 alone, 0.921) |

### E37 — ConvNeXt-small at 384 px, round-3 labels (2026-10-03 04:10, folds 0–1)

E32 recipe with `convnext_small.fb_in22k_ft_in1k` instead of tiny (`slurm/runs/e37_01.txt`, 2-GPU job
1019430, ~08:30). At 224 px on older labels, small gave +0.006 on fold 0 (E08c) for 1.7× the compute,
within seed noise. With cleaner labels and 384 px, capacity may matter more. Compare with E32 folds
0–1 (silver 0.9567 / 0.9570, gold(fold) 0.932 / 0.927); folds 2–4 only if both folds gain.

**E37 folds 0–1 done** (08:00): ConvNeXt-small alone = tiny alone (folds 0–1 pooled: silver 0.9567 vs E32
0.9566, gold 0.927 vs 0.926; per fold 0.9571 / 0.9565 vs 0.9567 / 0.9570) — capacity is not the lever.
But it blends better than another tiny run (same 1,764 studies): E36 + E32 0.9583 → E36 + E32 + E37
**0.9597**; the six of Version 16 0.9595 → seven with E37 0.9600. Folds 2–3 (job 1019580, 2 GPUs) and 4
(1019581) launched 08:05 (~12:30), for the ensemble.

**E37 complete** (11:55): silver 0.9553, gold 0.9247 over 5 folds (E32 0.9563 / 0.9268) — a bit weaker alone,
still the better blend partner: E37 + E32 0.9584 vs E36 + E32 0.9580; the three 384 px runs E37 + E36 + E32
0.9590 / 0.9316; Version 16's six + E37 **0.9590** / 0.9289 (six: 0.9585 / 0.9283). Subsets of the seven
reach 0.9592, not chosen (+0.0002 tuned on silver). **Version 18** = Version 16 + E37, so its LB against
Version 16 measures E37 alone. Versions 14–17 submitted by the user at 11:23 (ledger refs 56794645–51,
matched to versions through the `scriptVersionId` the API returns).

**Version 15 (E35 alone, round 4): public LB 0.930** (scored 12:15) vs Version 13 (E30 alone, round 3) 0.927 →
**+0.003 for round 4**, which our validation called flat (+0.0005 silver, −0.001 gold). One 224 px model on
round-4 labels beats the round-2 three-model ensemble (Version 11, 0.929). LB gains per labelling round so far:
~+0.002 (1→2: E07 0.918 → E23 0.921, ~0.001 of it from 24-window inference), +0.006 (2→3), +0.003 (3→4). Our validation can no longer see
label gains: silver is scored against the report labels the rounds move away from, and gold has 58 studies.

### Round 5: E38 (224 px), E39 (384 px); E40 = ConvNeXt-small 384 on round 4 (2026-10-03 12:25)

knee-submit **Version 18** (V16 + E37, 7 runs, 3.1 min on 3 studies) and **Version 19** (`['*e36*', '*e35*']`, the
two round-4 runs, 1.3 min) pushed 12:40–12:44; the dataset of 12:00 holds E31–E37. Only one submission
is left today: Version 19 if Versions 14/16 show that mixing round-3 runs in does not help (Version 15
says one round-4 model beats one round-3 model by 0.003), otherwise Version 18.

Teacher = mean OOF of the round-4 runs E35 and E36 (`runs/2026-10-03_1220_teacher_r4_oof_mean/`; OOF silver
0.9572, gold 0.928). Only round-4 runs, so the labels keep moving one round at a time (E37, round 3, left out).
- **E38** = E35 recipe on round-5 labels, job 1019853 (folds 0–2 / 3–4, ~15:00) → LB against Version 15 (0.930).
- **E39** = E36 recipe (384 px) on round-5 labels, folds 0–1 job 1019854 (~15:00); folds 2–4 after.
- **E40** = E37 recipe (ConvNeXt-small 384) on round-4 labels, the round the LB has validated: fold 0
  job 1019856 (~16:45; one GPU, no node had two free), other folds as slots free.

**Version 17 (E31 alone): public LB 0.928** (scored 13:30) vs Version 10 (E23 alone) 0.921, same teacher
(round-1 OOF), same 224 px recipe → **+0.007 for the 4-table base** (mean of Qwen + pilkwang_v2 + steven_v2 +
lixin_sol56, α 0.5 instead of 0.7: both changed, so the split between base and α is unknown). As much as a
whole bootstrapping round, and our validation said −0.002 silver (silver is scored against Qwen's labels).
→ **E41** = E31 recipe with the round-4 teacher (E35, E36), `slurm/runs/e41.txt`, job 1019919 (queued behind
the job limit, starts when E38 ends ~15:00, ~17:45). Against E35 (0.930) and E38 (round 5, Qwen base).
CV note: a score from public-table labels is still our model, but the labels are no longer only ours —
say so if it ever reaches the CV line.

**Version 14 (round-3 ensemble E32, E30, E34, E33): public LB 0.932** (scored 14:45) — our best with our own
labels: +0.005 over E30 alone (0.927), +0.003 over the round-2 ensemble (Version 11, 0.929). At round 2 the
ensemble added +0.008 over its 224 px member alone (Version 11 vs Version 10, 384 px included). A round-4
single model (0.930) does not reach a round-3 ensemble: ensembles of the latest round are the target.

**E38 complete** (14:45, round 5, 224 px): silver 0.9542, gold 0.923 (E35, round 4: 0.9530 / 0.921; E30, round 3:
0.9525 / 0.922) — our validation barely moves across rounds while the LB did (+0.006, +0.003), so only the
LB decides (E38 alone against Version 15, 0.930). E41 started 14:49 on node40 (~17:35).

E39 folds 0–1 done (14:55): silver 0.9559 / 0.9580 at epoch 12 (E36 same folds, round 4: 0.9594 / 0.9543).
Folds 2–3 job 1020034 (~17:40), fold 4 queued (`slurm/runs/e39_4.txt`, starts when a job slot frees).

**Version 16 (six runs, rounds 3 + 4: E36, E32, E35, E34, E30, E33): public LB 0.933** (scored 15:45) — new best
with our own labels, +0.001 over the round-3 four (Version 14). Today's fifth submission: Version 18 (Version 16 +
E37) rather than Version 19 (the two round-4 runs alone), since more members of the latest rounds keep helping.

**E40 fold 0** (16:15, ConvNeXt-small 384 px on round-4 labels): silver 0.9540, gold(fold) 0.938 — below E36 (tiny,
round 4) 0.9594 and E37 (small, round 3) 0.9571 on the same fold, and it lowers the blends it joins (E36 + E35
0.9608 → with E40 0.9605; E40 + E36 0.9597 vs E37 + E36 0.9611). One fold, but no sign of a gain: stopped
there (the other four folds would cost ~17 GPU-hours). E39 fold 4 took the freed slot (job 1020036, ~18:50).

**E42** queued (16:40): E41's labels (4-table base, α 0.5, round-4 teacher) at 384 px, the E39 recipe
(`slurm/runs/e42_01.txt`, `e42_23.txt`, `e42_4.txt`, jobs 1020095–97, start as E41/E39 free their slots).
Launched before E41's LB because the 4-table base gave +0.007 at round 2 and 384 px helped at every round.

**E41 complete** (17:15, 4-table base + round-4 teacher, 224 px): silver 0.9510, gold 0.906 — below E38 (0.9542 /
0.923) and E35 (0.9530 / 0.921) on our validation, the same profile as E31 (0.9498 / 0.906), which still beat
E23 (0.9516 / 0.901) by 0.007 on the LB. Our validation scores against Qwen's labels (silver) and 58 studies
(gold), so it cannot judge the public-table base: E41 alone goes to the LB against E38 and E35.

**E39 complete** (18:40, round 5, 384 px): silver 0.9566, gold 0.922 (E36, round 4: 0.9559 / 0.929; E32, round 3:
0.9563 / 0.927) — flat on our validation, as every round since 3. Blends: E39 + E38 0.9579; rounds 3–5 at
384 px + 224 px (E39, E38, E36, E35, E37, E32) **0.9594** / 0.931; all ten runs 0.9595. E42 folds 2–3 (job
1020096) wait for two free GPUs on one node (none had two at 18:40; ~20:10 when fold 4 frees node40).

E42 fold 4 done (19:40); folds 2–3 started 19:42 on node40 (~22:20). **E43** (19:45) = E34 recipe (256 px, seed 1)
on round-5 labels, all five folds on the one free GPU (job 1020210, ~01:30) → round-5 ensemble member.

E42 folds 0, 1, 4 done (20:04; last-epoch silver 0.9577 / 0.9550 / 0.9486). **E44** (20:05) = E41's labels with
the 256 px seed-1 recipe, 2-GPU job 1020225 (~23:00) → third member of the public-table family, used if E41
wins on the LB tomorrow.

**E42 complete** (22:05, public-table base + round-4 teacher, 384 px): silver 0.9552, gold 0.914 (E41, 224 px:
0.9510 / 0.906; E39, round 5 Qwen base, 384 px: 0.9566 / 0.922). Version 18 (Version 16 + E37) submitted by the
user at 20:45 (ref 56804782, confirmed as Version 18 on the Submissions page).

**E45** (22:10) = next round of the public-table family: teacher = mean OOF of E41 and E42
(`runs/2026-10-03_2210_teacher_mixpub_r5_oof_mean/`), E41 recipe (224 px, 4-table base, α 0.5), job 1020318
(~00:55). Launched ahead of E41's LB so that it is ready if the public-table family wins.

**E44 complete** (23:25, public-table base, 256 px seed 1): silver 0.9522, gold 0.906. Public-table family E41 + E42
+ E44 (rank blend) 0.9595 / 0.919. **E46** (23:30) = E45 at 384 px (E42 recipe, public-table teacher E41 + E42):
folds 0–1 job 1020382, folds 2–3 job 1020383 and fold 4 job 1020384 queued (`slurm/runs/e46_*.txt`).

**E45 complete** (00:35, public-table family, next round, 224 px): silver 0.9496, gold 0.898 (E41: 0.9510 / 0.906) —
the family drifts away from the Qwen labels our validation scores against, as expected; E45 alone vs E41
alone on the LB decides whether this family keeps gaining per round.

**E43 complete** (01:05, round 5, 256 px seed 1): silver 0.9553, gold 0.923. E46 folds 2–3 moved to one GPU,
sequential (job 1020456, ~06:30; no node had two free). Ensembles on our validation: Version 16's six 0.9585,
Version 18's seven 0.9590, rounds 3–5 with our labels (10 runs) 0.9594, public-table four 0.9596, all 14 runs
**0.9604**.

knee-submit versions for 2026-10-04 (dataset of 01:10: the 14 runs E30, E32–E39, E41–E45), one question each:

| version | MODELS | question |
|---|---|---|
| 20 | E38 | round 5 vs round 4 (E35 alone, 0.930) |
| 21 | E41 | public-table base + round-4 teacher vs our base (E38, E35) |
| 22 | E45 | one more public-table round (vs E41) |
| 23 | E39, E36, E38, E43, E35, E37, E32, E34, E30, E33 | best ensemble on our own labels (vs Version 16, 0.933) |
| 24 | the 14 runs, latest first | does the public-table family add to it (vs Version 23)? |

Order in 23/24 = latest rounds first, since `knee.infer.run` skips the last models if the 8 h budget runs out.

Versions 20–22 pushed and checked (02:00, one run each, ~0.9 min on 3 studies); 23 and 24 in progress.
E46 folds 0–1 done (02:00). **E47** (02:05) = E38 recipe on round-6 labels, teacher = mean OOF of the round-5 runs
E38, E39, E43 (`runs/2026-10-04_0205_teacher_r5_oof_mean/`), job 1020474 (~04:50) — ready if Version 20 says
round 5 still helps.

Versions 23 (10 runs, 3.1 min on 3 studies) and 24 (14 runs, 5.8 min) pushed and checked at 02:05: every run
loaded and predicted, none skipped. All five versions of the 2026-10-04 plan are ready.

E46 fold 4 done (03:00). **E48** (03:05) = E39 recipe (384 px) on round-6 labels (same teacher as E47), folds 0
then 1 on the one free GPU (`slurm/runs/e48_01.txt`, ~08:20); folds 2–4 once Version 20 shows whether round 5
still helps.

**Version 18 (Version 16 + E37): public LB 0.934** (scored 03:05) — +0.001 for E37 (ConvNeXt-small, 384 px),
as its blend gain on our validation suggested. New best with our own labels.

E47 complete (04:45, round 6, 224 px): silver 0.9551, gold 0.930. E46 complete (05:50, public-table family next
round, 384 px): silver 0.9578, gold 0.920. E48 folds 0–1 done (07:40, round 6, 384 px): silver 0.9597 / 0.9571 per
fold (E39, round 5: 0.9559 / 0.9580; E36, round 4: 0.9594 / 0.9543). No job ran after 07:40: the session on
nodemm01 ended at 03:10 (node no longer reachable for the user), so nothing launched the next steps.

**LB of the 2026-10-04 plan** (all five submitted by the user at 07:30 UTC, refs 56817552–73; versions matched
through `scriptVersionId`, pushed in order 20 → 24):

| version | content | public LB | question → answer |
|---|---|---|---|
| 20 | E38 (round 5, 224 px) | 0.929 | round 5 vs round 4 (E35, 0.930): no gain — the Qwen-base rounds have saturated |
| 21 | E41 (4-table base + round-4 teacher) | **0.931** | best single model, +0.001 over E35 |
| 22 | E45 (4-table base, one more round) | 0.929 | no gain over E41: the public-table family saturates after one round too |
| 23 | 10 runs, own labels, rounds 3–5 | 0.934 | = Version 18 (7 runs): round-5 members add nothing |
| 24 | Version 23 + E41, E42, E44, E45 | **0.935** | new best: the public-table family adds +0.001 as ensemble members |

Conclusions: label bootstrapping is done (rounds 5 and 6 are not worth more GPU-hours as single models);
gains now come from ensemble diversity (another label base), +0.001 per batch. On our validation, Version 24's
14 runs blend to 0.9604 / 0.927; + E46 + E47 0.9609 / 0.928 (+0.0005, the size of E37's gain that gave +0.001 LB).

### E49 / E50 — public-table members at 384 px; queue blocked (2026-10-04 22:10)

Leave-one-out on the 16-run blend (Version 24's 14 + E46 + E47, silver 0.9609): removing E46 costs −0.0005,
E42 −0.0003, E44 −0.0002, E37 −0.0001, every other run ≤ 0 — the 384 px public-table runs are the most
useful members. **E49** = E42's labels (LB-validated through E41) with ConvNeXt-small at 384 px (E37 recipe),
**E50** = E42 recipe with seed 1 (`slurm/runs/e49_*.txt`, `e50_*.txt`). E50 fold 2 (job 1021315) and E49 fold 4
(1021318) started at once as 1-GPU jobs (30G); the 2-GPU jobs for the other folds (1021313/14/16/17) are planned
by Slurm for 2026-10-08 even at 28G and 3.5–5 h: our fairshare priority (2845) is the lowest on the 3090
partition, behind 11 jobs of revol-24 (64G each, one per node) and 6 of hle-25. A 1-GPU test job submitted at
22:12 also pended (cancelled). Training memory: ~3–6 GB anonymous per training; the cgroup fills with the
page cache of the unpacked cache (reclaimable).

knee-submit **Version 25** (Version 24 + E46 + E47, 16 runs, `['*e46*', '*e39*', '*e42*', '*e36*', '*e47*', ...]`)
being pushed (dataset of 22:00). **knee-blend v3** built (not pushed: the user pushes third-party code):
public 0.85 + Version 24's 14 runs 0.15 at 12 windows (v2: 0.929 set at 0.3 → 0.942, public alone 0.943).

**Public pipeline vs ours** (code reading of `kaggle/blend/public_src.ipynb` and the receipts of our blend
run, 2026-10-04): no input bug found on our side. Series are picked from the CSV plane / fat-sat flags (never
from descriptions), laterality behaves (the 73 studies with unknown side score like the rest; swapping their
medial/lateral predictions drops medial meniscus 0.969 → 0.907). The public CoAt readers score 0.90–0.93 on
the 58 gold studies (checkpoints picked on them), and the notebook's own single Raptor model is 0.924 LB vs
0.931 for E41: its 0.943 comes from blending five families with different inputs. Differences worth testing,
most likely first: (1) pooling inside each window — their CoAt readers use per-finding gated attention over
the 12×12 feature map, A5 label queries over patch tokens; we average-pool each window (`knee/model.py`);
(2) our `sag_t` slot mixes T1 (38 %), PD (35 %) and T2 (25 %) — a contrast embedding from TR/TE; (3) 0.34
mm/px (384 px over 130 mm) vs our 0.39; (4) depth as channels (A5: 16 slices as 16 channels) — for
diversity; (5) top-3 checkpoint averaging; (6) intensity clip p2–p98.

### E51 — avg + max pooling inside each window (2026-10-04 22:45, P100)

`--pool avgmax` (`knee/model.py`): the window embedding concatenates the backbone's own average-pooled head
output with a max over the feature map passed through the same head norm (ConvNeXt's LayerNorm2d); timm's
`catavgmax` breaks ConvNeXt's NormMlpClassifierHead. Default stays `avg` (old checkpoints load unchanged;
`knee/infer.py` reads `pool` from config.json). E51 = E41 recipe with `--pool avgmax`, folds 0–1, on two P100s
(job 1021332, ~35 min/epoch, ~06:00) — the only free GPUs. Against E41 folds 0–1 (same labels, 3090/bf16 vs
P100/fp16: precision differences are within seed noise, E11 on P100 vs E07).

knee-submit **Version 25** pushed and checked (22:45): all 16 runs loaded and predicted, 4.6 min on the 3 visible
studies (dataset of 22:07). `logs/push_version.sh` waited 30 min for nothing: `kaggle datasets files` lists 20
files per page, so the last run's fold4 was never seen; now `--page-size 200`. Two blend notebooks ready for
the user to push (third-party code): `kaggle/blend/` (v3, our weight 0.15) and `kaggle/blend_w25/` (same kernel,
weight 0.25), both with Version 24's 14 runs at 12 windows → one LB run each brackets the weight.

Slurm checks (22:15–22:30): `sbatch --test-only` on the 3090 partition gives a 2026-10-08 start even for a 1-GPU,
14G, 3.5 h job — the backfill plan serialises revol-24's 64G jobs per node and hle-25's 1-day jobs before ours, so
lowering memory/time does not help; only jobs ending early can free a slot sooner. The 2-GPU jobs stay queued in
case. `scontrol show assoc_mgr`: no per-user GPU cap in qos-student, MaxJobsPU = 4 (the interactive job counts).

E51 epochs 1–2 (P100, 33 min/epoch, end ~05:30): silver 0.8419 → 0.8960 (fold 0), 0.8285 → 0.8868 (fold 1); E41 on
the same folds: 0.8445 → 0.9012, 0.8505 → 0.8893 (final 0.9518 / 0.9536) — early epochs, not a verdict.
E51 folds 2–3 (job 1021394, 2 P100) and 4 (1021395, 1 P100) queued at 23:08 behind the job limit: they start when
E50 fold 2 / E49 fold 4 end, so a full 5-fold E51 can go to the LB alone against E41 (0.931) if folds 0–1 hold.

**Leaderboard snapshot** (2026-10-04 21:58 UTC, `kaggle competitions leaderboard -d`): 5,161 teams; ours rank 1,404
with 0.942 (knee-blend v2 of 10-02; our own models alone: 0.935). Teams at or above 0.943: 1,352 — the public
notebook score, copied by ~1,050 teams between 0.943 and 0.945; ≥ 0.945: 297; ≥ 0.950: 110; ≥ 0.955: 50;
≥ 0.960: 5; first 0.963. Public notebooks sorted by score (`kaggle kernels list --sort-by scoreDescending`):
the top four (pjmathematician `rsna-knee-d4-blend` / `d4-lite`, aastikrajan15 `knee-s75-w50`, evgendvorkin
`rsna-versia-5`) all state 0.943 or less and use the same sources (sofiaanjenje e11/e13 kernels, DINOv2) as the
yamadan96 notebook we blend with — no better public base exists; 0.945 (top 300) needs something of our own on top.

**knee-blend version 3** (public 0.85 + Version 24's 14 runs 0.15) pushed by me at 00:01 on 2026-10-05 with the
user's go ("pousse les blends"; from now on the user only clicks Submit, every push is mine); version 4 (weight
0.25, `kaggle/blend_w25/`) follows once version 3's visible-test run is checked.

knee-blend version 3 checked (00:17): public pipeline then our 14 runs on GPU (2.4 min), "BLEND written: public 0.85 +
ours 0.15", 0.11 h in total (`logs/blend_1005_0001/`). Version 4 (weight 0.25) pushed at 00:17.

**E50 fold 2 done** (00:19; E42 recipe, seed 1, 384 px): silver 0.9543, gold(fold) 0.919 — E42 (seed 42) on the same
fold 0.9567 / 0.923, E46 0.9580 / 0.922: one fold of seed noise, −0.002. On the 881 fold-2 studies: E42 + E50
0.9596 vs E42 + E46 0.9608; the 16-run blend 0.9607 → 0.9609 with E50 (+0.0002). A seed replicate adds little;
E51 folds 2–3 took the freed slot (job 1021394, node19, P100, ~07:30).

knee-blend version 4 checked (00:35): 14 runs predicted, "BLEND written: public 0.75 + ours 0.25", 0.14 h
(`logs/blend_1005_0017/`). Plan for 2026-10-05 (from 02:00 Paris): knee-blend v3 (0.15) and v4 (0.25) — the two
weights bracket the blend against the public 0.943 —, knee-submit Version 25 (Version 24 + E46 + E47, vs 0.935);
two submissions kept for E51 (full 5 folds alone vs E41, 0.931) if folds 0–1 hold.

**E49 fold 4 done** (01:53; ConvNeXt-small 384 px, E42 labels): silver 0.9454, gold(fold) 0.943 — below ConvNeXt-tiny
on the same labels (E42 0.9486) and below E37 (small, round 3: 0.9499) on this fold. On the 881 fold-4 studies:
E42 + E49 0.9529 vs E42 + E37 0.9538; the 16-run blend 0.9541 → 0.9544 with E49 (+0.0003). Second ConvNeXt-small
fold weaker than tiny (E40 fold 0 was the first): the other E49 folds (3090 jobs 1021313 / 1021316, ~16 GPU-hours)
cancelled; E50's queued 3090 folds kept (cheaper seed member). E51 fold 4 took the freed slot (job 1021395, P100).
E51 at epoch 4: fold 0 0.9315, fold 1 0.9272 (E41: 0.9334 / 0.9331) — trailing so far.

**E51 folds 0–1 done** (05:10; avg + max pooling, E41 recipe, P100): silver 0.9529 / 0.9470 (E41: 0.9518 / 0.9536);
pooled over the 1,764 studies 0.9497 vs 0.9524 (−0.0027, paired bootstrap 95 % CI [−0.0054, +0.0003]), gold 0.894
vs 0.906. Per finding: lateral meniscus −0.015, PF OA −0.012, MCL −0.009, medial meniscus −0.005; fracture +0.007,
contusion +0.003, ACL +0.002. The 16-run blend 0.9618 → 0.9617 with E51 (folds 0–1 studies). **Max pooling inside
the window does not help** (P100/fp16 vs 3090/bf16 is a confound, but not one that explains −0.007 on a fold):
✗, folds 2–4 (jobs 1021394/95, half-way) cancelled at 05:15. `--pool avgmax` stays in the code, default `avg`.

3090s free again at 05:10 (hle-25's jobs on node40 ended; 5 GPUs free): E50 folds 0–1 started on node40 (job
1021314, 3.5 h limit) and folds 3–4 on nodemm01 (1021317) once the E51 jobs freed the slots.
**E52** (05:20) = E42 recipe with **ConvNeXt V2-tiny** (`convnextv2_tiny.fcmae_ft_in22k_in1k_384`: masked-autoencoder
pretraining, IN-22k fine-tune at 384 px) — a different pretraining for ensemble diversity, against E42 per fold
(`slurm/runs/e52_*.txt`): fold 4 started on nodemm05 (job 1021713), folds 0–1 / 2–3 (1021711/12) queued behind the
job limit (4 running jobs, the interactive one included).

E52 at bs 8 ran out of GPU memory at the first training step (05:19, job 1021713: ConvNeXt V2's GRN layers allocate
4.2 GB tensors at stage 1 for 8 × 5 × 8 windows of 384 px; 13.2 GB allocated, 6.6 GB reserved but fragmented).
Relaunched at 05:22 with `--bs 6` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (jobs 1021717–19; fold 4
started on nodemm05, folds 0–1 and 2–3 queued) — a recipe difference against E42 (bs 8) to keep in mind.

E52 at bs 6 ran out of memory too (05:25, backward pass: 21.2 GB in use, 3.2 GB more asked). Relaunched at
**bs 4** (05:27, jobs 1021722–24): fold 4 trains on nodemm05 at 20.4 GB / 24.6 GB, GPU at 100 % after 2 min.
E52 therefore differs from E42 by backbone and batch (4 vs 8, same lr, twice the steps).

**E52 abandoned** (05:57): ConvNeXt V2-tiny at bs 4, fold 4 epoch 1: loss 0.618, **silver 0.484 / gold 0.484** — chance
level, i.e. collapsed outputs (ConvNeXt V1 runs: loss ~0.58, silver ~0.85 after epoch 1), at 25 min/epoch (2× V1).
Not debugged (suspects: lr 2e-4 too high for FCMAE weights, GRN in bf16 over 96×96 maps); the 3090 window is worth
more on a known member. ✗ for now. **E53** (06:00) = E46 recipe with seed 1 (E46 = most useful member of the 16-run
blend), `slurm/runs/e53_*.txt`: fold 4 on nodemm05 (job 1021738), folds 0–1 / 2–3 (1021736/37) queued.
E50 at epoch 3 (05:55): folds 0 / 1 / 3 / 4 silver 0.918 / 0.912 / 0.928 / 0.902.

**E50 folds 0–1 done** (07:33): silver 0.9540 / 0.9566, gold(fold) 0.929 / 0.932 (E42, seed 42: 0.9577 / 0.9550) —
seed noise both ways. E53 folds 0–1 took node40 (job 1021736); E53 fold 4 at epoch 1: silver 0.832 (normal).

Public write-ups found on GitHub (2026-10-05, web search; context only, nothing reused):
- `TranBaDat2607/RSNA-Knee-Abnormality-Detection`: public CoAtNet checkpoints (v5/v10/v8) + residual-gated CoAtNet at
  0.4 → **0.939 LB**, 0.930 on the 58 gold studies; labels from an OpenAI model (gold agreement 0.869).
- `raahimnawaz/rsna-knee-abnormality-detection`: ViT-L (DINOv3 continued-pretrained on knees), 7 slots, up to 82
  three-slice windows per study (96 slices at 384 px, 140 mm FOV) → **0.929 alone**; rank-averaged into the public
  0.943 ensemble as an extra family → **no change (0.943)**; OOF image teachers mixed into report labels gave at most
  +0.005 LB; ViT-L and CoAtNet on the same labels within 0.005 of each other ("labels/recipe > architecture");
  the 58 gold studies overstated LB gains, while a 691-study held-out fold correlated 0.996 with the LB.
Our numbers sit in the same place (0.931 single, 0.935 ensemble, 0.942 blended), and the second write-up warns that
an extra family may not move the 0.943 public ensemble — the blends submitted today (v3 / v4) test exactly that.

Submitted by the user at 07:36–07:37 Paris (2026-10-05): knee-blend versions 3 (ref 56843645) and 4 (56843667),
knee-submit Version 25 (56843671) — matched through `scriptVersionId` (knee-blend 355281624 < 355283800 = pushed
00:01 < 00:17; knee-submit 355269060 = Version 25, pushed 22:45). Two submissions left today (E51 abandoned).

**E50 complete** (07:40; E42 recipe, seed 1, 384 px, public-table labels): silver 0.9551, gold 0.911 over 5 folds
(E42, seed 42: 0.9552 / 0.914) — a seed replicate. Version 25's 16 runs blend to 0.9609 / 0.928; with E50 (17 runs)
0.9611 / 0.927: +0.0002, below what moved the LB so far (+0.0005 for +0.001) — not worth a submission alone.
E53 folds 2–3 started (job 1021737, nodemm01) when E50's job ended; E53 folds 0–1 (1021736) and 4 (1021738) running.

**Kaggle inference path replayed on validation folds** (08:00, `scripts/check_infer.py`, P100 = fp16-only like the
T4s). Motive: on the 58 gold studies our single models sit near Archit Konde's (~0.93 vs his 0.930 OOF), yet he
scores 0.950 on the LB and we 0.931 — a gap that could hide a train/test mismatch (58 studies: weak evidence).
Checked by reading: `test_series.csv` has the same columns, dtypes and values as `train_series.csv` (slot choice
identical); `knee/infer.py` decodes with the training functions and defaults (150 mm, 64 slices, `study_side`
over header-only series ≡ the training meta). Checked by running: fold 0 through the inference code path in fp16
(Kaggle) vs the run's own OOF (bf16, `scripts/train.py`), same 12 windows:

| run | studies | OOF silver | fp16 replay silver | mean per-label correlation | max abs diff | non-finite logits |
|---|---|---|---|---|---|---|
| E35 (224 px) | 120 | 0.9586 | 0.9587 | 1.0000 | 0.0073 | 0 |
| E36 (384 px) | 60 | 0.9630 | 0.9629 | 1.0000 | 0.0054 | 0 |
| E37 (ConvNeXt-small 384) | 60 | 0.9673 | 0.9677 | 1.0000 | 0.0074 | 0 |

No precision or code-path loss: the LB gap to the top single models is not an inference bug. (Remaining
unchecked link: the remote-archive DICOMs vs the files Kaggle mounts — same bytes by construction.)

### E54 — full-data training (2026-10-05 08:30)

Every network so far trained on 4 folds (80 % of the studies) and the submission averages the 5 fold networks.
`scripts/train.py --full-fit` trains each network on all 4,407 studies instead (one network per `--folds` entry,
seeded by it; the printed silver/gold are in-sample, so no `oof_fold*.csv` is written and the run cannot feed
`eval_oof`, teachers or blends; checkpoints keep the `fold<k>.pt` names that `knee.infer` averages). Smoke test on
the P100 (ResNet-34, 3 steps): checkpoint and `full0_history.json` written, no OOF. **E54** = E41 recipe (224 px,
LB 0.931 alone) with `--full-fit`, 3 networks, one 3090 in sequence (`slurm/runs/e54.txt`, queued behind the job
limit, ~3.5 h) → E54 alone against E41 alone: 25 % more data per network vs 5 networks averaged over 3.

Auto (2026-10-06 00:23, scripts/auto_followup.py): Kaggle: ref 56843645 (knee-blend version 3) scored 0.944; ref 56843667 (knee-blend version 4) scored 0.943; ref 56843671 (knee-submit version 25) scored 0.936.

**LB of 2026-10-05** (read 2026-10-06 00:25): knee-blend v3 (public 0.85 + Version 24's 14 runs 0.15) **0.944** — the
first blend above the public pipeline alone (0.943): our ensemble now adds to it, at a low weight; v4 (weight 0.25)
0.943; v2 (0.929 set at 0.3) was 0.942 → the optimum weight is ≤ 0.15. knee-submit Version 25 (16 runs) **0.936**
(+0.001 over Version 24: E46 + E47). Our best with our own labels only stays Version 18/23 (0.934).

**E53 complete** (E46 recipe, seed 1, 384 px): silver 0.9573, gold 0.922 (E46: 0.9578 / 0.920); E46 + E53 0.9618.
Added to Version 25's 16 runs on OOF: +E53 +0.0004, +E53 +E50 **+0.0006** (0.9609 → 0.9615).
**E54 complete** (E41 recipe, full data, 3 networks, ~65 min each): in-sample monitor silver 0.978 (not a
validation); only the LB can judge it. No job ran after E53/E54 ended (~11:00 on 2026-10-05): the unattended
follow-up job was being written when the user left, and was not submitted.

**Plan for 2026-10-06** (00:30): knee-submit **Version 26** = E54 alone (full-data training vs E41 alone, 0.931) and
**Version 27** = Version 25 + E53 + E50 (18 runs, +0.0006 on OOF) — being packaged (dataset with the 19 runs). knee-blend
with the same 18 runs at our weight **0.15** (`kaggle/blend/`, vs v3 0.944: a better set at the best weight so far)
and **0.10** (`kaggle/blend_w10/`, new folder, notebook git-ignored like the other copies of public code): 0.25 < 0.15,
so the optimum may sit lower. `kaggle/blend/knee-blend-v3.ipynb` keeps the v3 build. **E55** (00:50) = E46 recipe with
seed 2 (`slurm/runs/e55_*.txt`, jobs 1023235/36/37).
**Unattended follow-up**: `slurm/auto_1006.sbatch` (job 1023238, CPU partition, nodecpu10; Kaggle API and GitHub reachable
from there) waits for the Claude session's knee-submit pushes (marker `logs/day10_pushed`, else redoes them at 02:30),
pushes both blends (`logs/push_blends.sh`, which now waits for the new version to start before waiting for its end),
writes `kaggle/A_SOUMETTRE.md`, then logs Kaggle scores and E55's OOF gain every 10 min (`scripts/auto_followup.py`).

**Versions 26/27 pushed** (01:02): the dataset now holds the 19 runs (models.zip 10 GB, packaged 00:24, pushed after Kaggle processed it). knee-submit **Version 26** (E54 alone) ran its 3 full-data networks on the visible test (1.3 min); **Version 27** (18 runs, MODELS in the order of Version 25 with E53 after E46 and E50 after E42) ran all 18 (5.1 min, no run skipped by the time budget); logs in `logs/push_1006_0024` and `logs/push_1006_0056`. Marker `logs/day10_pushed` written at 01:03; job 1023238 read it at 01:03 and went on to the blends.

Auto (2026-10-06 01:37, scripts/auto_followup.py): knee-blend versions 5 (public 0.85 + 18 runs 0.15) and 6 (public 0.90 + 18 runs 0.10) pushed and run on the visible test: 2 of 2 wrote the blend (logs/day10_blends.txt).

Auto (2026-10-06 05:38, scripts/auto_followup.py): **E55 complete** (E46 recipe, seed 2, 384 px): silver 0.9558, gold 0.9196. OOF gain on Version 25's 16 runs: +0.0008 V25 0.9609/0.9280 -> +e53,e50,e55 0.9617/0.9273.

`kaggle/submissions.csv` (2026-10-02 16:30): ledger of every Kaggle submission (API ref, date, notebook
version, models, public score), so the LB scores quoted here can be checked against the Kaggle page.
The AUCs in this journal can only be recomputed where the competition data lives (`data/` is never
versioned: rule on redistribution): `python -m scripts.eval_oof runs/<run>` on the cluster, from the
committed out-of-fold predictions.
