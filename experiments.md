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

Auto (2026-10-06 07:59, scripts/auto_followup.py): Kaggle: new submission ref 56871945 (knee-submit, scriptVersionId 355580807); new submission ref 56871950 (knee-submit, scriptVersionId 355582643); new submission ref 56871958 (knee-blend, scriptVersionId 355583502); new submission ref 56871962 (knee-blend, scriptVersionId 355585333).

Submitted by the user at 07:57 Paris (2026-10-06): refs 56871945 / 56871950 = knee-submit Versions 26 / 27 and
56871958 / 56871962 = knee-blend versions 5 / 6 (matched through `scriptVersionId`, increasing with push order).
E55 added alone to Version 25's 16 runs: +0.0003 on OOF (0.9609 → 0.9612): a third seed of E46 adds less than the
second (E53 +0.0004) — seeds of one recipe are running out.

### E56 / E57 — full-data 384 px, sequence embedding (2026-10-06 08:00)

The 3090s sat idle from 05:38 (E55 done) to 08:04. Two jobs (Slurm here: 2 GPUs per job at most,
QOSMaxGRESPerJob — a 3-GPU job is refused; 4 CPUs per GPU; 4 running jobs per user, knee_auto and the
interactive session included):
- **E56** = E46 recipe (384 px, 4-table base, round-5 mixpub teacher; the most useful member) with `--full-fit`,
  networks 0–1 in parallel (job 1023550, nodemm05, ~5.5 h; network 2 in `slurm/runs/e56_2.txt` if E54 wins).
  Launched ahead of the LB verdict on full-data training (Version 26 = E54 alone vs E41 alone 0.931, scored ~11:00):
  if it wins, E56 joins or replaces E46 in the next Version.
- **E57** = E41 recipe (224 px) + `--seq-emb`, folds 0–1 (job 1023557, node40, ~1 h per fold), against E41 per
  fold (0.9518 / 0.9536). Motive (item 2 of the public-vs-ours list of 2026-10-04): a slot can hold several
  weightings and the model is not told which — `sag_t` holds PD 35 %, T1 32 %, T2 20 %, gradient echo 7 % of the
  training series, `ax_fs` PD 68 %, T2 12 %, STIR 9 %, GRE 6 %. `knee/data.py` `sequence_classes` maps the header
  times to unknown / T1 / PD / T2 / STIR / GRE (TI 80–300 ms → STIR, ScanningSequence GR → GRE, TR < 1000 ms → T1,
  TE ≥ 60 ms → T2, else PD; 5.5 % unknown, no TR/TE); `assign_slots` adds `<slot>_seq`, `KneeNet(seq_emb=True)`
  adds a zero-initialised embedding of it to the slot's tokens. Test time reads the same header fields from the
  same `load_volume` meta that built `series_meta.parquet` (`knee/infer.py`), so train and test classes match.
  Old checkpoints are unchanged: `check_infer` on E35 fold 0 (120 studies, fp16) gives the same numbers as on
  10-05 (0.9586 / 0.9587, correlation 1.0000, max |diff| 0.0073); a ResNet-34 smoke run (3 steps, P100) writes
  `"seq_emb": true` and its checkpoint reloads with every key. Commit 9b98f70.

The user leaves their PC on from now on (2026-10-06 08:30, "quartier libre"): the Claude session stays up, so
**knee_auto (job 1023238) was cancelled at 08:26** — it held one of the 4 job slots for a 10-min score poll; the
poll now runs in the session (`python -m scripts.auto_followup scores`). Its slot took **E57 folds 2–4** (job
1023567, 1 GPU, folds in sequence, ~3 h; a 2-GPU job pended on Resources: only single 3090s were free), launched
before the folds 0–1 verdict and to be cancelled if they show no gain. E57 at epoch 3: silver 0.9198 / 0.9184
(E41: 0.9228 / 0.9222).

### Label sources on the gold studies, per finding (2026-10-06 08:40)

Gold AUC (58 studies) of each training-label source, before any image model (`knee.labels`, public tables,
teacher OOF; per finding, then macro):

| source | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Baker | Cont | Frac | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen report | 0.986 | 0.984 | 0.951 | 0.863 | 0.888 | 0.830 | 0.868 | 0.804 | 0.746 | 0.911 | 0.809 | 0.801 | 0.870 |
| steven v2 | 0.993 | 0.964 | 0.954 | 0.862 | 0.931 | 0.830 | 0.901 | 0.853 | 0.790 | 0.946 | 0.831 | 0.793 | 0.887 |
| 4-table mean | 0.989 | 0.980 | 0.959 | 0.879 | 0.914 | 0.809 | 0.891 | 0.865 | 0.769 | 0.928 | 0.864 | 0.805 | 0.888 |
| teacher r4 (OOF) | 0.957 | 0.941 | 0.969 | 0.906 | 0.994 | 0.861 | 0.856 | 0.981 | 0.847 | 0.964 | 0.943 | 0.921 | 0.928 |
| E41 labels (α 0.5) | 0.978 | 0.975 | 0.980 | 0.925 | 0.981 | 0.843 | 0.902 | 0.993 | 0.843 | 0.971 | 0.914 | 0.903 | 0.934 |
| V25 ensemble OOF | 0.950 | 0.950 | 0.970 | 0.889 | 0.992 | 0.853 | 0.864 | 0.983 | 0.839 | 0.976 | 0.947 | 0.922 | 0.928 |

Reports are near-perfect for the ligaments (ACL 0.99, MCL 0.98) where the image models are not (0.93–0.95), and
weak where the image shows more than the text (effusion, synovitis, contusion, fracture, medial OA). A single α
mixes the teacher's ligament errors into labels that were right. Curves of (1 − α)·4-table + α·teacher r4 per
finding (α 0 → 1): ACL/MCL best at 0.1–0.3 (0.988/0.984), menisci 0.5–0.7, medial OA → 1, lateral OA 0.8,
PF OA flat 0.3–0.7, effusion 0.5–0.6, synovitis 0.7, Baker's 0.5–0.6, contusion/fracture 0.8; uniform α 0.5
0.934, 0.7 0.940, 0.8 0.940. Weak findings for every source and model: lateral OA (≤ 0.86), synovitis (≤ 0.87),
PF OA (≤ 0.91), lateral meniscus. Patient age (would help OA) is absent from every header (`PatientAge` null);
sex is present (M/F/O).
**E58** (queued, job 1023582, starts when E57 folds 0–1 free their GPUs) = E41 with `--bootstrap-alpha-labels`
(new, `scripts/train.py`; `_` for spaces in finding names): ACL 0.2, MCL 0.2, MM 0.5, LM 0.6, MOA 0.9, LOA 0.8,
PFOA 0.4, effusion 0.6, synovitis 0.7, Baker's 0.6, contusion 0.8, fracture 0.8 — family-smoothed choices from
the curves; label gold agreement 0.934 → 0.944 (chosen on the same 58 studies, so optimistic). Only the LB can
judge a label change: E58 alone against E41 alone (0.931). Label change per cell vs α 0.5 (non-gold studies):
0.015 (Baker's) to 0.072 (medial OA) on average.

### Inference throughput: is knee-submit cut by its 8 h budget? (2026-10-06 08:35)

`knee.infer.run` predicts one model after another on **one** GPU, re-reading and re-warping every series for
each model, and skips the last models when the measured pace says they would end past 8 h. knee-submit runs on
Kaggle's T4 × 2 machine (`machine_shape` NvidiaTeslaT4 in the pulled metadata), so one T4 sits idle; the
hidden-test log is invisible, so whether Versions 25/27 ran all their models is unknown. Version 9 (35 networks,
three 384 px models, 24 windows) took ~5 h on the hidden test; Version 27 holds ~2.8× its compute (eight 384 px
models, ConvNeXt-small among them) — likely past the budget, i.e. the tail of the list may never run.
- **knee-bench** (`kaggle/bench/`, private notebook, version 1 pushed 08:37): 60 training studies posed as a test
  set on the same T4 × 2 image; decode time, per-model time at 24 and 12 windows (384 / 256 / 224 px, ConvNeXt-small,
  RadImageNet), DataLoader alone (CPU bound?), two models at once on the two T4s.
- **Grouped two-GPU inference** (`knee.infer.run(grouped=True)`, default off): models sharing an input (cache,
  slice size, windows, span, zoom) run together — each batch read once, every fold network of every model in the
  group spread round-robin over the GPUs, launches interleaved so both compute at once; a first pass on 16
  studies measures the pace, the models that fit the budget are kept in the given priority order, and the pace
  is re-checked before each group. On the P100 (24 training studies, E35 + E41, 12 windows) it gives **exactly**
  the predictions of the per-model path (max |diff| 0.0, one or two device slots) — 58 s vs 70 s on one GPU.

Auto (2026-10-06 08:46, scripts/auto_followup.py): Kaggle: ref 56871945 (knee-submit version 26) scored 0.935.

**Version 26 on the public LB: 0.935** (scored 08:46) — **E54 alone**, i.e. the E41 recipe trained on all 4,407
studies (3 networks, seeds 42/43/44), against **0.931** for E41 alone (5 fold networks on 80 % each): **+0.004**,
the largest single-model step since the round-3 labels, and as much as the whole 14-run Version 24 (0.935). More
data per network beats averaging five 80 % networks here (noisy report labels: data-hungry), and every gold study
(58, weight 3) is now in training instead of ~46. Full-data training becomes the default for submissions; 5-fold
runs stay for OOF (teachers, ensemble selection), full-data ones go to Kaggle. Queue changes (08:50):
- **E58** switched to full data before it started (job 1023584, 2 GPUs: networks 0–1 / 2, seeds as E54), so the
  per-finding α is compared with E54 (0.935) — same seeds, only the labels differ; the 5-fold version (1023582)
  was cancelled while pending.
- **E56 network 2** queued (job 1023585, 1 GPU, `slurm/runs/e56_2.txt`): three full-data E46 networks like E54.
- E46-recipe network on a full 3090: ~15 min/epoch on all studies (12 on 80 %), ~3 h per network.

**knee-bench version 1** (COMPLETE 09:00, 23 min, `logs/bench_1006_v1/`): Kaggle T4 × 2, 4 CPUs, 31 GB RAM; 60
training studies (320 series) as a test set.

| step (24 windows unless said) | s / study | per network |
|---|---|---|
| decode both cache sizes (4 processes) | 1.11 | — |
| E46, ConvNeXt-tiny 384 px, 5 networks, 1 T4 | 3.01 (12 windows: 1.47) | 0.60 |
| E43, 256 px | 1.45 (0.69) | 0.29 |
| E41, 224 px | 1.01 (0.53) | 0.20 |
| E37, ConvNeXt-small 384 px | 4.67 (2.35) | 0.93 |
| E33, RadImageNet ResNet-50 224 px | 0.82 (0.39) | 0.16 |
| DataLoader alone, 384 px / 224 px | 0.11 / 0.05 | — |
| two 384 px models at once, one per T4 | 179 s wall vs 361 s in sequence | — |

GPU-bound (the loader needs 4 % of the model time) and the second T4 doubles the throughput. Version 9 (7 models,
13.5 s/study + decode) took ~5 h on the hidden test → **≈ 1,250 hidden studies**. Version 27's 18 models need
37 s/study, ~13 h on one T4: its budget rule (skip a model when elapsed + 1.2 × predicted > 8 h) ran only the
first **~9** (e46, e53, e39, e42, e50, e36, e47, e38, e41), Version 25 the first ~12. It cost little: on OOF the
cut ensembles score as the full lists (V27: 9 models silver 0.9621 / gold 0.928 vs 18 models 0.9615 / 0.928; the
tail = older rounds), so Versions 25/27 measured mostly their first models. The second T4 matters for what comes
next: more strong (full-data) models within 8 h.
Greedy forward selection on OOF silver over the 32 complete 5-fold runs: E46 0.9578 → +E53 0.9618 → +E55 0.9627
→ +E37 0.9631 → +E42 0.9632, then flat (gold 0.920 → 0.929): three seeds of the E46 recipe carry most of it,
ConvNeXt-small (E37) is the useful different member. Full-data priorities follow: E46 recipe (E56, more seeds),
then ConvNeXt-small 384, then E42.
**knee-bench version 2** pushed (09:05): the repository's new `knee/infer.py` (grouped, two T4s) replaces the
dataset's copy; the whole Version 27 list grouped on the same 60 studies, peak GPU memory, per-model predictions
compared with the old path (E46, E41, E37), and the budget rule with a 300 s budget.

**E57 folds 0–1 done** (09:07; sequence embedding, E41 recipe): silver 0.9538 / 0.9524 (E41 0.9518 / 0.9536), gold(fold)
0.926 / 0.895 (E41 0.922 / 0.914); pooled over the 1,764 studies 0.9533 vs 0.9524 (**+0.0009**, paired bootstrap 95 %
CI [−0.0023, +0.0045]), gold 0.901 vs 0.906. Per finding (silver): ACL +0.018, lateral meniscus +0.009, fracture
+0.006 / MCL −0.013, PF OA −0.010, contusion −0.009 — the sagittal gain fits the T1/PD/T2 mix of `sag_t`, the
rest looks like noise. **Not a clear gain**: folds 2–4 (job 1023567) cancelled at 09:09 for E58's GPUs — fold 2 was
at epoch 9 of 12 (silver 0.9503), lost (should have waited 12 min). `--seq-emb` stays in the code (default off);
it could be judged later on the LB as E54 + `--seq-emb` (full data, 3 networks, ~3.3 GPU-hours).
Queue at 09:10: E56 networks 0–1 (nodemm05) and 2 (node40), **E58** (node40, started 09:09), E59 waiting.

**E61** queued (job 1023610, 09:13) = E54 with `--gold-weight 10` (3 before): the 58 expert-labelled studies are the
only labels made like the test's (images, two MSK radiologists + adjudication); at weight 3 they are ~4 % of the
loss, at 10 ~12 % → alone against E54 (0.935); E54's own gain may partly come from seeing all 58 instead of ~46.
Put ahead of E59 (`scontrol update … Nice=100` on 1023605/06): a cheap 224 px question before more seeds.
Notebooks switched to the grouped two-T4 inference for the next pushes: knee-submit's cell calls
`run(..., grouped=True)`; `scripts/build_blend_nb.py` gives our part every GPU with ≥ 6 GB free (both T4s
had 14.7 GB free after the public part in blend versions 5/6) and `grouped=True`.

**knee-bench version 2** (COMPLETE 09:40, 33 min, `logs/bench_1006_v2/`) — the grouped two-T4 path on the same 60
studies, whole Version 27 list at 24 windows: groups 384 px (8 models) 757 s, 224 px (7) 232 s, 256 px (3) 140 s →
**18.8 s/study vs ~37 s/study** one model after another on one T4 (2.0×); peak memory 8.9 / 8.8 GiB per T4 (16 GB);
predictions **identical** to the old path (max |diff| 0.0 for E46, E41 and E37). Budget rule with a 300 s budget:
the 16-study calibration (41 networks loaded, 0.6 s each) left 2 min, so only E46 was kept and the call took 309 s
(3 % over: negligible against a budget of hours). On ~1,250 hidden studies the whole 18-model list would need
~6.5 h + 0.4 h decoding: it now fits, where one T4 ran only the first ~9.

**Queue for 2026-10-06** (09:40; Slurm runs pending jobs by priority, `--nice` sets the order). Running: E56
(full-data E46 recipe, 384 px, networks 0–1 nodemm05 ~11:15, network 2 node40 ~12:15), E58 (full-data E54 +
per-finding α, node40, ~11:20). Pending, in order: **E61** (E54 + gold weight 10), **E62** (E54 + `--seq-emb`,
`slurm/runs/e62.txt`: the LB test of E57's idea), **E59** (E56 with seeds 142–144), **E60** (E37 recipe —
ConvNeXt-small 384 px, round-3 labels — on all studies, `slurm/runs/e60_*.txt`, ~4.8 h per network).
Each 224 px question (E58, E61, E62) is one full-data run judged alone against E54 (0.935), seeds unchanged.
`logs/chain_v28.sh` (background) waits for E56 and E58's six networks, packages 22 runs (Version 27's 18, E54,
E55, E56, E58) with the grouped `knee/infer.py`, then pushes knee-submit **Version 28 = E56 alone** (full data at
384 px vs E54 at 224 px) and **Version 29 = E58 alone**.

**E56 and E58 complete** (E56 networks 0–1 at 11:03, network 2 at 12:01; E58 ~11:10; in-sample monitors only).
`logs/chain_v28.sh` then packaged the 22 runs (12:03) and pushed knee-submit **Version 28 = E56 alone**
(`logs/push_1006_1202`: 3 networks, one 384 px group, 35 s, 1.3 min on the visible test) and **Version 29 =
E58 alone** (`logs/push_1006_1230`: 3 networks, 30 s) — the first versions with the grouped two-T4 inference.
On ~1,250 hidden studies: 3 × 0.60 / 2 ≈ 0.9 s/study → ~20 min of models + ~25 min decoding.
E61 (gold weight 10) and E62 (sequence embedding) started at 11:03 / 11:13 (2 GPUs each, ~13:20 / 13:30);
E59 and E60 wait: all ten 3090s busy (six by other users) at 12:35.
One submission left today: Version 29 first (its answer decides whether E59/E60, not started, switch to
per-finding α), Version 28 tomorrow.

**E61 and E62 complete** (13:17, three full-data networks each). `logs/chain_v30.sh` packaged 24 runs (11 GB upload)
and pushed knee-submit **Versions 30 (E61) and 31 (E62): both ERROR on the visible test, never to be submitted**.
Cause: `logs/push_version.sh` waited for "<last run>/fold4" in the dataset listing, and the last run given was
E55, already in the previous version — the wait passed at once and both notebooks ran against the previous
dataset version, without E61/E62. MODELS matched nothing (`[]`), `prepare_cache` got no slice size (each series
failed with UnboundLocalError), `assign_slots` received an empty table → KeyError 'sag_fs'. Fixes: the script now
waits for the last run's own last checkpoint (`fold2.pt` for a full-data run) for up to 1 h and refuses to push
when a MODELS pattern matches no run of the dataset; `knee.infer.run` asserts that the model list is not empty.
Versions 28/29 were fine (E55 was new in their package). `logs/chain_v32.sh` waits for the processed version and
pushes E61 alone and E62 alone again (next version numbers: 32, 33).

The dataset version of 13:17 was processed at 13:38; knee-submit **Version 32 = E61 alone** (`logs/push_1006_1338`,
3 networks, 31 s) and **Version 33 = E62 alone** (`logs/push_1006_1342`, 28 s) checked on the visible test.
Ready to submit, each one change against E54 (0.935): **28** E56 (384 px), **29** E58 (per-finding α), **32** E61
(gold weight 10), **33** E62 (sequence embedding). E59 (seeds 142–144 of E56) started 13:13 / 13:16.

**E64** queued after E60 (jobs 1024059/60, nice 300) = E39 recipe on all studies: 384 px, **own labels only**
(Qwen base, round-4 teacher α 0.7; lineage checked: teacher r4 = E35 + E36 ← teacher r3 = E30/E32/E33 ← teacher r2 =
E23/E24/E25/E27, none with `--mix-public`) — the CV quotes own-label scores (best: Versions 18/23, 0.934) and E54's
gain suggests a full-data own-label set can beat it. E60 (ConvNeXt-small, round-3 labels) is own-label too.

Auto (2026-10-06 15:59, scripts/auto_followup.py): Kaggle: ref 56871950 (knee-submit version 27) scored 0.936.

**Version 27 on the public LB: 0.936** (scored 15:59, ~8 h after submission: it ran to its budget) — the same as
Version 25. As the bench predicted, the one-T4 path ran only its first ~9 models (E46, E53, E39, E42, E50, E36,
E47, E38, E41), and on OOF those score like the full lists; E53 + E50 replaced older tail models that never ran.
The 5-fold ensemble line is flat at 0.936; full-data single models (E54 0.935) are the way up.

Auto (2026-10-06 16:09, scripts/auto_followup.py): Kaggle: ref 56871962 (knee-blend version 6) scored 0.944.

**knee-blend version 6 on the public LB: 0.944** (public 0.90 + Version 27's 18 runs 0.10, scored 16:09) — same as
version 3 (0.85 + Version 24's 14 runs 0.15); version 5 (0.15 with Version 27's runs) pending. The blend sits at
0.944 with 5-fold ensembles as our part; the next step is a stronger part of ours.
**Evening chain** (`logs/chain_v34.sh`, started 16:10, log `logs/chain_v34.out`): when E59 and E60 have their three
networks each, package 26 runs (adds E59, E60; E60 last so the wait works) and push knee-submit **Version 34 =
full-data ensemble E56 + E59 + E60 + E54** (two E46-recipe 384 px trios, ConvNeXt-small 384 px, 224 px), then
rebuild and push **knee-blend version 7** = public pipeline + the same four runs, both T4s, 24 windows, weight from
`logs/blend7_weight` (0.15 unless version 5's score says otherwise; the version 5 build is kept as
`kaggle/blend/knee-blend-v5.ipynb`). Expected ~21:30.

Auto (2026-10-06 16:19, scripts/auto_followup.py): Kaggle: ref 56871958 (knee-blend version 5) scored 0.944.

**knee-blend version 5: 0.944** (public 0.85 + Version 27's runs 0.15, scored 16:19): weights 0.10 and 0.15 tie at
0.944 (0.25: 0.943, 0.30 with a weaker set: 0.942) → version 7 keeps 0.15. **E59 complete** (~16:10, three
full-data networks, seeds 142–144); E60 started 16:10/16:12 on three 3090s (~4.8 h per network → ~21:00).

**Night queue** (16:30, after E60 and E64): **E65** = E47 recipe on all studies (224 px, own labels: Qwen base +
round-5 teacher E38/E39/E43, α 0.7; `slurm/runs/e65.txt`, nice 400) — with E60 and E64, an own-label full-data set;
**E66** = E42 recipe on all studies (384 px, 4-table base, round-4 teacher; `slurm/runs/e66_*.txt`, nice 500).

**E60 complete** (21:04; E64 network 2 too, ~19:30). `logs/chain_v34.sh` packaged 26 runs and pushed knee-submit
**Version 34 = E56 + E59 + E60 + E54** (`logs/push_1006_21*`: one 384 px group of 3 models in 55 s + one 224 px group,
2.2 min on the visible test) and **knee-blend version 7** (public 0.85 + the same four runs 0.15, 24 windows, ours on
GPUs 0,1: "BLEND written", 0.11 h; `logs/blend_1006_2130`). Order for the user (`kaggle/A_SOUMETTRE.md`): Version 29
tonight if a submission is left; tomorrow 34, blend 7, 28, (29), 32; then 33. Running: E64 networks 0–1 (~00:10),
E65 (node40), E66 network 2 (nodemm01); E66 networks 0–1 waiting.

**Night of 2026-10-06/07** (read 01:16): the Claude session and the interactive job ended during the evening (new
interactive job 1025494 at ~01:15); the batch jobs were unaffected. **E64** (own-label E39 recipe, 384 px, full
data) and **E65** (own-label E47 recipe, 224 px, full data) complete, **E66** network 2 complete; E66 networks 0–1
(job 1024334) wait on Priority: all ten 3090s held by other users, our fair-share priority low after a full day
(P100s idle, but ConvNeXt is ~8× slower there). Version 29 was not submitted on 10-06 (one submission unused).
`logs/chain_v35.sh` (01:20) packages 28 runs (adds E64, E65, the last one new so the wait works) and pushes
knee-submit **Version 35 = own-label full-data ensemble E64 + E60 + E65** — no public table in any lineage, the
candidate to beat the CV's own-label 0.934.

Auto (2026-10-07 01:18, scripts/auto_followup.py): Kaggle: new submission ref 56892847 (knee-submit, scriptVersionId 355721387).

knee-submit **Version 35 = E64 + E60 + E65** pushed and checked (01:44, `logs/push_1006_*`/`push_1007_*`: the three
own-label full-data runs, one 384 px group of 2 models + one 224 px group, 1.8 min on the visible test). Version 29
was submitted by the user at 23:17 UTC on 10-06 (ref 56892847). Order for 10-07: 34, blend 7, 35, 28, 32.

Auto (2026-10-07 01:59, scripts/auto_followup.py): Kaggle: ref 56892847 (knee-submit version 29) scored 0.931.

**Version 29 on the public LB: 0.931** (scored 01:59) — **E58 alone** (E54 + per-finding teacher weight) against
**0.935** for E54: **−0.004**. Per-finding α chosen on the 58 gold studies raised the labels' gold agreement
(0.934 → 0.944) and lowered the LB: once more the gold set misleads label choices (cf. round 2, where +0.03 gold
gave +0.003 LB). Leaning the ligaments on the reports (α 0.2) likely teaches report wording rather than what the
image shows, while the test is read from the images. ✗ — uniform α stays; no queued or packaged run besides E58
uses `--bootstrap-alpha-labels` (E59, E60, E64, E65, E66 all uniform). E61 (gold weight 10) is the same kind of
bet on the gold set: Version 32 will tell.

### 2026-10-07: new questions while the 3090s are taken

E66 complete at 08:18 (E42 recipe, full data); the queue was then empty until 10:15. All ten 3090s are held by
other users and a higher-priority backlog (hle-25: 16 pending jobs at priority 3478 vs ~2850 for ours; our
fair-share factor 0.08 after the last days), while 11 P100s sat idle → today's questions run on P100s (ConvNeXt
~8× slower: ~40 min per epoch at 224 px on all studies, ~8 h per network), three 2-GPU jobs (4-job limit):
- **E70** = E54 + bigger window/series aggregator (`--dim 384 --depth 4`, vs 256 × 2), job 1026089 (node17), networks
  0–1; network 2 shares job 1026124 with E69's.
- **E69** = E54 + **auxiliary report-grade loss** (`--aux-grades 0.3`, new): the per-finding pooled vectors also
  predict the 6-way Qwen grade (absent / equivocal / mild / moderate / severe / not mentioned) with a cross-entropy;
  the head (`aux_grade_head`, knee/model.py) is dropped when the checkpoint is saved, so inference is unchanged.
  Smoke test (ResNet-34, 3 steps, P100): loss 1.56, checkpoint without aux keys, reloads into a plain KneeNet.
  Jobs 1026123 (node18) and 1026124 (node19). P100 nodes 18–21 have 64 GB: a 2-GPU job must ask ≤ 56 GB
  (`scontrol update … MinMemoryNode=57344`), 64G pends on Resources.
Checked and dropped: **patient overlap** — every study has its own PatientID (4,410 IDs for 4,410 studies), so no
train/test patient matching is possible; **finer resolution** — native pixel spacing is 0.32–0.33 mm (median, 512
reconstructed rows over ~160 mm, acquired matrices often lower), our 384 px cache is 0.39 mm/px: a 512 px cache
would mostly interpolate (and home is at 96/100 GB); **laterality errors** — studies whose report names only one
meniscus while the 16-run OOF clearly favours the other: 10 of 1,065 (0.9 %), OA 0 of 206 → not a lever.
Public label tables re-scored on gold: steven `llm_labels_v4_blend.csv` 0.893 (v2, used: 0.887) — not adopted
(gold gains do not predict the LB, Version 29).

Auto (2026-10-07 10:21, scripts/auto_followup.py): Kaggle: new submission ref 56905329 (knee-submit, scriptVersionId 355860913); new submission ref 56905335 (knee-blend, scriptVersionId 355861601); new submission ref 56905336 (knee-submit, scriptVersionId 355900841); new submission ref 56905340 (knee-submit, scriptVersionId 355719628); new submission ref 56905346 (knee-submit, scriptVersionId 355738008).

Submitted by the user at 10:20 Paris (2026-10-07): knee-submit Versions **34, 35, 28, 32** and knee-blend **7**
(refs 56905329–56905346; Version 32 identified from the push times, scriptVersionId 355738008 ≈ 13:35–13:40 on
10-06 — Versions 30/31 had no output and cannot be submitted).
**Disk**: home at 96/100 GB. With the user's OK, `kaggle/_dataset` (14 GB, the staging copy of the 28 packaged runs;
every checkpoint is in `runs/`, the wheels in `kaggle/wheels`, the code in git) was deleted; `scripts/package_kaggle.py`
now stages in node-local `/tmp/knee_dataset_$USER` (home 82 GB).

### 2026-10-07 morning: where the gap to the top is not

**Leaderboard** (full CSV, 08:32 UTC): 5,373 teams, first **0.964**, 10th 0.960, ≥ 0.955: 56, ≥ 0.950: 116; ours rank
637 with 0.944 (≥ 0.946: 223 teams). Public notebooks by score (`kaggle kernels list --sort-by scoreDescending`, hotness):
the newest ones (Tri-Specialist Superstack, Pure CoAtNet 0946, 0944 Fast Stack ConvNeXt Blend, Fast Parent "0.957"
— a local diagnostic, not the LB) are all the same 0.943 community stack, + an own 2.5D ConvNeXt reader at 30 % for
0.944 at best — exactly our knee-blend. No better public base. Public GitHub write-ups found today (homeshwarnelakurthi,
NTejas-1, anhquan1111, Daniel766hi; context only, nothing reused): all at or below 0.941; ideas they list that we
already have or tested: soft targets, per-finding attention queries, rank blending, weight EMA (E18: no gain).
Our blend weights bracket flat: 0.10 / 0.15 → 0.944, 0.25 → 0.943, 0.30 (weaker ours) → 0.942: our 0.936 adds
almost nothing to the stack's 0.943 — an own model near the stack's level is what is missing.

**Scanner / sex prior** (CPU, E41 OOF, `python -m scripts.oof_analysis site-prior`): per finding, a logistic regression on the OOF
logit + one-hot of 28 scanner groups (maker | model | field) + sex, nested over the 5 folds: silver 0.9510 →
0.9512, gold 0.906 → 0.896. Silver prevalence varies widely by scanner (ACL 2–53 %), but the image model already
carries it. PatientAge is blank in every header. ✗

**Subgroups of the OOF** (`python -m scripts.oof_analysis subgroups`, 3-run blend E41 + E46 + E53, silver 0.9615):
studies with 8 / 9+ series (10 %) score 0.933 / 0.926 against 0.959–0.972 for 4–7 series — two Philips sites (Ingenia Ambition X 0.894, Achieva dStream 0.918; ~9 series:
T1, T2 and PD FS in each plane, an oblique ACL series), English reports full of "suspicious …". In them the sag_t /
cor_t slot takes T1 or T2 at random (equal slice counts, ties broken by UID): a natural experiment — the 345
studies with both a T1 and a PD/T2 sagittal non-FS series score 0.919 with T1 in the slot and 0.908 with PD/T2,
both far below the other 4,062 studies (0.965). The random pick is not the cause; these sites' reports (or case
mix) are harder. Also: studies with longer series score lower (longest series ≤ 30 slices 0.971, 30–45 0.960, 45–80 0.954).

**New public label tables** (Kaggle datasets of 10-06, copied to `data/public_labels/`, licences read from the
dataset metadata). Gold agreement (macro AUC on the 58) and Spearman with our 4-table mean over all studies:

| table | licence | gold | ρ vs 4-table | verdict |
|---|---|---|---|---|
| kiritohayate95 MedGemma-27B (p3 prompt, 0–10) | CC0 | 0.879 | 0.78 | new reader |
| kiritohayate95 Mistral-Small-24B AWQ | CC0 | 0.879 | 0.76 | new reader |
| kiritohayate95 Qwen-32B | CC0 | 0.886 | 0.80 | new reader |
| mean of these three (mutual ρ 0.77) | | **0.898** | | |
| mandavisingh labels_v1 | CC0 | 0.893 | 0.97 | a mix of the tables we have |
| tunminhhunh report_labels_v2 | CC0 | 0.866 | — | identical to pilkwang v2 |
| codeandodev weak_labels | MIT | 1.000 | 0.58 | gold rows overwritten with the truth; lexicon rules |

The three readers are local open-weight models run with vLLM on one prompt ("how likely experts re-reading the
IMAGES mark the finding", 0 = stated normal, 1 = not mentioned, … 8–10 = definite/significant; scripts in the
dataset). Base means: 4-table 0.888 → 7-table (equal weights) 0.896, gains on synovitis 0.769 → 0.830 and lateral OA
0.809 → 0.838, loss on effusion 0.865 → 0.848. Training labels as E54 builds them (base, round-4 teacher at α 0.5,
gold rows excluded from the check): **0.934 → 0.938** (synovitis 0.843 → 0.873). Not a choice fitted on the gold
set: just more independent readers averaged, the step that took E38 → E41 from 0.929 to 0.931 on the LB.
**E72** = E54 with the 7-table base (`--mix-public … medgemma27_p3 mistral24_p3 qwen32_p3`, `scripts/eval_labels.py`
divides the 0–10 grades by 10), seeds 42/43/44, alone against E54 (0.935). Queued at 11:05 on P100 (jobs 1026319:
networks 0–1, 1026320: network 2), pending on the 4-job limit until E69/E70 end (~18:30); ~8 h per network.
Teacher check (gold agreement): the round-4 teacher (E35 + E36 OOF) 0.928 stays the best — the mean OOF of the eight
mixpub 5-fold runs 0.923, of the five 384 px ones 0.926 (labels 7-table + α 0.5: 0.938 / 0.935 / 0.936): later
self-distillation rounds drift, E72 keeps the round-4 teacher.

**Test-time slice span** (`scripts/tta_eval.py`, now with `--settings`, `--save` and a paired bootstrap against
24 windows; E41 fold networks, interactive P100). Fold 0, first 300 studies: base (12 windows) 0.9536, 24 windows
0.9553, zoom +8 % 0.9525, zoom −7 % 0.9543, span 0.04–0.96 **0.9570**, 0.12–0.88 0.9561; means of two settings add
nothing beyond the best single one. Re-checked on all of fold 1 (882 studies, 24 windows each):

| span | silver | Δ vs 0.08–0.92 [95 % CI] | gold (~12 studies) |
|---|---|---|---|
| 0.08–0.92 (training) | 0.9542 | — | 0.910 |
| 0.04–0.96 | 0.9545 | +0.0003 [−0.0008, +0.0016] | 0.908 |
| 0.02–0.98 | 0.9546 | +0.0004 [−0.0005, +0.0015] | 0.902 |
| 0.00–1.00 | 0.9550 | +0.0008 [−0.0004, +0.0025] | 0.904 |

The fold-0 gain (+0.0017 on 300 studies) shrinks to noise on the full fold, as E17 found at 16 windows (2–98 %:
−0.0006). ✗ — inference keeps the training span. `knee.infer.run(span_pad=…)` (widens every model's range, default 0)
stays available. Means of two to four span settings (2–4× the inference): at most +0.0008 [+0.0000, +0.0018].

**Queue rebuilt after Version 28 (11:45)**: 384 px beat 224 px on the LB, so tonight's GPUs go to 384 px runs.
E73 (E72 + seq-emb at 224 px; jobs 1026354/1026355/1026359, cancelled before they started:
`slurm/runs/*_cancelled.txt`) is dropped — Version 33 answers seq-emb at 224 px. Training time measured: 224 px
5.0 min/epoch on a 3090, 40.7 on a P100 (8.1×); 384 px 14.5 min/epoch on a 3090 → ~118 min on a P100, **~24 h per
network** (the 24 h default limit of `slurm/multi_gpu.sbatch` would kill it: 384 px jobs get `--time=36:00:00`).
- 1026319 = E72 networks 0–1, 1026432 = E72 network 2 (224 px, ~8 h, start ~18:30 when E69/E70 end);
- **E74** = E72's labels at 384 px (= E66 recipe + 7-table base; E66 = E54's labels at 384 px, not yet on the LB):
  1026433 networks 0–1 (~18:30 → ~18:30 on 10-08), 1026434 network 2 (starts at the next free slot, 01:14 or 02:30);
- **E75** = E56 recipe (Version 28: 384 px, mixpub round-5 teacher) with the 7-table base, 2 networks (seeds
  42/43), ensemble member: 1026436, starts after the unattended E72 push (dependency, so the push gets a slot first).
- **Unattended E72 push**: `slurm/auto_1007.sbatch` (E72 only; job 1026435, CPU, `--dependency=afterany` on
  1026319 and 1026432) packages the 13 full-data runs (`logs/day12c_runs.txt`, E72 last), pushes knee-submit
  Version N = E72 alone, notes it here and in `kaggle/A_SOUMETTRE.md`, commits and pushes. The evening pushes run
  from this session (`logs/chain_v36.sh`): Version 36 = E69 alone, Version 37 = E70 alone, Version 38 = **E66 alone**
  (E54's labels at 384 px, complete since 08:18: against E54 it isolates the slice size, against E56 the teacher)
  (31 runs); E72 then becomes Version 39.
Checked while rebuilding: every Slurm job gets a private `/tmp` (`/etc/slurm/plugstack.conf`: `auto_tmpdir.so
mount=/tmp mount=/var/tmp`), so `/tmp/knee_raw` is per job and a 224 px run can never read a 384 px unpacked
cache left by another job (`/tmp` usage back to its 99 GB baseline at each job start on node40).

**Slice spacing** (OOF of the 3-run blend, by the sag_fs series): studies whose series have more than 30 slices
score lower than ≤ 30 within 3 of the 4 sites that have both (Philips Achieva 0.934 vs 0.942, Ingenia 0.952 vs
0.967, Siemens Aera 0.944 vs 0.953; Avanto fit the reverse, 0.959 vs 0.944); by slice gap: ≤ 2.5 mm 0.952 (285
studies), 2.5–3.5 mm 0.962 (2,212), 3.5–4.5 mm 0.958 (1,325), > 4.5 mm 0.971 (579). A 3-slice window spans 6 mm at
2 mm spacing but 15 mm at 5 mm, and 8 training windows cover a 45-slice stack only in part. Candidate (not run):
windows whose neighbours are a fixed ~3 mm apart (stride from the cached slice positions) — needs a training
run; confounded with sites, so expected gain unclear.

Auto (2026-10-07 11:05, scripts/auto_followup.py): Kaggle: ref 56905346 (knee-submit version 32) scored 0.928.

**Version 32 on the public LB: 0.928** — **E61 alone** (E54 with `--gold-weight 10` instead of 3: the 58 expert studies
~12 % of the loss instead of ~4 %) against **0.935** for E54: **−0.007**, the largest drop of any single change. With
58 studies seen 12 epochs at weight 10, the network fits them (and their sites) rather than learning more from
them. ✗ — gold weight stays 3; the open question is the other direction (weight 1, i.e. gold rows like any
other row), a cheap later test. Same lesson as Versions 29 and the round-2 labels: the 58 gold studies help as
a check, not as a target to fit harder.

Auto (2026-10-07 11:25, scripts/auto_followup.py): Kaggle: ref 56905340 (knee-submit version 28) scored 0.938.

**Version 28 on the public LB: 0.938 — our best own score** (previous: 0.936, Versions 25/27 with 16–18 five-fold
runs). **E56 alone** = E46 recipe (384 px, 4-table base, mixpub round-5 teacher) on all studies, 3 networks, against
**0.935** for E54 (224 px, round-4 teacher): **+0.003**, more than any ensemble of five-fold runs. Two things differ
(slice size and teacher); E66 (= E54's labels at 384 px, complete, not yet on the LB) separates them. Consequence for
the GPU plan: new full-data runs go to 384 px even on P100s (~24 h per network instead of ~8 h at 224 px), 224 px
stays for quick A/B questions.

Auto (2026-10-07 12:08, scripts/auto_followup.py): Kaggle: ref 56905336 (knee-submit version 35) scored 0.934.

**Version 35 on the public LB: 0.934** — own-label full-data ensemble **E64 + E60 + E65** (Qwen-base labels, no public
table anywhere: 384 px round 5, ConvNeXt-small 384 px round 3, 224 px round 6). Equal to the best own-label score
so far (Versions 18/23, 0.934, five-fold ensembles): full data does not lift the own-label line, while E56 alone
(384 px, 4-table base) reaches 0.938 — at 384 px the public-table base is worth ~+0.004 over our Qwen-only base
(at 224 px: E41 0.931 vs E38 0.929). More independent readers in the base (E72/E74/E75, 7 tables) is the right
direction. CV line unchanged (own-label score stays 0.934).

### Outside review (Gemini, 2026-10-07 ~12:30) and what came of it

The user gave a full brief of the project to another AI. Its proposals, checked against our own results:

| proposal | verdict |
|---|---|
| per-finding spatial attention instead of average-pooled windows | **tested now: E76** — the one model-side idea not tried; the public CoAt readers do it |
| 6th slot for the oblique ACL / 3D cartilage series | low: oblique ACL series in 275 studies (6 %), 3D cartilage ~450 (10 %); their presence as a feature adds +0.0002 (protocol test below); the ACL is weak everywhere, not only there |
| per-finding teacher weight (text for ligaments, teacher for synovitis/effusion) | already tested as E58: −0.004 on the LB |
| per-finding blend weights fitted on the 58 gold studies | no: the public models trained on those 58 (in-sample predictions), 12 weights on 58 studies overfit, and the 58 misled every label choice so far |
| "extended gold" = gold + studies where the 7 readers agree | no: easy cases, where label recipes do not differ |
| fixed physical slice spacing | low: only 285 studies (6 %) have ≤ 2.5 mm slices (analysis above) |
| test-time pseudo-labelling inside the notebook | not now: heavy under the 9 h limit, uncertain gain |
| MRNet pretraining | possible (allowed, click-through): needs the user's registration; ~1,130 labelled exams (ACL, meniscus without side), +25 % data on two findings |

**Protocol composition** (CPU, E41 OOF, nested logistic regression per finding on the OOF logit + log counts of
series per plane, fat-suppressed, 3D, T1, STIR, GRE and description keywords — oblique/ACL, contrast, cartilage
mapping, Dixon, diffusion, anonymised): silver 0.9510 → 0.9512 (the recalibration alone 0.9502), gold 0.906 → 0.904.
An oblique ACL series goes with ACL tears (silver prevalence 46 % vs 15 %), but the images already carry it. ✗

**E76 — spatial queries** (`--spatial-queries`, `knee/model.py`): one learned query per finding attends over every
feature-map cell of every window (7 × 7 per window at 224 px, 12 × 12 at 384), each cell tagged with its series,
depth (the window position MLP) and in-plane position (row/column MLP); the attention output goes through a
zero-initialised linear layer and is added to the plain model's logits, so training starts from the plain model.
Checks: with the flag off, old checkpoints predict exactly as before (`scripts/check_infer` on E35 fold 0, 120
studies: 0.9586 / 0.9587, correlation 1.0000, max |diff| 0.0073, the numbers of 10-05; grouped vs per-model
`knee/infer.py` on 12 studies for E54 and E66: max |diff| 0.0); with it on, outputs at initialisation equal the
plain model's, gradients reach the new layers after one step, the checkpoint reloads through the new
`KneeNet.from_config` (now used by `knee/infer.py`, `check_infer`, `eval_variants`, `tta_eval`). Cost measured on
the P100 (synthetic batch, 8 studies × 5 series × 8 windows, fp16, gradient checkpointing): 224 px 4.80 vs 4.81
s/step, 5.8 GiB both. E76 = E41 recipe + `--spatial-queries`, folds 0–1, against E41 (0.9518 / 0.9536): job 1026676
(2 P100s), after the unattended E72 push (~03:30 → ~10:30 on 10-08). It takes the slot of **E75**, postponed
(`slurm/runs/e75_01_postponed.txt`): E74 already answers "7 tables at 384 px".

**384 px does not fit a P100 at batch 8** (found by the same benchmark): ConvNeXt-tiny, 8 studies × 40 windows of
384 px, out of memory at 15.9 GiB even with gradient checkpointing — every 384 px run so far trained on 3090s
(24 GB), so **E74 would have crashed at its first step tonight**. New `--accum k` in `scripts/train.py` (k batches per
optimiser step; the schedule counts optimiser steps; an incomplete last group is left out): E74's runs files now
use `--bs 4 --accum 2` (8 studies per step, as on the 3090s): 6.92 s per 4 studies, 8.3 GiB, i.e. ~2 h per epoch,
~25 h per network. Full-data runs now compute their in-sample monitor at the last epoch only (it cost ~10 % of the
run each epoch and measures nothing). Smoke tests on the P100: a full-data run with `--spatial-queries --bs 4
--accum 2` (2 epochs × 6 steps: monitor only at epoch 2, checkpoint reloads, the new head's weights moved off zero)
and E72's exact path (`--bs 8`, accum 1).

Auto (2026-10-07 13:09, scripts/auto_followup.py): Kaggle: ref 56905329 (knee-submit version 34) scored 0.938.

**Version 34 on the public LB: 0.938** — full-data ensemble **E56 + E59 + E60 + E54** (two E46-recipe 384 px trios,
ConvNeXt-small 384 px own labels, 224 px) = **E56 alone** (Version 28, 0.938). Six more networks of the same
families add nothing: our members are too alike (same 2.5D input, close labels), as both outside reviews said.
The way up is decorrelated members (spatial queries E76, other label bases, other inputs), not more seeds.

### Two more outside reviews (two Claude sessions, 2026-10-07 ~13:30) — checks and tonight's tests

Shared diagnosis: our members are too alike (same 2.5D input, close labels), silent cells are under-used, average
pooling loses where a lesion is (medial vs lateral sit on fixed image sides after mirroring: gold lateral meniscus
0.889 vs medial 0.970, lateral OA 0.853 vs medial 0.992). Version 34 (0.938 = E56 alone with six more networks of
the same families) agrees. Checked on CPU:
- **Mixed contrasts inside a series** (dual-echo / Dixon images at one position, so a 3-slice window would stack two
  contrasts): none — all 20,792 selected series have one image per position (`n_images == n_slices`; the loader
  keeps one image per position anyway).
- **Stacking the 12 OOF predictions** per finding (nested logistic regression): silver −0.0006 to −0.0013, gold
  −0.01 (E41, E46). ✗ The 12 queries already share the evidence.
- **Reporting habits by site** (report language × scanner maker, 20 groups; `knee.labels.report_language`, 99.5 %
  agreement with a public language column): the share of reports that mention a finding varies hugely — synovitis
  17 % (bg|Philips) to 76 % (es|GE), contusion 21 % (es|Philips) to 91 %, Baker's 22–99 %, effusion 42–99 % — while the
  image teacher's per-site means vary about as much as the labels (it learned the sites' habits too). Silent cells
  carry habits, not image content.
- DDP on two P100s would halve the wall time of one network, not the GPU-hours; our three networks per run already
  run in parallel. Not pursued.

Tonight's queue rebuilt around these (14:00; 224 px tests, ~8 h each; the 384 px E74 is postponed —
`slurm/runs/e74_*_postponed.txt` — until E72's LB and E79 say how to make 384 px models):
- **E77** = E54 + teacher weight by what the report says, fixed a priori: silent cells 0.9 with full loss weight
  (`--alpha-silent 0.9 --silence-weight 1.0`), cells graded absent/severe 0.25 (`--alpha-clear 0.25`), the rest 0.5.
  Labels' gold agreement 0.934 → 0.940 (informative only, not used to pick the values). Alone vs E54 (0.935).
- **E78** = E54 + `--site-bias`: a learned logit offset per (site, finding) on silent cells, training only (saved
  in `site_bias_fold*.csv`, never used at inference). Alone vs E54.
- **E79** = E41 fold 0 network (224 px) continued 4 epochs at 384 px (`--init-from`, lr 1e-4 / 5e-4, bs 4 × 2):
  fold-0 silver against E42 fold 0 (384 px from scratch, same labels, 0.9577). If it matches, a 384 px version
  costs ~8 h of P100 instead of ~25 h.
- **E76** (spatial queries, folds 0–1) and **E72** (7-table base) as before.
New code: `scripts/train.py` `--alpha-silent`, `--alpha-clear`, `--site-bias`, `--init-from`; `StudyDataset` also
returns its row index (`"i"`); `slurm/multi_gpu.sbatch` takes optional `VAR=value` words before a run's arguments
(that run gets its own cache, e.g. 384 px next to a 224 px run in one job, and skips the job's unpacked copy).
Checks: with the new options off, `build_targets` returns exactly the previous labels; with E77's options, silent
cells move 0.049 on average, mentioned ones 0.018, gold rows not at all; a P100 smoke run with `--site-bias
--alpha-silent --alpha-clear --init-from` (20 sites, 25.6 % silent cells; 224/224 tensors loaded; in-sample monitor
0.952 after 8 steps, i.e. the loaded network) writes its checkpoint and `site_bias_fold0.csv`; the per-run
environment parser dry-run gives the expected two commands.
Jobs: 1026319 (E72 n0–1), 1026813 (E72 n2 + E79), 1026814 (E77 n0–1), 1026815 (E77 n2 + E78 n0), 1026816 (E78 n1–2),
1026817 (unattended E72 push, after 1026319 + 1026813), 1026676 (E76, after the push).

### 2026-10-07 afternoon: user decisions, rules facts, E75 on a 3090

- User: **no team merge** (merger deadline 2026-10-15, confirmed in the rules timeline: entry and team-merger
  deadline 10-15, final submission 10-22); MRNet explained and not pursued (2 of our 12 findings, meniscus
  without side, report-derived labels, 3 pre-cropped series); the two final submissions get chosen in the last week.
- Rules pages are readable from the cluster (`kaggle competitions pages rsna-knee-abnormality-detection list
  rsna-knee-abnormality-detection --content`; the forum still is not). Efficiency Prize: the team's selected final
  submissions are ranked by AUC / (Benchmark - max AUC) + RuntimeSeconds / 32400 (lower is better; 0.01 of AUC is
  worth ~700 s of runtime); no CPU-only condition (CPU or GPU notebooks, <= 9 h).
- 3090 partition at 14:15: all 10 cards held (hle-25, revol-24, agranados; nodemm05's fourth card has no free CPU),
  two jobs pending ahead of ours, our fair-share factor 773 vs 1210-1691. **E75** now queued on one 3090 as a batch
  job (1026904, `slurm/runs/e75_3090.txt`): E56 recipe (Version 28, best own single) with the 7-table base, three
  full-data networks one after the other (~3 h each on a 3090, vs ~25 h on a P100), 14 h limit. It is our
  lowest-priority job (2766), so it takes a job slot only when a card is free once the P100 jobs above have started.
  The user's interactive 3090 request (1026675) is theirs to keep or cancel.

### Public members' out-of-fold predictions; E80, a family that reads each series differently (2026-10-07 15:00)

Two families of the 0.943 public stack (antoinegg1 RadImageNet query heads e9 v15 and e11 v20, CC-BY-NC-SA-4.0)
publish their five-fold OOF tables (`v52_oof.csv`, `v52_e11_oof.csv`; read in the scratchpad, not versioned): the
per-family public predictions Claude 1 asked about. Against our targets, same 4,407 studies:

| predictions | silver | gold | mean rank correlation with E41 |
|---|---|---|---|
| E41 (5-fold, 224 px) | 0.9510 | 0.906 | — |
| public e9 heads | 0.9197 | 0.854 | 0.75 |
| public e11 heads | 0.9158 | 0.826 | 0.73 |
| E41 + 15 % e9 / + 15 % e11 (rank average) | 0.9534 / 0.9539 | 0.909 / 0.907 | |
| our runs: E46, E42 (384 px), E38 (own labels), E25 / E33 (RadImageNet ResNet-50) | | | 0.895, 0.899, 0.917, 0.869 / 0.868 |

Much weaker members still lift E41 by +0.0025–0.003 silver at 15 % because their errors differ, while our own runs
stay at 0.87–0.92 of each other whatever the backbone (same 2.5D input, close labels): the reason Version 34 = E56
alone while the stack gains from its five families. Priority: own families with a different input.

**E80 — stacked slices** (`--stack-slices 16`, new): one image per series, 16 slices spread evenly over the span and
stacked as channels (depth as channels, the A5 design inside the public stack), instead of 8 (24 at test) windows of
3 adjacent slices. The ConvNeXt stem takes 16 channels (timm adapts the pretrained weights); the grey slices are
normalised with the colour-averaged ImageNet statistics; training shifts the covered range by up to ±3 %.
`--windows` / `--windows-eval` are forced to 1, and `knee/infer.py` gives such models one window whatever the
test-time setting (`model_windows`, `model_dataset`, the stack in `input_key`). Checks: 2.5D samples byte-identical
to HEAD's `knee/data.py` (6 studies, train and eval); a smoke model predicts identically through `predict_model` and
`predict_group` and matches its training OOF (max |diff| 0.0004, CPU fp32 vs GPU fp16). Speed on the P100 at 224 px:
0.59 s per batch of 8 studies, ~4.3 min per fold epoch, ~8x faster than the 2.5D windows (384 px: ~3 h per
full-data network). Probe: E54's labels (E41 recipe), folds 0–1 on the interactive P100 (started 15:05, ~1 h per
fold), against E41 (0.9518 / 0.9536) and above all for its correlation with E41.

**E80 fold 0** (882 studies; the run folder says 1505, it started ~14:20): silver **0.9266** vs E41 0.9518 (−0.025),
gold(fold) 0.865 vs 0.922 (11 studies). Mean rank correlation with E41 **0.822** (E42–E41 0.903, public e9–E41
0.756, E80–e9 0.718). As good as E41 on the diffuse findings (effusion 0.993 / 0.989, synovitis 0.982 / 0.975,
Baker's, ACL 0.934 / 0.934), far behind on focal ones (lateral meniscus 0.839 / 0.926, medial meniscus 0.923 /
0.965, contusion 0.912 / 0.972, fracture 0.893 / 0.936): mixing 16 slices in the first layer dilutes a lesion seen
on two or three of them. Blend on fold 0: E41 + 20 % E80 0.9542 (+0.0024), but E41 + 20 % E42 (the 384 px 2.5D run)
0.9561, and E41 + E42 + 15 % E80 0.9602 vs 0.9596 without it (+0.0006). Not worth a full-data trio as it is ✗; the
option stays in the code (one image per series: very cheap inference). Fold 1 stopped at its start to free the
P100 for **E79 fold 1** (the 224 → 384 px continuation from E41's fold 1 network, same arguments as E79 fold 0,
`2026-10-07_1525_e79_convnext_384_from_e41_f1`, against E42 fold 1), started ~15:30 on the interactive P100, ~7 h.

Auto (2026-10-07 18:31, scripts/auto_followup.py): Kaggle: ref 56905335 (knee-blend version 7) scored 0.942.

**knee-blend version 7 on the public LB: 0.942** (public 0.85 + full-data E56 + E59 + E60 + E54 at 0.15, 24 windows,
both T4s) — below blends 3 / 5 / 6 (0.944; their own part: the 5-fold sets of Versions 24 / 27, mostly 384 px runs
of both label families, cut by the time budget to the first runs) although our part alone is better (Version 34
0.938 vs Version 27 0.936). Our part now leans on the mixpub family (E56, E59: 4-table base, mixpub teacher), whose
labels share the public label tables the stack's members were trained on: the likeliest reason, in line with the
correlations above. **knee-blend version 8** (built in `kaggle/blend_v8/`, pushed after the evening chain): public
0.85 + the own-label full-data set of Version 35 (E64 384 px, E60 ConvNeXt-small 384 px, E65 224 px; Qwen labels
and own teachers only, 0.934 alone) at 0.15, 24 windows: does the label family decide what a member adds to the
stack?

**E75 started on a 3090** (job 1026904, node40, 18:21): a card freed as E69 / E70 ended, ~3 h per network, the
three done ~04:30. E77's first job (1026814) now waits for the next slot (01:14, when the interactive job ends):
E77 and E78 complete ~10:30 as planned (their third networks were already in the later jobs).

**Evening chain done (19:07)**: knee-submit **Version 36 = E69 alone** (auxiliary report grades), **37 = E70 alone**
(aggregator 384 × 4), **38 = E66 alone** (E54's labels at 384 px), each 3 networks on the visible test, dataset of the
31 runs of `logs/day12_runs.txt`, packaged with the new `knee/infer.py` (E80 option; the 2.5D path unchanged).
**knee-blend version 8** pushed (19:10, `kaggle/blend_v8/`). `kaggle/A_SOUMETTRE.md` rewritten for 10-08: blend 8,
E72's version (added by the night job), Version 36 in the morning; E77 and E78 in the afternoon (Versions 38 / 37 if
they are late); Version 33 later. Night automation: `slurm/auto_1007.sbatch` (job 1026817, after E72: Version N = E72
alone) and **`slurm/auto_1008.sbatch`** (job 1027476, after E77's jobs and 1026817): packages the 13 full-data runs +
E75 (when complete) + E77 and pushes E77 alone, then E75 alone; waits for E78 (until 16:00), packages again and
pushes E78 alone; rows appended to the night table of `kaggle/A_SOUMETTRE.md`.

**E79 fold 1 done (22:34; interactive P100, 4 epochs of ~107 min at 384 px from E41's fold-1 network)**: silver
0.9533 → 0.9537 → 0.9577 → **0.9580**, against **0.9550 for E42** (same labels, 384 px from scratch, 12 epochs) and
0.9561 for E46; gold(fold) 0.913 (E42 0.923, 12 studies). The continuation is at least as good as training at 384 px
from scratch for a third of the 384 px epochs, and it combines like a scratch model with other 384 px runs (fold 1:
E79 + E46 0.9607 vs E42 + E46 0.9589; E79 + E42 0.9604), though not with its own parent (E41 + E79 0.9579, rank
correlation 0.944): the 384 px version replaces its 224 px parent. Fold 0 (job 1026813) ends ~01:45.
**E81** queued on one 3090 (job 1027771, lowest priority, `slurm/runs/e81_3090.txt`): E54's three full-data networks
continued 4 epochs at 384 px (~1 h each on a 3090): the LB test of the continuation against Version 38 (E66, E54's
recipe at 384 px from scratch). E75 on its 3090: 12.8 min per epoch, network 0 done, all three ~02:20.

Auto (2026-10-08 04:43, scripts/auto_followup.py): knee-submit **Version 39 = E72 alone** (7-table report base) pushed by the unattended job (`slurm/auto_1007.sbatch`): dataset of the 13 full-data runs, 1 model(s) predicted on the visible test (`logs/push_1008_0432`).

Auto (2026-10-08 09:52, scripts/auto_followup.py): knee-submit **Version 40 = E77 alone** (teacher weight by what the report says: silent cells 0.9, absent/severe 0.25) pushed by the unattended job (`slurm/auto_1008.sbatch`).

Auto (2026-10-08 09:55, scripts/auto_followup.py): knee-submit **Version 41 = E75 alone** (E56 recipe with the 7-table base, 384 px, trained on a 3090) pushed by the unattended job.

Auto (2026-10-08 10:10, scripts/auto_followup.py): knee-submit **Version 42 = E78 alone** (per-site bias on silent cells, training only) pushed by the unattended job.

`kaggle/submissions.csv` (2026-10-02 16:30): ledger of every Kaggle submission (API ref, date, notebook
version, models, public score), so the LB scores quoted here can be checked against the Kaggle page.
The AUCs in this journal can only be recomputed where the competition data lives (`data/` is never
versioned: rule on redistribution): `python -m scripts.eval_oof runs/<run>` on the cluster, from the
committed out-of-fold predictions.
