# RSNA Knee Abnormality Detection

Solution for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
predict, for every knee MRI study, the probability of 12 findings — ACL, MCL, medial and lateral
meniscus tears, medial / lateral / patellofemoral osteoarthritis, effusion, synovitis, Baker's
cyst, bone contusion and fracture. Metric: macro ROC AUC over the 12 findings.

The difficulty is the supervision: only **58** of the 4,407 training studies carry expert labels,
the other 4,349 come with a free-text radiology report in one of a dozen languages, and the
**test set has images only** (no report). The pipeline therefore has two stages:

1. **Report → labels.** A local open-weights LLM grades each finding in every report
   (absent / equivocal / mild / moderate / severe / not mentioned). Grades become soft labels.
   Report text never leaves the cluster (competition rule on data security: no hosted LLM API).
2. **Images → predictions.** A 2.5D multi-series network, trained on those soft labels, reads
   the MRI series; it is the only thing that runs at test time.

## Pipeline

| Step | Code | Notes |
|---|---|---|
| Remote archive access | [`knee/remote_zip.py`](knee/remote_zip.py), [`scripts/build_zip_index.py`](scripts/build_zip_index.py) | The archive is 265 GB zipped (570 GB of DICOMs); members are fetched with HTTP range requests, nothing but the cache is stored |
| Image cache | [`scripts/build_cache.py`](scripts/build_cache.py), [`knee/dicom_io.py`](knee/dicom_io.py), [`knee/cache_io.py`](knee/cache_io.py) | Slices sorted by physical position, canonical orientation per plane, every knee mirrored into a right knee (one side per study), 150 mm centre crop, 256 px, JPEG q95 |
| Report labels | [`scripts/llm_labels.py`](scripts/llm_labels.py), [`knee/labels.py`](knee/labels.py), [`scripts/eval_labels.py`](scripts/eval_labels.py) | Qwen3.6-27B (int4, vLLM) on one RTX 3090; agreement with the 58 gold studies: macro AUC 0.873 |
| Folds | [`scripts/make_folds.py`](scripts/make_folds.py) | 5 folds, near-identical reports grouped, gold studies spread over folds |
| Model | [`knee/data.py`](knee/data.py), [`knee/model.py`](knee/model.py), [`scripts/train.py`](scripts/train.py) | 5 series per study (sagittal/coronal/axial fat-suppressed fluid-sensitive + sagittal/coronal other), 3-slice windows through an ImageNet CNN, transformer over windows, one attention query per finding |
| Submission | [`knee/infer.py`](knee/infer.py), [`scripts/package_kaggle.py`](scripts/package_kaggle.py), [`kaggle/submit/`](kaggle/submit/) | Offline Kaggle notebook; same preprocessing code as training (inputs checked byte-identical) |

Every experiment — kept or not — is logged in [`experiments.md`](experiments.md) with the
change, the commit and the before/after scores.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # training / cache
python -m scripts.build_zip_index                                     # needs Kaggle API credentials
sbatch slurm/build_cache.sbatch 0 2; sbatch slurm/build_cache.sbatch 1 2
python -m scripts.build_cache --fix-laterality
# labels: separate venv with vLLM, one 24 GB GPU
python -m scripts.llm_labels --name qwen36_27b_v1 --no-evidence
python -m scripts.make_folds qwen36_27b_v1
sbatch slurm/train_3090.sbatch --name r34_base --labels qwen36_27b_v1
python -m scripts.package_kaggle runs/<run>          # private Kaggle dataset, then push kaggle/submit/
```

The first, text-only prototype (September 2026) is kept in [`legacy/`](legacy/): it could not be
submitted, since the test set has no reports.

## Competition constraints

Code competition: Kaggle notebook, ≤ 9 h, internet off, `submission.csv`. External data and
pre-trained models must be public and free. Competition data is never versioned here (`data/` is
git-ignored) and the repository stays private until the end of the competition.
