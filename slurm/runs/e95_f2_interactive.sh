#!/bin/bash
# E95 fold 2 again from its start: the interactive job ended early on 2026-10-10 (~08:40) at fold 2 epoch 6 of 12.
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
.venv/bin/python -m scripts.train --name e95_own_r6_384_s142 --run-dir 2026-10-09_1835_e95_convnext_384_labels_r6_full3_seed142 --labels qwen36_27b_v1 --bootstrap-run 2026-10-04_0205_teacher_r5_oof_mean --bootstrap-alpha 0.7 --grad-ckpt --img-size 384 --full-fit --folds 2 --raw-dir /tmp/knee_raw384 --workers 4 --epochs 12 --seed 142 --backbone convnext_tiny.fb_in22k_ft_in1k
