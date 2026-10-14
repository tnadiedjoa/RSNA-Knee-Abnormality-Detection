#!/bin/bash
# E118 (queued 2026-10-14 17:30, 3090 batch): E114's 3D CV test (fold 0, R(2+1)D-18, 224 px) with 32 slices per clip instead of 24 (finer depth sampling; compare: python -m scripts.cv_compare e118 e114 --folds 0).
cd ~/RSNA-Knee-Abnormality-Detection
LAB="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e118_3d_32sl_f0 --run-dir 2026-10-14_1730_e118_cv_r2plus1d18_stack32_224 $LAB --backbone r2plus1d_18 --stack-slices 32 --img-size 224 --bs 2 --accum 4 --epochs 10 --grad-ckpt --raw-dir /tmp/knee_raw384 --workers 4 --folds 0 > logs/e118_3d_32sl_f0.log 2>&1
