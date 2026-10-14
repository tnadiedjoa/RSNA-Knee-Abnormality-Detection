#!/bin/bash
# E122 (queued 2026-10-14 19:25, 3090 batch): E114 3D CV test at 256 px, bs 2 x 4 as E114. 160 px (E120): 0.9540, 224 px (E114): 0.9573, 320 px at bs 1 (E117): 0.9381 - does 256 px help when the batch size stays 2?
cd ~/RSNA-Knee-Abnormality-Detection
LAB="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e122_3d_256_f0 --run-dir 2026-10-14_1925_e122_cv_r2plus1d18_stack24_256 $LAB --backbone r2plus1d_18 --stack-slices 24 --img-size 256 --bs 2 --accum 4 --epochs 10 --grad-ckpt --raw-dir /tmp/knee_raw384 --workers 4 --folds 0 > logs/e122_3d_256_f0.log 2>&1
