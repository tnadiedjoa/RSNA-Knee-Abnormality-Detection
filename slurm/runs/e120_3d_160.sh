#!/bin/bash
# E120 (queued 2026-10-14 17:30, 3090 batch): E114's 3D CV test at 160 px. E117 at 320 px fell to silver 0.938 (E114 at 224: 0.957): the Kinetics models were trained on 112 px clips, so the best input size may sit below 224.
cd ~/RSNA-Knee-Abnormality-Detection
LAB="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e120_3d_160_f0 --run-dir 2026-10-14_1730_e120_cv_r2plus1d18_stack24_160 $LAB --backbone r2plus1d_18 --stack-slices 24 --img-size 160 --bs 2 --accum 4 --epochs 10 --grad-ckpt --raw-dir /tmp/knee_raw384 --workers 4 --folds 0 > logs/e120_3d_160_f0.log 2>&1
