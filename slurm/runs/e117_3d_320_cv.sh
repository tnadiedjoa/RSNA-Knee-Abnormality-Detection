#!/bin/bash
# E117 (queued 2026-10-13 20:10, 3090 batch job): E114's 3D CV test (fold 0, same labels, R(2+1)D-18, 24 slices, 10 epochs)
# at 320 px instead of 224 (bs 1 x 8: the same memory as 224 px at bs 2): is resolution what holds the 3D model back
# (alone silver 0.9573 at 224 px vs 0.958 for the 2.5D runs at 384 px)? Compared with
# python -m scripts.cv_compare e117 e114 --folds 0.
cd ~/RSNA-Knee-Abnormality-Detection
LAB="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e117_r2p1d_320_f0 --run-dir 2026-10-13_2010_e117_cv_r2plus1d18_stack24_320 $LAB --backbone r2plus1d_18 --stack-slices 24 --img-size 320 --epochs 10 --bs 1 --accum 8 --grad-ckpt --raw-dir /tmp/knee_raw384 --workers 4 --folds 0 > logs/e117_f0.log 2>&1
