#!/bin/bash
# E116 (queued 2026-10-13 19:25, 3090 batch job): E114's 3D CV test (fold 0, same labels, 24 slices, 224 px, 10 epochs) with
# R3D-18 (full 3x3x3 convolutions, Kinetics-400) instead of R(2+1)D-18: a second 3D backbone, compared with
# python -m scripts.cv_compare e116 e114 --folds 0 (alone, correlation with E79 / E46 / E114, gain when added).
cd ~/RSNA-Knee-Abnormality-Detection
LAB="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e116_r3d_f0 --run-dir 2026-10-13_1925_e116_cv_r3d18_stack24_224 $LAB --backbone r3d_18 --stack-slices 24 --img-size 224 --epochs 10 --bs 2 --accum 4 --grad-ckpt --raw-dir /tmp/knee_raw384 --workers 4 --folds 0 > logs/e116_f0.log 2>&1
