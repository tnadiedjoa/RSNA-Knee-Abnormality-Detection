#!/bin/bash
# E119 (queued 2026-10-14 17:30, 3090 batch): E114's 3D CV test with MC3-18 (3D convolutions in the first stages, 2D after; Kinetics-400): a third 3D backbone beside R(2+1)D-18 (E114) and R3D-18 (E116, +0.0005 silver over E79 + E46 + E114).
cd ~/RSNA-Knee-Abnormality-Detection
LAB="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e119_3d_mc3_f0 --run-dir 2026-10-14_1730_e119_cv_mc3_18_stack24_224 $LAB --backbone mc3_18 --stack-slices 24 --img-size 224 --bs 2 --accum 4 --epochs 10 --grad-ckpt --raw-dir /tmp/knee_raw384 --workers 4 --folds 0 > logs/e119_3d_mc3_f0.log 2>&1
