#!/bin/bash
# E82 on the interactive 3090 (nodemm01): E77's three full-data 224 px networks continued 4 epochs at 384 px (E79 recipe)
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
.venv/bin/python -m scripts.train --name e82_e77_to_384 --run-dir 2026-10-08_1915_e82_convnext_384_from_e77_full3 --labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5 --backbone convnext_tiny.fb_in22k_ft_in1k --grad-ckpt --full-fit --alpha-silent 0.9 --alpha-clear 0.25 --silence-weight 1.0 --img-size 384 --epochs 4 --lr 1e-4 --lr-head 5e-4 --init-from "runs/2026-10-07_1400_e77_convnext_mixpub_r5_silentalpha_full3/checkpoints/fold{fold}.pt" --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4
