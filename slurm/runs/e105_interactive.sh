#!/bin/bash
# On the second interactive 3090 (job 1033826), queued 2026-10-11 12:40 while the 10-11 LB (zoomed members) is scored:
# E105 = E90's networks (E58 continued at 384 on E75's targets) continued 4 epochs on the central 100 mm (--zoom 1.5).
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4  # each job has its own /tmp
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e105_e90_zoom150 --run-dir 2026-10-11_1240_e105_convnext_384_zoom100mm_from_e90_mix7_r6_full3 $MIX7 --grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4 --epochs 4 --lr 1e-4 --lr-head 5e-4 --zoom 1.5 --backbone convnext_tiny.fb_in22k_ft_in1k --init-from "runs/2026-10-09_1000_e90_convnext_384_from_e58_mix7_r6_full3/checkpoints/fold{fold}.pt"
