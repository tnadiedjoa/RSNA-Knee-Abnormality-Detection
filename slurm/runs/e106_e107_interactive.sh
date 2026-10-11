#!/bin/bash
# On the second interactive 3090 (job 1033826), queued 2026-10-11 15:55. LB of 10-11: base 0.8 + the five zoomed members
# (130 mm E91-E93 and 100 mm E96-E97) 0.2 = 0.951 (blend 39), the first move above 0.950, while the 130 mm members alone
# (blend 36) stayed at 0.950: the tighter view may be what adds. A tighter one still: the central 75 mm (--zoom 2.0,
# 0.20 mm/px), 4 epochs continued from 384 px networks:
#  E106 = E75's networks on E75's targets; E107 = E64's networks on the round-6 own targets.
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4  # each job has its own /tmp
COMMON="--grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4 --epochs 4 --lr 1e-4 --lr-head 5e-4 --zoom 2.0 --backbone convnext_tiny.fb_in22k_ft_in1k"
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
OWN6="--labels qwen36_27b_v1 --bootstrap-run 2026-10-04_0205_teacher_r5_oof_mean --bootstrap-alpha 0.7"
.venv/bin/python -m scripts.train --name e106_e75_zoom200 --run-dir 2026-10-11_1555_e106_convnext_384_zoom75mm_from_e75_mix7_r6_full3 $MIX7 $COMMON --init-from "runs/2026-10-07_1420_e75_convnext_384_mix7_r6_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e107_e64_zoom200 --run-dir 2026-10-11_1555_e107_convnext_384_zoom75mm_from_e64_labels_r6_full3 $OWN6 $COMMON --init-from "runs/2026-10-06_1400_e64_convnext_384_labels_r5_full3/checkpoints/fold{fold}.pt"
