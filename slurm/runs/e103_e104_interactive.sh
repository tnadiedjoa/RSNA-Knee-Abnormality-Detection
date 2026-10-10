#!/bin/bash
# On the first interactive 3090 (job 1033306, until 2026-10-11 ~11:40), after E101 (queued 2026-10-11 01:45): two more
# members on the central 100 mm (--zoom 1.5), 4 epochs continued from 384 px networks, ~3.3 h each:
#  E103 = E56's networks (mixpub r6 recipe, 384 px) on E75's targets;
#  E104 = E93's networks (E65 at 384 px, 130 mm, own labels) on the round-6 own targets.
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
COMMON="--grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4 --epochs 4 --lr 1e-4 --lr-head 5e-4 --zoom 1.5 --backbone convnext_tiny.fb_in22k_ft_in1k"
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
OWN6="--labels qwen36_27b_v1 --bootstrap-run 2026-10-04_0205_teacher_r5_oof_mean --bootstrap-alpha 0.7"
.venv/bin/python -m scripts.train --name e103_e56_zoom150 --run-dir 2026-10-11_0300_e103_convnext_384_zoom100mm_from_e56_mix7_r6_full3 $MIX7 $COMMON --init-from "runs/2026-10-06_0830_e56_convnext_384_mixpub_r6_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e104_e93_zoom150 --run-dir 2026-10-11_0300_e104_convnext_384_zoom100mm_from_e93_labels_r6_full3 $OWN6 $COMMON --init-from "runs/2026-10-09_1835_e93_convnext_384_zoom130_from_e65_full3/checkpoints/fold{fold}.pt"
