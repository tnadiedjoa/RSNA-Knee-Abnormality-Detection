#!/bin/bash
# On the interactive 3090 (nodemm01, job 1033306), queued 2026-10-10 20:50 after E97: a tighter, higher-resolution view
# for diversity. The public parts are worth 0.950 whatever their weights (10-10 LB) and our 150 mm 2.5D members add
# nothing to them; heliosli's compact model, which did add (+0.002 on the CoAtNet), differed mainly by its input
# (anatomy-anchored 352 px crops). A cheap step in that direction: the central 100 mm of the cached 150 mm field at 384 px
# (--zoom 1.5, 0.26 mm/px instead of 0.39), 4 epochs continued from 384 px networks:
#  E100 = E85's networks (E72 continued at 384 on E75's targets) on E75's targets;
#  E101 = E95's networks (own labels, 384 px from scratch, seed 142) on the round-6 own targets.
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
COMMON="--grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4 --epochs 4 --lr 1e-4 --lr-head 5e-4 --zoom 1.5 --backbone convnext_tiny.fb_in22k_ft_in1k"
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
OWN6="--labels qwen36_27b_v1 --bootstrap-run 2026-10-04_0205_teacher_r5_oof_mean --bootstrap-alpha 0.7"
.venv/bin/python -m scripts.train --name e100_e85_zoom150 --run-dir 2026-10-10_2050_e100_convnext_384_zoom100mm_from_e85_mix7_r6_full3 $MIX7 $COMMON --init-from "runs/2026-10-08_2255_e85_convnext_384_from_e72_mix7_r6_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e101_e95_zoom150 --run-dir 2026-10-10_2050_e101_convnext_384_zoom100mm_from_e95_labels_r6_full3 $OWN6 $COMMON --init-from "runs/2026-10-09_1835_e95_convnext_384_labels_r6_full3_seed142/checkpoints/fold{fold}.pt"
