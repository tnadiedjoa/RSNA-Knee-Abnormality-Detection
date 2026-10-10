#!/bin/bash
# On the interactive 3090 (nodemm01, job 1033306), queued 2026-10-10 15:00 after E95: a tighter, higher-resolution view
# for diversity. The public parts are worth 0.950 whatever their weights (10-10 LB) and our 150 mm 2.5D members add
# nothing to them; heliosli's compact model, which did add (+0.002 on the CoAtNet), differed mainly by its input
# (anatomy-anchored 352 px crops). A cheap step in that direction: the central 100 mm of the cached 150 mm field at 384 px
# (--zoom 1.5, 0.26 mm/px instead of 0.39), 4 epochs continued from 384 px networks:
#  E96 = E75's networks (E56 recipe, 7-table base + mixpub r5 teacher; 0.940 alone) on E75's targets;
#  E97 = E64's networks (own labels) on the round-6 own targets.
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
COMMON="--grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4 --epochs 4 --lr 1e-4 --lr-head 5e-4 --zoom 1.5 --backbone convnext_tiny.fb_in22k_ft_in1k"
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
OWN6="--labels qwen36_27b_v1 --bootstrap-run 2026-10-04_0205_teacher_r5_oof_mean --bootstrap-alpha 0.7"
.venv/bin/python -m scripts.train --name e96_e75_zoom150 --run-dir 2026-10-10_1500_e96_convnext_384_zoom100mm_from_e75_mix7_r6_full3 $MIX7 $COMMON --init-from "runs/2026-10-07_1420_e75_convnext_384_mix7_r6_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e97_e64_zoom150 --run-dir 2026-10-10_1500_e97_convnext_384_zoom100mm_from_e64_labels_r6_full3 $OWN6 $COMMON --init-from "runs/2026-10-06_1400_e64_convnext_384_labels_r5_full3/checkpoints/fold{fold}.pt"
