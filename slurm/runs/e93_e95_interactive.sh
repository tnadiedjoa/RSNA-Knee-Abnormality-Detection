#!/bin/bash
# On the interactive 3090 (nodemm01, job 1030891 until 2026-10-10 15:16), queued 2026-10-09 18:35 after E90: own-label
# members at 384 px (Qwen labels + own round-5 teacher at alpha 0.7, the round-6 targets of E65 / E84, no public table).
# On the 58 gold studies the own-label CV runs pair best with heliosli's compact model (e47 + S5 0.936, e36 + S5 0.934;
# mixpub runs lower), and on the 0.943 stack the own-label set blended one tick better than the mixpub one (blend 8 vs 7).
#  E93 = E65's networks continued 4 epochs at 384 px with a 130 mm field of view (--zoom 1.154): E84 + zoom (~1 h each);
#  E94 = E60's ConvNeXt-small networks (own labels round 3) continued 4 epochs on the round-6 targets (~1.6 h each);
#  E95 = E64's recipe (384 px from scratch, 12 epochs) on the round-6 targets, seed 142 (~2.9 h each).
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
OWN6="--labels qwen36_27b_v1 --bootstrap-run 2026-10-04_0205_teacher_r5_oof_mean --bootstrap-alpha 0.7 --grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4"
CONT="--epochs 4 --lr 1e-4 --lr-head 5e-4"
.venv/bin/python -m scripts.train --name e93_e65_384_zoom130 --run-dir 2026-10-09_1835_e93_convnext_384_zoom130_from_e65_full3 $OWN6 $CONT --zoom 1.154 --backbone convnext_tiny.fb_in22k_ft_in1k --init-from "runs/2026-10-06_1630_e65_convnext_labels_r6_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e94_e60_own_r6 --run-dir 2026-10-09_1835_e94_convnext_small_384_from_e60_labels_r6_full3 $OWN6 $CONT --backbone convnext_small.fb_in22k_ft_in1k --init-from "runs/2026-10-06_1000_e60_convnext_small_384_labels_r3_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e95_own_r6_384_s142 --run-dir 2026-10-09_1835_e95_convnext_384_labels_r6_full3_seed142 $OWN6 --epochs 12 --seed 142 --backbone convnext_tiny.fb_in22k_ft_in1k
