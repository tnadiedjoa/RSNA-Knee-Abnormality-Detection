#!/bin/bash
# On the interactive 3090 (nodemm01), queued 2026-10-09 10:00 after E86, until the job ends (~19:07): two more members on
# E75's targets (7-table base + mixpub r5 teacher; E75 = Version 41, 0.940, best own single), 4 epochs each:
#  E89 = E60's networks (ConvNeXt-small, 384 px, own labels round 3) continued on E75's targets: backbone diversity
#        (~1.6 h per network);
#  E90 = E58's networks (224 px, per-finding teacher weights) continued at 384 px on E75's targets (~1 h per network).
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5 --grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4 --epochs 4 --lr 1e-4 --lr-head 5e-4"
.venv/bin/python -m scripts.train --name e89_e60_mix7_r6 --run-dir 2026-10-09_1000_e89_convnext_small_384_from_e60_mix7_r6_full3 $MIX7 --backbone convnext_small.fb_in22k_ft_in1k --init-from "runs/2026-10-06_1000_e60_convnext_small_384_labels_r3_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e90_e58_to_384_r6 --run-dir 2026-10-09_1000_e90_convnext_384_from_e58_mix7_r6_full3 $MIX7 --backbone convnext_tiny.fb_in22k_ft_in1k --init-from "runs/2026-10-06_0915_e58_convnext_mixpub_r5_alphalabels_full3/checkpoints/fold{fold}.pt"
