#!/bin/bash
# E121 (2026-10-14 17:35, interactive 3090): E115 with R3D-18 (E116: alone 0.9553, +0.0005 silver over E79 + E46 + E114). Earlier notes, E115: E114's fold-0 CV test: R(2+1)D-18 alone silver 0.9573 (as a
# 2.5D 384 px run), rank correlation 0.875 with E79 / E46 (2.5D runs 0.90-0.94), +0.0016 silver added at 0.2 to E79 + E46
# (another 2.5D run: +0.0006): the most diverse member measured so far. Same settings (24 slices, 224 px, 10 epochs, bs 2 x 4)
# on E75's targets (7 tables + mixpub r5 teacher, the best LB line). Usage: bash e115_3d_full.sh <network index ...>
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
for n in "$@"; do
  .venv/bin/python -m scripts.train --name e121_r3d_full_n$n --run-dir 2026-10-14_1735_e121_r3d18_stack24_224_mix7_r6_full3 $MIX7 --backbone r3d_18 --stack-slices 24 --img-size 224 --epochs 10 --bs 2 --accum 4 --grad-ckpt --full-fit --raw-dir /tmp/knee_raw384 --workers 4 --folds $n
done
