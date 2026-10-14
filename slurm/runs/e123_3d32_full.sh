#!/bin/bash
# E123 (queued 2026-10-15 00:00, 3090 batch): E115 with 32 slices per clip (E118 fold 0: alone silver 0.9583 vs 0.9573 at 24 slices, +0.0021 at 0.33 over E79 + E46, the best 3D test so far). Earlier notes, E115: E114's fold-0 CV test: R(2+1)D-18 alone silver 0.9573 (as a
# 2.5D 384 px run), rank correlation 0.875 with E79 / E46 (2.5D runs 0.90-0.94), +0.0016 silver added at 0.2 to E79 + E46
# (another 2.5D run: +0.0006): the most diverse member measured so far. Same settings (24 slices, 224 px, 10 epochs, bs 2 x 4)
# on E75's targets (7 tables + mixpub r5 teacher, the best LB line). Usage: bash e115_3d_full.sh <network index ...>
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
for n in "$@"; do
  .venv/bin/python -m scripts.train --name e123_3d32_full_n$n --run-dir 2026-10-15_0000_e123_r2plus1d18_stack32_224_mix7_r6_full3 $MIX7 --backbone r2plus1d_18 --stack-slices 32 --img-size 224 --epochs 10 --bs 2 --accum 4 --grad-ckpt --full-fit --raw-dir /tmp/knee_raw384 --workers 4 --folds $n
done
