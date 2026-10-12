#!/bin/bash
# On the interactive 3090 of job 1036315 (nodemm05, queued 2026-10-12 19:45): E113 = E75's recipe (7-table base + mixpub r5
# teacher, 384 px, 12 epochs, full data) with a ConvNeXt-small backbone from scratch (E60 / E89 are small networks on other
# targets): a stronger, different member for the blends on heliosli's 0.954 base. ~5.5 h per network.
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4  # each job has its own /tmp
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5"
.venv/bin/python -m scripts.train --name e113_small_384_mix7 --run-dir 2026-10-12_1945_e113_convnext_small_384_mix7_r6_full3 $MIX7 --grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4 --backbone convnext_small.fb_in22k_ft_in1k
