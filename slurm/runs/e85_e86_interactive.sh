#!/bin/bash
# On the interactive 3090 (nodemm01), queued 2026-10-08 22:55 after E82, two more members of E75's line (E56 recipe +
# 7-table base + mixpub r5 teacher, 384 px; Version 41, 0.940, best own single):
#  E85 = E72's three full-data 224 px networks (same 7 tables, teacher r4) continued 4 epochs at 384 px on E75's
#        targets (teacher mixpub r5): an E75 reached through another path, a third of the cost (E79 recipe);
#  E86 = E75 from scratch with seed 142 (as E59 is to E56).
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
MIX7="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 medgemma27_p3 mistral24_p3 qwen32_p3 --bootstrap-run 2026-10-03_2210_teacher_mixpub_r5_oof_mean --bootstrap-alpha 0.5 --backbone convnext_tiny.fb_in22k_ft_in1k --grad-ckpt --img-size 384 --full-fit --folds 0 1 2 --raw-dir /tmp/knee_raw384 --workers 4"
.venv/bin/python -m scripts.train --name e85_e72_to_384_r6 --run-dir 2026-10-08_2255_e85_convnext_384_from_e72_mix7_r6_full3 $MIX7 --epochs 4 --lr 1e-4 --lr-head 5e-4 --init-from "runs/2026-10-07_1100_e72_convnext_mix7_r5_full3/checkpoints/fold{fold}.pt"
.venv/bin/python -m scripts.train --name e86_full_384_mix7_r6_seed142 --run-dir 2026-10-08_2255_e86_convnext_384_mix7_r6_full3_seed142 $MIX7 --seed 142
