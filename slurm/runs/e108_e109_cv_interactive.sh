#!/bin/bash
# CV test of the tighter views (second interactive 3090, job 1033826, queued 2026-10-11 22:10, until 10-12 01:23).
# E79's recipe (E41's fold networks continued 4 epochs at 384 px, E41's labels) with --zoom 1.5 (E108, central 100 mm) and
# --zoom 2.0 (E109, central 75 mm), folds 0 then 1, against E79 (zoom 1.0: 0.9597 / 0.9580) and E42 (0.9577 / 0.9550):
# their own silver/gold, and above all their rank correlation with E79 / E42 / E46 and what they add to them on the folds.
cd ~/RSNA-Knee-Abnormality-Detection
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4  # each job has its own /tmp
E79="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5 --backbone convnext_tiny.fb_in22k_ft_in1k --grad-ckpt --img-size 384 --epochs 4 --lr 1e-4 --lr-head 5e-4 --raw-dir /tmp/knee_raw384 --workers 4 --init-from runs/2026-10-03_1340_e41_convnext_mixpub_r5_5f/checkpoints/fold{fold}.pt"
for f in 0 1; do
  .venv/bin/python -m scripts.train --name e108_zoom150_f$f --run-dir 2026-10-11_2210_e108_cv_zoom100mm_from_e41 $E79 --zoom 1.5 --folds $f
  .venv/bin/python -m scripts.train --name e109_zoom200_f$f --run-dir 2026-10-11_2210_e109_cv_zoom75mm_from_e41 $E79 --zoom 2.0 --folds $f
done
