#!/bin/bash
# E114: a 3D model, CV test (queued 2026-10-12 22:30, 3090 batch job). Each series becomes one clip of 24 slices spread over
# its span at 224 px (--stack-slices 24) read by a video ResNet(2+1)D-18 pretrained on Kinetics-400 (3D convolutions,
# knee/model.py Video3D); its depth axis (24 / 8 = 3 positions) stays as tokens for the same transformer and finding
# queries. Labels of E41 / E79 (Qwen + 3 public tables + round-4 teacher at 0.5) so that folds 0 and 1 compare with E79,
# E42, E46: silver / gold alone, rank correlation, gain when added (scratchpad zoom_cv.py logic).
cd ~/RSNA-Knee-Abnormality-Detection
LAB="--labels qwen36_27b_v1 --mix-public pilkwang_v2 steven_v2 lixin_sol56 --bootstrap-run 2026-10-03_1220_teacher_r4_oof_mean --bootstrap-alpha 0.5"
for f in 0 1; do
  .venv/bin/python -m scripts.train --name e114_r2p1d_f$f --run-dir 2026-10-12_2230_e114_cv_r2plus1d18_stack24_224 $LAB --backbone r2plus1d_18 --stack-slices 24 --img-size 224 --epochs 10 --bs 2 --accum 4 --grad-ckpt --raw-dir /tmp/knee_raw384 --workers 4 --folds $f
done

# appended 2026-10-13 02:50, read by this job after its fold-1 CV run: the 3D model's full-data network 1 (E115)
bash slurm/runs/e115_3d_full.sh 1 > logs/e115_n1.log 2>&1
