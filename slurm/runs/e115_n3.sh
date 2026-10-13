#!/bin/bash
# E115 network 3 (queued 2026-10-13 19:25, 3090 batch job): one more full-data 3D network of the same recipe, seeded by its
# index, so that the 3D member of the Kaggle blends averages 5 networks instead of 3.
cd ~/RSNA-Knee-Abnormality-Detection
bash slurm/runs/e115_3d_full.sh 3 > logs/e115_n3.log 2>&1
