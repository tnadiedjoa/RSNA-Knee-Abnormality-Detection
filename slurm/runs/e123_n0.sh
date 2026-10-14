#!/bin/bash
# E123 network 0 (queued 2026-10-15 00:00, 3090 batch job)
cd ~/RSNA-Knee-Abnormality-Detection
bash slurm/runs/e123_3d32_full.sh 0 > logs/e123_n0.log 2>&1
