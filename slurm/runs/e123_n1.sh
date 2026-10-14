#!/bin/bash
# E123 network 1 (queued 2026-10-15 00:00, 3090 batch job)
cd ~/RSNA-Knee-Abnormality-Detection
bash slurm/runs/e123_3d32_full.sh 1 > logs/e123_n1.log 2>&1
