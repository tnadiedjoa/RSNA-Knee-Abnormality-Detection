#!/bin/bash
#SBATCH --job-name=knee_train
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch slurm/train_p100.sbatch <scripts.train args>
# P100 (sm_60) needs the CUDA 12.6 torch build of .venv-p100 (fp16 + grad scaling, no bf16).
set -e
cd "$SLURM_SUBMIT_DIR"
.venv-p100/bin/python -m scripts.train "$@"
