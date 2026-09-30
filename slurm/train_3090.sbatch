#!/bin/bash
#SBATCH --job-name=knee_train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch slurm/train_3090.sbatch <scripts.train args>
set -e
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -m scripts.train "$@"
