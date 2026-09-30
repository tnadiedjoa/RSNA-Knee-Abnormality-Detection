#!/bin/bash
#SBATCH --job-name=knee_cache
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch slurm/build_cache.sbatch <shard> <nshards>
set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -m scripts.build_cache --convert --workers 10 --shard "$1" --nshards "$2"
python -m scripts.build_cache --workers 32 --shard "$1" --nshards "$2"
