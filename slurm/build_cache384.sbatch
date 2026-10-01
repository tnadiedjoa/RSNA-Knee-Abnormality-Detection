#!/bin/bash
#SBATCH --job-name=knee_c384
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# 384 px cache of the model's input series, in its own folder (data/series_meta.parquet untouched).
# usage: sbatch slurm/build_cache384.sbatch <shard> <nshards>
set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
KNEE_CACHE_DIR=data/cache384 python -m scripts.build_cache --size 384 --sides-from-meta --slots-only \
    --workers 32 --shard "$1" --nshards "$2"
