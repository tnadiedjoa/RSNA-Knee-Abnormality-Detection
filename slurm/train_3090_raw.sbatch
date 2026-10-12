#!/bin/bash
#SBATCH --job-name=knee_raw
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch slurm/train_3090_raw.sbatch <script>: unpack the 384 px cache to this job's /tmp, then run <script> (bash),
# which finds it at /tmp/knee_raw384 (each Slurm job has its own /tmp here).
set -e
cd "$SLURM_SUBMIT_DIR"
export KNEE_CACHE_DIR=$PWD/data/cache384 KNEE_CACHE_SIZE=384
[ -d /tmp/knee_raw384 ] || .venv/bin/python -m scripts.unpack_cache /tmp/knee_raw384 --workers 4
bash "$1"
