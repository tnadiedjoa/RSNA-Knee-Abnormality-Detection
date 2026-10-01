#!/bin/bash
#SBATCH --job-name=knee_multi
#SBATCH --partition=3090
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch [-p P100] [--gres=gpu:N] slurm/multi_gpu.sbatch <runs file>   (Slurm here: <= 10 CPUs per job, <= 4 running jobs per user)
# One training per GPU in a single job (Slurm allows few running jobs per user, not few GPUs).
# Each line of the runs file holds the scripts.train arguments of one training. The cache is
# unpacked once to the node's local disk when there is room: with 4 CPUs per GPU, JPEG decoding
# would otherwise limit small backbones.
set -e
cd "$SLURM_SUBMIT_DIR"
PY=.venv/bin/python
[ "$SLURM_JOB_PARTITION" = P100 ] && PY=.venv-p100/bin/python  # sm_60 needs the CUDA 12.6 torch build
mapfile -t RUNS < <(grep -v '^\s*#' "$1" | grep -v '^\s*$')
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
[ ${#RUNS[@]} -le ${#GPUS[@]} ] || { echo "${#RUNS[@]} runs for ${#GPUS[@]} GPUs"; exit 1; }
WORKERS=$(( SLURM_CPUS_PER_TASK / ${#GPUS[@]} ))
RAW=()
df -h /tmp
if [ "$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)" -gt "${RAW_GB:-60}" ]; then  # RAW_GB=100 for the 384 px cache
  $PY -m scripts.unpack_cache /tmp/knee_raw --workers "$SLURM_CPUS_PER_TASK"
  RAW=(--raw-dir /tmp/knee_raw)
fi
for i in "${!RUNS[@]}"; do
  name=$(echo "${RUNS[$i]}" | grep -oP '(?<=--name )\S+')
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY -m scripts.train ${RUNS[$i]} "${RAW[@]}" --workers "$WORKERS" \
    > "logs/${name}_${SLURM_JOB_ID}.log" 2>&1 &
done
wait
