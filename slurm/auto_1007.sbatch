#!/bin/bash
#SBATCH --job-name=knee_auto
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
# Unattended push of 2026-10-08 (the interactive job, and the Claude session in it, ends at 01:14): starts after
# both E72 jobs end (--dependency), packages the 13 full-data runs only (E54 ... E70, E72 last; ~5 GB instead of
# 15, the 5-fold runs stay in the earlier dataset versions that older notebook versions pin), pushes knee-submit
# Version N = E72 alone, notes it in experiments.md and kaggle/A_SOUMETTRE.md, commits and pushes to GitHub.
cd ~/RSNA-Knee-Abnormality-Detection
PY=.venv/bin/python
export GIT_COMMITTER_NAME="Théophile Nadiedjoa" GIT_COMMITTER_EMAIL="theophile.nadied@gmail.com"
AUTHOR="Théophile Nadiedjoa <theophile.nadied@gmail.com>"
E72=runs/2026-10-07_1100_e72_convnext_mix7_r5_full3
log() { echo "$(date '+%F %T') $*"; }
note() { $PY -m scripts.auto_followup note "$1"; }
sync_commit() {  # sync_commit "<message>" <paths...>: rebase on GitHub, commit what changed, push (3 tries)
  local msg=$1; shift
  for i in 1 2 3; do
    if git pull --rebase --autostash -q && git add "$@" && { git diff --cached --quiet || git commit -q --author="$AUTHOR" -m "$msg"; } && git push -q; then
      log "pushed: $msg"; return 0
    fi
    sleep 60
  done
  log "commit/push FAILED: $msg"
}
log "start on $(hostname)"; df -h /tmp | tail -1
curl -s -o /dev/null -w "kaggle.com HTTP %{http_code}\n" https://www.kaggle.com
if ! [ -f $E72/checkpoints/fold0.pt ] || ! [ -f $E72/checkpoints/fold1.pt ] || ! [ -f $E72/checkpoints/fold2.pt ]; then
  note "E72 incomplete when its jobs ended ($(ls $E72/checkpoints 2>/dev/null | tr '\n' ' ')): nothing pushed by the unattended job."
  sync_commit "Auto: E72 incomplete" experiments.md; exit 1
fi
# the evening pushes (Versions 36/37, Claude session) must be over: wait for their marker, at most until 06:00
until [ -f logs/chain_v36.done ] || [ "$(date +%s)" -ge "$(date -d '06:00' +%s)" ]; do sleep 60; done
bash logs/push_version.sh "['*e72*']" "13 full-data runs (E54 ... E70) + E72 (7-table report base)" $(cat logs/day12c_runs.txt)
L=$(cat logs/push_version.last)
V=$(grep -oE "version [0-9]+" $L/push.txt | head -1 | awk '{print $2}'); C=$(grep -c "fold networks done" $L/check.txt)
log "pushed: $(cat $L/push.txt) | $(cat $L/status.txt)"
note "knee-submit **Version ${V:-?} = E72 alone** (7-table report base) pushed by the unattended job (\`slurm/auto_1007.sbatch\`): dataset of the 13 full-data runs, $C model(s) predicted on the visible test (\`$L\`)."
printf '\n| knee-submit | **%s** | E72 seul : labels moyennés sur 7 lecteurs LLM (3 nouveaux, CC0) | bat-il E54 (0,935) ? |\n' "${V:-?}" >> kaggle/A_SOUMETTRE.md
sync_commit "Auto: knee-submit Version ${V:-?} (E72 alone) pushed" experiments.md kaggle/A_SOUMETTRE.md kaggle/submit/knee-submit.ipynb $E72
date '+%F %T' > logs/chain_v38.done
