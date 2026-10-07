#!/bin/bash
#SBATCH --job-name=knee_auto
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
# Unattended push of 2026-10-08 (the interactive job, and the Claude session in it, ends at 01:14): starts after
# the E72 / E73 jobs end (--dependency), packages the 14 full-data runs only (E54 ... E70, E72, E73 last; ~5 GB
# instead of 15, the 5-fold runs stay in the earlier dataset versions that older notebook versions pin), pushes
# knee-submit Version N = E72 alone and N+1 = E73 alone, notes them in experiments.md and kaggle/A_SOUMETTRE.md,
# commits and pushes to GitHub.
cd ~/RSNA-Knee-Abnormality-Detection
PY=.venv/bin/python
export GIT_COMMITTER_NAME="Théophile Nadiedjoa" GIT_COMMITTER_EMAIL="theophile.nadied@gmail.com"
AUTHOR="Théophile Nadiedjoa <theophile.nadied@gmail.com>"
E72=runs/2026-10-07_1100_e72_convnext_mix7_r5_full3; E73=runs/2026-10-07_1115_e73_convnext_mix7_r5_seqemb_full3
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
for R in $E72 $E73; do
  if ! [ -f $R/checkpoints/fold0.pt ] || ! [ -f $R/checkpoints/fold1.pt ] || ! [ -f $R/checkpoints/fold2.pt ]; then
    note "$(basename $R) incomplete when its jobs ended ($(ls $R/checkpoints 2>/dev/null | tr '\n' ' ')): nothing pushed by the unattended job."
    sync_commit "Auto: $(basename $R) incomplete" experiments.md; exit 1
  fi
done
# the evening pushes (Versions 36/37, Claude session) must be over: wait for their marker, at most until 06:00
until [ -f logs/chain_v36.done ] || [ "$(date +%s)" -ge "$(date -d '06:00' +%s)" ]; do sleep 60; done
bash logs/push_version.sh "['*e72*']" "14 full-data runs (E54 ... E70) + E72 (7-table report base), E73 (+ seq-emb)" $(cat logs/day12d_runs.txt)
L=$(cat logs/push_version.last)
V=$(grep -oE "version [0-9]+" $L/push.txt | head -1 | awk '{print $2}'); C=$(grep -c "fold networks done" $L/check.txt)
log "pushed: $(cat $L/push.txt) | $(cat $L/status.txt)"
sleep 60
SKIP_PACKAGE=1 bash logs/push_version.sh "['*e73*']" "E73 alone"
L2=$(cat logs/push_version.last)
V2=$(grep -oE "version [0-9]+" $L2/push.txt | head -1 | awk '{print $2}'); C2=$(grep -c "fold networks done" $L2/check.txt)
log "pushed: $(cat $L2/push.txt) | $(cat $L2/status.txt)"
note "knee-submit **Version ${V:-?} = E72 alone** (7-table report base) and **Version ${V2:-?} = E73 alone** (E72 + seq-emb) pushed by the unattended job (\`slurm/auto_1007.sbatch\`): dataset of the 14 full-data runs; $C and $C2 model(s) predicted on the visible test (\`$L\`, \`$L2\`)."
printf '\n| knee-submit | **%s** | E72 seul : labels moyennés sur 7 lecteurs LLM (3 nouveaux, CC0) | bat-il E54 (0,935) ? |\n| knee-submit | **%s** | E73 seul : E72 + type de séquence IRM | bat-il E72 ? |\n' "${V:-?}" "${V2:-?}" >> kaggle/A_SOUMETTRE.md
sync_commit "Auto: knee-submit Versions ${V:-?} (E72) and ${V2:-?} (E73) pushed" experiments.md kaggle/A_SOUMETTRE.md kaggle/submit/knee-submit.ipynb $E72 $E73
date '+%F %T' > logs/chain_v38.done
