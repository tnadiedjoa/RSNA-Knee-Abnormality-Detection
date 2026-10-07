#!/bin/bash
#SBATCH --job-name=knee_auto
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Unattended pushes of 2026-10-08 (the Claude session may not be back): starts after E77's two jobs end
# (--dependency=afterany:<E77 jobs>), waits for the E72 push of slurm/auto_1007.sbatch (logs/chain_v38.done), then
#  1. packages the 13 full-data runs of logs/day12c_runs.txt + E75 (3090 run, when complete) + E77 (last: new in the
#     package) and pushes knee-submit "E77 alone", then "E75 alone" (same dataset version) when E75 is complete;
#  2. waits for E78's three networks (at most until 16:00), packages again with E78 last, pushes "E78 alone";
# each version noted in experiments.md and kaggle/A_SOUMETTRE.md, committed and pushed to GitHub.
cd ~/RSNA-Knee-Abnormality-Detection
PY=.venv/bin/python
export GIT_COMMITTER_NAME="Théophile Nadiedjoa" GIT_COMMITTER_EMAIL="theophile.nadied@gmail.com"
AUTHOR="Théophile Nadiedjoa <theophile.nadied@gmail.com>"
E75=runs/2026-10-07_1420_e75_convnext_384_mix7_r6_full3
E77=runs/2026-10-07_1400_e77_convnext_mixpub_r5_silentalpha_full3
E78=runs/2026-10-07_1400_e78_convnext_mixpub_r5_sitebias_full3
log() { echo "$(date '+%F %T') $*"; }
note() { $PY -m scripts.auto_followup note "$1"; }
complete() { [ -f $1/checkpoints/fold0.pt ] && [ -f $1/checkpoints/fold1.pt ] && [ -f $1/checkpoints/fold2.pt ]; }
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
row() {  # row "<version>" "<contenu>" "<question>": one line of the night table of kaggle/A_SOUMETTRE.md
  [ -n "$(tail -c1 kaggle/A_SOUMETTRE.md)" ] && echo >> kaggle/A_SOUMETTRE.md
  printf '| knee-submit | **%s** | %s | %s |\n' "$1" "$2" "$3" >> kaggle/A_SOUMETTRE.md
}
push() {  # push "<patterns>" "<description>" [runs...]: knee-submit version; prints the version number
  bash logs/push_version.sh "$@" >&2
  local L=$(cat logs/push_version.last)
  log "pushed: $(cat $L/push.txt) | $(cat $L/status.txt) | $(grep -c 'fold networks done' $L/check.txt) model(s) on the visible test ($L)" >&2
  grep -oE "version [0-9]+" $L/push.txt | head -1 | awk '{print $2}'
}
log "start on $(hostname)"; df -h /tmp | tail -1
curl -s -o /dev/null -w "kaggle.com HTTP %{http_code}\n" https://www.kaggle.com
until [ -f logs/chain_v38.done ] || [ "$(date +%s)" -ge "$(date -d '12:00' +%s)" ]; do sleep 60; done  # E72 push first
if ! complete $E77; then
  note "E77 incomplete when its jobs ended ($(ls $E77/checkpoints 2>/dev/null | tr '\n' ' ')): not pushed by the unattended job."
else
  extra=(); complete $E75 && extra=($E75)
  V=$(push "['*e77*']" "13 full-data runs + E75 + E77 (teacher weight by mention)" $(cat logs/day12c_runs.txt) "${extra[@]}" $E77)
  note "knee-submit **Version ${V:-?} = E77 alone** (teacher weight by what the report says: silent cells 0.9, absent/severe 0.25) pushed by the unattended job (\`slurm/auto_1008.sbatch\`)."
  row "${V:-?}" "E77 seul : cases non mentionnées → on suit l'image" "bat-il E54 (0,935) ?"
  if complete $E75; then
    V=$(SKIP_PACKAGE=1 push "['*e75*']" "E75 alone")
    note "knee-submit **Version ${V:-?} = E75 alone** (E56 recipe with the 7-table base, 384 px, trained on a 3090) pushed by the unattended job."
    row "${V:-?}" "E75 seul : meilleure recette 384 px (E56) + labels à 7 lecteurs" "bat-il E56 (0,938) ?"
  fi
  sync_commit "Auto: knee-submit E77 (and E75) alone pushed" experiments.md kaggle/A_SOUMETTRE.md kaggle/submit/knee-submit.ipynb $E77 $E75
fi
until complete $E78 || [ "$(date +%s)" -ge "$(date -d '16:00' +%s)" ]; do sleep 120; done
if complete $E78; then
  sleep 60
  extra=(); complete $E75 && extra=($E75); complete $E77 && extra+=($E77)
  V=$(push "['*e78*']" "13 full-data runs + E75 + E77 + E78 (site bias)" $(cat logs/day12c_runs.txt) "${extra[@]}" $E78)
  note "knee-submit **Version ${V:-?} = E78 alone** (per-site bias on silent cells, training only) pushed by the unattended job."
  row "${V:-?}" "E78 seul : biais par site pendant l'entraînement" "bat-il E54 (0,935) ?"
  sync_commit "Auto: knee-submit E78 alone pushed" experiments.md kaggle/A_SOUMETTRE.md kaggle/submit/knee-submit.ipynb $E78
else
  note "E78 incomplete at 16:00 ($(ls $E78/checkpoints 2>/dev/null | tr '\n' ' ')): not pushed by the unattended job."
  sync_commit "Auto: E78 incomplete" experiments.md
fi
date '+%F %T' > logs/auto_1008.done
