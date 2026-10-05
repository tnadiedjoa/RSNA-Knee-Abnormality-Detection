#!/bin/bash
#SBATCH --job-name=knee_auto
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x_%j.out
# Unattended follow-up of 2026-10-06, so nothing depends on the user's interactive session (a Claude session
# running there dies with it). Deterministic steps only, each one logged in experiments.md and pushed to GitHub:
#  1. knee-submit Versions 26 (E54 alone) and 27 (Version 25 + E53 + E50) are pushed by the Claude session, which
#     touches logs/day10_pushed when done; if that marker is still missing at 02:30, they are pushed from here;
#  2. knee-blend from kaggle/blend (our weight 0.15) then kaggle/blend_w10 (0.10), both with those 18 runs;
#  3. kaggle/A_SOUMETTRE.md: the exact versions for the user to submit;
#  4. every 10 min until the time limit: Kaggle scores -> kaggle/submissions.csv; E55 complete -> its OOF gain.
cd ~/RSNA-Knee-Abnormality-Detection
PY=.venv/bin/python
export GIT_COMMITTER_NAME="Théophile Nadiedjoa" GIT_COMMITTER_EMAIL="theophile.nadied@gmail.com"
AUTHOR="Théophile Nadiedjoa <theophile.nadied@gmail.com>"
V27_MODELS="['*e46*', '*e53*', '*e39*', '*e42*', '*e50*', '*e36*', '*e47*', '*e38*', '*e41*', '*e43*', '*e44*', '*e35*', '*e45*', '*e37*', '*e32*', '*e34*', '*e30*', '*e33*']"
E55=runs/2026-10-06_0050_e55_convnext_384_mixpub_r6_seed2_5f

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
version_of() { grep -oE "version [0-9]+" "$1" | head -1 | awk '{print $2}'; }

log "start on $(hostname)"
curl -s -o /dev/null -w "kaggle.com HTTP %{http_code}\n" https://www.kaggle.com; git ls-remote -q origin HEAD | cut -c1-12
$PY -c "from kaggle.api.kaggle_api_extended import KaggleApi as K; a=K(); a.authenticate(); print('kaggle api ok', len(a.competition_submissions('rsna-knee-abnormality-detection')))"

# 1. knee-submit Versions 26/27
until [ -f logs/day10_pushed ] || [ "$(date +%s)" -ge "$(date -d '2026-10-06 02:30' +%s)" ]; do sleep 60; done
if [ -f logs/day10_pushed ]; then
  read -r V26 V27 < logs/day10_pushed
  log "Versions $V26 / $V27 pushed by the Claude session"
else
  log "marker missing at 02:30: pushing Versions 26/27 from here"
  bash logs/push_version.sh "['*e54*']" "rounds 3-6 + public-table runs, E50, E53, E54 (full data)" $(cat logs/day10_runs.txt)
  V26=$(version_of "$(cat logs/push_version.last)/push.txt"); C26=$(grep -c elapsed "$(cat logs/push_version.last)/check.txt")
  sleep 60
  SKIP_PACKAGE=1 bash logs/push_version.sh "$V27_MODELS" "Version 25 + E53 + E50 (18 runs)"
  V27=$(version_of "$(cat logs/push_version.last)/push.txt"); C27=$(grep -c elapsed "$(cat logs/push_version.last)/check.txt")
  note "knee-submit Version $V26 (E54 alone; $C26 run(s) predicted on the visible test) and Version $V27 (Version 25 + E53 + E50; $C27 of 18 runs predicted) pushed by the unattended job (the Claude session did not finish them)."
  sync_commit "Auto: knee-submit Versions $V26 and $V27 pushed" experiments.md kaggle/submit/knee-submit.ipynb
fi

# 2. knee-blend at weights 0.15 and 0.10 (same 18 runs, dataset of 2026-10-06)
bash logs/push_blends.sh kaggle/blend kaggle/blend_w10 > logs/day10_blends.txt 2>&1
B=($(grep -oE "version [0-9]+ successfully" logs/day10_blends.txt | awk '{print $2}'))
BL=$(grep -c "BLEND written" logs/day10_blends.txt)
note "knee-blend versions ${B[0]:-?} (public 0.85 + 18 runs 0.15) and ${B[1]:-?} (public 0.90 + 18 runs 0.10) pushed and run on the visible test: $BL of 2 wrote the blend (logs/day10_blends.txt)."

# 3. what the user submits
cat > kaggle/A_SOUMETTRE.md <<EOF
# À soumettre (écrit automatiquement le $(date '+%d/%m à %H:%M'))

Sur Kaggle : Submit → choisir le notebook → choisir la version dans le menu (Kaggle propose la dernière par défaut).

| Notebook | Version | Contenu | Question |
|---|---|---|---|
| knee-submit | **$V26** | E54 seul : la recette d'E41 entraînée sur 100 % des données | bat-il E41 seul (0,931) ? |
| knee-submit | **$V27** | Version 25 + E53 + E50 (18 modèles) | bat-il la Version 25 (0,936) ? |
| knee-blend | **${B[0]:-?}** | public 85 % + nos 18 modèles 15 % | bat-il le blend v3 (0,944) ? |
| knee-blend | **${B[1]:-?}** | public 90 % + nos 18 modèles 10 % | quel poids est le meilleur ? |

Les blends mettent environ 9 h à être notés, knee-submit environ 3 h.
EOF
sync_commit "Auto: knee-blend versions ${B[0]:-?}/${B[1]:-?} pushed, kaggle/A_SOUMETTRE.md" experiments.md kaggle/A_SOUMETTRE.md

# 4. scores and E55
E55_DONE=0
while true; do
  if $PY -m scripts.auto_followup scores >> logs/auto_scores.log 2>&1; then
    sync_commit "Auto: Kaggle scores" kaggle/submissions.csv experiments.md
  fi
  if [ $E55_DONE = 0 ] && [ "$(ls $E55/oof_fold*.csv 2>/dev/null | wc -l)" -ge 5 ]; then
    E55_DONE=1
    S=$($PY -m scripts.eval_oof $E55 2>/dev/null | awk 'NR==2{print "silver " $2 ", gold " $3}')
    G=$($PY -m scripts.auto_followup gain e53 e50 e55 2>/dev/null | tail -1)
    note "**E55 complete** (E46 recipe, seed 2, 384 px): $S. OOF gain on Version 25's 16 runs: $G."
    sync_commit "Auto: E55 complete" experiments.md $E55
  fi
  sleep 600
done
