"""Helpers of the unattended follow-up job (slurm/auto_1005.sh): no decision here needs judgement.

    python -m scripts.auto_followup scores          # fill pending rows of kaggle/submissions.csv, add new refs
    python -m scripts.auto_followup gain <run> ...  # OOF blend gain of adding these runs to Version 25's set
    python -m scripts.auto_followup note "<text>"   # add a paragraph to experiments.md

`scores` prints one line per change and exits 0 if something changed, 1 otherwise.
"""

import sys
from datetime import datetime
from pathlib import Path

from knee.config import LABELS, ROOT_DIR

LEDGER = ROOT_DIR / "kaggle" / "submissions.csv"
JOURNAL = ROOT_DIR / "experiments.md"
ANCHOR = "`kaggle/submissions.csv` (2026-10-02 16:30)"
COMPETITION = "rsna-knee-abnormality-detection"
V25 = ["e46", "e39", "e42", "e36", "e47", "e38", "e41", "e43", "e44", "e35", "e45", "e37", "e32", "e34", "e30", "e33"]


def note(text: str) -> None:
    s = JOURNAL.read_text()
    assert ANCHOR in s, "journal anchor missing"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    JOURNAL.write_text(s.replace(ANCHOR, f"Auto ({stamp}, slurm/auto_1005.sh): {text}\n\n{ANCHOR}", 1))


def scores() -> bool:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    subs = {str(s.ref): s for s in api.competition_submissions(COMPETITION)}
    lines = LEDGER.read_text().splitlines()
    known = {line.split(",")[0] for line in lines}
    changed = []
    for i, line in enumerate(lines):
        ref = line.split(",")[0]
        if not line.endswith(",,pending") or ref not in subs:
            continue
        s = subs[ref]
        status = str(s.status)
        if "COMPLETE" in status and s.public_score not in (None, ""):
            lines[i] = line[: -len(",,pending")] + f",{s.public_score},"
            changed.append(f"ref {ref} ({line.split(',')[2]} version {line.split(',')[3]}) scored {s.public_score}")
        elif "ERROR" in status:
            lines[i] = line[: -len(",,pending")] + ",,error"
            changed.append(f"ref {ref} ({line.split(',')[2]} version {line.split(',')[3]}) ended in ERROR")
    for ref, s in sorted(subs.items()):
        if ref in known or str(s.date) < "2026-10-05":
            continue
        notebook = str(s.url).split("/code/theophilenadiedjoa/")[-1].split("?")[0]
        script_id = str(s.url).split("scriptVersionId=")[-1]
        score = s.public_score if "COMPLETE" in str(s.status) and s.public_score not in (None, "") else ""
        tail = f"{score}," if score else ",pending"
        lines.append(f"{ref},{s.date:%Y-%m-%d %H:%M},{notebook},?,scriptVersionId {script_id} (version to match),{tail}")
        changed.append(f"new submission ref {ref} ({notebook}, scriptVersionId {script_id})" + (f" scored {score}" if score else ""))
    if changed:
        LEDGER.write_text("\n".join(lines) + "\n")
        note("Kaggle: " + "; ".join(changed) + ".")
        print("\n".join(changed))
    return bool(changed)


def gain(extra: list[str]) -> None:
    from scripts.eval_oof import load_oof, score, targets

    t = targets("qwen36_27b_v1")
    runs = {p.name.split("_")[2]: p for p in (ROOT_DIR / "runs").glob("2026-10-0*_e*_5f")}
    names = V25 + [e for e in extra if e not in V25]
    preds = {k: load_oof(runs[k]) for k in names}
    idx = preds["e30"].index

    def blend(keys):
        return sum(preds[k].loc[idx, LABELS].rank(pct=True) for k in keys) / len(keys)

    a, b = score(blend(V25), t), score(blend(names), t)
    print(f"{b['silver'] - a['silver']:+.4f} V25 {a['silver']:.4f}/{a['gold']:.4f} -> +{','.join(extra)} {b['silver']:.4f}/{b['gold']:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scores":
        sys.exit(0 if scores() else 1)
    elif cmd == "gain":
        gain(sys.argv[2:])
    elif cmd == "note":
        note(sys.argv[2])
