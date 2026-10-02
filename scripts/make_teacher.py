"""Store the mean OOF probability of finished 5-fold runs as a teacher pseudo-run for soft bootstrapping.

    python -m scripts.make_teacher <out run dir> "<what>" <run> [<run> ...]

Writes runs/<out>/oof_fold{0..4}.csv (same layout as a training run, so --bootstrap-run reads it)
and teacher.json (what it is, which runs). Every run must have all 5 OOF files with the same studies.
"""

import json
import sys

import pandas as pd

from knee.config import ROOT_DIR

RUNS_DIR = ROOT_DIR / "runs"


def main():
    out, what, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    assert runs, "no runs given"
    out_dir = RUNS_DIR / out
    out_dir.mkdir(exist_ok=False)
    for f in range(5):
        dfs = [pd.read_csv(RUNS_DIR / r / f"oof_fold{f}.csv", dtype={"StudyInstanceUID": str}) for r in runs]
        keys = dfs[0][["StudyInstanceUID", "fold"]]
        for r, d in zip(runs, dfs):
            assert d[["StudyInstanceUID", "fold"]].equals(keys), f"{r} fold {f}: different studies"
        labels = [c for c in dfs[0].columns if c not in ("StudyInstanceUID", "fold")]
        mean = sum(d[labels] for d in dfs) / len(dfs)
        pd.concat([keys, mean], axis=1).to_csv(out_dir / f"oof_fold{f}.csv", index=False)
        print(f"fold {f}: {len(keys)} studies")
    (out_dir / "teacher.json").write_text(json.dumps({"what": what, "runs": runs}, indent=1))
    print(out_dir)


if __name__ == "__main__":
    main()
