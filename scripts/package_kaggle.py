"""Bundle the `knee` package and trained runs into a private Kaggle dataset for the offline
submission notebook (kaggle/submit/).

    python -m scripts.package_kaggle runs/<run1> [runs/<run2> ...] [--message "..."]

The dataset holds knee/*.py plus, per run, its config and fold checkpoints. It is private:
nothing derived from the competition is published during the competition.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from knee.config import ROOT_DIR

DATASET = "theophilenadiedjoa/knee-models"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--message", default="update")
    p.add_argument("--dataset", default=DATASET, help="owner/slug (default: knee-models)")
    # staging copy (~14 GB): node-local disk, not the 100 GB home (kaggle/_dataset until 2026-10-07)
    p.add_argument("--out", default=f"/tmp/knee_dataset_{os.environ.get('USER', 'user')}")
    args = p.parse_args()

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "knee").mkdir(parents=True)
    for f in (ROOT_DIR / "knee").glob("*.py"):
        if f.name != "remote_zip.py":  # needs internet, useless offline
            shutil.copy(f, out / "knee" / f.name)
    shutil.copytree(ROOT_DIR / "kaggle" / "wheels", out / "wheels")  # DICOM decoders missing on Kaggle
    for run in map(Path, args.runs):
        dst = out / "models" / run.name
        dst.mkdir(parents=True)
        cfg = sorted(run.glob("config*.json"))[-1]
        shutil.copy(cfg, dst / "config.json")
        for ckpt in sorted((run / "checkpoints").glob("fold*.pt")):
            shutil.copy(ckpt, dst / ckpt.name)
        print(run.name, [c.name for c in (run / "checkpoints").glob("fold*.pt")])
    meta = {"title": args.dataset.split("/")[1], "id": args.dataset, "licenses": [{"name": "CC-BY-NC-4.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta))

    kaggle = str(ROOT_DIR / ".venv" / "bin" / "kaggle")
    exists = subprocess.run([kaggle, "datasets", "status", args.dataset], capture_output=True, text=True).returncode == 0
    cmd = ([kaggle, "datasets", "version", "-p", str(out), "-m", args.message, "--dir-mode", "zip"] if exists
           else [kaggle, "datasets", "create", "-p", str(out), "--dir-mode", "zip"])
    print(subprocess.run(cmd, capture_output=True, text=True).stdout[-500:])


if __name__ == "__main__":
    main()
