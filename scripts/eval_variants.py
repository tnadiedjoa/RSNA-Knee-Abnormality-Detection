"""Out-of-fold predictions of a trained run under other inference settings (no retraining): every fold
network predicts its own validation studies with, e.g., more windows or a wider slice span.

    python -m scripts.eval_variants <run> --variant w12:windows_eval=12 --variant w16:windows_eval=16 \
        --variant span:windows_eval=16,span=0.02-0.98

Writes runs/<run>/variants/oof_<name>.csv and prints silver / gold for each variant next to the
run's own OOF (the first variant with the training settings should reproduce it).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from knee.config import CACHE_DIR, DATA_DIR, LABELS, RUNS_DIR
from knee.data import StudyDataset, assign_slots, series_table
from knee.labels import load_soft
from knee.model import KneeNet
from scripts.eval_oof import load_oof, score, targets


def parse_variant(text: str, cfg: dict) -> tuple[str, dict]:
    name, _, kv = text.partition(":")
    over = {"windows_eval": cfg["windows_eval"], "span": tuple(cfg.get("span", (0.08, 0.92))), "zoom": cfg.get("zoom", 1.0)}
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        over[k] = tuple(map(float, v.split("-"))) if k == "span" else (int(v) if k == "windows_eval" else float(v))
    return name, over


@torch.no_grad()
def predict(model, ds, device, bs):
    out = []
    for batch in DataLoader(ds, bs, num_workers=ds_workers, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = model(batch["x"].to(device), batch["pos"].to(device), batch["mask"].to(device), batch["seq"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def main():
    global ds_workers
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--variant", action="append", required=True, help="name:key=value,... (windows_eval, span=a-b, zoom)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--raw-dir", default=None)
    args = p.parse_args()
    ds_workers = args.workers
    run = RUNS_DIR / args.run
    cfg = json.loads(sorted(run.glob("config*.json"))[-1].read_text())
    cache_dir = Path(cfg.get("cache_dir", CACHE_DIR))
    raw = Path(args.raw_dir) if args.raw_dir else None

    # the same labelled studies and folds as scripts/train.py
    studies = pd.read_csv(DATA_DIR / "train.csv").merge(pd.read_csv(DATA_DIR / "folds.csv")[["StudyInstanceUID", "fold"]],
                                                        on="StudyInstanceUID")
    labelled = load_soft(cfg["labels"]).dropna(subset=LABELS)["StudyInstanceUID"]
    studies = studies[studies["StudyInstanceUID"].isin(labelled)].reset_index(drop=True)
    slots = assign_slots(series_table("train")).set_index("StudyInstanceUID").reindex(studies["StudyInstanceUID"]).reset_index()
    device = torch.device("cuda")
    variants = [parse_variant(v, cfg) for v in args.variant]
    (run / "variants").mkdir(exist_ok=True)
    preds = {name: [] for name, _ in variants}
    for fold in range(5):
        va = np.where(studies["fold"].to_numpy() == fold)[0]
        model = KneeNet.from_config(cfg)
        model.load_state_dict(torch.load(run / "checkpoints" / f"fold{fold}.pt", map_location="cpu"))
        model = model.to(device).eval().to(memory_format=torch.channels_last)
        for name, o in variants:
            ds = StudyDataset(slots.iloc[va], n_windows=o["windows_eval"], img_size=cfg["img_size"], span=o["span"],
                              zoom=o["zoom"], cache_dir=cache_dir, raw_dir=raw)
            bs = max(1, 4 * 12 // o["windows_eval"])  # as knee/infer.py (4 studies x 12 windows), fewer if denser
            df = pd.DataFrame(predict(model, ds, device, bs), columns=LABELS)
            df.insert(0, "StudyInstanceUID", studies["StudyInstanceUID"].to_numpy()[va])
            df.insert(1, "fold", fold)
            preds[name].append(df)
        print(f"fold {fold} done", flush=True)

    t = targets(cfg["labels"])
    rows = {"stored OOF": score(load_oof(run), t)}
    for name, o in variants:
        oof = pd.concat(preds[name])
        oof.to_csv(run / "variants" / f"oof_{name}.csv", index=False)
        rows[f"{name} {o}"] = score(oof.set_index("StudyInstanceUID"), t)
    print(pd.DataFrame(rows).T[["silver", "gold"]].round(4).to_string())


if __name__ == "__main__":
    main()
