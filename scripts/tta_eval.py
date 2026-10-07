"""Test-time augmentation on one validation fold: predict it under several input settings (windows, zoom, slice
span) with the fold's own network and score each setting and their averages against the same targets.

    python -m scripts.tta_eval runs/<run> --fold 0 [--limit 300] [--settings w24 "span wide"] [--save preds.npz]

Each setting and mean is also compared with "w24" (24 windows, training span and zoom) by a paired bootstrap
over studies: mean silver delta and its 95 % interval.

Every setting must stay something the network saw in training: zoom (field of view = 150 mm / zoom) and the
slice span are only nudged around the training values, windows only made denser.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from knee.config import LABELS, ROOT_DIR
from knee.data import StudyDataset, assign_slots, series_table
from knee.model import KneeNet
from scripts.eval_oof import score, targets
from scripts.train import macro_auc


@torch.no_grad()
def predict(model, slots, cfg, cache_dir, windows, zoom, span, dtype, workers):
    ds = StudyDataset(slots, n_windows=windows, img_size=cfg["img_size"], span=span, zoom=zoom, cache_dir=cache_dir)
    out = []
    for batch in DataLoader(ds, batch_size=max(1, 48 // windows), num_workers=workers):
        x, pos, mask, seq = (batch[k].cuda() for k in ("x", "pos", "mask", "seq"))
        with torch.autocast("cuda", dtype=dtype):
            out.append(torch.sigmoid(model(x, pos, mask, seq).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--settings", nargs="+", help="subset of the settings below (default: all)")
    p.add_argument("--save", help="npz of every setting's predictions")
    p.add_argument("--boot", type=int, default=200)
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.loads(sorted(run.glob("config*.json"))[0].read_text())
    size = cfg.get("cache_size", 256)
    cache_dir = ROOT_DIR / "data" / ("cache" if size == 256 else f"cache{size}")
    oof = pd.read_csv(run / f"oof_fold{args.fold}.csv").set_index("StudyInstanceUID")
    if args.limit:
        oof = oof.iloc[: args.limit]
    slots = assign_slots(series_table("train")).set_index("StudyInstanceUID").reindex(oof.index).reset_index()
    model = KneeNet.from_config(cfg)
    model.load_state_dict(torch.load(run / "checkpoints" / f"fold{args.fold}.pt", map_location="cpu"))
    model = model.cuda().eval().to(memory_format=torch.channels_last)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    w, z, s = cfg["windows_eval"], cfg.get("zoom", 1.0), tuple(cfg.get("span", (0.08, 0.92)))
    settings = {
        "base": (w, z, s),
        "w24": (24, z, s),
        "zoom+8%": (24, z * 1.08, s),
        "zoom-7%": (24, z * 0.93, s),
        "span wide": (24, z, (s[0] - 0.04, s[1] + 0.04)),
        "span narrow": (24, z, (s[0] + 0.04, s[1] - 0.04)),
        "span wider": (24, z, (max(0.0, s[0] - 0.06), min(1.0, s[1] + 0.06))),
        "span full": (24, z, (0.0, 1.0)),
        "w32 wide": (32, z, (s[0] - 0.04, s[1] + 0.04)),
    }
    if args.settings:
        settings = {n: settings[n] for n in ["w24"] + [n for n in args.settings if n != "w24"]}
    t = targets("qwen36_27b_v1")
    preds = {}
    for name, (wi, zo, sp) in settings.items():
        preds[name] = pd.DataFrame(predict(model, slots, cfg, cache_dir, wi, zo, sp, dtype, args.workers),
                                   columns=LABELS, index=oof.index)
        sc = score(preds[name], t)
        print(f"{name:12s} windows {wi:2d} zoom {zo:.2f} span {sp[0]:.2f}-{sp[1]:.2f}: silver {sc['silver']:.4f} "
              f"gold {sc['gold']:.4f}", flush=True)
    if args.save:
        np.savez(args.save, index=np.asarray(oof.index), **{n: p.to_numpy() for n, p in preds.items()})
    sc = score(oof[LABELS], t)
    print(f"{'OOF file':12s} silver {sc['silver']:.4f} gold {sc['gold']:.4f} (training-time prediction, bf16)")
    names = [n for n in settings if n != "base"]
    variants = {n: preds[n] for n in settings if n != "w24"}
    for k in range(2, len(names) + 1):
        for combo in combinations(names, k):
            if k == len(names) or "w24" in combo:
                variants["mean " + "+".join(combo)] = sum(preds[n] for n in combo) / k
    y = t.loc[oof.index, LABELS].to_numpy()
    ref = preds["w24"].to_numpy()
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(args.boot)]
    print(f"\nsilver delta vs w24, paired bootstrap over {len(y)} studies ({args.boot} resamples):")
    for name, pr in variants.items():
        d = [macro_auc(y[b], pr.to_numpy()[b])[0] - macro_auc(y[b], ref[b])[0] for b in boots]
        full = macro_auc(y, pr.to_numpy())[0] - macro_auc(y, ref)[0]
        print(f"  {name:40s} {full:+.4f}  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
              f"P(>0) {np.mean(np.array(d) > 0):.2f}", flush=True)


if __name__ == "__main__":
    main()
