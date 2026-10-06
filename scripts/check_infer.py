"""Replay the Kaggle inference path on one validation fold and compare it with the run's OOF.

    python -m scripts.check_infer runs/<run> --fold 0 [--dtype fp16|bf16|fp32] [--windows 12] [--limit 200]

The OOF predictions were written by scripts/train.py (bf16 autocast on 3090s); the Kaggle notebook runs
knee/infer.py (fp16 autocast on T4s, no bf16). Same slots, same cache, same fold network: any gap in
AUC or correlation comes from the precision or from the inference code itself.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from knee.config import DATA_DIR, LABELS, ROOT_DIR
from knee.data import StudyDataset, assign_slots, series_table
from knee.model import KneeNet
from scripts.eval_oof import score, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--windows", type=int, default=0, help="0 = the run's windows_eval, as its OOF")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.loads(sorted(run.glob("config*.json"))[0].read_text())
    windows = args.windows or cfg["windows_eval"]
    size = cfg.get("cache_size", 256)
    cache_dir = ROOT_DIR / "data" / ("cache" if size == 256 else f"cache{size}")

    oof = pd.read_csv(run / f"oof_fold{args.fold}.csv").set_index("StudyInstanceUID")
    slots = assign_slots(series_table("train")).set_index("StudyInstanceUID").reindex(oof.index).reset_index()
    if args.limit:
        slots, oof = slots.iloc[: args.limit], oof.iloc[: args.limit]

    device = torch.device("cuda")
    model = KneeNet(cfg["backbone"], pretrained=False, dim=cfg["dim"], depth=cfg["depth"], drop_path=0,
                    pool=cfg.get("pool", "avg"), seq_emb=cfg.get("seq_emb", False))
    model.load_state_dict(torch.load(run / "checkpoints" / f"fold{args.fold}.pt", map_location="cpu"))
    model = model.to(device).eval().to(memory_format=torch.channels_last)

    ds = StudyDataset(slots, n_windows=windows, img_size=cfg["img_size"], span=tuple(cfg.get("span", (0.08, 0.92))),
                      zoom=cfg.get("zoom", 1.0), cache_dir=cache_dir)
    dl = DataLoader(ds, batch_size=max(1, 48 // windows), num_workers=args.workers)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[args.dtype]
    out, nonfinite = [], 0
    with torch.no_grad():
        for batch in dl:
            x, pos, mask, seq = (batch[k].to(device) for k in ("x", "pos", "mask", "seq"))
            with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                logits = model(x, pos, mask, seq).float()
            nonfinite += int((~torch.isfinite(logits)).sum())
            out.append(torch.sigmoid(logits).cpu().numpy())
    pred = pd.DataFrame(np.concatenate(out), columns=LABELS, index=oof.index)

    t = targets("qwen36_27b_v1")
    a, b = score(oof[LABELS], t), score(pred, t)
    corr = np.mean([np.corrcoef(oof[c], pred[c])[0, 1] for c in LABELS])
    print(f"{run.name} fold {args.fold}, {len(pred)} studies, {windows} windows, {args.dtype}, non-finite logits: {nonfinite}")
    print(f"OOF (train.py)      silver {a['silver']:.4f}  gold {a['gold']:.4f} (gold n {a['n_gold']})")
    print(f"replay (infer path) silver {b['silver']:.4f}  gold {b['gold']:.4f}")
    print(f"mean per-label correlation {corr:.4f}, max |diff| {np.abs(oof[LABELS].to_numpy() - pred.to_numpy()).max():.4f}")
    print("silver per label, OOF vs replay:")
    print(pd.DataFrame({"oof": a["silver_per"], "replay": b["silver_per"]}, index=[l[:10] for l in LABELS]).T.round(3).to_string())


if __name__ == "__main__":
    main()
