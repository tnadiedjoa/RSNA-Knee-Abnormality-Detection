"""Train the 2.5D model with cross-validation. One folder per run in runs/<date>_<name>/:
config.json, per-fold metrics, out-of-fold predictions (oof.csv), checkpoints (not versioned).

    python -m scripts.train --name r34_base --labels qwen36_v1 --folds 0 1 2 3 4
"""

import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader

from knee.config import CACHE_DIR, CACHE_SIZE, DATA_DIR, LABELS, RUNS_DIR
from knee.data import StudyDataset, assign_slots, series_table
from knee.labels import load_soft
from knee.model import KneeNet


def build_targets(studies: pd.DataFrame, args):
    """Soft report labels, overridden by the gold labels where they exist; per-cell loss weights."""
    soft = load_soft(args.labels).set_index("StudyInstanceUID").reindex(studies["StudyInstanceUID"])
    y = soft[LABELS].to_numpy(np.float32)
    mentioned = soft[[f"{l}__mentioned" for l in LABELS]].to_numpy(bool)
    w = np.where(mentioned, 1.0, args.silence_weight).astype(np.float32)
    if args.mix_public:  # average with public report-label tables (CC0, data/public_labels/), missing cells keep ours
        from scripts.eval_labels import public_sets
        pub = public_sets()
        tables = [pub[n].set_index("StudyInstanceUID").reindex(studies["StudyInstanceUID"])[LABELS].to_numpy(np.float32)
                  for n in args.mix_public]
        y = ((y + sum(np.where(np.isnan(t), y, t) for t in tables)) / (1 + len(tables))).astype(np.float32)
    if args.bootstrap_run:  # soft bootstrapping: pull report labels towards an earlier run's out-of-fold predictions
        oof = pd.concat([pd.read_csv(f) for f in sorted((RUNS_DIR / args.bootstrap_run).glob("oof_fold*.csv"))])
        oof = oof.set_index("StudyInstanceUID").reindex(studies["StudyInstanceUID"])[LABELS].to_numpy(np.float32)
        alpha = np.array([args.bootstrap_alpha_labels.get(l, args.bootstrap_alpha) for l in LABELS], np.float32)
        y = np.where(np.isnan(oof), y, (1 - alpha) * y + alpha * oof).astype(np.float32)
    gold = studies[LABELS].to_numpy(np.float32)
    is_gold = ~np.isnan(gold).any(1)
    y[is_gold], w[is_gold] = gold[is_gold], args.gold_weight
    # confident report labels for evaluation: absent -> 0, moderate/severe -> 1, else excluded
    grades = pd.read_parquet(DATA_DIR / "labels" / f"{args.labels}.parquet").set_index("StudyInstanceUID")
    g = grades.reindex(studies["StudyInstanceUID"])[LABELS]
    silver = np.where(g.isin(["moderate", "severe"]), 1.0, np.where(g == "absent", 0.0, np.nan)).astype(np.float32)
    silver[is_gold] = gold[is_gold]
    return y, w, silver, is_gold


def _label_values(text: str) -> dict:
    """"ACL=0.2,Medial_Meniscus=0.5" -> {"ACL": 0.2, "Medial Meniscus": 0.5}: knee.config.LABELS names, "_" for
    spaces (slurm/multi_gpu.sbatch splits its runs-file lines on spaces)."""
    out = {k.strip().replace("_", " "): float(v) for k, v in (item.split("=") for item in text.split(",") if item.strip())}
    unknown = set(out) - set(LABELS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown findings: {sorted(unknown)}")
    return out


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list]:
    per = []
    for k in range(y_true.shape[1]):
        ok = ~np.isnan(y_true[:, k])
        t = y_true[ok, k]
        per.append(roc_auc_score(t, y_pred[ok, k]) if 0 < t.sum() < len(t) else float("nan"))
    return float(np.nanmean(per)), per


def lr_lambda(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


AMP_DTYPE = torch.bfloat16  # set in main(): float16 + grad scaling on GPUs without bf16 (P100)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            logits = model(batch["x"].to(device, non_blocking=True), batch["pos"].to(device), batch["mask"].to(device),
                           batch["seq"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(fold, studies, slots, y, w, silver, is_gold, folds, args, run_dir, device):
    tr, va = np.where(folds != fold)[0], np.where(folds == fold)[0]
    if args.full_fit:  # every study in training; "fold" only seeds the network and picks the (in-sample) monitor set
        tr = np.arange(len(folds))
    raw = Path(args.raw_dir) if args.raw_dir else None
    span = tuple(args.span)
    ds_tr = StudyDataset(slots.iloc[tr], y[tr], w[tr], n_windows=args.windows, img_size=args.img_size, train=True,
                         aug_strength=args.aug, span=span, zoom=args.zoom, seed=args.seed + fold, raw_dir=raw)
    ds_va = StudyDataset(slots.iloc[va], n_windows=args.windows_eval, img_size=args.img_size, span=span, zoom=args.zoom,
                         raw_dir=raw)
    dl_tr = DataLoader(ds_tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                       persistent_workers=True)
    bs_va = args.bs if args.windows_eval <= 12 else max(1, args.bs * 12 // args.windows_eval)  # denser eval: smaller batches
    dl_va = DataLoader(ds_va, bs_va, num_workers=args.workers, pin_memory=True)

    torch.manual_seed(args.seed + fold)
    model = KneeNet(args.backbone, dim=args.dim, depth=args.depth, drop_path=args.drop_path, pool=args.pool,
                    seq_emb=args.seq_emb).to(device)
    model = model.to(memory_format=torch.channels_last)
    if args.grad_ckpt:  # recompute backbone activations in the backward pass: bigger backbones fit a 16 GB P100
        model.backbone.set_grad_checkpointing(True)
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    others = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": others, "lr": args.lr_head}],
                            weight_decay=args.wd)
    total = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, int(0.05 * total)))

    scaler = torch.amp.GradScaler(enabled=AMP_DTYPE == torch.float16)
    # weight EMA: passive (training is unchanged), evaluated and saved next to the raw weights
    ema = ModelEmaV3(model, decay=args.ema, use_warmup=True) if args.ema else None
    history = []
    for epoch in range(args.epochs):
        model.train()
        ds_tr.epoch = epoch
        t, losses = time.time(), []
        for step, batch in enumerate(dl_tr):
            if args.debug_steps and step >= args.debug_steps:
                break
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                logits = model(batch["x"].to(device, non_blocking=True), batch["pos"].to(device), batch["mask"].to(device),
                           batch["seq"].to(device))
            loss = F.binary_cross_entropy_with_logits(logits.float(), batch["y"].to(device), reduction="none")
            loss = (loss * batch["w"].to(device)).sum() / batch["w"].sum().to(device)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model, step=epoch * len(dl_tr) + step)
            losses.append(loss.item())
        pred = predict(model, dl_va, device)
        silver_auc, _ = macro_auc(silver[va], pred)
        gold_va = is_gold[va]
        gold_auc = macro_auc(silver[va][gold_va], pred[gold_va])[0] if gold_va.sum() >= 5 else float("nan")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "silver_auc": silver_auc,
                        "gold_auc_fold": gold_auc, "minutes": (time.time() - t) / 60})
        ema_txt = ""
        if ema is not None:
            pred_ema = predict(ema.module, dl_va, device)
            history[-1]["silver_auc_ema"] = macro_auc(silver[va], pred_ema)[0]
            ema_txt = f" ema {history[-1]['silver_auc_ema']:.4f}"
        print(f"fold {fold} epoch {epoch + 1}: loss {np.mean(losses):.4f} silver {silver_auc:.4f}{ema_txt} "
              f"gold(fold) {gold_auc:.4f} [{(time.time() - t) / 60:.1f} min]", flush=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), run_dir / "checkpoints" / f"fold{fold}.pt")  # knee.infer averages every fold*.pt
    (run_dir / f"{'full' if args.full_fit else 'fold'}{fold}_history.json").write_text(json.dumps(history, indent=1))
    if ema is not None:  # kept apart: package_kaggle and load_oof only pick fold*.pt / oof_fold*.csv
        torch.save(ema.module.state_dict(), run_dir / "checkpoints" / f"ema_fold{fold}.pt")
        oof = pd.DataFrame(pred_ema, columns=LABELS)
        oof.insert(0, "StudyInstanceUID", studies["StudyInstanceUID"].to_numpy()[va])
        oof.insert(1, "fold", fold)
        oof.to_csv(run_dir / f"oof_ema_fold{fold}.csv", index=False)
    return va, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--labels", default="qwen36_27b_v1")
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--windows-eval", type=int, default=12)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--pool", default="avg", choices=["avg", "avgmax"], help="pooling inside each window (knee/model.py)")
    p.add_argument("--seq-emb", action="store_true", help="add a learned T1/PD/T2/... embedding to each slot's tokens (knee/model.py)")
    p.add_argument("--aug", type=float, default=1.0)
    p.add_argument("--silence-weight", type=float, default=0.5)
    p.add_argument("--gold-weight", type=float, default=3.0)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default=None, help="resume/extend an existing run folder")
    p.add_argument("--span", type=float, nargs=2, default=[0.08, 0.92], help="slice range covered by the windows")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--zoom", type=float, default=1.0, help="field of view = 150 mm / zoom (cache crop is 150 mm)")
    p.add_argument("--exclude-folds", type=int, nargs="*", default=[])
    p.add_argument("--full-fit", action="store_true",
                   help="train every network on all studies (one per --folds entry, seeded by it); no OOF is written")
    p.add_argument("--bootstrap-run", default=None, help="run folder whose OOF predictions are mixed into the labels")
    p.add_argument("--bootstrap-alpha", type=float, default=0.5)
    p.add_argument("--bootstrap-alpha-labels", type=_label_values, default={},
                   help='per-finding teacher weight overriding --bootstrap-alpha, e.g. "ACL=0.2,MCL=0.2"')
    p.add_argument("--mix-public", nargs="*", default=[], help="public label tables averaged with ours (pilkwang_v2 steven_v2 lixin_sol56)")
    p.add_argument("--raw-dir", default=None, help="unpacked cache on local disk (scripts/unpack_cache.py)")
    p.add_argument("--debug-steps", type=int, default=0, help="cap training steps per epoch (smoke tests)")
    p.add_argument("--ema", type=float, default=0.0, help="weight EMA decay (e.g. 0.998), 0 = off")
    args = p.parse_args()

    run_dir = RUNS_DIR / (args.run_dir or f"{datetime.now():%Y-%m-%d_%H%M%S}_{args.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    config = vars(args) | {"commit": commit, "gpu": torch.cuda.get_device_name(0),
                           "cache_dir": str(CACHE_DIR), "cache_size": CACHE_SIZE}
    (run_dir / f"config_folds{''.join(map(str, args.folds))}.json").write_text(json.dumps(config, indent=1))

    train = pd.read_csv(DATA_DIR / "train.csv")
    folds_df = pd.read_csv(DATA_DIR / "folds.csv")
    studies = train.merge(folds_df[["StudyInstanceUID", "fold"]], on="StudyInstanceUID")
    labelled = load_soft(args.labels).dropna(subset=LABELS)["StudyInstanceUID"]
    studies = studies[studies["StudyInstanceUID"].isin(labelled)].reset_index(drop=True)  # e.g. LLM parse errors
    if args.exclude_folds:  # nested CV: these studies are neither trained on nor predicted
        studies = studies[~studies["fold"].isin(args.exclude_folds)].reset_index(drop=True)
    print(f"{len(studies)} labelled studies", flush=True)
    slots = assign_slots(series_table("train")).set_index("StudyInstanceUID").reindex(studies["StudyInstanceUID"])
    slots = slots.reset_index()
    y, w, silver, is_gold = build_targets(studies, args)
    folds = studies["fold"].to_numpy()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    global AMP_DTYPE
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16

    for fold in args.folds:
        va, pred = train_fold(fold, studies, slots, y, w, silver, is_gold, folds, args, run_dir, device)
        if args.full_fit:  # in-sample predictions: never written as OOF (eval_oof, teachers and blends read those)
            continue
        oof = pd.DataFrame(pred, columns=LABELS)
        oof.insert(0, "StudyInstanceUID", studies["StudyInstanceUID"].to_numpy()[va])
        oof.insert(1, "fold", fold)
        oof.to_csv(run_dir / f"oof_fold{fold}.csv", index=False)

    if args.full_fit:
        return
    # summary over every fold present in the run folder
    oof = pd.concat([pd.read_csv(f) for f in sorted(run_dir.glob("oof_fold*.csv"))])
    idx = studies.reset_index().set_index("StudyInstanceUID").loc[oof["StudyInstanceUID"], "index"].to_numpy()
    pred = oof[LABELS].to_numpy()
    silver_auc, silver_per = macro_auc(silver[idx], pred)
    g = is_gold[idx]
    gold_auc, gold_per = macro_auc(silver[idx][g], pred[g])
    summary = {"folds": sorted(oof["fold"].unique().tolist()), "silver_auc": silver_auc, "gold_auc": gold_auc,
               "n_gold": int(g.sum()), "silver_per_label": dict(zip(LABELS, silver_per)),
               "gold_per_label": dict(zip(LABELS, gold_per))}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
