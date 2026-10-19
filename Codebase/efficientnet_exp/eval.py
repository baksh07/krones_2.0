"""
Krones — Evaluation script (multi-architecture)
------------------------------------------------
Loads a trained checkpoint and reports binary metrics + per-defect-class
breakdown. Auto-detects model architecture from checkpoint args, so works
with any backbone trained via the new train.py (resnet50, EffNet, Swin, etc.)

Usage:
    python eval.py --root /path --ckpt runs/baseline_resnet50/best.pt --split val
    python eval.py --root /path --ckpt runs/effnet_b4_640/best.pt --split test
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

from dataset import KronesDataset
from train import build_model


@torch.no_grad()
def gather_predictions(model, loader, device, autocast_dtype):
    model.eval()
    all_logits, all_labels, all_classes = [], [], []
    for imgs, labels, cls_idx in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", dtype=autocast_dtype):
            logits = model(imgs).squeeze(1)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
        all_classes.append(cls_idx)
    return (torch.cat(all_logits), torch.cat(all_labels).long(), torch.cat(all_classes))


def find_best_threshold(probs, labels):
    thrs = np.linspace(0.05, 0.95, 91)
    best = (0.5, 0.0)
    for t in thrs:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt["args"]["img_size"]
    splits_csv = ckpt["args"].get("splits_csv", "splits_aug.csv")
    # auto-detect model architecture from checkpoint args (defaults to resnet50)
    model_name = ckpt["args"].get("model_name", "resnet50")

    print(
        f"loaded checkpoint epoch {ckpt['epoch']}  "
        f"trained val F1 {ckpt.get('val_f1', float('nan')):.4f}"
    )
    print(f"  model:      {model_name}")
    print(f"  img_size:   {img_size}")
    print(f"  splits_csv: {splits_csv}")

    ds = KronesDataset(args.root, args.split, img_size=img_size, splits_csv=splits_csv)
    print(f"  {args.split} samples: {len(ds)}")

    model = build_model(num_classes=1, pretrained=False, model_name=model_name).to(
        device
    )
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    autocast_dtype = torch.bfloat16

    logits, labels, cls_idx = gather_predictions(model, loader, device, autocast_dtype)
    probs = torch.sigmoid(logits).numpy()
    labels = labels.numpy()
    cls_idx = cls_idx.numpy()

    preds_05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(labels, preds_05, zero_division=0)
    prec_05 = precision_score(labels, preds_05, zero_division=0)
    rec_05 = recall_score(labels, preds_05, zero_division=0)
    cm_05 = confusion_matrix(labels, preds_05, labels=[0, 1])

    best_thr, best_f1 = find_best_threshold(probs, labels)
    preds_best = (probs >= best_thr).astype(int)
    prec_best = precision_score(labels, preds_best, zero_division=0)
    rec_best = recall_score(labels, preds_best, zero_division=0)
    cm_best = confusion_matrix(labels, preds_best, labels=[0, 1])

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    rows = []
    for i, c in enumerate(classes):
        mask = cls_idx == i
        if mask.sum() == 0:
            continue
        sub_lbl = labels[mask]
        sub_pred = preds_best[mask]
        n = mask.sum()
        n_correct = (sub_lbl == sub_pred).sum()
        n_faulty = sub_lbl.sum()
        rows.append(
            {
                "dominant": c,
                "n": int(n),
                "n_faulty": int(n_faulty),
                "n_good": int(n - n_faulty),
                "accuracy": float(n_correct / n),
                "n_misclass": int(n - n_correct),
            }
        )
    per_class = pd.DataFrame(rows).sort_values("n_misclass", ascending=False)

    print("\n" + "=" * 70)
    print(f"EVAL on {args.split.upper()}")
    print("=" * 70)
    print(f"@ threshold 0.5    F1 {f1_05:.4f}   P {prec_05:.4f}   R {rec_05:.4f}")
    print(f"  confusion:\n{cm_05}")
    print()
    print(
        f"@ best threshold {best_thr:.2f}   F1 {best_f1:.4f}   "
        f"P {prec_best:.4f}   R {rec_best:.4f}"
    )
    print(f"  confusion:\n{cm_best}")
    print()
    print(f"ROC-AUC: {auc:.4f}")
    print()
    print("-" * 70)
    print("per-defect-class breakdown @ best threshold (sorted by n_misclass)")
    print("-" * 70)
    print(per_class.to_string(index=False))

    out = Path(args.ckpt).parent / f"eval_{args.split}.csv"
    pd.DataFrame(
        {
            "label": labels,
            "pred05": preds_05,
            "predbest": preds_best,
            "prob": probs,
            "class_idx": cls_idx,
            "class": [classes[i] if i >= 0 else "" for i in cls_idx],
        }
    ).to_csv(out, index=False)
    print(f"\nper-image predictions written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    main(args)
