"""
Krones — Test-Time Augmentation + threshold tuning
---------------------------------------------------
For a trained classifier checkpoint, runs TTA on the val split to find
the optimal probability threshold. Then optionally evaluates on test split
with the chosen threshold.

TTA augmentations: 4 rotations (0, 90, 180, 270 degrees) + horizontal flip
= 8 forward passes per image. Probabilities are averaged before thresholding.

Run order:
    python infer_tta.py --root <DATASET> --ckpt runs/baseline_resnet50/best.pt
                        --split val      # finds best threshold on val
    python infer_tta.py --root <DATASET> --ckpt runs/baseline_resnet50/best.pt
                        --split test --threshold 0.55
                                         # uses chosen threshold for test
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from dataset import KronesDataset
from train import build_model


def tta_forward(model, imgs, autocast_dtype):
    """Run 8 TTA passes (4 rotations x 2 flips) and average probabilities."""
    probs_list = []
    # 4 rotations on both unflipped and flipped views
    for k in range(4):  # k = number of 90-deg rotations
        x = torch.rot90(imgs, k, dims=[-2, -1])
        with autocast("cuda", dtype=autocast_dtype):
            p = torch.sigmoid(model(x).squeeze(1).float())
        probs_list.append(p)
        # horizontally flipped version
        xf = torch.flip(x, dims=[-1])
        with autocast("cuda", dtype=autocast_dtype):
            pf = torch.sigmoid(model(xf).squeeze(1).float())
        probs_list.append(pf)
    return torch.stack(probs_list, dim=0).mean(dim=0)


@torch.no_grad()
def gather_tta(model, loader, device, autocast_dtype):
    model.eval()
    all_probs, all_labels, all_classes = [], [], []
    for imgs, labels, cls_idx in loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = tta_forward(model, imgs, autocast_dtype)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
        all_classes.append(cls_idx)
    return (
        torch.cat(all_probs).numpy(),
        torch.cat(all_labels).long().numpy(),
        torch.cat(all_classes).numpy(),
    )


def find_best_threshold(probs, labels):
    thrs = np.linspace(0.05, 0.95, 181)  # 0.005 step, finer than before
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

    print(
        f"loaded ckpt epoch {ckpt['epoch']}  trained val F1 {ckpt.get('val_f1', float('nan')):.4f}"
    )
    print(f"img_size={img_size}  splits_csv={splits_csv}")

    ds = KronesDataset(args.root, args.split, img_size=img_size, splits_csv=splits_csv)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"{args.split}: {len(ds)} samples")

    model = build_model(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    autocast_dtype = torch.bfloat16

    print("running TTA (8 passes per image: 4 rotations x 2 flips)...")
    probs, labels, cls_idx = gather_tta(model, loader, device, autocast_dtype)

    # ---- threshold logic ----
    if args.threshold is not None:
        thr = args.threshold
        print(f"using user-specified threshold: {thr}")
    else:
        thr, best_f1 = find_best_threshold(probs, labels)
        print(f"best threshold (sweep): {thr:.3f}  F1 {best_f1:.4f}")

    preds = (probs >= thr).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # also report @ 0.5 for comparison
    preds05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(labels, preds05, zero_division=0)

    print()
    print("=" * 70)
    print(f"TTA on {args.split.upper()}")
    print("=" * 70)
    print(f"@ thr 0.50  F1 {f1_05:.4f}     (no TTA reference: see eval.py output)")
    print(f"@ thr {thr:.3f}  F1 {f1:.4f}   P {prec:.4f}   R {rec:.4f}")
    print(f"  confusion (rows=true, cols=pred):\n{cm}")

    # per-class breakdown
    rows = []
    for i, c in enumerate(classes):
        mask = cls_idx == i
        if mask.sum() == 0:
            continue
        sub_lbl = labels[mask]
        sub_pred = preds[mask]
        n = mask.sum()
        n_correct = (sub_lbl == sub_pred).sum()
        rows.append(
            {
                "dominant": c,
                "n": int(n),
                "n_misclass": int(n - n_correct),
                "accuracy": float(n_correct / n),
            }
        )
    per_class = pd.DataFrame(rows).sort_values("n_misclass", ascending=False)
    print("\nper-class breakdown after TTA:")
    print(per_class.to_string(index=False))

    # save
    out = Path(args.ckpt).parent / f"tta_{args.split}.csv"
    pd.DataFrame(
        {
            "label": labels,
            "pred": preds,
            "prob": probs,
            "class_idx": cls_idx,
            "class": [classes[i] if i >= 0 else "" for i in cls_idx],
        }
    ).to_csv(out, index=False)
    print(f"\nper-image probs+preds written to: {out}")
    print(f"\nUSE THIS THRESHOLD FOR KAGGLE SUBMISSION: {thr:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="if not given, sweep to find best on this split",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    main(args)
