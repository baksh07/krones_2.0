"""
Krones — Multi-scale Test-Time Augmentation
--------------------------------------------
Heavier TTA than infer_tta.py. For each image:
  - 3 scales (576, 640, 704)
  - 4 rotations x 2 flips = 8 augs per scale
  - Total 24 forward passes per image
  - Logit averaging across all 24 (instead of probability averaging),
    then sigmoid once at the end. Better-calibrated than averaging probs.

Run order:
    python infer_tta_multiscale.py --root <DATASET> --ckpt <CKPT> --split val
        # finds best threshold on val
    python infer_tta_multiscale.py --root <DATASET> --ckpt <CKPT> --split test --threshold 0.48
        # uses chosen threshold on test
"""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from dataset import KronesDataset
from train import build_model


# Default scales relative to the training image size. Should match training
# distribution closely - too far off and the model's features degrade.
# 0.9x / 1.0x / 1.1x is standard.
DEFAULT_SCALES = [0.9, 1.0, 1.1]


def tta_forward_logits(model, imgs, autocast_dtype):
    """Run 8 TTA passes (4 rotations x 2 flips) and return averaged LOGITS.
    Logit averaging is statistically better than probability averaging because
    it preserves the model's confidence information through the average."""
    parts = []
    for k in range(4):
        x = torch.rot90(imgs, k, dims=[-2, -1])
        with autocast("cuda", dtype=autocast_dtype):
            logits = model(x).squeeze(1).float()
        parts.append(logits)
        xf = torch.flip(x, dims=[-1])
        with autocast("cuda", dtype=autocast_dtype):
            lf = model(xf).squeeze(1).float()
        parts.append(lf)
    return torch.stack(parts, dim=0).mean(dim=0)  # (B,) averaged logits


def multiscale_tta(model, imgs, base_size, scales, autocast_dtype):
    """For each scale: resize image, run 8-pass TTA, collect logits.
    Average logits across all (n_scales * 8) passes -> sigmoid once.
    Returns (B,) probabilities."""
    all_logits = []
    for scale in scales:
        target = int(round(base_size * scale))
        # round to multiple of 32 for clean ResNet downsampling
        target = max(32, (target // 32) * 32)
        if target == base_size:
            scaled = imgs
        else:
            scaled = F.interpolate(
                imgs, size=(target, target), mode="bilinear", align_corners=False
            )
        scale_logits = tta_forward_logits(model, scaled, autocast_dtype)  # (B,)
        all_logits.append(scale_logits)
    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    return torch.sigmoid(avg_logits)


@torch.no_grad()
def gather_multiscale(model, loader, device, base_size, scales, autocast_dtype):
    model.eval()
    all_probs, all_labels, all_classes = [], [], []
    n_total = len(loader)
    t0 = time.time()
    for batch_idx, (imgs, labels, cls_idx) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        probs = multiscale_tta(model, imgs, base_size, scales, autocast_dtype)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
        all_classes.append(cls_idx)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_total:
            elapsed = time.time() - t0
            eta = elapsed * (n_total - batch_idx - 1) / max(1, batch_idx + 1)
            print(
                f"  batch {batch_idx + 1}/{n_total}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s"
            )
    return (
        torch.cat(all_probs).numpy(),
        torch.cat(all_labels).long().numpy(),
        torch.cat(all_classes).numpy(),
    )


def find_best_threshold(probs, labels):
    thrs = np.linspace(0.05, 0.95, 181)
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

    scales = args.scales if args.scales else DEFAULT_SCALES
    print(
        f"multi-scale TTA: scales={scales}  (8 augs per scale = {8 * len(scales)} passes/image)"
    )

    ds = KronesDataset(args.root, args.split, img_size=img_size, splits_csv=splits_csv)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"{args.split}: {len(ds)} samples  -> {len(loader)} batches")

    model = build_model(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    autocast_dtype = torch.bfloat16

    print("\nrunning multi-scale TTA...")
    probs, labels, cls_idx = gather_multiscale(
        model, loader, device, img_size, scales, autocast_dtype
    )

    # ---- threshold ----
    if args.threshold is not None:
        thr = args.threshold
        print(f"using user-specified threshold: {thr}")
    else:
        thr, best_f1 = find_best_threshold(probs, labels)
        print(f"\nbest threshold (sweep): {thr:.3f}  F1 {best_f1:.4f}")

    preds = (probs >= thr).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    preds05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(labels, preds05, zero_division=0)

    print()
    print("=" * 70)
    print(f"MULTI-SCALE TTA on {args.split.upper()}")
    print("=" * 70)
    print(f"@ thr 0.50      F1 {f1_05:.4f}")
    print(f"@ thr {thr:.3f}    F1 {f1:.4f}   P {prec:.4f}   R {rec:.4f}")
    print(f"  confusion (rows=true, cols=pred):\n{cm}")

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
    print("\nper-class breakdown after multi-scale TTA:")
    print(per_class.to_string(index=False))

    out = Path(args.ckpt).parent / f"tta_ms_{args.split}.csv"
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
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="lower than infer_tta.py because multi-scale uses more VRAM",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=None,
        help="scale multipliers relative to training img_size. "
        "Default: 0.9 1.0 1.1 (e.g. 576/640/704 for img_size=640)",
    )
    args = p.parse_args()
    main(args)
