"""
Krones — Eval classifier on uncropped full images (no YOLO ROI)
----------------------------------------------------------------
Hypothesis test: does ROI cropping actually help, or could the model do as
well or better on full uncropped images?
 
Loads your trained classifier checkpoint, but instead of using the cropped
dataset (KronesDataset), it runs on the original train images directly,
filtered to the val/test split. Just resizes to img_size and predicts.
 
Usage:
    python eval_no_crop.py \\
        --root "/path/to/dataset" \\
        --ckpt runs/resnet50_640/best.pt \\
        --split val
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              confusion_matrix, roc_auc_score)
from tqdm import tqdm
 
 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
 
 
class FullImageDataset(Dataset):
    """Like KronesDataset but loads the original train_images/<file>.png
    instead of the cropped ROI from audit_output/crops/."""
    def __init__(self, splits_df, img_root, img_size):
        self.df = splits_df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, i):
        row = self.df.iloc[i]
        # the file_name in splits.csv matches both crop name AND original image name
        # because crops were generated 1:1 from the original images.
        # original images are at <root>/train_images/<file_name>
        img_path = self.img_root / row["file_name"]
        img = Image.open(img_path).convert("RGB")
        # support both 'label' and 'label_recomputed' column naming
        label = int(row.get("label_recomputed", row.get("label")))
        return self.tfm(img), label, int(row["dominant_idx"])
 
 
def build_classifier(num_classes=1):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
 
 
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
 
    # load the splits we trained on
    splits_csv = Path(args.root) / "audit_output" / "splits.csv"
    print(f"reading splits: {splits_csv}")
    splits = pd.read_csv(splits_csv)
    sub = splits[splits["split"] == args.split].reset_index(drop=True)
    print(f"  {args.split} split: {len(sub)} samples")
    print(f"  columns: {list(splits.columns)}")
 
    # need a class index column for per-class breakdown. some pipelines
    # save it as 'dominant_idx', some as 'class_idx'. detect.
    if "dominant_idx" in splits.columns:
        pass
    elif "class_idx" in splits.columns:
        sub["dominant_idx"] = sub["class_idx"]
    else:
        # fall back: compute on the fly from a class list
        unique_classes = sorted(splits["dominant"].unique())
        cls_to_idx = {c: i for i, c in enumerate(unique_classes)}
        sub["dominant_idx"] = sub["dominant"].map(cls_to_idx)
 
    # original full images live at <root>/train_images
    img_root = Path(args.root) / "train_images"
    if not img_root.exists():
        # try other common names
        for alt in ["images", "train", "raw_images"]:
            if (Path(args.root) / alt).exists():
                img_root = Path(args.root) / alt
                break
    print(f"image root: {img_root}")
    assert img_root.exists(), f"can't find original image directory at {img_root}"
 
    # load classifier
    print(f"loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt["args"]["img_size"]
    print(f"  trained val F1: {ckpt.get('val_f1', float('nan')):.4f}")
    print(f"  img_size: {img_size}")
 
    model = build_classifier(num_classes=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
 
    ds = FullImageDataset(sub, img_root, img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
 
    autocast_dtype = torch.bfloat16
    print(f"\nrunning inference on full uncropped images...")
    all_logits, all_labels, all_classes = [], [], []
    with torch.no_grad():
        for imgs, labels, cls_idx in tqdm(loader):
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", dtype=autocast_dtype):
                logits = model(imgs).squeeze(1).float()
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_classes.append(cls_idx)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).long().numpy()
    cls_idx = torch.cat(all_classes).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
 
    # threshold sweep
    thrs = np.linspace(0.05, 0.95, 91)
    best = (0.5, 0.0)
    for t in thrs:
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best[1]:
            best = (float(t), float(f1))
    best_thr, best_f1 = best
 
    preds_05 = (probs >= 0.5).astype(int)
    preds_best = (probs >= best_thr).astype(int)
 
    f1_05 = f1_score(labels, preds_05, zero_division=0)
    p_05 = precision_score(labels, preds_05, zero_division=0)
    r_05 = recall_score(labels, preds_05, zero_division=0)
    cm_05 = confusion_matrix(labels, preds_05, labels=[0, 1])
 
    f1_b = best_f1
    p_b = precision_score(labels, preds_best, zero_division=0)
    r_b = recall_score(labels, preds_best, zero_division=0)
    cm_b = confusion_matrix(labels, preds_best, labels=[0, 1])
 
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
 
    print()
    print("=" * 70)
    print(f"NO-CROP EVAL on {args.split.upper()} (full uncropped images)")
    print("=" * 70)
    print(f"@ threshold 0.5    F1 {f1_05:.4f}   P {p_05:.4f}   R {r_05:.4f}")
    print(f"  confusion:\n{cm_05}")
    print()
    print(f"@ best threshold {best_thr:.2f}  F1 {f1_b:.4f}   P {p_b:.4f}   R {r_b:.4f}")
    print(f"  confusion:\n{cm_b}")
    print(f"\nROC-AUC: {auc:.4f}")
 
    # per-class
    print("\nper-defect-class breakdown @ best threshold:")
    rows = []
    for i, c in enumerate(classes):
        mask = (cls_idx == i)
        if mask.sum() == 0: continue
        sub_lbl = labels[mask]; sub_pred = preds_best[mask]
        n = mask.sum(); n_correct = (sub_lbl == sub_pred).sum()
        rows.append({"dominant": c, "n": int(n),
                     "accuracy": float(n_correct / n),
                     "n_misclass": int(n - n_correct)})
    df = pd.DataFrame(rows).sort_values("n_misclass", ascending=False)
    print(df.to_string(index=False))
 
    print()
    print("=" * 70)
    print("COMPARE TO YOUR CROPPED-INPUT BASELINE:")
    print("=" * 70)
    print(f"Cropped (with ROI):  test F1 0.9643 (best thr) - your prior result")
    print(f"Uncropped (this run): F1 {best_f1:.4f}")
    diff = best_f1 - 0.9643
    print(f"Difference:           {diff:+.4f}")
    if diff > 0.001:
        print("=> Uncropped does BETTER. Cropping is hurting.")
    elif diff < -0.005:
        print("=> Cropping helps significantly. ROI is doing real work.")
    else:
        print("=> Roughly tied. Cropping is neutral / not the bottleneck.")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    main(args)