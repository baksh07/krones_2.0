"""
build_area_targets.py
---------------------
For the multi-task ResNet experiment. Computes per-image, per-conditional-class
SUMMED polygon areas from train_annotations.json (COCO format), and joins them
with your existing train_clean.csv / splits CSV.

Output: a CSV with one row per training image, containing:
    file_name, image_id, label,
    area_scuffing, area_air_bubble, area_contamination_light,
    area_chip, area_scuffing_heavy, area_glass_imperfection

Areas are in raw pixel-squared units (matching the threshold rules from your
labeling spec). Areas of 0 mean the class is not annotated in that image.

We confirmed via check_area_rule.py that the SUM rule reconstructs labels
with 100.000% accuracy across all 35,342 training images.

Usage:
    python build_area_targets.py \
        --annotations /path/to/train_annotations.json \
        --train_csv   /path/to/train.csv \
        --out_csv     ./area_targets.csv

Optional:
    --splits_csv  /path/to/splits_aug.csv
        If provided, will join split assignment too (train/val/test).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


# Conditional class IDs and thresholds (verified against the labeling rule)
CONDITIONAL = {
    1: ("air_bubble", 500),
    3: ("chip", 200),
    6: ("contamination_light", 180),
    13: ("glass_imperfection", 100),
    23: ("scuffing", 75000),
    24: ("scuffing_heavy", 1200),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--annotations",
        required=True,
        help="Path to train_annotations.json (COCO format)",
    )
    p.add_argument(
        "--train_csv",
        required=True,
        help="Path to train.csv with columns: image_id, target",
    )
    p.add_argument(
        "--out_csv", required=True, help="Output path for the area-targets CSV"
    )
    p.add_argument(
        "--splits_csv",
        default=None,
        help="Optional: splits CSV to join split assignments",
    )
    args = p.parse_args()

    # ---- Load annotations ----
    print(f"[1/4] Loading {args.annotations} ...")
    with open(args.annotations) as f:
        coco = json.load(f)
    print(f"      {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    fname_to_cocoid = {img["file_name"]: img["id"] for img in coco["images"]}

    # Group annotations by image, summing areas per conditional class
    per_image_area = defaultdict(lambda: defaultdict(float))
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid in CONDITIONAL:
            per_image_area[ann["image_id"]][cid] += ann["area"]

    # ---- Load labels ----
    print(f"[2/4] Loading {args.train_csv} ...")
    df = pd.read_csv(args.train_csv)
    df["file_name"] = df["image_id"].astype(str)
    if not df["file_name"].iloc[0].endswith(".png"):
        df["file_name"] = df["file_name"] + ".png"
    print(f"      {len(df)} rows")

    # ---- Build output rows ----
    print("[3/4] Computing per-image area targets ...")
    out_rows = []
    missing = 0
    for _, row in df.iterrows():
        fname = row["file_name"]
        label = int(row["target"])
        cocoid = fname_to_cocoid.get(fname)
        if cocoid is None:
            missing += 1
            continue

        areas = per_image_area.get(cocoid, {})
        out = {
            "file_name": fname,
            "image_id": cocoid,
            "label": label,
        }
        for cid, (name, _thresh) in CONDITIONAL.items():
            out[f"area_{name}"] = float(areas.get(cid, 0.0))
        out_rows.append(out)

    if missing:
        print(f"      WARNING: {missing} rows had no matching image in JSON")

    out_df = pd.DataFrame(out_rows)

    # ---- Optional: join splits ----
    if args.splits_csv:
        print(f"      Joining splits from {args.splits_csv} ...")
        splits = pd.read_csv(args.splits_csv)
        # Splits CSVs vary; try common column names
        if "file_name" in splits.columns:
            key = "file_name"
        elif "image_id" in splits.columns:
            key = "image_id"
        else:
            raise ValueError(
                f"splits CSV needs file_name or image_id column, "
                f"got: {list(splits.columns)}"
            )
        if "split" not in splits.columns:
            raise ValueError(
                f"splits CSV needs 'split' column, got: {list(splits.columns)}"
            )
        out_df = out_df.merge(
            splits[[key, "split"]], left_on="file_name", right_on=key, how="left"
        )
        if key != "file_name":
            out_df = out_df.drop(columns=[key])
        n_unmatched = out_df["split"].isna().sum()
        if n_unmatched:
            print(f"      WARNING: {n_unmatched} rows have no split assignment")

    # ---- Write + summary ----
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[4/4] Wrote {out_path}  ({len(out_df)} rows)")

    # Summary statistics per conditional class
    print()
    print("=" * 75)
    print("Per-class area stats across all training images:")
    print("=" * 75)
    print(
        f"{'class':25s} {'thresh':>8} {'n_pos':>8} {'min>0':>10} "
        f"{'median>0':>10} {'max':>12}"
    )
    print("-" * 75)
    for cid, (name, thresh) in CONDITIONAL.items():
        col = f"area_{name}"
        positives = out_df[out_df[col] > 0][col]
        n_pos = len(positives)
        if n_pos > 0:
            print(
                f"{name:25s} {thresh:>8} {n_pos:>8} "
                f"{positives.min():>10.0f} {positives.median():>10.0f} "
                f"{positives.max():>12.0f}"
            )
        else:
            print(f"{name:25s} {thresh:>8} {n_pos:>8}  (none)")

    # How many images cross each threshold?
    print()
    print("Images crossing the threshold (would be Faulty by this class alone):")
    for cid, (name, thresh) in CONDITIONAL.items():
        col = f"area_{name}"
        n_cross = (out_df[col] > thresh).sum()
        n_pos = (out_df[col] > 0).sum()
        if n_pos > 0:
            print(
                f"  {name:25s} {n_cross:>6} / {n_pos:>6} "
                f"({n_cross / n_pos * 100:>5.1f}% of images that have this class)"
            )


if __name__ == "__main__":
    main()
