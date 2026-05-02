"""
Krones — Build expanded-ROI crops + stratified train/val/test split
-------------------------------------------------------------------
For every training image:
  1. Read the ROI bbox from train_annotations.json
  2. Expand it by PAD pixels on all sides (clipped to image bounds)
  3. Save the cropped image to <out_dir>/crops/<file_name>

Then build a splits.csv with stratified 80/10/10 train/val/test based on:
  - binary label (Faulty/Good)
  - dominant defect class (so rare classes appear in val and test)
  - conditional-borderline flag (so val/test contain just-above and
    just-below threshold examples)

Usage:
    python build_crops_and_splits.py --root /path/to/dataset --pad 150

Outputs (under <root>/audit_output/):
    crops/<file_name>.png       expanded-ROI cropped images
    crops_meta.csv              per-image: orig size, roi bbox, crop bbox, crop size
    splits.csv                  per-image: split in {train, val, test}
    splits_summary.txt          counts & balance per split
"""

import os, json, argparse, math, random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from PIL import Image

# Conditional thresholds — must match label_audit.py
CONDITIONAL = {
    "Scuffing":            75000,
    "Air bubble":            500,
    "Contamination light":   180,
    "Chip":                  200,
    "Scuffing heavy":       1200,
    "Glass imperfection":    100,
}
HARD_GOOD = {"Water drop", "No fault", "Foam residue", "Embossing"}
META = {"Roi", "No fault"}

# Conditional borderline ratio: an image is "borderline" if its dominant defect
# is a conditional class with total area within +/- BORDERLINE_PCT of threshold.
# For small thresholds, we also widen by an absolute margin so small classes
# (Glass imperfection, Air bubble) get enough borderline coverage.
BORDERLINE_PCT = 0.50
BORDERLINE_ABS_MIN = 200    # pixels — minimum half-width of borderline band


def expand_bbox(bbox, pad, W, H):
    """Expand [x,y,w,h] by `pad` on each side, clip to image bounds."""
    x, y, w, h = bbox
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(W, int(x + w) + pad)
    y1 = min(H, int(y + h) + pad)
    return x0, y0, x1, y1


def stratified_split(df, key_col, ratios=(0.8, 0.1, 0.1), seed=0):
    """Stratify by key_col. Returns dict image_id -> split."""
    rng = random.Random(seed)
    splits = {}
    for key, group in df.groupby(key_col):
        ids = group["image_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios[0]))
        n_val   = int(round(n * ratios[1]))
        # Remaining go to test (handles rounding)
        train_ids = ids[:n_train]
        val_ids   = ids[n_train:n_train + n_val]
        test_ids  = ids[n_train + n_val:]
        for i in train_ids: splits[i] = "train"
        for i in val_ids:   splits[i] = "val"
        for i in test_ids:  splits[i] = "test"
    return splits


def main(root, pad, n_workers, dry_run):
    img_dir   = os.path.join(root, "train_images")
    ann_path  = os.path.join(root, "train_annotations.json")
    out_dir   = os.path.join(root, "audit_output")
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    print(f"[1/5] loading annotations ...")
    ann = json.load(open(ann_path, "r"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    name2id = {v: k for k, v in cats.items()}
    ROI_ID = name2id["Roi"]

    img_to_roi = {}
    img_class_count = defaultdict(Counter)
    img_class_area  = defaultdict(Counter)

    for a in ann["annotations"]:
        if a["category_id"] == ROI_ID:
            img_to_roi[a["image_id"]] = a["bbox"]
    for a in ann["annotations"]:
        cname = cats[a["category_id"]]
        if cname in META:
            continue
        iid = a["image_id"]
        img_class_count[iid][cname] += 1
        img_class_area[iid][cname] += float(a.get("area", 0.0))

    images = ann["images"]
    print(f"      {len(images)} images, {len(img_to_roi)} have ROI bboxes")

    # -----------------------------------------------------------------
    # 2. crop loop
    # -----------------------------------------------------------------
    print(f"[2/5] cropping with pad={pad}px ...")
    meta_rows = []
    missing_files = 0
    missing_roi = 0
    failures = 0

    for idx, im in enumerate(images, 1):
        if idx % 2000 == 0:
            print(f"      {idx}/{len(images)}")
        iid   = im["id"]
        fname = im["file_name"]
        in_path = os.path.join(img_dir, fname)
        if not os.path.exists(in_path):
            missing_files += 1
            continue

        try:
            with Image.open(in_path) as img:
                W, H = img.size
                roi = img_to_roi.get(iid)
                if roi is None:
                    # no ROI: use whole image
                    x0, y0, x1, y1 = 0, 0, W, H
                    missing_roi += 1
                else:
                    x0, y0, x1, y1 = expand_bbox(roi, pad, W, H)
                if dry_run:
                    pass
                else:
                    crop = img.crop((x0, y0, x1, y1))
                    out_path = os.path.join(crops_dir, fname)
                    crop.save(out_path, "PNG", optimize=False)
                meta_rows.append({
                    "image_id":  iid,
                    "file_name": fname,
                    "orig_w":    W,
                    "orig_h":    H,
                    "roi_x":     roi[0] if roi else None,
                    "roi_y":     roi[1] if roi else None,
                    "roi_w":     roi[2] if roi else None,
                    "roi_h":     roi[3] if roi else None,
                    "crop_x0":   x0,
                    "crop_y0":   y0,
                    "crop_x1":   x1,
                    "crop_y1":   y1,
                    "crop_w":    x1 - x0,
                    "crop_h":    y1 - y0,
                })
        except Exception as e:
            failures += 1
            print(f"      FAIL {fname}: {e}")

    print(f"      done. missing files: {missing_files}, missing ROI: {missing_roi}, failures: {failures}")
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(out_dir, "crops_meta.csv"), index=False)

    # crop size statistics
    if len(meta_df):
        print(f"      crop size stats: "
              f"w mean={meta_df['crop_w'].mean():.0f} median={meta_df['crop_w'].median():.0f} "
              f"max={meta_df['crop_w'].max()} | "
              f"h mean={meta_df['crop_h'].mean():.0f} median={meta_df['crop_h'].median():.0f} "
              f"max={meta_df['crop_h'].max()}")

    # -----------------------------------------------------------------
    # 3. build per-image label table for splitting
    # -----------------------------------------------------------------
    print(f"[3/5] computing labels and stratification keys ...")
    HARD_FAULTY = {
        "Contamination dark", "Yeast residue", "No base visible", "Crown cap",
        "Mold", "Foreign object - manual cleaning", "Glass shard",
        "Foreign object - washing machine", "Break / Crack", "Label",
        "Liquid", "Paint residue", "Foil / Semitransparent", "Circlip",
        "Straw", "Insect",
    }

    label_rows = []
    for im in images:
        iid = im["id"]
        counts = img_class_count.get(iid, Counter())
        areas  = img_class_area.get(iid, Counter())

        hard_hits = [c for c in counts if c in HARD_FAULTY]
        cond_hits = [(c, areas[c]) for c in counts if c in CONDITIONAL and areas[c] > CONDITIONAL[c]]
        is_faulty = bool(hard_hits) or bool(cond_hits)

        # dominant defect = max area among defect classes
        defect_only = {c: a for c, a in areas.items() if c not in HARD_GOOD}
        dominant = max(defect_only, key=defect_only.get) if defect_only else "Good_no_defect"

        # borderline flag: dominant is conditional and area is near threshold.
        # band = max(BORDERLINE_PCT * threshold, BORDERLINE_ABS_MIN)
        borderline = 0
        if dominant in CONDITIONAL:
            thr = CONDITIONAL[dominant]
            a = areas[dominant]
            band = max(BORDERLINE_PCT * thr, BORDERLINE_ABS_MIN)
            if (thr - band) <= a <= (thr + band):
                borderline = 1

        # stratification key: (label, dominant, borderline)
        strat_key = f"{int(is_faulty)}|{dominant}|{borderline}"

        label_rows.append({
            "image_id":   iid,
            "file_name":  im["file_name"],
            "label":      int(is_faulty),
            "dominant":   dominant,
            "borderline": borderline,
            "strat_key":  strat_key,
        })

    label_df = pd.DataFrame(label_rows)

    # -----------------------------------------------------------------
    # 4. stratified split
    # -----------------------------------------------------------------
    print(f"[4/5] stratified split (80/10/10) ...")
    # any stratum with <10 images: lump into a fallback bucket per binary-label
    grp_sizes = label_df["strat_key"].value_counts()
    rare_keys = grp_sizes[grp_sizes < 10].index.tolist()
    print(f"      strata: {len(grp_sizes)}, rare (<10 imgs): {len(rare_keys)} (will be merged by binary label)")

    label_df["strat_key_eff"] = label_df.apply(
        lambda r: (f"rare_{r['label']}" if r["strat_key"] in rare_keys else r["strat_key"]),
        axis=1)

    splits = stratified_split(label_df, "strat_key_eff", (0.8, 0.1, 0.1), seed=42)
    label_df["split"] = label_df["image_id"].map(splits)

    # Post-hoc guarantee: for every borderline+dominant class with >=3 images,
    # ensure val and test each have >=1. If not, move from train to val/test.
    print(f"      checking borderline coverage per class ...")
    moves = 0
    for (dom, _), g in label_df[label_df["borderline"] == 1].groupby(["dominant", "borderline"]):
        if len(g) < 3:
            continue
        for needed_split in ["val", "test"]:
            if (g["split"] == needed_split).sum() == 0:
                # find a candidate in train and move it
                candidates = g[g["split"] == "train"]
                if len(candidates) > 0:
                    move_id = candidates.iloc[0]["image_id"]
                    label_df.loc[label_df["image_id"] == move_id, "split"] = needed_split
                    g = label_df[(label_df["dominant"] == dom) & (label_df["borderline"] == 1)]
                    moves += 1
                    print(f"        moved 1 borderline '{dom}' image  train -> {needed_split}")
    if moves == 0:
        print(f"        no moves needed")

    label_df.to_csv(os.path.join(out_dir, "splits.csv"), index=False)

    # -----------------------------------------------------------------
    # 5. summary
    # -----------------------------------------------------------------
    print(f"[5/5] summary\n")
    lines = []
    L = lines.append
    L("="*70); L("SPLITS SUMMARY"); L("="*70)
    for sp in ["train", "val", "test"]:
        sub = label_df[label_df["split"] == sp]
        n = len(sub)
        nf = int(sub["label"].sum())
        ng = n - nf
        L(f"{sp:5s}: {n:6d} images   Faulty={nf:6d} ({nf/n*100:5.2f}%)   Good={ng:6d} ({ng/n*100:5.2f}%)")
    L("")
    L("-"*70)
    L("rare-class coverage (per split, by dominant defect)")
    L("-"*70)
    rare_classes = ["Insect", "Straw", "Circlip", "Foil / Semitransparent",
                    "Paint residue", "Liquid", "Glass imperfection",
                    "Scuffing heavy", "Break / Crack", "Label",
                    "Foreign object - washing machine"]
    pivot = (label_df[label_df["dominant"].isin(rare_classes)]
             .groupby(["dominant", "split"]).size().unstack(fill_value=0))
    for col in ["train", "val", "test"]:
        if col not in pivot.columns: pivot[col] = 0
    pivot = pivot[["train", "val", "test"]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    L(str(pivot))
    # warn on zero coverage in val/test
    L("")
    warnings = []
    for cls, row in pivot.iterrows():
        if row["val"] == 0:
            warnings.append(f"  WARNING: '{cls}' has 0 images in VAL")
        if row["test"] == 0:
            warnings.append(f"  WARNING: '{cls}' has 0 images in TEST")
        if row["train"] == 0:
            warnings.append(f"  CRITICAL: '{cls}' has 0 images in TRAIN")
    if warnings:
        L("COVERAGE WARNINGS:")
        for w in warnings: L(w)
    else:
        L("  all rare classes have >=1 image in train, val, and test  [OK]")
    L("")
    L("-"*70)
    L("borderline (conditional-class ±30% of threshold) coverage per split")
    L("-"*70)
    bp = (label_df[label_df["borderline"] == 1]
          .groupby(["dominant", "split"]).size().unstack(fill_value=0))
    if len(bp):
        bp["total"] = bp.sum(axis=1)
        L(str(bp))
    else:
        L("  (no borderline images found — check BORDERLINE_PCT)")
    L("")
    L("="*70)
    L(f"outputs: {out_dir}")
    L(f"  crops/                 ({len(meta_df)} cropped images, pad={pad}px)")
    L(f"  crops_meta.csv         (per-image crop bookkeeping)")
    L(f"  splits.csv             (per-image train/val/test assignment)")
    L(f"  splits_summary.txt     (this report)")
    L("="*70)

    report = "\n".join(lines)
    with open(os.path.join(out_dir, "splits_summary.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--pad", type=int, default=150)
    p.add_argument("--workers", type=int, default=1,
                   help="(reserved for future parallelization)")
    p.add_argument("--dry-run", action="store_true",
                   help="do everything except writing crop PNGs (for quick testing)")
    args = p.parse_args()
    main(args.root, args.pad, args.workers, args.dry_run)   