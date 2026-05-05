"""
Krones — Build 8-channel segmentation masks for the full pipeline
------------------------------------------------------------------
Generates per-channel binary masks for every crop in splits_aug.csv:

  Channel 0: Background  (computed at training time as 1 - union, NOT saved)
  Channel 1: Scuffing                conditional, threshold 75000 px
  Channel 2: Air bubble              conditional, threshold   500 px
  Channel 3: Contamination light     conditional, threshold   180 px
  Channel 4: Chip                    conditional, threshold   200 px
  Channel 5: Scuffing heavy          conditional, threshold  1200 px
  Channel 6: Glass imperfection      conditional, threshold   100 px
  Channel 7: "Always Faulty" merged  (any of the 16 hard-Faulty classes)

For real crops:
  - Look up polygons from train_annotations.json
  - Rasterize conditional class polygons into their respective channels
  - Rasterize ALL hard-Faulty class polygons together into channel 7
  - Translate polygon coords from full-image space to crop space

For synth crops:
  - Channels 1-6 are empty (synth defects are all hard-Faulty rare classes)
  - Channel 7 = the mask saved in crops_aug_masks/ by build_copypaste_aug_with_masks.py

Storage: per-channel subdirs containing only crops with content in that channel.
Missing files = all-zero mask (handled by dataset class at training time).

Usage:
    python build_seg_masks.py --root /path/to/dataset
    python build_seg_masks.py --root /path/to/dataset --fresh   # nuke existing
"""
import os, json, argparse, shutil
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


# (channel_index, class_name, output_subdir_name)
CONDITIONAL_CHANNELS = [
    (1, "Scuffing",            "ch1_Scuffing"),
    (2, "Air bubble",          "ch2_Air_bubble"),
    (3, "Contamination light", "ch3_Contamination_light"),
    (4, "Chip",                "ch4_Chip"),
    (5, "Scuffing heavy",      "ch5_Scuffing_heavy"),
    (6, "Glass imperfection",  "ch6_Glass_imperfection"),
]
ALWAYS_FAULTY_CHANNEL = (7, "ch7_always_faulty")

# Classes that go into channel 7 (always-Faulty). Hard-Faulty classes from your
# training data. Adjust if your annotation names differ.
HARD_FAULTY_CLASSES = {
    "Yeast residue", "Crown cap", "Mold", "No base visible",
    "Glass shard", "Break / Crack", "Label",
    "Foreign object - manual cleaning", "Foreign object - washing machine",
    "Insect", "Foil / Semitransparent", "Paint residue", "Liquid",
    "Straw", "Circlip", "Contamination dark",
}

# Classes that are always Good (excluded from all channels)
ALWAYS_GOOD_CLASSES = {
    "Good_no_defect", "Water drop", "No fault", "Foam residue", "Embossing",
}


def rasterize_polygons(polygons, width, height, crop_x, crop_y):
    """Rasterize COCO polygons (full-image coords) into a binary mask of crop size."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        pts = [(poly[i] - crop_x, poly[i+1] - crop_y) for i in range(0, len(poly), 2)]
        draw.polygon(pts, fill=255)
    return np.array(mask, dtype=np.uint8)


def main(args):
    root = Path(args.root)
    out_dir = root / "audit_output"
    crops_dir = out_dir / "crops"
    crops_aug_masks_dir = out_dir / "crops_aug_masks"
    meta_csv = out_dir / "crops_meta.csv"
    splits_csv = out_dir / "splits_aug.csv"
    ann_path = root / "train_annotations.json"

    seg_dir = out_dir / "seg_masks"
    if seg_dir.exists() and args.fresh:
        print(f"removing existing {seg_dir} (--fresh)")
        shutil.rmtree(seg_dir)

    for _, _, subdir in CONDITIONAL_CHANNELS:
        (seg_dir / subdir).mkdir(parents=True, exist_ok=True)
    (seg_dir / ALWAYS_FAULTY_CHANNEL[1]).mkdir(parents=True, exist_ok=True)

    print(f"crops dir         : {crops_dir}")
    print(f"crops aug masks   : {crops_aug_masks_dir}")
    print(f"meta csv          : {meta_csv}")
    print(f"splits aug csv    : {splits_csv}")
    print(f"annotations       : {ann_path}")
    print(f"output            : {seg_dir}")
    for p in (crops_dir, meta_csv, splits_csv, ann_path):
        if not p.exists():
            print(f"ERROR: missing {p}"); return

    # 1. load annotations
    print("\n[1/4] loading annotations...")
    with open(ann_path) as f:
        ann = json.load(f)
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    name_to_id = {v: k for k, v in cats.items()}
    print(f"  found {len(cats)} categories")

    cond_names = {n for _, n, _ in CONDITIONAL_CHANNELS}
    missing_cond = cond_names - set(name_to_id.keys())
    if missing_cond:
        print(f"  WARNING: conditional class names not in annotations: {missing_cond}")
    missing_hard = HARD_FAULTY_CLASSES - set(name_to_id.keys())
    if missing_hard:
        print(f"  WARNING: hard-Faulty class names not in annotations: {missing_hard}")
    extras = (set(name_to_id.keys()) - cond_names - HARD_FAULTY_CLASSES
              - ALWAYS_GOOD_CLASSES - {"Roi", "Bottle"})
    if extras:
        print(f"  classes not categorized (will be ignored): {extras}")

    # group polygons by image_id and class name
    print("  indexing annotations...")
    img_to_class_polys = defaultdict(lambda: defaultdict(list))
    n_relevant = 0
    for a in ann["annotations"]:
        cname = cats[a["category_id"]]
        seg = a.get("segmentation")
        if not seg or isinstance(seg, dict):
            continue
        if cname in cond_names or cname in HARD_FAULTY_CLASSES:
            for poly in seg:
                if isinstance(poly, list) and len(poly) >= 6:
                    img_to_class_polys[a["image_id"]][cname].append(poly)
                    n_relevant += 1
    print(f"  {n_relevant} relevant polygons across {len(img_to_class_polys)} images")

    # 2. load crop metadata and splits
    print("\n[2/4] loading crop metadata and splits...")
    meta = pd.read_csv(meta_csv)
    splits = pd.read_csv(splits_csv)
    print(f"  crops_meta: {len(meta)} rows  columns: {list(meta.columns)}")
    print(f"  splits_aug: {len(splits)} rows")

    # detect crop bbox column convention
    if all(k in meta.columns for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")):
        get_crop_bbox = lambda r: (int(r["crop_x0"]), int(r["crop_y0"]),
                                   int(r["crop_x1"] - r["crop_x0"]),
                                   int(r["crop_y1"] - r["crop_y0"]))
        bbox_style = "x0,y0,x1,y1"
    elif all(k in meta.columns for k in ("crop_x", "crop_y", "crop_w", "crop_h")):
        get_crop_bbox = lambda r: (int(r["crop_x"]), int(r["crop_y"]),
                                   int(r["crop_w"]), int(r["crop_h"]))
        bbox_style = "x,y,w,h"
    else:
        print(f"ERROR: can't find crop bbox columns. Available: {list(meta.columns)}")
        return
    print(f"  bbox column style: {bbox_style}")

    meta_by_fn = {row["file_name"]: row for _, row in meta.iterrows()}

    # 3. process REAL crops
    print("\n[3/4] generating masks for real crops...")
    real_rows = (splits[splits.get("is_synth", 0) != 1]
                 if "is_synth" in splits.columns else splits)
    real_rows = real_rows.copy()
    print(f"  {len(real_rows)} real crops")

    per_channel_counts = Counter()
    n_real_processed = 0
    n_real_with_mask = 0
    for _, row in tqdm(real_rows.iterrows(), total=len(real_rows), desc="real crops"):
        file_name = row["file_name"]
        image_id = int(row["image_id"]) if pd.notna(row["image_id"]) else None
        if image_id is None or image_id < 0:
            continue
        meta_row = meta_by_fn.get(file_name)
        if meta_row is None:
            continue

        crop_x, crop_y, crop_w, crop_h = get_crop_bbox(meta_row)
        cls_polys = img_to_class_polys.get(image_id, {})
        if not cls_polys:
            n_real_processed += 1
            continue

        wrote_any = False
        # conditional channels
        for ch_idx, cname, subdir in CONDITIONAL_CHANNELS:
            polys = cls_polys.get(cname, [])
            if not polys:
                continue
            mask = rasterize_polygons(polys, crop_w, crop_h, crop_x, crop_y)
            if mask.sum() == 0:
                continue
            Image.fromarray(mask).save(seg_dir / subdir / file_name)
            per_channel_counts[ch_idx] += 1
            wrote_any = True

        # always-Faulty channel (union of all hard-Faulty class polygons)
        all_hard_polys = []
        for cname in HARD_FAULTY_CLASSES:
            all_hard_polys.extend(cls_polys.get(cname, []))
        if all_hard_polys:
            mask = rasterize_polygons(all_hard_polys, crop_w, crop_h, crop_x, crop_y)
            if mask.sum() > 0:
                Image.fromarray(mask).save(seg_dir / ALWAYS_FAULTY_CHANNEL[1] / file_name)
                per_channel_counts[7] += 1
                wrote_any = True

        n_real_processed += 1
        if wrote_any:
            n_real_with_mask += 1

    print(f"\n  real crops processed  : {n_real_processed}")
    print(f"  real crops with masks : {n_real_with_mask}")

    # 4. process SYNTH crops (channel 7 only)
    print("\n[4/4] copying synth masks into channel 7...")
    synth_rows = (splits[splits.get("is_synth", 0) == 1]
                  if "is_synth" in splits.columns else pd.DataFrame())
    print(f"  {len(synth_rows)} synth crops")

    n_synth_processed = 0
    n_synth_missing_mask = 0
    for _, row in tqdm(synth_rows.iterrows(), total=len(synth_rows), desc="synth crops"):
        file_name = row["file_name"]
        src_mask = crops_aug_masks_dir / file_name
        if not src_mask.exists():
            n_synth_missing_mask += 1
            continue
        mask = np.asarray(Image.open(src_mask).convert("L"), dtype=np.uint8)
        if mask.sum() == 0:
            continue
        # synth masks are already binary 0/255. Save into channel 7.
        Image.fromarray(mask).save(seg_dir / ALWAYS_FAULTY_CHANNEL[1] / file_name)
        per_channel_counts[7] += 1
        n_synth_processed += 1

    print(f"\n  synth crops processed     : {n_synth_processed}")
    print(f"  synth crops missing masks : {n_synth_missing_mask}")
    if n_synth_missing_mask > 0:
        print(f"  WARNING: re-run build_copypaste_aug_with_masks.py to fix this.")

    # final summary
    print("\n" + "=" * 60)
    print("SEGMENTATION MASK GENERATION COMPLETE")
    print("=" * 60)
    print(f"output dir: {seg_dir}")
    print()
    print("per-channel mask file counts:")
    for ch_idx, cname, subdir in CONDITIONAL_CHANNELS:
        n = per_channel_counts.get(ch_idx, 0)
        print(f"  ch{ch_idx} {cname:<22s}: {n:>5d} files  ({subdir})")
    print(f"  ch{ALWAYS_FAULTY_CHANNEL[0]} {'always-Faulty (merged)':<22s}: "
          f"{per_channel_counts.get(7, 0):>5d} files  ({ALWAYS_FAULTY_CHANNEL[1]})")

    total_files = sum(per_channel_counts.values())
    print(f"\n  total mask files: {total_files}")

    # disk usage estimate
    sample_files = list((seg_dir / CONDITIONAL_CHANNELS[0][2]).glob("*.png"))[:50]
    if sample_files:
        avg_size_kb = np.mean([f.stat().st_size for f in sample_files]) / 1024
        est_total_mb = avg_size_kb * total_files / 1024
        print(f"  approx disk size: {est_total_mb:.0f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset root")
    p.add_argument("--fresh", action="store_true",
                   help="delete existing seg_masks/ before regenerating")
    args = p.parse_args()
    main(args)