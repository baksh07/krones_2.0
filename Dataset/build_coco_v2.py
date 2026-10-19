"""
Build COCO instance-seg JSONs from the audit_output dataset (v2).

What this does:
  - Reads the cleaned 80/10/10 splits from audit_output/splits.csv.
  - Reads per-image crop bboxes from audit_output/crops_meta.csv (pad=150 crops).
  - Reads polygons from train_annotations.json.
  - Translates each polygon to cropped-image coords (subtract crop_x0/y0).
  - Drops 5 categories (Roi, No fault, Embossing, Foam residue, Water drop).
  - Remaps remaining 22 categories to contiguous IDs 1..22 (alphabetical-by-old-id,
    same as build_krones_coco_cropped.py so eval evaluators line up).
  - Writes 3 COCO JSONs (train, val, test) plus krones_classes.json.

Real images only — synth crops_aug/ images have no polygons saved by
build_copypaste_aug.py. Adding synth requires modifying that script first.

Challenge dataset inputs (read-only):
  train_annotations.json
  audit_output/splits.csv
  audit_output/crops_meta.csv
  audit_output/crops/<file_name>             # actual cropped PNGs (referenced, not copied)

Outputs (under the repository's Dataset/coco_v2/ directory):
  krones_train.json
  krones_val.json
  krones_test.json
  krones_classes.json
  build_summary.txt

Usage:
  python build_coco_v2.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from project_paths import DATA_ROOT, REPO_ROOT

ANN_FILE = DATA_ROOT / "train_annotations.json"
SPLITS_CSV = DATA_ROOT / "audit_output" / "splits.csv"
CROPS_META = DATA_ROOT / "audit_output" / "crops_meta.csv"
CROPS_DIR = DATA_ROOT / "audit_output" / "crops"  # referenced by trainer, not copied

OUT_DIR = REPO_ROOT / "Dataset" / "coco_v2"

# Same drop list as build_krones_coco_cropped.py — keeps the verdict-rule
# class set consistent between v1 and v2 datasets.
DROP_CATEGORIES = {
    "Roi",
    "No fault",
    "Embossing",
    "Foam residue",
    "Water drop",
}


def shift_polygon(poly, dx, dy):
    out = []
    for i in range(0, len(poly), 2):
        out.append(poly[i] - dx)
        out.append(poly[i + 1] - dy)
    return out


def shoelace_area(poly):
    xs = poly[0::2]
    ys = poly[1::2]
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------
    print(f"Loading {ANN_FILE.name} ...")
    with open(ANN_FILE) as f:
        coco = json.load(f)
    print(
        f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations, "
        f"{len(coco['categories'])} raw categories"
    )

    print("Loading splits.csv ...")
    splits_df = pd.read_csv(SPLITS_CSV)
    split_by_iid = dict(zip(splits_df["image_id"], splits_df["split"]))
    print(f"  {len(splits_df)} rows; splits: {Counter(splits_df['split'])}")

    print("Loading crops_meta.csv ...")
    meta_df = pd.read_csv(CROPS_META)
    meta_by_iid = {
        row["image_id"]: (
            int(row["crop_x0"]),
            int(row["crop_y0"]),
            int(row["crop_x1"]),
            int(row["crop_y1"]),
            int(row["crop_w"]),
            int(row["crop_h"]),
        )
        for _, row in meta_df.iterrows()
    }
    print(
        f"  {len(meta_df)} crops; crop_w mean={meta_df['crop_w'].mean():.0f} "
        f"median={meta_df['crop_w'].median():.0f} max={meta_df['crop_w'].max()}"
    )

    # -----------------------------------------------------------------
    # Build category remap (drop 5, contiguous 1..22)
    # -----------------------------------------------------------------
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    keep_old_ids = sorted(
        cid for cid, name in cats.items() if name not in DROP_CATEGORIES
    )
    old_to_new = {old: new for new, old in enumerate(keep_old_ids, start=1)}
    new_categories = [
        {"id": old_to_new[old], "name": cats[old], "supercategory": cats[old]}
        for old in keep_old_ids
    ]
    new_id_to_name = {c["id"]: c["name"] for c in new_categories}
    n_classes = len(new_categories)
    print(
        f"\nKept {n_classes} categories (dropped {len(DROP_CATEGORIES)}: "
        f"{sorted(DROP_CATEGORIES)})"
    )

    # Group annotations by image, filtering kept categories only
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] not in old_to_new:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    # -----------------------------------------------------------------
    # Build per-image entries
    # -----------------------------------------------------------------
    print("\nTranslating polygons to cropped frame ...")
    images_by_split: dict[str, list[dict]] = defaultdict(list)
    anns_by_split: dict[str, list[dict]] = defaultdict(list)
    n_skipped_no_meta = 0
    n_skipped_no_split = 0
    n_dropped_anns_outside = 0
    n_kept_anns = 0

    next_ann_id_global = 1  # IDs are reassigned per-split when writing JSONs

    for im in tqdm(coco["images"], unit="img"):
        iid = im["id"]
        if iid not in meta_by_iid:
            n_skipped_no_meta += 1
            continue
        if iid not in split_by_iid:
            n_skipped_no_split += 1
            continue
        split = split_by_iid[iid]
        cx0, cy0, cx1, cy1, cw, ch = meta_by_iid[iid]

        new_image = {
            "id": iid,
            "file_name": im["file_name"],
            "width": cw,
            "height": ch,
            "date_captured": im.get("date_captured", ""),
        }
        images_by_split[split].append(new_image)

        for ann in anns_by_img[iid]:
            new_segs = []
            for poly in ann.get("segmentation", []):
                if len(poly) < 6:
                    continue
                shifted = shift_polygon(poly, cx0, cy0)
                xs = shifted[0::2]
                ys = shifted[1::2]
                # Drop polygons fully outside the cropped region
                if max(xs) < 0 or min(xs) > cw or max(ys) < 0 or min(ys) > ch:
                    continue
                new_segs.append(shifted)
            if not new_segs:
                n_dropped_anns_outside += 1
                continue

            all_xs, all_ys = [], []
            for p in new_segs:
                all_xs.extend(p[0::2])
                all_ys.extend(p[1::2])
            xmin = max(0.0, min(all_xs))
            ymin = max(0.0, min(all_ys))
            xmax = min(float(cw), max(all_xs))
            ymax = min(float(ch), max(all_ys))
            new_bbox = [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]
            new_area = sum(shoelace_area(p) for p in new_segs)

            anns_by_split[split].append(
                {
                    "id": next_ann_id_global,
                    "image_id": iid,
                    "category_id": old_to_new[ann["category_id"]],
                    "bbox": new_bbox,
                    "segmentation": new_segs,
                    "area": new_area,
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            next_ann_id_global += 1
            n_kept_anns += 1

    print("\nTranslation complete:")
    print(f"  Images skipped (no crop meta):  {n_skipped_no_meta}")
    print(f"  Images skipped (no split):      {n_skipped_no_split}")
    print(f"  Annotations kept:               {n_kept_anns}")
    print(f"  Annotations dropped (outside):  {n_dropped_anns_outside}")

    # -----------------------------------------------------------------
    # Write JSONs
    # -----------------------------------------------------------------
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("COCO v2 BUILD SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"crops dir (referenced): {CROPS_DIR}")
    summary_lines.append(f"output dir:             {OUT_DIR}")
    summary_lines.append(f"classes:                {n_classes} (dropped 5)")
    summary_lines.append(
        f"images skipped:         no_meta={n_skipped_no_meta}, no_split={n_skipped_no_split}"
    )
    summary_lines.append(f"annotations kept:       {n_kept_anns:,}")
    summary_lines.append(
        f"annotations dropped:    {n_dropped_anns_outside:,} (poly fully outside crop)"
    )
    summary_lines.append("")

    for split in ("train", "val", "test"):
        ims = images_by_split[split]
        anns = anns_by_split[split]
        # reassign annotation IDs to be contiguous within each split JSON
        for new_id, a in enumerate(anns, start=1):
            a["id"] = new_id
        out = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "type": "instances",
            "images": ims,
            "annotations": anns,
            "categories": new_categories,
        }
        path = OUT_DIR / f"krones_{split}.json"
        with open(path, "w") as f:
            json.dump(out, f)

        # Per-class image count (how many images contain at least one ann of this class)
        per_cls_imgs: Counter = Counter()
        ann_by_img_split = defaultdict(set)
        for a in anns:
            ann_by_img_split[a["image_id"]].add(a["category_id"])
        for cids in ann_by_img_split.values():
            for c in cids:
                per_cls_imgs[c] += 1

        n_with_anns = len(ann_by_img_split)
        line = (
            f"{split:5s}: {len(ims):>5} images ({n_with_anns} with kept anns), "
            f"{len(anns):>6} anns"
        )
        print(line)
        summary_lines.append(line)

        # Print per-class image count for this split (so you can spot zero-coverage classes)
        summary_lines.append(f"  per-class image counts ({split}):")
        for cid in sorted(new_id_to_name):
            n = per_cls_imgs.get(cid, 0)
            marker = "  (!)" if n == 0 else ""
            summary_lines.append(f"    {new_id_to_name[cid]:35s} {n:>5}{marker}")
        summary_lines.append("")

    # Classes file
    classes_path = OUT_DIR / "krones_classes.json"
    with open(classes_path, "w") as f:
        json.dump([c["name"] for c in new_categories], f, indent=2)
    print(f"\n  {classes_path.name}: {n_classes} classes")

    # Summary file
    summary_path = OUT_DIR / "build_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  {summary_path.name}: written")


if __name__ == "__main__":
    main()
