"""
build_yolo_seg_dataset.py
-------------------------
Convert the Krones COCO-format annotations into YOLO segmentation format
for Ultralytics training.

DESIGN A: 7 classes
  0: Scuffing             (conditional, threshold 75,000 px)
  1: Air bubble           (conditional, threshold 500 px)
  2: Contamination light  (conditional, threshold 180 px)
  3: Chip                 (conditional, threshold 200 px)
  4: Scuffing heavy       (conditional, threshold 1,200 px)
  5: Glass imperfection   (conditional, threshold 100 px)
  6: Hard-Faulty          (any of 16 always-Faulty classes merged)

Excluded (no label written):
  Good_no_defect, Water drop, No fault, Foam residue, Embossing, Roi, Bottle

The crops in audit_output/crops/ are GT-ROI crops cut from 1280x1024 originals.
Polygon coordinates in train_annotations.json are in original-image coords;
we shift them into crop-space using crops_meta.csv (crop_x0/y0/x1/y1) and
normalize to crop dimensions.

Output structure:
    <out_dir>/
      images/train/  (symlinks to crop PNGs)
      images/val/
      images/test/
      labels/train/  (one .txt per image, YOLO seg polygon format)
      labels/val/
      labels/test/
      krones_seg.yaml

YOLO seg label format (one line per polygon):
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    coords normalized to [0, 1] relative to crop width/height

Usage:
    python build_yolo_seg_dataset.py \\
        --root <challenge-data-root> \\
        --crops_dir audit_output/crops \\
        --crops_meta audit_output/crops_meta.csv \\
        --splits_csv audit_output/splits.csv \\
        --annotations train_annotations.json \\
        --out_dir audit_output/yolo_seg_7cls
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ---- Class definitions ----------------------------------------------------

# YOLO id -> COCO category name. Order matters: the .yaml lists names in this order.
CONDITIONAL = [
    (0, "Scuffing"),
    (1, "Air bubble"),
    (2, "Contamination light"),
    (3, "Chip"),
    (4, "Scuffing heavy"),
    (5, "Glass imperfection"),
]

# These 16 classes all map to YOLO class 6 (Hard-Faulty merged)
HARD_FAULTY_NAMES = [
    "Yeast residue",
    "Crown cap",
    "Mold",
    "Contamination dark",
    "No base visible",
    "Glass shard",
    "Break / Crack",
    "Label",
    "Foreign object - manual cleaning",
    "Foreign object - washing machine",
    "Insect",
    "Foil / Semitransparent",
    "Paint residue",
    "Liquid",
    "Straw",
    "Circlip",
]

# These get no label entry (excluded entirely)
EXCLUDED = {
    "Good_no_defect",
    "Water drop",
    "No fault",
    "Foam residue",
    "Embossing",
    "Roi",
    "Bottle",
}

YOLO_CLASS_NAMES = [name for _, name in CONDITIONAL] + ["Hard-Faulty"]


# ---- Polygon shifting ----------------------------------------------------


def polygon_to_yolo_seg(segmentation, crop_box, crop_w, crop_h):
    """Convert COCO polygon(s) to YOLO seg coords (normalized).

    Args:
        segmentation: list of polygons. Each polygon is [x1,y1,x2,y2,...] in
                      original-image pixel coords.
        crop_box: (x0, y0, x1, y1) of the crop in original-image coords.
        crop_w, crop_h: dimensions of the crop image (after the cut).

    Returns:
        list of strings, one per polygon, formatted as "x1 y1 x2 y2 ..."
        with normalized coords. Skips polygons fully outside the crop.
    """
    if not isinstance(segmentation, list):
        # COCO can also have RLE encoding; not used in this dataset.
        return []

    x0_c, y0_c, x1_c, y1_c = crop_box
    out_polys = []

    for poly in segmentation:
        if len(poly) < 6:  # need at least 3 points (6 coords)
            continue
        coords = []
        any_inside = False
        for j in range(0, len(poly), 2):
            ox = poly[j]
            oy = poly[j + 1]
            # Track whether at least one vertex was originally within the crop
            if x0_c <= ox <= x1_c and y0_c <= oy <= y1_c:
                any_inside = True
            # Shift to crop space and clip
            x = max(0.0, min(float(crop_w), ox - x0_c))
            y = max(0.0, min(float(crop_h), oy - y0_c))
            coords.append(x / crop_w)
            coords.append(y / crop_h)

        if not any_inside:
            continue
        out_polys.append(" ".join(f"{c:.6f}" for c in coords))

    return out_polys


# ---- Main ----------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        help="Dataset root (contains train_annotations.json, audit_output/)",
    )
    p.add_argument(
        "--crops_dir",
        default="audit_output/crops",
        help="Path to GT-ROI crops dir, relative to --root",
    )
    p.add_argument(
        "--crops_meta",
        default="audit_output/crops_meta.csv",
        help="Crop metadata: file_name, image_id, crop_x0/y0/x1/y1",
    )
    p.add_argument(
        "--splits_csv",
        default="audit_output/splits.csv",
        help="Splits CSV with file_name and split columns",
    )
    p.add_argument(
        "--annotations",
        default="train_annotations.json",
        help="COCO annotations JSON, relative to --root",
    )
    p.add_argument(
        "--out_dir",
        default="audit_output/yolo_seg_7cls",
        help="Output directory, relative to --root",
    )
    p.add_argument(
        "--copy_images",
        action="store_true",
        default=False,
        help="Copy images instead of symlinking (slower, more disk)",
    )
    args = p.parse_args()

    root = Path(args.root)
    crops_dir = root / args.crops_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load annotations and build mappings ----
    print(f"[1/6] Loading {args.annotations} ...")
    with open(root / args.annotations) as f:
        coco = json.load(f)
    print(
        f"      {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, "
        f"{len(coco['categories'])} categories"
    )

    # Build COCO-id -> YOLO-class-id mapping
    name_to_cocoid = {c["name"]: c["id"] for c in coco["categories"]}
    cocoid_to_yolo = {}
    print("[2/6] Building class mapping (Design A, 7 classes) ...")
    for yolo_id, name in CONDITIONAL:
        if name not in name_to_cocoid:
            raise KeyError(f"Conditional class '{name}' not in annotations")
        cocoid_to_yolo[name_to_cocoid[name]] = yolo_id
        print(f"        {yolo_id}: {name:25s}  (conditional)")
    n_hard_mapped = 0
    for name in HARD_FAULTY_NAMES:
        if name not in name_to_cocoid:
            print(f"        WARNING: hard-Faulty class '{name}' missing from JSON")
            continue
        cocoid_to_yolo[name_to_cocoid[name]] = 6
        n_hard_mapped += 1
    print(
        f"        6: Hard-Faulty (merged)    "
        f"({n_hard_mapped}/{len(HARD_FAULTY_NAMES)} source classes)"
    )

    # Sanity: anything in JSON not handled?
    accounted = set(n for _, n in CONDITIONAL) | set(HARD_FAULTY_NAMES) | EXCLUDED
    unexpected = set(name_to_cocoid.keys()) - accounted
    if unexpected:
        print(
            f"        WARNING: classes in JSON not in mapping or exclude list: "
            f"{unexpected}"
        )

    # COCO image_id -> file_name lookup (these are ORIGINAL filenames, not crops)
    # Group annotations by COCO image id
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # ---- Load crop metadata and splits ----
    print("[3/6] Loading crop metadata and splits ...")
    crops_meta = pd.read_csv(root / args.crops_meta)
    splits = pd.read_csv(root / args.splits_csv)
    print(
        f"      crops_meta: {len(crops_meta)} rows, columns: {list(crops_meta.columns)}"
    )
    print(f"      splits:     {len(splits)} rows, columns: {list(splits.columns)}")

    needed = ["file_name", "image_id", "crop_x0", "crop_y0", "crop_x1", "crop_y1"]
    for c in needed:
        if c not in crops_meta.columns:
            raise ValueError(
                f"crops_meta missing column '{c}'. Got: {list(crops_meta.columns)}"
            )
    # crops_meta uses 'image_id' as the COCO numeric id (per audit_report.txt)
    crops_lookup = crops_meta.set_index("file_name")[
        ["image_id", "crop_x0", "crop_y0", "crop_x1", "crop_y1"]
    ].to_dict("index")

    if "split" not in splits.columns:
        raise ValueError(
            f"splits CSV needs 'split' column. Got: {list(splits.columns)}"
        )
    # ---- Create output structure ----
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ---- Iterate crops, write labels and link images ----
    print(f"[4/6] Building YOLO-seg labels for {len(splits)} crops ...")

    stats = defaultdict(lambda: defaultdict(int))  # per-split counts
    skipped_no_crop_meta = 0
    skipped_no_anns = 0
    for _, row in tqdm(splits.iterrows(), total=len(splits), desc="crops"):
        fname = row["file_name"]
        split = row["split"]
        if split not in ("train", "val", "test"):
            continue

        meta = crops_lookup.get(fname)
        if meta is None:
            skipped_no_crop_meta += 1
            continue

        cocoid = int(meta["image_id"])
        x0_c = float(meta["crop_x0"])
        y0_c = float(meta["crop_y0"])
        x1_c = float(meta["crop_x1"])
        y1_c = float(meta["crop_y1"])
        crop_w = int(round(x1_c - x0_c))
        crop_h = int(round(y1_c - y0_c))
        if crop_w <= 0 or crop_h <= 0:
            skipped_no_crop_meta += 1
            continue

        # Image path on disk
        img_path = crops_dir / fname
        if not img_path.exists():
            skipped_no_crop_meta += 1
            continue

        # Verify crop dims match the actual file (cheap sanity check on a few)
        # We trust the metadata for speed but log mismatches once.
        # (Skipping to avoid opening every PNG)

        # Build label content
        label_lines = []
        anns = anns_by_image.get(cocoid, [])
        for ann in anns:
            cid = ann["category_id"]
            if cid not in cocoid_to_yolo:
                continue  # excluded class
            yolo_id = cocoid_to_yolo[cid]
            polys = polygon_to_yolo_seg(
                ann.get("segmentation", []), (x0_c, y0_c, x1_c, y1_c), crop_w, crop_h
            )
            for p in polys:
                label_lines.append(f"{yolo_id} {p}")
                stats[split][f"class_{yolo_id}"] += 1

        if not anns:
            skipped_no_anns += 1  # tracked but still write empty label

        # Write label file (empty file = "no defects in this crop")
        label_path = out_dir / "labels" / split / (Path(fname).stem + ".txt")
        with open(label_path, "w") as lf:
            lf.write("\n".join(label_lines))
            if label_lines:
                lf.write("\n")

        # Link or copy image
        link_target = out_dir / "images" / split / fname
        if link_target.exists() or link_target.is_symlink():
            link_target.unlink()
        if args.copy_images:
            import shutil

            shutil.copy(img_path, link_target)
        else:
            # Use absolute symlink so Ultralytics finds the file regardless of cwd
            link_target.symlink_to(img_path.resolve())

        stats[split]["images"] += 1
        if not label_lines:
            stats[split]["empty_labels"] += 1

    print()
    print("[5/6] Per-split statistics:")
    for split in ["train", "val", "test"]:
        s = stats[split]
        print(
            f"  {split:5s}: {s['images']:5d} images  "
            f"({s.get('empty_labels', 0)} with no defect annotations)"
        )
        for yolo_id in range(7):
            cnt = s.get(f"class_{yolo_id}", 0)
            cls_name = YOLO_CLASS_NAMES[yolo_id]
            print(f"           class {yolo_id} {cls_name:25s} {cnt:6d} polygons")
    if skipped_no_crop_meta:
        print(
            f"  WARNING: skipped {skipped_no_crop_meta} crops without metadata "
            f"or missing image files"
        )

    # ---- Write krones_seg.yaml ----
    print("[6/6] Writing krones_seg.yaml ...")
    yaml_path = out_dir / "krones_seg.yaml"
    yaml_content = (
        f"# YOLO seg dataset config — Krones (Design A, 7 classes)\n"
        f"# Auto-generated by build_yolo_seg_dataset.py\n"
        f"\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: 7\n"
        f"names:\n"
    )
    for i, name in enumerate(YOLO_CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    yaml_path.write_text(yaml_content)
    print(f"      wrote {yaml_path}")

    print()
    print("=" * 70)
    print("DATASET READY")
    print("=" * 70)
    print(f"Config: {yaml_path}")
    print("\nKick off training with:")
    print("  yolo segment train \\")
    print(f"      data={yaml_path} \\")
    print("      model=yolo11s-seg.pt \\")
    print("      epochs=50 imgsz=640 batch=16 \\")
    print("      device=0 \\")
    print("      project=runs/seg name=krones_7cls_yolo11s")


if __name__ == "__main__":
    main()
