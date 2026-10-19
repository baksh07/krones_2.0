"""
Convert original train_annotations.json into ROI-cropped COCO splits.

What this does:
  - For every train image, crops to its bottle-base ROI annotation
    (the original COCO category 'Roi') with 50 px padding on each side
    (clipped to the image boundary).
  - Writes cropped PNGs to Dataset/cropped_images/ (preserving filenames).
  - Translates every kept polygon annotation into the cropped coordinate frame.
  - Drops 5 categories that don't help instance-seg training:
      Roi, No fault, Embossing, Foam residue, Water drop.
  - Remaps the remaining 22 categories to contiguous IDs 1..22.
  - Stratified 90/10 train/val split by image_id (same seed as
    build_krones_coco.py for direct comparison with v0).
  - Drops annotations whose polygons fall entirely outside the cropped
    region (audit § 6.6: ~3.9% of defect annotations).

Outputs (under <repo>/Dataset/coco_cropped/):
  krones_train.json
  krones_val.json
  krones_classes.json     (the 22 names, contiguous order — same as v0)
  roi_lookup.json         (filename -> original ROI [x, y, w, h])

Cropped images at <repo>/Dataset/cropped_images/<filename>.png
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from PIL import Image
from tqdm import tqdm

from project_paths import DATA_ROOT, REPO_ROOT

ANN_FILE = DATA_ROOT / "train_annotations.json"  # ORIGINAL: has Roi category
SRC_IMG_DIR = DATA_ROOT / "train_images"
OUT_IMG_DIR = REPO_ROOT / "Dataset" / "cropped_images"
OUT_DIR = REPO_ROOT / "Dataset" / "coco_cropped"

DROP_CATEGORIES = {
    "Roi",
    "No fault",
    "Embossing",
    "Foam residue",
    "Water drop",
}

ROI_CATEGORY_NAME = "Roi"


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


def main(seed: int = 42, val_frac: float = 0.10, padding: int = 50):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ANN_FILE} ...")
    with open(ANN_FILE) as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    roi_cat_id = next(cid for cid, n in cats.items() if n == ROI_CATEGORY_NAME)
    print(f"Roi category id = {roi_cat_id}")

    roi_by_img: dict[int, list[float]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] == roi_cat_id:
            roi_by_img[ann["image_id"]] = ann["bbox"]
    print(f"Found ROI for {len(roi_by_img)} / {len(coco['images'])} images")

    # Build kept-category remap (drop 5 categories, contiguous IDs 1..22)
    keep_old_ids = sorted(
        cid for cid, name in cats.items() if name not in DROP_CATEGORIES
    )
    old_to_new = {old: new for new, old in enumerate(keep_old_ids, start=1)}
    new_categories = [
        {"id": old_to_new[old], "name": cats[old], "supercategory": cats[old]}
        for old in keep_old_ids
    ]
    new_id_to_name = {c["id"]: c["name"] for c in new_categories}
    print(f"Kept {len(new_categories)} categories (dropped {len(DROP_CATEGORIES)})")

    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] not in old_to_new:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    print(f"\nCropping images (padding={padding}) and shifting annotations ...")
    new_images = []
    new_annotations = []
    next_ann_id = 1
    n_skipped_imgs = 0
    n_dropped_anns = 0
    n_kept_anns = 0
    roi_lookup: dict[str, list[float]] = {}

    for im in tqdm(coco["images"], unit="img"):
        if im["id"] not in roi_by_img:
            n_skipped_imgs += 1
            continue
        roi = roi_by_img[im["id"]]
        rx, ry, rw, rh = roi
        roi_lookup[im["file_name"]] = roi

        W, H = im["width"], im["height"]
        cx0 = max(0, int(round(rx - padding)))
        cy0 = max(0, int(round(ry - padding)))
        cx1 = min(W, int(round(rx + rw + padding)))
        cy1 = min(H, int(round(ry + rh + padding)))
        cw = cx1 - cx0
        ch = cy1 - cy0
        if cw <= 0 or ch <= 0:
            n_skipped_imgs += 1
            continue

        src_path = SRC_IMG_DIR / im["file_name"]
        dst_path = OUT_IMG_DIR / im["file_name"]
        if not dst_path.exists():
            try:
                img = Image.open(src_path)
                cropped = img.crop((cx0, cy0, cx1, cy1))
                cropped.save(dst_path, "PNG")
            except Exception as e:
                print(f"WARN: failed to crop {im['file_name']}: {e}")
                n_skipped_imgs += 1
                continue

        new_images.append(
            {
                "id": im["id"],
                "file_name": im["file_name"],
                "width": cw,
                "height": ch,
                "date_captured": im.get("date_captured", ""),
            }
        )

        for ann in anns_by_img[im["id"]]:
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
                n_dropped_anns += 1
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

            new_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": im["id"],
                    "category_id": old_to_new[ann["category_id"]],
                    "bbox": new_bbox,
                    "segmentation": new_segs,
                    "area": new_area,
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            next_ann_id += 1
            n_kept_anns += 1

    print("\nCropping complete:")
    print(f"  Images written: {len(new_images)} (skipped {n_skipped_imgs})")
    print(
        f"  Annotations kept: {n_kept_anns} "
        f"(dropped {n_dropped_anns} fully outside crop)"
    )

    # Stratified 90/10 split (mirror build_krones_coco.py logic)
    anns_by_img_kept: dict[int, list[dict]] = defaultdict(list)
    for a in new_annotations:
        anns_by_img_kept[a["image_id"]].append(a)

    class_image_counts: Counter = Counter()
    for anns in anns_by_img_kept.values():
        for cid in {a["category_id"] for a in anns}:
            class_image_counts[cid] += 1

    print("\nImages per kept class (cropped dataset):")
    for cid in sorted(new_id_to_name):
        print(f"  {new_id_to_name[cid]:35s} {class_image_counts.get(cid, 0):>6}")

    def stratum(img_id: int) -> int:
        anns = anns_by_img_kept.get(img_id, [])
        if not anns:
            return -1
        cids = {a["category_id"] for a in anns}
        return min(cids, key=lambda c: class_image_counts[c])

    img_by_stratum: dict[int, list[int]] = defaultdict(list)
    for im in new_images:
        img_by_stratum[stratum(im["id"])].append(im["id"])

    rng = random.Random(seed)
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    for s, ids in img_by_stratum.items():
        rng.shuffle(ids)
        if len(ids) < 2:
            train_ids.update(ids)
            continue
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_ids.update(ids[:n_val])
        train_ids.update(ids[n_val:])

    print(f"\nSplit: train={len(train_ids)}, val={len(val_ids)}")

    val_class_counts: Counter = Counter()
    for img_id in val_ids:
        for cid in {a["category_id"] for a in anns_by_img_kept.get(img_id, [])}:
            val_class_counts[cid] += 1
    print("\nPer-class val image counts (! = zero):")
    for cid in sorted(new_id_to_name):
        n = val_class_counts.get(cid, 0)
        marker = "  (!)" if n == 0 else ""
        print(f"  {new_id_to_name[cid]:35s} {n:>4}{marker}")

    img_by_id_new = {im["id"]: im for im in new_images}

    def write_split(name: str, ids: set[int]):
        out = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "type": "instances",
            "images": [img_by_id_new[i] for i in sorted(ids) if i in img_by_id_new],
            "annotations": [a for a in new_annotations if a["image_id"] in ids],
            "categories": new_categories,
        }
        for new_id, ann in enumerate(out["annotations"], start=1):
            ann["id"] = new_id
        path = OUT_DIR / f"krones_{name}.json"
        with open(path, "w") as f:
            json.dump(out, f)
        n_with = sum(1 for im in out["images"] if anns_by_img_kept.get(im["id"]))
        print(
            f"  {path.name}: {len(out['images'])} images "
            f"({n_with} with kept anns), {len(out['annotations'])} anns"
        )

    print("\nWriting splits:")
    write_split("train", train_ids)
    write_split("val", val_ids)

    classes_path = OUT_DIR / "krones_classes.json"
    with open(classes_path, "w") as f:
        json.dump([c["name"] for c in new_categories], f, indent=2)
    print(f"  {classes_path.name}: {len(new_categories)} classes")

    roi_lookup_path = OUT_DIR / "roi_lookup.json"
    with open(roi_lookup_path, "w") as f:
        json.dump(roi_lookup, f)
    print(f"  {roi_lookup_path.name}: ROI bboxes for {len(roi_lookup)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument(
        "--padding", type=int, default=50, help="Pixels of padding around the ROI bbox"
    )
    args = parser.parse_args()
    main(seed=args.seed, val_frac=args.val_frac, padding=args.padding)
