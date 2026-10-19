"""
Convert Krones COCO splits (krones_train.json / krones_val.json) into the
Ultralytics YOLO segmentation format used by yolo train segment.

Output layout (under <repo>/Dataset/yolo/):
  images/train/<img_stem>.png  ← symlinks to original PNGs (no copies)
  images/val/<img_stem>.png
  labels/train/<img_stem>.txt
  labels/val/<img_stem>.txt
  krones.yaml                  ← Ultralytics dataset config

YOLO label format (one annotation per line):
  <class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xN_norm> <yN_norm>

Where:
  - class_id is 0..N-1 (we map COCO id 1..22 -> YOLO id 0..21).
  - Coords are normalized polygon vertices in [0, 1].

Multi-polygon annotations are split into separate lines (each polygon = one
YOLO instance). Same approach as Ultralytics' JSON2YOLO tool.

Empty-annotation images: by default we EXCLUDE them from train (mirroring
Detectron2's FILTER_EMPTY_ANNOTATIONS=True for v0 fairness) but INCLUDE them
in val (so verdict eval covers clean bottles too). Pass --include-empty-train
to include them in train as well (acts as hard negatives).

Run:
  python build_krones_yolo.py
  python build_krones_yolo.py --include-empty-train
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from project_paths import DATA_ROOT, REPO_ROOT

COCO_DIR = REPO_ROOT / "Dataset" / "coco"
IMG_DIR = DATA_ROOT / "train_images"
OUT_DIR = REPO_ROOT / "Dataset" / "yolo"


def coco_to_yolo_label(ann, img_w, img_h, coco_id_to_yolo_id):
    """One COCO annotation -> list of YOLO label lines (one per polygon)."""
    yolo_id = coco_id_to_yolo_id[ann["category_id"]]
    lines = []
    for poly in ann.get("segmentation", []):
        if len(poly) < 6:  # need at least 3 points
            continue
        # Normalize each x,y by image width/height
        norm = []
        for i in range(0, len(poly), 2):
            x = poly[i] / img_w
            y = poly[i + 1] / img_h
            # Clamp to [0, 1] in case annotation goes slightly outside
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            norm.append(f"{x:.6f}")
            norm.append(f"{y:.6f}")
        lines.append(f"{yolo_id} " + " ".join(norm))
    return lines


def convert_split(
    split_name, coco_path, out_img_dir, out_lbl_dir, coco_id_to_yolo_id, include_empty
):
    with open(coco_path) as f:
        coco = json.load(f)

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_total = 0
    n_with_anns = 0
    n_empty = 0
    n_skipped = 0
    n_polys = 0

    for im in coco["images"]:
        anns = anns_by_img.get(im["id"], [])
        is_empty = len(anns) == 0
        if is_empty and not include_empty:
            n_skipped += 1
            continue

        # Symlink the source image (force-replace if exists)
        src = IMG_DIR / im["file_name"]
        dst = out_img_dir / im["file_name"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Fallback: hardlink (works across same filesystem)
            import shutil

            shutil.copy2(src, dst)

        # Write label file
        label_lines = []
        for a in anns:
            label_lines.extend(
                coco_to_yolo_label(a, im["width"], im["height"], coco_id_to_yolo_id)
            )
        n_polys += len(label_lines)
        lbl_path = out_lbl_dir / (Path(im["file_name"]).stem + ".txt")
        lbl_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

        n_total += 1
        if is_empty:
            n_empty += 1
        else:
            n_with_anns += 1

    print(
        f"{split_name}: wrote {n_total} images "
        f"({n_with_anns} with anns, {n_empty} empty, "
        f"{n_skipped} skipped). {n_polys} polygons total."
    )


def main(include_empty_train: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Load class list from the canonical record
    with open(COCO_DIR / "krones_classes.json") as f:
        krones_classes: list[str] = json.load(f)

    # Load one of the splits to get the COCO id -> name mapping (so we can
    # build coco_id -> yolo_id precisely)
    with open(COCO_DIR / "krones_train.json") as f:
        train = json.load(f)
    coco_id_to_name = {c["id"]: c["name"] for c in train["categories"]}
    name_to_yolo_id = {n: i for i, n in enumerate(krones_classes)}
    coco_id_to_yolo_id = {
        cid: name_to_yolo_id[name] for cid, name in coco_id_to_name.items()
    }

    print("Class mapping (COCO id -> YOLO id : name):")
    for cid in sorted(coco_id_to_yolo_id):
        print(f"  {cid:>2} -> {coco_id_to_yolo_id[cid]:>2} : {coco_id_to_name[cid]}")
    print()

    convert_split(
        "train",
        COCO_DIR / "krones_train.json",
        OUT_DIR / "images" / "train",
        OUT_DIR / "labels" / "train",
        coco_id_to_yolo_id,
        include_empty=include_empty_train,
    )
    convert_split(
        "val",
        COCO_DIR / "krones_val.json",
        OUT_DIR / "images" / "val",
        OUT_DIR / "labels" / "val",
        coco_id_to_yolo_id,
        include_empty=True,  # always include empty in val
    )

    # Write the Ultralytics dataset config
    yaml_path = OUT_DIR / "krones.yaml"
    yaml_lines = [
        "# Auto-generated by build_krones_yolo.py",
        f"path: {OUT_DIR}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(krones_classes)}",
        "names:",
    ]
    for i, name in enumerate(krones_classes):
        yaml_lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    print(f"\nWrote {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-empty-train",
        action="store_true",
        help="Include train images with no kept annotations "
        "(default: exclude, mirroring Detectron2 behavior)",
    )
    args = parser.parse_args()
    main(include_empty_train=args.include_empty_train)
