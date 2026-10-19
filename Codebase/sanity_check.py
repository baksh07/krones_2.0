"""
Visual sanity check on the converter output.

Loads the COCO splits via Detectron2's loader (the same path the trainer uses),
draws annotations on a sample of images, writes PNGs to Dataset/viz_out/sanity/.

Goals:
  - Confirm Detectron2 parses the JSON without errors.
  - Confirm class names + polygons render correctly on the original images.
  - Inspect rare-class images explicitly (Insect, Straw, Circlip, Foil, Liquid,
    Paint residue) since they have only 3-8 examples in val.

Run:
  python sanity_check.py [--per-split 10] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

from project_paths import DATA_ROOT, REPO_ROOT

COCO_DIR = REPO_ROOT / "Dataset" / "coco"
IMG_DIR = str(DATA_ROOT / "train_images")
OUT_DIR = REPO_ROOT / "Dataset" / "viz_out" / "sanity"

RARE_CLASSES = [
    "Insect",
    "Straw",
    "Circlip",
    "Foil / Semitransparent",
    "Liquid",
    "Paint residue",
]


def register():
    for split in ("train", "val"):
        name = f"krones_{split}"
        if name in DatasetCatalog.list():
            continue
        register_coco_instances(
            name, {}, str(COCO_DIR / f"krones_{split}.json"), IMG_DIR
        )


def draw_one(rec, meta, out_path: Path):
    img = cv2.imread(rec["file_name"])
    if img is None:
        print(f"  WARN: failed to read {rec['file_name']}")
        return False
    vis = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.7)
    drawn = vis.draw_dataset_dict(rec)
    cv2.imwrite(str(out_path), drawn.get_image()[:, :, ::-1])
    return True


def sample_split(split: str, per_split: int, rng: random.Random):
    name = f"krones_{split}"
    records = DatasetCatalog.get(name)
    meta = MetadataCatalog.get(name)
    with_anns = [r for r in records if r.get("annotations")]
    print(f"{name}: {len(records)} records, {len(with_anns)} with kept anns")
    sample = rng.sample(with_anns, min(per_split, len(with_anns)))
    n = 0
    for i, rec in enumerate(sample):
        out_path = OUT_DIR / f"{split}_{i:02d}_id{rec['image_id']}.png"
        if draw_one(rec, meta, out_path):
            n += 1
    print(f"  wrote {n} samples to {OUT_DIR}/")


def sample_rare(rng: random.Random, per_class: int = 3):
    name = "krones_train"
    records = DatasetCatalog.get(name)
    meta = MetadataCatalog.get(name)
    name_to_id = {n: i for i, n in enumerate(meta.thing_classes)}
    rare_dir = OUT_DIR / "rare"
    rare_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRare-class samples (up to {per_class} per class):")
    for cname in RARE_CLASSES:
        cid = name_to_id.get(cname)
        if cid is None:
            print(f"  {cname}: NOT FOUND in metadata (mapping mismatch?)")
            continue
        matching = [
            r
            for r in records
            if any(a["category_id"] == cid for a in r.get("annotations", []))
        ]
        if not matching:
            print(f"  {cname}: 0 matching images")
            continue
        sample = rng.sample(matching, min(per_class, len(matching)))
        safe = cname.replace(" / ", "_").replace(" ", "_")
        n = 0
        for i, rec in enumerate(sample):
            out_path = rare_dir / f"{safe}_{i:02d}_id{rec['image_id']}.png"
            if draw_one(rec, meta, out_path):
                n += 1
        print(f"  {cname:30s} {n}/{len(matching)} written")


def main(per_split: int, seed: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    register()
    rng = random.Random(seed)
    sample_split("train", per_split, rng)
    sample_split("val", per_split, rng)
    sample_rare(rng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(per_split=args.per_split, seed=args.seed)
