"""
Ensemble two checkpoints by averaging detection scores per (image, class).

Strategy:
  - For each test image, run inference with both models.
  - Group detections by class name within each model.
  - For each class present in either model, take the union of detections;
    detections seen only by one model get their score halved (soft penalty).
  - Apply verdict rule with threshold 0.50 on the merged detection list.

This is a "soft" ensemble — agreement boosts confidence, single-model hits
contribute but at reduced weight.

Usage:
  python infer_ensemble.py \\
      --weights1 ../output/maskrcnn_r50_v0/model_0015999.pth \\
      --weights2 ../output/maskrcnn_r50_v0/model_final.pth \\
      --threshold 0.50 \\
      --out ../submissions/sub04_ensemble_15999plus19999_t050.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectron2_train import (  # noqa: E402
    KRONES_CLASSES,
    NUM_CLASSES,
    verdict_from_polygons,
)

from project_paths import DATA_ROOT

TEST_IMG_DIR = DATA_ROOT / "test_images"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

UNMATCHED_PENALTY = 0.5  # detections seen by only one model get score * 0.5


def build_predictor(weights_path: Path) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.TEST.DETECTIONS_PER_IMAGE = 60
    return DefaultPredictor(cfg)


def detections_from_output(
    out, class_names: list[str]
) -> list[tuple[str, float, float]]:
    """Return list of (class_name, score, mask_pixel_count) for one image's predictions."""
    inst = out["instances"].to("cpu")
    if len(inst) == 0:
        return []
    scores = inst.scores.numpy()
    classes = inst.pred_classes.numpy()
    masks = inst.pred_masks.numpy()
    dets = []
    for i in range(len(inst)):
        cname = class_names[int(classes[i])]
        dets.append((cname, float(scores[i]), float(masks[i].sum())))
    return dets


def merge_detections(dets1, dets2):
    """Score-average ensemble:
    - Group detections by class within each model.
    - Sort each group by score descending.
    - Pair the top-K detections per class across the two models, averaging score and area.
    - Excess unmatched detections in either model are kept with score * UNMATCHED_PENALTY.
    """
    by_class_1: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_class_2: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c, s, a in dets1:
        by_class_1[c].append((s, a))
    for c, s, a in dets2:
        by_class_2[c].append((s, a))
    for c in by_class_1:
        by_class_1[c].sort(reverse=True)
    for c in by_class_2:
        by_class_2[c].sort(reverse=True)

    merged: list[tuple[str, float, float]] = []
    all_classes = set(by_class_1) | set(by_class_2)
    for c in all_classes:
        list1 = by_class_1.get(c, [])
        list2 = by_class_2.get(c, [])
        k = min(len(list1), len(list2))
        for i in range(k):
            s1, a1 = list1[i]
            s2, a2 = list2[i]
            merged.append((c, (s1 + s2) / 2.0, (a1 + a2) / 2.0))
        for i in range(k, len(list1)):
            s, a = list1[i]
            merged.append((c, s * UNMATCHED_PENALTY, a))
        for i in range(k, len(list2)):
            s, a = list2[i]
            merged.append((c, s * UNMATCHED_PENALTY, a))
    return merged


def predict_verdict_ensemble(p1, p2, image_path: Path, threshold: float) -> int:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"WARN: could not read {image_path}; predicting Reusable=0")
        return 0
    out1 = p1(img)
    out2 = p2(img)
    dets1 = detections_from_output(out1, KRONES_CLASSES)
    dets2 = detections_from_output(out2, KRONES_CLASSES)
    merged = merge_detections(dets1, dets2)

    areas: dict[str, list[float]] = defaultdict(list)
    for cname, score, area in merged:
        if score < threshold:
            continue
        areas[cname].append(area)
    return verdict_from_polygons(areas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights1", type=Path, required=True)
    parser.add_argument("--weights2", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.weights1.exists():
        sys.exit(f"ERROR: weights1 not found: {args.weights1}")
    if not args.weights2.exists():
        sys.exit(f"ERROR: weights2 not found: {args.weights2}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(SAMPLE_SUB)
    print(f"Loaded {len(sub)} test image_ids")
    print(f"Loading model 1: {args.weights1}")
    p1 = build_predictor(args.weights1)
    print(f"Loading model 2: {args.weights2}")
    p2 = build_predictor(args.weights2)
    print(
        f"Running ensemble inference (threshold={args.threshold}, "
        f"unmatched_penalty={UNMATCHED_PENALTY}) ..."
    )

    verdicts = []
    for img_id in tqdm(sub["image_id"], unit="img"):
        verdicts.append(
            predict_verdict_ensemble(p1, p2, TEST_IMG_DIR / img_id, args.threshold)
        )

    out_df = pd.DataFrame({"image_id": sub["image_id"], "target": verdicts})
    out_df.to_csv(args.out, index=False)

    n0 = sum(1 for v in verdicts if v == 0)
    n1 = sum(1 for v in verdicts if v == 1)
    print(f"\nWrote {args.out}")
    print(f"  total: {len(verdicts)}")
    print(f"  Reusable (0):       {n0} ({100 * n0 / len(verdicts):.1f}%)")
    print(f"  Not Reusable (1):   {n1} ({100 * n1 / len(verdicts):.1f}%)")


if __name__ == "__main__":
    main()
