"""
Run YOLO26-seg inference on Krones test set, apply the verdict rule,
and produce a Kaggle submission CSV.

Imports the verdict rule from detectron2_train.py to keep one source of truth.

Usage:
  python yolo_infer_to_submission.py \\
      --weights ../output/yolo26m_v1/weights/best.pt \\
      --threshold 0.50 \\
      --imgsz 1024 \\
      --out ../submissions/sub_yolo26m_v1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectron2_train import (  # noqa: E402
    KRONES_CLASSES,
    verdict_from_polygons,
)

REPO_ROOT = Path('/home/dll0202/baksh/krones_2.0')
DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
TEST_IMG_DIR = DATA_ROOT / 'test_images'
SAMPLE_SUB = DATA_ROOT / 'sample_submission.csv'


def predict_verdict(model, image_path: Path, threshold: float, imgsz: int) -> int:
    """Run YOLO on one image, apply the verdict rule, return 0 or 1."""
    # YOLO Results: list of length 1 for a single image
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=0.05,        # keep low; verdict rule uses --threshold
        iou=0.5,
        device=0,
        verbose=False,
        save=False,
    )
    res = results[0]

    if res.masks is None or len(res.boxes) == 0:
        return 0

    # Per-detection (class_name, score, mask_pixel_area)
    scores = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    masks = res.masks.data.cpu().numpy()  # (N, H, W) bool/float

    areas: dict[str, list[float]] = defaultdict(list)
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue
        cname = KRONES_CLASSES[int(classes[i])]
        areas[cname].append(float(masks[i].sum()))

    return verdict_from_polygons(areas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True,
                        help='Path to YOLO .pt checkpoint (e.g. best.pt)')
    parser.add_argument('--threshold', type=float, default=0.50,
                        help='Min detection score for verdict aggregation')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Inference image size (must match training)')
    parser.add_argument('--out', type=Path, required=True,
                        help='Output submission.csv path')
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f'ERROR: checkpoint not found: {args.weights}')
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(SAMPLE_SUB)
    print(f'Loaded {len(sub)} test image_ids from {SAMPLE_SUB}')
    print(f'Loading checkpoint: {args.weights}')
    model = YOLO(str(args.weights))
    print(f'Running inference (imgsz={args.imgsz}, threshold={args.threshold}) ...')

    verdicts = []
    for img_id in tqdm(sub['image_id'], unit='img'):
        verdicts.append(
            predict_verdict(model, TEST_IMG_DIR / img_id, args.threshold, args.imgsz)
        )

    out_df = pd.DataFrame({'image_id': sub['image_id'], 'target': verdicts})
    out_df.to_csv(args.out, index=False)

    n0 = sum(1 for v in verdicts if v == 0)
    n1 = sum(1 for v in verdicts if v == 1)
    print(f'\nWrote {args.out}')
    print(f'  total: {len(verdicts)}')
    print(f'  Reusable (0):       {n0} ({100*n0/len(verdicts):.1f}%)')
    print(f'  Not Reusable (1):   {n1} ({100*n1/len(verdicts):.1f}%)')


if __name__ == '__main__':
    main()
