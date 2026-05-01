"""
Run inference on the Krones test set and produce a Kaggle submission CSV.

Imports the verdict rule and class list from detectron2_train.py so there's
one source of truth.

Usage examples:
  python infer_to_submission.py \\
      --weights ../output/maskrcnn_r50_v0/model_0013999.pth \\
      --threshold 0.5 \\
      --out ../submissions/sub_iter13999_t05.csv

  python infer_to_submission.py \\
      --weights ../output/maskrcnn_r50_v0/model_0011999.pth \\
      --threshold 0.40 \\
      --out ../submissions/sub_iter11999_t04.csv
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

REPO_ROOT = Path('/home/dll0202/baksh/krones_2.0')
DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
TEST_IMG_DIR = DATA_ROOT / 'test_images'
SAMPLE_SUB = DATA_ROOT / 'sample_submission.csv'


def build_predictor(weights_path: Path) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    ))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # keep low so verdict logic chooses
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.TEST.DETECTIONS_PER_IMAGE = 60
    return DefaultPredictor(cfg)


def predict_one(predictor: DefaultPredictor,
                image_path: Path,
                threshold: float) -> int:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f'WARN: could not read {image_path}; predicting Reusable=0')
        return 0
    out = predictor(img)
    inst = out['instances'].to('cpu')
    if len(inst) == 0:
        return 0
    scores = inst.scores.numpy()
    classes = inst.pred_classes.numpy()
    masks = inst.pred_masks.numpy()
    areas: dict[str, list[float]] = defaultdict(list)
    for i in range(len(inst)):
        if scores[i] < threshold:
            continue
        cname = KRONES_CLASSES[int(classes[i])]
        areas[cname].append(float(masks[i].sum()))
    return verdict_from_polygons(areas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True,
                        help='Path to a .pth checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum detection score to count for verdict')
    parser.add_argument('--out', type=Path, required=True,
                        help='Output submission.csv path')
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f'ERROR: checkpoint not found: {args.weights}')
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(SAMPLE_SUB)
    print(f'Loaded {len(sub)} test image_ids from {SAMPLE_SUB}')
    print(f'Loading checkpoint: {args.weights}')
    predictor = build_predictor(args.weights)
    print(f'Running inference (threshold={args.threshold}) ...')

    verdicts = []
    for img_id in tqdm(sub['image_id'], unit='img'):
        verdicts.append(predict_one(predictor, TEST_IMG_DIR / img_id, args.threshold))

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
