"""
Sweep score thresholds on the val set to pick the best for verdict F1 / acc.

Strategy:
  1. Run inference on val ONCE with a low score floor (0.05).
  2. Cache per-detection (class_name, score, mask_pixel_count) per image.
  3. Sweep thresholds in pure Python — extremely fast since we only re-aggregate.

Usage:
  python tune_threshold.py --weights ../output/maskrcnn_r50_v0/model_0013999.pth

  # Reuse cached predictions (faster on repeated runs):
  python tune_threshold.py --weights ../output/maskrcnn_r50_v0/model_0013999.pth \\
      --cache /tmp/val_preds_iter13999.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
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
COCO_DIR = REPO_ROOT / 'Dataset' / 'coco'
IMG_DIR = str(DATA_ROOT / 'train_images')
TRAIN_CSV = DATA_ROOT / 'train.csv'

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def build_predictor(weights_path: Path) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    ))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.TEST.DETECTIONS_PER_IMAGE = 60
    return DefaultPredictor(cfg)


def collect_predictions(predictor, records, label_by_filename):
    """Run model on each val image; return dict[filename -> list[(cname, score, area)]]."""
    cache: dict[str, list[tuple[str, float, float]]] = {}
    truth: dict[str, int] = {}
    for rec in tqdm(records, unit='img'):
        fn = Path(rec['file_name']).name
        if fn not in label_by_filename:
            continue
        img = cv2.imread(rec['file_name'])
        if img is None:
            continue
        out = predictor(img)
        inst = out['instances'].to('cpu')
        dets: list[tuple[str, float, float]] = []
        if len(inst) > 0:
            scores = inst.scores.numpy()
            classes = inst.pred_classes.numpy()
            masks = inst.pred_masks.numpy()
            for i in range(len(inst)):
                cname = KRONES_CLASSES[int(classes[i])]
                dets.append((cname, float(scores[i]), float(masks[i].sum())))
        cache[fn] = dets
        truth[fn] = label_by_filename[fn]
    return cache, truth


def evaluate_at_threshold(cache, truth, thr: float):
    cm = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for fn, dets in cache.items():
        true = truth[fn]
        areas: dict[str, list[float]] = defaultdict(list)
        for cname, score, area in dets:
            if score < thr:
                continue
            areas[cname].append(area)
        pred = verdict_from_polygons(areas)
        cm[(true, pred)] += 1
    tn, fp = cm[(0, 0)], cm[(0, 1)]
    fn_, tp = cm[(1, 0)], cm[(1, 1)]
    n = tn + fp + fn_ + tp
    acc = (tn + tp) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'tn': tn, 'fp': fp, 'fn': fn_, 'tp': tp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=None,
                        help='Pickle path to cache val preds (skip inference if exists)')
    args = parser.parse_args()

    name = 'krones_val'
    if name not in DatasetCatalog.list():
        register_coco_instances(name, {}, str(COCO_DIR / 'krones_val.json'), IMG_DIR)
    records = DatasetCatalog.get(name)

    df = pd.read_csv(TRAIN_CSV)
    label_by_filename = dict(zip(df['image_id'].astype(str), df['target'].astype(int)))

    if args.cache and args.cache.exists():
        print(f'Loading cached predictions from {args.cache}')
        with open(args.cache, 'rb') as f:
            cache, truth = pickle.load(f)
    else:
        print(f'Loading checkpoint: {args.weights}')
        predictor = build_predictor(args.weights)
        print(f'Running inference on {len(records)} val records ...')
        cache, truth = collect_predictions(predictor, records, label_by_filename)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cache, 'wb') as f:
                pickle.dump((cache, truth), f)
            print(f'Cached predictions to {args.cache}')

    print(f'\nVal images evaluated: {len(cache)}')
    print(f'{"thresh":>7} | {"acc":>6} | {"prec":>6} | {"recall":>6} | {"f1":>6} | '
          f'{"tn":>5} {"fp":>5} {"fn":>5} {"tp":>5}')
    print('-' * 76)
    rows = []
    for thr in THRESHOLDS:
        r = evaluate_at_threshold(cache, truth, thr)
        rows.append((thr, r))
        print(f'  {thr:.2f} | {r["acc"]:.4f} | {r["prec"]:.4f} | {r["rec"]:.4f} '
              f'| {r["f1"]:.4f} | {r["tn"]:5d} {r["fp"]:5d} {r["fn"]:5d} {r["tp"]:5d}')

    best_acc_thr, best_acc = max(((t, r['acc']) for t, r in rows), key=lambda x: x[1])
    best_f1_thr, best_f1 = max(((t, r['f1']) for t, r in rows), key=lambda x: x[1])
    print()
    print(f'Best by accuracy: threshold={best_acc_thr}  acc={best_acc:.4f}')
    print(f'Best by F1:       threshold={best_f1_thr}   f1={best_f1:.4f}')


if __name__ == '__main__':
    main()
