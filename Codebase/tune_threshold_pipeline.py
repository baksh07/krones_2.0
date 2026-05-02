"""
Run the FULL inference pipeline (ROI ONNX → crop → defect Mask R-CNN) on the
val split, then sweep thresholds. This is the LB-predictive evaluation —
unlike the in-trainer VerdictEvaluator which uses GT crops, this one exercises
the YOLO ROI detector exactly like submission time.

Pipeline:
  1. Read val image filenames from coco_cropped/krones_val.json.
  2. For each val image, load the ORIGINAL (uncropped) image from train_images.
  3. ROI ONNX → padded crop box → crop.
  4. Defect Mask R-CNN on cropped image, score floor 0.05.
  5. Cache (cname, score, mask_area) per image.
  6. Sweep thresholds in pure Python.

Usage:
  python tune_threshold_pipeline.py \\
      --weights ../output/maskrcnn_r50_cropped/model_0008999.pth \\
      --cache /tmp/val_pipeline_iter8999.pkl

  # Reuse cache:
  python tune_threshold_pipeline.py \\
      --weights ../output/maskrcnn_r50_cropped/model_0008999.pth \\
      --cache /tmp/val_pipeline_iter8999.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
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
from detectron2_train_cropped import (  # noqa: E402
    KRONES_CLASSES,
    NUM_CLASSES,
    verdict_from_polygons,
)
from infer_to_submission import build_roi_session, predict_roi  # noqa: E402

REPO_ROOT = Path('/home/dll0202/baksh/krones_2.0')
DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
COCO_VAL = REPO_ROOT / 'Dataset' / 'coco_cropped' / 'krones_val.json'
ORIG_IMG_DIR = DATA_ROOT / 'train_images'
TRAIN_CSV = DATA_ROOT / 'train.csv'
DEFAULT_ONNX = REPO_ROOT / 'output' / 'ROI_Detector' / 'best.onnx'

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def build_predictor(weights_path: Path,
                    min_size_test: int = 768,
                    max_size_test: int = 1024) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    ))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.INPUT.MIN_SIZE_TEST = min_size_test
    cfg.INPUT.MAX_SIZE_TEST = max_size_test
    cfg.TEST.DETECTIONS_PER_IMAGE = 60
    return DefaultPredictor(cfg)


def collect_predictions(predictor, roi_session, val_filenames, label_by_filename):
    cache: dict[str, list[tuple[str, float, float]]] = {}
    truth: dict[str, int] = {}
    n_no_roi = 0
    for fn in tqdm(val_filenames, unit='img'):
        if fn not in label_by_filename:
            continue
        img = cv2.imread(str(ORIG_IMG_DIR / fn))
        if img is None:
            continue
        crop_box = predict_roi(roi_session, img)
        if crop_box is None:
            n_no_roi += 1
            cropped = img
        else:
            cx0, cy0, cx1, cy1 = crop_box
            cropped = img[cy0:cy1, cx0:cx1]

        out = predictor(cropped)
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
    print(f'ROI fallback (full image): {n_no_roi} / {len(cache)}')
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
    parser.add_argument('--weights', type=Path, required=True,
                        help='Path to defect Mask R-CNN .pth checkpoint')
    parser.add_argument('--roi-onnx', type=Path, default=DEFAULT_ONNX)
    parser.add_argument('--cache', type=Path, default=None,
                        help='Pickle path to cache val preds (skip inference if exists)')
    parser.add_argument('--min-size-test', type=int, default=768,
                        help='Defect predictor MIN_SIZE_TEST')
    parser.add_argument('--max-size-test', type=int, default=1024)
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f'ERROR: defect checkpoint not found: {args.weights}')
    if not args.roi_onnx.exists():
        sys.exit(f'ERROR: ROI ONNX not found: {args.roi_onnx}')

    df = pd.read_csv(TRAIN_CSV)
    label_by_filename = dict(zip(df['image_id'].astype(str), df['target'].astype(int)))

    with open(COCO_VAL) as f:
        val_coco = json.load(f)
    val_filenames = [im['file_name'] for im in val_coco['images']]
    print(f'Val split: {len(val_filenames)} images from {COCO_VAL.name}')

    if args.cache and args.cache.exists():
        print(f'Loading cached predictions from {args.cache}')
        with open(args.cache, 'rb') as f:
            cache, truth = pickle.load(f)
    else:
        print(f'Loading ROI detector: {args.roi_onnx}')
        roi_session = build_roi_session(args.roi_onnx)
        print(f'Loading defect checkpoint: {args.weights}')
        predictor = build_predictor(
            args.weights, args.min_size_test, args.max_size_test
        )
        print(f'Running pipeline on {len(val_filenames)} val images '
              f'(MIN_SIZE_TEST={args.min_size_test}, MAX_SIZE_TEST={args.max_size_test}) ...')
        cache, truth = collect_predictions(
            predictor, roi_session, val_filenames, label_by_filename
        )
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
    print()
    print(f'Reference: in-trainer val (GT crops) was 0.9355 at iter 8999.')
    print(f'           v0 LB = 0.94310, cropped iter 8999 LB (t=0.5) = 0.94258.')


if __name__ == '__main__':
    main()
