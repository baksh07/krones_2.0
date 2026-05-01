"""
Evaluate a trained YOLO26-seg checkpoint on the Krones val split using the
same verdict rule as v0/v2 — gives a directly-comparable number to
Detectron2 v0's 0.9295 verdict acc.

Strategy mirrors tune_threshold.py:
  1. Run inference on val ONCE with a low score floor (0.05).
  2. Cache per-detection (class_name, score, mask_pixel_count) per image.
  3. Sweep thresholds in pure Python — extremely fast since we only re-aggregate.

Usage:
  python yolo_eval_val.py \\
      --weights ../output/yolo26m_v1-2/weights/best.pt \\
      --imgsz 1024

  # Reuse cached predictions on subsequent runs:
  python yolo_eval_val.py \\
      --weights ../output/yolo26m_v1-2/weights/best.pt \\
      --imgsz 1024 \\
      --cache /tmp/yolo_v1_val_preds.pkl
"""

from __future__ import annotations

import argparse
import pickle
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
TRAIN_CSV = DATA_ROOT / 'train.csv'
VAL_IMG_DIR = REPO_ROOT / 'Dataset' / 'yolo' / 'images' / 'val'

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def collect_predictions(model, image_paths, imgsz):
    """Run YOLO once per val image; return dict[filename -> list[(cname, score, area)]]."""
    cache: dict[str, list[tuple[str, float, float]]] = {}
    for img_path in tqdm(image_paths, unit='img'):
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=0.05,
            iou=0.5,
            device=0,
            verbose=False,
            save=False,
        )
        res = results[0]
        dets: list[tuple[str, float, float]] = []
        if res.masks is not None and len(res.boxes) > 0:
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            masks = res.masks.data.cpu().numpy()
            for i in range(len(scores)):
                cname = KRONES_CLASSES[int(classes[i])]
                dets.append((cname, float(scores[i]), float(masks[i].sum())))
        cache[img_path.name] = dets
    return cache


def evaluate_at_threshold(cache, truth, thr: float):
    cm = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for fn, dets in cache.items():
        if fn not in truth:
            continue
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
            'tn': tn, 'fp': fp, 'fn': fn_, 'tp': tp, 'n': n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True,
                        help='Path to YOLO .pt checkpoint (e.g. best.pt)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Inference image size (must match training)')
    parser.add_argument('--cache', type=Path, default=None,
                        help='Pickle path to cache val preds (skip inference if exists)')
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f'ERROR: checkpoint not found: {args.weights}')
    if not VAL_IMG_DIR.exists():
        sys.exit(f'ERROR: val image dir not found: {VAL_IMG_DIR}\n'
                 f'Run Dataset/build_krones_yolo.py first.')

    df = pd.read_csv(TRAIN_CSV)
    truth: dict[str, int] = dict(
        zip(df['image_id'].astype(str), df['target'].astype(int))
    )

    if args.cache and args.cache.exists():
        print(f'Loading cached predictions from {args.cache}')
        with open(args.cache, 'rb') as f:
            cache = pickle.load(f)
    else:
        image_paths = sorted(VAL_IMG_DIR.glob('*.png'))
        print(f'Loading checkpoint: {args.weights}')
        model = YOLO(str(args.weights))
        print(f'Running inference on {len(image_paths)} val images at imgsz={args.imgsz} ...')
        cache = collect_predictions(model, image_paths, args.imgsz)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cache, 'wb') as f:
                pickle.dump(cache, f)
            print(f'Cached predictions to {args.cache}')

    # Filter cache to only images that have a label in train.csv
    cache_with_truth = {fn: dets for fn, dets in cache.items() if fn in truth}
    print(f'\nEvaluating verdict on {len(cache_with_truth)} val images '
          f'(of {len(cache)} cached, {len(truth)} truth labels)')
    print()
    print(f'{"thresh":>7} | {"acc":>6} | {"prec":>6} | {"recall":>6} | {"f1":>6} | '
          f'{"tn":>5} {"fp":>5} {"fn":>5} {"tp":>5}')
    print('-' * 76)
    rows = []
    for thr in THRESHOLDS:
        r = evaluate_at_threshold(cache_with_truth, truth, thr)
        rows.append((thr, r))
        print(f'  {thr:.2f} | {r["acc"]:.4f} | {r["prec"]:.4f} | {r["rec"]:.4f} '
              f'| {r["f1"]:.4f} | {r["tn"]:5d} {r["fp"]:5d} {r["fn"]:5d} {r["tp"]:5d}')

    best_acc_thr, best_acc = max(((t, r['acc']) for t, r in rows), key=lambda x: x[1])
    best_f1_thr, best_f1 = max(((t, r['f1']) for t, r in rows), key=lambda x: x[1])
    print()
    print(f'Best by accuracy: threshold={best_acc_thr}  acc={best_acc:.4f}')
    print(f'Best by F1:       threshold={best_f1_thr}   f1={best_f1:.4f}')
    print()
    print(f'Reference (Detectron2 val): v0 best=0.9318 (iter 15999), '
          f'v0 final=0.9295, v2 final=0.9154')


if __name__ == '__main__':
    main()
