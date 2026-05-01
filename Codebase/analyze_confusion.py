"""
Diagnostic: per-class confusion matrix + per-image verdict failure breakdown.

For a given checkpoint, runs inference on val, then:
  (A) Builds an instance-level confusion matrix using greedy bbox-IoU matching
      (IoU >= 0.5). Rows = true class, columns = predicted class. The extra
      'BACKGROUND' row/column captures false negatives (true ann with no
      matching pred) and false positives (pred with no matching ann).
  (B) Builds a per-image verdict failure breakdown: every val image gets a
      row tagged correct / FP / FN with the most likely cause.
  (C) Writes a Markdown summary listing the top FN/FP causes and the worst
      off-diagonal confusion pairs.

Usage:
  python analyze_confusion.py \\
      --weights ../output/maskrcnn_r50_v0/model_final.pth \\
      --threshold 0.50 \\
      --out ../output/diagnostics/iter19999/

Outputs (in --out dir):
  confusion_matrix.csv   Raw counts, 23x23 (22 classes + BACKGROUND).
  confusion_heatmap.png  Row-normalized heatmap.
  verdict_failures.csv   Per-image: filename, true, pred, error, cause.
  failure_summary.md     Top-line summary, top 20 FN/FP causes, top 30
                         off-diagonal confusion pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectron2_train import (  # noqa: E402
    ALWAYS_FAULTY,
    CONDITIONAL_AGGREGATED,
    CONDITIONAL_PER_REGION,
    KRONES_CLASSES,
    NUM_CLASSES,
    verdict_from_polygons,
)

REPO_ROOT = Path('/home/dll0202/baksh/krones_2.0')
DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
COCO_VAL = REPO_ROOT / 'Dataset' / 'coco' / 'krones_val.json'
IMG_DIR = DATA_ROOT / 'train_images'
TRAIN_CSV = DATA_ROOT / 'train.csv'

IOU_THRESH = 0.5
BG_LABEL = 'BACKGROUND'


def build_predictor(weights_path: Path) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.TEST.DETECTIONS_PER_IMAGE = 60
    return DefaultPredictor(cfg)


def load_gt(coco_path: Path):
    with open(coco_path) as f:
        data = json.load(f)
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    img_id_to_filename = {im['id']: im['file_name'] for im in data['images']}
    gt_by_filename: dict[str, list[dict]] = defaultdict(list)
    for ann in data['annotations']:
        fn = img_id_to_filename[ann['image_id']]
        x, y, w, h = ann['bbox']
        gt_by_filename[fn].append({
            'class': cat_id_to_name[ann['category_id']],
            'bbox_xyxy': [x, y, x + w, y + h],
            'area': float(ann.get('area', w * h)),
        })
    # Include images that exist but have no kept anns
    for im in data['images']:
        gt_by_filename.setdefault(im['file_name'], [])
    return gt_by_filename


def greedy_match(pred_boxes_xyxy, gt_boxes_xyxy, iou_thresh: float):
    """Greedy 1-to-1 bbox-IoU matching, predictions assumed score-sorted desc.
    Returns list of (pred_idx, gt_idx_or_None)."""
    if not pred_boxes_xyxy:
        return []
    if not gt_boxes_xyxy:
        return [(i, None) for i in range(len(pred_boxes_xyxy))]
    iou = box_iou(
        torch.tensor(pred_boxes_xyxy, dtype=torch.float32),
        torch.tensor(gt_boxes_xyxy, dtype=torch.float32),
    ).numpy()
    matches: list[tuple[int, int | None]] = []
    used_gt: set[int] = set()
    for p in range(len(pred_boxes_xyxy)):
        best_iou, best_g = 0.0, None
        for g in range(len(gt_boxes_xyxy)):
            if g in used_gt:
                continue
            if iou[p, g] > best_iou:
                best_iou, best_g = iou[p, g], g
        if best_g is not None and best_iou >= iou_thresh:
            matches.append((p, best_g))
            used_gt.add(best_g)
        else:
            matches.append((p, None))
    return matches


def build_confusion(predictions, gt_by_filename) -> pd.DataFrame:
    classes = list(KRONES_CLASSES) + [BG_LABEL]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for filename, preds in predictions.items():
        gts = gt_by_filename.get(filename, [])
        preds_sorted = sorted(preds, key=lambda x: -x[1])
        pred_boxes = [p[3] for p in preds_sorted]
        pred_classes = [p[0] for p in preds_sorted]
        gt_boxes = [g['bbox_xyxy'] for g in gts]
        gt_classes = [g['class'] for g in gts]
        matches = greedy_match(pred_boxes, gt_boxes, IOU_THRESH)
        used_gt: set[int] = set()
        for p_idx, g_idx in matches:
            if g_idx is not None:
                counts[(gt_classes[g_idx], pred_classes[p_idx])] += 1
                used_gt.add(g_idx)
            else:
                counts[(BG_LABEL, pred_classes[p_idx])] += 1
        for g_idx in range(len(gts)):
            if g_idx not in used_gt:
                counts[(gt_classes[g_idx], BG_LABEL)] += 1
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    for (t, p), n in counts.items():
        matrix.loc[t, p] = n
    return matrix


def plot_heatmap(matrix: pd.DataFrame, out_path: Path):
    row_sums = matrix.sum(axis=1).replace(0, 1)
    norm = matrix.div(row_sums, axis=0)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Per-class confusion (row-normalized, bbox IoU >= {IOU_THRESH})')
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            n = matrix.iloc[i, j]
            if n > 0:
                color = 'white' if norm.iloc[i, j] > 0.5 else 'black'
                ax.text(j, i, str(int(n)), ha='center', va='center', fontsize=6, color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def diagnose_image(filename, preds, gts, true_label, threshold):
    pred_areas = defaultdict(list)
    for cname, score, area, _ in preds:
        if score < threshold:
            continue
        pred_areas[cname].append(area)
    pred_verdict = verdict_from_polygons(pred_areas)

    gt_classes = sorted({g['class'] for g in gts})
    pred_classes = sorted({c for c, s, a, b in preds if s >= threshold})

    if pred_verdict == true_label:
        return {
            'image_id': filename, 'true': true_label, 'pred': pred_verdict,
            'error': 'correct', 'cause': '',
            'gt_classes': ','.join(gt_classes),
            'pred_classes': ','.join(pred_classes),
        }
    if true_label == 1 and pred_verdict == 0:
        # FN — what GT class would have driven verdict?
        true_drivers_af = [c for c in gt_classes if c in ALWAYS_FAULTY]
        # Conditional aggregated: did GT carry these?
        ca_gt = [c for c in gt_classes if c in CONDITIONAL_AGGREGATED]
        cpr_gt = [c for c in gt_classes if c in CONDITIONAL_PER_REGION]
        if true_drivers_af:
            cause = 'missed_always_faulty:' + ','.join(true_drivers_af)
        elif ca_gt:
            cause = 'missed_aggregated:' + ','.join(ca_gt)
        elif cpr_gt:
            cause = 'missed_per_region:' + ','.join(cpr_gt)
        else:
            cause = 'missed_unknown'
        return {
            'image_id': filename, 'true': true_label, 'pred': pred_verdict,
            'error': 'FN', 'cause': cause,
            'gt_classes': ','.join(gt_classes),
            'pred_classes': ','.join(pred_classes),
        }
    # FP
    fired_af = [c for c in pred_classes if c in ALWAYS_FAULTY]
    fired_ca = [c for c in pred_classes if c in CONDITIONAL_AGGREGATED]
    fired_cpr = [c for c in pred_classes if c in CONDITIONAL_PER_REGION]
    if fired_af:
        cause = 'fired_always_faulty:' + ','.join(fired_af)
    elif fired_ca:
        cause = 'fired_aggregated:' + ','.join(fired_ca)
    elif fired_cpr:
        cause = 'fired_per_region:' + ','.join(fired_cpr)
    else:
        cause = 'fired_unknown'
    return {
        'image_id': filename, 'true': true_label, 'pred': pred_verdict,
        'error': 'FP', 'cause': cause,
        'gt_classes': ','.join(gt_classes),
        'pred_classes': ','.join(pred_classes),
    }


def verdict_breakdown(predictions, gt_by_filename, label_by_filename, threshold):
    rows = []
    for filename, preds in predictions.items():
        true_label = label_by_filename.get(filename)
        if true_label is None:
            continue
        gts = gt_by_filename.get(filename, [])
        rows.append(diagnose_image(filename, preds, gts, true_label, threshold))
    return pd.DataFrame(rows)


def write_summary(args, breakdown: pd.DataFrame, matrix: pd.DataFrame, out_path: Path):
    n = len(breakdown)
    n_correct = (breakdown['error'] == 'correct').sum()
    n_fp = (breakdown['error'] == 'FP').sum()
    n_fn = (breakdown['error'] == 'FN').sum()
    lines = [
        '# Diagnostic summary',
        f'**Checkpoint:** `{args.weights}`',
        f'**Threshold:** {args.threshold}',
        f'**Val images:** {n}',
        f'**Verdict accuracy:** {n_correct/n:.4f}',
        f'**False positives:** {n_fp}  ({100*n_fp/n:.2f}%)',
        f'**False negatives:** {n_fn}  ({100*n_fn/n:.2f}%)',
        '',
        '## Top FN causes (true=Not-Reusable, model predicted Reusable)',
    ]
    fn_causes = breakdown[breakdown['error'] == 'FN']['cause'].value_counts().head(20)
    for cause, count in fn_causes.items():
        lines.append(f'- {count:4d}  `{cause}`')
    lines.append('')
    lines.append('## Top FP causes (true=Reusable, model predicted Not-Reusable)')
    fp_causes = breakdown[breakdown['error'] == 'FP']['cause'].value_counts().head(20)
    for cause, count in fp_causes.items():
        lines.append(f'- {count:4d}  `{cause}`')
    lines.append('')
    lines.append('## Top off-diagonal confusion pairs (true → predicted)')
    classes = list(KRONES_CLASSES) + [BG_LABEL]
    pairs = []
    for t in classes:
        for p in classes:
            if t == p:
                continue
            n_pp = matrix.loc[t, p]
            if n_pp > 0:
                pairs.append((n_pp, t, p))
    pairs.sort(reverse=True)
    for n_pp, t, p in pairs[:30]:
        lines.append(f'- {n_pp:4d}  {t}  →  {p}')
    out_path.write_text('\n'.join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--threshold', type=float, default=0.50)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    gt_by_filename = load_gt(COCO_VAL)
    df_csv = pd.read_csv(TRAIN_CSV)
    label_by_filename = dict(zip(df_csv['image_id'], df_csv['target'].astype(int)))

    print(f'Loading checkpoint: {args.weights}')
    predictor = build_predictor(args.weights)

    print(f'Running inference on {len(gt_by_filename)} val images ...')
    predictions: dict[str, list[tuple[str, float, float, list[float]]]] = {}
    for filename in tqdm(sorted(gt_by_filename.keys()), unit='img'):
        img = cv2.imread(str(IMG_DIR / filename))
        if img is None:
            continue
        out = predictor(img)
        inst = out['instances'].to('cpu')
        dets = []
        if len(inst) > 0:
            scores = inst.scores.numpy()
            classes = inst.pred_classes.numpy()
            masks = inst.pred_masks.numpy()
            boxes = inst.pred_boxes.tensor.numpy()
            for i in range(len(inst)):
                cname = KRONES_CLASSES[int(classes[i])]
                dets.append((cname, float(scores[i]), float(masks[i].sum()),
                             boxes[i].tolist()))
        predictions[filename] = dets

    print(f'\nBuilding confusion matrix at threshold {args.threshold} ...')
    filtered = {fn: [d for d in dets if d[1] >= args.threshold]
                for fn, dets in predictions.items()}
    matrix = build_confusion(filtered, gt_by_filename)
    matrix.to_csv(args.out / 'confusion_matrix.csv')
    plot_heatmap(matrix, args.out / 'confusion_heatmap.png')
    print(f'  Wrote {args.out/"confusion_matrix.csv"}')
    print(f'  Wrote {args.out/"confusion_heatmap.png"}')

    print(f'\nBuilding verdict failure breakdown ...')
    breakdown = verdict_breakdown(predictions, gt_by_filename, label_by_filename,
                                  args.threshold)
    breakdown.to_csv(args.out / 'verdict_failures.csv', index=False)
    print(f'  Wrote {args.out/"verdict_failures.csv"}')

    summary_lines = write_summary(args, breakdown, matrix, args.out / 'failure_summary.md')
    print(f'  Wrote {args.out/"failure_summary.md"}')
    print('\n=== Summary ===')
    print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
