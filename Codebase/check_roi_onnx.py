"""
Sanity-check the ROI detector ONNX export.

Picks one image from the original train set, runs the ONNX through
infer_to_submission.predict_roi(), and compares the predicted (padded) crop
box to the ground-truth ROI from train_annotations.json.

Prints:
  - ONNX input/output shapes
  - Top-5 raw detections (so you can eyeball the layout: NMS vs raw, coord units)
  - Predicted crop box vs ground-truth ROI bbox + 50 px padding
  - IoU of the two boxes

Usage:
  python check_roi_onnx.py
  python check_roi_onnx.py --image-id <some_id>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_to_submission import (  # noqa: E402
    REPO_ROOT,
    ROI_PADDING,
    build_roi_session,
    letterbox,
    predict_roi,
)

DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
ANN_FILE = DATA_ROOT / 'train_annotations.json'
SRC_IMG_DIR = DATA_ROOT / 'train_images'
DEFAULT_ONNX = REPO_ROOT / 'output' / 'ROI_Detector' / 'best.onnx'


def gt_roi_box(image_id: str) -> tuple[int, int, int, int] | None:
    with open(ANN_FILE) as f:
        coco = json.load(f)
    name_to_id = {im['file_name']: im['id'] for im in coco['images']}
    img_by_id = {im['id']: im for im in coco['images']}
    if image_id not in name_to_id:
        return None
    img_id = name_to_id[image_id]
    im = img_by_id[img_id]
    W, H = im['width'], im['height']
    roi_cat_id = next(c['id'] for c in coco['categories'] if c['name'] == 'Roi')
    for ann in coco['annotations']:
        if ann['image_id'] == img_id and ann['category_id'] == roi_cat_id:
            rx, ry, rw, rh = ann['bbox']
            cx0 = max(0, int(round(rx - ROI_PADDING)))
            cy0 = max(0, int(round(ry - ROI_PADDING)))
            cx1 = min(W, int(round(rx + rw + ROI_PADDING)))
            cy1 = min(H, int(round(ry + rh + ROI_PADDING)))
            return cx0, cy0, cx1, cy1
    return None


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    a_area = (ax1 - ax0) * (ay1 - ay0)
    b_area = (bx1 - bx0) * (by1 - by0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=Path, default=DEFAULT_ONNX)
    parser.add_argument('--image-id', type=str, default=None,
                        help='Filename to test (default: first one in annotations)')
    args = parser.parse_args()

    if not args.onnx.exists():
        sys.exit(f'ERROR: ONNX not found: {args.onnx}')

    session = build_roi_session(args.onnx)
    inp = session.get_inputs()[0]
    outs = session.get_outputs()
    print(f'Input:  name={inp.name}  shape={inp.shape}  dtype={inp.type}')
    for o in outs:
        print(f'Output: name={o.name}  shape={o.shape}  dtype={o.type}')
    print()

    # Pick test image
    if args.image_id is None:
        with open(ANN_FILE) as f:
            coco = json.load(f)
        args.image_id = coco['images'][0]['file_name']
    img_path = SRC_IMG_DIR / args.image_id
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f'ERROR: could not read {img_path}')
    H, W = img.shape[:2]
    print(f'Image: {args.image_id}  ({W}x{H})')

    # Raw output dump (top-5 by score, assuming column 4 is score)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lb, scale, pad_x, pad_y = letterbox(img_rgb)
    x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    raw = session.run(None, {inp.name: x})[0]
    print(f'\nRaw output shape: {raw.shape}')
    print(f'Letterbox: scale={scale:.4f}, pad=({pad_x}, {pad_y})')

    if raw.ndim == 3 and raw.shape[-1] == 6:
        dets = raw[0]
        # Top-5 by what we believe is the score column
        order = np.argsort(-dets[:, 4])[:5]
        print('\nTop-5 detections (assuming x1,y1,x2,y2,score,cls in 640-letterbox coords):')
        for i, idx in enumerate(order):
            d = dets[idx]
            print(f'  [{i}] x1={d[0]:8.2f} y1={d[1]:8.2f} '
                  f'x2={d[2]:8.2f} y2={d[3]:8.2f} '
                  f'score={d[4]:.4f} cls={d[5]:.0f}')
    else:
        print('\nUnexpected output layout — print first row for inspection:')
        print(raw.reshape(-1, raw.shape[-1])[:3])

    # Run the actual function and compare to GT
    pred_box = predict_roi(session, img)
    gt_box = gt_roi_box(args.image_id)

    print()
    if pred_box is None:
        print('predict_roi() returned None — script will fall back to full image.')
    else:
        print(f'Predicted padded crop box: {pred_box}  '
              f'({pred_box[2]-pred_box[0]}x{pred_box[3]-pred_box[1]})')
    if gt_box is None:
        print('No GT ROI annotation found for this image.')
    else:
        print(f'GT padded crop box:        {gt_box}  '
              f'({gt_box[2]-gt_box[0]}x{gt_box[3]-gt_box[1]})')
    if pred_box is not None and gt_box is not None:
        print(f'IoU(pred, gt) = {iou(pred_box, gt_box):.4f}')
        print('  Expect IoU > 0.95 if the ONNX layout assumption is correct.')


if __name__ == '__main__':
    main()
