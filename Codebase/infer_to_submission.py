"""
Run inference on the Krones test set and produce a Kaggle submission CSV.

Pipeline (matches the cropped training pipeline):
  1. ROI detector (YOLO26n ONNX) predicts bottle-base bbox on the full image.
  2. Crop to that bbox + 50 px padding (clipped to image bounds), exactly as
     Dataset/build_krones_coco_cropped.py does for training.
  3. Defect Mask R-CNN (Detectron2 .pth) runs on the cropped image.
  4. Verdict rule aggregates per-class mask areas.

Imports the verdict rule and class list from detectron2_train.py so there's
one source of truth.

Usage:
  python infer_to_submission.py \\
      --weights ../output/maskrcnn_r50_cropped/model_0008999.pth \\
      --roi-onnx ../output/ROI_Detector/best.onnx \\
      --threshold 0.5 \\
      --out ../submissions/sub_cropped_iter8999_t05.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
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

from project_paths import DATA_ROOT, REPO_ROOT

TEST_IMG_DIR = DATA_ROOT / "test_images"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

ROI_IMGSZ = 640
ROI_PADDING = 50  # must match build_krones_coco_cropped.py


def build_roi_session(onnx_path: Path) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def letterbox(img: np.ndarray, size: int = ROI_IMGSZ):
    """Resize preserving aspect ratio with gray padding. Returns (img, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def predict_roi(
    session: ort.InferenceSession, img_bgr: np.ndarray
) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) in original image coords, padded by ROI_PADDING and clipped.
    Falls back to None if no detection."""
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb, scale, pad_x, pad_y = letterbox(img_rgb, ROI_IMGSZ)
    x = lb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None, ...]  # NCHW

    in_name = session.get_inputs()[0].name
    out = session.run(None, {in_name: x})[0]  # [1, 300, 6] (x1,y1,x2,y2,score,cls)
    dets = out[0]
    if dets.shape[0] == 0:
        return None
    # Pick highest-scoring detection
    best = dets[np.argmax(dets[:, 4])]
    score = float(best[4])
    if score <= 0:
        return None
    x1, y1, x2, y2 = best[:4]

    # Reverse letterbox
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Apply 50 px padding, clip to image
    cx0 = max(0, int(round(x1 - ROI_PADDING)))
    cy0 = max(0, int(round(y1 - ROI_PADDING)))
    cx1 = min(W, int(round(x2 + ROI_PADDING)))
    cy1 = min(H, int(round(y2 + ROI_PADDING)))
    if cx1 - cx0 <= 0 or cy1 - cy0 <= 0:
        return None
    return cx0, cy0, cx1, cy1


def build_predictor(weights_path: Path) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # keep low so verdict logic chooses
    cfg.INPUT.MIN_SIZE_TEST = 768
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.TEST.DETECTIONS_PER_IMAGE = 60
    return DefaultPredictor(cfg)


def predict_one(
    predictor: DefaultPredictor,
    roi_session: ort.InferenceSession,
    image_path: Path,
    threshold: float,
) -> tuple[int, bool]:
    """Returns (verdict, roi_found)."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"WARN: could not read {image_path}; predicting Reusable=0")
        return 0, False

    crop_box = predict_roi(roi_session, img)
    if crop_box is None:
        # Fallback: run on full image (rare; ROI mAP50=0.99)
        cropped = img
        roi_found = False
    else:
        cx0, cy0, cx1, cy1 = crop_box
        cropped = img[cy0:cy1, cx0:cx1]
        roi_found = True

    out = predictor(cropped)
    inst = out["instances"].to("cpu")
    if len(inst) == 0:
        return 0, roi_found
    scores = inst.scores.numpy()
    classes = inst.pred_classes.numpy()
    masks = inst.pred_masks.numpy()
    areas: dict[str, list[float]] = defaultdict(list)
    for i in range(len(inst)):
        if scores[i] < threshold:
            continue
        cname = KRONES_CLASSES[int(classes[i])]
        areas[cname].append(float(masks[i].sum()))
    return verdict_from_polygons(areas), roi_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to defect Mask R-CNN .pth checkpoint",
    )
    parser.add_argument(
        "--roi-onnx",
        type=Path,
        default=REPO_ROOT / "output" / "ROI_Detector" / "best.onnx",
        help="Path to ROI detector ONNX",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum detection score to count for verdict",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output submission.csv path"
    )
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f"ERROR: defect checkpoint not found: {args.weights}")
    if not args.roi_onnx.exists():
        sys.exit(f"ERROR: ROI ONNX not found: {args.roi_onnx}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(SAMPLE_SUB)
    print(f"Loaded {len(sub)} test image_ids from {SAMPLE_SUB}")
    print(f"Loading ROI detector: {args.roi_onnx}")
    roi_session = build_roi_session(args.roi_onnx)
    print(f"Loading defect checkpoint: {args.weights}")
    predictor = build_predictor(args.weights)
    print(f"Running pipeline (threshold={args.threshold}) ...")

    verdicts = []
    n_no_roi = 0
    for img_id in tqdm(sub["image_id"], unit="img"):
        v, roi_found = predict_one(
            predictor, roi_session, TEST_IMG_DIR / img_id, args.threshold
        )
        verdicts.append(v)
        if not roi_found:
            n_no_roi += 1

    out_df = pd.DataFrame({"image_id": sub["image_id"], "target": verdicts})
    out_df.to_csv(args.out, index=False)

    n0 = sum(1 for v in verdicts if v == 0)
    n1 = sum(1 for v in verdicts if v == 1)
    print(f"\nWrote {args.out}")
    print(f"  total: {len(verdicts)}")
    print(f"  Reusable (0):       {n0} ({100 * n0 / len(verdicts):.1f}%)")
    print(f"  Not Reusable (1):   {n1} ({100 * n1 / len(verdicts):.1f}%)")
    print(f"  ROI fallback (full image): {n_no_roi}")


if __name__ == "__main__":
    main()
