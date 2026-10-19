"""
infer_yolo_seg.py
-----------------
Run the trained YOLO-seg model on a set of crop images and save per-image
per-class predictions to a CSV that evaluate_rule.py consumes.

For each image, we record:
  - Max confidence per class (or 0 if no detection of that class)
  - Total predicted mask area in raw crop pixels per class
    (sum of pixel counts across all detected instances of that class)

Usage:
    python infer_yolo_seg.py \\
        --model /path/to/best.pt \\
        --images_dir /path/to/yolo_seg_7cls/images/val \\
        --out_csv yolo_seg_val_predictions.csv \\
        --imgsz 640 --conf 0.05 --device 0

Notes:
  - Uses very low base confidence (0.05) so we capture even weak detections.
    The actual decision threshold is applied later in evaluate_rule.py.
  - Mask areas are computed in the crop's native pixel space (not normalized).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# Class names matching the YOLO-seg dataset (Design A, 7 classes)
CLASS_NAMES = [
    "Scuffing",  # 0  (conditional, threshold 75,000 px)
    "Air bubble",  # 1  (conditional, threshold 500)
    "Contamination light",  # 2  (conditional, threshold 180)
    "Chip",  # 3  (conditional, threshold 200)
    "Scuffing heavy",  # 4  (conditional, threshold 1,200)
    "Glass imperfection",  # 5  (conditional, threshold 100)
    "Hard-Faulty",  # 6  (always Faulty)
]
N_CLASSES = len(CLASS_NAMES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", required=True, help="Path to trained YOLO-seg .pt checkpoint"
    )
    p.add_argument(
        "--images_dir",
        required=True,
        help="Directory of crop images to run inference on",
    )
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Base confidence floor for raw inference (low = catch all)",
    )
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    # Validate paths
    model_path = Path(args.model)
    images_dir = Path(args.images_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    # Collect image files (Ultralytics handles dir input but we want explicit ordering)
    img_files = sorted(
        [
            f
            for f in images_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    )
    if not img_files:
        raise RuntimeError(f"No images found in {images_dir}")
    print(f"Found {len(img_files)} images in {images_dir}")

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Build CSV header
    header = ["file_name"]
    for c in CLASS_NAMES:
        safe = c.replace(" ", "_").replace("-", "_")
        header.append(f"conf_{safe}")
        header.append(f"area_{safe}")
        header.append(f"n_det_{safe}")

    # Run inference in batches and write rows progressively
    rows = []
    print(f"Running inference (imgsz={args.imgsz}, conf={args.conf}, iou={args.iou})")

    # Process in batches via Ultralytics' built-in batching
    for i in tqdm(range(0, len(img_files), args.batch), desc="batches"):
        batch_files = [str(f) for f in img_files[i : i + args.batch]]
        results = model.predict(
            source=batch_files,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            save=False,
            stream=False,
        )

        for img_path, result in zip(batch_files, results):
            file_name = Path(img_path).name

            # Per-class accumulators
            max_conf = [0.0] * N_CLASSES
            total_area = [0] * N_CLASSES
            n_det = [0] * N_CLASSES

            if result.boxes is not None and len(result.boxes) > 0:
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                # Mask areas: count nonzero pixels per detected mask
                if result.masks is not None and result.masks.data is not None:
                    masks = result.masks.data.cpu().numpy()  # shape: (N, H, W) bool
                    # Note: masks are in MODEL space (imgsz), not original crop dims.
                    # We need to scale back to original image space for accurate
                    # pixel counts. result.orig_shape = (H, W) of the original image.
                    orig_h, orig_w = result.orig_shape
                    model_h, model_w = masks.shape[1], masks.shape[2]
                    # Scale factor for area (linear scale squared)
                    area_scale = (orig_h * orig_w) / (model_h * model_w)

                    mask_areas = (
                        masks.sum(axis=(1, 2)) * area_scale
                    )  # area in orig pixels
                else:
                    # No mask data -> fall back to bbox area
                    bboxes = result.boxes.xyxy.cpu().numpy()
                    mask_areas = (bboxes[:, 2] - bboxes[:, 0]) * (
                        bboxes[:, 3] - bboxes[:, 1]
                    )

                for cid, conf, area in zip(cls_ids, confs, mask_areas):
                    if 0 <= cid < N_CLASSES:
                        if conf > max_conf[cid]:
                            max_conf[cid] = float(conf)
                        total_area[cid] += int(area)
                        n_det[cid] += 1

            row = [file_name]
            for c in range(N_CLASSES):
                row.extend([f"{max_conf[c]:.4f}", total_area[c], n_det[c]])
            rows.append(row)

    # Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(f"Columns: {len(header)} ({len(header) - 1} per-class fields + file_name)")

    # Summary stats
    print("\nPer-class detection summary:")
    print(
        f"  {'Class':<24s} {'images_with_det':>16s} {'avg_max_conf':>14s} "
        f"{'avg_area_when_det':>18s}"
    )
    for c in range(N_CLASSES):
        confs = [float(r[1 + c * 3]) for r in rows]
        areas = [int(r[2 + c * 3]) for r in rows]
        n_with_det = sum(1 for x in confs if x > 0)
        avg_conf = np.mean([x for x in confs if x > 0]) if n_with_det > 0 else 0.0
        avg_area = (
            np.mean([a for a, c_ in zip(areas, confs) if c_ > 0])
            if n_with_det > 0
            else 0.0
        )
        print(
            f"  {CLASS_NAMES[c]:<24s} {n_with_det:>16d} {avg_conf:>14.3f} "
            f"{avg_area:>18.0f}"
        )


if __name__ == "__main__":
    main()
