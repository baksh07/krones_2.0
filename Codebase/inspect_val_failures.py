"""
Save random val false-positives and false-negatives as side-by-side images for
manual inspection.

For each picked image:
  Left panel  = original image with GROUND-TRUTH polygons drawn (true classes).
  Right panel = original image with MODEL prediction polygons (with class +
                confidence score, only above --threshold).

Use this to manually classify failure causes:
  - Dropped-class confusion (water drop fired as Air bubble, foam as Mold, etc.)
  - Class-pair confusion (Mold vs Yeast residue, Scuffing vs Scuffing heavy)
  - Threshold edge case (score 0.49 vs 0.51)
  - Genuine miss (model didn't detect the defect at all)
  - Aggregation borderline (Scuffing summed area near 75000)
  - Annotator disagreement (looks ambiguous to a human)

Reads val ground truth from COCO JSON (same path the trainer uses), pulls
the per-image verdict (true) from train.csv. Picks --num random FPs and
--num random FNs.

Usage:
  python inspect_val_failures.py \\
      --weights ../output/maskrcnn_r50_v0/model_final.pth \\
      --threshold 0.50 \\
      --num 30 \\
      --out ../output/diagnostics/iter19999/failures/
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectron2_train import (  # noqa: E402
    KRONES_CLASSES,
    NUM_CLASSES,
    verdict_from_polygons,
)

from project_paths import DATA_ROOT, REPO_ROOT

COCO_VAL = REPO_ROOT / "Dataset" / "coco" / "krones_val.json"
IMG_DIR = DATA_ROOT / "train_images"
TRAIN_CSV = DATA_ROOT / "train.csv"


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


def predict_verdict_and_keep(predictor, image_path: Path, threshold: float):
    """Run inference, return (verdict, full_instances, image_bgr)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None
    out = predictor(img)
    inst = out["instances"].to("cpu")
    if len(inst) == 0:
        return 0, inst, img
    scores = inst.scores.numpy()
    classes = inst.pred_classes.numpy()
    masks = inst.pred_masks.numpy()
    areas: dict[str, list[float]] = defaultdict(list)
    for i in range(len(inst)):
        if scores[i] < threshold:
            continue
        cname = KRONES_CLASSES[int(classes[i])]
        areas[cname].append(float(masks[i].sum()))
    return verdict_from_polygons(areas), inst, img


def make_panel(
    image_bgr,
    gt_record,
    pred_instances,
    metadata,
    threshold,
    filename,
    true_verdict,
    pred_verdict,
    out_path: Path,
):
    """Save a 2-panel figure: left=GT polygons, right=predictions filtered by threshold."""
    img_rgb = image_bgr[:, :, ::-1]

    # Left: ground truth via Visualizer.draw_dataset_dict
    vis_gt = Visualizer(img_rgb, metadata=metadata, scale=0.7)
    vis_gt.draw_dataset_dict(gt_record)
    gt_panel = vis_gt.output.get_image()

    # Right: filter pred_instances to threshold and draw
    if len(pred_instances) > 0:
        keep_mask = pred_instances.scores >= threshold
        pred_filtered = pred_instances[keep_mask]
    else:
        pred_filtered = pred_instances

    vis_pred = Visualizer(img_rgb, metadata=metadata, scale=0.7)
    vis_pred.draw_instance_predictions(pred_filtered)
    pred_panel = vis_pred.output.get_image()

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    axes[0].imshow(gt_panel)
    axes[0].set_title(f"Ground truth — true verdict = {true_verdict}", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(pred_panel)
    axes[1].set_title(
        f"Prediction (thr {threshold}) — pred verdict = {pred_verdict}", fontsize=14
    )
    axes[1].axis("off")
    fig.suptitle(filename, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument(
        "--num", type=int, default=30, help="Number of FPs and FNs each to save"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fp").mkdir(exist_ok=True)
    (args.out / "fn").mkdir(exist_ok=True)

    name = "krones_val_inspect"
    if name not in DatasetCatalog.list():
        register_coco_instances(name, {}, str(COCO_VAL), str(IMG_DIR))
        MetadataCatalog.get(name).thing_classes = KRONES_CLASSES
    metadata = MetadataCatalog.get(name)
    records = DatasetCatalog.get(name)
    record_by_filename = {Path(r["file_name"]).name: r for r in records}

    df_csv = pd.read_csv(TRAIN_CSV)
    label_by_filename = dict(zip(df_csv["image_id"], df_csv["target"].astype(int)))

    print(f"Loading checkpoint: {args.weights}")
    predictor = build_predictor(args.weights)

    fp_imgs: list[tuple[str, int, int, object, np.ndarray]] = []
    fn_imgs: list[tuple[str, int, int, object, np.ndarray]] = []

    print(f"Scanning {len(records)} val images for FP/FN ...")
    for r in tqdm(records, unit="img"):
        filename = Path(r["file_name"]).name
        true_label = label_by_filename.get(filename)
        if true_label is None:
            continue
        pred_verdict, inst, img = predict_verdict_and_keep(
            predictor, IMG_DIR / filename, args.threshold
        )
        if pred_verdict is None:
            continue
        if pred_verdict == true_label:
            continue
        if true_label == 0 and pred_verdict == 1:
            fp_imgs.append((filename, true_label, pred_verdict, inst, img))
        elif true_label == 1 and pred_verdict == 0:
            fn_imgs.append((filename, true_label, pred_verdict, inst, img))

    print(f"\nFound {len(fp_imgs)} FPs and {len(fn_imgs)} FNs in val.")

    rng = random.Random(args.seed)
    fp_sample = rng.sample(fp_imgs, min(args.num, len(fp_imgs)))
    fn_sample = rng.sample(fn_imgs, min(args.num, len(fn_imgs)))

    print(f"Writing {len(fp_sample)} FP panels and {len(fn_sample)} FN panels ...")
    for i, (filename, true_v, pred_v, inst, img) in enumerate(
        tqdm(fp_sample, desc="FP")
    ):
        gt_record = record_by_filename[filename]
        out_path = args.out / "fp" / f"fp_{i:02d}_{Path(filename).stem}.png"
        make_panel(
            img,
            gt_record,
            inst,
            metadata,
            args.threshold,
            filename,
            true_v,
            pred_v,
            out_path,
        )
    for i, (filename, true_v, pred_v, inst, img) in enumerate(
        tqdm(fn_sample, desc="FN")
    ):
        gt_record = record_by_filename[filename]
        out_path = args.out / "fn" / f"fn_{i:02d}_{Path(filename).stem}.png"
        make_panel(
            img,
            gt_record,
            inst,
            metadata,
            args.threshold,
            filename,
            true_v,
            pred_v,
            out_path,
        )

    print("\nDone. Inspect:")
    print(f"  {args.out / 'fp'}/  ({len(fp_sample)} false-positive panels)")
    print(f"  {args.out / 'fn'}/  ({len(fn_sample)} false-negative panels)")


if __name__ == "__main__":
    main()
