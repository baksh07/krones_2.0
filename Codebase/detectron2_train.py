"""
Mask R-CNN R50-FPN fine-tune on Krones (22 kept classes).

Pipeline:
  1. Register krones_train + krones_val from COCO JSON in Dataset/coco/.
  2. Configure Mask R-CNN R50-FPN starting from COCO-pretrained weights.
  3. Override the train data loader to add brightness/contrast jitter on top
     of the default ResizeShortestEdge + RandomFlip.
  4. Override the evaluator to run COCOEvaluator AND a custom VerdictEvaluator
     that reproduces the Krones audit verdict rule on each prediction and
     reports verdict accuracy vs train.csv.

Run:
  python detectron2_train.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
)
from detectron2.utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from project_paths import DATA_ROOT, REPO_ROOT

COCO_DIR = REPO_ROOT / "Dataset" / "coco"
IMG_DIR = str(DATA_ROOT / "train_images")
TRAIN_CSV = DATA_ROOT / "train.csv"
OUTPUT_DIR = REPO_ROOT / "output" / "maskrcnn_r50_v0"

with open(COCO_DIR / "krones_classes.json") as f:
    KRONES_CLASSES: list[str] = json.load(f)
NUM_CLASSES = len(KRONES_CLASSES)
assert NUM_CLASSES == 22, f"expected 22 classes, got {NUM_CLASSES}"


# ---------------------------------------------------------------------------
# Verdict rule (from data-audit-report.md §2.2, §2.3, §6.4)
# ---------------------------------------------------------------------------
ALWAYS_FAULTY = {
    "Break / Crack",
    "Circlip",
    "Contamination dark",
    "Crown cap",
    "Foil / Semitransparent",
    "Foreign object - manual cleaning",
    "Foreign object - washing machine",
    "Glass shard",
    "Insect",
    "Label",
    "Liquid",
    "Mold",
    "No base visible",
    "Paint residue",
    "Straw",
    "Yeast residue",
}
CONDITIONAL_PER_REGION = {
    "Air bubble": 500,
    "Chip": 200,
    "Glass imperfection": 100,
}
CONDITIONAL_AGGREGATED = {
    "Scuffing": 75000,
    "Scuffing heavy": 1200,
    "Contamination light": 180,
}


def verdict_from_polygons(areas_per_class: dict[str, list[float]]) -> int:
    """Return 1 (Not Reusable) or 0 (Reusable)."""
    for cls in ALWAYS_FAULTY:
        if areas_per_class.get(cls):
            return 1
    for cls, thr in CONDITIONAL_PER_REGION.items():
        if any(a > thr for a in areas_per_class.get(cls, [])):
            return 1
    for cls, thr in CONDITIONAL_AGGREGATED.items():
        if sum(areas_per_class.get(cls, [])) > thr:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Custom verdict evaluator
# ---------------------------------------------------------------------------
class VerdictEvaluator(DatasetEvaluator):
    """Compute verdict accuracy against train.csv labels.

    Accumulates predictions per image, applies the verdict rule, compares to
    the ground-truth label from train.csv (which lives outside the COCO JSON).
    """

    def __init__(
        self,
        dataset_name: str,
        train_csv_path: Path,
        class_names: list[str],
        score_threshold: float = 0.5,
    ):
        self._dataset_name = dataset_name
        self._class_names = class_names
        self._score_threshold = score_threshold

        df = pd.read_csv(train_csv_path)
        self._label_by_filename: dict[str, int] = dict(
            zip(df["image_id"].astype(str), df["target"].astype(int))
        )

        records = DatasetCatalog.get(dataset_name)
        self._filename_by_id: dict[int, str] = {
            r["image_id"]: Path(r["file_name"]).name for r in records
        }

    def reset(self):
        self._preds: dict[int, int] = {}

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            inst = out["instances"].to("cpu")
            if len(inst) == 0:
                self._preds[inp["image_id"]] = 0
                continue

            scores = inst.scores.numpy()
            classes = inst.pred_classes.numpy()
            masks = inst.pred_masks.numpy()  # (N, H, W) bool

            areas: dict[str, list[float]] = defaultdict(list)
            for i in range(len(inst)):
                if scores[i] < self._score_threshold:
                    continue
                cname = self._class_names[int(classes[i])]
                areas[cname].append(float(masks[i].sum()))

            self._preds[inp["image_id"]] = verdict_from_polygons(areas)

    def evaluate(self):
        n_total = 0
        cm = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for img_id, pred in self._preds.items():
            fn = self._filename_by_id.get(img_id)
            if fn is None or fn not in self._label_by_filename:
                continue
            true = self._label_by_filename[fn]
            cm[(true, pred)] += 1
            n_total += 1
        if n_total == 0:
            return {"verdict": {"acc": 0.0, "n": 0}}
        tn, fp = cm[(0, 0)], cm[(0, 1)]
        fn, tp = cm[(1, 0)], cm[(1, 1)]
        acc = (tn + tp) / n_total
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {
            "verdict": {
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n": n_total,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        }


# ---------------------------------------------------------------------------
# Trainer with custom mapper + evaluator
# ---------------------------------------------------------------------------
class KronesTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomBrightness(0.85, 1.15),
            T.RandomContrast(0.85, 1.15),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        return DatasetEvaluators(
            [
                COCOEvaluator(
                    dataset_name,
                    tasks=("bbox", "segm"),
                    distributed=False,
                    output_dir=output_folder,
                ),
                VerdictEvaluator(
                    dataset_name,
                    train_csv_path=TRAIN_CSV,
                    class_names=KRONES_CLASSES,
                ),
            ]
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def register_datasets():
    for split in ("train", "val"):
        name = f"krones_{split}"
        if name in DatasetCatalog.list():
            continue
        register_coco_instances(
            name, {}, str(COCO_DIR / f"krones_{split}.json"), IMG_DIR
        )
        MetadataCatalog.get(name).thing_classes = KRONES_CLASSES
        DatasetCatalog.get(name)
        meta_classes = list(MetadataCatalog.get(name).thing_classes)
        assert meta_classes == KRONES_CLASSES, (
            f"Class mismatch in {name}: "
            f"metadata={meta_classes} vs file={KRONES_CLASSES}"
        )


def build_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.DATASETS.TRAIN = ("krones_train",)
    cfg.DATASETS.TEST = ("krones_val",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.001
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    cfg.SOLVER.IMS_PER_BATCH = 12
    cfg.SOLVER.BASE_LR = 0.015
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.STEPS = (14000, 18000)
    cfg.SOLVER.WARMUP_ITERS = 1500
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000

    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1024, 1280)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.INPUT.MASK_FORMAT = "polygon"

    cfg.TEST.EVAL_PERIOD = 2000
    cfg.TEST.DETECTIONS_PER_IMAGE = 60

    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    return cfg


def main():
    setup_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    register_datasets()
    cfg = build_cfg()

    print(f"OUTPUT_DIR: {cfg.OUTPUT_DIR}")
    print(f"NUM_CLASSES: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    trainer = KronesTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
