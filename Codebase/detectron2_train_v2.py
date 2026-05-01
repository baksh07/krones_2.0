"""
Cascade Mask R-CNN R50-FPN fine-tune on Krones (22 kept classes) — v2.

Diagnosed v0 failures dictate this design:
  FN (104 in val): dominated by recall misses — Mold/Contamination dark/Scuffing
                   instances completely absent from predictions (BACKGROUND
                   column dominates the confusion matrix).
  FP (145 in val): dominated by mask over-segmentation — Scuffing predictions
                   draw polygons 3-5x larger than GT, summing past the 75000
                   px aggregation threshold.

v2 changes from v0:
  1. Architecture: Cascade Mask R-CNN R50-FPN (cascaded RoI heads at IoU
     [0.5, 0.6, 0.7]) — better recall AND mask precision in one model.
  2. RPN: more proposals retained pre/post NMS so small defects don't get
     filtered out before the head sees them.
  3. Anchors: add 0.25 and 4.0 aspect ratios for elongated Scuffing scratches
     (default ratios are [0.5, 1.0, 2.0] — 5:1 scratches have no good anchor).
  4. Test-time: DETECTIONS_PER_IMAGE=100 (was 60), SCORE_THRESH_TEST=0.03
     (was 0.05) so verdict aggregation has more low-confidence signal.
  5. Mask loss weight 1.5 (was 1.0): push polygon precision up for Scuffing.
  6. Box head loss weight 1.5: sharpen Mold/Yeast/Contamination class
     boundaries.
  7. RFS threshold 0.005 (was 0.001): now actually upsamples Insect, Straw,
     Circlip, Foil/Semitransparent, Liquid, Paint residue.
  8. Schedule: 16000 iters (v0 peaked at iter 16000 anyway), batch 8 (cascade
     is heavier than vanilla Mask R-CNN), LR 0.01 (linear-scaled).

Run:
  python detectron2_train_v2.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
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
REPO_ROOT = Path('/home/dll0202/baksh/krones_2.0')
DATA_ROOT = Path('/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge')
COCO_DIR = REPO_ROOT / 'Dataset' / 'coco'
IMG_DIR = str(DATA_ROOT / 'train_images')
TRAIN_CSV = DATA_ROOT / 'train.csv'
OUTPUT_DIR = REPO_ROOT / 'output' / 'cascade_v2'

with open(COCO_DIR / 'krones_classes.json') as f:
    KRONES_CLASSES: list[str] = json.load(f)
NUM_CLASSES = len(KRONES_CLASSES)
assert NUM_CLASSES == 22, f'expected 22 classes, got {NUM_CLASSES}'


# ---------------------------------------------------------------------------
# Verdict rule (from data-audit-report.md §2.2, §2.3, §6.4) — UNCHANGED from v0
# ---------------------------------------------------------------------------
ALWAYS_FAULTY = {
    'Break / Crack', 'Circlip', 'Contamination dark', 'Crown cap',
    'Foil / Semitransparent', 'Foreign object - manual cleaning',
    'Foreign object - washing machine', 'Glass shard', 'Insect',
    'Label', 'Liquid', 'Mold', 'No base visible', 'Paint residue',
    'Straw', 'Yeast residue',
}
CONDITIONAL_PER_REGION = {
    'Air bubble': 500,
    'Chip': 200,
    'Glass imperfection': 100,
}
CONDITIONAL_AGGREGATED = {
    'Scuffing': 75000,
    'Scuffing heavy': 1200,
    'Contamination light': 180,
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
# Custom verdict evaluator — UNCHANGED from v0
# ---------------------------------------------------------------------------
class VerdictEvaluator(DatasetEvaluator):
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
            zip(df['image_id'].astype(str), df['target'].astype(int))
        )
        records = DatasetCatalog.get(dataset_name)
        self._filename_by_id: dict[int, str] = {
            r['image_id']: Path(r['file_name']).name for r in records
        }

    def reset(self):
        self._preds: dict[int, int] = {}

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            inst = out['instances'].to('cpu')
            if len(inst) == 0:
                self._preds[inp['image_id']] = 0
                continue
            scores = inst.scores.numpy()
            classes = inst.pred_classes.numpy()
            masks = inst.pred_masks.numpy()
            areas: dict[str, list[float]] = defaultdict(list)
            for i in range(len(inst)):
                if scores[i] < self._score_threshold:
                    continue
                cname = self._class_names[int(classes[i])]
                areas[cname].append(float(masks[i].sum()))
            self._preds[inp['image_id']] = verdict_from_polygons(areas)

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
            return {'verdict': {'acc': 0.0, 'n': 0}}
        tn, fp = cm[(0, 0)], cm[(0, 1)]
        fn, tp = cm[(1, 0)], cm[(1, 1)]
        acc = (tn + tp) / n_total
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            'verdict': {
                'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
                'n': n_total, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            }
        }


# ---------------------------------------------------------------------------
# Trainer — augmentations unchanged, evaluator unchanged
# ---------------------------------------------------------------------------
class KronesTrainerV2(DefaultTrainer):

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
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference', dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        return DatasetEvaluators([
            COCOEvaluator(
                dataset_name,
                tasks=('bbox', 'segm'),
                distributed=False,
                output_dir=output_folder,
            ),
            VerdictEvaluator(
                dataset_name,
                train_csv_path=TRAIN_CSV,
                class_names=KRONES_CLASSES,
            ),
        ])


# ---------------------------------------------------------------------------
# Dataset registration — unchanged
# ---------------------------------------------------------------------------
def register_datasets():
    for split in ('train', 'val'):
        name = f'krones_{split}'
        if name in DatasetCatalog.list():
            continue
        register_coco_instances(
            name, {}, str(COCO_DIR / f'krones_{split}.json'), IMG_DIR
        )
        MetadataCatalog.get(name).thing_classes = KRONES_CLASSES
        DatasetCatalog.get(name)
        meta_classes = list(MetadataCatalog.get(name).thing_classes)
        assert meta_classes == KRONES_CLASSES, (
            f'Class mismatch in {name}: '
            f'metadata={meta_classes} vs file={KRONES_CLASSES}'
        )


# ---------------------------------------------------------------------------
# v2 config
# ---------------------------------------------------------------------------
CASCADE_CFG = 'Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml'


def build_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CASCADE_CFG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CASCADE_CFG)

    cfg.DATASETS.TRAIN = ('krones_train',)
    cfg.DATASETS.TEST = ('krones_val',)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.SAMPLER_TRAIN = 'RepeatFactorTrainingSampler'
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.005     # v2: was 0.001 — actually upsamples now
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    # Cascade head class count
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    # === FN fix: more proposals reach the head ===
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 4000     # v2: was 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000    # v2: was 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000      # v2: was 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500     # v2: was 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.8              # v2: was 0.7

    # === FN fix: thin anchors for elongated Scuffing scratches ===
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]

    # === FN fix: keep more candidate detections at test time ===
    cfg.TEST.DETECTIONS_PER_IMAGE = 100         # v2: was 60
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.03  # v2: was 0.05

    # === FP fix: tighter mask polygons ===
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.5   # v2: was default 1.0

    # === FP fix: sharpen class boundaries (Mold/Yeast/Contamination ring) ===
    cfg.MODEL.ROI_BOX_HEAD.LOSS_WEIGHT = 1.5    # v2: was default 1.0

    # === Schedule: cascade is heavier, batch 8 instead of 12 ===
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01                   # 0.015 * (8/12) linear scaling
    cfg.SOLVER.MAX_ITER = 16000                 # v0 peaked at iter 16000
    cfg.SOLVER.STEPS = (12000, 14500)           # 75% / 90% decay
    cfg.SOLVER.WARMUP_ITERS = 1500
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000

    # === Input pipeline (unchanged from v0) ===
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1024, 1280)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.INPUT.MASK_FORMAT = 'polygon'

    cfg.TEST.EVAL_PERIOD = 2000

    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    return cfg


def main():
    setup_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    register_datasets()
    cfg = build_cfg()

    print(f'OUTPUT_DIR:       {cfg.OUTPUT_DIR}')
    print(f'NUM_CLASSES:      {cfg.MODEL.ROI_HEADS.NUM_CLASSES}')
    print(f'IMS_PER_BATCH:    {cfg.SOLVER.IMS_PER_BATCH}')
    print(f'BASE_LR:          {cfg.SOLVER.BASE_LR}')
    print(f'MAX_ITER:         {cfg.SOLVER.MAX_ITER}')
    print(f'REPEAT_THRESHOLD: {cfg.DATALOADER.REPEAT_THRESHOLD}')
    print(f'ASPECT_RATIOS:    {cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}')
    print(f'CUDA available:   {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU:              {torch.cuda.get_device_name(0)}')

    trainer = KronesTrainerV2(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    main()
