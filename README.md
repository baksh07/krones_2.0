# Krones Vision AI Challenge

This repository contains computer-vision experiments developed for the
Krones Vision AI Challenge. The work explores how image-level
**Reusable / Not Reusable** decisions can be made from bottle images, using
both direct classification and defect-localization pipelines.

The experiments cover data preparation, model training, validation, inference,
and generation of submission CSVs. They are research and development code;
results in individual experiment folders should not be treated as a single,
reproducible benchmark without checking the corresponding configuration.

## Approaches explored

- **Instance segmentation:** Detectron2 Mask R-CNN and Ultralytics YOLO
  segmentation models for localizing defects before deriving an image-level
  verdict.
- **Region-of-interest (ROI) pipelines:** locating and cropping the bottle
  base before running defect segmentation.
- **Image classification:** ResNet, Swin Transformer, and EfficientNet
  experiments for direct binary classification.
- **Data and evaluation experiments:** COCO/YOLO dataset conversion, crop and
  augmentation generation, threshold tuning, test-time augmentation,
  ensembling, and validation/failure analysis.

## Repository layout

| Path | Contents |
| --- | --- |
| `Codebase/` | Training, evaluation, inference, ROI, segmentation, and analysis scripts, organized by experiment. |
| `Dataset/` | Dataset conversion scripts and derived COCO/YOLO annotations and images. |
| `runs/` | Experiment outputs and training logs. |
| `submissions/` | Generated submission CSV files from selected experiments. |

## Data and environment

The scripts expect the challenge dataset and, depending on the experiment,
additional audit outputs and pretrained/model checkpoints. These inputs are
not all provided by this repository. Check each script's configuration before
running it: several scripts define `DATA_ROOT` and `REPO_ROOT` as absolute
Linux paths and need to be updated for your machine.

There is no repository-wide dependency lockfile. The experiments use different
computer-vision frameworks, including PyTorch/torchvision, Detectron2, and
Ultralytics; set up an environment with versions compatible with your Python,
CUDA, and selected experiment.

The supplied challenge materials include a confidentiality notice. They are
not reproduced here; confirm that any dataset, annotations, derived assets, or
results are authorized for sharing before distributing them.

## Running an experiment

From the repository root, configure the dataset and repository paths in the
scripts for the experiment you want to run. For example, the full-image
instance-segmentation flow is:

```bash
python Dataset/build_krones_coco.py
python Codebase/detectron2_train.py
```

For the YOLO segmentation flow:

```bash
python Dataset/build_krones_coco.py
python Dataset/build_krones_yolo.py
python Codebase/yolo_train.py
```

These commands require the original dataset, the relevant framework
dependencies, and an appropriately configured machine. See each script's
header and configuration for its expected inputs, options, and output paths.

## Notes

- `Dataset/build_krones_coco.py` creates COCO train/validation annotations;
  `Dataset/build_krones_yolo.py` converts those annotations for YOLO
  segmentation.
- Some experiments use different datasets, preprocessing, class selections,
  and evaluation rules. Compare results only after checking those details.
- Model checkpoints and other generated artifacts may be stored outside the
  repository or excluded by `.gitignore`.
