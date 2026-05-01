"""
Train YOLO26m-seg on Krones (22 classes) — y0 baseline (YOLO version of v0).

Hyperparameters chosen to match Krones data characteristics:
  - imgsz=1024: defects (Scuffing) are ~1% of ROI; default 640 is too small.
  - mosaic=0.0, mixup=0.0, copy_paste=0.0: composing/mixing images destroys
    defect context (the "is this on the bottle base?" cue matters).
  - degrees=0.0, flipud=0.0: bottles are always upright; rotation/v-flip noise.
  - hsv_h=0.0, hsv_s=0.2, hsv_v=0.2: mild lighting jitter only (the inspection
    line is grayscale-ish; aggressive HSV would distort defect appearance).
  - fliplr=0.5: horizontal flip is fine (bottle base is symmetric).
  - SGD, lr0=0.01, cos_lr=True: standard YOLO recipe.
  - patience=15: early-stop if val plateaus.

Output: /home/dll0202/baksh/krones_2.0/output/yolo26m_v1/

Run:
  python yolo_train.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path('/home/dll0202/baksh/krones_2.0')
DATASET_YAML = REPO_ROOT / 'Dataset' / 'yolo' / 'krones.yaml'
OUTPUT_PROJECT = REPO_ROOT / 'output'
RUN_NAME = 'yolo26m_v1'


def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f'{DATASET_YAML} not found. Run Dataset/build_krones_yolo.py first.'
        )

    model = YOLO('yolo26m-seg.pt')  # auto-downloads on first use

    model.train(
        data=str(DATASET_YAML),
        epochs=30,
        imgsz=1024,
        batch=8,
        device=0,
        project=str(OUTPUT_PROJECT),
        name=RUN_NAME,
        exist_ok=False,            # set True to overwrite existing run
        seed=42,
        deterministic=True,

        # === Augmentation: minimal, defect-friendly ===
        mosaic=0.0,                # off — destroys defect context
        mixup=0.0,                 # off
        copy_paste=0.0,            # off
        hsv_h=0.0,                 # bottles aren't color-coded
        hsv_s=0.2,                 # mild saturation jitter
        hsv_v=0.2,                 # mild brightness jitter
        degrees=0.0,               # bottles always upright
        translate=0.05,            # small position jitter
        scale=0.2,                 # modest zoom
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,                # OK — base is roughly symmetric
        flipud=0.0,                # off
        bgr=0.0,
        close_mosaic=0,            # mosaic is already off

        # === Training ===
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        amp=True,
        patience=15,

        # === Loss weights (defaults; tune later if needed) ===
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # === Misc ===
        save=True,
        save_period=5,             # checkpoint every 5 epochs
        cache=False,               # don't cache images (save RAM)
        workers=8,
        plots=True,
        verbose=True,
        val=True,
        rect=False,                # rectangular padding hurts dense scenes
    )


if __name__ == '__main__':
    main()
