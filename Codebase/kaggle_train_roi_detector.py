"""
ROI detector training notebook for Kaggle (paste into cells).

Trains YOLO11n-detect (or YOLO26n if available) to predict the bottle-base
ROI bounding box on Krones train images. Output: ONNX model that takes a
1280x1024 PNG and returns one bbox.

Pipeline at inference time will be:
  test image -> ROI detector (this) -> crop with 50 px padding ->
  defect model (Mask R-CNN trained on cropped images) -> verdict rule

Coordinate handling:
  - Labels normalized using each image's OWN width/height from COCO JSON.
  - ROI bbox clamped to image bounds before normalizing.
  - YOLO handles letterbox + imgsz internally, predictions come back in
    original-image coordinate space.
  - imgsz=640 at both train and inference — must match.

Train/val split: replicates v0's stratified 90/10 split (seed=42) by re-running
the same logic. So the ROI detector's val set = same images as defect model's val.
"""

# ============================================================================
# CELL 1 — Setup, paths, install
# ============================================================================
# !pip install -q ultralytics
# Note: install Ultralytics in a Kaggle cell with !pip install -q ultralytics
# (uncomment the line above when running in Kaggle).

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2  # for sanity-check viz
import matplotlib.pyplot as plt
from tqdm import tqdm

# Kaggle paths — adjust if competition slug differs
COMP_DIR = Path("/kaggle/input/1st-krones-vision-ai-challenge")
SRC_IMG_DIR = COMP_DIR / "train_images"
ANN_FILE = COMP_DIR / "train_annotations.json"

# Output paths in /kaggle/working
WORK = Path("/kaggle/working")
YOLO_ROOT = WORK / "yolo_roi"
LBL_TRAIN = YOLO_ROOT / "labels" / "train"
LBL_VAL = YOLO_ROOT / "labels" / "val"
IMG_TRAIN = YOLO_ROOT / "images" / "train"
IMG_VAL = YOLO_ROOT / "images" / "val"
for d in (LBL_TRAIN, LBL_VAL, IMG_TRAIN, IMG_VAL):
    d.mkdir(parents=True, exist_ok=True)

print("COMP_DIR exists       :", COMP_DIR.exists())
print("SRC_IMG_DIR exists    :", SRC_IMG_DIR.exists())
print("ANN_FILE exists       :", ANN_FILE.exists())
print("Working dir prepared  :", YOLO_ROOT)


# ============================================================================
# CELL 2 — Replicate v0's stratified 90/10 split
# ============================================================================
# Same logic as Dataset/build_krones_coco.py — same seed=42 — produces the same
# train/val partition. This is the key correctness guarantee.

DROP_CATEGORIES = {
    "Roi",
    "No fault",
    "Embossing",
    "Foam residue",
    "Water drop",
}

print("Loading annotations ...")
with open(ANN_FILE) as f:
    coco = json.load(f)

cats = {c["id"]: c["name"] for c in coco["categories"]}
roi_cat_id = next(cid for cid, n in cats.items() if n == "Roi")

# Same kept-category remap logic as v0
keep_old_ids = sorted(cid for cid, name in cats.items() if name not in DROP_CATEGORIES)
old_to_new = {old: new for new, old in enumerate(keep_old_ids, start=1)}

# Group kept (non-dropped) annotations by image — used for stratification only
anns_by_img = defaultdict(list)
for ann in coco["annotations"]:
    if ann["category_id"] in old_to_new:
        a = dict(ann)
        a["category_id"] = old_to_new[ann["category_id"]]
        anns_by_img[ann["image_id"]].append(a)

# Per-class image counts (across the whole dataset)
class_image_counts = Counter()
for anns in anns_by_img.values():
    for cid in {a["category_id"] for a in anns}:
        class_image_counts[cid] += 1


def stratum(img_id):
    anns = anns_by_img.get(img_id, [])
    if not anns:
        return -1
    cids = {a["category_id"] for a in anns}
    return min(cids, key=lambda c: class_image_counts[c])


img_by_stratum = defaultdict(list)
for im in coco["images"]:
    img_by_stratum[stratum(im["id"])].append(im["id"])

rng = random.Random(42)  # same seed as v0
val_frac = 0.10
train_ids, val_ids = set(), set()
for s, ids in img_by_stratum.items():
    rng.shuffle(ids)
    if len(ids) < 2:
        train_ids.update(ids)
        continue
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids.update(ids[:n_val])
    train_ids.update(ids[n_val:])

print(f"Replicated v0 split: train={len(train_ids)}, val={len(val_ids)}")
# Should match the numbers from build_krones_coco.py: 31809 / 3533


# ============================================================================
# CELL 3 — Build ROI lookup, write YOLO labels, symlink images
# ============================================================================

# image_id -> ROI bbox [x, y, w, h] in original-image pixels
roi_by_img_id = {}
for ann in coco["annotations"]:
    if ann["category_id"] == roi_cat_id:
        roi_by_img_id[ann["image_id"]] = ann["bbox"]

print(f"ROI annotations found for {len(roi_by_img_id)} / {len(coco['images'])} images")

# Per-image dim lookup from COCO (don't assume 1280x1024)
images_by_id = {im["id"]: im for im in coco["images"]}


def roi_to_yolo_line(roi, W, H):
    """Convert COCO [x, y, w, h] in pixels to YOLO normalized format."""
    x, y, w, h = roi
    # Clamp to image bounds (annotator-error safety)
    x = max(0.0, min(float(W), x))
    y = max(0.0, min(float(H), y))
    w = max(0.0, min(float(W) - x, w))
    h = max(0.0, min(float(H) - y, h))
    if w <= 0 or h <= 0:
        return None
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    wn = w / W
    hn = h / H
    # Class 0 = ROI (single class)
    return f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def setup_split(split_name, img_ids, lbl_dir, img_dir):
    n_written = 0
    n_no_roi = 0
    n_bad = 0
    for img_id in tqdm(sorted(img_ids), desc=f"{split_name}", unit="img"):
        im = images_by_id[img_id]
        fn = im["file_name"]
        W, H = im["width"], im["height"]

        if img_id not in roi_by_img_id:
            n_no_roi += 1
            continue

        line = roi_to_yolo_line(roi_by_img_id[img_id], W, H)
        if line is None:
            n_bad += 1
            continue

        # Write label file
        (lbl_dir / (Path(fn).stem + ".txt")).write_text(line + "\n")

        # Symlink image (Kaggle filesystem allows symlinks within /kaggle/working)
        dst = img_dir / fn
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to((SRC_IMG_DIR / fn).resolve())
        except OSError:
            shutil.copy2(SRC_IMG_DIR / fn, dst)
        n_written += 1
    print(f"  {split_name}: {n_written} written, {n_no_roi} no_roi, {n_bad} bad_bbox")
    return n_written


print("\nSetting up YOLO-format dataset:")
n_train = setup_split("train", train_ids, LBL_TRAIN, IMG_TRAIN)
n_val = setup_split("val", val_ids, LBL_VAL, IMG_VAL)
print(f"\nTotal: train={n_train}, val={n_val}")


# ============================================================================
# CELL 4 — Sanity check: draw labels on 4 random images, save as a grid
# ============================================================================
# THIS IS THE BUG-CATCHER. If conversion is wrong, the boxes won't be on the
# bottle. Look at the saved figure before training.


def draw_yolo_label(img_path, lbl_path):
    """Read image + label, draw the box, return RGB array."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    with open(lbl_path) as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        xc = float(parts[1]) * W
        yc = float(parts[2]) * H
        bw = float(parts[3]) * W
        bh = float(parts[4]) * H
        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            img,
            f"cls={cls}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return img[:, :, ::-1]  # BGR -> RGB


sample_files = list(LBL_TRAIN.glob("*.txt"))[:4]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, lbl_path in zip(axes.flat, sample_files):
    img_path = IMG_TRAIN / (lbl_path.stem + ".png")
    img = draw_yolo_label(img_path, lbl_path)
    if img is not None:
        ax.imshow(img)
        ax.set_title(lbl_path.stem[:30], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("/kaggle/working/roi_sanity.png", dpi=80)
plt.show()
print("Saved sanity check to /kaggle/working/roi_sanity.png")
print("LOOK AT THIS FIGURE — green box must be on the bottle base.")


# ============================================================================
# CELL 5 — Write dataset YAML
# ============================================================================

yaml_path = YOLO_ROOT / "roi.yaml"
yaml_text = f"""# Krones ROI detector dataset
path: {YOLO_ROOT}
train: images/train
val: images/val

nc: 1
names:
  0: ROI
"""
yaml_path.write_text(yaml_text)
print("Wrote", yaml_path)
print(yaml_text)


# ============================================================================
# CELL 6 — Train YOLO11n-detect (fallback to YOLO26n if available)
# ============================================================================
from ultralytics import YOLO  # noqa: E402

# Try YOLO26n first; fallback to YOLO11n if not available in this Ultralytics version
model_choices = ["yolo26n.pt", "yolo11n.pt"]
model = None
for choice in model_choices:
    try:
        model = YOLO(choice)
        print(f"Loaded {choice}")
        break
    except Exception as e:
        print(f"Skipping {choice}: {e}")
if model is None:
    raise RuntimeError("No YOLO nano detection model available")

results = model.train(
    data=str(yaml_path),
    epochs=15,  # ROI is trivial — 10-15 epochs is plenty
    imgsz=640,
    batch=32,  # T4 / P100 / A100 all handle this fine
    device=0,
    project="/kaggle/working",
    name="roi_detector_v1",
    exist_ok=True,
    seed=42,
    patience=5,  # early stop if no val gain
    # Augmentation: minimal — ROI is in roughly the same spot in every image
    mosaic=0.0,  # disable: corrupts "one ROI per image"
    mixup=0.0,
    copy_paste=0.0,
    hsv_h=0.0,
    hsv_s=0.1,
    hsv_v=0.1,
    degrees=0.0,
    translate=0.02,  # tiny shifts only
    scale=0.1,
    fliplr=0.5,
    flipud=0.0,
    optimizer="SGD",
    lr0=0.01,
    cos_lr=True,
    plots=True,
    verbose=True,
)


# ============================================================================
# CELL 7 — Validate (should be ~99% mAP50)
# ============================================================================
metrics = model.val()
print("Box mAP50    :", metrics.box.map50)
print("Box mAP50-95 :", metrics.box.map)
print("Box recall   :", metrics.box.mr)
print("Box precision:", metrics.box.mp)
# Expect mAP50 > 0.99, mAP50-95 > 0.95 for this trivial single-class task.


# ============================================================================
# CELL 8 — Spot-check predictions on 4 val images
# ============================================================================
val_imgs = list(IMG_VAL.glob("*.png"))[:4]
results = model.predict(
    source=[str(p) for p in val_imgs],
    imgsz=640,
    conf=0.5,
    save=False,
    verbose=False,
)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, img_path, res in zip(axes.flat, val_imgs, results):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(
                img,
                f"ROI {conf:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
    ax.imshow(img[:, :, ::-1])
    ax.set_title(img_path.stem[:30], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("/kaggle/working/roi_predictions.png", dpi=80)
plt.show()


# ============================================================================
# CELL 9 — Export to ONNX
# ============================================================================

best_pt = Path("/kaggle/working/roi_detector_v1/weights/best.pt")
print("Best checkpoint:", best_pt, "exists:", best_pt.exists())

# Reload best weights and export
m = YOLO(str(best_pt))
onnx_path = m.export(
    format="onnx",
    imgsz=640,
    dynamic=False,  # fixed input shape — better for ONNX runtime
    simplify=True,
    opset=18,
)
print("ONNX exported to:", onnx_path)
print("Download this file from /kaggle/working/ for offline use.")


# ============================================================================
# CELL 10 — Summary
# ============================================================================
print("=" * 60)
print("ROI DETECTOR TRAINING COMPLETE")
print("=" * 60)
print(f"Train images: {n_train}")
print(f"Val images:   {n_val}")
print(f"Best mAP50:   {metrics.box.map50:.4f}  (expect > 0.99)")
print(f"Best mAP50-95:{metrics.box.map:.4f}    (expect > 0.95)")
print()
print("Files in /kaggle/working/ :")
for p in [
    "roi_detector_v1/weights/best.pt",
    "roi_detector_v1/weights/best.onnx",
    "roi_sanity.png",
    "roi_predictions.png",
]:
    full = Path("/kaggle/working") / p
    if full.exists():
        size = full.stat().st_size / 1024 / 1024
        print(f"  {p:50s} {size:7.2f} MB")
