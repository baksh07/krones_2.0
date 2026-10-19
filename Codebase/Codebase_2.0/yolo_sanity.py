"""
Krones — YOLO ROI sanity check
-------------------------------
Loads a few test images, runs YOLO ROI prediction, draws the predicted bbox
on the image, and saves a grid for visual inspection. Also computes statistics
on bbox size/position consistency.

Usage:
    python yolo_sanity.py \\
        --yolo /path/to/best.onnx \\
        --test-dir /path/to/test_images \\
        --n-samples 30 \\
        --out yolo_sanity_grid.png

If you don't have a separate test directory locally, you can also run on a
sample of train images:
    python yolo_sanity.py --yolo ... --test-dir <train_imgs> --n-samples 30
"""

import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.patches as patches


YOLO_INPUT = 640
ROI_PAD = 150


def preprocess_for_yolo(img_pil):
    img = img_pil.convert("RGB").resize((YOLO_INPUT, YOLO_INPUT))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]
    return arr


def parse_yolo_output(out, orig_w, orig_h, score_thresh=0.1):
    """Returns (x, y, w, h, score) in original-image pixels, or None."""
    arr = out[0] if out.ndim == 3 else out
    if arr.shape[0] == 0 or arr.shape[1] < 6:
        return None
    scores = arr[:, 4]
    idx = int(np.argmax(scores))
    if scores[idx] < score_thresh:
        return None
    x1, y1, x2, y2 = arr[idx, :4]
    sx, sy = orig_w / YOLO_INPUT, orig_h / YOLO_INPUT
    return (
        float(x1 * sx),
        float(y1 * sy),
        float((x2 - x1) * sx),
        float((y2 - y1) * sy),
        float(scores[idx]),
    )


def main(args):
    print(f"loading YOLO ONNX: {args.yolo}")
    sess = ort.InferenceSession(
        str(args.yolo), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"  providers: {sess.get_providers()}")

    # gather images
    test_dir = Path(args.test_dir)
    all_imgs = sorted(list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")))
    print(f"found {len(all_imgs)} images in {test_dir}")
    if len(all_imgs) == 0:
        print("ERROR: no images found")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(all_imgs, min(args.n_samples, len(all_imgs)))
    print(f"sampling {len(sample)} images for visual inspection")

    # also collect statistics over a larger subset
    print(f"\ncollecting bbox statistics over up to {args.n_stats} images...")
    stats_sample = rng.sample(all_imgs, min(args.n_stats, len(all_imgs)))
    stats = []
    n_no_detection = 0
    for img_path in stats_sample:
        with Image.open(img_path) as im:
            W, H = im.size
            x = preprocess_for_yolo(im)
        out = sess.run(out_names, {in_name: x})[0]
        bbox = parse_yolo_output(out, W, H)
        if bbox is None:
            n_no_detection += 1
            continue
        bx, by, bw, bh, score = bbox
        stats.append(
            {
                "file": img_path.name,
                "img_W": W,
                "img_H": H,
                "bx": bx,
                "by": by,
                "bw": bw,
                "bh": bh,
                "score": score,
                "bbox_frac_w": bw / W,
                "bbox_frac_h": bh / H,
                "bbox_center_x_frac": (bx + bw / 2) / W,
                "bbox_center_y_frac": (by + bh / 2) / H,
                "aspect_ratio": bw / max(1, bh),
            }
        )
    sdf = pd.DataFrame(stats)
    print(
        f"  detection rate: {len(sdf)}/{len(stats_sample)}  "
        f"(no-detection: {n_no_detection})"
    )
    print()
    print("BBox size as fraction of image (should be ~ same for similar bottles):")
    print(
        f"  width fraction:    mean {sdf['bbox_frac_w'].mean():.3f}  "
        f"std {sdf['bbox_frac_w'].std():.3f}  "
        f"range [{sdf['bbox_frac_w'].min():.3f}, {sdf['bbox_frac_w'].max():.3f}]"
    )
    print(
        f"  height fraction:   mean {sdf['bbox_frac_h'].mean():.3f}  "
        f"std {sdf['bbox_frac_h'].std():.3f}  "
        f"range [{sdf['bbox_frac_h'].min():.3f}, {sdf['bbox_frac_h'].max():.3f}]"
    )
    print()
    print("BBox center (should cluster around 0.5, 0.5 if bottles are centered):")
    print(
        f"  center x fraction: mean {sdf['bbox_center_x_frac'].mean():.3f}  "
        f"std {sdf['bbox_center_x_frac'].std():.3f}"
    )
    print(
        f"  center y fraction: mean {sdf['bbox_center_y_frac'].mean():.3f}  "
        f"std {sdf['bbox_center_y_frac'].std():.3f}"
    )
    print()
    print(
        "Aspect ratio (close to 1.0 means square bbox - expected for top-down bottles):"
    )
    print(
        f"  mean {sdf['aspect_ratio'].mean():.3f}  std {sdf['aspect_ratio'].std():.3f}"
    )
    print()
    print("Detection confidence:")
    print(
        f"  mean {sdf['score'].mean():.3f}  "
        f"min {sdf['score'].min():.3f}  max {sdf['score'].max():.3f}"
    )
    print(f"  low-confidence (<0.5): {(sdf['score'] < 0.5).sum()}")

    # save stats csv
    out_csv = Path(args.out).with_suffix(".csv")
    sdf.to_csv(out_csv, index=False)
    print(f"\nbbox stats saved: {out_csv}")

    # ---- visual grid ----
    print(f"\ngenerating visual grid of {len(sample)} samples...")
    cols = 5
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for i, img_path in enumerate(sample):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        with Image.open(img_path) as im:
            W, H = im.size
            x = preprocess_for_yolo(im)
            img_rgb = im.convert("RGB")
        out = sess.run(out_names, {in_name: x})[0]
        bbox = parse_yolo_output(out, W, H)

        ax.imshow(img_rgb, cmap="gray")
        ax.set_title(f"{img_path.name[:40]}", fontsize=7)
        ax.axis("off")

        if bbox is None:
            ax.text(
                W / 2,
                H / 2,
                "NO DETECTION",
                color="red",
                fontsize=12,
                ha="center",
                va="center",
                weight="bold",
            )
            continue

        bx, by, bw, bh, score = bbox
        # YOLO's tight bbox (red)
        rect_yolo = patches.Rectangle(
            (bx, by),
            bw,
            bh,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="-",
        )
        ax.add_patch(rect_yolo)
        # Expanded ROI (yellow) - what the classifier actually crops
        x0 = max(0, bx - ROI_PAD)
        y0 = max(0, by - ROI_PAD)
        x1 = min(W, bx + bw + ROI_PAD)
        y1 = min(H, by + bh + ROI_PAD)
        rect_exp = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect_exp)
        ax.text(
            bx + 5,
            by + 25,
            f"{score:.2f}",
            color="red",
            fontsize=9,
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.7, pad=1),
        )

    for j in range(len(sample), rows * cols):
        axes[j // cols, j % cols].axis("off")
    plt.suptitle(
        "YOLO ROI sanity check\n"
        "RED = YOLO bbox      YELLOW dashed = expanded ROI (what classifier crops)",
        fontsize=11,
        y=1.0,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=80, bbox_inches="tight")
    print(f"visual grid saved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", required=True, help="path to YOLO ONNX file")
    p.add_argument(
        "--test-dir",
        required=True,
        help="directory of test images (or train images for sanity)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="how many images to draw in the visual grid",
    )
    p.add_argument(
        "--n-stats",
        type=int,
        default=300,
        help="how many images to compute bbox statistics over",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="yolo_sanity_grid.png")
    args = p.parse_args()
    main(args)
