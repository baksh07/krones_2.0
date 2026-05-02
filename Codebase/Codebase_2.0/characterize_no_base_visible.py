"""
Krones — Characterize "No base visible" images
-----------------------------------------------
For every image whose dominant defect (or any annotation) is "No base visible",
compute statistics of the ROI interior:
  - mean intensity
  - std intensity
  - fraction of pixels below threshold (i.e. how "blank/black" it looks)

Then bucket each image into:
  - 'blank'   : ROI is mostly dark/empty (an ROI-cropped model can still detect)
  - 'partial' : ROI looks like a normal bottle (ROI-cropped model would miss this)
  - 'middle'  : ambiguous

Output:
  - nbv_characterization.csv   : per-image stats and bucket
  - nbv_summary.txt            : ratios per bucket + recommendation
  - nbv_samples/               : grid of sample crops per bucket (for eyeballing)

Usage:
    python characterize_no_base_visible.py --root /path/to/dataset
"""

import os, json, argparse, random
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

# threshold for "dark pixel" — bottles in your set are bright, defects are not
DARK_PIXEL_THR = 40       # 0..255  (anything below this is "dark/empty")
BLANK_FRAC_HI  = 0.70     # ROI >70% dark pixels  -> 'blank'
BLANK_FRAC_LO  = 0.30     # ROI <30% dark pixels  -> 'partial' (looks normal)
N_SAMPLES_PER_BUCKET = 12


def roi_stats(img_path, roi_bbox):
    """Return (mean, std, dark_frac) inside the ROI, in 0..255 grayscale."""
    img = Image.open(img_path).convert("L")
    x, y, w, h = [int(v) for v in roi_bbox]
    x = max(0, x); y = max(0, y)
    crop = img.crop((x, y, x + w, y + h))
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.size == 0:
        return None
    mean = float(arr.mean())
    std  = float(arr.std())
    dark = float((arr < DARK_PIXEL_THR).mean())
    return mean, std, dark


def make_grid(img_paths, out_path, tile=256, cols=4):
    rows = (len(img_paths) + cols - 1) // cols
    grid = Image.new("L", (tile * cols, tile * rows), 0)
    for i, p in enumerate(img_paths):
        try:
            im = Image.open(p).convert("L").resize((tile, tile))
            grid.paste(im, ((i % cols) * tile, (i // cols) * tile))
        except Exception as e:
            print(f"      could not load {p}: {e}")
    grid.save(out_path)


def main(root):
    ann_path  = os.path.join(root, "train_annotations.json")
    img_dir   = os.path.join(root, "train_images")
    out_dir   = os.path.join(root, "audit_output")
    sample_dir = os.path.join(out_dir, "nbv_samples")
    os.makedirs(sample_dir, exist_ok=True)

    print("[1/4] loading annotations ...")
    ann = json.load(open(ann_path, "r"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    name2id = {v: k for k, v in cats.items()}
    if "No base visible" not in name2id:
        print("ERROR: 'No base visible' not in categories. Found:", list(name2id))
        return
    NBV_ID = name2id["No base visible"]
    ROI_ID = name2id["Roi"]

    img_to_roi = {}
    for a in ann["annotations"]:
        if a["category_id"] == ROI_ID:
            img_to_roi[a["image_id"]] = a["bbox"]

    nbv_image_ids = set()
    for a in ann["annotations"]:
        if a["category_id"] == NBV_ID:
            nbv_image_ids.add(a["image_id"])
    print(f"      {len(nbv_image_ids)} unique images with at least one 'No base visible' annotation")

    fname_of = {im["id"]: im["file_name"] for im in ann["images"]}

    print("[2/4] computing ROI stats per image ...")
    rows = []
    missing = 0
    for i, iid in enumerate(sorted(nbv_image_ids), 1):
        if i % 500 == 0:
            print(f"      processed {i}/{len(nbv_image_ids)}")
        fname = fname_of[iid]
        path  = os.path.join(img_dir, fname)
        if not os.path.exists(path):
            missing += 1
            continue
        roi = img_to_roi.get(iid)
        if roi is None:
            # no ROI annotation — use full image
            img = Image.open(path).convert("L")
            roi = [0, 0, img.width, img.height]
        s = roi_stats(path, roi)
        if s is None:
            missing += 1; continue
        mean, std, dark = s
        rows.append({
            "image_id":  iid,
            "file_name": fname,
            "roi_mean":  mean,
            "roi_std":   std,
            "roi_dark_frac": dark,
        })
    if missing:
        print(f"      WARNING: {missing} images skipped (file not found or unreadable)")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("ERROR: no images processed.")
        return

    print("[3/4] bucketing ...")
    def bucket(d):
        if d >= BLANK_FRAC_HI:
            return "blank"
        if d <= BLANK_FRAC_LO:
            return "partial"
        return "middle"
    df["bucket"] = df["roi_dark_frac"].apply(bucket)

    # save csv
    df.sort_values("roi_dark_frac").to_csv(
        os.path.join(out_dir, "nbv_characterization.csv"), index=False)

    # summary
    counts = df["bucket"].value_counts().to_dict()
    n = len(df)
    lines = []
    L = lines.append
    L("="*70)
    L("'No base visible' characterization")
    L("="*70)
    L(f"images analyzed: {n}")
    L(f"  blank   (ROI mostly dark, >{int(BLANK_FRAC_HI*100)}% dark px): "
      f"{counts.get('blank',0):5d}  ({counts.get('blank',0)/n*100:5.2f}%)")
    L(f"  middle                                               : "
      f"{counts.get('middle',0):5d}  ({counts.get('middle',0)/n*100:5.2f}%)")
    L(f"  partial (ROI looks normal, <{int(BLANK_FRAC_LO*100)}% dark px): "
      f"{counts.get('partial',0):5d}  ({counts.get('partial',0)/n*100:5.2f}%)")
    L("")
    L("ROI stats by bucket:")
    L(str(df.groupby("bucket")[["roi_mean","roi_std","roi_dark_frac"]]
            .agg(["mean","median","std"]).round(2)))
    L("")
    L("INTERPRETATION:")
    blank_pct = counts.get('blank', 0) / n
    partial_pct = counts.get('partial', 0) / n
    if partial_pct < 0.10:
        L(f"  >>> Most NBV images are blank/dark ({blank_pct*100:.0f}%).")
        L("  >>> An ROI-cropped model CAN still detect them (ROI interior is dark).")
        L("  >>> ROI-crop training is viable for this class.")
    elif partial_pct > 0.30:
        L(f"  >>> {partial_pct*100:.0f}% of NBV images have a normal-looking ROI interior.")
        L("  >>> An ROI-cropped model will MISS these — the defect is on the rim.")
        L("  >>> Recommend: full image, or expanded ROI (ROI + N pixels padding).")
    else:
        L(f"  >>> Mixed: {blank_pct*100:.0f}% blank, {partial_pct*100:.0f}% partial.")
        L("  >>> Borderline — expanded ROI (ROI + 50–100 px padding) is the safe bet.")
    L("="*70)

    print("\n".join(lines))
    with open(os.path.join(out_dir, "nbv_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    print("[4/4] saving sample grids per bucket ...")
    random.seed(0)
    for b in ["blank", "middle", "partial"]:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            continue
        # take a spread: lowest, middle, highest dark_frac
        sub = sub.sort_values("roi_dark_frac")
        idx_pick = np.linspace(0, len(sub)-1, min(N_SAMPLES_PER_BUCKET, len(sub))).astype(int)
        picks = sub.iloc[idx_pick]
        paths = [os.path.join(img_dir, fn) for fn in picks["file_name"]]
        out_grid = os.path.join(sample_dir, f"grid_{b}.png")
        make_grid(paths, out_grid)
        # also save the file list so you can find them
        picks.to_csv(os.path.join(sample_dir, f"grid_{b}.csv"), index=False)
        print(f"      wrote {out_grid}  ({len(picks)} samples)")

    print(f"\nALL OUTPUTS: {out_dir}")
    print("  nbv_characterization.csv  - per-image stats")
    print("  nbv_summary.txt           - bucket ratios + recommendation")
    print("  nbv_samples/grid_*.png    - sample grids for visual check")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="dataset root with train_images/ and train_annotations.json")
    args = p.parse_args()
    main(args.root)