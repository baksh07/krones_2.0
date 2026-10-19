"""
Krones — Copy-paste augmentation (with segmentation masks)
-----------------------------------------------------------
SAME BEHAVIOR as build_copypaste_aug.py, except:
  - Also writes binary masks under audit_output/crops_aug_masks/
  - Each mask matches its synth_<class>_<n>.png in crops_aug/
  - Mask = white (255) where the paste happened, black (0) elsewhere
  - Same dimensions as the synth crop (so it aligns 1:1)

For the 8-channel segmentation pipeline, all synth crops contribute to
the merged "always Faulty" channel (channel 7). The mask we save here
IS that channel's ground truth for the synth crop.

Outputs (under <root>/audit_output/):
    crops_aug/synth_<class>_<n>.png            synthesized images (Faulty)   <-- already
    crops_aug_masks/synth_<class>_<n>.png      binary masks (paste region)   <-- NEW
    splits_aug.csv                             splits + synth rows           <-- already
    aug_summary.txt                            counts, paste positions       <-- already

Usage:
    python build_copypaste_aug_with_masks.py --root /path/to/dataset --target 200 --seed 42
"""

import os
import json
import argparse
import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

RARE_CLASSES = {
    "Insect": dict(
        placement="anywhere", max_tile_frac=0.05, rot=(0, 360), scale=(0.85, 1.20)
    ),
    "Foil / Semitransparent": dict(
        placement="anywhere", max_tile_frac=0.10, rot=(0, 360), scale=(0.85, 1.20)
    ),
    "Paint residue": dict(
        placement="anywhere", max_tile_frac=0.05, rot=(0, 360), scale=(0.85, 1.20)
    ),
    "Liquid": dict(
        placement="anywhere", max_tile_frac=0.10, rot=(0, 360), scale=(0.85, 1.20)
    ),
}
DROPPED_CLASSES = ["Circlip", "Straw"]

MIN_DEFECT_AREA = 80
FEATHER_PX = 3
EDGE_MARGIN_FRAC = 0.10
MAX_PLACE_ATTEMPTS = 30


def polygon_to_mask(seg_list, w, h, offset=(0, 0)):
    mask = Image.new("L", (w, h), 0)
    drawer = ImageDraw.Draw(mask)
    ox, oy = offset
    for poly in seg_list:
        xy = [(poly[i] - ox, poly[i + 1] - oy) for i in range(0, len(poly), 2)]
        drawer.polygon(xy, fill=255)
    return np.asarray(mask, dtype=np.uint8)


def extract_defect_tile(img, poly_list, bbox):
    x, y, w, h = [int(round(v)) for v in bbox]
    W, H = img.size
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img.crop((x0, y0, x1, y1)).convert("L")
    mask = polygon_to_mask(poly_list, x1 - x0, y1 - y0, offset=(x0, y0))
    if FEATHER_PX > 0:
        mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(FEATHER_PX))
        mask = np.asarray(mask_img, dtype=np.uint8)
    if mask.sum() < MIN_DEFECT_AREA * 255:
        return None
    rgba = Image.new("LA", (x1 - x0, y1 - y0))
    rgba.putalpha(Image.fromarray(mask))
    rgba.paste(crop, (0, 0))
    rgba.putalpha(Image.fromarray(mask))
    return rgba


def random_transform(tile, rng, rot_range, scale_range):
    angle = rng.uniform(*rot_range)
    scale = rng.uniform(*scale_range)
    new_w = max(8, int(tile.width * scale))
    new_h = max(8, int(tile.height * scale))
    out = tile.resize((new_w, new_h), Image.LANCZOS)
    out = out.rotate(angle, resample=Image.BICUBIC, expand=True)
    return out


def find_paste_location(
    dest_w, dest_h, tile_w, tile_h, roi_in_crop, rng, placement="anywhere"
):
    rx0, ry0, rx1, ry1 = roi_in_crop
    rw = rx1 - rx0
    rh = ry1 - ry0
    mx = int(rw * EDGE_MARGIN_FRAC)
    my = int(rh * EDGE_MARGIN_FRAC)
    rx0 += mx
    ry0 += my
    rx1 -= mx
    ry1 -= my
    if placement == "bottom_half":
        ry0 = ry0 + (ry1 - ry0) // 2
    valid_x_min = rx0
    valid_x_max = rx1 - tile_w
    valid_y_min = ry0
    valid_y_max = ry1 - tile_h
    if valid_x_max <= valid_x_min or valid_y_max <= valid_y_min:
        return None
    px = rng.randint(valid_x_min, valid_x_max)
    py = rng.randint(valid_y_min, valid_y_max)
    return px, py


def alpha_paste(dest_img, tile_la, pos):
    if dest_img.mode != "L":
        dest_img = dest_img.convert("L")
    tile_l = tile_la.convert("L")
    tile_a = tile_la.split()[-1]
    dest_img.paste(tile_l, pos, tile_a)
    return dest_img


def build_paste_mask(dest_w, dest_h, tile_la, pos, alpha_threshold=64):
    """Build a binary mask matching dest_w x dest_h, with the tile's alpha
    pasted at pos. Threshold the soft alpha at alpha_threshold so we get a
    clean binary mask without feathered edges (which the segmenter could
    learn as background noise otherwise).

    NOTE on the threshold: feathered alpha edges are 0..255 with a soft falloff
    of FEATHER_PX pixels. We threshold at 64 (~25%) which captures most of the
    paste while excluding the very softest fringe. This gives a tight binary
    mask of the actual defect region.
    """
    canvas = Image.new("L", (dest_w, dest_h), 0)
    tile_a = tile_la.split()[-1]  # extract alpha channel
    canvas.paste(tile_a, pos)  # place at paste position
    arr = np.asarray(canvas, dtype=np.uint8)
    binary = (arr >= alpha_threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def main(root, target_per_class, seed):
    img_dir = os.path.join(root, "train_images")
    ann_path = os.path.join(root, "train_annotations.json")
    out_dir = os.path.join(root, "audit_output")
    crops_dir = os.path.join(out_dir, "crops")
    aug_dir = os.path.join(out_dir, "crops_aug")
    aug_mask_dir = os.path.join(out_dir, "crops_aug_masks")  # NEW
    os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)  # NEW

    rng = random.Random(seed)
    np.random.seed(seed)

    # 1. load metadata
    print("[1/5] loading metadata ...")
    ann = json.load(open(ann_path, "r"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    name2id = {v: k for k, v in cats.items()}
    fname_of = {im["id"]: im["file_name"] for im in ann["images"]}
    splits_df = pd.read_csv(os.path.join(out_dir, "splits.csv"))
    meta_df = pd.read_csv(os.path.join(out_dir, "crops_meta.csv"))

    available = [c for c in RARE_CLASSES.keys() if c in name2id]
    missing = [c for c in RARE_CLASSES.keys() if c not in name2id]
    if missing:
        print(f"      WARNING: missing classes in annotations: {missing}")

    # 2. collect candidate defect tiles per class
    print("[2/5] collecting defect tiles per class (train split only) ...")
    train_image_ids = set(splits_df[splits_df["split"] == "train"]["image_id"].tolist())

    class_anns = defaultdict(list)
    for a in ann["annotations"]:
        cname = cats[a["category_id"]]
        if cname not in available:
            continue
        if a["image_id"] not in train_image_ids:
            continue
        seg = a.get("segmentation")
        if not seg:
            continue
        if isinstance(seg, dict):
            continue
        class_anns[cname].append((a["image_id"], seg, a["bbox"]))
    for c in available:
        print(f"      {c:30s}: {len(class_anns[c])} candidate annotations")

    # 3. collect Good destination crops
    print("[3/5] collecting destination crops (Good, train only) ...")
    good_train = splits_df[(splits_df["split"] == "train") & (splits_df["label"] == 0)]
    good_train = good_train.merge(meta_df, on=["image_id", "file_name"], how="left")
    print(f"      {len(good_train)} Good train crops available as backgrounds")

    # 4. synthesize
    print(f"[4/5] synthesizing up to {target_per_class} images per class ...")
    n_real_train_per_class = (
        splits_df[
            (splits_df["split"] == "train") & (splits_df["dominant"].isin(available))
        ]
        .groupby("dominant")
        .size()
        .to_dict()
    )

    new_rows = []
    sanity_log = []
    rejected_too_big = Counter()
    for cname in available:
        cfg = RARE_CLASSES[cname]
        anns = class_anns[cname]
        if len(anns) == 0:
            print(f"      [{cname}] no candidate annotations, skipping")
            continue
        n_real = n_real_train_per_class.get(cname, 0)
        n_synth = max(0, target_per_class - n_real)
        if n_synth == 0:
            print(
                f"      [{cname}] already has {n_real} >= target {target_per_class}, no synth"
            )
            continue
        print(
            f"      [{cname}] real={n_real}  synth_target={n_synth}  "
            f"placement={cfg['placement']}  max_tile_frac={cfg['max_tile_frac']}"
        )

        produced = 0
        attempts = 0
        max_attempts = n_synth * 8
        while produced < n_synth and attempts < max_attempts:
            attempts += 1
            src_iid, seg, sbbox = anns[rng.randrange(len(anns))]
            src_path = os.path.join(img_dir, fname_of[src_iid])
            if not os.path.exists(src_path):
                continue
            with Image.open(src_path) as src_img:
                tile = extract_defect_tile(src_img, seg, sbbox)
            if tile is None:
                continue
            tile = random_transform(tile, rng, cfg["rot"], cfg["scale"])

            d = good_train.sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
            dest_path = os.path.join(crops_dir, d["file_name"])
            if not os.path.exists(dest_path):
                continue
            with Image.open(dest_path) as dest_img:
                dW, dH = dest_img.size

                roi_x0 = max(0, d["roi_x"] - d["crop_x0"])
                roi_y0 = max(0, d["roi_y"] - d["crop_y0"])
                roi_x1 = min(dW, roi_x0 + d["roi_w"])
                roi_y1 = min(dH, roi_y0 + d["roi_h"])
                roi_in_crop = (int(roi_x0), int(roi_y0), int(roi_x1), int(roi_y1))

                roi_area = max(1, (roi_x1 - roi_x0) * (roi_y1 - roi_y0))
                tile_area = tile.width * tile.height
                if tile_area > cfg["max_tile_frac"] * roi_area:
                    rejected_too_big[cname] += 1
                    continue

                pos = None
                for _ in range(MAX_PLACE_ATTEMPTS):
                    pos = find_paste_location(
                        dW,
                        dH,
                        tile.width,
                        tile.height,
                        roi_in_crop,
                        rng,
                        placement=cfg["placement"],
                    )
                    if pos:
                        break
                if pos is None:
                    continue

                out_img = alpha_paste(dest_img.copy(), tile, pos)
                out_name = f"synth_{cname.replace(' ', '_').replace('/', '_')}_{produced:04d}.png"
                out_path = os.path.join(aug_dir, out_name)
                out_img.save(out_path, "PNG")

                # NEW: also save the binary paste mask. Same name as composite,
                # under crops_aug_masks/. dW x dH dimensions to match composite.
                mask_img = build_paste_mask(dW, dH, tile, pos)
                mask_path = os.path.join(aug_mask_dir, out_name)
                mask_img.save(mask_path, "PNG")

                new_rows.append(
                    {
                        "image_id": -1,
                        "file_name": out_name,
                        "label": 1,
                        "dominant": cname,
                        "borderline": 0,
                        "strat_key": f"1|{cname}|0",
                        "strat_key_eff": f"1|{cname}|0",
                        "split": "train",
                        "is_synth": 1,
                        "synth_source_image_id": int(src_iid),
                        "synth_dest_file": d["file_name"],
                        "paste_x": int(pos[0]),
                        "paste_y": int(pos[1]),
                        "paste_w": int(tile.width),
                        "paste_h": int(tile.height),
                    }
                )
                produced += 1
        sanity_log.append((cname, n_real, produced, attempts, rejected_too_big[cname]))
        print(
            f"      [{cname}] produced {produced}/{n_synth}  "
            f"(attempts={attempts}, size-rejected={rejected_too_big[cname]})"
        )

    # 5. write augmented splits + summary
    print("[5/5] writing splits_aug.csv and summary ...")
    if "is_synth" not in splits_df.columns:
        splits_df["is_synth"] = 0
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        for col in splits_df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[
            list(splits_df.columns)
            + [c for c in new_df.columns if c not in splits_df.columns]
        ]
        out_df = pd.concat(
            [
                splits_df.assign(
                    is_synth=0,
                    synth_source_image_id=None,
                    synth_dest_file=None,
                    paste_x=None,
                    paste_y=None,
                    paste_w=None,
                    paste_h=None,
                ),
                new_df,
            ],
            ignore_index=True,
        )
    else:
        out_df = splits_df.assign(is_synth=0)

    out_df.to_csv(os.path.join(out_dir, "splits_aug.csv"), index=False)

    lines = []
    L = lines.append
    L("=" * 70)
    L("COPY-PASTE AUGMENTATION SUMMARY (with segmentation masks)")
    L("=" * 70)
    L(f"target per class : {target_per_class}")
    L(f"seed             : {seed}")
    L(f"composite dir    : {aug_dir}")
    L(f"mask dir         : {aug_mask_dir}")
    L("")
    L("DROPPED from copy-paste:")
    for c in DROPPED_CLASSES:
        n_real_dropped = splits_df[
            (splits_df["split"] == "train") & (splits_df["dominant"] == c)
        ].shape[0]
        L(f"  - {c:<28s} ({n_real_dropped} real train images, no synth)")
    L("")
    L(
        f"{'class':<28s} {'real_train':>11s} {'synth_made':>11s} {'final':>7s} {'attempts':>9s} {'too_big':>9s}"
    )
    for cname, n_real, produced, attempts, too_big in sanity_log:
        L(
            f"{cname:<28s} {n_real:>11d} {produced:>11d} {n_real + produced:>7d} {attempts:>9d} {too_big:>9d}"
        )
    L("")
    total_real = sum(r for _, r, _, _, _ in sanity_log)
    total_synth = sum(p for _, _, p, _, _ in sanity_log)
    L(
        f"totals: real={total_real}  synth={total_synth}  combined={total_real + total_synth}"
    )
    L("")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "aug_summary.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument(
        "--target", type=int, default=200, help="target real+synth count per rare class"
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.root, args.target, args.seed)
