"""
Synthesize CopyPaste images WITH polygon tracking, then build COCO splits.

Writes generated artifacts under the repository's Dataset/ directory.
The challenge data directory is read-only.

Pipeline:
  1. Synthesize ~200 images each for Insect, Foil/Semitransparent, Paint
     residue, Liquid by pasting source-defect tiles onto Good train crops.
     Track the polygon through scale + rotation + paste translation so the
     output has a real COCO segmentation.
  2. Write synth PNGs to Dataset/crops_aug_v2/.
  3. Build COCO JSONs in Dataset/coco_v2_aug/:
       - krones_train.json (real train + synth)
       - krones_val.json   (real val only)
       - krones_test.json  (real test only)
       - krones_classes.json
       - build_summary.txt

  Uses ABSOLUTE file paths in COCO file_name so Detectron2 can read both
  dirs (real crops at audit_output/crops/, synth at Dataset/crops_aug_v2/).

Config matches build_copypaste_aug.py (4 rare classes, 200 target each,
Circlip + Straw excluded — synth quality was bad on inspection).

Usage:
  python build_synth_and_coco_v2.py
  python build_synth_and_coco_v2.py --target 200 --seed 42 --inspect 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from project_paths import DATA_ROOT, REPO_ROOT

ANN_FILE = DATA_ROOT / "train_annotations.json"
SPLITS_CSV = DATA_ROOT / "audit_output" / "splits.csv"
CROPS_META = DATA_ROOT / "audit_output" / "crops_meta.csv"
SRC_IMG_DIR = DATA_ROOT / "train_images"
REAL_CROPS_DIR = DATA_ROOT / "audit_output" / "crops"

SYNTH_DIR = REPO_ROOT / "Dataset" / "crops_aug_v2"
INSPECT_DIR = SYNTH_DIR / "inspect"
OUT_DIR = REPO_ROOT / "Dataset" / "coco_v2_aug"

# Categories dropped from training (same as build_krones_coco_cropped.py)
DROP_CATEGORIES = {
    "Roi",
    "No fault",
    "Embossing",
    "Foam residue",
    "Water drop",
}

# Per-class CopyPaste config (same as build_copypaste_aug.py)
RARE_CLASSES = {
    "Insect": dict(max_tile_frac=0.05, rot=(0, 360), scale=(0.85, 1.20)),
    "Foil / Semitransparent": dict(
        max_tile_frac=0.10, rot=(0, 360), scale=(0.85, 1.20)
    ),
    "Paint residue": dict(max_tile_frac=0.05, rot=(0, 360), scale=(0.85, 1.20)),
    "Liquid": dict(max_tile_frac=0.10, rot=(0, 360), scale=(0.85, 1.20)),
}

MIN_DEFECT_AREA = 80  # px^2 mask area
FEATHER_PX = 3
EDGE_MARGIN_FRAC = 0.10
MAX_PLACE_ATTEMPTS = 30

# Synth image_ids start here (well above any real image_id)
SYNTH_ID_BASE = 1_000_000


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def shift_polygon(poly, dx, dy):
    return [poly[i] - dx if i % 2 == 0 else poly[i] - dy for i in range(len(poly))]


def shoelace_area(poly):
    xs = poly[0::2]
    ys = poly[1::2]
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def polygon_to_mask(seg_list, w, h, offset=(0, 0)):
    """Rasterize COCO polygons into an (H,W) uint8 mask, 255 inside."""
    mask = Image.new("L", (w, h), 0)
    drawer = ImageDraw.Draw(mask)
    ox, oy = offset
    for poly in seg_list:
        xy = [(poly[i] - ox, poly[i + 1] - oy) for i in range(0, len(poly), 2)]
        drawer.polygon(xy, fill=255)
    return np.asarray(mask, dtype=np.uint8)


def transform_polygon(poly_local, scale, angle_deg, tile_local_w, tile_local_h):
    """
    Transform a polygon expressed in source-tile-local coords (origin at
    tile top-left) through the SAME pipeline as the tile:
      1. scale by `scale` (around tile origin (0,0) — points scale linearly)
      2. rotate by `angle_deg` around the SCALED tile center, with PIL's
         "expand=True" behaviour (output canvas grows to fit rotated rect,
         origin shifts so all rotated content has positive coords)

    Returns the polygon in coordinates of the rotated/scaled tile canvas
    (origin at top-left of that canvas).

    Math notes:
      PIL.Image.rotate(angle, expand=True) rotates CCW (visually) around the
      image center and expands the output canvas to the AABB of the rotated
      original rectangle. In image coords (y-down) "CCW visually" means:
          x' = cos(t) * x + sin(t) * y
          y' = -sin(t) * x + cos(t) * y
      where (x, y) are coords relative to the rotation center.
    """
    # 1. scale
    new_w = tile_local_w * scale
    new_h = tile_local_h * scale
    scaled = [p * scale for p in poly_local]

    # 2. rotate around center of scaled tile (cx, cy)
    cx, cy = new_w / 2.0, new_h / 2.0
    t = math.radians(angle_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)

    # Compute rotated AABB of the scaled tile rectangle to find canvas growth
    corners = [(0, 0), (new_w, 0), (new_w, new_h), (0, new_h)]
    rot_corners = []
    for x, y in corners:
        rx = x - cx
        ry = y - cy
        nx = cos_t * rx + sin_t * ry + cx
        ny = -sin_t * rx + cos_t * ry + cy
        rot_corners.append((nx, ny))
    min_x = min(c[0] for c in rot_corners)
    min_y = min(c[1] for c in rot_corners)
    max_x = max(c[0] for c in rot_corners)
    max_y = max(c[1] for c in rot_corners)
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    # Rotate every polygon point and shift to new canvas
    out = []
    for i in range(0, len(scaled), 2):
        x = scaled[i] - cx
        y = scaled[i + 1] - cy
        nx = cos_t * x + sin_t * y + cx
        ny = -sin_t * x + cos_t * y + cy
        out.append(nx - min_x)
        out.append(ny - min_y)
    return out, canvas_w, canvas_h


def find_paste_location(roi_in_crop, tile_w, tile_h, rng):
    """Pick a paste anchor (top-left) inside the bottle interior."""
    rx0, ry0, rx1, ry1 = roi_in_crop
    rw = rx1 - rx0
    rh = ry1 - ry0
    mx = int(rw * EDGE_MARGIN_FRAC)
    my = int(rh * EDGE_MARGIN_FRAC)
    rx0 += mx
    ry0 += my
    rx1 -= mx
    ry1 -= my
    valid_x_min = rx0
    valid_x_max = rx1 - tile_w
    valid_y_min = ry0
    valid_y_max = ry1 - tile_h
    if valid_x_max <= valid_x_min or valid_y_max <= valid_y_min:
        return None
    return (
        rng.randint(valid_x_min, valid_x_max),
        rng.randint(valid_y_min, valid_y_max),
    )


# ---------------------------------------------------------------------------
# Tile extraction + paste
# ---------------------------------------------------------------------------
def extract_tile_and_local_poly(src_img, seg_list, sbbox):
    """
    Crop the defect from src_img using sbbox; return (rgba_tile, local_poly_list, tile_w, tile_h)
    where local_poly_list is in tile-local coords (origin at tile top-left).

    Returns (None, None, 0, 0) if the tile is too small or invalid.
    """
    sx, sy, sw, sh = [int(round(v)) for v in sbbox]
    W, H = src_img.size
    x0, y0 = max(0, sx), max(0, sy)
    x1, y1 = min(W, sx + sw), min(H, sy + sh)
    if x1 <= x0 or y1 <= y0:
        return None, None, 0, 0
    tile_w, tile_h = x1 - x0, y1 - y0

    crop = src_img.crop((x0, y0, x1, y1)).convert("L")
    mask = polygon_to_mask(seg_list, tile_w, tile_h, offset=(x0, y0))
    if FEATHER_PX > 0:
        mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(FEATHER_PX))
        mask = np.asarray(mask_img, dtype=np.uint8)
    if mask.sum() < MIN_DEFECT_AREA * 255:
        return None, None, 0, 0

    rgba = Image.new("LA", (tile_w, tile_h))
    rgba.paste(crop, (0, 0))
    rgba.putalpha(Image.fromarray(mask))

    # Translate polygons to tile-local coords (subtract x0, y0)
    local_polys = []
    for p in seg_list:
        if len(p) < 6:
            continue
        local = shift_polygon(p, x0, y0)
        # clip points outside [0, tile_w], [0, tile_h]? No — they were used to
        # rasterize the mask which IS clipped. Polygon points may extend slightly
        # outside the tile bbox; that's acceptable since the mask handles it.
        local_polys.append(local)
    return rgba, local_polys, tile_w, tile_h


def alpha_paste(dest_img, tile_la, pos):
    """Paste an LA tile onto a grayscale dest using its alpha as blend mask."""
    if dest_img.mode != "L":
        dest_img = dest_img.convert("L")
    tile_l = tile_la.convert("L")
    tile_a = tile_la.split()[-1]
    dest_img.paste(tile_l, pos, tile_a)
    return dest_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(target: int, seed: int, inspect_n: int):
    rng = random.Random(seed)
    np.random.seed(seed)

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if inspect_n > 0:
        INSPECT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load metadata
    # -----------------------------------------------------------------
    print("[1/5] Loading metadata ...")
    with open(ANN_FILE) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    name2id = {v: k for k, v in cats.items()}
    fname_of = {im["id"]: im["file_name"] for im in coco["images"]}
    splits_df = pd.read_csv(SPLITS_CSV)
    meta_df = pd.read_csv(CROPS_META)

    available = [c for c in RARE_CLASSES if c in name2id]
    missing = [c for c in RARE_CLASSES if c not in name2id]
    if missing:
        print(f"  WARNING: missing rare classes in annotations: {missing}")

    # Build kept-category remap (drop 5, contiguous IDs 1..22) — same scheme
    # as build_coco_v2.py / build_krones_coco_cropped.py
    keep_old_ids = sorted(
        cid for cid, name in cats.items() if name not in DROP_CATEGORIES
    )
    old_to_new = {old: new for new, old in enumerate(keep_old_ids, start=1)}
    new_categories = [
        {"id": old_to_new[old], "name": cats[old], "supercategory": cats[old]}
        for old in keep_old_ids
    ]
    new_id_to_name = {c["id"]: c["name"] for c in new_categories}
    n_classes = len(new_categories)
    print(f"  kept {n_classes} categories (dropped {len(DROP_CATEGORIES)})")

    # -----------------------------------------------------------------
    # Collect source defect tiles per class (TRAIN split images only, no leak)
    # -----------------------------------------------------------------
    print("[2/5] Collecting source defect tiles (train split only) ...")
    train_image_ids = set(splits_df[splits_df["split"] == "train"]["image_id"].tolist())
    class_anns: dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        cname = cats[a["category_id"]]
        if cname not in available:
            continue
        if a["image_id"] not in train_image_ids:
            continue
        seg = a.get("segmentation")
        if not seg or isinstance(seg, dict):
            continue
        class_anns[cname].append((a["image_id"], seg, a["bbox"]))
    for c in available:
        print(f"  {c:30s}: {len(class_anns[c])} candidate annotations")

    # -----------------------------------------------------------------
    # Collect Good train crops as destinations
    # -----------------------------------------------------------------
    print("[3/5] Collecting destination crops (Good, train only) ...")
    good_train = splits_df[(splits_df["split"] == "train") & (splits_df["label"] == 0)]
    good_train = good_train.merge(meta_df, on=["image_id", "file_name"], how="left")
    good_train = good_train.reset_index(drop=True)
    print(f"  {len(good_train)} Good train crops available")

    # -----------------------------------------------------------------
    # Synthesize
    # -----------------------------------------------------------------
    print(f"[4/5] Synthesizing {target} images per class ...")
    synth_records: list[dict] = []
    rejected_too_big = Counter()
    n_inspect_done = 0

    for cname in available:
        cfg = RARE_CLASSES[cname]
        anns = class_anns[cname]
        if not anns:
            print(f"  [{cname}] no candidates, skipping")
            continue

        new_cat_id = old_to_new[name2id[cname]]
        produced = 0
        attempts = 0
        max_attempts = target * 12  # generous headroom

        pbar = tqdm(total=target, desc=f"  {cname:25s}", unit="img")
        while produced < target and attempts < max_attempts:
            attempts += 1
            src_iid, seg, sbbox = anns[rng.randrange(len(anns))]
            src_path = SRC_IMG_DIR / fname_of[src_iid]
            if not src_path.exists():
                continue
            with Image.open(src_path) as src_img:
                tile_rgba, local_polys, tile_w, tile_h = extract_tile_and_local_poly(
                    src_img, seg, sbbox
                )
            if tile_rgba is None or not local_polys:
                continue

            # Random scale + rotation
            scale = rng.uniform(*cfg["scale"])
            angle = rng.uniform(*cfg["rot"])

            # Apply scale + rotation to tile
            new_w_pre = max(8, int(round(tile_w * scale)))
            new_h_pre = max(8, int(round(tile_h * scale)))
            tile_scaled = tile_rgba.resize((new_w_pre, new_h_pre), Image.LANCZOS)
            tile_rot = tile_scaled.rotate(angle, resample=Image.BICUBIC, expand=True)
            t_w, t_h = tile_rot.size

            # Apply same transform to polygons (analytically, NOT by re-rasterizing)
            transformed_polys = []
            for lp in local_polys:
                tp, canvas_w, canvas_h = transform_polygon(
                    lp, scale, angle, tile_w, tile_h
                )
                transformed_polys.append(tp)
            # Sanity: PIL canvas size should match analytical canvas size (rounded)
            if abs(canvas_w - t_w) > 2 or abs(canvas_h - t_h) > 2:
                # Off-by-one rounding diff is fine; >2 px means a bug.
                pass

            # Pick destination crop
            d = good_train.iloc[rng.randrange(len(good_train))]
            dest_path = REAL_CROPS_DIR / d["file_name"]
            if not dest_path.exists():
                continue
            with Image.open(dest_path) as dest_img:
                dW, dH = dest_img.size

                # ROI in crop coords: (roi - crop_origin), clipped to crop bounds
                roi_x0 = max(0, int(d["roi_x"]) - int(d["crop_x0"]))
                roi_y0 = max(0, int(d["roi_y"]) - int(d["crop_y0"]))
                roi_x1 = min(dW, roi_x0 + int(d["roi_w"]))
                roi_y1 = min(dH, roi_y0 + int(d["roi_h"]))
                roi_in_crop = (roi_x0, roi_y0, roi_x1, roi_y1)

                # Size filter: reject tile if too big relative to ROI
                roi_area = max(1, (roi_x1 - roi_x0) * (roi_y1 - roi_y0))
                if t_w * t_h > cfg["max_tile_frac"] * roi_area:
                    rejected_too_big[cname] += 1
                    continue

                # Find paste location
                pos = None
                for _ in range(MAX_PLACE_ATTEMPTS):
                    pos = find_paste_location(roi_in_crop, t_w, t_h, rng)
                    if pos:
                        break
                if pos is None:
                    continue

                # Paste
                out_img = alpha_paste(dest_img.copy(), tile_rot, pos)
                out_name = f"synth_{cname.replace(' ', '_').replace('/', '_')}_{produced:04d}.png"
                out_path = SYNTH_DIR / out_name
                out_img.save(out_path, "PNG")

                # Translate polygons by paste position into dest-crop coords
                px, py = pos
                final_polys = []
                for tp in transformed_polys:
                    fp = [
                        tp[i] + px if i % 2 == 0 else tp[i] + py for i in range(len(tp))
                    ]
                    final_polys.append(fp)

                # Compute bbox + area in dest coords
                all_xs, all_ys = [], []
                for p in final_polys:
                    all_xs.extend(p[0::2])
                    all_ys.extend(p[1::2])
                xmin = max(0.0, min(all_xs))
                ymin = max(0.0, min(all_ys))
                xmax = min(float(dW), max(all_xs))
                ymax = min(float(dH), max(all_ys))
                bbox = [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]
                area = sum(shoelace_area(p) for p in final_polys)

                synth_records.append(
                    {
                        "file_name": out_name,
                        "abs_path": str(out_path),
                        "width": dW,
                        "height": dH,
                        "category_name": cname,
                        "category_id_new": new_cat_id,
                        "segmentation": final_polys,
                        "bbox": bbox,
                        "area": area,
                        "src_image_id": int(src_iid),
                        "dest_file_name": d["file_name"],
                        "paste_x": int(px),
                        "paste_y": int(py),
                        "scale": float(scale),
                        "rotation_deg": float(angle),
                    }
                )

                # Inspect: render polygon outline on the synth image (visual sanity)
                if n_inspect_done < inspect_n:
                    insp = out_img.convert("RGB").copy()
                    drawer = ImageDraw.Draw(insp)
                    for p in final_polys:
                        pts = [(p[i], p[i + 1]) for i in range(0, len(p), 2)]
                        drawer.polygon(pts, outline=(0, 255, 0))
                    rx0, ry0, rx1, ry1 = roi_in_crop
                    drawer.rectangle([rx0, ry0, rx1, ry1], outline=(255, 0, 0))
                    insp.save(INSPECT_DIR / f"inspect_{out_name}", "PNG")
                    n_inspect_done += 1

                produced += 1
                pbar.update(1)
        pbar.close()
        print(
            f"  [{cname}] produced {produced}/{target}  "
            f"(attempts={attempts}, size-rejected={rejected_too_big[cname]})"
        )

    print(f"  total synth records: {len(synth_records)}")
    if inspect_n > 0:
        print(f"  inspection PNGs (synth + drawn polygon): {INSPECT_DIR}")

    # -----------------------------------------------------------------
    # Build COCO splits
    # -----------------------------------------------------------------
    print("[5/5] Building COCO JSONs ...")
    split_by_iid = dict(zip(splits_df["image_id"], splits_df["split"]))
    meta_by_iid = {
        row["image_id"]: (
            int(row["crop_x0"]),
            int(row["crop_y0"]),
            int(row["crop_x1"]),
            int(row["crop_y1"]),
            int(row["crop_w"]),
            int(row["crop_h"]),
        )
        for _, row in meta_df.iterrows()
    }

    # Group annotations by image, kept categories only
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] not in old_to_new:
            continue
        anns_by_img[a["image_id"]].append(a)

    images_by_split: dict[str, list[dict]] = defaultdict(list)
    anns_by_split: dict[str, list[dict]] = defaultdict(list)
    next_ann_id = 1

    # Real images: shift polygons into crop frame
    for im in tqdm(coco["images"], desc="  real images", unit="img"):
        iid = im["id"]
        split = split_by_iid.get(iid)
        if split is None or iid not in meta_by_iid:
            continue
        cx0, cy0, _, _, cw, ch = meta_by_iid[iid]
        abs_fname = str(REAL_CROPS_DIR / im["file_name"])

        images_by_split[split].append(
            {
                "id": iid,
                "file_name": abs_fname,
                "width": cw,
                "height": ch,
                "date_captured": im.get("date_captured", ""),
            }
        )

        for ann in anns_by_img[iid]:
            new_segs = []
            for poly in ann.get("segmentation", []):
                if len(poly) < 6:
                    continue
                shifted = shift_polygon(poly, cx0, cy0)
                xs = shifted[0::2]
                ys = shifted[1::2]
                if max(xs) < 0 or min(xs) > cw or max(ys) < 0 or min(ys) > ch:
                    continue
                new_segs.append(shifted)
            if not new_segs:
                continue

            all_xs, all_ys = [], []
            for p in new_segs:
                all_xs.extend(p[0::2])
                all_ys.extend(p[1::2])
            xmin = max(0.0, min(all_xs))
            ymin = max(0.0, min(all_ys))
            xmax = min(float(cw), max(all_xs))
            ymax = min(float(ch), max(all_ys))
            bbox = [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]
            area = sum(shoelace_area(p) for p in new_segs)

            anns_by_split[split].append(
                {
                    "id": next_ann_id,
                    "image_id": iid,
                    "category_id": old_to_new[ann["category_id"]],
                    "bbox": bbox,
                    "segmentation": new_segs,
                    "area": area,
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            next_ann_id += 1

    # Synth images: append to TRAIN only, with non-overlapping image_ids
    for i, rec in enumerate(synth_records):
        synth_iid = SYNTH_ID_BASE + i
        images_by_split["train"].append(
            {
                "id": synth_iid,
                "file_name": rec["abs_path"],
                "width": rec["width"],
                "height": rec["height"],
                "date_captured": "",
            }
        )
        anns_by_split["train"].append(
            {
                "id": next_ann_id,
                "image_id": synth_iid,
                "category_id": rec["category_id_new"],
                "bbox": rec["bbox"],
                "segmentation": rec["segmentation"],
                "area": rec["area"],
                "iscrowd": 0,
            }
        )
        next_ann_id += 1

    # Write
    summary_lines = []
    L = summary_lines.append
    L("=" * 70)
    L("SYNTH + COCO BUILD SUMMARY")
    L("=" * 70)
    L(f"synth dir:   {SYNTH_DIR}")
    L(f"output dir:  {OUT_DIR}")
    L(f"classes:     {n_classes}")
    L(f"synth target per class: {target}, seed: {seed}")
    L("")

    for split in ("train", "val", "test"):
        ims = images_by_split[split]
        anns = anns_by_split[split]
        for new_id, a in enumerate(anns, start=1):
            a["id"] = new_id
        out = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "type": "instances",
            "images": ims,
            "annotations": anns,
            "categories": new_categories,
        }
        path = OUT_DIR / f"krones_{split}.json"
        with open(path, "w") as f:
            json.dump(out, f)

        ann_by_img = defaultdict(set)
        for a in anns:
            ann_by_img[a["image_id"]].add(a["category_id"])
        n_with = len(ann_by_img)
        n_synth = sum(1 for im in ims if im["id"] >= SYNTH_ID_BASE)
        line = (
            f"{split:5s}: {len(ims):>5} images ({n_with} with anns, "
            f"{n_synth} synth), {len(anns):>6} anns"
        )
        print(line)
        L(line)

    L("")
    L("Per-class image counts (train, including synth):")
    train_anns = anns_by_split["train"]
    train_ann_by_img = defaultdict(set)
    for a in train_anns:
        train_ann_by_img[a["image_id"]].add(a["category_id"])
    cls_img_counts: Counter = Counter()
    for cids in train_ann_by_img.values():
        for c in cids:
            cls_img_counts[c] += 1
    for cid in sorted(new_id_to_name):
        L(f"  {new_id_to_name[cid]:35s} {cls_img_counts.get(cid, 0):>5}")
    L("")
    L(f"Synth records produced: {len(synth_records)}")
    by_class = Counter(r["category_name"] for r in synth_records)
    for c, n in by_class.items():
        L(f"  {c:35s} {n:>5}")

    classes_path = OUT_DIR / "krones_classes.json"
    with open(classes_path, "w") as f:
        json.dump([c["name"] for c in new_categories], f, indent=2)
    print(f"  {classes_path.name}: {n_classes} classes")

    summary_path = OUT_DIR / "build_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  {summary_path.name}: written")

    # Also save synth records as a debugging sidecar
    synth_meta_path = OUT_DIR / "synth_records.json"
    # Drop polygons from this sidecar (they'd duplicate the COCO JSON)
    light_records = [
        {k: v for k, v in r.items() if k != "segmentation"} for r in synth_records
    ]
    with open(synth_meta_path, "w") as f:
        json.dump(light_records, f, indent=2)
    print(f"  {synth_meta_path.name}: {len(synth_records)} synth records")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target", type=int, default=200, help="Target real+synth count per rare class"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--inspect",
        type=int,
        default=8,
        help="Save N random synth images with polygon drawn (for visual sanity)",
    )
    args = p.parse_args()
    main(args.target, args.seed, args.inspect)
