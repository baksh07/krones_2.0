"""
evaluate_rule.py
----------------
Apply the post-processing rule to YOLO-seg predictions and find the binary-F1
optimum by sweeping confidence/area thresholds.

The rule (any match means Faulty):
  Hard-Faulty       conf >= conf_hard           -> Faulty
  Chip              conf >= conf_chip           -> Faulty   (99.7% cross threshold)
  Glass imp.        conf >= conf_glass          -> Faulty   (97% cross)
  Scuffing heavy    conf >= conf_sheavy         -> Faulty   (65% cross, weak detection)
  Scuffing          conf >= conf_scuf  AND area >= 75000 -> Faulty
  Air bubble        conf >= conf_air   AND area >= 500   -> Faulty
  Cont. light       conf >= conf_cl    AND area >= 180   -> Faulty
  Otherwise -> Good

Usage:
    python evaluate_rule.py \\
        --predictions yolo_seg_val_predictions.csv \\
        --ground_truth /path/to/train_clean.csv \\
        --splits_csv /path/to/splits.csv \\
        --split val
"""

import argparse
from itertools import product

import numpy as np
import pandas as pd


# (safe_name, display_name, area_threshold_or_None)
# area_threshold None => treat as unconditional (any detection above conf -> Faulty)
CLASSES = [
    ("Scuffing", "Scuffing", 75000),
    ("Air_bubble", "Air bubble", 500),
    ("Contamination_light", "Contamination light", 180),
    ("Chip", "Chip", None),
    ("Scuffing_heavy", "Scuffing heavy", None),
    ("Glass_imperfection", "Glass imperfection", None),
    ("Hard_Faulty", "Hard-Faulty", None),
]


def apply_rule(df, conf_thresholds):
    """Apply rule. Returns ndarray of int (1=Faulty, 0=Good)."""
    n = len(df)
    pred = np.zeros(n, dtype=int)
    for safe_name, _, area_th in CLASSES:
        conf_col = f"conf_{safe_name}"
        area_col = f"area_{safe_name}"
        if conf_col not in df.columns:
            continue
        conf_th = conf_thresholds[safe_name]
        cond_conf = df[conf_col].values >= conf_th
        if area_th is not None:
            cond_area = df[area_col].values >= area_th
            cond = cond_conf & cond_area
        else:
            cond = cond_conf
        pred = pred | cond.astype(int)
    return pred


def binary_f1(pred, gt):
    pred = np.asarray(pred).astype(int)
    gt = np.asarray(gt).astype(int)
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1, precision, recall, (tp, fp, fn, tn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument(
        "--ground_truth",
        required=True,
        help="train_clean.csv with file_name and label column",
    )
    p.add_argument("--splits_csv", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--label_col", default="label_recomputed")
    p.add_argument("--out_csv", default=None, help="Save full sweep results")
    p.add_argument("--coarse", action="store_true", help="Coarser sweep (fewer points)")
    args = p.parse_args()

    # Load
    print(f"Loading predictions:  {args.predictions}")
    pred_df = pd.read_csv(args.predictions)
    print(f"  {len(pred_df)} rows")

    print(f"Loading ground truth: {args.ground_truth}")
    gt_df = pd.read_csv(args.ground_truth)

    print(f"Loading splits:       {args.splits_csv}")
    splits_df = pd.read_csv(args.splits_csv)
    split_files = set(splits_df[splits_df["split"] == args.split]["file_name"].tolist())
    print(f"  {len(split_files)} files in split '{args.split}'")

    # Filter and merge
    pred_df = pred_df[pred_df["file_name"].isin(split_files)].copy()
    if args.label_col not in gt_df.columns:
        raise KeyError(f"'{args.label_col}' not in GT cols: {list(gt_df.columns)}")
    label_lookup = dict(zip(gt_df["file_name"], gt_df[args.label_col]))
    pred_df["label"] = pred_df["file_name"].map(label_lookup)
    n_missing = pred_df["label"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: dropping {n_missing} rows with missing labels")
        pred_df = pred_df.dropna(subset=["label"])
    pred_df["label"] = pred_df["label"].astype(int)

    n = len(pred_df)
    n_pos = int(pred_df["label"].sum())
    print(f"  Final: {n} images, {n_pos} Faulty ({n_pos / n:.1%})")

    gt = pred_df["label"].values

    # ---- Stage 1: per-class diagnostic baseline ----
    # Apply each class rule alone, see how it performs.
    print("\n" + "=" * 70)
    print("STAGE 1 — single-class diagnostic (rule = only this class)")
    print("=" * 70)
    print(
        f"  {'Class':<22s} {'conf':>6s} {'F1':>7s} {'P':>7s} {'R':>7s} "
        f"{'TP':>5s} {'FP':>5s} {'FN':>5s}"
    )
    diag_conf = 0.25  # reasonable middle ground for diagnostics
    for safe_name, disp_name, _ in CLASSES:
        confs = {sn: 1.01 for sn, _, _ in CLASSES}  # all OFF
        confs[safe_name] = diag_conf
        pred = apply_rule(pred_df, confs)
        f1, prec, rec, (tp, fp, fn, _) = binary_f1(pred, gt)
        print(
            f"  {disp_name:<22s} {diag_conf:>6.2f} {f1:>7.4f} "
            f"{prec:>7.4f} {rec:>7.4f} {tp:>5d} {fp:>5d} {fn:>5d}"
        )

    # ---- Stage 2: full grid sweep ----
    print("\n" + "=" * 70)
    print("STAGE 2 — full-rule grid sweep")
    print("=" * 70)

    if args.coarse:
        conf_grid = {
            "Hard_Faulty": [0.20, 0.30, 0.40, 0.50],
            "Chip": [0.20, 0.30, 0.40, 0.50],
            "Glass_imperfection": [0.20, 0.30, 0.40, 0.50],
            "Scuffing_heavy": [0.10, 0.20, 0.30, 1.01],  # 1.01 = OFF
            "Scuffing": [0.10, 0.20, 0.30, 1.01],
            "Air_bubble": [0.10, 0.20, 0.30, 1.01],
            "Contamination_light": [0.10, 0.20, 0.30, 1.01],
        }
    else:
        conf_grid = {
            "Hard_Faulty": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
            "Chip": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "Glass_imperfection": [0.10, 0.15, 0.20, 0.30, 1.01],
            "Scuffing_heavy": [0.05, 0.10, 0.20, 1.01],
            "Scuffing": [0.05, 0.10, 0.15, 0.20, 0.30, 1.01],
            "Air_bubble": [0.05, 0.10, 0.15, 0.20, 1.01],
            "Contamination_light": [0.05, 0.10, 0.15, 0.20, 0.30, 1.01],
        }

    keys = list(conf_grid.keys())
    grids = [conf_grid[k] for k in keys]
    n_combos = 1
    for g in grids:
        n_combos *= len(g)
    print(f"  Grid size: {n_combos} combinations across {len(keys)} thresholds")

    best_f1 = 0.0
    best_conf = None
    best_stats = None
    rows = []

    # Iterate, with progress every 5%
    progress_step = max(1, n_combos // 20)
    for i, vals in enumerate(product(*grids)):
        confs = dict(zip(keys, vals))
        pred = apply_rule(pred_df, confs)
        f1, prec, rec, (tp, fp, fn, tn) = binary_f1(pred, gt)
        rows.append(
            {
                **{f"conf_{k}": v for k, v in confs.items()},
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_conf = dict(confs)
            best_stats = (prec, rec, tp, fp, fn, tn)
        if (i + 1) % progress_step == 0:
            print(f"  swept {i + 1}/{n_combos}  current best F1 = {best_f1:.4f}")

    print(f"\n  swept all {n_combos} combos.")

    # ---- Best result ----
    print("\n" + "=" * 70)
    print("BEST CONFIG")
    print("=" * 70)
    prec, rec, tp, fp, fn, tn = best_stats
    print(f"  Binary F1: {best_f1:.4f}")
    print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("\n  Confidence thresholds:")
    for k, v in best_conf.items():
        on_off = " (OFF)" if v >= 1.01 else ""
        print(f"    {k:<24s} {v:.2f}{on_off}")

    # Optional: save full sweep
    if args.out_csv:
        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values("f1", ascending=False).head(200)
        out_df.to_csv(args.out_csv, index=False)
        print(f"\n  Saved top 200 sweep results to {args.out_csv}")

    # Sensitivity analysis on the best config: per-class +/- effect
    print("\n" + "=" * 70)
    print("SENSITIVITY — what happens if we turn each class OFF in best config?")
    print("=" * 70)
    for k in keys:
        confs_off = dict(best_conf)
        confs_off[k] = 1.01
        pred = apply_rule(pred_df, confs_off)
        f1_off, _, _, _ = binary_f1(pred, gt)
        delta = f1_off - best_f1
        print(f"  {k:<24s}  F1 without = {f1_off:.4f}   delta = {delta:+.4f}")


if __name__ == "__main__":
    main()
