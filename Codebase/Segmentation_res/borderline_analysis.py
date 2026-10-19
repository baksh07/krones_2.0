"""
Krones — Borderline analysis
-----------------------------
Reads existing TTA predictions and answers:
1. What fraction of test images are borderline (prob 0.3-0.7)?
2. Which dominant defect classes are over-represented in the borderline zone?
3. For Scuffing specifically, what's the area distribution of borderline cases?
   Are they near the 75k threshold (ambiguous border) or spread out (model bias)?
4. Would routing borderline cases to a segmenter make sense?

This is purely a diagnostic - no model running, just analysis on existing
predictions + annotation areas.

Usage:
    python borderline_analysis.py \\
        --eval-csv runs/resnet50_640/eval_test.csv \\
        --root /path/to/dataset
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

CONDITIONAL = {
    "Scuffing": 75000,
    "Air bubble": 500,
    "Contamination light": 180,
    "Chip": 200,
    "Scuffing heavy": 1200,
    "Glass imperfection": 100,
}

LOWER_BORDER = 0.3
UPPER_BORDER = 0.7


def main(args):
    print(f"reading eval csv: {args.eval_csv}")
    df = pd.read_csv(args.eval_csv)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # join with splits.csv to get image_id, then with annotations to get areas
    splits = pd.read_csv(Path(args.root) / "audit_output" / "splits.csv")
    split_name = "test" if "test" in args.eval_csv else "val"
    split_rows = splits[splits["split"] == split_name].reset_index(drop=True)
    if len(split_rows) != len(df):
        print(f"WARNING length mismatch: split_rows={len(split_rows)} eval={len(df)}")
        return
    df["image_id"] = split_rows["image_id"].values
    df["file_name"] = split_rows["file_name"].values

    # Load annotations to compute per-image conditional class areas
    print(f"loading annotations from {args.root}/train_annotations.json...")
    with open(Path(args.root) / "train_annotations.json") as f:
        ann = json.load(f)
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    img_class_area = defaultdict(lambda: Counter())
    for a in ann["annotations"]:
        cn = cats[a["category_id"]]
        if cn in CONDITIONAL:
            img_class_area[a["image_id"]][cn] += float(a.get("area", 0.0))
    df["cond_area"] = df.apply(
        lambda r: img_class_area.get(r["image_id"], {}).get(r["class"], 0.0),
        axis=1,
    )
    df["error"] = (df["label"] != df["predbest"]).astype(int)

    # Question 1: what fraction is borderline?
    border = df[(df["prob"] >= LOWER_BORDER) & (df["prob"] <= UPPER_BORDER)]
    print()
    print("=" * 70)
    print(f"BORDERLINE ZONE: prob in [{LOWER_BORDER}, {UPPER_BORDER}]")
    print("=" * 70)
    print(f"borderline: {len(border)} / {len(df)} = {len(border) / len(df) * 100:.2f}%")

    # of these borderline cases, how many are errors vs correct?
    n_b_err = border["error"].sum()
    print(
        f"  of which errors: {n_b_err}/{len(border)} = "
        f"{n_b_err / max(1, len(border)) * 100:.1f}%"
    )
    print(
        f"  (compare: overall error rate is {df['error'].sum() / len(df) * 100:.2f}%)"
    )

    # Question 2: class composition of borderline
    print()
    print("per-class composition of the borderline zone:")
    print("  (n_in_borderline = how many of this class are in [0.3, 0.7])")
    print("  (frac = what fraction of THIS CLASS lives in the borderline zone)")
    rows = []
    for cls in sorted(df["class"].unique()):
        n_total = (df["class"] == cls).sum()
        n_b = (
            (df["class"] == cls)
            & (df["prob"] >= LOWER_BORDER)
            & (df["prob"] <= UPPER_BORDER)
        ).sum()
        n_b_err = (
            (df["class"] == cls)
            & (df["prob"] >= LOWER_BORDER)
            & (df["prob"] <= UPPER_BORDER)
            & (df["error"] == 1)
        ).sum()
        rows.append(
            {
                "class": cls,
                "n_total": int(n_total),
                "n_in_borderline": int(n_b),
                "frac_borderline": float(n_b / max(1, n_total)),
                "n_b_errors": int(n_b_err),
                "is_conditional": cls in CONDITIONAL,
            }
        )
    bdf = pd.DataFrame(rows).sort_values("n_in_borderline", ascending=False)
    print(bdf.to_string(index=False))

    n_cond_in_b = bdf[bdf["is_conditional"]]["n_in_borderline"].sum()
    print(f"\n  total borderline from conditional classes: {n_cond_in_b}")
    print(
        f"  total borderline from non-conditional classes: {len(border) - n_cond_in_b}"
    )

    # Question 3: For Scuffing, where do the borderline cases sit on the area axis?
    print()
    print("=" * 70)
    print("SCUFFING area distribution in the borderline zone (threshold=75000)")
    print("=" * 70)
    scuff = df[df["class"] == "Scuffing"].copy()
    scuff_b = scuff[
        (scuff["prob"] >= LOWER_BORDER) & (scuff["prob"] <= UPPER_BORDER)
    ].copy()
    print(f"Scuffing samples in test: {len(scuff)}")
    print(f"Scuffing in borderline zone: {len(scuff_b)}")
    if len(scuff_b) > 0:
        print("\n  area buckets of borderline Scuffing:")
        bins = [
            (0, 1000),
            (1000, 10000),
            (10000, 50000),
            (50000, 75000),
            (75000, 100000),
            (100000, 1e9),
        ]
        for lo, hi in bins:
            n = ((scuff_b["cond_area"] >= lo) & (scuff_b["cond_area"] < hi)).sum()
            n_correct = (
                (scuff_b["cond_area"] >= lo)
                & (scuff_b["cond_area"] < hi)
                & (scuff_b["error"] == 0)
            ).sum()
            print(
                f"    area [{lo:>7}, {hi:>7}): {n:>3d} cases  "
                f"correct {n_correct}, errors {n - n_correct}"
            )

    # how many borderline Scuffing cases are NEAR the threshold (might benefit from segmenter)
    near_thr = (
        (scuff_b["cond_area"] >= 50000) & (scuff_b["cond_area"] <= 100000)
    ).sum()
    far_below = (scuff_b["cond_area"] < 50000).sum()
    far_above = (scuff_b["cond_area"] > 100000).sum()
    print(
        f"\n  near threshold (50k-100k): {near_thr}  "
        f"(true ambiguous cases — segmenter can help)"
    )
    print(
        f"  far below (< 50k):         {far_below}  "
        f"(should be Good — segmenter would say no)"
    )
    print(
        f"  far above (> 100k):        {far_above}  "
        f"(should be Faulty — segmenter would confirm)"
    )

    # Question 4: How many errors specifically would the hybrid catch?
    print()
    print("=" * 70)
    print("WOULD HYBRID INFERENCE FIX THESE ERRORS?")
    print("=" * 70)
    print("Reasoning: the segmenter computes Scuffing area, applies threshold 75000.")
    print("For each error in the borderline zone, what does the area threshold say?")
    print()
    # for borderline cases that are ERRORS in conditional classes, would
    # applying the area threshold yield the correct answer?
    borderline_errors = df[
        (df["prob"] >= LOWER_BORDER) & (df["prob"] <= UPPER_BORDER) & (df["error"] == 1)
    ]
    n_savable = 0
    n_total_borderline_errors = 0
    for _, row in borderline_errors.iterrows():
        cls = row["class"]
        if cls not in CONDITIONAL:
            n_total_borderline_errors += 1
            continue
        n_total_borderline_errors += 1
        thresh = CONDITIONAL[cls]
        true_label = row["label"]
        # what would area threshold say?
        area_says_faulty = row["cond_area"] > thresh
        area_label = 1 if area_says_faulty else 0
        if area_label == true_label:
            n_savable += 1
    print(f"borderline errors: {n_total_borderline_errors}")
    print(f"  of which would be FIXED by area threshold: {n_savable}")
    print(f"  remaining unfixed: {n_total_borderline_errors - n_savable}")
    print()
    print("NOTE: this is an UPPER BOUND — assumes the segmenter perfectly recovers")
    print("the annotation polygon area. In practice the segmenter has its own errors,")
    print("so realistic gain is ~50-70% of this upper bound.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-csv",
        required=True,
        help="path to eval_test.csv or eval_val.csv from eval.py",
    )
    p.add_argument("--root", required=True, help="dataset root")
    args = p.parse_args()
    main(args)
