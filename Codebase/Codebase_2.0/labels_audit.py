"""
Krones Vision AI Challenge — Label Audit
----------------------------------------
Recomputes the binary (Faulty / Good) label for every training image
directly from train_annotations.json using the area thresholds from the
challenge brief, then compares against train.csv.

Outputs:
  - train_clean.csv     : per-image clean label + diagnostics
  - audit_report.txt    : summary statistics
  - disagreements.csv   : images where csv label != recomputed label

Run:
    python label_audit.py --root /path/to/1st-krones-vision-ai-challenge
"""

import os
import json
import argparse
from collections import defaultdict, Counter
import pandas as pd

# -------------------------------------------------------------------
# Class taxonomy (from challenge brief + EDA)
# -------------------------------------------------------------------

# Hard Faulty: presence of any annotation -> Faulty
HARD_FAULTY = {
    "Contamination dark",
    "Yeast residue",
    "No base visible",
    "Crown cap",
    "Mold",
    "Foreign object - manual cleaning",
    "Glass shard",
    "Foreign object - washing machine",
    "Break / Crack",
    "Label",
    "Liquid",
    "Paint residue",
    "Foil / Semitransparent",
    "Circlip",
    "Straw",
    "Insect",
}

# Hard Good: presence does NOT make image Faulty (still Good)
HARD_GOOD = {
    "Water drop",
    "No fault",
    "Foam residue",
    "Embossing",
}

# Conditional Faulty: Faulty only if total area in image > threshold
CONDITIONAL = {
    "Scuffing": 75000,
    "Air bubble": 500,
    "Contamination light": 180,
    "Chip": 200,
    "Scuffing heavy": 1200,
    "Glass imperfection": 100,
}

# Meta categories that are not defects
META = {"Roi", "No fault"}


# -------------------------------------------------------------------
def main(root: str):
    csv_path = os.path.join(root, "train.csv")
    ann_path = os.path.join(root, "train_annotations.json")
    out_dir = os.path.join(root, "audit_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/6] loading data from {root} ...")
    csv = pd.read_csv(csv_path)
    with open(ann_path, "r") as f:
        ann = json.load(f)

    cats = {c["id"]: c["name"] for c in ann["categories"]}
    name2id = {v: k for k, v in cats.items()}
    images = {im["id"]: im for im in ann["images"]}
    print(
        f"      {len(images)} images, {len(ann['annotations'])} annotations, {len(cats)} categories"
    )

    # sanity check that every named class exists
    all_named = HARD_FAULTY | HARD_GOOD | set(CONDITIONAL.keys()) | META
    missing = all_named - set(cats.values())
    if missing:
        print(f"      WARNING: classes in taxonomy but not in annotations: {missing}")
    extra = set(cats.values()) - all_named
    if extra:
        print(f"      WARNING: classes in annotations not in taxonomy: {extra}")

    # -----------------------------------------------------------------
    # Step 2: aggregate per-image annotation info
    # -----------------------------------------------------------------
    print("[2/6] aggregating annotations per image ...")
    img_to_roi = {}  # image_id -> bbox
    img_class_count = defaultdict(lambda: Counter())  # image_id -> {class_name: n_anns}
    img_class_area = defaultdict(
        lambda: Counter()
    )  # image_id -> {class_name: total_area}
    img_outside_roi = defaultdict(set)  # image_id -> {classes with anns outside ROI}

    ROI_ID = name2id.get("Roi")

    # first pass: ROI bboxes
    for a in ann["annotations"]:
        if a["category_id"] == ROI_ID:
            img_to_roi[a["image_id"]] = a["bbox"]  # [x, y, w, h]

    def inside(b, r):
        return (
            b[0] >= r[0]
            and b[1] >= r[1]
            and b[0] + b[2] <= r[0] + r[2]
            and b[1] + b[3] <= r[1] + r[3]
        )

    # second pass: defect annotations
    for a in ann["annotations"]:
        cname = cats[a["category_id"]]
        if cname in META:
            continue
        iid = a["image_id"]
        img_class_count[iid][cname] += 1
        img_class_area[iid][cname] += float(a.get("area", 0.0))

        roi = img_to_roi.get(iid)
        if roi is not None and not inside(a["bbox"], roi):
            img_outside_roi[iid].add(cname)

    # -----------------------------------------------------------------
    # Step 3: recompute binary label per image
    # -----------------------------------------------------------------
    print("[3/6] recomputing binary labels ...")

    rows = []
    id_to_fname = {im["id"]: im["file_name"] for im in ann["images"]}

    for iid, fname in id_to_fname.items():
        counts = img_class_count.get(iid, Counter())
        areas = img_class_area.get(iid, Counter())

        hard_faulty_hits = [c for c in counts if c in HARD_FAULTY]
        cond_hits = [
            (c, areas[c], CONDITIONAL[c])
            for c in counts
            if c in CONDITIONAL and areas[c] > CONDITIONAL[c]
        ]
        cond_below = [
            (c, areas[c], CONDITIONAL[c])
            for c in counts
            if c in CONDITIONAL and areas[c] <= CONDITIONAL[c]
        ]

        is_faulty = bool(hard_faulty_hits) or bool(cond_hits)
        # reasoning string for debugging
        reason_bits = []
        if hard_faulty_hits:
            reason_bits.append("hard:" + ",".join(sorted(hard_faulty_hits)))
        if cond_hits:
            reason_bits.append(
                "cond_over:" + ",".join(f"{c}({a:.0f}>{t})" for c, a, t in cond_hits)
            )
        if cond_below and not is_faulty:
            reason_bits.append(
                "cond_below:" + ",".join(f"{c}({a:.0f}<={t})" for c, a, t in cond_below)
            )
        if not reason_bits:
            reason_bits.append("no_defect_anns")

        # dominant defect = class with largest total area among defect classes (excl. hard good)
        defect_only = {c: a for c, a in areas.items() if c not in HARD_GOOD}
        dominant = max(defect_only, key=defect_only.get) if defect_only else ""

        rows.append(
            {
                "file_name": fname,
                "image_id": iid,
                "label_recomputed": int(is_faulty),
                "dominant_defect": dominant,
                "n_annotations": int(sum(counts.values())),
                "n_defect_classes": int(len([c for c in counts if c not in HARD_GOOD])),
                "has_outside_roi": int(bool(img_outside_roi.get(iid))),
                "outside_roi_classes": ";".join(
                    sorted(img_outside_roi.get(iid, set()))
                ),
                "reason": " | ".join(reason_bits),
            }
        )

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Step 4: merge with train.csv to find disagreements
    # -----------------------------------------------------------------
    print("[4/6] comparing with train.csv ...")
    # train.csv schema: try to auto-detect label column
    label_col = None
    for cand in ["label", "is_faulty", "faulty", "target", "class", "y"]:
        if cand in csv.columns:
            label_col = cand
            break
    if label_col is None:
        # take last column if not found
        label_col = csv.columns[-1]
        print(f"      could not find label column, using last column: '{label_col}'")
    fname_col = None
    for cand in ["file_name", "filename", "image", "image_name", "id"]:
        if cand in csv.columns:
            fname_col = cand
            break
    if fname_col is None:
        fname_col = csv.columns[0]
        print(
            f"      could not find filename column, using first column: '{fname_col}'"
        )

    csv_small = csv[[fname_col, label_col]].rename(
        columns={fname_col: "file_name", label_col: "label_csv"}
    )
    df = df.merge(csv_small, on="file_name", how="left")

    df["agree"] = (df["label_recomputed"] == df["label_csv"]).astype(int)
    n_total = len(df)
    n_with_csv = df["label_csv"].notna().sum()
    n_disagree = (df["agree"] == 0).sum() if n_with_csv else 0

    # -----------------------------------------------------------------
    # Step 5: write outputs
    # -----------------------------------------------------------------
    print("[5/6] writing outputs ...")
    df.to_csv(os.path.join(out_dir, "train_clean.csv"), index=False)
    df[df["agree"] == 0].to_csv(os.path.join(out_dir, "disagreements.csv"), index=False)

    # -----------------------------------------------------------------
    # Step 6: summary report
    # -----------------------------------------------------------------
    print("[6/6] summary report\n")
    lines = []
    L = lines.append
    L("=" * 70)
    L("KRONES LABEL AUDIT REPORT")
    L("=" * 70)
    L(f"images total                    : {n_total}")
    L(f"images with csv label           : {n_with_csv}")
    L(
        f"recomputed Faulty               : {int(df['label_recomputed'].sum())}  ({df['label_recomputed'].mean() * 100:.2f}%)"
    )
    L(
        f"recomputed Good                 : {int((1 - df['label_recomputed']).sum())}  ({(1 - df['label_recomputed']).mean() * 100:.2f}%)"
    )
    if n_with_csv:
        L(f"csv Faulty                      : {int(df['label_csv'].sum())}")
        L(f"csv Good                        : {int((1 - df['label_csv']).sum())}")
        L(
            f"DISAGREEMENTS                   : {n_disagree}  ({n_disagree / n_with_csv * 100:.2f}%)"
        )
        # confusion
        c = pd.crosstab(
            df["label_csv"],
            df["label_recomputed"],
            rownames=["csv"],
            colnames=["recomputed"],
            dropna=False,
        )
        L("\nconfusion (csv vs recomputed):")
        L(str(c))
    L("")
    L("-" * 70)
    L("out-of-ROI defects")
    L("-" * 70)
    L(f"images with >=1 defect outside ROI: {int(df['has_outside_roi'].sum())}")
    outside_classes = Counter()
    for s in df["outside_roi_classes"]:
        if s:
            for c in s.split(";"):
                outside_classes[c] += 1
    if outside_classes:
        L("breakdown by class:")
        for c, n in outside_classes.most_common():
            L(f"  {c:<40s} {n}")
    L("")
    L("-" * 70)
    L("dominant defect distribution (after recomputation)")
    L("-" * 70)
    dom = (
        df.groupby(["dominant_defect", "label_recomputed"]).size().unstack(fill_value=0)
    )
    dom["total"] = dom.sum(axis=1)
    dom = dom.sort_values("total", ascending=False)
    L(str(dom))
    L("")
    L("-" * 70)
    L("conditional classes — borderline (annotations present but below threshold)")
    L("-" * 70)
    cond_below_counts = Counter()
    cond_above_counts = Counter()
    for _, r in df.iterrows():
        if "cond_below:" in r["reason"]:
            for tok in r["reason"].split("cond_below:")[1].split(" | ")[0].split(","):
                cls = tok.split("(")[0]
                cond_below_counts[cls] += 1
        if "cond_over:" in r["reason"]:
            for tok in r["reason"].split("cond_over:")[1].split(" | ")[0].split(","):
                cls = tok.split("(")[0]
                cond_above_counts[cls] += 1
    L(f"{'class':<25s} {'above_thr':>10s} {'below_thr':>10s}")
    for c in CONDITIONAL:
        L(
            f"{c:<25s} {cond_above_counts.get(c, 0):>10d} {cond_below_counts.get(c, 0):>10d}"
        )
    L("")
    L("=" * 70)
    L(f"outputs written to: {out_dir}")
    L("  train_clean.csv     - per-image labels + diagnostics")
    L("  disagreements.csv   - csv vs recomputed mismatches")
    L("  audit_report.txt    - this report")
    L("=" * 70)

    report = "\n".join(lines)
    with open(os.path.join(out_dir, "audit_report.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        help="folder containing train.csv and train_annotations.json",
    )
    args = p.parse_args()
    main(args.root)
