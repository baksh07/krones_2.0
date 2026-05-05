"""
Krones — Multi-task Dataset (binary + 6 area regression)
---------------------------------------------------------
Extends the original KronesDataset to also return per-image SUMMED polygon
areas for the 6 conditional defect classes, joined from area_targets.csv.

Returned tuple from __getitem__:
    img       : Tensor (3, H, W)  -- normalized RGB
    label     : Tensor scalar     -- binary 0/1 (float for BCE)
    cls_idx   : Tensor scalar     -- dominant defect class index (long)
    areas     : Tensor (6,)       -- log(area + 1) for the 6 conditional classes,
                                      in the order [scuffing, air_bubble,
                                      contamination_light, chip, scuffing_heavy,
                                      glass_imperfection].
                                      For synth images areas are zeros (masked
                                      out in loss).
    has_areas : Tensor scalar     -- 1.0 if real image (use regression loss),
                                      0.0 if synth (skip regression loss)

The order of conditional classes matches the order in build_area_targets.py
and CONDITIONAL_NAMES below.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Order matters: the regression head outputs are aligned to this order.
CONDITIONAL_NAMES = [
    "scuffing",
    "air_bubble",
    "contamination_light",
    "chip",
    "scuffing_heavy",
    "glass_imperfection",
]

CONDITIONAL_THRESHOLDS = {
    "scuffing":            75000.0,
    "air_bubble":          500.0,
    "contamination_light": 180.0,
    "chip":                200.0,
    "scuffing_heavy":      1200.0,
    "glass_imperfection":  100.0,
}


class KronesMTDataset(Dataset):
    """Multi-task dataset for binary + 6-area-regression training.

    Args:
        root:             dataset root directory.
        split:            'train' | 'val' | 'test'.
        img_size:         resize to (img_size, img_size).
        splits_csv:       splits filename inside audit_output/.
        area_targets_csv: area-targets filename inside audit_output/.
                          Default: 'area_targets.csv'.
    """

    def __init__(self, root, split, img_size=640,
                 splits_csv="splits_aug.csv",
                 area_targets_csv="area_targets.csv"):
        assert split in ("train", "val", "test"), f"bad split: {split}"
        self.root      = Path(root)
        self.split     = split
        self.img_size  = img_size
        self.crops_dir     = self.root / "audit_output" / "crops"
        self.crops_aug_dir = self.root / "audit_output" / "crops_aug"

        # ---- splits ----
        df = pd.read_csv(self.root / "audit_output" / splits_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        for col in ("file_name", "label", "dominant", "is_synth"):
            if col not in df.columns:
                if col == "is_synth":
                    df["is_synth"] = 0
                else:
                    raise ValueError(f"missing column '{col}' in {splits_csv}")

        # ---- area targets ----
        # area_targets.csv has columns: file_name, image_id, label,
        #   area_scuffing, area_air_bubble, area_contamination_light,
        #   area_chip, area_scuffing_heavy, area_glass_imperfection
        # Synth images do NOT appear in area_targets.csv (they have no
        # train_annotations.json polygons). Real images all do.
        area_df = pd.read_csv(self.root / "audit_output" / area_targets_csv)
        area_cols = [f"area_{c}" for c in CONDITIONAL_NAMES]
        keep_cols = ["file_name"] + area_cols
        for c in keep_cols:
            if c not in area_df.columns:
                raise ValueError(
                    f"area_targets.csv missing column '{c}'. "
                    f"Got: {list(area_df.columns)}")
        area_df = area_df[keep_cols]

        # Left-join. Synth rows get NaN for area columns -> we fill with 0.
        merged = df.merge(area_df, on="file_name", how="left")
        n_with_areas = merged[area_cols[0]].notna().sum()
        n_synth = int(merged["is_synth"].sum())
        print(f"  [KronesMTDataset/{split}] joined area targets: "
              f"{n_with_areas}/{len(merged)} have areas (expected ~real-count). "
              f"is_synth={n_synth}")
        # Sanity: every real should have an area row, every synth should NOT
        real_missing = ((merged["is_synth"] == 0) &
                        (merged[area_cols[0]].isna())).sum()
        if real_missing > 0:
            print(f"  WARNING: {real_missing} real images missing area targets")

        # Fill missing areas with 0 (will be masked out by has_areas anyway)
        for c in area_cols:
            merged[c] = merged[c].fillna(0.0)

        self.df = merged
        self.area_cols = area_cols

        # ---- class index map (carried over from KronesDataset) ----
        self.classes = sorted(df["dominant"].dropna().unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # ---- transforms (identical to KronesDataset, label-preserving only) ----
        if split == "train":
            self.transform = T.Compose([
                T.Resize((img_size, img_size), antialias=True),
                T.RandomRotation(degrees=180, expand=False),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        is_synth = int(row["is_synth"])
        base_dir = self.crops_aug_dir if is_synth == 1 else self.crops_dir
        img_path = base_dir / row["file_name"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(int(row["label"]), dtype=torch.float32)
        cls_idx = self.class_to_idx.get(row["dominant"], -1)
        cls_idx = torch.tensor(cls_idx, dtype=torch.long)

        # Area targets, log-transformed.
        # log1p(0) = 0, log1p(75000) ≈ 11.225, log1p(400000) ≈ 12.9
        # so output range is [0, ~13] - well-behaved for regression.
        raw_areas = np.array([row[c] for c in self.area_cols], dtype=np.float32)
        log_areas = np.log1p(raw_areas)
        areas = torch.from_numpy(log_areas)

        # has_areas = 1.0 for real, 0.0 for synth (synth is masked from regression loss)
        has_areas = torch.tensor(0.0 if is_synth == 1 else 1.0,
                                 dtype=torch.float32)

        return img, label, cls_idx, areas, has_areas

    def make_sampler_weights(self):
        """Same as the binary KronesDataset: weight by 1/n_dominant_class."""
        counts = self.df["dominant"].value_counts().to_dict()
        weights = self.df["dominant"].map(lambda c: 1.0 / counts[c]).astype("float32")
        return torch.tensor(weights.values, dtype=torch.float32)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for sp in ["train", "val", "test"]:
        ds = KronesMTDataset(root, sp, img_size=640)
        print(f"{sp:5s}: {len(ds)} samples, {len(ds.classes)} dominant classes")
    # peek
    ds = KronesMTDataset(root, "train", img_size=640)
    img, lbl, cidx, areas, has_areas = ds[0]
    print(f"sample 0: image {tuple(img.shape)}, label {lbl.item()}, "
          f"class {ds.idx_to_class[cidx.item()]}")
    print(f"  log-areas: {areas.numpy()}")
    print(f"  raw areas: {np.expm1(areas.numpy())}")
    print(f"  has_areas: {has_areas.item()}")
    # find a synth one
    for i in range(len(ds)):
        if int(ds.df.iloc[i]['is_synth']) == 1:
            img, lbl, cidx, areas, has_areas = ds[i]
            print(f"\nsynth sample {i}: has_areas={has_areas.item()}, "
                  f"areas={areas.numpy()}")
            break
