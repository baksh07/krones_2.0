"""
Krones — PyTorch Dataset for binary classification
----------------------------------------------------
Loads expanded-ROI crops (real + synth) from disk, applies augmentation
during training only, and returns (image_tensor, binary_label, defect_class_idx).

Designed so the model and training script never need to touch raw paths
or know about real-vs-synth — the dataset hides that complexity.

Key choices documented inline:
  - Resize all crops to a single fixed size (default 512x512). Justified because
    crops vary 700-1056 px and we need a consistent batch shape.
  - Train augmentations are LABEL-PRESERVING ONLY:
      * 360-degree rotation (top-down circular images, fully symmetric)
      * horizontal/vertical flip (same reasoning)
      * brightness/contrast jitter (sensor variation in production)
    We do NOT apply random crop / random erase / cutout, because these can
    silently change the area of conditional defects (Scuffing, Air bubble, ...)
    and flip a borderline-Faulty into a Good. That would corrupt labels.
  - Val and test get NO augmentation, only resize + normalize.
"""

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as T


# Mean/std for ImageNet (we use ImageNet-pretrained ResNet-50, so normalize the same way)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KronesDataset(Dataset):
    """
    Args:
        root:        dataset root (contains audit_output/crops/, audit_output/crops_aug/)
        split:       'train' | 'val' | 'test'
        img_size:    int, resize to (img_size, img_size).  default 512.
        splits_csv:  filename inside audit_output/.  default 'splits_aug.csv'.
                     Use 'splits.csv' if you want to train without synth images.
    """

    # canonical ordering of dominant defect labels — index used by sampler weights
    # and by the per-class breakdown in eval.py.  Unknown classes get bucket -1.
    def __init__(self, root, split, img_size=512, splits_csv="splits_aug.csv"):
        assert split in ("train", "val", "test"), f"bad split: {split}"
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.crops_dir = self.root / "audit_output" / "crops"
        self.crops_aug_dir = self.root / "audit_output" / "crops_aug"

        df = pd.read_csv(self.root / "audit_output" / splits_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        # required columns
        for col in ("file_name", "label", "dominant", "is_synth"):
            if col not in df.columns:
                # splits.csv (without synth) has no is_synth column — treat all as real
                if col == "is_synth":
                    df["is_synth"] = 0
                else:
                    raise ValueError(f"missing required column '{col}' in {splits_csv}")
        self.df = df

        # build dominant class -> integer index map (stable ordering)
        self.classes = sorted(df["dominant"].dropna().unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # image transforms
        if split == "train":
            self.transform = T.Compose(
                [
                    T.Resize((img_size, img_size), antialias=True),
                    # full 360 rotation - top-down circular images are rotationally symmetric
                    T.RandomRotation(degrees=180, expand=False),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    # mild photometric jitter to handle sensor/lighting variation
                    T.ColorJitter(brightness=0.15, contrast=0.15),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            # deterministic transforms for val / test
            self.transform = T.Compose(
                [
                    T.Resize((img_size, img_size), antialias=True),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # synth images live in crops_aug/, real images in crops/
        base_dir = self.crops_aug_dir if int(row["is_synth"]) == 1 else self.crops_dir
        img_path = base_dir / row["file_name"]
        # crops are grayscale - convert to 3-channel so ImageNet-pretrained backbone
        # (which expects RGB) gets a consistent input
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(int(row["label"]), dtype=torch.float32)  # BCE wants float
        cls_idx = self.class_to_idx.get(row["dominant"], -1)
        cls_idx = torch.tensor(cls_idx, dtype=torch.long)
        return img, label, cls_idx

    # -----------------------------------------------------------------
    def make_sampler_weights(self):
        """Return per-sample weights for WeightedRandomSampler.
        Weight = 1 / count_of_dominant_class_for_this_sample.
        Effect: every defect class has roughly equal expected count per epoch.
        """
        counts = self.df["dominant"].value_counts().to_dict()
        weights = self.df["dominant"].map(lambda c: 1.0 / counts[c]).astype("float32")
        return torch.tensor(weights.values, dtype=torch.float32)


# Quick smoke test if run directly
if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for sp in ["train", "val", "test"]:
        ds = KronesDataset(root, sp)
        print(f"{sp:5s}: {len(ds)} samples, {len(ds.classes)} dominant classes")
    # peek one
    ds = KronesDataset(root, "train")
    img, lbl, cidx = ds[0]
    print(
        f"sample 0: image {tuple(img.shape)}, label {lbl.item()}, class {ds.idx_to_class[cidx.item()]}"
    )
    w = ds.make_sampler_weights()
    print(
        f"sampler weights: shape {tuple(w.shape)}, min {w.min():.2e}, max {w.max():.2e}"
    )
