"""
Krones — Training script for binary Faulty/Good classification
---------------------------------------------------------------
ResNet-50 (ImageNet-pretrained) fine-tuned on expanded-ROI crops.

Pipeline:
  - Dataloader uses WeightedRandomSampler (1/n_dominant_class) for class balance
  - Loss: BCEWithLogitsLoss
  - Optimizer: AdamW with cosine schedule + linear warmup
  - Mixed precision (BF16) on by default
  - Best checkpoint kept by val F1
  - TensorBoard logs: train/val loss, val F1/precision/recall, LR, confusion matrix

Usage:
    python train.py --root /path/to/dataset --epochs 20 --batch-size 32

For a quick smoke test:
    python train.py --root ... --epochs 1 --max-steps 50
"""

import argparse
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import timm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from dataset import KronesDataset


# ---------------------------------------------------------------------------
def build_model(
    num_classes=1,
    pretrained=True,
    model_name="swin_tiny_patch4_window7_224",
    img_size=224,
):
    """Build a backbone with a binary classification head.
    Default: Swin-Tiny (timm) pretrained on ImageNet-1k.

    img_size MUST match the actual input the model will receive at runtime.
    Swin's patch embedding has a hardcoded size assert; passing the wrong
    img_size produces 'Input height (X) doesn't match model (Y)' at the first
    forward pass.

    Special case 'resnet50' uses torchvision and ignores img_size (resnets are
    fully convolutional and accept any input).
    """
    if model_name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    # Swin / ConvNeXt / etc via timm. Pass img_size explicitly.
    # timm interpolates pretrained position embeddings to the new size if needed.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes, img_size=img_size
    )
    return model


def split_param_groups(model, head_lr, backbone_lr, weight_decay):
    """Return optimizer param groups: head gets higher LR, backbone lower.
    'Head' = the final classifier (fc/head/classifier - timm names it 'head').
    """
    # timm uses model.head for most architectures (Swin/ConvNeXt/ViT/EfficientNet)
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            name.startswith("head")
            or name.startswith("classifier")
            or name.startswith("fc")
        ):
            head_params.append(p)
        else:
            backbone_params.append(p)
    print(
        f"      param groups: head={sum(p.numel() for p in head_params):,}  "
        f"backbone={sum(p.numel() for p in backbone_params):,}"
    )
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    """Linear warmup -> cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    """Return dict with binary metrics on the given loader."""
    model.eval()
    all_logits = []
    all_labels = []
    all_classes = []
    for imgs, labels, cls_idx in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", dtype=autocast_dtype):
            logits = model(imgs).squeeze(1)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
        all_classes.append(cls_idx)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).long()
    classes = torch.cat(all_classes)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    f1 = f1_score(labels.numpy(), preds.numpy(), zero_division=0)
    prec = precision_score(labels.numpy(), preds.numpy(), zero_division=0)
    rec = recall_score(labels.numpy(), preds.numpy(), zero_division=0)
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=[0, 1])
    return dict(
        f1=f1,
        precision=prec,
        recall=rec,
        confusion=cm,
        logits=logits,
        labels=labels,
        classes=classes,
    )


# ---------------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # -----------------------------------------------------------------
    # data
    # -----------------------------------------------------------------
    train_ds = KronesDataset(
        args.root, "train", img_size=args.img_size, splits_csv=args.splits_csv
    )
    val_ds = KronesDataset(
        args.root, "val", img_size=args.img_size, splits_csv=args.splits_csv
    )
    print(
        f"train: {len(train_ds)}  val: {len(val_ds)}  classes: {len(train_ds.classes)}"
    )

    # WeightedRandomSampler balances by dominant defect class
    sampler_weights = train_ds.make_sampler_weights()
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(train_ds),  # one "epoch" = one full dataset pass worth of draws
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # -----------------------------------------------------------------
    # model, loss, optimizer
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # model, loss, optimizer
    # -----------------------------------------------------------------
    model = build_model(
        num_classes=1,
        pretrained=True,
        model_name=args.model_name,
        img_size=args.img_size,
    ).to(device)
    if args.compile:
        model = torch.compile(model)
    criterion = nn.BCEWithLogitsLoss()

    # Differential LR: head learns faster (it's fresh), backbone learns slower
    # (its features are good already, we don't want to wreck them).
    # head_lr / backbone_lr factor of 3 is a standard fine-tuning recipe.
    head_lr = args.lr
    backbone_lr = args.lr * args.backbone_lr_mult
    print(f"      LR: head={head_lr:.1e}  backbone={backbone_lr:.1e}")
    param_groups = split_param_groups(model, head_lr, backbone_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(100, int(0.05 * total_steps))  # 5% of total, min 100 steps

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=(args.precision == "fp16"))  # bf16 doesn't need scaler

    # -----------------------------------------------------------------
    # logging + checkpointing
    # -----------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir / "tb")
    best_val_f1 = 0.0
    ckpt_path = out_dir / "best.pt"
    last_ckpt_path = out_dir / "last.pt"

    print(f"total steps: {total_steps}  warmup: {warmup_steps}")
    print(f"output dir: {out_dir}")

    # -----------------------------------------------------------------
    # training loop
    # -----------------------------------------------------------------
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for batch_idx, (imgs, labels, _) in enumerate(train_loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # update LR with manual cosine warmup-and-decay.
            # Each param group gets scaled by the same fraction of its base LR,
            # so the head:backbone ratio (e.g. 3:1) is preserved across training.
            sched_factor = cosine_warmup_lr(global_step, total_steps, warmup_steps, 1.0)
            backbone_group_lr = args.lr * args.backbone_lr_mult * sched_factor
            head_group_lr = args.lr * sched_factor
            optimizer.param_groups[0]["lr"] = (
                backbone_group_lr  # backbone group (added first)
            )
            optimizer.param_groups[1]["lr"] = head_group_lr  # head group (added second)
            lr = head_group_lr  # what we log to TB

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=autocast_dtype):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

            if args.precision == "fp16":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            global_step += 1

            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                print(
                    f"  epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {loss.item():.4f}  lr {lr:.2e}"
                )

        avg_loss = running_loss / max(1, n_seen)
        epoch_time = time.time() - t0
        print(f"epoch {epoch} done. avg_loss {avg_loss:.4f}  time {epoch_time:.0f}s")

        # ---------------- val ----------------
        val_metrics = evaluate(model, val_loader, device, autocast_dtype)
        f1, prec, rec = (
            val_metrics["f1"],
            val_metrics["precision"],
            val_metrics["recall"],
        )
        cm = val_metrics["confusion"]
        print(f"  val: F1 {f1:.4f}  P {prec:.4f}  R {rec:.4f}")
        print(f"  confusion (rows=true, cols=pred):\n{cm}")
        writer.add_scalar("val/f1", f1, epoch)
        writer.add_scalar("val/precision", prec, epoch)
        writer.add_scalar("val/recall", rec, epoch)
        writer.add_scalar("train/avg_loss", avg_loss, epoch)

        # per-class accuracy on val - especially track Scuffing since it's the
        # diagnostic we care about. Logging "Scuffing acc" alongside global F1
        # tells us in real time whether the new model is fixing the bottleneck.
        v_labels = val_metrics["labels"]
        v_classes = val_metrics["classes"]
        v_probs = torch.sigmoid(val_metrics["logits"])
        v_preds = (v_probs >= 0.5).long()
        watch_classes = [
            "Scuffing",
            "Air bubble",
            "Contamination light",
            "Glass imperfection",
            "Scuffing heavy",
            "Good_no_defect",
            "Contamination dark",
        ]
        for cname in watch_classes:
            if cname not in train_ds.class_to_idx:
                continue
            cidx = train_ds.class_to_idx[cname]
            mask = v_classes == cidx
            if mask.sum() == 0:
                continue
            acc = (v_labels[mask] == v_preds[mask]).float().mean().item()
            writer.add_scalar(
                f"val_per_class/{cname.replace(' ', '_').replace('/', '_')}", acc, epoch
            )
            if cname == "Scuffing":
                # also print to console
                print(f"  val Scuffing acc: {acc:.4f}  ({mask.sum().item()} samples)")

        # save last
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": (
                    model._orig_mod.state_dict()
                    if hasattr(model, "_orig_mod")
                    else model.state_dict()
                ),
                "optimizer": optimizer.state_dict(),
                "val_f1": f1,
                "args": vars(args),
                "classes": train_ds.classes,
            },
            last_ckpt_path,
        )
        # save best
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": (
                        model._orig_mod.state_dict()
                        if hasattr(model, "_orig_mod")
                        else model.state_dict()
                    ),
                    "val_f1": f1,
                    "args": vars(args),
                    "classes": train_ds.classes,
                },
                ckpt_path,
            )
            print(f"  saved new best -> val F1 {f1:.4f}")

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"reached --max-steps {args.max_steps}, stopping early.")
            break

    writer.close()
    print(f"\ntraining done. best val F1: {best_val_f1:.4f}")
    print(f"checkpoints: {ckpt_path}  /  {last_ckpt_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset root")
    p.add_argument("--out-dir", default="runs/baseline_resnet50")
    p.add_argument(
        "--splits-csv",
        default="splits_aug.csv",
        help="splits.csv (no synth) or splits_aug.csv (with synth)",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many steps (for smoke testing)",
    )
    p.add_argument(
        "--compile", action="store_true", help="torch.compile the model (PyTorch 2.0+)"
    )
    p.add_argument(
        "--model-name",
        default="swin_tiny_patch4_window7_224",
        help="timm model name. Examples: swin_tiny_patch4_window7_224, "
        "convnext_tiny.fb_in22k, efficientnetv2_s",
    )
    p.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.33,
        help="backbone LR = lr * this factor. Default 0.33 = 3:1 head:backbone",
    )
    args = p.parse_args()
    main(args)
