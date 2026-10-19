"""
Krones — Multi-task training (binary + 6-area-regression)
----------------------------------------------------------
ResNet-50 backbone with 7 heads:
  - 1 binary classifier head (Faulty/Good) - the primary task
  - 6 regression heads, one per conditional defect class, each predicting
    log(summed_polygon_area + 1) for that class in the image.

Hypothesis: forcing the backbone to also predict per-class areas gives it
a stronger inductive bias toward representing defect SIZE, not just presence.
At inference, predicted areas can be used to override the binary head when
they fall below the labeling thresholds (rule-based post-processing).

Loss:
  total = w_bin * BCE(binary_logit, label)
        + sum_c (w_class[c] * Huber(log_area_pred_c, log_area_true_c))
                                                      [only for real images;
                                                       synth has has_areas=0
                                                       and is masked out]

Default loss weights (from area-distribution analysis):
  w_bin       = 1.00
  w_scuffing  = 0.50   (16.8% threshold cross rate; high impact)
  w_clight    = 0.30   (56.2% cross; meaningful)
  w_chip      = 0.10   (99.7% cross; nearly always Faulty if present)
  w_glass     = 0.10   (97.0% cross; nearly always Faulty)
  w_sheavy    = 0.10   (65.1% cross)
  w_airb      = 0.05   (0.3% cross; threshold barely matters)

Usage:
    python train_mt.py --root /path/to/dataset --epochs 25 --batch-size 32 \\
        --out-dir runs/resnet50_mt_640
"""

import argparse
import time
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from dataset_mt import KronesMTDataset, CONDITIONAL_NAMES, CONDITIONAL_THRESHOLDS


# ---- Default per-head loss weights -----------------------------------------
DEFAULT_LOSS_WEIGHTS = {
    "binary": 1.00,
    "scuffing": 0.50,
    "contamination_light": 0.30,
    "chip": 0.10,
    "glass_imperfection": 0.10,
    "scuffing_heavy": 0.10,
    "air_bubble": 0.05,
}


# ---- Model -----------------------------------------------------------------
class ResNet50MT(nn.Module):
    """ResNet-50 backbone + 7 heads (binary + 6 area regression).

    The original `model.fc` is replaced by an Identity so we get the 2048-dim
    pooled feature. From there we branch to 7 independent linear heads.

    Forward returns a dict:
        {
          "binary":               (B,)   logits
          "area_scuffing":        (B,)   log-area prediction
          "area_air_bubble":      (B,)
          ...
        }
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head_binary = nn.Linear(self.feat_dim, 1)
        # One regression head per conditional class.
        self.heads_area = nn.ModuleDict(
            {name: nn.Linear(self.feat_dim, 1) for name in CONDITIONAL_NAMES}
        )

    def forward(self, x):
        feats = self.backbone(x)  # (B, 2048)
        out = {"binary": self.head_binary(feats).squeeze(1)}
        for name, head in self.heads_area.items():
            out[f"area_{name}"] = head(feats).squeeze(1)
        return out


def build_model(pretrained=True):
    return ResNet50MT(pretrained=pretrained)


# ---- LR schedule (unchanged from train.py) ---------------------------------
def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---- Multi-task loss -------------------------------------------------------
def multitask_loss(out, label, areas, has_areas, weights, huber_delta=1.0):
    """Compute the weighted multi-task loss.

    Args:
        out:        dict from ResNet50MT.forward
        label:      (B,) float, binary 0/1
        areas:      (B, 6) float, log(area+1) targets aligned to CONDITIONAL_NAMES
        has_areas:  (B,) float, 1.0 if real (use regression), 0.0 if synth (mask)
        weights:    dict of {head_name: weight}, see DEFAULT_LOSS_WEIGHTS
        huber_delta: Huber transition point in log-area units (1.0 ≈ factor of e)

    Returns:
        total_loss: scalar
        log_dict: per-head loss values for tensorboard
    """
    # Binary loss applies to all images
    bin_loss = nn.functional.binary_cross_entropy_with_logits(
        out["binary"], label, reduction="mean"
    )

    log = {"loss/binary": bin_loss.detach()}
    total = weights["binary"] * bin_loss

    # Regression losses, masked by has_areas (synth -> 0 contribution)
    # has_areas mean over batch = fraction of real images. Use it for normalization.
    n_real = has_areas.sum().clamp(min=1.0)

    for i, name in enumerate(CONDITIONAL_NAMES):
        pred = out[f"area_{name}"]  # (B,)
        target = areas[:, i]  # (B,)
        # per-sample Huber, then mask out synth, then mean over real samples
        per_sample = nn.functional.huber_loss(
            pred, target, reduction="none", delta=huber_delta
        )
        masked = (per_sample * has_areas).sum() / n_real
        log[f"loss/area_{name}"] = masked.detach()
        total = total + weights[name] * masked

    log["loss/total"] = total.detach()
    return total, log


# ---- Eval ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    """Return dict with binary metrics + per-head regression MAE on log-areas."""
    model.eval()
    all_binary_logits = []
    all_labels = []
    all_classes = []
    all_areas_pred = []
    all_areas_true = []
    all_has_areas = []

    for imgs, labels, cls_idx, areas, has_areas in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", dtype=autocast_dtype):
            out = model(imgs)
        all_binary_logits.append(out["binary"].float().cpu())
        # stack regression heads -> (B, 6) in CONDITIONAL_NAMES order
        ap = torch.stack(
            [out[f"area_{n}"].float().cpu() for n in CONDITIONAL_NAMES], dim=1
        )
        all_areas_pred.append(ap)
        all_areas_true.append(areas)
        all_has_areas.append(has_areas)
        all_labels.append(labels)
        all_classes.append(cls_idx)

    binary_logits = torch.cat(all_binary_logits)
    labels = torch.cat(all_labels).long()
    classes = torch.cat(all_classes)
    areas_pred = torch.cat(all_areas_pred)
    areas_true = torch.cat(all_areas_true)
    has_areas = torch.cat(all_has_areas)

    probs = torch.sigmoid(binary_logits)
    preds = (probs >= 0.5).long()

    f1 = f1_score(labels.numpy(), preds.numpy(), zero_division=0)
    prec = precision_score(labels.numpy(), preds.numpy(), zero_division=0)
    rec = recall_score(labels.numpy(), preds.numpy(), zero_division=0)
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=[0, 1])

    # per-head log-area MAE on real images only
    real_mask = has_areas > 0.5
    area_mae = {}
    if real_mask.sum() > 0:
        for i, name in enumerate(CONDITIONAL_NAMES):
            mae = (
                (areas_pred[real_mask, i] - areas_true[real_mask, i])
                .abs()
                .mean()
                .item()
            )
            area_mae[name] = mae

    return dict(
        f1=f1,
        precision=prec,
        recall=rec,
        confusion=cm,
        binary_logits=binary_logits,
        labels=labels,
        classes=classes,
        areas_pred=areas_pred,
        areas_true=areas_true,
        has_areas=has_areas,
        area_mae=area_mae,
    )


# ---- Inference rule (override binary with area thresholds) -----------------
def apply_area_override(
    probs, areas_pred_raw, threshold=0.5, upper_confidence=0.85, lower_confidence=0.50
):
    """Rule-based post-processing using predicted per-class areas.

    Logic:
      1. P(Faulty) >= upper_confidence:
           Trust binary head -> Faulty (high-conf, likely a hard-Faulty class)
      2. Any conditional area > threshold:
           -> Faulty (some defect crosses the size rule)
      3. P(Faulty) >= threshold AND no area crosses threshold AND
         P(Faulty) <  upper_confidence:
           Binary said Faulty but no defect class shows enough area.
           This is the "small-Scuffing FP" case - flip to Good.
      4. Otherwise -> Good

    Returns:
        preds:        (N,) int 0/1
        flipped_pos:  (N,) bool, override flipped Good->Faulty (rule 2)
        flipped_neg:  (N,) bool, override flipped Faulty->Good (rule 3)
    """
    n = len(probs)
    preds = np.zeros(n, dtype=int)
    flipped_pos = np.zeros(n, dtype=bool)
    flipped_neg = np.zeros(n, dtype=bool)

    binary_pred = (probs >= threshold).astype(int)
    for i in range(n):
        if probs[i] >= upper_confidence:
            preds[i] = 1
            continue

        any_cross = False
        for j, name in enumerate(CONDITIONAL_NAMES):
            if areas_pred_raw[i, j] > CONDITIONAL_THRESHOLDS[name]:
                any_cross = True
                break

        if any_cross:
            preds[i] = 1
            if binary_pred[i] == 0:
                flipped_pos[i] = True
            continue

        if binary_pred[i] == 1:
            preds[i] = 0
            flipped_neg[i] = True
        else:
            preds[i] = 0

    return preds, flipped_pos, flipped_neg


# ---- Main ------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- data ----
    train_ds = KronesMTDataset(
        args.root,
        "train",
        img_size=args.img_size,
        splits_csv=args.splits_csv,
        area_targets_csv=args.area_targets_csv,
    )
    val_ds = KronesMTDataset(
        args.root,
        "val",
        img_size=args.img_size,
        splits_csv=args.splits_csv,
        area_targets_csv=args.area_targets_csv,
    )
    print(
        f"train: {len(train_ds)}  val: {len(val_ds)}  classes: {len(train_ds.classes)}"
    )

    sampler_weights = train_ds.make_sampler_weights()
    sampler = WeightedRandomSampler(
        weights=sampler_weights, num_samples=len(train_ds), replacement=True
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

    # ---- model ----
    model = build_model(pretrained=True).to(device)
    if args.compile:
        model = torch.compile(model)

    # ---- loss weights ----
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    # Allow CLI overrides like --w-binary 1.5 --w-scuffing 1.0
    for name in weights:
        cli = getattr(args, f"w_{name}", None)
        if cli is not None:
            weights[name] = cli
    print("loss weights:")
    for k, v in weights.items():
        print(f"  {k:25s} = {v:.3f}")

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(100, int(0.05 * total_steps))

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=(args.precision == "fp16"))

    # ---- logging / checkpointing ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir / "tb")
    best_val_f1 = 0.0
    ckpt_path = out_dir / "best.pt"
    last_ckpt_path = out_dir / "last.pt"
    print(f"total steps: {total_steps}  warmup: {warmup_steps}")
    print(f"output dir: {out_dir}")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0

        for batch_idx, (imgs, labels, _cls, areas, has_areas) in enumerate(
            train_loader
        ):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            areas = areas.to(device, non_blocking=True)
            has_areas = has_areas.to(device, non_blocking=True)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=autocast_dtype):
                out = model(imgs)
                loss, log = multitask_loss(
                    out, labels, areas, has_areas, weights, huber_delta=args.huber_delta
                )

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
                for k, v in log.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                print(
                    f"  epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {loss.item():.4f}  lr {lr:.2e}  "
                    f"bin {log['loss/binary'].item():.3f}  "
                    f"scuff {log['loss/area_scuffing'].item():.3f}"
                )

        avg_loss = running_loss / max(1, n_seen)
        epoch_time = time.time() - t0
        print(f"epoch {epoch} done. avg_loss {avg_loss:.4f}  time {epoch_time:.0f}s")

        # ---- val ----
        val_metrics = evaluate(model, val_loader, device, autocast_dtype)
        f1, prec, rec = (
            val_metrics["f1"],
            val_metrics["precision"],
            val_metrics["recall"],
        )
        cm = val_metrics["confusion"]
        print(f"  val: F1 {f1:.4f}  P {prec:.4f}  R {rec:.4f}")
        print(f"  confusion:\n{cm}")
        print("  area MAE (log-units, on real images):")
        for name, mae in val_metrics["area_mae"].items():
            print(f"    {name:25s} {mae:.4f}")

        writer.add_scalar("val/f1", f1, epoch)
        writer.add_scalar("val/precision", prec, epoch)
        writer.add_scalar("val/recall", rec, epoch)
        writer.add_scalar("train/avg_loss", avg_loss, epoch)
        for name, mae in val_metrics["area_mae"].items():
            writer.add_scalar(f"val_area_mae/{name}", mae, epoch)

        # ---- evaluate the area-override rule on val ----
        probs = torch.sigmoid(val_metrics["binary_logits"]).numpy()
        areas_pred_log = val_metrics["areas_pred"].numpy()
        areas_pred_raw = np.expm1(areas_pred_log)
        areas_pred_raw = np.clip(areas_pred_raw, 0, None)
        labels_np = val_metrics["labels"].numpy()
        preds_rule, flipped_pos, flipped_neg = apply_area_override(
            probs, areas_pred_raw, threshold=0.5, upper_confidence=0.85
        )
        f1_rule = f1_score(labels_np, preds_rule, zero_division=0)
        n_pos = int(flipped_pos.sum())
        n_neg = int(flipped_neg.sum())
        print(
            f"  val with area-override rule: F1 {f1_rule:.4f}  "
            f"(+{n_pos} flipped Good->Faulty, -{n_neg} flipped Faulty->Good)"
        )
        writer.add_scalar("val/f1_with_rule", f1_rule, epoch)
        writer.add_scalar("val/n_overrides_pos", n_pos, epoch)
        writer.add_scalar("val/n_overrides_neg", n_neg, epoch)

        # ---- per-defect-class breakdown (using rule-based preds) ----
        v_classes = val_metrics["classes"].numpy()
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
            acc = (labels_np[mask] == preds_rule[mask]).astype(float).mean()
            tag = f"val_per_class/{cname.replace(' ', '_').replace('/', '_')}"
            writer.add_scalar(tag, acc, epoch)
            if cname == "Scuffing":
                print(
                    f"  val Scuffing acc (with rule): {acc:.4f}  ({mask.sum()} samples)"
                )

        # ---- save checkpoints ----
        save_state = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": save_state,
                "optimizer": optimizer.state_dict(),
                "val_f1": f1,
                "val_f1_with_rule": f1_rule,
                "args": vars(args),
                "classes": train_ds.classes,
                "conditional_names": CONDITIONAL_NAMES,
                "loss_weights": weights,
            },
            last_ckpt_path,
        )

        # Track BEST by either binary F1 or rule F1, whichever is higher.
        # (More robust: pick whichever metric we'll actually use at submission.)
        primary = max(f1, f1_rule)
        if primary > best_val_f1:
            best_val_f1 = primary
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": save_state,
                    "val_f1": f1,
                    "val_f1_with_rule": f1_rule,
                    "args": vars(args),
                    "classes": train_ds.classes,
                    "conditional_names": CONDITIONAL_NAMES,
                    "loss_weights": weights,
                },
                ckpt_path,
            )
            print(
                f"  saved new best -> max(F1) = {primary:.4f} "
                f"(binary {f1:.4f}, rule {f1_rule:.4f})"
            )

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"reached --max-steps {args.max_steps}, stopping early.")
            break

    writer.close()
    print(f"\ntraining done. best val F1: {best_val_f1:.4f}")
    print(f"checkpoints: {ckpt_path}  /  {last_ckpt_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", default="runs/resnet50_mt_640")
    p.add_argument("--splits-csv", default="splits_aug.csv")
    p.add_argument("--area-targets-csv", default="area_targets.csv")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Huber loss delta in log-area units",
    )
    # Per-head loss weight overrides
    for name in [
        "binary",
        "scuffing",
        "air_bubble",
        "contamination_light",
        "chip",
        "scuffing_heavy",
        "glass_imperfection",
    ]:
        p.add_argument(
            f"--w-{name.replace('_', '-')}", dest=f"w_{name}", type=float, default=None
        )
    args = p.parse_args()
    main(args)
