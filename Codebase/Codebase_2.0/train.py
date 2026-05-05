"""
Krones — Training script for binary Faulty/Good classification
---------------------------------------------------------------
ResNet-50 (ImageNet-pretrained) fine-tuned on expanded-ROI crops.

Pipeline:
  - Dataloader uses WeightedRandomSampler (1/n_dominant_class) for class balance
  - Loss: BCEWithLogitsLoss (optional pos_weight to rebalance FP vs FN penalty)
  - Optimizer: AdamW with cosine schedule + linear warmup
  - Optional EMA of weights (smooths noisy val curve, often gives ensemble effect)
  - Mixed precision (BF16) on by default
  - Best checkpoint kept by val F1 (on EMA weights if --ema enabled)
  - TensorBoard logs: train/val loss, val F1/precision/recall, LR, per-class val acc

Usage:
    # baseline (your previous successful run)
    python train.py --root /path/to/dataset --epochs 25 --batch-size 32

    # experiment 1: pos_weight + EMA stacked
    python train.py --root /path/to/dataset --epochs 25 --batch-size 32 \\
            --pos-weight 0.7 --ema --out-dir runs/resnet50_640_exp1

For a quick smoke test:
    python train.py --root ... --epochs 1 --max-steps 50
"""
import os, argparse, time, math, copy
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

from dataset import KronesDataset


# ---------------------------------------------------------------------------
def build_model(num_classes=1, pretrained=True):
    """ResNet-50 with binary classification head (1 logit output)."""
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    """Linear warmup -> cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
@torch.no_grad()
def update_ema(ema_model, model, decay):
    """In-place EMA update: ema = decay * ema + (1 - decay) * model."""
    msd = (model._orig_mod.state_dict()
           if hasattr(model, "_orig_mod") else model.state_dict())
    for k, v in ema_model.state_dict().items():
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
        else:
            # buffers like batchnorm running counts: just copy
            v.copy_(msd[k])


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
    rec  = recall_score(labels.numpy(), preds.numpy(), zero_division=0)
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=[0, 1])
    return dict(f1=f1, precision=prec, recall=rec,
                confusion=cm, logits=logits, labels=labels, classes=classes)


# ---------------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # -----------------------------------------------------------------
    # data
    # -----------------------------------------------------------------
    train_ds = KronesDataset(args.root, "train", img_size=args.img_size,
                             splits_csv=args.splits_csv)
    val_ds   = KronesDataset(args.root, "val",   img_size=args.img_size,
                             splits_csv=args.splits_csv)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  classes: {len(train_ds.classes)}")

    sampler_weights = train_ds.make_sampler_weights()
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # -----------------------------------------------------------------
    # model, loss, optimizer
    # -----------------------------------------------------------------
    model = build_model(num_classes=1, pretrained=True).to(device)
    if args.compile:
        model = torch.compile(model)

    # BCE with optional pos_weight to rebalance FP vs FN penalty.
    # pos_weight < 1 -> false positives get penalized more (model becomes more
    # conservative about predicting Faulty -- targets the Scuffing-FP bias we
    # diagnosed). pos_weight = 1 (default) is standard unweighted BCE.
    if args.pos_weight is not None and abs(args.pos_weight - 1.0) > 1e-8:
        pw = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"      loss: BCE with pos_weight={args.pos_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"      loss: BCE (unweighted)")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    # EMA shadow model. Updated every step. At eval time, val is run on the EMA
    # copy if --ema is enabled, and the EMA weights are saved as best.pt.
    # Effect: val curve is smoother, often equivalent to a small ensemble.
    ema_model = None
    if args.ema:
        ema_model = copy.deepcopy(model._orig_mod
                                  if hasattr(model, "_orig_mod") else model)
        ema_model = ema_model.to(device).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"      EMA: enabled, decay={args.ema_decay}")

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(100, int(0.05 * total_steps))   # 5% of total, min 100 steps

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=(args.precision == "fp16"))

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

            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

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

            # EMA update happens AFTER the optimizer step
            if ema_model is not None:
                update_ema(ema_model, model, args.ema_decay)

            running_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            global_step += 1

            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr",   lr,           global_step)
                print(f"  epoch {epoch} step {global_step}/{total_steps} "
                      f"loss {loss.item():.4f}  lr {lr:.2e}")

        avg_loss = running_loss / max(1, n_seen)
        epoch_time = time.time() - t0
        print(f"epoch {epoch} done. avg_loss {avg_loss:.4f}  time {epoch_time:.0f}s")

        # ---------------- val ----------------
        # If EMA enabled, validate on the EMA model (this is what gets saved
        # as best.pt). Also run val on the live model for comparison logging.
        eval_model = ema_model if ema_model is not None else model
        val_metrics = evaluate(eval_model, val_loader, device, autocast_dtype)
        f1, prec, rec = val_metrics["f1"], val_metrics["precision"], val_metrics["recall"]
        cm = val_metrics["confusion"]
        which = "EMA" if ema_model is not None else "live"
        print(f"  val ({which}): F1 {f1:.4f}  P {prec:.4f}  R {rec:.4f}")
        print(f"  confusion (rows=true, cols=pred):\n{cm}")
        writer.add_scalar("val/f1",        f1,   epoch)
        writer.add_scalar("val/precision", prec, epoch)
        writer.add_scalar("val/recall",    rec,  epoch)
        writer.add_scalar("train/avg_loss", avg_loss, epoch)

        # If EMA, also log the live-model F1 so we can compare in TB
        if ema_model is not None:
            live_metrics = evaluate(model, val_loader, device, autocast_dtype)
            writer.add_scalar("val_live/f1",        live_metrics["f1"],        epoch)
            writer.add_scalar("val_live/precision", live_metrics["precision"], epoch)
            writer.add_scalar("val_live/recall",    live_metrics["recall"],    epoch)
            print(f"  val (live for comparison): F1 {live_metrics['f1']:.4f}")

        # per-class accuracy on val. Track classes that have been the primary
        # failure modes so we can see in real-time which lever moves them.
        v_labels  = val_metrics["labels"]
        v_classes = val_metrics["classes"]
        v_probs   = torch.sigmoid(val_metrics["logits"])
        v_preds   = (v_probs >= 0.5).long()
        watch_classes = ["Scuffing", "Air bubble", "Contamination light",
                         "Glass imperfection", "Scuffing heavy", "Good_no_defect",
                         "Contamination dark"]
        for cname in watch_classes:
            if cname not in train_ds.class_to_idx:
                continue
            cidx = train_ds.class_to_idx[cname]
            mask = (v_classes == cidx)
            if mask.sum() == 0: continue
            acc = (v_labels[mask] == v_preds[mask]).float().mean().item()
            tag = f"val_per_class/{cname.replace(' ', '_').replace('/', '_')}"
            writer.add_scalar(tag, acc, epoch)
            if cname == "Scuffing":
                print(f"  val Scuffing acc: {acc:.4f}  ({mask.sum().item()} samples)")

        # -------------- save checkpoints --------------
        # When EMA is active, the saved weights ARE the EMA weights, because
        # that's the model we want at inference/submission time.
        save_state = (ema_model.state_dict() if ema_model is not None
                      else (model._orig_mod.state_dict()
                            if hasattr(model, "_orig_mod") else model.state_dict()))

        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model": save_state,
            "optimizer": optimizer.state_dict(),
            "val_f1": f1, "args": vars(args),
            "classes": train_ds.classes,
            "ema": ema_model is not None,
        }, last_ckpt_path)

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "model": save_state,
                "val_f1": f1, "args": vars(args),
                "classes": train_ds.classes,
                "ema": ema_model is not None,
            }, ckpt_path)
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
    p.add_argument("--root",        required=True, help="dataset root")
    p.add_argument("--out-dir",     default="runs/baseline_resnet50")
    p.add_argument("--splits-csv",  default="splits_aug.csv",
                   help="splits.csv (no synth) or splits_aug.csv (with synth)")
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size",    type=int, default=512)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--precision",   choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--log-every",   type=int, default=20)
    p.add_argument("--max-steps",   type=int, default=None,
                   help="stop after this many steps (for smoke testing)")
    p.add_argument("--compile",     action="store_true",
                   help="torch.compile the model (PyTorch 2.0+)")
    # ---- new in this version ----
    p.add_argument("--pos-weight",  type=float, default=None,
                   help="BCE pos_weight. <1 penalizes FP more, >1 penalizes FN more. "
                        "Default unset = standard unweighted BCE.")
    p.add_argument("--ema",         action="store_true",
                   help="enable EMA of model weights for evaluation/saving")
    p.add_argument("--ema-decay",   type=float, default=0.999,
                   help="EMA decay rate. Higher = slower averaging. 0.999 is standard.")
    args = p.parse_args()
    main(args)