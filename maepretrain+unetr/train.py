
import os
import time
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from vesuvius_data_2 import VesuviusDatasetConfig, VesuviusSegPatchDataset, debug_print_batch_stats_once
from unetr import VideoMAEUNETR2D, load_videomae_encoder_from_mae_ckpt, masked_bce_dice_loss, logits_stats

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_loaders(args):
    train_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.train_ids),
        tile_size=args.tile_size,
        stride=args.stride,
        num_frames=args.num_frames,
        depth_mode=args.depth_mode,
        clip_min=0.0,
        clip_max=200.0,
        pos_ratio=0.5,                # 50/50
        pos_tile_min_frac=args.pos_tile_min_frac,
        valid_tile_min_frac=args.valid_tile_min_frac,
        repeat=args.repeat,
    )
    val_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.val_ids),
        tile_size=args.tile_size,
        stride=args.stride,
        num_frames=args.num_frames,
        depth_mode="center_contig",
        clip_min=0.0,
        clip_max=200.0,
        pos_ratio=0.0,                # not used in val
        pos_tile_min_frac=args.pos_tile_min_frac,
        valid_tile_min_frac=args.valid_tile_min_frac,
        repeat=1,
    )

    train_ds = VesuviusSegPatchDataset(train_cfg, is_train=True)
    val_ds = VesuviusSegPatchDataset(val_cfg, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader

def evaluate(model, loader, device, args):
    model.eval()
    losses = []
    with torch.no_grad():
        for step, (x, y, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss, _, _ = masked_bce_dice_loss(
                logits, y,
                pos_weight=args.pos_weight,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
            )
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if len(losses) else float("nan")

def train_one_epoch(model, loader, optimizer, device, args, epoch, scaler=None):
    model.train()
    t0 = time.time()

    running = []
    skip_steps = 0
    grad_bad_steps = 0

    for step, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        debug_print_batch_stats_once("train", x, y)

        optimizer.zero_grad(set_to_none=True)

        # forward (we already force encoder fp32 inside model)
        logits = model(x)

        # check logits finite first (hard gate)
        if not torch.isfinite(logits).all():
            skip_steps += 1
            continue

        loss, bce, dice = masked_bce_dice_loss(
            logits, y,
            pos_weight=args.pos_weight,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )

        if not torch.isfinite(loss):
            skip_steps += 1
            continue

        loss.backward()

        # grad clip + finite check
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if not torch.isfinite(total_norm):
            grad_bad_steps += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()

        running.append(float(loss.detach().cpu()))

        if (step + 1) % args.log_every == 0:
            stats = logits_stats(logits)
            lr0 = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch:02d} | step {step+1:04d}/{len(loader)} "
                f"| loss={np.mean(running[-args.log_every:]):.4f} "
                f"(bce={float(bce):.4f}, dice={float(dice):.4f}) "
                f"| grad_norm={float(total_norm):.3f} | lr={lr0:.2e} "
                f"| logits(mean={stats['mean']:.3f}, std={stats['std']:.3f}, min={stats['min']:.2f}, max={stats['max']:.2f}) "
                f"| skip={skip_steps} grad_bad={grad_bad_steps}"
            )
            print(msg)

    dt = time.time() - t0
    train_loss = float(np.mean(running)) if len(running) else float("nan")
    print(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f}  skip_steps={skip_steps}  grad_bad_steps={grad_bad_steps}  time={dt:.1f}s")
    return train_loss, skip_steps, grad_bad_steps

def freeze_encoder(model: VideoMAEUNETR2D, freeze: bool = True):
    for p in model.encoder.parameters():
        p.requires_grad = not freeze

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/kaggle/input/vesuvius-challenge-ink-detection")
    ap.add_argument("--train_ids", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--val_ids", nargs="+", default=["1"])

    ap.add_argument("--tile_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--depth_mode", type=str, default="rand_contig")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--pos_tile_min_frac", type=float, default=0.01)
    ap.add_argument("--valid_tile_min_frac", type=float, default=0.5)
    ap.add_argument("--repeat", type=int, default=1)

    ap.add_argument("--pos_weight", type=float, default=10.0)  # 50/50 下建议 5~20
    ap.add_argument("--bce_weight", type=float, default=0.5)
    ap.add_argument("--dice_weight", type=float, default=0.5)

    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mae_ckpt", type=str, default="/kaggle/working/mae_outputs/best_mae.pt")
    ap.add_argument("--out_dir", type=str, default="/kaggle/working/seg_outputs_run")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=1)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print("Device:", device)

    train_loader, val_loader = make_loaders(args)

    model = VideoMAEUNETR2D(tile_size=args.tile_size, num_frames=args.num_frames).to(device)

    if args.mae_ckpt and os.path.exists(args.mae_ckpt):
        print("Loading MAE encoder from:", args.mae_ckpt)
        load_videomae_encoder_from_mae_ckpt(model.encoder, args.mae_ckpt)

    # param groups: encoder lr smaller
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [p for n, p in model.named_parameters() if (not n.startswith("encoder.")) and p.requires_grad]

    # (start frozen optionally)
    if args.freeze_encoder_epochs > 0:
        freeze_encoder(model, True)
        enc_params = []  # frozen => empty
        print(f"[Warmup] Freeze encoder for {args.freeze_encoder_epochs} epoch(s)")

    optimizer = torch.optim.AdamW(
        [
            {"params": dec_params, "lr": args.lr},
            {"params": enc_params, "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")

    # print val batch stats once
    for xb, yb, _ in val_loader:
        debug_print_batch_stats_once("val", xb, yb)
        break

    for epoch in range(1, args.epochs + 1):
        # unfreeze after warmup
        if (epoch == args.freeze_encoder_epochs + 1) and (args.freeze_encoder_epochs > 0):
            freeze_encoder(model, False)
            # rebuild optimizer with encoder params
            enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
            dec_params = [p for n, p in model.named_parameters() if (not n.startswith("encoder.")) and p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": dec_params, "lr": args.lr},
                    {"params": enc_params, "lr": args.encoder_lr},
                ],
                weight_decay=args.weight_decay,
            )
            print("[Warmup] Encoder unfrozen. Optimizer rebuilt.")

        train_loss, skip_steps, grad_bad_steps = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_loss = evaluate(model, val_loader, device, args)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | skip={skip_steps} | grad_bad={grad_bad_steps}")

        # save best
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.out_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
            print("  [Saved] best ->", ckpt_path)

if __name__ == "__main__":
    main()
