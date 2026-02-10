

import os
import csv
import math
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import VideoMAEConfig, VideoMAEForPreTraining

# You should already have this file in /kaggle/working
from vesuvius_data_1 import VesuviusDatasetConfig, VesuviusMAEPatchDataset


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for r in rows:
            w.writerow(r)


def build_lr_scheduler(optimizer, total_steps, warmup_steps, min_lr=1e-6):
    """
    Linear warmup + Cosine decay to min_lr.
    Returns a function step() that updates lr each optimizer step.
    """
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_at(step):
        if step < warmup_steps:
            scale = (step + 1) / max(1, warmup_steps)
        else:
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))
        return scale

    def step_fn(step):
        scale = lr_at(step)
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = min_lr + (base_lrs[i] - min_lr) * scale

    return step_fn


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="/kaggle/input/vesuvius-challenge-ink-detection")
    ap.add_argument("--train_ids", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--valid_ids", nargs="+", default=["1"])

    ap.add_argument("--tile_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)

    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument(
        "--depth_mode", type=str, default="rand_contig",
        choices=["rand_contig", "center_contig", "odd_subsample", "rand_stride2"]
    )

    ap.add_argument("--mask_ratio", type=float, default=0.85)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--out_dir", type=str, default="/kaggle/working/mae_outputs")
    ap.add_argument("--seed", type=int, default=42)

    # by default True (you can disable by passing --no_fp16 if you want)
    ap.add_argument("--fp16", action="store_true", default=True)
    return ap.parse_args()


@torch.no_grad()
def make_bool_masked_pos(batch_size: int, num_patches: int, mask_ratio: float, device: torch.device):
    """
    Create bool_masked_pos of shape (B, num_patches) where about mask_ratio patches are masked.
    """
    num_mask = int(mask_ratio * num_patches)
    num_mask = max(1, min(num_mask, num_patches - 1))  # safe range

    bool_masked_pos = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(num_patches, device=device)[:num_mask]
        bool_masked_pos[i, idx] = True
    return bool_masked_pos


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # ------------------------
    # Dataset / Loader
    # ------------------------
    train_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.train_ids),
        tile_size=args.tile_size,
        stride=args.stride,
        num_frames=args.num_frames,
        depth_mode=args.depth_mode,
        repeat=args.repeat,
    )
    valid_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.valid_ids),
        tile_size=args.tile_size,
        stride=args.tile_size,
        num_frames=args.num_frames,
        depth_mode="center_contig",
        repeat=1,
    )
    train_cfg.mask_ratio = args.mask_ratio
    valid_cfg.mask_ratio = args.mask_ratio

    train_ds = VesuviusMAEPatchDataset(train_cfg, is_train=True)
    val_ds = VesuviusMAEPatchDataset(valid_cfg, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # ------------------------
    # Model (IMPORTANT: config must match finetune encoder)
    # ------------------------
    tubelet_size = 2 if (args.num_frames % 2 == 0) else 1

    vcfg = VideoMAEConfig(
        image_size=args.tile_size,
        patch_size=16,
        num_channels=1,
        num_frames=args.num_frames,
        tubelet_size=tubelet_size,

        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,

        decoder_num_hidden_layers=4,
        decoder_hidden_size=512,
        decoder_num_attention_heads=8,
        decoder_intermediate_size=2048,

        norm_pix_loss=True,
        mask_ratio=args.mask_ratio,
    )

    model = VideoMAEForPreTraining(vcfg).to(device)
    model.train()

    # ---- compute num_patches for bool_masked_pos ----
    patch = vcfg.patch_size
    tube = vcfg.tubelet_size
    Hp = args.tile_size // patch
    Wp = args.tile_size // patch
    Tp = args.num_frames // tube
    num_patches = Hp * Wp * Tp
    print(f"[info] patch_size={patch} tubelet_size={tube} Hp={Hp} Wp={Wp} Tp={Tp} num_patches={num_patches}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    lr_step = build_lr_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=args.min_lr
    )

    # new AMP API (still works on kaggle)
    use_amp = bool(args.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_val = float("inf")
    global_step = 0

    history_path = os.path.join(args.out_dir, f"training_history_videomae_{args.tile_size}_{args.num_frames}.csv")
    best_ckpt_path = os.path.join(args.out_dir, "best_mae.pt")
    last_ckpt_path = os.path.join(args.out_dir, "last_mae.pt")

    for epoch in range(1, args.epochs + 1):
        # ------------------------
        # Train
        # ------------------------
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"[Train] epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)  # (B,T,1,H,W)

            lr_step(global_step)
            optimizer.zero_grad(set_to_none=True)

            # create bool_masked_pos for this batch
            B = batch.size(0)
            bool_masked_pos = make_bool_masked_pos(B, num_patches, args.mask_ratio, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(pixel_values=batch, bool_masked_pos=bool_masked_pos)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            global_step += 1

            pbar.set_postfix(loss=float(np.mean(train_losses)), lr=optimizer.param_groups[0]["lr"])

        train_loss = float(np.mean(train_losses))

        # ------------------------
        # Val
        # ------------------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"[Val] epoch {epoch}/{args.epochs}", leave=False)
            for batch in pbar:
                batch = batch.to(device, non_blocking=True)
                B = batch.size(0)
                bool_masked_pos = make_bool_masked_pos(B, num_patches, args.mask_ratio, device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(pixel_values=batch, bool_masked_pos=bool_masked_pos)
                    loss = out.loss

                val_losses.append(loss.item())
                pbar.set_postfix(val_loss=float(np.mean(val_losses)))

        val_loss = float(np.mean(val_losses))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr_now:.2e}")

        history.append([epoch, train_loss, val_loss, lr_now])
        save_history_csv(history_path, history)

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": vcfg.to_dict(),
                "args": vars(args),
            },
            last_ckpt_path
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": vcfg.to_dict(),
                    "args": vars(args),
                },
                best_ckpt_path
            )
            print("  -> saved BEST to", best_ckpt_path)

    print("Done.")
    print("Best ckpt:", best_ckpt_path)
    print("History csv:", history_path)


if __name__ == "__main__":
    main()
