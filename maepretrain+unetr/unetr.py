
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEConfig, VideoMAEModel

IGNORE_INDEX = 127

def _strip_prefix(k: str) -> str:
    for p in ["model.", "net.", "encoder.", "videomae.", "module.", "model_state."]:
        if k.startswith(p):
            return k[len(p):]
    return k

def _pick_state_dict(ckpt: dict):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]
    return ckpt

@torch.no_grad()
def load_videomae_encoder_from_mae_ckpt(encoder: VideoMAEModel, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _pick_state_dict(ckpt)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {type(state)}")

    new_state = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        kk = _strip_prefix(k)

        # drop obvious decoder heads / bridges
        if any(s in kk for s in ["decoder", "mask_token", "decoder_pos_embed", "encoder_to_decoder"]):
            continue

        # normalize possible nesting
        for p in ["videomae.videomae.", "model.videomae.", "videomae."]:
            if kk.startswith(p):
                kk = kk[len(p):]
                break

        new_state[kk] = v

    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    print(f"[load_videomae_encoder_from_mae_ckpt] loaded from {ckpt_path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("  unexpected (first 5):", unexpected[:5])
    return missing, unexpected

def _gn_groups(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while g > 1:
        if ch % g == 0:
            return g
        g -= 1
    return 1

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, max_groups=8):
        super().__init__()
        g = _gn_groups(out_ch, max_groups)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, max_groups=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvGNAct(out_ch + skip_ch, out_ch, max_groups=max_groups)
        self.conv2 = ConvGNAct(out_ch, out_ch, max_groups=max_groups)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x

class VideoMAEUNETR2D(nn.Module):
    """
    Input:  (B, T, 1, H, W)
    Output: (B, 1, H, W)
    """
    def __init__(self, tile_size=64, num_frames=24, intermediate_size=3072, use_layers=(3, 6, 9, 12), ch=256, max_gn_groups=8):
        super().__init__()
        tubelet_size = 2 if (num_frames % 2 == 0) else 1

        self.vcfg = VideoMAEConfig(
            image_size=tile_size,
            patch_size=16,
            num_channels=1,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=intermediate_size,
        )
        self.encoder = VideoMAEModel(self.vcfg)
        self.use_layers = tuple(use_layers)

        self.patch = self.vcfg.patch_size
        self.Hp = tile_size // self.patch
        self.Wp = tile_size // self.patch
        self.Tp = num_frames // tubelet_size
        self.D = self.vcfg.hidden_size

        self.proj = nn.ModuleDict({str(l): nn.Conv2d(self.D, ch, kernel_size=1) for l in self.use_layers})

        self.up1 = UpBlock(ch, ch, ch, max_groups=max_gn_groups)
        self.up2 = UpBlock(ch, ch, ch, max_groups=max_gn_groups)
        self.up3 = UpBlock(ch, ch, ch, max_groups=max_gn_groups)
        self.up4 = UpBlock(ch, 0,  ch, max_groups=max_gn_groups)
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def tokens_to_2d(self, hs: torch.Tensor) -> torch.Tensor:
        # hs: (B, L, D)
        B, L, D = hs.shape
        n_hw = self.Hp * self.Wp

        # expected tokens (without cls)
        expected = self.Tp * n_hw

        if L == expected + 1:
            x = hs[:, 1:, :]
            Tp = self.Tp
        elif L == expected:
            x = hs
            Tp = self.Tp
        else:
            # try infer Tp from L
            if (L - 1) % n_hw == 0:
                x = hs[:, 1:, :]
                Tp = (L - 1) // n_hw
            elif L % n_hw == 0:
                x = hs
                Tp = L // n_hw
            else:
                raise RuntimeError(f"Cannot reshape tokens: hs={hs.shape}, Hp={self.Hp},Wp={self.Wp},Tp={self.Tp}")

        x = x.reshape(B, Tp, self.Hp, self.Wp, D)
        x = x.mean(dim=1)                     # (B,Hp,Wp,D)
        x = x.permute(0, 3, 1, 2).contiguous() # (B,D,Hp,Wp)
        return x

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # --- Force encoder in fp32 for stability (even if outer AMP is enabled) ---
        with torch.cuda.amp.autocast(enabled=False):
            out = self.encoder(video.float(), output_hidden_states=True, return_dict=True)
            hss = out.hidden_states  # len=13

            feats = {}
            for l in self.use_layers:
                f2d = self.tokens_to_2d(hss[l])
                feats[l] = self.proj[str(l)](f2d)

        x = feats[self.use_layers[-1]]
        x = self.up1(x, feats[self.use_layers[-2]])
        x = self.up2(x, feats[self.use_layers[-3]])
        x = self.up3(x, feats[self.use_layers[-4]])
        x = self.up4(x, None)
        return self.head(x)

def masked_bce_dice_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    pos_weight: float = 10.0,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
):
    """
    logits,y in any dtype -> compute in fp32
    y: (B,1,H,W) with {0,1,127}
    """
    logits = logits.float()
    y = y.float()

    valid = (y != float(ignore_index)).float()
    y_bin = (y > 0.5).float() * valid

    pw = torch.tensor([pos_weight], device=logits.device, dtype=torch.float32)

    bce = F.binary_cross_entropy_with_logits(logits, y_bin, reduction="none", pos_weight=pw)
    bce = (bce * valid).sum() / (valid.sum() + eps)

    p = torch.sigmoid(logits) * valid
    inter = (p * y_bin).sum()
    den = p.sum() + y_bin.sum()
    dice = (2.0 * inter + eps) / (den + eps)
    dice_loss = 1.0 - dice

    loss = bce_weight * bce + dice_weight * dice_loss
    return loss, bce.detach(), dice_loss.detach()

@torch.no_grad()
def logits_stats(logits: torch.Tensor) -> dict:
    t = logits.float()
    return {
        "mean": float(t.mean().cpu()),
        "std": float(t.std().cpu()),
        "min": float(t.min().cpu()),
        "max": float(t.max().cpu()),
    }
