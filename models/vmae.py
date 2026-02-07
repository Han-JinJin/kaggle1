import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor
import torchvision.transforms as T
import numpy as np
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import segmentation_models_pytorch as smp

from torch.optim import AdamW
from warmup_scheduler import GradualWarmupScheduler

import utils

from torchvision.models.video import swin_transformer
import albumentations as A
from transformers import VideoMAEConfig, VideoMAEForPreTraining
from transformers import VideoMAEModel

class VideoMaeModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, in_chans=8, scheduler=None, wandb_logger=None, freeze=False):
        super(VideoMaeModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15,ignore_index=self.IGNORE_INDEX)

        self.loss_func= lambda x,y: 0.6 * self.loss_func1(x,y)+0.4*self.loss_func2(x,y)

        # VideoMAE expects pixel_values with shape (B, num_frames, num_channels, H, W).
        # In this repo, we treat the depth stack (e.g., 24 slices) as "frames" (num_frames).
        # Each frame is a single-channel grayscale image.

        # IMPORTANT: Use a small tubelet_size (2 by default) to preserve depth/temporal structure.
        # This also matches the original VideoMAE pretraining setting better than collapsing all frames.
        tubelet_size = 2 if in_chans % 2 == 0 else 1

        videomae_config = VideoMAEConfig(
            image_size=size,
            patch_size=16,
            num_channels=1,
            num_frames=in_chans,          # here in_chans == number of frames (usually CFG.valid_chans)
            tubelet_size=tubelet_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,       # match videomae-base
            norm_pix_loss=True,
        )

        # Prefer loading pretrained weights (will re-init any mismatched shapes automatically).
        # If offline / download fails, fall back to random init.
        try:
            self.encoder = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base",
                config=videomae_config,
                ignore_mismatched_sizes=True,
            )
            print("✅ Loaded VideoMAE pretrained weights (MCG-NJU/videomae-base) with ignore_mismatched_sizes=True")
        except Exception as e:
            print(f"⚠️ Could not load pretrained VideoMAE weights, falling back to random init. Reason: {e}")
            self.encoder = VideoMAEModel(videomae_config)

        # Remove classifier head, keep norm
        if hasattr(self.encoder, "head"):
            self.encoder.head = nn.Identity()
    
        embed_dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 4, (self.hparams.size // 16) ** 2)
        )

    def forward(self, x):
        """Forward.

        Expected input from DataLoader (default in this repo):
            x: (B, T, C, H, W)
        where:
            T = number of depth slices used as frames (typically CFG.valid_chans)
            C = 1 (grayscale)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B,T,C,H,W or B,C,T,H,W). Got shape: {tuple(x.shape)}")

        # VideoMAE expects (B, T, C, H, W).
        if x.shape[2] == self.encoder.config.num_channels:
            pixel_values = x  # already (B, T, C, H, W)
        elif x.shape[1] == self.encoder.config.num_channels:
            pixel_values = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        else:
            raise ValueError(
                f"Channel mismatch: got input shape {tuple(x.shape)} but VideoMAE config expects num_channels={self.encoder.config.num_channels}. "
                f"Make sure VideoDataset outputs C={self.encoder.config.num_channels}."
            )

        # Sanity checks (fail fast with clear message)
        if pixel_values.shape[1] != self.encoder.config.num_frames:
            raise ValueError(
                f"Frame mismatch: pixel_values has T={pixel_values.shape[1]}, but VideoMAE config has num_frames={self.encoder.config.num_frames}. "
                f"For this repo, set model num_frames = CFG.valid_chans (NOT CFG.in_chans)."
            )

        if not hasattr(self, "_printed_input_shape"):
            print(
                f"[VMAE] pixel_values shape={tuple(pixel_values.shape)} (B,T,C,H,W); "
                f"num_frames={self.encoder.config.num_frames}, num_channels={self.encoder.config.num_channels}"
            )
            self._printed_input_shape = True

        features = self.encoder(pixel_values)
        tokens = features.last_hidden_state
        cls = tokens[:, 0]
        out = self.classifier(cls)

        # low-res logits: (B, 1, size//16, size//16)
        out = out.view(-1, 1, self.hparams.size // 16, self.hparams.size // 16)
        return out

        
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # print(outputs.shape)
        # print(y.shape)
        loss = self.loss_func(outputs, y)
        if torch.isnan(loss):
            print("Loss nan encountered")
        self.log("train/total_loss", loss.item(),on_step=True, on_epoch=True, prog_bar=True)
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=(self.hparams.size // y_preds.shape[-1]),mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
        
    def configure_optimizers(self):
        weight_decay = 0.
        base_lr = self.hparams.lr

        # 1️⃣ Backbone parameters in their own group
        backbone_params = list(self.parameters())

        # 5 4Optimizer
        optimizer = AdamW(backbone_params, lr=base_lr, weight_decay=weight_decay)

        # 6 Scheduler
        return [optimizer]



    def on_validation_epoch_end(self):
        mask_pred_tensor = torch.tensor(self.mask_pred, dtype=torch.float32, device=self.device)
        mask_count_tensor = torch.tensor(self.mask_count, dtype=torch.float32, device=self.device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mask_pred_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mask_count_tensor, op=dist.ReduceOp.SUM)

        if self.trainer.is_global_zero:
            mask_pred_np = mask_pred_tensor.cpu().numpy()
            mask_count_np = mask_count_tensor.cpu().numpy()
            final_mask = np.divide(
                mask_pred_np,
                mask_count_np,
                out=np.zeros_like(mask_pred_np),
                where=mask_count_np != 0
            )

            self.hparams.wandb_logger.log_image(key="masks", images=[np.clip(final_mask, 0, 1)], caption=["probs"])

        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        
        
def load_weights(model, ckpt_path, strict=True, map_location='cpu'):
    """
    Loads weights from a checkpoint into the model.
    
    Args:
        model: An instance of TimesfomerModel.
        ckpt_path: Path to the .ckpt file saved by PyTorch Lightning.
        strict: Whether to strictly enforce that the keys in state_dict match.
        map_location: Where to load the checkpoint (e.g., 'cpu', 'cuda').

    Returns:
        model: The model with loaded weights.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    
    # For Lightning checkpoints, weights are under 'state_dict'
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Strip 'model.' prefix if saved with Lightning
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
    
    print("Loaded checkpoint from:", ckpt_path)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return model
        
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(optimizer, scheduler=None, epochs=15, steps_per_epoch=10):
    if scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=epochs * 50,
            pct_start=0.1,       # 10% warmup
            anneal_strategy='cos',  # Cosine decay after warmup
            cycle_momentum=False  # Turn off momentum scheduling (common for AdamW)
        )
    elif scheduler == 'cosine':
        scheduler_after = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=1e-6)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=1.0, total_epoch=4, after_scheduler=scheduler_after)
    elif scheduler == 'linear':
        scheduler_after = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.05, total_iters=epochs)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=5.0, total_epoch=4, after_scheduler=scheduler_after)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)