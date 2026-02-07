import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim import AdamW
from models.resnetall import generate_model


import utils


def init_weights(m):
    """Initialize weights for Conv2d layers using Kaiming normal initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class Decoder(nn.Module):
    """Decoder module for upsampling and combining encoder features."""

    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    encoder_dims[i] + encoder_dims[i - 1],
                    encoder_dims[i - 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))
        ])

        self.logit = nn.Conv2d(encoder_dims[0], 1, kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

class RegressionPLModel(pl.LightningModule):
    """PyTorch Lightning model for ink detection using 3D ResNet backbone."""

    def __init__(
        self,
        pred_shape,
        size=None,
        enc='',
        with_norm=False,
        total_steps=780,
        a=0.7,
        b=0.3,
        smooth=0.25,
        dropout=None,
        max=True
    ):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        # Loss functions
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=self.hparams.smooth)
        self.loss_func = lambda x, y: (
            self.hparams.a * self.loss_func1(x, y) +
            self.hparams.b * self.loss_func2(x, y)
        )
        self.mask_gt = np.zeros(self.hparams.pred_shape)

        # Backbone: 3D ResNet-101
        self.backbone = generate_model(
            model_depth=101,
            n_input_channels=1,
            forward_features=True,
            n_classes=1039
        )
        state_dict = torch.load('./checkpoints/r3d101_KM_200ep.pth')["state_dict"]
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.backbone.load_state_dict(state_dict, strict=False)

        # Decoder
        dummy_input = torch.rand(1, 1, 20, 256, 256)
        encoder_dims = [x.size(1) for x in self.backbone(dummy_input)]
        self.decoder = Decoder(encoder_dims=encoder_dims, upscale=1)

        # Normalization
        self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        """Forward pass through backbone and decoder."""
        if self.hparams.with_norm:
            x = self.normalization(x)

        feat_maps = self.backbone(x)

        # Pool temporal dimension
        if self.hparams.max:
            feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        else:
            feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]

        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        outputs = self(x)
        loss = self.loss_func(outputs, y)

        if torch.isnan(loss):
            print("Loss NaN encountered")

        # Log metrics
        self.log("train/total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=True, sync_dist=True)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y, xyxys = batch
        outputs = self(x)

        loss = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        y_cpu = y.to('cpu')

        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            # Prediction
            pred_patch = F.interpolate(
                y_preds[i].unsqueeze(0).float(),
                size=(self.hparams.size, self.hparams.size),
                mode='bilinear'
            ).squeeze(0).squeeze(0).numpy()
            self.mask_pred[y1:y2, x1:x2] += pred_patch

            # Ground truth
            gt_patch = F.interpolate(
                y_cpu[i].unsqueeze(0).float(),
                size=(self.hparams.size, self.hparams.size),
                mode='bilinear'
            ).squeeze(0).squeeze(0).numpy()
            self.mask_gt[y1:y2, x1:x2] = gt_patch

            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        mask_pred_tensor = torch.tensor(self.mask_pred, dtype=torch.float32, device=self.device)
        mask_count_tensor = torch.tensor(self.mask_count, dtype=torch.float32, device=self.device)

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

    # def on_validation_epoch_end(self):
    #     """Calculate and log validation metrics at the end of each epoch."""
    #     # Average overlapping predictions
    #     self.mask_pred = np.divide(
    #         self.mask_pred,
    #         self.mask_count,
    #         out=np.zeros_like(self.mask_pred),
    #         where=self.mask_count != 0
    #     )

    #     pred_binary = (self.mask_pred > 0.5).astype(np.float32)

    #     # Calculate metrics
    #     if hasattr(self, 'mask_gt'):
    #         gt_binary = self.mask_gt.astype(np.float32)

    #         # Intersection and Union for IoU
    #         intersection = np.logical_and(pred_binary, gt_binary).sum()
    #         union = np.logical_or(pred_binary, gt_binary).sum()
    #         iou = intersection / (union + 1e-8)

    #         # Dice coefficient
    #         dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)

    #         # Pixel accuracy
    #         correct = (pred_binary == gt_binary).sum()
    #         total = gt_binary.size
    #         pixel_acc = correct / total

    #         # Precision and Recall
    #         tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    #         fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
    #         fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()

    #         precision = tp / (tp + fp + 1e-8)
    #         recall = tp / (tp + fn + 1e-8)
    #         f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    #         # Log metrics
    #         self.log("val/iou", iou, prog_bar=True, sync_dist=True)
    #         self.log("val/dice", dice, prog_bar=True, sync_dist=True)
    #         self.log("val/pixel_acc", pixel_acc, prog_bar=True, sync_dist=True)
    #         self.log("val/precision", precision, sync_dist=True)
    #         self.log("val/recall", recall, sync_dist=True)
    #         self.log("val/f1", f1, sync_dist=True)

    #         # Print only on master process
    #         if self.trainer.is_global_zero:
    #             print(f"\n=== Validation Metrics ===")
    #             print(f"IoU: {iou:.4f}")
    #             print(f"Dice: {dice:.4f}")
    #             print(f"Pixel Accuracy: {pixel_acc:.4f}")
    #             print(f"Precision: {precision:.4f}")
    #             print(f"Recall: {recall:.4f}")
    #             print(f"F1 Score: {f1:.4f}")

    #     # Save metrics locally per experiment
    #     if self.trainer.is_global_zero:
    #         exp_dir = os.path.join(self.trainer.default_root_dir, "metrics")
    #         os.makedirs(exp_dir, exist_ok=True)

    #         csv_path = os.path.join(exp_dir, "metrics.csv")
    #         file_exists = os.path.isfile(csv_path)

    #         with open(csv_path, "a", newline="") as f:
    #             writer = csv.writer(f)

    #             if not file_exists:
    #                 writer.writerow([
    #                     "timestamp",
    #                     "epoch",
    #                     "iou",
    #                     "dice",
    #                     "pixel_acc",
    #                     "precision",
    #                     "recall",
    #                     "f1"
    #                 ])

    #             writer.writerow([
    #                 datetime.now().isoformat(),
    #                 self.current_epoch,
    #                 float(iou),
    #                 float(dice),
    #                 float(pixel_acc),
    #                 float(precision),
    #                 float(recall),
    #                 float(f1)
    #             ])

    #     # Log image only on master process
    #     if self.trainer.is_global_zero:
    #         wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"])

    #     # Reset masks
    #     self.mask_pred = np.zeros(self.hparams.pred_shape)
    #     self.mask_count = np.zeros(self.hparams.pred_shape)
    #     if hasattr(self, 'mask_gt'):
    #         self.mask_gt = np.zeros(self.hparams.pred_shape)
                
    def configure_optimizers(self):
        weight_decay = 0.
        base_lr = self.hparams.lr

        # 1️⃣ Backbone parameters in their own group
        backbone_params = list(self.parameters())

        # 5 4Optimizer
        optimizer = AdamW(backbone_params, lr=base_lr, weight_decay=weight_decay)

        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[base_lr],
            pct_start=0.35,
            steps_per_epoch=143,
            epochs=25,
            final_div_factor=1e2
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        x, y, xyxys = batch

        outputs = self(x)
        y_preds = torch.sigmoid(outputs)

        return {
            'preds': y_preds.cpu(),
            'xyxys': xyxys
        }


def scheduler_step(scheduler, avg_val_loss, epoch):
    """Step the learning rate scheduler."""
    scheduler.step(epoch)
