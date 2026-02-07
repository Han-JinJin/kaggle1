import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, TimesformerModel
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
import sys
sys.path.append('/home/ubuntu/TimeSformer')  # adjust to your path
from timesformer.models.vit import TimeSformer 


# Convert to PIL and then to 3 channels
pil_transform = T.Compose([
    T.ToPILImage(),                    # convert (C, H, W) to PIL
    T.Grayscale(num_output_channels=3),  # convert to 3 channels
])

class TimesformerDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.transform = transform
        self.xyxys=xyxys
        
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600", use_fast=True)
        self.processor.do_resize = False
        self.processor.size = {"height": cfg.size, "width": cfg.size}
        self.pil_transform  = pil_transform
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32), 
            # T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
            ])
        
    def __len__(self):
        return len(self.images)
    
    def fourth_augment(self,image, in_chans=8):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(in_chans-6, in_chans)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image
    
    def shuffle_d_axis(self,image):
        # image shape: (H, W, D)
        d = image.shape[2]
        shuffled_indices = np.arange(d)
        np.random.shuffle(shuffled_indices)

        # Reorder along D axis
        image_shuffled = image[:, :, shuffled_indices]
        return image_shuffled
    
    def z_circular_shift_np(self, volume, max_shift=4, prob=0.5, cutout_size=1, cutout_prob=0.5):
        """
        Circularly shift slices along Z-axis (last dim) by a random integer in [-max_shift, max_shift].
        Then randomly cut out (zero out) a contiguous block of slices along Z-axis.
        
        Args:
            volume: np.ndarray shape (H, W, D) or (C, H, W, D)
            max_shift: max absolute shift (int). shift=0 means no-op.
            prob: probability to apply shift
            cutout_size: number of consecutive slices to cut out
            cutout_prob: probability to apply cutout
        
        Returns:
            volume after augmentation
        """

        if (random.random() > prob) or (max_shift == 0):
            shifted_volume = volume
        else:
            D = volume.shape[-1]
            shift = random.randint(-max_shift, max_shift)
            if shift == 0:
                shifted_volume = volume
            else:
                shifted_volume = np.roll(volume, shift=shift, axis=-1)
                        # Apply cutout with given probability
        if (random.random() < cutout_prob) and (cutout_size > 0):
            D = shifted_volume.shape[-1]
            # Ensure cutout size is not larger than volume depth
            cutout_size_clamped = min(cutout_size, D)
            start_idx = random.randint(0, D - cutout_size_clamped)
            # Zero out the block along the last axis
            if shifted_volume.ndim == 3:
                # shape: (H, W, D)
                shifted_volume[:, :, start_idx:start_idx + cutout_size_clamped] = 0
            elif shifted_volume.ndim == 4:
                # shape: (C, H, W, D)
                shifted_volume[:, :, :, start_idx:start_idx + cutout_size_clamped] = 0
            else:
                raise ValueError("Unsupported volume shape for cutout")

        return shifted_volume

    def __getitem__(self, idx):
        if self.xyxys is not None: #VALID
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            
            image = image.permute(1,0,2,3)
            image = image.repeat(1, 3, 1, 1)
            image = torch.stack([self.video_transform(f) for f in image])
            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]

            # image=self.fourth_augment(image, self.cfg.in_chans)
            # image = self.shuffle_d_axis(image)
            image = self.z_circular_shift_np(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
                
            image = image.permute(1,0,2,3)
            image = image.repeat(1, 3, 1, 1)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            # print('image',image.shape)
            return image, label
                
            
            



class TimesfomerModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None):
        super(TimesfomerModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)
        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        
        # # self.backbone=TimeSformer(img_size=128, num_classes=0, num_frames=4,attention_type='divided_space_time', in_chans=3)
        # from timesformer.models.vit import default_cfgs

        # # Add a config for patch 4
        # default_cfgs['vit_base_patch4_224'] = ''


        # # Now you can safely instantiate the backbone
        # self.backbone = TimeSformer(
        #     img_size=224,
        #     num_classes=0,
        #     num_frames=96,
        #     attention_type='divided_space_time',
        #     in_chans=3,
        #     patch_size=4
        # )
        # self.backbone.default_cfg = checkpoint['config']
        # self.backbone = TimeSformer(img_size=224, num_classes=0, num_frames=96, attention_type='divided_space_time')#,  pretrained_model='checkpoints/TimeSformer_divST_96x4_224_K600.pyth')
        # Freeze all backbone parameters
        # print(self.backbone)
        # checkpoint = torch.load(
        #     'checkpoints/TimeSformer_divST_96x4_224_K600.pyth',
        #     map_location='cpu',
        #     weights_only=False  # ✅ important!
        # )
        self.backbone=TimeSformer(img_size=224, num_classes=0, num_frames=96,attention_type='divided_space_time', in_chans=3)
        # # self.backbone.default_cfgs['vit_base_patch4_224'] = self.backbone.default_cfgs['vit_base_patch16_224'].copy()

        # checkpoint = torch.load('checkpoints/TimeSformer_divST_8x32_224_HowTo100M.pyth')
        checkpoint = torch.load('checkpoints/TimeSformer_divST_96x4_224_K600.pyth')
        
        state_dict = checkpoint['model_state']
        # print("Original checkpoint keys:", list(state_dict.keys()))
        state_dict.pop('model.head.weight', None)
        state_dict.pop('model.head.bias', None)
        # # 3️⃣ Interpolate temporal positional embeddings
        # def interpolate_temporal_pos_embed(model, state_dict, new_num_frames):
        #     pos_embed_ckpt = state_dict['pos_embed']      # (1, N+1, C)
        #     cls_token = pos_embed_ckpt[:, 0:1, :]
        #     patch_pos = pos_embed_ckpt[:, 1:, :]         # (1, N, C)
            
        #     N = patch_pos.shape[1]
        #     T_old = 96                                   # checkpoint frames
        #     H_W = N // T_old
            
        #     patch_pos = patch_pos.view(1, T_old, H_W, -1)
        #     patch_pos = F.interpolate(patch_pos, size=(new_num_frames, H_W), mode='bilinear', align_corners=False)
        #     patch_pos = patch_pos.view(1, new_num_frames * H_W, -1)
            
        #     state_dict['pos_embed'] = torch.cat([cls_token, patch_pos], dim=1)

        # interpolate_temporal_pos_embed(self.backbone, state_dict, 8)

        # 4️⃣ Load state_dict
        # self.backbone.load_state_dict(state_dict, strict=False)
        # print("Weights loaded with interpolated temporal embeddings!")

        try:
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Backbone weights loaded successfully.")
        except RuntimeError as e:
            print(f"Failed to load backbone weights: {e}")
        
        self.classifier = nn.Sequential(
            nn.Linear(768, (self.hparams.size //16)**2)
        )
        
    def forward(self, x):

        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
        features = self.backbone(x)  # (B, 768)
        preds = self.classifier(features)  # (B, H'*W')
        preds = preds.reshape(-1,1,self.hparams.size//16,self.hparams.size//16)
        return preds
   
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        outputs = outputs
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
        outputs = outputs
        # print(outputs.shape)
        # print(y.shape)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def configure_optimizers(self):
        # Separate the head parameters
        head_params = list(self.classifier.parameters())
        other_params = [p for n, p in self.backbone.named_parameters() if "classifier" not in n]

        param_groups = [
            {'params': other_params, 'lr': self.hparams.lr},
            {'params': head_params, 'lr': self.hparams.lr },  # 10x LR for the head
        ]

        optimizer = AdamW(param_groups)
    
        scheduler = get_scheduler(optimizer,scheduler=self.hparams.scheduler)
        return [optimizer], [scheduler]
    # def configure_optimizers(self):
    #     weight_decay = 0.001
    #     base_lr = self.hparams.lr

    #     # 1️⃣ Backbone parameters in their own group
    #     backbone_params = list(self.backbone.backbone.parameters())

    #     # 2️⃣ Everything else (decoder, head, etc.)
    #     other_params = [
    #         p for n, p in self.backbone.named_parameters()
    #         if p.requires_grad and not any(k in n for k in ["backbone"])
    #     ]

    #     # 3️⃣ Sanity checks
    #     assert len(other_params) > 0, "No parameters found for 'other_params' group"
    #     assert len(backbone_params) > 0, "No parameters found for 'backbone_params' group"
        
    #     total_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
    #     backbone_count = sum(p.numel() for p in backbone_params)
    #     other_count = sum(p.numel() for p in other_params)

    #     assert backbone_count + other_count == total_params, \
    #         f"Mismatch: total={total_params}, backbone={backbone_count}, other={other_count}"

    #     # 4️⃣ Define param groups
    #     param_groups = [
    #         {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
    #         {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
    #     ]

    #     # 5 4Optimizer
    #     optimizer = AdamW(param_groups)

    #     # 6 Scheduler
    #     scheduler = get_scheduler(optimizer, scheduler="cosine", epochs=15)

    #     return [optimizer], [scheduler]


        
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

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict,weights_only=True)
    
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

def get_scheduler(optimizer, scheduler=None, epochs=30):
    if scheduler == 'cosine':
        scheduler_after = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 20, eta_min=1e-6)
    else:
        scheduler_after = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,   # start at the current learning rate
            end_factor=0.05,    # end at 1% of the current learning rate
            total_iters = epochs
        )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_after)
    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)