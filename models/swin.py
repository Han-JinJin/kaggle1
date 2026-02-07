import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
# from transformers import AutoImageProcessor
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
#  torchvision.models.swin_transformer.SwinTransformer 
import albumentations as A
# from transformers import VideoMAEConfig, VideoMAEForPreTraining
# from transformers import VideoMAEModel



class TimesformerDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None, norm=False, aug=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.rotate = A.Compose([A.Rotate(8,p=1)])
        self.xyxys=xyxys
        self.aug = aug
        self.scale_factor = 16
        
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),  # scales to [0.0, 1.0]
        ])
        
        # Conditionally add Normalize transformation
        if norm:
            self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32), 
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225]),
            T.Normalize(mean=[0.5], std=[0.5])
            ])
                
    def __len__(self):
        return len(self.images)
    
    # def fourth_augment(self, image):
    #     """
    #     Custom channel augmentation that returns exactly 24 channels.
    #     """
    #     # always select 8
    #     cropping_num =  self.cfg.valid_chans 

    #     # pick crop indices
    #     start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
    #     crop_indices = np.arange(start_idx, start_idx + cropping_num)

    #     # pick where to paste them
    #     start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

    #     # container
    #     image_tmp = np.zeros_like(image)

    #     # paste cropped channels
    #     image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

    #     # # optional random cutout
    #     # cutout_idx = random.randint(0, 1)
    #     # temporal_random_cutout_idx = np.arange(start_paste_idx, start_paste_idx + cutout_idx)
    #     # if random.random() > 0.4:
    #     #     image_tmp[..., temporal_random_cutout_idx] = 0

    #     # finally, keep only 24 channels
    #     image = image_tmp[..., start_paste_idx:start_paste_idx + cropping_num]

    #     return image
    def fourth_augment(self, image):
        """
        Custom channel augmentation that returns exactly 24 channels.
        """
        # always select 24 channels
        remove_n = 0
        cropping_num = self.cfg.valid_chans  + remove_n

        # pick crop indices
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        # pick where to paste them
        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        
        # container
        image_tmp = np.zeros_like(image)
        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]
        
        # --- remove 2 random channels (instead of blacking out) ---
        all_indices = np.arange(start_paste_idx, start_paste_idx + cropping_num)
        remove_idx = np.random.choice(all_indices, remove_n, replace=False)
        # keep all except removed ones
        keep_idx = np.array([i for i in all_indices if i not in remove_idx])
        image_tmp = image_tmp[..., keep_idx]

        return image_tmp
    
    def shuffle(self, image):
        """
        Channel shuffle augmentation that returns exactly 24 channels.
        """
        # Shuffle channels randomly
        shuffled_indices = np.random.permutation(self.cfg.valid_chans)
        image_shuffled = image[..., shuffled_indices]

        return image_shuffled


    def __getitem__(self, idx):
        if self.xyxys is not None: #VALID
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//self.scale_factor,self.cfg.size//self.scale_factor)).squeeze(0)
                
            image = image.permute(1,0,2,3)
            # image = image.repeat(1,3,1,1)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label,xy
            
        else:
            image = self.images[idx]
            label = self.labels[idx]

            if self.aug == 'fourth':
                image=self.fourth_augment(image)
                # image = self.shuffle(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//self.scale_factor,self.cfg.size//self.scale_factor)).squeeze(0)

            image = image.permute(1,0,2,3)
            
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label

# HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH  GOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
class Patch3DTransformerSegmentation(nn.Module):
    def __init__(self, num_classes=1, embed_dim=768, num_heads=8, depth=2, patch_output=4):
        super().__init__()
        self.patch_output = patch_output
        self.num_classes = num_classes

        backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

        # Modify first conv layer to accept 1 channel instead of 3
        old_proj = backbone.patch_embed.proj
        new_proj = nn.Conv3d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )
        # Initialize weights by summing across RGB channels
        with torch.no_grad():
            # old_proj.weight shape: [out_channels, 3, kT, kH, kW]
            summed = old_proj.weight.sum(dim=1, keepdim=True)   # -> [out_channels, 1, kT, kH, kW]
            new_proj.weight.copy_(summed)

            # If bias exists, copy it too
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        # Replace the old conv with the new one
        backbone.patch_embed.proj = new_proj
        
        backbone = nn.Sequential(*list(backbone.children())[:-2]) 
        
        # ckpt_path = "pretraining/checkpoints/tiny_epoch=72.ckpt"
        # ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)  # CPU first, move to GPU later if needed

        # if 'state_dict' in ckpt:
        #     state_dict = ckpt['state_dict']
        # else:
        #     state_dict = ckpt
        # # Filter only encoder weights (keys start with 'encoder.')
        # encoder_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('encoder.'):
        #         # remove 'encoder.' prefix to match backbone keys
        #         encoder_state_dict[k.replace('encoder.', '')] = v

        # missing, unexpected =  backbone.load_state_dict(encoder_state_dict, strict=False)
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)
        self.backbone = backbone
        # # Freeze the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim,
            dropout=0.2,
            activation="gelu",
            batch_first=True 
        )
        self.embed_dim = embed_dim

        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, (patch_output ** 2))
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        # print(x.shape)
        feats = self.backbone(x)  # (B, T', H', W',embed_dim)

        # Temporal pooling
        feats = feats.max(dim=1)[0]# (B, 1, Hf, Wf, embed_dim,)
        
        B, Hf, Wf, D = feats.shape

        # Move embed_dim to dim 1 and flatten patches
        patch_tokens = feats.permute(0, 3, 1, 2).contiguous()  # (B, 768, 8, 7, 7)
        patch_tokens = patch_tokens.view(B, Hf*Wf, -1)        # (B, 768, 8*7*7=392)
       
        # print(patch_tokens.shape)
        transformed_tokens = self.decoder(patch_tokens) 
        logits = self.classifier(transformed_tokens)
        logits = logits.permute(0, 2, 1).view(B, self.patch_output**2, Hf, Wf)  # (B, patch_output^2, Hf, Wf)
        
        # logits = self.head(transformed_tokens)   # (B, num_classes, Hf, Wf)
        logits = F.pixel_shuffle(logits, upscale_factor=self.patch_output) 
        return logits
    
class SwinModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
        super(SwinModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15,ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y) +0.5*self.loss_func2(x,y)
        self.backbone = Patch3DTransformerSegmentation(num_classes=1, patch_output=4)
    
    def forward(self, x):

        x = x.permute(0,2,1,3,4)
        output = self.backbone(x)  # runs backbone, sets self.feature
        return output

# # class Swin3DEncoder(nn.Module):
# #     def __init__(self,pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=16.ckpt'):
# #         super().__init__()
# #         # backbone = swin_transformer.SwinTransformer3d(
# #         #     patch_size=[2, 4, 4],
# #         #     embed_dim=96,
# #         #     depths=[2, 2, 12, 2],
# #         #     num_heads=[3, 6, 12, 24],
# #         #     window_size=[8, 7, 7],
# #         #     stochastic_depth_prob=0.1,
# #         # )
# #         # backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')
# #         #         # Get old weights
# #         # old_conv = backbone.patch_embed.proj  # Conv3d(3, 128, ...)
# #         # weight = old_conv.weight  # [128, 3, 2, 4, 4]
# #         # bias = old_conv.bias      # [128]

# #         # # Adapt weights: average across RGB → 1 channel
# #         # new_weight = weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]

# #         # # Replace conv with new one (keep out_channels=128!)
# #         # backbone.patch_embed.proj = nn.Conv3d(
# #         #     in_channels=1,
# #         #     out_channels=128,
# #         #     kernel_size=(2, 4, 4),
# #         #     stride=(2, 4, 4),
# #         #     bias=True
# #         # )

# #         # # # Load adapted weights
#         # backbone.patch_embed.proj.weight = nn.Parameter(new_weight)
#         # backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())  # shape [128]
#         # self.backbone = backbone
#         self.backbone = swin_transformer.swin3d_b(weights='KINETICS400_IMAGENET22K_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # remove classifier head, eep norm
#         self.backbone.head = nn.Identity()

#         # if pretrained_ckpt:
#         #     ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#         #     ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#         #     model_state_dict = self.backbone.state_dict()

#         #     new_state_dict = {}
#         #     used_ckpt_keys = set()  # keep track of keys already used


#         #     for k_model in model_state_dict.keys():
#         #         # Skip decoder stuff
#         #         if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#         #             continue
#         #         print(k_model)
#         #         # Find corresponding key in checkpoint
#         #         if k_model in ckpt_state_dict:
#         #             new_state_dict[k_model] = ckpt_state_dict[k_model]
#         #         else:
#         #             # Fallback: find first unused key with matching shape
#         #             for k_ckpt, v_ckpt in ckpt_state_dict.items():
#         #                 if k_ckpt in used_ckpt_keys:
#         #                     continue  # skip already used keys
#         #                 if v_ckpt.shape == model_state_dict[k_model].shape:
#         #                     new_state_dict[k_model] = v_ckpt
#         #                     used_ckpt_keys.add(k_ckpt)
#         #                     print(f"Fallback match: {k_model} <- {k_ckpt}")
#         #                     break
                            
#         #     # Load into model
#         #     msg = self.backbone.load_state_dict(new_state_dict, strict=False)
#         #     print("Loaded:", msg)
        
        

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips  # [s1, s2, s3, s4]




# UNT BATM

# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=17.ckpt', in_chans=1):
#         super().__init__()
#         # Load Swin3D
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # remove classifier head, keep norm
#         self.backbone.head = nn.Identity()

#         # pretrained_ckpt = 'pretraining/checkpoints/tiny_epoch=71.ckpt'
#         # # optionally load ckpt
#         # if pretrained_ckpt:
#         #     ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#         #     ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#         #     model_state_dict = self.backbone.state_dict()

#         #     new_state_dict = {}
#         #     used_ckpt_keys = set()  # keep track of keys already used


#         #     for k_model in model_state_dict.keys():
#         #         # Skip decoder stuff
#         #         if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#         #             continue
#         #         print(k_model)
#         #         # Find corresponding key in checkpoint
#         #         if k_model in ckpt_state_dict:
#         #             new_state_dict[k_model] = ckpt_state_dict[k_model]
#         #         else:
#         #             # Fallback: find first unused key with matching shape
#         #             for k_ckpt, v_ckpt in ckpt_state_dict.items():
#         #                 if k_ckpt in used_ckpt_keys:
#         #                     continue  # skip already used keys
#         #                 if v_ckpt.shape == model_state_dict[k_model].shape:
#         #                     new_state_dict[k_model] = v_ckpt
#         #                     used_ckpt_keys.add(k_ckpt)
#         #                     print(f"Fallback match: {k_model} <- {k_ckpt}")
#         #                     break
                            
#         #     # Load into model
#         #     msg = self.backbone.load_state_dict(new_state_dict, strict=False)
#         #     print("Loaded:", msg)

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips  # [s1, s2, s3, s4]

# class UpBlock2D(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x, skip):
#         # squeeze away the temporal dim if present
#         if x.dim() == 5:  # (B, C, 1, H, W)
#             x = x.squeeze(2)
#         if skip.dim() == 5:
#             skip = skip.squeeze(2)

#         x = self.up(x)

#         # pad if needed
#         if x.shape[2:] != skip.shape[2:]:
#             diff = [s - d for s, d in zip(skip.shape[2:], x.shape[2:])]
#             x = nn.functional.pad(x, [0, diff[1], 0, diff[0]])

#         x = torch.cat([x, skip], dim=1)
#         return self.conv(x)
    
# class TransformerHead2D(nn.Module):
#     def __init__(self, in_dim, num_heads=8, mlp_ratio=4.0, out_dim=1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(in_dim)
#         self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
#         self.norm2 = nn.LayerNorm(in_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, int(in_dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(in_dim * mlp_ratio), in_dim)
#         )
#         self.final = nn.Linear(in_dim, out_dim)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape
#         x_seq = x.flatten(2).permute(2, 0, 1)       # (H*W, B, C)
#         x_attn = self.attn(self.norm1(x_seq),
#                            self.norm1(x_seq),
#                            self.norm1(x_seq))[0]
#         x_seq = x_seq + x_attn
#         x_seq = x_seq + self.mlp(self.norm2(x_seq))
#         x_seq = self.final(x_seq)                   # (H*W, B, out_dim)
#         x = x_seq.permute(1, 2, 0).view(B, -1, H, W)
#         return x
    
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15,ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y) +0.5*self.loss_func2(x,y)
#         self.backbone = Swin3DEncoder()
#         self.normalization=nn.BatchNorm3d(num_features=1) 

#         # Channels per stage (tiny config)
#         self.enc_channels = [96, 192, 384, 768]

#         # Bottleneck conv to reshape final feature
#         self.bottleneck = nn.Conv3d(self.enc_channels[-1], self.enc_channels[-1], kernel_size=1)

#         # Decoder (reverse order)
#         self.up3 = UpBlock2D(self.enc_channels[-1], self.enc_channels[-2])
#         self.up2 = UpBlock2D(self.enc_channels[-2], self.enc_channels[-3])
#         self.up1 = UpBlock2D(self.enc_channels[-3], self.enc_channels[-4])

#         # Final segmentation head
#         self.head = TransformerHead2D(in_dim=192, num_heads=8, mlp_ratio=4.0, out_dim=1)
#         self.aux_head3 = nn.Conv2d(self.enc_channels[-2], 1, kernel_size=1)
#         self.aux_head2 = nn.Conv2d(self.enc_channels[-3], 1, kernel_size=1)

#     def forward(self, x):
#         x= x.permute(0,2,1,3,4)
#         x=self.normalization(x)
#         skips = self.backbone(x) 
#         s1, s2, s3, s4 = skips

#         # Swin returns (B, T, H, W, C) → convert to (B, C, T, H, W)
#         def permute_fmap(fmap):
#             return fmap.permute(0, 4, 1, 2, 3).contiguous()

#         feat_maps = map(permute_fmap, (s1, s2, s3, s4))

#         s1, s2, s3, s4 = [torch.max(f, dim=2, keepdim=True)[0] for f in feat_maps]

#         x = self.up3(s4, s3)
#         x = self.up2(x, s2)
#         # x = self.up1(x, s1)
#         out = self.head(x)

#         return out
    
# CLAUD    
    
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import numpy as np
# import segmentation_models_pytorch as smp
# from torchvision.models.video import swin_transformer

# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=17.ckpt', in_chans=1):
#         super().__init__()
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # Adapt patch_embed for 1 channel
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())
        
#         ckpt_path = "pretraining/checkpoints/tiny_epoch=72.ckpt"
#         ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)  # CPU first, move to GPU later if needed

#         if 'state_dict' in ckpt:
#             state_dict = ckpt['state_dict']
#         else:
#             state_dict = ckpt
#         # Filter only encoder weights (keys start with 'encoder.')
#         encoder_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('encoder.'):
#                 # remove 'encoder.' prefix to match backbone keys
#                 encoder_state_dict[k.replace('encoder.', '')] = v

#         missing, unexpected =  self.backbone.load_state_dict(encoder_state_dict, strict=False)
#         print("Missing keys:", missing)
#         print("Unexpected keys:", unexpected)
#         # self.backbone = backbone
#         self.backbone.head = nn.Identity()

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):
#                 skips.append(x)

#         return skips


# class FeatureAlignmentBlock(nn.Module):
#     """Aligns Swin transformer features to CNN-friendly format"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.align = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x):
#         return self.align(x)


# class UpBlock2D(nn.Module):
#     def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
#         # Now we know exact skip channel size
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x, skip):
#         if x.dim() == 5:
#             x = x.squeeze(2)
#         if skip.dim() == 5:
#             skip = skip.squeeze(2)

#         x = self.up(x)

#         # Handle size mismatch
#         if x.shape[2:] != skip.shape[2:]:
#             diff_h = skip.shape[2] - x.shape[2]
#             diff_w = skip.shape[3] - x.shape[3]
#             x = nn.functional.pad(x, [diff_w//2, diff_w - diff_w//2, 
#                                        diff_h//2, diff_h - diff_h//2])

#         x = torch.cat([x, skip], dim=1)
#         return self.conv(x)


# class TemporalAggregation(nn.Module):
#     """Attention-based temporal aggregation"""
#     def __init__(self, channels):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Conv3d(channels, channels // 4, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(channels // 4, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         # x: (B, C, T, H, W)
#         weights = self.attention(x)  # (B, 1, T, H, W)
#         weighted = x * weights
#         return weighted.sum(dim=2, keepdim=True) / (weights.sum(dim=2, keepdim=True) + 1e-8)


# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, 
#                  freeze=False, dropout=0.1):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary', smooth=0.15, ignore_index=self.IGNORE_INDEX)
#         self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25, ignore_index=self.IGNORE_INDEX)
#         self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

#         self.backbone = Swin3DEncoder()
#         self.normalization = nn.BatchNorm3d(num_features=1)

#         # Swin3D-tiny channels: [96, 192, 384, 768]
#         self.enc_channels = [96, 192, 384, 768]
        
#         # Decoder output channels (more balanced pyramid)
#         self.dec_channels = [512, 256, 128, 64]
        
#         # CRITICAL FIX: Feature alignment blocks for each skip connection
#         # These convert Swin features to CNN-friendly representations
#         self.align4 = FeatureAlignmentBlock(self.enc_channels[3], self.dec_channels[0])
#         self.align3 = FeatureAlignmentBlock(self.enc_channels[2], self.dec_channels[1])
#         self.align2 = FeatureAlignmentBlock(self.enc_channels[1], self.dec_channels[2])
#         self.align1 = FeatureAlignmentBlock(self.enc_channels[0], self.dec_channels[3])
        
#         # Temporal aggregation for each stage
#         self.temp_agg4 = TemporalAggregation(self.enc_channels[3])
#         self.temp_agg3 = TemporalAggregation(self.enc_channels[2])
#         self.temp_agg2 = TemporalAggregation(self.enc_channels[1])
#         self.temp_agg1 = TemporalAggregation(self.enc_channels[0])

#         # Decoder with aligned channel dimensions
#         # Note: skip_ch matches aligned feature channels
#         self.up3 = UpBlock2D(
#             in_ch=self.dec_channels[0], 
#             skip_ch=self.dec_channels[1],  # aligned s3 channels
#             out_ch=self.dec_channels[1],
#             dropout=dropout
#         )
#         self.up2 = UpBlock2D(
#             in_ch=self.dec_channels[1], 
#             skip_ch=self.dec_channels[2],  # aligned s2 channels
#             out_ch=self.dec_channels[2],
#             dropout=dropout
#         )
#         self.up1 = UpBlock2D(
#             in_ch=self.dec_channels[2], 
#             skip_ch=self.dec_channels[3],  # aligned s1 channels
#             out_ch=self.dec_channels[3],
#             dropout=dropout
#         )
        
#         # Segmentation head (no upsampling)
#         self.head = nn.Conv2d(self.dec_channels[3], 1, kernel_size=1)

#     def forward(self, x):
#         x = x.permute(0, 2, 1, 3, 4)
#         x = self.normalization(x)
#         skips = self.backbone(x)
#         s1, s2, s3, s4 = skips
#         # print(s1.shape,s2.shape,s3.shape,s4.shape)

#         # Convert (B, T, H, W, C) → (B, C, T, H, W)
#         def permute_fmap(fmap):
#             return fmap.permute(0, 4, 1, 2, 3).contiguous()

#         s1, s2, s3, s4 = map(permute_fmap, (s1, s2, s3, s4))

#         # Temporal aggregation
#         s1 = self.temp_agg1(s1).squeeze(2)  # → (B, 96, H, W)
#         s2 = self.temp_agg2(s2).squeeze(2)  # → (B, 192, H, W)
#         s3 = self.temp_agg3(s3).squeeze(2)  # → (B, 384, H, W)
#         s4 = self.temp_agg4(s4).squeeze(2)  # → (B, 768, H, W)

#         # CRITICAL: Align Swin features to CNN-friendly format
#         s4_aligned = self.align4(s4)  # (B, 512, H, W)
#         s3_aligned = self.align3(s3)  # (B, 256, H, W)
#         s2_aligned = self.align2(s2)  # (B, 128, H, W)
#         s1_aligned = self.align1(s1)  # (B, 64, H, W)

#         # Decoder with properly aligned features
#         x = self.up3(s4_aligned, s3_aligned)  # (B, 256, H, W)
#         x = self.up2(x, s2_aligned)            # (B, 128, H, W)
#         x = self.up1(x, s1_aligned)            # (B, 64, H, W)
#         # x = self.final_up(x)                   # (B, 32, H, W)
        
#         out = self.head(x)  # (B, 1, H, W)
#         return out
    
    ##########################################################
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# import numpy as np
# import segmentation_models_pytorch as smp
# from torchvision.models.video import swin_transformer


# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt=None, in_chans=1):
#         super().__init__()
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # Adapt patch_embed for 1 channel
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())
#         pretrained_ckpt = 'pretraining/checkpoints/tiny_epoch=72.ckpt'
#         # optionally load ckpt
#         if pretrained_ckpt:
#             ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#             ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#             model_state_dict = self.backbone.state_dict()

#             new_state_dict = {}
#             used_ckpt_keys = set()  # keep track of keys already used


#             for k_model in model_state_dict.keys():
#                 # Skip decoder stuff
#                 if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#                     continue
#                 print(k_model)
#                 # Find corresponding key in checkpoint
#                 if k_model in ckpt_state_dict:
#                     new_state_dict[k_model] = ckpt_state_dict[k_model]
#                 else:
#                     # Fallback: find first unused key with matching shape
#                     for k_ckpt, v_ckpt in ckpt_state_dict.items():
#                         if k_ckpt in used_ckpt_keys:
#                             continue  # skip already used keys
#                         if v_ckpt.shape == model_state_dict[k_model].shape:
#                             new_state_dict[k_model] = v_ckpt
#                             used_ckpt_keys.add(k_ckpt)
#                             print(f"Fallback match: {k_model} <- {k_ckpt}")
#                             break

#         missing, unexpected =  self.backbone.load_state_dict(new_state_dict, strict=False)
#         print("Missing keys:", missing)
#         print("Unexpected keys:", unexpected)
#         # self.backbone = backbone
#         # self.backbone.head = nn.Identity()
#         self.backbone.head = nn.Identity()

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):
#                 skips.append(x)

#         return skips


# class TransformerUpBlock(nn.Module):
#     """Transformer-based upsampling block similar to your patch model"""
#     def __init__(self, in_channels, skip_channels, out_channels, 
#                  num_heads=8, depth=2, upscale_factor=2):
#         super().__init__()
#         self.upscale_factor = upscale_factor
#         self.out_channels = out_channels
        
#         # Project skip connection to match working dimension
#         self.skip_proj = nn.Linear(skip_channels, in_channels) if skip_channels != in_channels else nn.Identity()
        
#         # Project input if needed
#         self.in_proj = nn.Linear(in_channels, in_channels) if in_channels != in_channels else nn.Identity()
        
#         # Transformer decoder (like your patch model)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=in_channels,
#             nhead=num_heads,
#             dim_feedforward=in_channels * 2,
#             dropout=0.1,
#             activation="gelu",
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
#         # Pixel shuffle upsampling (like your patch model)
#         self.classifier = nn.Linear(in_channels, out_channels * (upscale_factor ** 2))
        
#     def forward(self, x, skip):
#         """
#         x: (B, H_x, W_x, C_in) - from previous decoder stage
#         skip: (B, H_s, W_s, C_skip) - from encoder
#         """
#         B, H_skip, W_skip, C_skip = skip.shape
        
#         # Upsample x to match skip spatial dimensions using bilinear interpolation
#         if x.shape[1:3] != skip.shape[1:3]:
#             # (B, H, W, C) -> (B, C, H, W) for interpolate
#             x_up = x.permute(0, 3, 1, 2)
#             x_up = F.interpolate(x_up, size=(H_skip, W_skip), mode='bilinear', align_corners=False)
#             x = x_up.permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        
#         # Project skip connection
#         skip_proj = self.skip_proj(skip)  # (B, H, W, C_in)
        
#         # Combine: concatenate then add
#         combined = x + skip_proj  # (B, H, W, C_in)
        
#         # Flatten spatial dimensions for transformer
#         B, H, W, C = combined.shape
#         tokens = combined.view(B, H * W, C)  # (B, N, C)
        
#         # Apply transformer
#         transformed = self.transformer(tokens)  # (B, N, C)
        
#         # Pixel shuffle upsampling
#         logits = self.classifier(transformed)  # (B, N, out_channels * upscale^2)
#         logits = logits.permute(0, 2, 1).view(B, self.out_channels * (self.upscale_factor ** 2), H, W)
#         output = F.pixel_shuffle(logits, upscale_factor=self.upscale_factor)
        
#         # Convert back to (B, H', W', C) format
#         output = output.permute(0, 2, 3, 1)  # (B, H', W', out_channels)
        
#         return output


# class SwinTransformerUNet(nn.Module):
#     def __init__(self, num_classes=1, num_heads=8, depth=2):
#         super().__init__()
        
#         self.encoder = Swin3DEncoder()
        
#         # Swin3D-tiny channels: [96, 192, 384, 768]
#         self.enc_channels = [96, 192, 384, 768]
        
#         # Temporal max pooling for each skip (like your patch model)
#         # No learnable parameters - just max pool
        
#         # Decoder with transformer blocks at each level
#         # Going from 768 -> 384 -> 192 -> 96
#         self.up3 = TransformerUpBlock(
#             in_channels=self.enc_channels[3],   # 768
#             skip_channels=self.enc_channels[2],  # 384
#             out_channels=self.enc_channels[2],   # 384
#             num_heads=num_heads,
#             depth=depth,
#             upscale_factor=2
#         )
        
#         self.up2 = TransformerUpBlock(
#             in_channels=self.enc_channels[2],   # 384
#             skip_channels=self.enc_channels[1],  # 192
#             out_channels=self.enc_channels[1],   # 192
#             num_heads=num_heads,
#             depth=depth,
#             upscale_factor=2
#         )
        
#         self.up1 = TransformerUpBlock(
#             in_channels=self.enc_channels[1],   # 192
#             skip_channels=self.enc_channels[0],  # 96
#             out_channels=self.enc_channels[0],   # 96
#             num_heads=num_heads,
#             depth=depth,
#             upscale_factor=2
#         )
        
#         # Final segmentation head
#         self.head = nn.Sequential(
#             nn.Linear(self.enc_channels[0], self.enc_channels[0]),
#             nn.GELU(),
#             nn.Linear(self.enc_channels[0], num_classes)
#         )
        
#     def forward(self, x):
#         """
#         x: (B, C, T, H, W)
#         """
#         skips = self.encoder(x)
#         s1, s2, s3, s4 = skips
        
#         # Convert (B, T, H, W, C) -> (B, C, T, H, W)
#         def permute_fmap(fmap):
#             return fmap.permute(0, 4, 1, 2, 3).contiguous()
        
#         s1, s2, s3, s4 = map(permute_fmap, (s1, s2, s3, s4))
        
#         # Temporal max pooling (like your patch model - simple and effective)
#         s1 = s1.max(dim=2)[0]  # (B, 96, H, W)
#         s2 = s2.max(dim=2)[0]  # (B, 192, H, W)
#         s3 = s3.max(dim=2)[0]  # (B, 384, H, W)
#         s4 = s4.max(dim=2)[0]  # (B, 768, H, W)
        
#         # Convert to (B, H, W, C) for transformer processing
#         s1 = s1.permute(0, 2, 3, 1)  # (B, H, W, 96)
#         s2 = s2.permute(0, 2, 3, 1)  # (B, H, W, 192)
#         s3 = s3.permute(0, 2, 3, 1)  # (B, H, W, 384)
#         s4 = s4.permute(0, 2, 3, 1)  # (B, H, W, 768)
        
#         # Decoder path with transformer blocks
#         x = self.up3(s4, s3)  # (B, H*2, W*2, 384)
#         x = self.up2(x, s2)   # (B, H*4, W*4, 192)
#         x = self.up1(x, s1)   # (B, H*8, W*8, 96)
        
#         # Final segmentation
#         B, H, W, C = x.shape
#         x = x.view(B, H * W, C)
#         logits = self.head(x)  # (B, H*W, num_classes)
#         logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, num_classes, H, W)
        
#         return logits


# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, 
#                  freeze=False, num_heads=8, depth=2):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary', smooth=0.15, ignore_index=self.IGNORE_INDEX)
#         self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25, ignore_index=self.IGNORE_INDEX)
#         self.loss_func = lambda x, y: 0.6 * self.loss_func1(x, y) + 0.4 * self.loss_func2(x, y)

#         self.backbone = SwinTransformerUNet(num_classes=1, num_heads=num_heads, depth=depth)
#         self.normalization = nn.BatchNorm3d(num_features=1)
    
#     def forward(self, x):
#         x = x.permute(0, 2, 1, 3, 4)  # Rearrange to (B, C, T, H, W)
#         # x = self.normalization(x)
#         output = self.backbone(x)  # (B, 1, H, W)
#         return output
    
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# import numpy as np
# import segmentation_models_pytorch as smp
# from torchvision.models.video import swin_transformer


# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt=None, in_chans=1):
#         super().__init__()
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # Adapt patch_embed for 1 channel
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())
#         pretrained_ckpt = 'pretraining/checkpoints/tiny_epoch=72.ckpt'
#         # optionally load ckpt
#         if pretrained_ckpt:
#             ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#             ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#             model_state_dict = self.backbone.state_dict()

#             new_state_dict = {}
#             used_ckpt_keys = set()  # keep track of keys already used


#             for k_model in model_state_dict.keys():
#                 # Skip decoder stuff
#                 if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#                     continue
#                 print(k_model)
#                 # Find corresponding key in checkpoint
#                 if k_model in ckpt_state_dict:
#                     new_state_dict[k_model] = ckpt_state_dict[k_model]
#                 else:
#                     # Fallback: find first unused key with matching shape
#                     for k_ckpt, v_ckpt in ckpt_state_dict.items():
#                         if k_ckpt in used_ckpt_keys:
#                             continue  # skip already used keys
#                         if v_ckpt.shape == model_state_dict[k_model].shape:
#                             new_state_dict[k_model] = v_ckpt
#                             used_ckpt_keys.add(k_ckpt)
#                             print(f"Fallback match: {k_model} <- {k_ckpt}")
#                             break

#         missing, unexpected =  self.backbone.load_state_dict(new_state_dict, strict=False)
#         print("Missing keys:", missing)
#         print("Unexpected keys:", unexpected)
#         # self.backbone = backbone
#         # self.backbone.head = nn.Identity()
#         self.backbone.head = nn.Identity()

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):
#                 skips.append(x)

#         return skips


# class TransformerUpBlock(nn.Module):
#     """Transformer-based upsampling block similar to your patch model"""
#     def __init__(self, in_channels, skip_channels, out_channels, 
#                  num_heads=8, depth=2, upscale_factor=2):
#         super().__init__()
#         self.upscale_factor = upscale_factor
#         self.out_channels = out_channels
        
#         # Project skip connection to match working dimension
#         self.skip_proj = nn.Linear(skip_channels, in_channels) if skip_channels != in_channels else nn.Identity()
        
#         # Project input if needed
#         self.in_proj = nn.Linear(in_channels, in_channels) if in_channels != in_channels else nn.Identity()
        
#         # Transformer decoder (like your patch model)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=in_channels,
#             nhead=num_heads,
#             dim_feedforward=in_channels * 2,
#             dropout=0.1,
#             activation="gelu",
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
#         # Pixel shuffle upsampling (like your patch model)
#         self.classifier = nn.Linear(in_channels, out_channels * (upscale_factor ** 2))
        
#     def forward(self, x, skip):
#         """
#         x: (B, H_x, W_x, C_in) - from previous decoder stage
#         skip: (B, H_s, W_s, C_skip) - from encoder
#         """
#         B, H_skip, W_skip, C_skip = skip.shape
        
#         # Upsample x to match skip spatial dimensions using bilinear interpolation
#         if x.shape[1:3] != skip.shape[1:3]:
#             # (B, H, W, C) -> (B, C, H, W) for interpolate
#             x_up = x.permute(0, 3, 1, 2)
#             x_up = F.interpolate(x_up, size=(H_skip, W_skip), mode='bilinear', align_corners=False)
#             x = x_up.permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        
#         # Project skip connection
#         skip_proj = self.skip_proj(skip)  # (B, H, W, C_in)
        
#         # Combine: concatenate then add
#         combined = x + skip_proj  # (B, H, W, C_in)
        
#         # Flatten spatial dimensions for transformer
#         B, H, W, C = combined.shape
#         tokens = combined.view(B, H * W, C)  # (B, N, C)
        
#         # Apply transformer
#         transformed = self.transformer(tokens)  # (B, N, C)
        
#         # Pixel shuffle upsampling
#         logits = self.classifier(transformed)  # (B, N, out_channels * upscale^2)
#         logits = logits.permute(0, 2, 1).view(B, self.out_channels * (self.upscale_factor ** 2), H, W)
#         output = F.pixel_shuffle(logits, upscale_factor=self.upscale_factor)
        
#         # Convert back to (B, H', W', C) format
#         output = output.permute(0, 2, 3, 1)  # (B, H', W', out_channels)
        
#         return output


# class TransformerSegmentationHead(nn.Module):
#     """Transformer-based segmentation head for final prediction"""
#     def __init__(self, in_channels, num_classes, num_heads=8, depth=2, hidden_dim=None):
#         super().__init__()
        
#         if hidden_dim is None:
#             hidden_dim = in_channels
        
#         # Input projection
#         self.input_proj = nn.Linear(in_channels, hidden_dim)
        
#         # Learnable positional embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
#         # Transformer encoder layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,
#             dropout=0.1,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True  # Pre-norm for better stability
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
#         # Output projection with residual connection
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.output_proj = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim // 2, num_classes)
#         )
        
#     def forward(self, x):
#         """
#         x: (B, H, W, C) or (B, H*W, C)
#         Returns: (B, num_classes, H, W)
#         """
#         if x.dim() == 4:
#             B, H, W, C = x.shape
#             x = x.view(B, H * W, C)
#         else:
#             B, N, C = x.shape
#             H = W = int(N ** 0.5)  # Assume square
        
#         # Project to hidden dimension
#         x = self.input_proj(x)  # (B, N, hidden_dim)
        
#         # Add positional embedding
#         x = x + self.pos_embed
        
#         # Apply transformer
#         x = self.transformer(x)  # (B, N, hidden_dim)
        
#         # Normalize and project to output classes
#         x = self.norm(x)
#         logits = self.output_proj(x)  # (B, N, num_classes)
        
#         # Reshape to spatial format
#         logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, num_classes, H, W)
        
#         return logits


# class SwinTransformerUNet(nn.Module):
#     def __init__(self, num_classes=1, num_heads=8, depth=2, head_depth=3):
#         super().__init__()
        
#         self.encoder = Swin3DEncoder()
        
#         # Swin3D-tiny channels: [96, 192, 384, 768]
#         self.enc_channels = [96, 192, 384, 768]
        
#         # Temporal max pooling for each skip (like your patch model)
#         # No learnable parameters - just max pool
        
#         # Decoder with transformer blocks at each level
#         # Going from 768 -> 384 -> 192 -> 96
#         self.up3 = TransformerUpBlock(
#             in_channels=self.enc_channels[3],   # 768
#             skip_channels=self.enc_channels[2],  # 384
#             out_channels=self.enc_channels[2],   # 384
#             num_heads=num_heads,
#             depth=depth,
#             upscale_factor=2
#         )
        
#         self.up2 = TransformerUpBlock(
#             in_channels=self.enc_channels[2],   # 384
#             skip_channels=self.enc_channels[1],  # 192
#             out_channels=self.enc_channels[1],   # 192
#             num_heads=num_heads,
#             depth=depth,
#             upscale_factor=2
#         )
        
#         self.up1 = TransformerUpBlock(
#             in_channels=self.enc_channels[1],   # 192
#             skip_channels=self.enc_channels[0],  # 96
#             out_channels=self.enc_channels[0],   # 96
#             num_heads=num_heads,
#             depth=depth,
#             upscale_factor=2
#         )
        
#         # Transformer-based segmentation head (replacing simple linear layers)
#         self.head = TransformerSegmentationHead(
#             in_channels=self.enc_channels[0],  # 96
#             num_classes=num_classes,
#             num_heads=num_heads,
#             depth=head_depth,  # Can be different from decoder depth
#             hidden_dim=self.enc_channels[0]  # Or use a different dimension
#         )
        
#     def forward(self, x):
#         """
#         x: (B, C, T, H, W)
#         """
#         skips = self.encoder(x)
#         s1, s2, s3, s4 = skips
        
#         # Convert (B, T, H, W, C) -> (B, C, T, H, W)
#         def permute_fmap(fmap):
#             return fmap.permute(0, 4, 1, 2, 3).contiguous()
        
#         s1, s2, s3, s4 = map(permute_fmap, (s1, s2, s3, s4))
        
#         # Temporal max pooling (like your patch model - simple and effective)
#         s1 = s1.max(dim=2)[0]  # (B, 96, H, W)
#         s2 = s2.max(dim=2)[0]  # (B, 192, H, W)
#         s3 = s3.max(dim=2)[0]  # (B, 384, H, W)
#         s4 = s4.max(dim=2)[0]  # (B, 768, H, W)
        
#         # Convert to (B, H, W, C) for transformer processing
#         s1 = s1.permute(0, 2, 3, 1)  # (B, H, W, 96)
#         s2 = s2.permute(0, 2, 3, 1)  # (B, H, W, 192)
#         s3 = s3.permute(0, 2, 3, 1)  # (B, H, W, 384)
#         s4 = s4.permute(0, 2, 3, 1)  # (B, H, W, 768)
        
#         # Decoder path with transformer blocks
#         x = self.up3(s4, s3)  # (B, H*2, W*2, 384)
#         x = self.up2(x, s2)   # (B, H*4, W*4, 192)
#         x = self.up1(x, s1)   # (B, H*8, W*8, 96)
        
#         # Transformer-based segmentation head
#         logits = self.head(x)  # (B, num_classes, H, W)
        
#         return logits


# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, 
#                  freeze=False, num_heads=8, depth=2, head_depth=3):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary', smooth=0.15, ignore_index=self.IGNORE_INDEX)
#         self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25, ignore_index=self.IGNORE_INDEX)
#         self.loss_func = lambda x, y: 0.6 * self.loss_func1(x, y) + 0.4 * self.loss_func2(x, y)

#         self.backbone = SwinTransformerUNet(
#             num_classes=1, 
#             num_heads=num_heads, 
#             depth=depth,
#             head_depth=head_depth
#         )
#         self.normalization = nn.BatchNorm3d(num_features=1)
    
#     def forward(self, x):
#         x = x.permute(0, 2, 1, 3, 4)  # Rearrange to (B, C, T, H, W)
#         # x = self.normalization(x)
#         output = self.backbone(x)  # (B, 1, H, W)
#         return output
# # ---------------- Full UNet ---------------- #
# class SwinUNet3D(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.encoder = Swin3DEncoder()

#         # Channels per stage (tiny config)
#         self.enc_channels = [96, 192, 384, 768]

#         # Bottleneck conv to reshape final feature
#         self.bottleneck = nn.Conv3d(self.enc_channels[-1], 768, kernel_size=1)

#         # Decoder (reverse order)
#         self.up3 = UpBlock3D(768, 384)
#         self.up2 = UpBlock3D(384, 192)
#         self.up1 = UpBlock3D(192, 96)

#         # Final segmentation head
#         self.head = nn.Conv3d(96, num_classes, kernel_size=1)

#     def forward(self, x):
#         skips = self.encoder(x)   # list of [stage1, stage2, stage3, stage4]
#         s1, s2, s3, s4 = skips

#         # Swin returns (B, T, H, W, C) → convert to (B, C, T, H, W)
#         def permute_fmap(fmap):
#             return fmap.permute(0, 4, 1, 2, 3).contiguous()

#         s1, s2, s3, s4 = map(permute_fmap, (s1, s2, s3, s4))

#         x = self.bottleneck(s4)
#         print(x.shape)

#         x = self.up3(x, s3)
#         print(x.shape)
#         x = self.up2(x, s2)
#         print(x.shape)
#         x = self.up1(x, s1)
#         print(x.shape)
#         out = self.head(x)
#         return out


# # GOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
# class Patch3DTransformerSegmentation(nn.Module):
#     def __init__(self, num_classes=1, embed_dim=1024, num_heads=4, depth=2, patch_output=4):
#         super().__init__()
#         self.patch_output = patch_output
#         self.num_classes = num_classes
        
#         # backbone = swin_transformer.swin3d_b(weights=None)
#         backbone = swin_transformer.swin3d_b(weights=None)


#         # Modify first conv layer to accept 1 channel instead of 3
#         old_proj = backbone.patch_embed.proj
#         backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
        
#         backbone = nn.Sequential(*list(backbone.children())[:-3]) 

#         # Load checkpoint
#         ckpt_path = "pretraining/checkpoints/epochepoch=20.ckpt"
#         ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)  # CPU first, move to GPU later if needed

#         # Check if it's a Lightning checkpoint
#         if 'state_dict' in ckpt:
#             state_dict = ckpt['state_dict']
#         else:
#             state_dict = ckpt
#                 # # Filter only encoder weights (keys start with 'encoder.')
#         encoder_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('encoder.'):
#                 # remove 'encoder.' prefix to match backbone keys
#                 encoder_state_dict[k.replace('encoder.', '')] = v
#         # new_state_dict = {}
#         # # for k, v in encoder_state_dict.items():
#         # #     new_key = k.replace("blocks", "features").replace("patch_embed", "0")  # example
#         # #     new_state_dict[new_key] = v
#         missing, unexpected =  backbone.load_state_dict(encoder_state_dict, strict=False)
#         print("Missing keys:", missing)
#         print("Unexpected keys:", unexpected)
#         self.backbone = backbone
#         # Freeze the backbone
#         # for param in self.backbone.parameters():
#         #     param.requires_grad = False
        

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim,
#             dropout=0.1,
#             activation="gelu",
#             batch_first=True  # (B, N, C)
#         )
#         self.embed_dim = embed_dim
#         self.pos_embedding = None  # will be initialized dynamically after seeing Hf, Wf
#         self.pos_embedding_x = None
#         self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # Classifier: per patch -> small 3D patch
#         self.classifier = nn.Linear(embed_dim, patch_output ** 2)
#         # self.proj = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)  # learns spatial filters
#         self.head = nn.Sequential(
#             nn.Conv2d(embed_dim,self.patch_output**2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.patch_output**2,self.patch_output**2, kernel_size=1),
#         )


#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         feats = self.backbone(x)  # (B, T', H', W',embed_dim)

#         # Adaptive temporal pooling
#         feats = feats.mean(dim=1) # (B, 1, Hf, Wf, embed_dim,)
        
#         B, Hf, Wf, D = feats.shape

#         # # Move embed_dim to dim 1 and flatten patches
#         patch_tokens = feats.permute(0, 3, 1, 2).contiguous()  # (B, 768, 8, 7, 7)
#         patch_tokens = patch_tokens.view(B, Hf*Wf, -1)        # (B, 768, 8*7*7=392)
#         # patch_tokens = patch_tokens.permute(1, 0, 2)               # (392, B, 768)
#         # if self.pos_embedding is None or self.pos_embedding.shape[1] != Hf*Wf:
#         #     self.pos_embedding = nn.Parameter(torch.zeros(1, Hf*Wf, self.embed_dim, device=x.device))
#         #     nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            
#         # Add positional embeddings
#         # patch_tokens = patch_tokens #+ self.pos_embedding  # (B, N, C)
#         # patch_tokens = self.input_proj(patch_tokens)  # (B, N, 16)
#         # Transformer
#         transformed_tokens = self.decoder(patch_tokens) 
#         transformed_tokens = transformed_tokens.permute(0, 2, 1).view(B, self.embed_dim, Hf, Wf)  # (B, C, Hf, Wf)

#         # if self.pos_embedding_x is None:
#         #     self.pos_embedding_x = nn.Parameter(torch.zeros(1,  self.embed_dim,  Hf, Wf, device=x.device))
#         #     nn.init.trunc_normal_(self.pos_embedding_x, std=0.02)
            
#         # transformed_tokens = transformed_tokens + self.pos_embedding_x  # (B, C, Hf, Wf)
#         # print(transformed_tsokens.shape)
#         transformed_tokens = transformed_tokens.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, -1)  # (B, N, C)
#         logits = self.classifier(transformed_tokens)
#         # logits = logits.permute(0, 2, 1).view(B, self.patch_output**2, Hf, Wf)  # (B, patch_output^2, Hf, Wf)
        
#         # # logits = self.head(transformed_tokens)   # (B, num_classes, Hf, Wf)
#         logits = F.pixel_shuffle(logits, upscale_factor=4) 
#         return logits
    
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
#         self.backbone = Patch3DTransformerSegmentation(num_classes=1, patch_output=4)
#         self.scale_factor = 8
    
#     def forward(self, x):

#         x = x.permute(0,2,1,3,4)
#         output = self.backbone(x) 
#         return output


# GOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD  
class SwinModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
        super(SwinModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')


        # Modify first conv layer to accept 1 channel instead of 3
        old_proj = backbone.patch_embed.proj
        new_proj = nn.Conv3d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )
        # Initialize weights by summing across RGB channels
        with torch.no_grad():
            # old_proj.weight shape: [out_channels, 3, kT, kH, kW]
            summed = old_proj.weight.sum(dim=1, keepdim=True)  # -> [out_channels, 1, kT, kH, kW]
            new_proj.weight.copy_(summed)

            # If bias exists, copy it too
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
            # embed_dim = out.shape[1]  # channel dimension


        # Replace the old conv with the new one
        backbone.patch_embed.proj = new_proj
        embed_dim = 768
        self.backbone = backbone
        self.backbone.head = nn.Identity()


        self.classifier = nn.Sequential(
                nn.Linear(embed_dim,(self.hparams.size//32)**2),
        )
    
    def forward(self, x):

        x = x.permute(0,2,1,3,4)
        preds = self.backbone(x)  # runs backbone, sets self.feature
        preds = self.classifier(preds)
        preds = preds.view(-1,1,self.hparams.size//32,self.hparams.size//32)
        return preds


# ## GOOOOOOOOOOOOOOOOOODDDDDDDDDDDDD
# class SwinModel(pl.LightningModule):

#     def __init__(self, pred_shape, size, lr, scheduler=None,wandb_logger=None,freeze=False):
#         super(SwinModel, self).__init__()
#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.25,ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
#         # model = swin_transformer.swin3d_t(weights="KINETICS400_V1")#KINETICS400_IMAGENET22K_V1 
        
        
#         # self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
#         # self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # Modify first conv layer to accept 1 channel instead of 3
#         old_proj = backbone.patch_embed.proj
#         new_proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
#         # Initialize weights by summing across RGB channels
#         with torch.no_grad():
#             # old_proj.weight shape: [out_channels, 3, kT, kH, kW]
#             summed = old_proj.weight.sum(dim=1, keepdim=True)   # -> [out_channels, 1, kT, kH, kW]
#             new_proj.weight.copy_(summed)

#             # If bias exists, copy it too
#             if old_proj.bias is not None:
#                 new_proj.bias.copy_(old_proj.bias)

#         # Replace the old conv with the new one
#         backbone.patch_embed.proj = new_proj
#         self.backbone = backbone
        
#         # backbone = nn.Sequential(*list(backbone.children())[:-2]) 

#         # self.backbone = swin_transformer.swin3d_t(weights="KINETICS400_V1") #KINETICS400_IMAGENET22K_V1 
#         self.backbone.head = nn.Identity()
#         # self.decoder = Decoder2D(in_channels=768, num_classes=1)
#         # self.decoder = Decoder2D(in_channels=512, num_classes=1)
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(384, 1),  
#         # )
#         self.classifier = nn.Sequential(
#             nn.Linear(768, 1),  
#         )

#         self.features = None
#         # self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)
#         # self.hook_handle = self.backbone.features[-1].register_forward_hook(self._hook_fn)
#         self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)
#         # self.post_norm = nn.LayerNorm(normalized_shape=(768,), eps=1e-05, elementwise_affine=True)
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))  # collapse T → 1


#         # self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)

#     def _hook_fn(self, module, input, output):
#         self.features = output

#     def forward(self, x):
#         x = x.permute(0,2,1,3,4)
#         _ = self.backbone(x)  # runs backbone, sets self.features
#         feat = self.features  # (B, T_patch, H_patch, W_patch, C)
#         # feat = self.post_norm(feat) 
#         feat = feat.permute(0, 4, 1, 2, 3)  # (B, C, T_patch, H_patch, W_patch)
#         # feat_2d = self.pool(feat).squeeze()
#         feat_2d = feat.max(dim=2)[0]  # average temporal patches: (B, C, H_patch, W_patch)
#         # print(feat_2d.shape)
#         seg_logits  = self.classifier(feat_2d.permute(0, 2, 3, 1)).view(-1,1,self.hparams.size//32,self.hparams.size//32)
#         # print(seg_logits.shape)
#         # seg_logits = self.decoder(feat_2d)  # (B, num_classes, 224, 224)
#         return seg_logits

# GOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD  
class SwinModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
        super(SwinModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        backbone = swin_transformer.swin3d_t(weights=None)#(weights="KINETICS400_V1")

        embed_dim = backbone.norm.normalized_shape[0]  # usually 1024 for swin3d_b

        # Modify first conv layer to accept 1 channel instead of 3
        old_proj = backbone.patch_embed.proj
        backbone.patch_embed.proj = nn.Conv3d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )
        
        # backbone = nn.Sequential(*list(backbone.children())[:-2]) 
        

        # # Load checkpoint
        # ckpt_path = "pretraining/checkpoints/epochepoch=93.ckpt"
        # ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)  # CPU first, move to GPU later if needed

        # # Check if it's a Lightning checkpoint
        # if 'state_dict' in ckpt:
        #     state_dict = ckpt['state_dict']
        # else:
        #     state_dict = ckpt
        #         # # Filter only encoder weights (keys start with 'encoder.')
        # encoder_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('encoder.'):
        #         # remove 'encoder.' prefix to match backbone keys
        #         encoder_state_dict[k.replace('encoder.', '')] = v
        # new_state_dict = {}
        # # for k, v in encoder_state_dict.items():
        # #     new_key = k.replace("blocks", "features").replace("patch_embed", "0")  # example
        # #     new_state_dict[new_key] = v
        # missing, unexpected =  backbone.load_state_dict(encoder_state_dict, strict=False)
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)
        backbone.head = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
                nn.Linear(embed_dim,(self.hparams.size//16)**2),
        )
    
    def forward(self, x):

        x = x.permute(0,2,1,3,4)
        preds = self.backbone(x)  # runs backbone, sets self.feature
        preds = self.classifier(preds)
        preds = preds.view(-1,1,self.hparams.size//16,self.hparams.size//16)
        return preds

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=15.ckpt', in_chans=1):
#         super().__init__()
#         # Load Swin3D
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # remove classifier head, keep norm
#         self.backbone.head = nn.Identity()
        
#         # for param in self.backbone.parameters():
#         #     param.requires_grad = False

#         # optionally load ckpt
#         if pretrained_ckpt:
#             ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#             ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#             model_state_dict = self.backbone.state_dict()

#             new_state_dict = {}
#             used_ckpt_keys = set()  # keep track of keys already used


#             for k_model in model_state_dict.keys():
#                 # Skip decoder stuff
#                 if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#                     continue
#                 print(k_model)
#                 # Find corresponding key in checkpoint
#                 if k_model in ckpt_state_dict:
#                     new_state_dict[k_model] = ckpt_state_dict[k_model]
#                 else:
#                     # Fallback: find first unused key with matching shape
#                     for k_ckpt, v_ckpt in ckpt_state_dict.items():
#                         if k_ckpt in used_ckpt_keys:
#                             continue  # skip already used keys
#                         if v_ckpt.shape == model_state_dict[k_model].shape:
#                             new_state_dict[k_model] = v_ckpt
#                             used_ckpt_keys.add(k_ckpt)
#                             print(f"Fallback match: {k_model} <- {k_ckpt}")
#                             break
                            
#             # Load into model
#             msg = self.backbone.load_state_dict(new_state_dict, strict=False)
#             print("Loaded:", msg)

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips  # [s1, s2, s3, s4]

# # patch
# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt='pretraining/checkpoints/tiny_epoch=70.ckpt', in_chans=1):
#         super().__init__()
#         # Load Swin3D
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # remove classifier head, keep norm
#         self.backbone.head = nn.Identity()

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips
    
          
# from einops import rearrange
# # ---------------- 2D PatchExpansion ---------------- #
# class PatchExpansion2D(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim_in = dim
#         self.expand = nn.Linear(dim, dim * 4, bias=False)  # expand for 2x2
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         # x: (B, H, W, C)
#         B, H, W, C = x.shape
#         x = self.expand(x)  # (B, H, W, 4*C)
#         x = x.view(B, H, W, 2, 2, C)  # split channels into 2×2 upsampling
#         x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H, 2, W, 2, C)
#         x = x.view(B, H * 2, W * 2, C)  # (B, H*2, W*2, C)
#         x = self.norm(x)
#         return x


# # ---------------- PatchExpansion2D Decoder Block ---------------- #
# class UpBlockPatch2D(nn.Module):
#     def __init__(self, in_ch, out_ch, skip_ch):
#         super().__init__()
#         self.dim_in = in_ch
#         self.dim_out = out_ch
#         self.expand = PatchExpansion2D(in_ch)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x, skip):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape

#         # (B, H, W, C) → expand → (B, H*2, W*2, C)
#         x_reshaped = x.permute(0, 2, 3, 1).contiguous()
#         x_ups = self.expand(x_reshaped)
#         # back to (B, C, H, W)
#         x_ups = x_ups.permute(0, 3, 1, 2).contiguous()

#         # match skip size if different
#         if x_ups.shape[2:] != skip.shape[2:]:
#             diff = [s - d for s, d in zip(skip.shape[2:], x_ups.shape[2:])]
#             x_ups = F.pad(x_ups, [0, diff[1], 0, diff[0]])

#         # concatenate skip connection
#         x_cat = torch.cat([x_ups, skip], dim=1)
#         return self.conv(x_cat)

    
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15,ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.6 * self.loss_func1(x,y) +0.4*self.loss_func2(x,y)

#         self.backbone = Swin3DEncoder()
#         self.enc_channels = [96, 192, 384, 768]
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))

#         # Decoder (2D)
#         self.up3 = UpBlockPatch2D(self.enc_channels[-1],self.enc_channels[-2], self.enc_channels[-2])
#         self.up2 = UpBlockPatch2D(self.enc_channels[-2],self.enc_channels[-3], self.enc_channels[-3])
#         self.up1 = UpBlockPatch2D(self.enc_channels[-3],self.enc_channels[-4], self.enc_channels[-4])

#         # Final head
#         self.head = nn.Conv2d(self.enc_channels[0], 1, kernel_size=1)

#     def forward(self, x):
#         # get swin feature maps [s1, s2, s3, s4]
#         x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) → (B, T, C, H, W)
#         feat_maps = self.backbone(x)
#         feat_maps = [f.permute(0, 4, 1, 2, 3).contiguous() for f in feat_maps]  # → (B, C, T, H, W)

        
#         # temporal aggregation
#         # s1, s2, s3, s4 = [torch.max(f, dim=2)[0] for f in feat_maps]  # → (B, C, H, W)
#         s1, s2, s3, s4 = [self.pool(f).squeeze() for f in feat_maps]
#         # print(s1.shape,s2.shape,s3.shape,s4.shape)


#         x = self.up3(s4, s3)
#         x = self.up2(x, s2)
#         x = self.up1(x, s1)

#         out = self.head(x)  # (B, 1, H, W)
#         return out
    

# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# import numpy as np
# import segmentation_models_pytorch as smp
# from torchvision.models.video import swin_transformer


# class PatchExpanding3D(nn.Module):
#     """3D Patch Expanding Layer (inverse of patch merging)"""
#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2 * dim, bias=False)
#         self.norm = norm_layer(dim // 2)

#     def forward(self, x):
#         """
#         x: B, T, H, W, C
#         """
#         B, T, H, W, C = x.shape
#         x = self.expand(x)  # B, T, H, W, 2*C
        
#         # Rearrange to upsample spatially
#         x = x.view(B, T, H, W, 2, 2, C // 2)
#         x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # B, T, H, 2, W, 2, C//2
#         x = x.view(B, T, H * 2, W * 2, C // 2)
#         x = self.norm(x)
#         return x


# class SwinUNet3D(nn.Module):
#     """Swin-Unet architecture adapted for 3D input"""
#     def __init__(self, num_classes=1, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], patch_output=4):
#         super().__init__()
#         self.num_classes = num_classes
#         self.patch_output = patch_output
        
#         # ===== ENCODER (Swin Transformer) =====
#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')
        
#         # Modify first conv layer to accept 1 channel
#         old_proj = backbone.patch_embed.proj
#         new_proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
#         with torch.no_grad():
#             summed = old_proj.weight.sum(dim=1, keepdim=True)
#             new_proj.weight.copy_(summed)
#             if old_proj.bias is not None:
#                 new_proj.bias.copy_(old_proj.bias)
        
#         backbone.patch_embed.proj = new_proj
#         backbone.head = nn.Identity()
#         backbone.norm = nn.Identity()
        
#         self.patch_embed = backbone.patch_embed
#         self.encoder_layers = backbone.layers  # 4 stages
        
#         # Feature dimensions: [96, 192, 384, 768]
#         self.dims = [96, 192, 384, 768]
        
#         # ===== BOTTLENECK =====
#         # Additional Swin block at bottleneck (optional, can reuse last encoder layer)
#         self.bottleneck = self.encoder_layers[3]  # Reuse the last encoder stage
        
#         # ===== DECODER (Swin Transformer blocks with Patch Expanding) =====
#         # Decoder stage 1: 768 -> 384
#         self.decoder1_expand = PatchExpanding3D(dim=self.dims[3])
#         self.decoder1_swin = self._make_decoder_layer(self.dims[2], depth=depths[2], num_heads=num_heads[2])
        
#         # Decoder stage 2: 384 -> 192
#         self.decoder2_expand = PatchExpanding3D(dim=self.dims[2])
#         self.decoder2_swin = self._make_decoder_layer(self.dims[1], depth=depths[1], num_heads=num_heads[1])
        
#         # Decoder stage 3: 192 -> 96
#         self.decoder3_expand = PatchExpanding3D(dim=self.dims[1])
#         self.decoder3_swin = self._make_decoder_layer(self.dims[0], depth=depths[0], num_heads=num_heads[0])
        
#         # Final expansion and patch expanding to original resolution
#         self.final_expand = PatchExpanding3D(dim=self.dims[0])
        
#         # ===== SEGMENTATION HEAD =====
#         # After temporal pooling, convert to 2D
#         final_dim = self.dims[0] // 2  # 48 after final expansion
#         self.segmentation_head = nn.Sequential(
#             nn.Conv2d(final_dim, final_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(final_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(final_dim, num_classes, kernel_size=1)
#         )
        
#     def _make_decoder_layer(self, dim, depth, num_heads):
#         """Create a decoder stage with Swin Transformer blocks"""
#         # Simplified: just use conv blocks since we don't have 3D Swin blocks readily available
#         # In the paper, these would be Swin Transformer blocks
#         return nn.Sequential(
#             nn.Conv3d(dim, dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dim, dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(dim),
#             nn.ReLU(inplace=True),
#         )
    
#     def forward(self, x):
#         B, C, T, H, W = x.shape
        
#         # ===== ENCODER =====
#         skip_connections = []
        
#         # Patch embedding
#         x = self.patch_embed(x)  # B, C, T', H', W'
#         x = x.permute(0, 2, 3, 4, 1)  # B, T', H', W', C (Swin format)
#         skip_connections.append(x)
        
#         # Encoder stage 1
#         x = self.encoder_layers[0](x)
#         skip_connections.append(x)
        
#         # Encoder stage 2
#         x = self.encoder_layers[1](x)
#         skip_connections.append(x)
        
#         # Encoder stage 3
#         x = self.encoder_layers[2](x)
#         skip_connections.append(x)
        
#         # ===== BOTTLENECK =====
#         x = self.encoder_layers[3](x)  # B, T', H', W', 768
        
#         # ===== DECODER WITH SKIP CONNECTIONS =====
#         # Decoder stage 1
#         x = self.decoder1_expand(x)  # Upsample
#         x = x + skip_connections[3]  # Skip connection from encoder stage 3
#         x = x.permute(0, 4, 1, 2, 3)  # B, C, T', H', W' for conv
#         x = self.decoder1_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)  # Back to B, T', H', W', C
        
#         # Decoder stage 2
#         x = self.decoder2_expand(x)
#         x = x + skip_connections[2]  # Skip connection from encoder stage 2
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.decoder2_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)
        
#         # Decoder stage 3
#         x = self.decoder3_expand(x)
#         x = x + skip_connections[1]  # Skip connection from encoder stage 1
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.decoder3_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)
        
#         # Final expansion
#         x = self.final_expand(x)  # B, T', H', W', 48
        
#         # ===== 3D TO 2D: TEMPORAL POOLING =====
#         x = x.permute(0, 4, 1, 2, 3)  # B, C, T', H', W'
#         x = x.mean(dim=2)  # B, C, H', W' - average across temporal dimension
        
#         # ===== SEGMENTATION HEAD =====
#         output = self.segmentation_head(x)  # B, num_classes, H', W'
        
#         return output


# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()
#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary', ignore_index=self.IGNORE_INDEX)
#         self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15, ignore_index=self.IGNORE_INDEX)
#         self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)
        
#         # Swin-Unet with patch expanding layers
#         self.backbone = SwinUNet3D(
# #             num_classes=1,
# #             embed_dim=96,
# #             depths=[2, 2, 2, 2],  # Depth of each stage
# #             num_heads=[3, 6, 12, 24]  # Number of attention heads per stage
# #         )
        
# #         if freeze:
# #             self._freeze_encoder()
    
# #     def _freeze_encoder(self):
# #         """Freeze encoder parameters"""
# #         for param in self.backbone.encoder_layers.parameters():
# #             param.requires_grad = False
    
# #     def forward(self, x):
# #         # x shape: [B, C, T, H, W] where C=1 (single channel)
# #         output = self.backbone(x)
# #         return output






# class PatchExpanding3D(nn.Module):
#     """3D Patch Expanding Layer (inverse of patch merging)"""
#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2 * dim, bias=False)
#         self.norm = norm_layer(dim // 2)

#     def forward(self, x):
#         """
#         x: B, T, H, W, C
#         """
#         B, T, H, W, C = x.shape
#         x = self.expand(x)  # B, T, H, W, 2*C
        
#         # Rearrange to upsample spatially
#         x = x.view(B, T, H, W, 2, 2, C // 2)
#         x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # B, T, H, 2, W, 2, C//2
#         x = x.view(B, T, H * 2, W * 2, C // 2)
#         x = self.norm(x)
#         return x


# class SwinUNet3D(nn.Module):
#     """Swin-Unet architecture adapted for 3D input"""
#     def __init__(self, num_classes=1, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], patch_output=4):
#         super().__init__()
#         self.num_classes = num_classes
#         self.patch_output = patch_output
        
#         # ===== ENCODER (Swin Transformer) =====
#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')
        
#         # Modify first conv layer to accept 1 channel
#         old_proj = backbone.patch_embed.proj
#         new_proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
#         with torch.no_grad():
#             summed = old_proj.weight.sum(dim=1, keepdim=True)
#             new_proj.weight.copy_(summed)
#             if old_proj.bias is not None:
#                 new_proj.bias.copy_(old_proj.bias)
        
#         backbone.patch_embed.proj = new_proj
#         backbone.head = nn.Identity()
#         backbone.norm = nn.Identity()
        
#         self.backbone = backbone
        
#         # Feature dimensions: [96, 192, 384, 768]
#         self.dims = [96, 192, 384, 768]
        
#         # ===== BOTTLENECK =====
#         # Additional Swin block at bottleneck (optional, can reuse last encoder layer)
#         self.bottleneck = self.encoder_layers[3]  # Reuse the last encoder stage
        
#         # ===== DECODER (Swin Transformer blocks with Patch Expanding) =====
#         # Decoder stage 1: 768 -> 384
#         self.decoder1_expand = PatchExpanding3D(dim=self.dims[3])
#         self.decoder1_swin = self._make_decoder_layer(self.dims[2], depth=depths[2], num_heads=num_heads[2])
        
#         # Decoder stage 2: 384 -> 192
#         self.decoder2_expand = PatchExpanding3D(dim=self.dims[2])
#         self.decoder2_swin = self._make_decoder_layer(self.dims[1], depth=depths[1], num_heads=num_heads[1])
        
#         # Decoder stage 3: 192 -> 96
#         self.decoder3_expand = PatchExpanding3D(dim=self.dims[1])
#         self.decoder3_swin = self._make_decoder_layer(self.dims[0], depth=depths[0], num_heads=num_heads[0])
        
#         # Final expansion and patch expanding to original resolution
#         self.final_expand = PatchExpanding3D(dim=self.dims[0])
        
#         # ===== SEGMENTATION HEAD =====
#         # After temporal pooling, convert to 2D
#         final_dim = self.dims[0] // 2  # 48 after final expansion
#         self.segmentation_head = nn.Sequential(
#             nn.Conv2d(final_dim, final_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(final_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(final_dim, num_classes, kernel_size=1)
#         )
        
#     def _make_decoder_layer(self, dim, depth, num_heads):
#         """Create a decoder stage with Swin Transformer blocks"""
#         # Simplified: just use conv blocks since we don't have 3D Swin blocks readily available
#         # In the paper, these would be Swin Transformer blocks
#         return nn.Sequential(
#             nn.Conv3d(dim, dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dim, dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(dim),
#             nn.ReLU(inplace=True),
#         )
    
#     def forward(self, x):
#         B, C, T, H, W = x.shape
        
#         # ===== ENCODER =====
#         skip_connections = []
        
#         # Patch embedding
#         x = self.backbone.patch_embed(x)  # B, T', H', W', C
#         x = self.backbone.pos_drop(x)
        
#         # Extract features through backbone.features
#         # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skip_connections.append(x)
        
#         # ===== BOTTLENECK =====
#         # Additional processing if needed (optional)
        
#         # ===== DECODER WITH SKIP CONNECTIONS =====
#         # skip_connections has 4 elements: [stage0, stage1, stage2, stage3]
#         # Decoder stage 1: fuse with stage2 (384)
#         x = self.decoder1_expand(x)  # Upsample
#         x = x + skip_connections[2]  # Skip connection from stage2 (384)
#         x = x.permute(0, 4, 1, 2, 3)  # B, C, T', H', W' for conv
#         x = self.decoder1_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)  # Back to B, T', H', W', C
        
#         # Decoder stage 2: fuse with stage1 (192)
#         x = self.decoder2_expand(x)
#         x = x + skip_connections[1]  # Skip connection from stage1 (192)
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.decoder2_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)
        
#         # Decoder stage 3: fuse with stage0 (96)
#         x = self.decoder3_expand(x)
#         x = x + skip_connections[0]  # Skip connection from stage0 (96)
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.decoder3_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)
        
#         # Final expansion
#         x = self.final_expand(x)  # B, T', H', W', 48
        
#         # ===== 3D TO 2D: TEMPORAL POOLING =====
#         x = x.permute(0, 4, 1, 2, 3)  # B, C, T', H', W'
#         x = x.mean(dim=2)  # B, C, H', W' - average across temporal dimension
        
#         # ===== SEGMENTATION HEAD =====
#         output = self.segmentation_head(x)  # B, num_classes, H', W'
        
#         return output


# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()
#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary', ignore_index=self.IGNORE_INDEX)
#         self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15, ignore_index=self.IGNORE_INDEX)
#         self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)
        
#         # Swin-Unet with patch expanding layers
#         self.backbone = SwinUNet3D(
#             num_classes=1,
#             embed_dim=96,
#             depths=[2, 2, 2, 2],  # Depth of each stage
#             num_heads=[3, 6, 12, 24]  # Number of attention heads per stage
#         )
        
#         if freeze:
#             self._freeze_encoder()
    
#     def _freeze_encoder(self):
#         """Freeze encoder parameters"""
#         for param in self.backbone.encoder_layers.parameters():
#             param.requires_grad = False


# class PatchExpanding3D(nn.Module):
#     """3D Patch Expanding Layer (inverse of patch merging)"""
#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2 * dim, bias=False)
#         self.norm = norm_layer(dim // 2)

#     def forward(self, x):
#         """
#         x: B, T, H, W, C
#         """
#         B, T, H, W, C = x.shape
#         x = self.expand(x)  # B, T, H, W, 2*C

#         # Rearrange to upsample spatially
#         x = x.view(B, T, H, W, 2, 2, C // 2)
#         x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # B, T, H, 2, W, 2, C//2
#         x = x.view(B, T, H * 2, W * 2, C // 2)
#         x = self.norm(x)
#         return x


# class Swin3DEncoder(nn.Module):
#     """Encapsulated Swin3D encoder with 1-channel input support"""
#     def __init__(self, pretrained_ckpt='pretraining/checkpoints/tiny_epoch=70.ckpt', in_chans=1):
#         super().__init__()
#         # Load Swin3D backbone
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # Remove classifier head, keep norm
#         self.backbone.head = nn.Identity()

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
#         for block in self.backbone.features:
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips  # [stage0, stage1, stage2, stage3]


# class SwinUNet3D(nn.Module):
#     """Swin-UNet architecture adapted for 3D input (using Swin3DEncoder)"""
#     def __init__(self, num_classes=1, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
#         super().__init__()
#         self.num_classes = num_classes

#         # ===== ENCODER =====
#         self.encoder = Swin3DEncoder(in_chans=1)

#         # Feature dimensions from Swin3D-tiny
#         self.dims = [96, 192, 384, 768]

#         # ===== DECODER (Patch Expanding + Simple Conv Blocks) =====
#         self.decoder1_expand = PatchExpanding3D(dim=self.dims[3])
#         self.decoder1_swin = self._make_decoder_layer(self.dims[2])

#         self.decoder2_expand = PatchExpanding3D(dim=self.dims[2])
#         self.decoder2_swin = self._make_decoder_layer(self.dims[1])

#         self.decoder3_expand = PatchExpanding3D(dim=self.dims[1])
#         self.decoder3_swin = self._make_decoder_layer(self.dims[0])

#         self.final_expand = PatchExpanding3D(dim=self.dims[0])

#         # ===== SEGMENTATION HEAD =====
#         final_dim = self.dims[0] #// 2  # 48
#         self.segmentation_head = nn.Sequential(
#             nn.Conv2d(final_dim, final_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(final_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(final_dim, num_classes, kernel_size=1)
#         )

#     def _make_decoder_layer(self, dim):
#         """Simple Conv3D-based decoder block"""
#         return nn.Sequential(
#             nn.Conv3d(dim, dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dim, dim, kernel_size=3, padding=1),
#             nn.BatchNorm3d(dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         B, C, T, H, W = x.shape

#         # ===== ENCODER =====
#         skips = self.encoder(x)  # [stage0, stage1, stage2, stage3]
#         x = skips[-1]  # bottleneck

#         # ===== DECODER =====
#         # stage3 (768) -> stage2 (384)
#         x = self.decoder1_expand(x)
#         x = x + skips[2]
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.decoder1_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)

#         # stage2 (384) -> stage1 (192)
#         x = self.decoder2_expand(x)
#         x = x + skips[1]
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.decoder2_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)

#         # stage1 (192) -> stage0 (96)
#         x = self.decoder3_expand(x)
#         x = x + skips[0]
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.decoder3_swin(x)
#         x = x.permute(0, 2, 3, 4, 1)

#         # Final upsample to original spatial scale
#         # x = self.final_expand(x)  # B, T, H', W', 48

#         # ===== 3D → 2D =====
#         x = x.permute(0, 4, 1, 2, 3)  # B, C, T, H, W
#         x = x.mean(dim=2)  # Temporal average: B, C, H, W

#         # ===== SEGMENTATION HEAD =====
#         out = self.segmentation_head(x)
#         return out


# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super().__init__()
#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         # Losses
#         self.loss_func1 = smp.losses.DiceLoss(mode='binary', ignore_index=self.IGNORE_INDEX)
#         self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15, ignore_index=self.IGNORE_INDEX)
#         self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

#         # Swin-Unet 3D backbone
#         self.backbone = SwinUNet3D(
#             num_classes=1,
#             embed_dim=96,
#             depths=[2, 2, 2, 2],
#             num_heads=[3, 6, 12, 24]
#         )

#         if freeze:
#             self._freeze_encoder()

#     def _freeze_encoder(self):
#         """Freeze encoder parameters"""
#         for param in self.backbone.encoder.parameters():
#             param.requires_grad = False

    
#     def forward(self, x):
#         # x shape: [B, C, T, H, W] where C=1 (single channel)
#         x = x.permute(0, 2, 1, 3, 4)  # Change to [B, T, C, H, W] for Swin3D
#         output = self.backbone(x)
#         return output
    

#########################GOOOOOOD

class PatchExpanding3D(nn.Module):
    """3D Patch Expanding Layer (inverse of patch merging)"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, T, H, W, C
        """
        B, T, H, W, C = x.shape
        x = self.expand(x)  # B, T, H, W, 2*C

        # Rearrange to upsample spatially
        x = x.view(B, T, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # B, T, H, 2, W, 2, C//2
        x = x.view(B, T, H * 2, W * 2, C // 2)
        x = self.norm(x)
        return x


class SwinDecoderStage(nn.Module):
    """Decoder stage using Swin Transformer blocks from encoder"""
    def __init__(self, encoder_stage):
        super().__init__()
        # Clone the encoder stage blocks
        self.blocks = nn.Sequential(*[block for block in encoder_stage])
    
    def forward(self, x):
        return self.blocks(x)


class Swin3DEncoder(nn.Module):
    """Encapsulated Swin3D encoder with 1-channel input support"""
    def __init__(self, pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=62.ckpt', in_chans=1):
        super().__init__()
        # Load Swin3D backbone
        self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')
        # self/backbone = swin_transformer.SwinTransformer3d(
        #     patch_size=[2, 4, 4],      # temporal=2, spatial=4x4 patches
        #     embed_dim=192,              # base dimension
        #     depths=[2, 2, 12],       # Tiny config
        #     num_heads=[3, 6, 12],  # heads per stage
        #     window_size=[8, 7, 7],     # attention window
        #     stochastic_depth_prob=0.1, # DropPath
        # )
        # self.backbone = swin_transformer.SwinTransformer3d(
        #     patch_size=[16, 4, 4],      # temporal=2, spatial=4x4 patches
        #     embed_dim=96,              # base dimension
        #     depths=[2, 2, 12, 2],       # Tiny config
        #     num_heads=[3, 6, 12, 24],  # heads per stage
        #     window_size=[2, 7, 7],     # attention window
        #     stochastic_depth_prob=0.1, # DropPath
        # )

        # --- patch_embed adaptation for 1 channel ---
        old_conv = self.backbone.patch_embed.proj
        weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
        bias = old_conv.bias

        self.backbone.patch_embed.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            bias=True
        )
        self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
        self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

        # Remove classifier head, keep norm
        self.backbone.head = nn.Identity()

    def forward(self, x):
        x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
        x = self.backbone.pos_drop(x)

        skips = []
        # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
        for block in self.backbone.features:
            x = block(x)
            if isinstance(block, nn.Sequential):  # stage output
                skips.append(x)

        return skips  # [stage0, stage1, stage2, stage3]


class SwinUNet3D(nn.Module):
    """Swin-UNet architecture adapted for 3D input (using Swin3D blocks in decoder)"""
    def __init__(self, num_classes=1, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.num_classes = num_classes

        # ===== ENCODER =====
        self.encoder = Swin3DEncoder(in_chans=1)

        # Feature dimensions from Swin3D-tiny
        self.dims = [96, 192, 384, 768]
        # self.dims = [128, 256, 512, 1024]

        # Extract encoder stages for reuse in decoder
        encoder_stages = []
        for block in self.encoder.backbone.features:
            if isinstance(block, nn.Sequential):
                encoder_stages.append(block)
        
        # ===== DECODER (Patch Expanding + Swin Transformer Blocks) =====
        self.decoder1_expand = PatchExpanding3D(dim=self.dims[3])
        self.decoder1_swin = SwinDecoderStage(encoder_stages[2])  # stage2 blocks

        self.decoder2_expand = PatchExpanding3D(dim=self.dims[2])
        self.decoder2_swin = SwinDecoderStage(encoder_stages[1])  # stage1 blocks

        self.decoder3_expand = PatchExpanding3D(dim=self.dims[1])
        self.decoder3_swin = SwinDecoderStage(encoder_stages[0])  # stage0 blocks

        # ===== SEGMENTATION HEAD (Transformer-based) =====
        final_dim = self.dims[1]  # 96
        self.segmentation_head = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, final_dim),
            nn.GELU(),
            nn.Linear(final_dim, num_classes)
        )


    def forward(self, x):
        B, C, T, H, W = x.shape

        # ===== ENCODER =====
        skips = self.encoder(x)  # [stage0, stage1, stage2, stage3]
        x = skips[-1]  # bottleneck

        # ===== DECODER =====
        # stage3 (768) -> stage2 (384)
        x = self.decoder1_expand(x)
        x = x + skips[2]
        x = self.decoder1_swin(x)

        # stage2 (384) -> stage1 (192)
        x = self.decoder2_expand(x)
        x = x + skips[1]
        x = self.decoder2_swin(x)

        # # stage1 (192) -> stage0 (96)
        # x = self.decoder3_expand(x)
        # x = x + skips[0]
        # x = self.decoder3_swin(x)

        # No final expand - output at 1/4 resolution
        # x shape: B, T, H/4, W/4, 96

        # ===== 3D → 2D =====
        # Temporal average: B, T, H, W, C -> B, H, W, C
        x = x.max(dim=1)[0]

        # ===== SEGMENTATION HEAD =====
        # Apply per-pixel classification
        B, H, W, C = x.shape
        x = x.view(B * H * W, C)
        out = self.segmentation_head(x)
        out = out.view(B, H, W, self.num_classes)
        out = out.permute(0, 3, 1, 2)  # B, num_classes, H, W

        return out


class SwinModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False, 
                 checkpoint_path=None, load_encoder_only=False):
        super().__init__()
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        # Losses
        self.loss_func1 = smp.losses.DiceLoss(mode='binary', ignore_index=self.IGNORE_INDEX)
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15, ignore_index=self.IGNORE_INDEX)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

        # Swin-Unet 3D backbone
        self.backbone = SwinUNet3D(
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24]
        )
        # checkpoint_path = 'pretraining/checkpoints/tiny_epoch=71.ckpt'
        # Load pretrained weights if provided
        if checkpoint_path:
            self.load_pretrained_weights(checkpoint_path, load_encoder_only)

        if freeze:
            self._freeze_encoder()

    def load_pretrained_weights(self, checkpoint_path, encoder_only=False):
        """Load pretrained weights from checkpoint"""
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        if encoder_only:
            # Load only encoder weights
            encoder_state = {k.replace('backbone.encoder.', ''): v 
                           for k, v in state_dict.items() 
                           if k.startswith('backbone.encoder.')}
            self.backbone.encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded encoder weights only ({len(encoder_state)} keys)")
        else:
            # Load full model weights
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded full model weights ({len(state_dict)} keys)")

    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")

    
    def forward(self, x):
        # x shape: [B, C, T, H, W] where C=1 (single channel)
        x = x.permute(0, 2, 1, 3, 4)  # Change to [B, T, C, H, W] for Swin3D
        output = self.backbone(x)
        return output

   
# # GOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD  
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

#         # backbone = swin_transformer.SwinTransformer3d(
#         #     patch_size=[16, 4, 4],      # temporal=2, spatial=4x4 patches
#         #     embed_dim=96,              # base dimension
#         #     depths=[2, 2, 12],       # Tiny config
#         #     num_heads=[3, 6, 12],  # heads per stage
#         #     window_size=[1, 7, 7],     # attention window
#         #     stochastic_depth_prob=0.1, # DropPath
#         # )
#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')



#         # Get old weights
#         old_conv = backbone.patch_embed.proj  # Conv3d(3, 128, ...)
#         weight = old_conv.weight  # [128, 3, 2, 4, 4]
#         bias = old_conv.bias      # [128]

#         # Adapt weights: average across RGB → 1 channel
#         new_weight = weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]

#         # Replace conv with new one (keep out_channels=128!)
#         backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=128,
#             kernel_size=(2, 4, 4),
#             stride=(2, 4, 4),
#             bias=True
#         )

#         # Load adapted weights
#         backbone.patch_embed.proj.weight = nn.Parameter(new_weight)
#         backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())  # shape [128]
#         self.backbone = backbone#nn.Sequential(*list(backbone.children())[:-2]) 
        
#         pretrained_ckpt = None#'pretraining/checkpoints/64_tiny_16_scratch_epoch=5-v3.ckpt'
#         if pretrained_ckpt:
#             ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#             ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#             model_state_dict = self.backbone.state_dict()

#             new_state_dict = {}
#             used_ckpt_keys = set()  # keep track of keys already used


#             for k_model in model_state_dict.keys():
#                 # Skip decoder stuff
#                 if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#                     continue
#                 print(k_model)
#                 # Find corresponding key in checkpoint
#                 if k_model in ckpt_state_dict:
#                     new_state_dict[k_model] = ckpt_state_dict[k_model]
#                 else:
#                     # Fallback: find first unused key with matching shape
#                     for k_ckpt, v_ckpt in ckpt_state_dict.items():
#                         if k_ckpt in used_ckpt_keys:
#                             continue  # skip already used keys
#                         if v_ckpt.shape == model_state_dict[k_model].shape:
#                             new_state_dict[k_model] = v_ckpt
#                             used_ckpt_keys.add(k_ckpt)
#                             print(f"Fallback match: {k_model} <- {k_ckpt}")
#                             break
                            
#             # Load into model
#             msg = self.backbone.load_state_dict(new_state_dict, strict=False)
#             print("Loaded:", msg)

#         # Remove classifier head, keep norm
#         self.backbone.head = nn.Identity()
    
#         embed_dim = 768
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(embed_dim // 4, (self.hparams.size // 16) ** 2)
#         )
#     # 
#     def forward(self, x):

#         x = x.permute(0,2,1,3,4)
#         preds = self.backbone(x)  # runs backbone, sets self.feature
#         # print(preds.shape)
#         preds = self.classifier(preds)
#         preds = preds.view(-1,1,self.hparams.size//16,self.hparams.size//16)
#         return preds
    


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
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=8,mode='bilinear').squeeze(0).squeeze(0).numpy()
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