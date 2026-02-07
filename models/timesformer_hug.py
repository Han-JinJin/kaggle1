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
import albumentations as A
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification, AutoModel


import utils
import torch
import torch.nn.functional as F
from transformers import TimesformerModel, TimesformerConfig



# Convert to PIL and then to 3 channels
pil_transform = T.Compose([
    T.ToPILImage(),                    # convert (C, H, W) to PIL
    T.Grayscale(num_output_channels=1),  # convert to 3 channels
])

class TimesformerDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.rotate = A.Compose([A.Rotate(8,p=1)])
        self.transform = transform
        self.xyxys=xyxys
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            # T.Normalize(mean=[0.5], std=[0.5])
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        

        self.pil_transform  = pil_transform
        
    def __len__(self):
        return len(self.images)
    
    def fourth_augment(self, image):
        """
        Custom channel augmentation that returns exactly 24 channels.
        """
        # always select valid_chans
        cropping_num =  self.cfg.valid_chans 

        # pick crop indices
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        # pick where to paste them
        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        # container
        image_tmp = np.zeros_like(image)

        # paste cropped channels
        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

        # optional random cutout
        cutout_idx = random.randint(0, 1)
        temporal_random_cutout_idx = np.arange(start_paste_idx, start_paste_idx + cutout_idx)
        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0

        # finally, keep only 24 channels
        image = image_tmp[..., start_paste_idx:start_paste_idx + cropping_num]

        return image
    def z_circular_shift_np(self, volume, max_shift=2, prob=0.5, cutout_size=1, cutout_prob=0.1):
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
    def shuffle_d_axis(self,image):
        # image shape: (H, W, D)
        d = image.shape[2]
        shuffled_indices = np.arange(d)
        np.random.shuffle(shuffled_indices)

        # Reorder along D axis
        image_shuffled = image[:, :, shuffled_indices]
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
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            
            # image = image.permute(1,0,2,3)
            # frames = [self.pil_transform(frame.squeeze(0)) for frame in image] 

            # # now run the Timesformer processor ONCE:
            # encoding = self.processor(
            #     frames,
            #     # [frame for frame in frames],   # list of PIL
            #     return_tensors='pt'
            #     )
            # # encoding["pixel_values"] is (1, T, C, H, W)
            # pixel_values = encoding["pixel_values"].squeeze(0)
            # return pixel_values, label,xy
            image = image.permute(1,0,2,3)
            image = image.repeat(1, 3, 1, 1)
            image = torch.stack([self.video_transform(f) for f in image])
            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]

            image=self.fourth_augment(image)
            image = self.shuffle_d_axis(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
                
            image = image.permute(1,0,2,3)
            image = image.repeat(1, 3, 1, 1)

            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label
    
class TimesfomerModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None,wandb_logger=None, with_norm=False):
        super(TimesfomerModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)
        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        
        # hf_repo = "facebook/vjepa2-vitl-fpc64-256"

        # backbone = AutoModel.from_pretrained(hf_repo)
        
        encoder = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k600")
        # self.encoder = backbone
        #         # --- Build encoder ---
        # config = TimesformerConfig(
        #     num_frames=16,
        #     image_size=64,
        #     patch_size=16,
        #     num_channels=1,
        #     attention_type="divided_space_time",
        #     hidden_size=768,           # embedding dimension
        #     num_attention_heads=8,
        #     intermediate_size=768,
        #     num_hidden_layers=6        # <--- THIS is the depth (# of transformer blocks)
        # )
        # self.encoder = TimesformerModel(config)

        # # --- Optionally load pretrained weights ---
        # pretrained_path= "pretraining/checkpoints/64_tf_8_epoch=30.ckpt"
        # if pretrained_path is not None:
        #     checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        #     state_dict = checkpoint["state_dict"]

        #     new_state_dict = {}
        #     for k, v in state_dict.items():
        #         # Keep only encoder weights
        #         if k.startswith("encoder.encoder."):
        #             new_state_dict[k.replace("encoder.encoder.", "encoder.")] = v
        #         elif k.startswith("encoder."):
        #             new_state_dict[k.replace("encoder.", "")] = v

        #     result = self.encoder.load_state_dict(new_state_dict, strict=False)

        #     print(f"✅ Loaded encoder weights from: {pretrained_path}")
        #     print("Missing keys:", result.missing_keys)
        #     print("Unexpected keys:", result.unexpected_keys)
        # else:
        #     print("⚠️ No pretrained weights loaded (training from scratch).")

                
        # config = TimesformerConfig(
        #     num_frames=16,
        #     image_size=64,
        #     patch_size=8,
        #     num_channels=1,
        #     attention_type="divided_space_time",
        #     hidden_size=768,           # embedding dimension
        #     num_attention_heads=8,
        #     intermediate_size=768,
        #     num_hidden_layers=6        # <--- THIS is the depth (# of transformer blocks)
        # )
        # encoder = TimesformerModel(config)

        # checkpoint = torch.load("pretraining/checkpoints/64_tf_16_fs_epoch=1-v2.ckpt", map_location="cpu", weights_only=False)
    
        # result = encoder.load_state_dict(checkpoint["state_dict"], strict=False)
        self.encoder = encoder

        # # Print result info
        # print("✅ Encoder load successful!")
        # print("Missing keys:", result.missing_keys)
        # print("Unexpected keys:", result.unexpected_keys)
  

        self.classifier = nn.Sequential(
            nn.Linear(768, (self.hparams.size//16)**2),  
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(768, 1),  
        # )
    #     self.resnet_to_rgb = nn.Conv2d(1, 3, kernel_size=1)
    #             # # # Segformer expects 2D input with shape (B, C, H, W)
    #     self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
    #         "nvidia/mit-b0",
    #         num_labels=1,
    #         ignore_mismatched_sizes=True,
    #         num_channels=3
    #     )
        
    #     self.upscaler1 = nn.ConvTranspose2d(
    #         1, 1, kernel_size=(4, 4), stride=2, padding=1)
    #     self.upscaler2 = nn.ConvTranspose2d(
    #         1, 1, kernel_size=(4, 4), stride=2, padding=1)
    #     self.decoder = TransformerDecoder(embed_dim=768, num_layers=2, num_heads=8)

    #     self.upsample_head = nn.Sequential(
    #         nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(64, 1, kernel_size=1)  # Binary mask, use more channels for multi-class
    #     )
    # def forward(self, x):
    #     x = x.permute(0,2,1,3,4)
    #     _ = self.backbone(x)  # runs backbone, sets self.features
    #     feat = self.features  # (B, T_patch, H_patch, W_patch, C)
    #     feat = feat.permute(0, 4, 1, 2, 3)  # (B, C, T_patch, H_patch, W_patch)
    #     feat_2d = feat.mean(dim=2)  # average temporal patches: (B, C, H_patch, W_patch)
    #     seg_logits = self.decoder(feat_2d)  # (B, num_classes, 224, 224)
    #     return seg_logits
    # def forward(self, x):
    #     outputs = self.backbone(x, output_hidden_states=True)
    #     last_hidden_state = outputs.last_hidden_state  # tuple of all hidden layers
    #     last_hidden_state = last_hidden_state[:,1:,:]
    #     last_hidden_state = last_hidden_state.reshape(x.shape[0],16,14,14,768)
    #     # feat = last_hidden_state.permute(0,4,1,2,3)  # (B, C, T_patch, H_patch, W_patch)
    #     feat_2d = last_hidden_state.max(dim=1)[0]
    #     seg_logits = self.classifier(feat_2d)
    #     # seg_logits = self.decoder(feat_2d)  # (B, num_classes, 224, 224)
    #     return seg_logits.permute(0,3,1,2)
    def forward(self, x):
        outputs = self.encoder(x, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state  # tuple of all hidden layers
        cls = last_hidden_state[:,0,:]
        preds = self.classifier(cls)
        preds = preds.view(-1,1,self.hparams.size//16,self.hparams.size//16)
        return preds
    
    
# # GOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
# class Patch3DTransformerSegmentation(nn.Module):
#     def __init__(self, num_classes=1, embed_dim=768, num_heads=8, depth=2, patch_output=4):
#         super().__init__()
#         self.patch_output = patch_output
#         self.num_classes = num_classes

                
#         config = TimesformerConfig(
#             num_frames=16,
#             image_size=64,
#             patch_size=16,
#             num_channels=1,
#             attention_type="divided_space_time",
#             hidden_size=768,           # embedding dimension
#             num_attention_heads=8,
#             intermediate_size=768,
#             num_hidden_layers=6        # <--- THIS is the depth (# of transformer blocks)
#         )
#         backbone = TimesformerModel(config)
        
#         checkpoint = torch.load("pretraining/checkpoints/64_tf_16_fs_epoch=31.ckpt", map_location="cpu",weights_only=False)
#         state_dict = checkpoint["state_dict"]
#         # filter only encoder weights and strip prefixes
#         # keep only encoder weights and remove only the first 'encoder.' prefix
#         encoder_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith("encoder."):
#                 new_key = k[len("encoder."): ]  # remove only the first 'encoder.'
#                 encoder_state_dict[new_key] = v

#         missing, unexpected = backbone.load_state_dict(encoder_state_dict, strict=False)

#         print("Missing keys:", missing)
#         print("Unexpected keys:", unexpected)


#         self.encoder = backbone
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim,
#             dropout=0.1,
#             activation="gelu",
#             batch_first=True  # (B, N, C)
#         )
#         self.embed_dim = embed_dim
#         self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # Classifier: per patch -> small 3D patch
#         self.classifier = nn.Linear(embed_dim, patch_output ** 2)

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         outputs = self.encoder(x, output_hidden_states=True)
#         feats = outputs.last_hidden_state[:,1:,:]  # tuple of all hidden layers
        
        
#         feats = feats.view(B,T,8,8,-1)
#         # print(feats.shape)
#         feats = feats.mean(dim=1) # (B, 1, Hf, Wf, embed_dim,)
#         # print(feats.shape)
#         B, Hf, Wf, D = feats.shape

#         # # Move embed_dim to dim 1 and flatten patches
#         # patch_tokens = feats.permute(0, 3, 1, 2).contiguous()  # (B, 768, 8, 7, 7)
#         patch_tokens = feats.view(B, Hf*Wf, -1)      # (B, 768, 8*7*7=392)
#         # print(patch_tokens.shape)
        
#         # Transformer
#         transformed_tokens = self.decoder(patch_tokens) 
#         transformed_tokens = transformed_tokens.permute(0, 2, 1).view(B, self.embed_dim, Hf, Wf)  # (B, C, Hf, Wf)

#         # print(transformed_tsokens.shape)
#         transformed_tokens = transformed_tokens.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, -1)  # (B, N, C)
#         logits = self.classifier(transformed_tokens)
#         logits = logits.view(B, 1, 8 ,8)  # (B, patch_output^2, Hf, Wf)
#         return logits


# class TimesfomerModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None,wandb_logger=None, with_norm=False):
#         super(TimesfomerModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)
#         self.loss_func= lambda x,y: 0.6 * self.loss_func1(x,y)+0.4*self.loss_func2(x,y)
        
#         self.backbone = Patch3DTransformerSegmentation(num_classes=1, patch_output=1)
    
#     def forward(self, x):

#         output = self.backbone(x)  # runs backbone, sets self.feature
# #         return output
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import numpy as np
# import segmentation_models_pytorch as smp
# from transformers import TimesformerModel
# from peft import LoraConfig, get_peft_model, TaskType


# # --- helper wrapper to bypass PEFT’s NLP-style forward ---
# class VisionPEFTModel(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, *args, **kwargs):
#         # Directly call the vision transformer
#         return self.model(*args, **kwargs)


# class TimesfomerModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, with_norm=False):
#         super().__init__()
#         self.save_hyperparameters()

#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         # ---- losses ----
#         self.loss_func1 = smp.losses.DiceLoss(mode="binary", ignore_index=self.IGNORE_INDEX)
#         self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25, ignore_index=self.IGNORE_INDEX)
#         self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

#         # ---- pretrained encoder ----
#         encoder = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")

#         # ---- LoRA config ----
#         lora_config = LoraConfig(
#             r=8,
#             lora_alpha=16,
#             target_modules=[
#                 "attention.attention.qkv",
#                 "attention.output.dense",
#                 "temporal_attention.attention.qkv",
#                 "temporal_attention.output.dense",
#             ],
#             lora_dropout=0.1,
#             bias="none",
#             task_type=TaskType.FEATURE_EXTRACTION,
#         )

#         encoder = get_peft_model(encoder, lora_config)

#         # --- fix: override PEFT forward so it accepts tensors instead of input_ids ---
#         old_forward = encoder.forward

#         def patched_forward(self, *args, **kwargs):
#             # If PEFT passed input_ids, treat them as pixel_values or the first arg
#             if "input_ids" in kwargs:
#                 kwargs["pixel_values"] = kwargs.pop("input_ids")
#             return old_forward(*args, **kwargs)

#         import types
#         encoder.forward = types.MethodType(patched_forward, encoder)

#         # freeze non-LoRA params
#         for n, p in encoder.named_parameters():
#             if "lora" not in n:
#                 p.requires_grad = False

#         encoder.print_trainable_parameters()
#         self.encoder = VisionPEFTModel(encoder)

#         # ---- classification head ----
#         self.classifier = nn.Linear(768, (self.hparams.size // 16) ** 2)

#     # ---------------------------------------------------------
#     def forward(self, x):
#         # x: (B, T, C, H, W)
#         outputs = self.encoder(x, output_hidden_states=True)
#         last_hidden_state = outputs.last_hidden_state
#         cls = last_hidden_state[:, 0, :]
#         preds = self.classifier(cls)
#         preds = preds.view(-1, 1, self.hparams.size // 16, self.hparams.size // 16)
#         return preds

    # ---------------------------------------------------------
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     preds = self.forward(x)
    #     loss = self.loss_func(preds, y)
    #     self.log("train_loss", loss, prog_bar=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     preds = self.forward(x)
    #     loss = self.loss_func(preds, y)
    #     self.log("val_loss", loss, prog_bar=True)
    #     return loss

    # def configure_optimizers(self):
    #     trainable_params = [p for p in self.parameters() if p.requires_grad]
    #     optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.lr, weight_decay=0.01)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
   
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
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
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.hparams.lr, weight_decay=0.01)
        scheduler = get_scheduler(optimizer, scheduler=self.hparams.scheduler)
        return [optimizer], [scheduler]
    
    # def configure_optimizers(self):
        # # # Separate the head parameters
        # # head_params = list(self.classifier.parameters())
        # # other_params = [p for n, p in self.named_parameters() if "classifier" not in n]
        
        # # assert len(head_params) > 0, "No parameters found for 'head_params' group"
        # # assert len(other_params) > 0, "No parameters found for 'other_params' group"

        # # param_groups = [
        # #     {'params': other_params, 'lr': self.hparams.lr},
        # #     {'params': head_params, 'lr': self.hparams.lr},  # 10x LR for the head
        # # ]

        # optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
    
        # scheduler = get_scheduler(optimizer,scheduler=self.hparams.scheduler)
        # return [optimizer], [scheduler]
        
    # def configure_optimizers(self):
    #     weight_decay = 0.001
    #     base_lr = self.hparams.lr

    #     # 1️⃣ Backbone parameters in their own group
    #     backbone_params = list(self.encoder.encoder.parameters())

    #     # 2️⃣ Everything else (decoder, head, etc.)
    #     other_params = [
    #         p for n, p in self.encoder.named_parameters()
    #         if p.requires_grad and not any(k in n for k in ["backbone"])
    #     ]

    #     # 3️⃣ Sanity checks
    #     assert len(other_params) > 0, "No parameters found for 'other_params' group"
    #     assert len(backbone_params) > 0, "No parameters found for 'backbone_params' group"
        
    #     total_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
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

    #     return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        
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