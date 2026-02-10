"""
- EfficientNet-B0 预训练 backbone
- UNet 架构（从公开库中调用segmentation_models_pytorch）
- 中间6层Z通道输入
- 3-Fold 交叉验证
-需要运行三次，每次从三个fragment里选一个当验证集，其他做训练，生成三个最佳模型权重
- valid_id = 1 运行第1次，后面每次修改 
"""

# 1. 导入库
import os
import gc
import sys
import time
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

# Kaggle 路径
sys.path.append('/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4')
sys.path.append('/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append('/kaggle/input/segmentation-models-pytorch/segmentation_models.pytorch-master')

import segmentation_models_pytorch as smp

# 2. 配置类

class CFG:
    # ============== 竞赛配置 =============
    comp_name = 'vesuvius'
    comp_dir_path = '/kaggle/input/'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'

    exp_name = 'vesuvius_2d_improved_v2'

    # ============== 模型配置 =============
    model_name = 'Unet'
    backbone = 'efficientnet-b0'
    in_chans = 6  # 中间6层Z通道
    target_size = 1
    use_attention = True  # 使用SCSE注意力机制

    # ============== 训练配置 =============
    size = 224
    tile_size = 224
    stride = tile_size // 2  # 50% 重叠

    train_batch_size = 16
    valid_batch_size = 32
    use_amp = True

    epochs = 15
    lr = 1e-4
    warmup_factor = 10

    # ============== Fold 配置 =============
    valid_id = 1  # 需要运行3次：1, 2, 3

    # ============== 优化器配置 =============
    weight_decay = 1e-6
    max_grad_norm = 1000
    min_lr = 1e-6

    # ============== 其他配置 =============
    num_workers = 4
    seed = 42

    # ============== 损失函数权重 =============
    dice_loss_weight = 0.3  # Dice Loss 权重

    # ============== 路径配置 =============
    outputs_path = f'/kaggle/working/outputs/{comp_name}/{exp_name}/'
    model_dir = outputs_path + f'{comp_name}-models/'
    figures_dir = outputs_path + 'figures/'
    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== 数据增强 =============
    train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3),
                       max_height=int(size * 0.3), mask_fill_value=0, p=0.5),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]


# 3. 工具函数
class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logger(log_file):
    """初始化日志"""
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def set_seed(seed=None):
    """设置随机种子"""
    if seed is None:
        seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    """创建目录"""
    for dir_path in [cfg.model_dir, cfg.figures_dir, cfg.log_dir]:
        os.makedirs(dir_path, exist_ok=True)


def cfg_init(cfg):
    """初始化配置"""
    set_seed(cfg.seed)
    make_dirs(cfg)

# 4. 数据加载
def read_image_mask(fragment_id, cfg):
    """
    读取 fragment 的图像和 mask
    使用中间6层 Z 通道 (29-34 of 65)
    """
    images = []

    # 中间6层
    mid = 65 // 2
    start = mid - cfg.in_chans // 2
    end = mid + cfg.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs, desc=f'加载 fragment {fragment_id}'):
        image_path = f"{cfg.comp_dataset_path}train/{fragment_id}/surface_volume/{i:02}.tif"
        image = cv2.imread(image_path, 0)

        # Padding 到 tile_size 的倍数
        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)

    # Stack: (H, W, C)
    images = np.stack(images, axis=2)

    # 读取 mask
    mask_path = f"{cfg.comp_dataset_path}train/{fragment_id}/inklabels.png"
    mask = cv2.imread(mask_path, 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype('float32') / 255.0

    return images, mask


def get_train_valid_dataset(cfg):
    """生成训练和验证数据集"""
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):
        image, mask = read_image_mask(fragment_id, cfg)

        # 滑动窗口切分
        x1_list = list(range(0, image.shape[1] - cfg.tile_size + 1, cfg.stride))
        y1_list = list(range(0, image.shape[0] - cfg.tile_size + 1, cfg.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size

                if fragment_id == cfg.valid_id:
                    # 验证集
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])
                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    # 训练集
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_transforms(data, cfg):
    """获取数据增强"""
    if data == 'train':
        return A.Compose(cfg.train_aug_list)
    else:
        return A.Compose(cfg.valid_aug_list)


class CustomDataset(Dataset):
    """自定义数据集"""
    def __init__(self, images, masks, cfg, transform=None):
        self.images = images
        self.masks = masks
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            data = self.transform(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        return image, mask


# 5. 模型定义
class CustomModel(nn.Module):
    """自定义分割模型"""
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        # ✅ 使用 EfficientNet-B0 预训练 UNet + SCSE注意力
        self.encoder = smp.Unet(
            encoder_name=cfg.backbone,
            encoder_weights=weight,  # 'imagenet'
            in_channels=cfg.in_chans,
            classes=cfg.target_size,
            activation=None,
            attention_type='scse' if cfg.use_attention else None,  # SCSE注意力
        )

    def forward(self, image):
        output = self.encoder(image)
        return output


def build_model(cfg, weight="imagenet"):
    """构建模型"""
    print(f'model_name: {cfg.model_name}')
    print(f'backbone: {cfg.backbone}')
    print(f'pretrained: {weight}')
    print(f'attention: {"SCSE" if cfg.use_attention else "None"}')

    model = CustomModel(cfg, weight)
    return model


# 6. 损失函数（改进版）
class CombinedLoss(nn.Module):
    """
    组合损失：BCE + Dice

    BCE: 像素级二分类损失
    Dice: 分割重叠度损失（提升边界质量）
    """
    def __init__(self, dice_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.dice_weight = dice_weight

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        return bce_loss + self.dice_weight * dice_loss


# 7. 评估指标
def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    计算 F_beta 分数
    beta=0.5 表示更强调 precision（精确率）
    竞赛官方评估指标
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)

    f_beta = (1 + beta_squared) * (c_precision * c_recall) / \
             (beta_squared * c_precision + c_recall + smooth)

    return f_beta


def calc_fbeta(mask, mask_pred, logger=None):
    """
    计算最佳 F0.5 分数和阈值
    搜索范围：0.10 - 0.50，步长 0.05
    """
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0

    for th in np.array(range(10, 50 + 1, 5)) / 100:
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        if dice > best_dice:
            best_dice = dice
            best_th = th

    if logger:
        logger.info(f'最佳阈值: {best_th:.2f}, F0.5: {best_dice:.4f}')

    return best_dice, best_th


# 8. 学习率调度器
class GradualWarmupSchedulerV2:
    """
    渐进式 Warmup 调度器

    先线性增加学习率，再使用余弦退火
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0

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
            return [base_lr * (float(self.last_epoch) / self.total_epoch)
                   for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                   for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_scheduler(cfg, optimizer):
    """获取学习率调度器"""
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=cfg.min_lr
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=cfg.warmup_factor, total_epoch=1,
        after_scheduler=scheduler_cosine
    )
    return scheduler

# 9. 训练和验证函数
def train_fn(train_loader, model, criterion, optimizer, scaler, cfg):
    """训练一个 epoch"""
    model.train()
    losses = AverageMeter()

    for step, (images, masks) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc='训练'
    ):
        images = images.to(device)
        masks = masks.to(device)
        batch_size = masks.size(0)

        # 混合精度训练
        with autocast(cfg.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, masks)

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.avg


def valid_fn(valid_loader, model, criterion, cfg, valid_xyxys, valid_mask_gt):
    """验证"""
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, masks) in tqdm(
        enumerate(valid_loader), total=len(valid_loader), desc='验证'
    ):
        images = images.to(device)
        masks = masks.to(device)
        batch_size = masks.size(0)

        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, masks)
        losses.update(loss.item(), batch_size)

        # 重构完整 mask
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        start_idx = step * cfg.valid_batch_size
        end_idx = start_idx + batch_size

        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))

    mask_pred /= mask_count
    return losses.avg, mask_pred


# 10. 主训练流程
def main():
    # 初始化
    cfg_init(CFG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = init_logger(log_file=CFG.log_path)

    # 加载数据
    print('加载数据...')
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = \
        get_train_valid_dataset(CFG)
    valid_xyxys = np.stack(valid_xyxys)
    print(f'训练样本: {len(train_images)}')
    print(f'验证样本: {len(valid_images)}')

    # 创建数据集
    train_dataset = CustomDataset(
        train_images, train_masks, CFG,
        transform=get_transforms('train', CFG)
    )
    valid_dataset = CustomDataset(
        valid_images, valid_masks, CFG,
        transform=get_transforms('valid', CFG)
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    # 构建模型
    print('\n构建模型...')
    model = build_model(CFG)
    model.to(device)

    # 损失函数和优化器
    criterion = CombinedLoss(dice_weight=CFG.dice_loss_weight)
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(CFG, optimizer)

    # 混合精度
    scaler = GradScaler(enabled=CFG.use_amp)

    # 验证集 mask
    valid_mask_gt = cv2.imread(
        CFG.comp_dataset_path + f"train/{CFG.valid_id}/inklabels.png", 0
    )
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    # 训练循环
    fold = CFG.valid_id
    best_score = -1
    best_loss = np.inf

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # 训练
        train_loss = train_fn(train_loader, model, criterion, optimizer, scaler, CFG)

        # 验证
        val_loss, mask_pred = valid_fn(
            valid_loader, model, criterion, CFG, valid_xyxys, valid_mask_gt
        )

        # 计算指标
        scheduler.step(epoch)
        best_dice, best_th = calc_fbeta(valid_mask_gt, mask_pred, logger)
        score = best_dice
        elapsed = time.time() - start_time

        # 日志
        logger.info(
            f'Epoch {epoch+1}/{CFG.epochs} - '
            f'训练损失: {train_loss:.4f} | '
            f'验证损失: {val_loss:.4f} | '
            f'F0.5: {score:.4f} (阈值: {best_th:.2f}) | '
            f'时间: {elapsed:.0f}s'
        )

        print(
            f'Epoch {epoch+1}/{CFG.epochs} - '
            f'训练损失: {train_loss:.4f} | '
            f'验证损失: {val_loss:.4f} | '
            f'F0.5: {score:.4f} (阈值: {best_th:.2f}) | '
            f'时间: {elapsed:.0f}s'
        )

        # 保存最佳模型
        if score > best_score:
            best_loss = val_loss
            best_score = score

            logger.info(f'保存最佳模型 (F0.5: {best_score:.4f})')

            torch.save({
                'model': model.state_dict(),
                'preds': mask_pred,
                'threshold': best_th,
                'score': best_score
            }, CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth')

    logger.info(f'\n训练完成!')
    logger.info(f'最佳 F0.5 分数: {best_score:.4f}')
    logger.info(f'最佳验证损失: {best_loss:.4f}')


if __name__ == '__main__':
    main()
