import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import tifffile
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import random
import gc
import math
from collections import defaultdict

# 配置
class Config:
    DATA_ROOT = "/kaggle/input"
    OUTPUT_DIR = "/kaggle/working"

    START_Z = 22
    END_Z = 42
    IN_CHANNELS = 20

    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 3e-5  # 增强正则化

    WINDOW_SIZE = 224
    STRIDE = 168

    MIN_INK_PIXELS = 15  # 折中
    POSITIVE_RATIO = 0.40  # 折中

    USE_HORIZONTAL_FLIP = True
    USE_VERTICAL_FLIP = True

    INFERENCE_BATCH_SIZE = 32
    USE_TTA = True
    USE_MORPHOLOGY = True

    BINARY_THRESHOLD = 0.50  # 0.60 → 0.50

    # 高斯加权窗口
    USE_GAUSSIAN_WEIGHT = True

    USE_HARD_NEGATIVE_MINING = True
    HARD_NEGATIVE_RATIO = 0.35  # 折中

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    SEED = 42
    PIN_MEMORY = True
    USE_MULTIPLE_GPU = torch.cuda.device_count() > 1
    GRADIENT_CLIP = 1.0

cfg = Config()

# 注意力机制模块
class AttentionGate(nn.Module):
    """注意力门 - 聚焦相关特征"""
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        # gate: 来自解码器的特征 [B, C_g, H, W]
        # skip: 来自编码器的跳跃连接 [B, C_s, H, W]
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)

        # 调整尺寸匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return skip * psi

# ================================
# 数据集 - 按需加载Z层
# ================================
class CachedDataset(Dataset):
    def __init__(self, fragment_id, config, split='train'):
        self.fragment_id = fragment_id
        self.config = config
        self.split = split
        self.base_path = self._get_base_path(fragment_id, config, split)

        self.mask = cv2.imread(str(self.base_path / "mask.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if split == 'train':
            self.label = cv2.imread(str(self.base_path / "inklabels.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            self.label = None

        self.height, self.width = self.mask.shape
        self.positions = self._generate_positions()

        # 按需加载
        self.slice_paths = [
            self.base_path / "surface_volume" / f"{i:02d}.tif"
            for i in range(config.START_Z, config.END_Z)
        ]
        print(f"    准备{config.IN_CHANNELS}个切片（按需加载）...", end=' ', flush=True)

        # 预加载以加速训练
        self.volume = self._preload_slices()

        self.positive_indices = []
        self.negative_indices = []
        if split == 'train':
            self._classify_samples()

    def _get_base_path(self, fragment_id, config, split):
        data_root = Path(config.DATA_ROOT) / "vesuvius-challenge-ink-detection"
        if not data_root.exists():
            import os
            for item in os.listdir(config.DATA_ROOT):
                if "vesuvius" in item.lower() and "challenge" in item.lower():
                    data_root = Path(config.DATA_ROOT) / item
                    if (data_root / split).exists():
                        break
        if not (data_root / split).exists():
            raise FileNotFoundError(f"找不到数据: {data_root / split / fragment_id}")
        return data_root / split / fragment_id

    def _preload_slices(self):
        """预加载所有Z层到内存"""
        slices = []
        for i in range(self.config.START_Z, self.config.END_Z):
            slice_path = self.base_path / "surface_volume" / f"{i:02d}.tif"
            slice_img = tifffile.imread(slice_path).astype(np.float32) / 65535.0
            slices.append(slice_img)
        return np.stack(slices, axis=0)

    def _generate_positions(self):
        positions = []
        ws = self.config.WINDOW_SIZE
        stride = self.config.STRIDE
        for y in range(0, self.height - ws + 1, stride):
            for x in range(0, self.width - ws + 1, stride):
                if self.mask[y:y+ws, x:x+ws].mean() > 0.1:
                    positions.append((y, x))
        return positions

    def _classify_samples(self):
        ws = self.config.WINDOW_SIZE
        for idx in range(len(self)):
            y, x = self.positions[idx]
            label_patch = self.label[y:y+ws, x:x+ws]
            # 提高正样本门槛
            if label_patch.sum() >= self.config.MIN_INK_PIXELS:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        y, x = self.positions[idx]
        ws = self.config.WINDOW_SIZE

        volume_patch = self.volume[:, y:y+ws, x:x+ws]
        mask_patch = self.mask[y:y+ws, x:x+ws]
        volume_patch = volume_patch * mask_patch[np.newaxis, :, :]
        volume_tensor = torch.from_numpy(volume_patch).float()

        if self.split == 'train':
            label_patch = self.label[y:y+ws, x:x+ws] * mask_patch
            label_tensor = torch.from_numpy(label_patch).float().unsqueeze(0)

            if self.config.USE_HORIZONTAL_FLIP and random.random() > 0.5:
                volume_tensor = torch.flip(volume_tensor, [2])
                label_tensor = torch.flip(label_tensor, [2])
            if self.config.USE_VERTICAL_FLIP and random.random() > 0.5:
                volume_tensor = torch.flip(volume_tensor, [1])
                label_tensor = torch.flip(label_tensor, [1])

            return volume_tensor, label_tensor, idx  #
        else:
            return volume_tensor, (y, x)

class HardNegativeSampler:
    """难例挖掘采样器 - 重点学习易误判的负样本"""
    def __init__(self, dataset, positive_indices, negative_indices,
                 total_samples, batch_size, positive_ratio=0.42,
                 hard_negative_ratio=0.3):
        self.dataset = dataset
        self.positive = positive_indices
        self.negative = negative_indices
        self.total = total_samples
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.hard_negative_ratio = hard_negative_ratio

        self.num_pos = max(1, int(batch_size * positive_ratio))
        self.num_neg = batch_size - self.num_pos
        self.num_hard_neg = max(1, int(self.num_neg * hard_negative_ratio))
        self.num_easy_neg = self.num_neg - self.num_hard_neg

        # 难例负样本池（初始化为空，训练时动态更新）
        self.hard_negatives = []

        pos_pct = 100 * len(self.positive) / self.total
        print(f"    正样本: {len(self.positive)} ({pos_pct:.1f}%)")
        print(f"    负样本: {len(self.negative)} ({100-pos_pct:.1f}%)")

    def update_hard_negatives(self, model, device, config, num_samples=150):
        """更新难例负样本池 - 增加采样数量"""
        if not self.negative:
            return

        model.eval()
        losses = []

        with torch.no_grad():
            # 随机采样负样本计算loss
            sample_indices = random.sample(self.negative, min(len(self.negative), num_samples))

            for idx in sample_indices:
                vol, lab, _ = self.dataset[idx]  
                vol = vol.unsqueeze(0).to(device)
                lab = lab.unsqueeze(0).to(device)

                # 只计算负样本区域的loss，更精确
                output = model(vol)
                loss = F.binary_cross_entropy(output, lab, reduction='none')

                # 关注被错误预测为正的区域
                false_positive_loss = loss[(lab < 0.5) & (output > 0.3)].mean().item() if ((lab < 0.5) & (output > 0.3)).any() else 0
                losses.append((idx, false_positive_loss))

        # 按loss排序，取前40%作为难例（增加难例比例）
        losses.sort(key=lambda x: x[1], reverse=True)
        num_hard = min(int(len(losses) * 0.4), len(losses))
        self.hard_negatives = [idx for idx, _ in losses[:num_hard]]

        model.train()

    def __iter__(self):
        num_batches = self.total // self.batch_size
        for _ in range(num_batches):
            # 正样本
            pos = random.choices(self.positive, k=self.num_pos) if self.positive else []

            # 负样本：难例 + 简单负样本
            hard_neg = []
            easy_neg = []

            if self.hard_negatives:
                hard_neg = random.choices(self.hard_negatives, k=self.num_hard_neg)

            remaining_neg = [idx for idx in self.negative if idx not in self.hard_negatives]
            if remaining_neg:
                easy_neg = random.choices(remaining_neg, k=min(len(remaining_neg), self.num_easy_neg))

            if len(hard_neg) < self.num_hard_neg:
                shortage = self.num_hard_neg - len(hard_neg)
                if remaining_neg:
                    hard_neg.extend(random.choices(remaining_neg, k=min(len(remaining_neg), shortage)))

            neg = hard_neg + easy_neg
            if len(neg) < self.num_neg:
                pos = pos[:self.num_neg - len(neg)]

            batch = pos + neg
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.total


class BalancedSampler:
    def __init__(self, positive_indices, negative_indices, total_samples, batch_size, positive_ratio=0.42):
        self.positive = positive_indices
        self.negative = negative_indices
        self.total = total_samples
        self.batch_size = batch_size
        self.num_pos = max(1, int(batch_size * positive_ratio))
        self.num_neg = batch_size - self.num_pos
        pos_pct = 100 * len(self.positive) / self.total
        print(f"    正样本: {len(self.positive)} ({pos_pct:.1f}%)")

    def __iter__(self):
        num_batches = self.total // self.batch_size
        for _ in range(num_batches):
            pos = random.choices(self.positive, k=self.num_pos) if self.positive else []
            neg = random.choices(self.negative, k=self.num_neg) if self.negative else []
            batch = pos + neg
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.total

# 改进的UNet + 注意力
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class AttentionUNet(nn.Module):
    """UNet + 注意力机制"""
    def __init__(self, in_channels=20):
        super().__init__()
        # 编码器添加轻量Dropout（0.1）
        self.enc1 = DoubleConv(in_channels, 40, use_dropout=True, dropout_rate=0.1)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(40, 80, use_dropout=True, dropout_rate=0.1)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(80, 160, use_dropout=True, dropout_rate=0.1)
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(160, 320, use_dropout=False)
        )

        # 添加注意力门
        self.attn3 = AttentionGate(320, 160, 80)
        self.attn2 = AttentionGate(160, 80, 40)
        self.attn1 = AttentionGate(80, 40, 20)

        # 解码器 - 保持较强的Dropout（0.15）防止过拟合
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(320 + 160, 160, use_dropout=True, dropout_rate=0.15)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(160 + 80, 80, use_dropout=True, dropout_rate=0.15)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(80 + 40, 40, use_dropout=True, dropout_rate=0.15)

        self.outc = nn.Conv2d(40, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        # 使用注意力门聚焦特征
        d1 = self.up1(b)
        e3_attended = self.attn3(d1, e3)  # 注意力
        d1 = torch.cat([d1, e3_attended], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        e2_attended = self.attn2(d2, e2)  # 注意力
        d2 = torch.cat([d2, e2_attended], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        e1_attended = self.attn1(d3, e1)  # 注意力
        d3 = torch.cat([d3, e1_attended], dim=1)
        d3 = self.dec3(d3)

        return torch.sigmoid(self.outc(d3))

# Focal Loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):  # 反复调优
        super().__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma  # 聚焦参数

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1) / (pred.sum() + target.sum() + 1)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)  
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.focal(pred, target) + 0.5 * self.dice(pred, target)

# 评估指标
def calculate_metrics(pred, target, threshold=0.5):
    """计算准确率、精确率、召回率、F0.5"""
    pred_binary = (pred > threshold).float()

    TP = (pred_binary * target).sum()
    FP = (pred_binary * (1 - target)).sum()
    FN = ((1 - pred_binary) * target).sum()
    TN = ((1 - pred_binary) * (1 - target)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)

    # F0.5分数
    beta = 0.5
    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-7)

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f0.5': f_beta.item()
    }

# 训练
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(config=None):
    if config is None:
        config = cfg

    set_seed(config.SEED)

    print(f"\n[创建模型]")
    model = AttentionUNet(in_channels=config.IN_CHANNELS).to(config.DEVICE)

    if config.USE_MULTIPLE_GPU:
        print(f"  使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)

    params = sum(p.numel() for p in model.parameters())
    print(f"  参数: {params:,}")

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                          weight_decay=config.WEIGHT_DECAY)

    # 使用余弦退火+预热
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    print(f"\n[开始训练]")
    best_loss = float('inf')
    best_f05 = 0.0
    patience = 0
    fragments = ["1", "2", "3"]

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for frag_idx, frag in enumerate(fragments):
            print(f"\n  [Epoch {epoch}/{config.NUM_EPOCHS}] 加载片段 {frag}...", end=' ', flush=True)

            ds = CachedDataset(frag, config, 'train')
            print(f"{len(ds)} patches")

            # 选择采样器
            if config.USE_HARD_NEGATIVE_MINING and epoch > 1:
                # 第一个epoch后更新难例
                sampler = HardNegativeSampler(
                    ds, ds.positive_indices, ds.negative_indices,
                    len(ds), config.BATCH_SIZE, config.POSITIVE_RATIO,
                    config.HARD_NEGATIVE_RATIO
                )
                sampler.update_hard_negatives(model, config.DEVICE, config, num_samples=200)
            else:
                sampler = BalancedSampler(
                    ds.positive_indices,
                    ds.negative_indices,
                    len(ds),
                    config.BATCH_SIZE,
                    config.POSITIVE_RATIO
                )

            loader = DataLoader(
                ds,
                batch_size=config.BATCH_SIZE,
                sampler=sampler,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                drop_last=True
            )

            pbar = tqdm(loader, desc=f"Fragment {frag}", leave=False)
            for volumes, labels, _ in pbar:  
                volumes = volumes.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)

                outputs = model(volumes)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()

                if config.GRADIENT_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})

                if num_batches % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del ds, loader, pbar
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches

        # 指标评估
        model.eval()
        val_loss = 0
        val_metrics = defaultdict(float)
        val_count = 0
        print(f"\n  验证...", end=' ', flush=True)
        ds = CachedDataset("1", config, 'train')
        with torch.no_grad():
            for _ in range(100):  # 增加到100个样本
                idx = random.randint(0, len(ds) - 1)
                vol, lab, _ = ds[idx]  
                vol = vol.unsqueeze(0).to(config.DEVICE)
                lab = lab.unsqueeze(0).to(config.DEVICE)
                out = model(vol)

                val_loss += criterion(out, lab).item()

                # 计算指标
                metrics = calculate_metrics(out, lab, threshold=config.BINARY_THRESHOLD)
                for k, v in metrics.items():
                    val_metrics[k] += v
                val_count += 1

        del ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_loss = val_loss / max(val_count, 1)
        for k in val_metrics:
            val_metrics[k] /= val_count

        # ✅ V3优化：阈值搜索 - 找最佳F0.5阈值
        best_threshold = config.BINARY_THRESHOLD
        best_f05_search = val_metrics['f0.5']

        if epoch >= 3:  # 从第3个epoch开始搜索
            threshold_search = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
            f05_scores = []

            ds_search = CachedDataset("1", config, 'train')
            with torch.no_grad():
                for thresh in threshold_search:
                    f05_sum = 0
                    for _ in range(30):  # 用30个样本快速搜索
                        idx = random.randint(0, len(ds_search) - 1)
                        vol, lab, _ = ds_search[idx]
                        vol = vol.unsqueeze(0).to(config.DEVICE)
                        lab = lab.unsqueeze(0).to(config.DEVICE)
                        out = model(vol)
                        metrics = calculate_metrics(out, lab, threshold=thresh)
                        f05_sum += metrics['f0.5']
                    f05_scores.append(f05_sum / 30)

            best_idx = np.argmax(f05_scores)
            best_threshold = threshold_search[best_idx]
            best_f05_search = f05_scores[best_idx]

            del ds_search
            gc.collect()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"✓\n  Train: {avg_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"  Val ( thresh={config.BINARY_THRESHOLD:.2f}) - Acc: {val_metrics['accuracy']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | "
              f"Rec: {val_metrics['recall']:.4f} | "
              f"F0.5: {val_metrics['f0.5']:.4f}")

        if epoch >= 3:
            print(f"  最佳阈值: {best_threshold:.2f} → F0.5: {best_f05_search:.4f}")
            val_metrics['f0.5'] = best_f05_search  # 使用最佳阈值的分数

        # 综合考虑loss和F0.5
        if val_metrics['f0.5'] > best_f05 or val_loss < best_loss:
            if val_metrics['f0.5'] > best_f05:
                best_f05 = val_metrics['f0.5']
            if val_loss < best_loss:
                best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/best_model.pth")
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f}, F0.5: {val_metrics['f0.5']:.4f})")
        else:
            patience += 1
            print(f"  Patience: {patience}/8")

        if patience >= 8:
            print(f"\n提前停止于epoch {epoch}")
            break

    print("\n✓ 训练完成!")
# 推理
# ================================

def custom_collate_fn(batch):
    """自定义collate_fn处理测试数据"""
    volumes = torch.stack([item[0] for item in batch])
    positions = list(zip(*[item[1] for item in batch]))  # 分离y和x坐标
    return volumes, positions

def rle_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend([b + 1, 0])
        else:
            run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))

def create_gaussian_weight(window_size, sigma=3.0):
    """创建高斯权重图 - 中心权重高"""
    ax = np.linspace(-window_size // 2, window_size // 2, window_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def inference(config=None):
    if config is None:
        config = cfg

    print("\n" + "=" * 60)
    print("开始推理 (V2优化)")
    print("=" * 60)
    print(f"  二值化阈值: {config.BINARY_THRESHOLD}")
    print(f"  高斯加权窗口: {config.USE_GAUSSIAN_WEIGHT}")

    model = AttentionUNet(in_channels=config.IN_CHANNELS).to(config.DEVICE)

    state_dict = torch.load(f"{config.OUTPUT_DIR}/best_model.pth")
    if config.USE_MULTIPLE_GPU:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # 高斯权重图
    if config.USE_GAUSSIAN_WEIGHT:
        gaussian_weight = create_gaussian_weight(config.WINDOW_SIZE, sigma=3.0)
    else:
        gaussian_weight = None

    submission_data = []

    for frag in ["a", "b"]:
        print(f"\n[片段 {frag}]")
        ds = CachedDataset(frag, config, 'test')
        loader = DataLoader(ds, batch_size=config.INFERENCE_BATCH_SIZE,
                           num_workers=0, collate_fn=custom_collate_fn)  # ✅ 添加collate_fn

        h, w = ds.height, ds.width
        pred_map = np.zeros((h, w))
        count_map = np.zeros((h, w))

        with torch.no_grad():
            for volumes, positions in tqdm(loader, desc="推理"):
                volumes = volumes.to(config.DEVICE, non_blocking=True)

                if config.USE_TTA:
                    preds = [model(volumes)]
                    preds.append(model(torch.flip(volumes, [3])))
                    preds.append(model(torch.flip(volumes, [2])))
                    output = torch.stack(preds).mean(0).cpu().numpy()
                else:
                    output = model(volumes).cpu().numpy()

                # positions是 ((y1, y2, ...), (x1, x2, ...))
                ys, xs = positions
                for i in range(len(ys)):
                    y = int(ys[i])
                    x = int(xs[i])

                    if config.USE_GAUSSIAN_WEIGHT:
                        # 高斯加权
                        pred_map[y:y+config.WINDOW_SIZE, x:x+config.WINDOW_SIZE] += \
                            output[i, 0] * gaussian_weight
                        count_map[y:y+config.WINDOW_SIZE, x:x+config.WINDOW_SIZE] += gaussian_weight
                    else:
                        pred_map[y:y+config.WINDOW_SIZE, x:x+config.WINDOW_SIZE] += output[i, 0]
                        count_map[y:y+config.WINDOW_SIZE, x:x+config.WINDOW_SIZE] += 1

        count_map[count_map == 0] = 1
        pred_map = pred_map / count_map * ds.mask

        # 提高二值化阈值
        binary = (pred_map > config.BINARY_THRESHOLD).astype(np.uint8)
        if config.USE_MORPHOLOGY:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        rle = rle_encode(binary)
        submission_data.append({'Id': frag, 'Predicted': rle})

        cv2.imwrite(f"{config.OUTPUT_DIR}/pred_{frag}.png", (pred_map * 255).astype(np.uint8))

        # 输出统计信息
        ink_ratio = binary.sum() / binary.size
        print(f"  墨水像素比例: {ink_ratio:.2%}")

        del ds, loader, pred_map, count_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    submission = pd.DataFrame(submission_data)
    submission.to_csv(f"{config.OUTPUT_DIR}/submission.csv", index=False)
    print(f"\n✓ 提交文件已保存")

    return submission


