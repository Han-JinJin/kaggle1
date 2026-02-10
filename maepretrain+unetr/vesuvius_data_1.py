
# vesuvius_data.py
import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tifffile
except Exception:
    tifffile = None


IGNORE_INDEX = 127  # keep consistent with your losses


# ---- per-process cache (DataLoader workers each has its own process) ----
_MEMMAP_CACHE: Dict[str, np.ndarray] = {}


def _read_slice_memmap(path: str) -> np.ndarray:
    arr = _MEMMAP_CACHE.get(path, None)
    if arr is not None:
        return arr
    if tifffile is not None:
        # Kaggle /kaggle/input is read-only, so MUST use mode="r"
        try:
            arr = tifffile.memmap(path, mode="r")
        except TypeError:
            # some tifffile versions don't expose mode; fallback to imread
            arr = tifffile.imread(path)
        except PermissionError:
            # fallback to normal read if memmap fails
            arr = tifffile.imread(path)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _MEMMAP_CACHE[path] = arr
    return arr



def _load_png_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _normalize_volume(x: np.ndarray, clip_min=0.0, clip_max=200.0) -> np.ndarray:
    # follow your existing normalization style (clip then scale to [-1, 1])
    x = np.clip(x, clip_min, clip_max)
    x = x / 255.0
    x = (x - 0.5) / 0.5
    return x.astype(np.float32)


def sample_depth_indices(total_slices: int, num_frames: int, mode: str) -> List[int]:
    """
    total_slices: usually 65 (00..64)
    num_frames: e.g. 24
    mode:
      - rand_contig: random consecutive window
      - center_contig: fixed center window
      - odd_subsample: pick 24 roughly-uniform indices from odd slices
      - rand_stride2: random start + stride 2 (if feasible), else fall back to rand_contig
    """
    if num_frames > total_slices:
        raise ValueError(f"num_frames={num_frames} > total_slices={total_slices}")

    if mode == "rand_contig":
        s = random.randint(0, total_slices - num_frames)
        return list(range(s, s + num_frames))

    if mode == "center_contig":
        s = (total_slices - num_frames) // 2
        return list(range(s, s + num_frames))

    if mode == "odd_subsample":
        odds = list(range(1, total_slices, 2))  # 1,3,5,...
        if len(odds) >= num_frames:
            # uniform pick num_frames from odds
            idx = np.linspace(0, len(odds) - 1, num_frames).round().astype(int)
            return [odds[i] for i in idx]
        # fallback
        return sample_depth_indices(total_slices, num_frames, "rand_contig")

    if mode == "rand_stride2":
        stride = 2
        needed = 1 + (num_frames - 1) * stride
        if needed <= total_slices:
            s = random.randint(0, total_slices - needed)
            return [s + i * stride for i in range(num_frames)]
        return sample_depth_indices(total_slices, num_frames, "rand_contig")

    raise ValueError(f"Unknown depth mode: {mode}")


def build_tile_coords(roi_mask: np.ndarray, tile_size: int, stride: int, min_roi_frac: float = 0.05) -> List[Tuple[int, int]]:
    """
    roi_mask: 0/255 or 0/1
    returns list of (x, y) upper-left
    min_roi_frac: tile must have at least this fraction of ROI pixels
    """
    h, w = roi_mask.shape
    roi = (roi_mask > 0).astype(np.uint8)

    coords = []
    tile_area = tile_size * tile_size
    thr = int(tile_area * min_roi_frac)

    # integral image for fast sum
    integ = cv2.integral(roi)

    def rect_sum(x1, y1, x2, y2):
        # sum over [y1:y2, x1:x2]
        return integ[y2, x2] - integ[y1, x2] - integ[y2, x1] + integ[y1, x1]

    for y in range(0, h - tile_size + 1, stride):
        y2 = y + tile_size
        for x in range(0, w - tile_size + 1, stride):
            x2 = x + tile_size
            s = rect_sum(x, y, x2, y2)
            if s >= thr:
                coords.append((x, y))
    return coords


@dataclass
class VesuviusDatasetConfig:
    data_root: str = "/kaggle/input/vesuvius-challenge-ink-detection"
    split: str = "train"  # train/test
    fragment_ids: Tuple[str, ...] = ("1", "2", "3")

    tile_size: int = 64
    stride: int = 64

    # depth
    num_frames: int = 24
    depth_mode: str = "rand_contig"
    total_slices: int = 65  # expect 00..64

    # normalization
    clip_min: float = 0.0
    clip_max: float = 200.0

    # training sampling
    repeat: int = 1
    pos_prob: float = 0.5   # for seg train: probability sampling from ink-positive tiles
    min_roi_frac: float = 0.05


class VesuviusMAEPatchDataset(Dataset):
    """
    Returns only video tensor: (T, 1, H, W) for MAE pretraining.
    """
    def __init__(self, cfg: VesuviusDatasetConfig, is_train: bool = True):
        self.cfg = cfg
        self.is_train = is_train

        self.fragments = []
        self.slice_paths: Dict[str, List[str]] = {}
        self.roi_masks: Dict[str, np.ndarray] = {}
        self.coords_all: List[Tuple[str, int, int]] = []

        for fid in cfg.fragment_ids:
            fid = str(fid).replace("Frag", "")
            base = os.path.join(cfg.data_root, cfg.split, fid)
            vol_dir = os.path.join(base, "surface_volume")
            mask_path = os.path.join(base, "mask.png")

            paths = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))
            if len(paths) == 0:
                raise FileNotFoundError(vol_dir)
            self.slice_paths[fid] = paths

            roi = _load_png_gray(mask_path)
            self.roi_masks[fid] = roi

            coords = build_tile_coords(roi, cfg.tile_size, cfg.stride, cfg.min_roi_frac)
            for (x, y) in coords:
                self.coords_all.append((fid, x, y))

        if len(self.coords_all) == 0:
            raise RuntimeError("No tiles found. Check mask.png / tile_size / stride.")

        self._len = len(self.coords_all) * max(1, cfg.repeat)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        fid, x, y = self.coords_all[idx % len(self.coords_all)]
        paths = self.slice_paths[fid]
        total = len(paths)

        depth_mode = self.cfg.depth_mode if self.is_train else "center_contig"
        z_idx = sample_depth_indices(total, self.cfg.num_frames, depth_mode)

        # read patch stack: (H, W, T)
        tile = np.empty((self.cfg.tile_size, self.cfg.tile_size, len(z_idx)), dtype=np.float32)
        for i, z in enumerate(z_idx):
            arr = _read_slice_memmap(paths[z])
            patch = np.asarray(arr[y:y+self.cfg.tile_size, x:x+self.cfg.tile_size], dtype=np.float32)
            tile[..., i] = patch

        tile = _normalize_volume(tile, self.cfg.clip_min, self.cfg.clip_max)  # (H,W,T)
        # (T,1,H,W)
        video = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(1)
        return video


class VesuviusSegPatchDataset(Dataset):
    """
    Returns (video, mask, (fid, x, y))
      video: (T,1,H,W)
      mask:  (1,H,W) with {0,1,IGNORE_INDEX}
    """
    def __init__(self, cfg: VesuviusDatasetConfig, is_train: bool = True):
        self.cfg = cfg
        self.is_train = is_train

        self.slice_paths: Dict[str, List[str]] = {}
        self.roi_masks: Dict[str, np.ndarray] = {}
        self.ink_labels: Dict[str, np.ndarray] = {}

        self.coords_all: List[Tuple[str, int, int]] = []
        self.coords_pos: List[Tuple[str, int, int]] = []

        for fid in cfg.fragment_ids:
            fid = str(fid).replace("Frag", "")
            base = os.path.join(cfg.data_root, cfg.split, fid)
            vol_dir = os.path.join(base, "surface_volume")
            mask_path = os.path.join(base, "mask.png")
            ink_path = os.path.join(base, "inklabels.png")

            paths = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))
            if len(paths) == 0:
                raise FileNotFoundError(vol_dir)
            self.slice_paths[fid] = paths

            roi = _load_png_gray(mask_path)
            ink = _load_png_gray(ink_path)

            self.roi_masks[fid] = roi
            self.ink_labels[fid] = ink

            coords = build_tile_coords(roi, cfg.tile_size, cfg.stride, cfg.min_roi_frac)
            for (x, y) in coords:
                self.coords_all.append((fid, x, y))
                ink_patch = ink[y:y+cfg.tile_size, x:x+cfg.tile_size]
                if (ink_patch > 0).mean() > 0.001:
                    self.coords_pos.append((fid, x, y))

        if len(self.coords_all) == 0:
            raise RuntimeError("No tiles found. Check mask.png / tile_size / stride.")

        self._len = len(self.coords_all) * max(1, cfg.repeat)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        # oversample positives in training
        if self.is_train and len(self.coords_pos) > 0 and random.random() < self.cfg.pos_prob:
            fid, x, y = random.choice(self.coords_pos)
        else:
            fid, x, y = self.coords_all[idx % len(self.coords_all)]

        paths = self.slice_paths[fid]
        total = len(paths)

        depth_mode = self.cfg.depth_mode if self.is_train else "center_contig"
        z_idx = sample_depth_indices(total, self.cfg.num_frames, depth_mode)

        tile = np.empty((self.cfg.tile_size, self.cfg.tile_size, len(z_idx)), dtype=np.float32)
        for i, z in enumerate(z_idx):
            arr = _read_slice_memmap(paths[z])
            patch = np.asarray(arr[y:y+self.cfg.tile_size, x:x+self.cfg.tile_size], dtype=np.float32)
            tile[..., i] = patch

        tile = _normalize_volume(tile, self.cfg.clip_min, self.cfg.clip_max)
        video = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(1)  # (T,1,H,W)

        ink = self.ink_labels[fid][y:y+self.cfg.tile_size, x:x+self.cfg.tile_size]
        roi = self.roi_masks[fid][y:y+self.cfg.tile_size, x:x+self.cfg.tile_size]
        mask = (ink > 0).astype(np.uint8)
        mask[roi == 0] = IGNORE_INDEX
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()  # (1,H,W)

        return video, mask_t, (fid, x, y)
