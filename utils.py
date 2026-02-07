import glob
import os
import cv2
from PIL import Image, ImageOps
import tifffile as tiff
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import wandb
from scipy.ndimage import zoom


import traceback

import numpy as np
from scipy.interpolate import CubicSpline

def create_lut(control_points, bit_depth=8):
    """
    Create a lookup table (LUT) based on control points for the given bit depth.

    Args:
        control_points (list of tuples): Control points in 8-bit range, e.g. [(0,0), (128,64), (255,255)]
        bit_depth (int): Bit depth of the image (8 or 16)

    Returns:
        np.ndarray: The generated LUT as a 1D numpy array.
    """
    if bit_depth == 16:
        # Scale control points from 8-bit to 16-bit (0-255 -> 0-65535)
        control_points = [(x * 257, y * 257) for x, y in control_points]
        x_points, y_points = zip(*control_points)
        x_range = np.linspace(0, 65535, 65536)
        spline = CubicSpline(x_points, y_points)
        lut = spline(x_range)
        lut = np.clip(lut, 0, 65535).astype('uint16')
    elif bit_depth == 8:
        x_points, y_points = zip(*control_points)
        x_range = np.linspace(0, 255, 256)
        spline = CubicSpline(x_points, y_points)
        lut = spline(x_range)
        lut = np.clip(lut, 0, 255).astype('uint8')
    else:
        raise ValueError("Unsupported bit depth: {}".format(bit_depth))
    return lut

def apply_lut_to_stack(img_stack, lut):
    """
    Apply a LUT to a 3D grayscale image stack (H, W, D).

    Args:
        img_stack (np.ndarray): Input stack with shape (H, W, D), dtype uint8.
        lut (np.ndarray): Lookup table of shape (256,) for 8-bit images.

    Returns:
        np.ndarray: Contrast-adjusted image stack.
    """
    if img_stack.dtype != np.uint8:
        raise ValueError("Image stack must be of dtype uint8.")
    if lut.shape[0] != 256:
        raise ValueError("LUT must have 256 values for 8-bit images.")

    # Apply LUT using NumPy fancy indexing — fast and vectorized
    return lut[img_stack]


def read_image_mask(fragment_id, CFG=None):
    """ 
    Reads a fragment image and its corresponding masks.    
    """

    control_points = [(0, 0), (128, 64), (255, 255)]

    # Create the LUT
    lut = create_lut(control_points, bit_depth=8)

    
    images = []
    start_idx = CFG.start_idx 
    end_idx = start_idx + CFG.in_chans
    
    idxs = range(start_idx, end_idx)
    image_shape = 0
    
    if fragment_id==CFG.valid_id:
        print('valid')
        start_idx = int(start_idx+(CFG.in_chans-CFG.valid_chans)//2)
        end_idx = start_idx + CFG.valid_chans
        idxs = range(start_idx, end_idx)
        print(start_idx)
    
    try:

        for i in tqdm(idxs):
            tif_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.tif")
            jpg_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.jpg")
            png_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.png") 
            
            if os.path.exists(tif_path):
                image = cv2.imread(tif_path, 0)
            elif os.path.exists(jpg_path):
                image = cv2.imread(jpg_path, 0)
            else:
                image = cv2.imread(png_path, 0)
            
            # Resize the image to match the expected size
            if (any(sub in fragment_id for sub in CFG.frags_ratio1)):
                scale = 1 / CFG.ratio1
                new_w = int(image.shape[1] * scale) 
                new_h = int(image.shape[0] * scale) 
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif (any(sub in fragment_id for sub in CFG.frags_ratio2)):
                scale = 1 / CFG.ratio2
                new_w = int(image.shape[1] * scale)
                new_h = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                scale = 1 / 1
                new_w = int(image.shape[1] * scale)
                new_h = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image_shape = (image.shape[1], image.shape[0])
                
            # pad0 = (CFG.size - image.shape[0] % CFG.size) % CFG.size
            # pad1 = (CFG.size - image.shape[1] % CFG.size) % CFG.size
            # image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
           
            image=np.clip(image,0,200)
            images.append(image)

        images = np.stack(images, axis=2)
        # images = apply_lut_to_stack(images, lut)
        print(f" Shape of {fragment_id} segment: {images.shape}")
        # if fragment_id == '20231024093300':
        #     images=images[:,:,::-1]
        # if fragment_id == CFG.valid_id:
        #     print('interpolate valid chans')
        #     # import numpy as np
        #     # from scipy.ndimage import zoom

        #     # Suppose images has shape (H, W, D)
        #     H, W, D = images.shape

        #     # Compute zoom factors
        #     zoom_factors = (1, 1, 16 / D)

        #     # Interpolate along depth (third axis)
        #     images = zoom(images, zoom_factors, order=1)  # order=1 → linear interpolation

        mask = np.zeros(images.shape[:2], dtype=np.uint8)  # shape = (H, W)
        fragment_mask = np.zeros(images.shape[:2], dtype=np.uint8)  # shape = (H, W)
        # # READ INK LABELS
        inklabel_files = glob.glob(f"{CFG.segment_path}/{fragment_id}/*inklabels.*")
        
        if len(inklabel_files) > 0:
            mask = cv2.imread(inklabel_files[0], 0)
        else:
            # print(f"Creating empty mask for {fragment_id}")
            # mask = np.zeros(images.shape[:2])
        
            mask = np.zeros(images.shape[:2], dtype=np.uint8)

            # # Build save path (same folder as images)
            save_dir = f"{CFG.segment_path}/{fragment_id}"

            save_path = os.path.join(save_dir, f"{fragment_id}_inklabels.png")

            # Save mask
            cv2.imwrite(save_path, mask)
            print(f"Saved empty mask to: {save_path}")

            
        mask =  cv2.resize(mask , image_shape, interpolation=cv2.INTER_AREA)
    
        path = f"{CFG.segment_path}{fragment_id}/{fragment_id}_mask.png"
        fragment_mask = cv2.imread(path,0)
            

        fragment_mask =  cv2.resize(fragment_mask , image_shape, interpolation=cv2.INTER_AREA)
        # pad0 = (CFG.size - fragment_mask.shape[0] % CFG.size) % CFG.size
        # pad1 = (CFG.size - fragment_mask.shape[1] % CFG.size) % CFG.size
        # fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

        # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
        
        
        mask = mask.astype('float32')
        mask/=255
        fragment_mask = fragment_mask.astype('float32')/255
        
            
    except Exception as e:
        print(f"Error processing fragment {fragment_id}: {e}")
        traceback.print_exc()

            
    return images, mask, fragment_mask

def get_train_valid_dataset(CFG=None):
    train_images = []
    train_masks = []
    
    valid_images = []
    valid_masks = []
    valid_xyxys = []
    
    segments = CFG.segments  # List of segment IDs (subdirectory names or file prefixes)
    path = CFG.segment_path  # Path to the directory containing all segments
    
    for fragment_id in segments:
        fragment_path = os.path.join(path, fragment_id)

        if not os.path.isdir(fragment_path):
            continue  # Skip

        print('reading', fragment_id)

        image, mask, fragment_mask = read_image_mask(fragment_id, CFG=CFG)

        x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, CFG.stride))
        y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, CFG.stride))
        windows_dict = {}

        for a in y1_list:
            for b in x1_list:
                if np.mean(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]) >= 1:
                                    
                    if fragment_id == CFG.valid_id or not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.05):
                        for yi in range(0, CFG.tile_size, CFG.size):
                            for xi in range(0, CFG.tile_size, CFG.size):
                                y1 = a + yi
                                x1 = b + xi
                                y2 = y1 + CFG.size
                                x2 = x1 + CFG.size
                                if fragment_id != CFG.valid_id :
                                    if (y1, y2, x1, x2) not in windows_dict:
                                        train_images.append(image[y1:y2, x1:x2])
                                        train_masks.append(mask[y1:y2, x1:x2, None])
                                        assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)
                                        windows_dict[(y1, y2, x1, x2)] = '1'
                                else:
                                    if (y1, y2, x1, x2) not in windows_dict:
                                        valid_images.append(image[y1:y2, x1:x2])
                                        valid_masks.append(mask[y1:y2, x1:x2, None])
                                        valid_xyxys.append([x1, y1, x2, y2])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.valid_chans)
                                        windows_dict[(y1, y2, x1, x2)] = '1'

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys



import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import albumentations as A


class VideoDataset(Dataset):

    def __init__(
        self,
        images,
        cfg,
        xyxys=None,
        labels=None,
        transform=None,
        norm=True,
        aug=None,
        out_chans=1,
        scale_factor=8
    ):
        """
        images: tensor [N, C, H, W]  OR  list of images
        cfg: cfg object with fields:
            - size
            - in_chans
            - valid_chans
            - out_chans (number of output video channels)
        xyxys: used for validation
        labels: masks
        transform: Albumentations transform
        norm: if True → use 3D normalization
        aug: augmentation choice: 'fourth', 'shuffle', None
        """
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys = xyxys
        self.aug = aug
        self.out_chans = out_chans

        self.scale_factor = scale_factor
        # ---------------------- #
        #   MAIN VIDEO TRANSFORM
        # ---------------------- #
        t_list = [T.ConvertImageDtype(torch.float32)]

        if norm:
            out_ch = self.out_chans

            if out_ch == 3:
                # Standard RGB ImageNet normalization
                mean = [0.485, 0.456, 0.406]
                std  = [0.229, 0.224, 0.225]

            elif out_ch == 1:
                # Grayscale normalization
                mean = [0.5]
                std  = [0.5]

            else:
                # Generic N-channel normalization
                mean = [0.5] * out_ch
                std  = [0.5] * out_ch

            t_list.append(T.Normalize(mean=mean, std=std))

        self.video_transform = T.Compose(t_list)

    def __len__(self):
        return len(self.images)

    # ------------------------------------------------------------ #
    #                      AUGMENTATIONS
    # ------------------------------------------------------------ #
    def fourth_augment(self, image):
        """
        Randomly crop K contiguous channels and zero out random others.
        """
        C = self.cfg.in_chans
        K = self.cfg.valid_chans

        # pick contiguous crop
        start_idx = random.randint(0, C - K)
        crop_idx = np.arange(start_idx, start_idx + K)
        cropped = image[..., crop_idx].copy()

        # randomly zero out some channels inside the cropped block
        zero_mask = np.random.rand(K) < 0.03 # 5% chance per channel
        cropped[..., zero_mask] = 0

        return cropped

    def shuffle_channels(self, image):
        """Random channel shuffle, returns exactly valid_chans."""
        K = self.cfg.valid_chans
        perm = np.random.permutation(K)
        return image[..., perm]

    # ------------------------------------------------------------ #
    #                 APPLY AUGMENTATION CHOICE
    # ------------------------------------------------------------ #

    def apply_aug(self, image):
        # if self.aug == "fourth":
        image = self.fourth_augment(image)
        if self.aug == "shuffle":
            image = self.shuffle_channels(image)
        return image

    # ------------------------------------------------------------ #
    #                      MAIN GETITEM
    # ------------------------------------------------------------ #

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx] if self.labels is not None else None

        # VALID CASE
        if self.xyxys is not None:
            xy = self.xyxys[idx]

            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data["image"].unsqueeze(0)
                label = data["mask"]
                if label is not None:
                    # ToTensorV2 usually returns (H, W) for masks; Dice/BCE expect (C, H, W).
                    if label.dim() == 2:
                        label = label.unsqueeze(0)  # (1, H, W)
                    # interpolate expects (N, C, H, W)
                    label = label.unsqueeze(0).float()  # (1, C, H, W)
                    label = F.interpolate(
                        label,
                        size=(self.cfg.size // self.scale_factor, self.cfg.size // self.scale_factor),
                        mode="nearest",
                    )
                    label = label.squeeze(0)  # (C, h, w)

            # permute to [frames, C, H, W]
            image = image.permute(1, 0, 2, 3)

            # convert frames
            image = torch.stack([self.video_transform(f) for f in image])

            # repeat channels if needed
            if image.shape[1] != self.out_chans:
                image = image.repeat(1, self.out_chans, 1, 1)

            return image, label, xy

        # TRAIN CASE
        else:
            # apply augmentation
            image = self.apply_aug(image)

            # apply albumentations
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data["image"].unsqueeze(0)
                label = data["mask"]
                if label is not None:
                    # ToTensorV2 usually returns (H, W) for masks; Dice/BCE expect (C, H, W).
                    if label.dim() == 2:
                        label = label.unsqueeze(0)  # (1, H, W)
                    # interpolate expects (N, C, H, W)
                    label = label.unsqueeze(0).float()  # (1, C, H, W)
                    label = F.interpolate(
                        label,
                        size=(self.cfg.size // self.scale_factor, self.cfg.size // self.scale_factor),
                        mode="nearest",
                    )
                    label = label.squeeze(0)  # (C, h, w)

            # permute → video frames
            image = image.permute(1, 0, 2, 3)

            # apply transforms frame-wise
            image = torch.stack([self.video_transform(f) for f in image])

            # repeat channels
            if image.shape[1] != self.out_chans:
                image = image.repeat(1, self.out_chans, 1, 1)

            return image, label


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug
    
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    
def make_dirs(cfg):
    for dir in [cfg.model_dir]:
        os.makedirs(dir, exist_ok=True)
        
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)