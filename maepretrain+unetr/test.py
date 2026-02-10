
import os
import glob
import argparse
from typing import List, Tuple, Optional, Dict, Any
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import tifffile
import csv
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---- your model code ----
from unetr import VideoMAEUNETR2D, load_videomae_encoder_from_mae_ckpt


def rle_encode(mask01: np.ndarray) -> str:
    """Fortran order RLE: transpose then flatten."""
    m = mask01.astype(np.uint8)
    pixels = m.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes.copy()
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _list_tif_slices(frag_root: str) -> List[str]:
    vol_dir = os.path.join(frag_root, "surface_volume")
    if not os.path.isdir(vol_dir):
        alt = os.path.join(frag_root, "surface_volumn")  # tolerate typo
        if os.path.isdir(alt):
            vol_dir = alt
    paths = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No tif slices found in: {vol_dir}")
    return paths
def write_submission_csv(path: str, ids: List[str], rles: List[str]) -> None:
    # 把所有空白（含 \n \r \t）压成一个空格，彻底消灭“拆行”
    clean = []
    for s in rles:
        s = "" if s is None else str(s)
        s = re.sub(r"\s+", " ", s).strip()
        clean.append(s)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        w.writerow(["Id", "Predicted"])
        for fid, rle in zip(ids, clean):
            w.writerow([str(fid), rle])


def _choose_z_indices(total_slices: int, num_frames: int, mode: str, start_z: Optional[int]) -> List[int]:
    if num_frames > total_slices:
        raise ValueError(f"num_frames={num_frames} > total_slices={total_slices}")
    if start_z is not None:
        s = max(0, min(int(start_z), total_slices - num_frames))
        return list(range(s, s + num_frames))
    if mode == "front_contig":
        return list(range(0, num_frames))
    s = (total_slices - num_frames) // 2
    return list(range(s, s + num_frames))


def load_mask_png(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)


def build_coords(mask01: np.ndarray, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    H, W = mask01.shape
    ys = list(range(0, max(H - tile_size + 1, 1), stride))
    xs = list(range(0, max(W - tile_size + 1, 1), stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    y_last = max(H - tile_size, 0)
    x_last = max(W - tile_size, 0)
    if ys[-1] != y_last: ys.append(y_last)
    if xs[-1] != x_last: xs.append(x_last)

    coords: List[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            if mask01[y:y+tile_size, x:x+tile_size].sum() > 0:
                coords.append((x, y))
    return coords


def normalize_patch_u16_to_model(patch_u16: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    # match your train normalization: clip -> /255 -> (x-0.5)/0.5
    x = patch_u16.astype(np.float32)
    x = np.clip(x, clip_min, clip_max)
    x = x / 255.0
    x = (x - 0.5) / 0.5
    return x.astype(np.float16)


def load_volume_slices_to_ram(slice_paths: List[str], z_idx: List[int]) -> np.ndarray:
    imgs = [tifffile.imread(slice_paths[z]) for z in z_idx]
    return np.stack(imgs, axis=0)  # (T,H,W)


class TestPatchDataset(Dataset):
    def __init__(
        self,
        vol_u16: np.ndarray,      # (T,H,W)
        coords: List[Tuple[int, int]],
        tile_size: int,
        clip_min: float,
        clip_max: float,
        cache_float16: bool = True,
    ):
        self.vol_u16 = vol_u16
        self.coords = coords
        self.tile_size = int(tile_size)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self.vol_f16 = None
        if cache_float16:
            try:
                x = vol_u16.astype(np.float32)
                x = np.clip(x, self.clip_min, self.clip_max)
                x = x / 255.0
                x = (x - 0.5) / 0.5
                self.vol_f16 = x.astype(np.float16)
            except MemoryError:
                self.vol_f16 = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx: int):
        x0, y0 = self.coords[idx]
        ts = self.tile_size

        if self.vol_f16 is not None:
            patch = self.vol_f16[:, y0:y0+ts, x0:x0+ts]  # (T,ts,ts)
        else:
            patch_u16 = self.vol_u16[:, y0:y0+ts, x0:x0+ts]
            patch = normalize_patch_u16_to_model(patch_u16, self.clip_min, self.clip_max)

        x = torch.from_numpy(patch).unsqueeze(1)  # (T,1,ts,ts)
        coord = torch.tensor([x0, y0], dtype=torch.int32)
        return x, coord


def _strip_outer_prefix(k: str) -> str:
    # do NOT strip "encoder." because it is a real submodule in your UNETR
    for p in ("model.", "net.", "module.", "pl_module.", "lit_model.", "vmae."):
        if k.startswith(p):
            return k[len(p):]
    return k


def find_ckpt_path(ckpt_dir_or_file: str) -> str:
    """
    Support:
      - a file path (xxx.pt / xxx.pth / xxx.ckpt)
      - a directory containing those files
    Preference: best > last > newest
    """
    p = ckpt_dir_or_file.rstrip("/")
    if os.path.isfile(p) and p.lower().endswith((".pt", ".pth", ".ckpt")):
        return p

    if not os.path.isdir(p):
        raise FileNotFoundError(f"ckpt_dir not found: {p}")

    exts = ("*.pt", "*.pth", "*.ckpt")
    cands = []
    for ext in exts:
        cands.extend(glob.glob(os.path.join(p, "**", ext), recursive=True))
        cands.extend(glob.glob(os.path.join(p, ext)))

    # dedup
    cands = sorted(list(set(cands)))
    if len(cands) == 0:
        raise FileNotFoundError(f"No .pt/.pth/.ckpt found under: {p}")

    def score(path: str) -> Tuple[int, int]:
        name = os.path.basename(path).lower()
        # higher is better
        s = 0
        if "best" in name: s += 30
        if "last" in name: s += 20
        if "final" in name: s += 10
        # newer is better
        mtime = int(os.path.getmtime(path))
        return (s, mtime)

    cands = sorted(cands, key=score, reverse=True)
    return cands[0]


def _extract_state_and_hp(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Handle:
      1) {"state_dict": ...}
      2) pure state_dict
      3) {"model": state_dict} / {"net": state_dict} (rare)
    """
    hp = {}
    if isinstance(ckpt_obj, dict):
        if "hyper_parameters" in ckpt_obj and isinstance(ckpt_obj["hyper_parameters"], dict):
            hp = ckpt_obj["hyper_parameters"]

        for key in ("state_dict", "model_state", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key], hp

        # if it already looks like a state dict
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj, hp

    raise TypeError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


def load_seg_model(ckpt_path: str, tile_size: int, num_frames: int, mae_ckpt: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state, hp = _extract_state_and_hp(ckpt)

    tile_size = int(hp.get("tile_size", tile_size))
    num_frames = int(hp.get("num_frames", num_frames))

    net = VideoMAEUNETR2D(tile_size=tile_size, num_frames=num_frames)

    # optional MAE fill: helpful if ckpt lacks encoder weights
    if mae_ckpt and os.path.exists(mae_ckpt):
        try:
            load_videomae_encoder_from_mae_ckpt(net.encoder, mae_ckpt)
        except Exception as e:
            print(f"[WARN] failed to load mae_ckpt='{mae_ckpt}': {e}")

    new_state = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            new_state[_strip_outer_prefix(k)] = v

    missing, unexpected = net.load_state_dict(new_state, strict=False)
    print(f"[CKPT] {ckpt_path}")
    print(f"  loaded with missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("  unexpected (first 10):", unexpected[:10])
    if missing:
        print("  missing (first 10):", missing[:10])

    return net, tile_size, num_frames


@torch.no_grad()
def predict_fragment(
    net: VideoMAEUNETR2D,
    vol_u16: np.ndarray,     # (T,H,W)
    mask01: np.ndarray,      # (H,W)
    tile_size: int,
    stride: int,
    clip_min: float,
    clip_max: float,
    batch_size: int,
    num_workers: int,
    cache_float16: bool,
    use_amp: bool,
) -> np.ndarray:
    device = next(net.parameters()).device
    coords = build_coords(mask01, tile_size=tile_size, stride=stride)
    print(f"[Predict] HxW={mask01.shape} tiles={len(coords)} tile={tile_size} stride={stride}")

    ds = TestPatchDataset(
        vol_u16=vol_u16,
        coords=coords,
        tile_size=tile_size,
        clip_min=clip_min,
        clip_max=clip_max,
        cache_float16=cache_float16,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    H, W = mask01.shape
    pred = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    for xb, cb in tqdm(dl, desc="tiles", leave=False):
        xb = xb.to(device, non_blocking=True)  # (B,T,1,ts,ts)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = net(xb)
        else:
            logits = net(xb)

        probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        cb = cb.cpu().numpy()

        for i in range(probs.shape[0]):
            x0, y0 = int(cb[i, 0]), int(cb[i, 1])
            pred[y0:y0+tile_size, x0:x0+tile_size] += probs[i]
            cnt[y0:y0+tile_size, x0:x0+tile_size] += 1.0

    pred = pred / np.maximum(cnt, 1.0)
    pred = pred * mask01.astype(np.float32)
    return pred


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/kaggle/input/vesuvius-challenge-ink-detection")
    ap.add_argument("--ckpt_dir", type=str, default="/kaggle/working/seg_outputs_run3")
    ap.add_argument("--mae_ckpt", type=str, default="/kaggle/working/mae_outputs/best_mae.pt")
    ap.add_argument("--out_csv", type=str, default="/kaggle/working/submission.csv")

    ap.add_argument("--tile_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--depth_mode", type=str, default="center_contig", choices=["center_contig", "front_contig"])
    ap.add_argument("--start_z", type=int, default=-1)

    ap.add_argument("--clip_min", type=float, default=0.0)
    ap.add_argument("--clip_max", type=float, default=200.0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--cache_float16", type=int, default=1)
    ap.add_argument("--no_amp", action="store_true")

    # notebook execution -> no argv -> use defaults
    if argv is None:
        import sys
        argv = sys.argv[1:]
        if ("ipykernel" in sys.argv[0]) or ("colab_kernel_launcher" in sys.argv[0]):
            argv = []
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    ckpt_path = find_ckpt_path(args.ckpt_dir)
    start_z = None if args.start_z < 0 else int(args.start_z)

    net, tile_size, num_frames = load_seg_model(
        ckpt_path=ckpt_path,
        tile_size=args.tile_size,
        num_frames=args.num_frames,
        mae_ckpt=args.mae_ckpt,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()

    sample_sub_path = os.path.join(args.data_root, "sample_submission.csv")
    sub = pd.read_csv(sample_sub_path)
    frag_ids = sub["Id"].tolist()

    test_root = os.path.join(args.data_root, "test")
    out_rle = []

    for fid in frag_ids:
        frag_root = os.path.join(test_root, str(fid))
        mask_path = os.path.join(frag_root, "mask.png")
        mask01 = load_mask_png(mask_path)

        slice_paths = _list_tif_slices(frag_root)
        z_idx = _choose_z_indices(len(slice_paths), num_frames, args.depth_mode, start_z)

        print(f"\n=== Fragment {fid} | slices={len(slice_paths)} use={z_idx[0]}..{z_idx[-1]} (T={num_frames}) ===")
        vol_u16 = load_volume_slices_to_ram(slice_paths, z_idx)

        prob = predict_fragment(
            net=net,
            vol_u16=vol_u16,
            mask01=mask01,
            tile_size=int(tile_size),
            stride=int(args.stride),
            clip_min=float(args.clip_min),
            clip_max=float(args.clip_max),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            cache_float16=bool(int(args.cache_float16)),
            use_amp=(not args.no_amp),
        )

        pred_bin = (prob > float(args.threshold)).astype(np.uint8)
        out_rle.append(rle_encode(pred_bin))

    sub["Id"] = sub["Id"].astype(str).str.strip()
    frag_ids = sub["Id"].tolist()

    write_submission_csv(args.out_csv, frag_ids, out_rle)

# 自检：必须只有 3 行（header + a + b）
    print("\n[CHECK] first 2 lines:")
    with open(args.out_csv, "r", encoding="utf-8") as f:
        for _ in range(2):
           print(f.readline().rstrip("\n"))

    print("[CHECK] line count =", sum(1 for _ in open(args.out_csv, "r", encoding="utf-8")))
    print(f"\n✅ saved: {args.out_csv}")
    print(sub.head())


if __name__ == "__main__":
    main()
