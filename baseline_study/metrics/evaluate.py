import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ski_ssim


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SSIM across all bands. Inputs: (C, H, W) in [0, 1]."""
    pred_np   = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    scores = [
        ski_ssim(pred_np[b], target_np[b], data_range=1.0)
        for b in range(pred_np.shape[0])
    ]
    return float(np.mean(scores))


def denorm_all_bands(
    frame_chw: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """
    Undo per-band HLS normalization and scale to [0, 1].

    mean/std are in raw HLS int16 reflectance units (0-10000 scale).
    After denorm we divide by 10000 to get true [0, 1] reflectance.
    """
    m = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    s = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)
    denormed = (frame_chw.cpu() * s) + m
    return (denormed / 10000.0).clamp(0, 1)


def evaluate_reconstruction(
    pred_frame: torch.Tensor,
    gt_frame: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> dict:
    """
    Full metric suite for one reconstructed frame vs. ground truth.

    Args:
        pred_frame: (C, H, W) normalized (straight from model output)
        gt_frame:   (C, H, W) normalized (straight from input tensor)
        mean, std:  HLS normalization constants for denormalization

    Returns:
        dict with keys: mae, psnr, ssim
    """
    pred = denorm_all_bands(pred_frame, mean, std)
    gt   = denorm_all_bands(gt_frame,   mean, std)

    return {
        "mae":  round(compute_mae(pred, gt),  6),
        "psnr": round(compute_psnr(pred, gt), 4),
        "ssim": round(compute_ssim(pred, gt), 6),
    }


def compute_gap_days(file_paths: list[str]) -> list[int]:
    """
    Parse acquisition dates from HLS filenames and return day-gaps between them.

    HLS filename format: ...<TILE>.<YYYYDDD>T<HHMMSS>...
    e.g. Mexico_HLS.S30.T13REM.2018026T173609.v2.0_cropped.tif
                                     ^^^^^^^  day of year 026 of 2018

    Returns:
        List of (n_frames - 1) integers: gap in days between consecutive acquisitions.
        e.g. for 4 frames → [80, 95, 65]
    """
    from datetime import datetime

    def parse_doy(filename: str) -> datetime:
        stem = Path(filename).stem          # strip .tif
        # find the segment that looks like 2018026
        for part in stem.split("."):
            if len(part) >= 7 and part[:4].isdigit() and part[4:7].isdigit():
                year = int(part[:4])
                doy  = int(part[4:7])
                return datetime(year, 1, 1) + __import__("datetime").timedelta(doy - 1)
        raise ValueError(f"Cannot parse date from filename: {filename}")

    from pathlib import Path
    dates = [parse_doy(f) for f in file_paths]
    return [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]