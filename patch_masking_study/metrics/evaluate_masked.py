"""
patch_masking_study/metrics/evaluate_masked.py

Extends the baseline evaluate.py with masked-region-only metrics.
For each metric we compute two versions:
  - global:  over the entire reconstructed image (same as baseline)
  - masked:  only over the pixels that were actually hidden from the model

The masked-region metric is the honest one — it measures how well the model
fills in what it did not see, rather than averaging over pixels it already had.
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ski_ssim
from pathlib import Path
import sys

# Reuse denorm from baseline evaluate.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline_study.metrics.evaluate import denorm_all_bands


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


def tokens_to_pixel_mask(
    masked_token_indices: torch.Tensor,
    patch_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Convert masked token indices into a (1, H, W) binary pixel mask.
    A pixel is marked as masked if its corresponding token was masked.

    Returns:
        pixel_mask: (1, H, W) float32 — 1.0 where masked, 0.0 elsewhere
    """
    tokens_per_frame = (H // patch_size) * (W // patch_size)
    patches_per_row  = W // patch_size

    pixel_mask = torch.zeros((1, H, W), dtype=torch.float32)

    for tok_idx in masked_token_indices.cpu().tolist():
        spatial_idx = tok_idx % tokens_per_frame
        row = spatial_idx // patches_per_row
        col = spatial_idx % patches_per_row

        r0, r1 = row * patch_size, (row + 1) * patch_size
        c0, c1 = col * patch_size, (col + 1) * patch_size
        pixel_mask[0, r0:r1, c0:c1] = 1.0

    return pixel_mask


def compute_masked_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    pixel_mask: torch.Tensor,
    patch_size: int,
) -> float:
    """
    Compute SSIM only over masked patches.

    Instead of using a bounding box (which covers nearly the whole image
    when patches are scattered), we extract each masked patch individually
    and compute SSIM patch-by-patch, then average.

    Args:
        pred:       (C, H, W) denormalized prediction
        gt:         (C, H, W) denormalized ground truth
        pixel_mask: (1, H, W) binary mask — 1.0 where masked
        patch_size: spatial patch size

    Returns:
        Mean SSIM across all masked patches and all bands
    """
    C, H, W = pred.shape
    patches_per_row = W // patch_size
    patches_per_col = H // patch_size

    pred_np = pred.numpy()
    gt_np   = gt.numpy()
    mask_np = pixel_mask[0].numpy()

    ssim_scores = []

    for row in range(patches_per_col):
        for col in range(patches_per_row):
            r0, r1 = row * patch_size, (row + 1) * patch_size
            c0, c1 = col * patch_size, (col + 1) * patch_size

            # Only process this patch if it was masked
            if mask_np[r0, c0] == 0.0:
                continue

            for b in range(C):
                pred_patch = pred_np[b, r0:r1, c0:c1]
                gt_patch   = gt_np[b,   r0:r1, c0:c1]
                score = ski_ssim(pred_patch, gt_patch, data_range=1.0)
                ssim_scores.append(score)

    if len(ssim_scores) == 0:
        return float("nan")

    return float(np.mean(ssim_scores))


def evaluate_reconstruction_with_masked_metrics(
    pred_frame: torch.Tensor,
    gt_frame: torch.Tensor,
    mean: list[float],
    std: list[float],
    masked_token_indices: torch.Tensor,
    patch_size: int,
    T: int,
) -> dict:
    """
    Full metric suite: global + masked-region-only.

    Args:
        pred_frame:            (C, H, W) normalized model output
        gt_frame:              (C, H, W) normalized ground truth
        mean, std:             HLS normalization constants
        masked_token_indices:  1D tensor of token indices that were masked
        patch_size:            spatial patch size
        T:                     number of frames in sequence

    Returns:
        dict with keys:
            mae, psnr, ssim              (global — entire image)
            masked_mae, masked_psnr, masked_ssim  (masked region only)
    """
    pred = denorm_all_bands(pred_frame, mean, std)
    gt   = denorm_all_bands(gt_frame,   mean, std)

    C, H, W = pred.shape

    # ── Global metrics ────────────────────────────────────────────────────────
    global_mae  = round(compute_mae(pred, gt),  6)
    global_psnr = round(compute_psnr(pred, gt), 4)
    global_ssim = round(compute_ssim(pred, gt), 6)

    # ── Masked-region-only metrics ────────────────────────────────────────────
    pixel_mask = tokens_to_pixel_mask(
        masked_token_indices,
        patch_size=patch_size,
        H=H, W=W,
    )  # (1, H, W)

    mask_chw    = pixel_mask.expand(C, H, W)
    masked_pred = pred[mask_chw == 1]
    masked_gt   = gt[mask_chw == 1]

    if masked_pred.numel() == 0:
        masked_mae  = float("nan")
        masked_psnr = float("nan")
        masked_ssim = float("nan")
    else:
        masked_mae  = round(F.l1_loss(masked_pred, masked_gt).item(), 6)
        mse = F.mse_loss(masked_pred, masked_gt).item()
        masked_psnr = round(10 * np.log10(1.0 / mse) if mse > 0 else float("inf"), 4)
        masked_ssim = round(compute_masked_ssim(pred, gt, pixel_mask, patch_size), 6)

    return {
        "mae":         global_mae,
        "psnr":        global_psnr,
        "ssim":        global_ssim,
        "masked_mae":  masked_mae,
        "masked_psnr": masked_psnr,
        "masked_ssim": masked_ssim,
    }
