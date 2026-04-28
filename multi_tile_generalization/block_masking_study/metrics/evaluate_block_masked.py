"""
evaluate_block_masked.py
------------------------
Computes MAE, PSNR, and SSIM for block masking experiments.
For block masking we work in PIXEL space because the masked region is one
contiguous rectangle — SSIM over that window is well-defined and meaningful.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_block_metrics(
    pred,
    target,
    pixel_mask,
    data_range: float = 1.0,
) -> dict:
    pred   = _to_numpy(pred)
    target = _to_numpy(target)
    mask   = _to_numpy(pixel_mask).astype(bool)

    C, H, W = pred.shape

    g_mae_list, g_psnr_list, g_ssim_list = [], [], []
    for c in range(C):
        p, t = pred[c], target[c]
        g_mae_list.append(float(np.mean(np.abs(t - p))))
        g_psnr_list.append(peak_signal_noise_ratio(t, p, data_range=data_range))
        g_ssim_list.append(structural_similarity(t, p, data_range=data_range))

    global_mae  = float(np.mean(g_mae_list))
    global_psnr = float(np.mean(g_psnr_list))
    global_ssim = float(np.mean(g_ssim_list))

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return {
            "global_mae": global_mae, "global_psnr": global_psnr,
            "global_ssim": global_ssim,
            "block_mae": global_mae, "block_psnr": global_psnr,
            "block_ssim": global_ssim,
            "block_h": 0, "block_w": 0, "block_area_frac": 0.0,
        }

    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    block_h = r1 - r0
    block_w = c1 - c0

    b_mae_list, b_psnr_list, b_ssim_list = [], [], []
    for c in range(C):
        p_block = pred[c, r0:r1, c0:c1]
        t_block = target[c, r0:r1, c0:c1]

        b_mae_list.append(float(np.mean(np.abs(t_block - p_block))))
        b_psnr_list.append(
            peak_signal_noise_ratio(t_block, p_block, data_range=data_range)
        )
        win = min(7, block_h, block_w)
        win = win if win % 2 == 1 else win - 1
        win = max(win, 3)
        if block_h >= win and block_w >= win:
            b_ssim_list.append(
                structural_similarity(t_block, p_block,
                                      data_range=data_range, win_size=win)
            )
        else:
            b_ssim_list.append(float("nan"))

    return {
        "global_mae":  global_mae,
        "global_psnr": global_psnr,
        "global_ssim": global_ssim,
        "block_mae":   float(np.mean(b_mae_list)),
        "block_psnr":  float(np.mean(b_psnr_list)),
        "block_ssim":  float(np.nanmean(b_ssim_list)),
        "block_h":     block_h,
        "block_w":     block_w,
        "block_area_frac": float(mask.sum()) / (H * W),
    }


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
