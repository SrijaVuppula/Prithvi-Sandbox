import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_metrics(pred, target, mask):
    mae = float(np.mean(np.abs(pred - target)))
    pred_hwc = pred.transpose(1, 2, 0)
    target_hwc = target.transpose(1, 2, 0)
    psnr = float(psnr_fn(target_hwc, pred_hwc, data_range=1.0))
    ssim = float(ssim_fn(target_hwc, pred_hwc, data_range=1.0, channel_axis=2))
    masked_mae = _masked_mae(pred, target, mask)
    masked_psnr = _masked_psnr(pred, target, mask)
    masked_ssim = _masked_ssim_patch_by_patch(pred, target, mask)
    return {
        "mae": round(mae, 6), "psnr": round(psnr, 4), "ssim": round(ssim, 6),
        "masked_mae": round(masked_mae, 6), "masked_psnr": round(masked_psnr, 4),
        "masked_ssim": round(masked_ssim, 6),
    }


def _masked_mae(pred, target, mask):
    diff = np.abs(pred - target)
    mask_3d = np.broadcast_to(mask[np.newaxis, :, :], pred.shape)
    return float(diff[mask_3d].mean())


def _masked_psnr(pred, target, mask):
    mask_3d = np.broadcast_to(mask[np.newaxis, :, :], pred.shape)
    mse = float(np.mean((pred[mask_3d] - target[mask_3d]) ** 2))
    if mse == 0:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def _masked_ssim_patch_by_patch(pred, target, mask, patch_size=16):
    H, W = mask.shape
    scores = []
    pred_hwc = pred.transpose(1, 2, 0)
    target_hwc = target.transpose(1, 2, 0)
    for row in range(0, H, patch_size):
        for col in range(0, W, patch_size):
            if not mask[row:row+patch_size, col:col+patch_size].any():
                continue
            p_pred = pred_hwc[row:row+patch_size, col:col+patch_size, :]
            p_target = target_hwc[row:row+patch_size, col:col+patch_size, :]
            if p_pred.shape[0] < 7 or p_pred.shape[1] < 7:
                continue
            scores.append(float(ssim_fn(p_target, p_pred, data_range=1.0, channel_axis=2)))
    return float(np.mean(scores)) if scores else 0.0
