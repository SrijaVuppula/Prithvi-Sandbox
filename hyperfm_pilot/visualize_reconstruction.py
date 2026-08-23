import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

N_BANDS = 291
MASK_RATIO = 0.20

def build_contiguous_mask(n_bands, mask_ratio, trial_seed):
    n_masked = round(n_bands * mask_ratio)
    rng = np.random.RandomState(trial_seed)
    start = rng.randint(0, n_bands - n_masked + 1)
    return np.arange(start, start + n_masked)

def build_scattered_mask(n_bands, mask_ratio, trial_seed):
    n_masked = round(n_bands * mask_ratio)
    rng = np.random.RandomState(trial_seed)
    return np.sort(rng.choice(n_bands, size=n_masked, replace=False))

def reconstruct_interp(arr, masked_idx, all_idx):
    unmasked_idx = np.setdiff1d(all_idx, masked_idx)
    known_vals = arr[..., unmasked_idx]
    f = interp1d(unmasked_idx, known_vals, axis=-1, kind='linear', fill_value='extrapolate')
    return f(masked_idx)

arr = np.load('cvpr_dataset/hsi/PACE_OCI.20240510T002414.L1B.V3_image_055.npy')
all_idx = np.arange(N_BANDS)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for row, (mask_type, mask_fn) in enumerate([('Contiguous', build_contiguous_mask), ('Scattered', build_scattered_mask)]):
    masked_idx = mask_fn(N_BANDS, MASK_RATIO, trial_seed=0)
    recon = reconstruct_interp(arr, masked_idx, all_idx)

    mid = len(masked_idx) // 2
    band_pos_in_mask = mid
    band = masked_idx[band_pos_in_mask]

    gt_img = arr[:, :, band]
    recon_img = recon[:, :, band_pos_in_mask]
    err_img = np.abs(gt_img - recon_img)

    vmin, vmax = np.nanmin(gt_img), np.nanmax(gt_img)

    im0 = axes[row, 0].imshow(gt_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[row, 0].set_title(f'{mask_type}: Ground Truth (band {band})')
    plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

    im1 = axes[row, 1].imshow(recon_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[row, 1].set_title(f'{mask_type}: Reconstructed')
    plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

    im2 = axes[row, 2].imshow(err_img, cmap='inferno')
    axes[row, 2].set_title(f'{mask_type}: Abs Error (MAE={np.nanmean(err_img):.4f})')
    plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

    for ax in axes[row]:
        ax.axis('off')

plt.tight_layout()
plt.savefig('reconstruction_comparison.png', dpi=150)
print("Saved reconstruction_comparison.png")
