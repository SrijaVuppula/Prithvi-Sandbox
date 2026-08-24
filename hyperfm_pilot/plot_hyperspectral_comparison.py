"""
Ground-truth / reconstructed / per-patch error table, styled to match
the earlier spatial-masking figures (dark bg, hot colormap, annotated
cells, PSNR in title). Since masking here is SPECTRAL (bands), not
spatial, the old cyan spatial-block boundary is replaced with a strip
showing which bands are masked -- there is no spatial region to box.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from pace_band_wavelengths import BAND_WAVELENGTH_NM

TILE_PATH = Path("cvpr_dataset/hsi/PACE_OCI.20250220T011900.L1B.V3_image_080.npy")
N_BANDS = 291
RATIO = 0.6
PATCH = 12  # 96/12 = 8x8 patch grid

rng = np.random.default_rng(99)
wave_axis = BAND_WAVELENGTH_NM

def contiguous_mask(frac):
    n_mask = int(round(N_BANDS * frac))
    start = rng.integers(0, N_BANDS - n_mask + 1)
    return np.arange(start, start + n_mask)

def scattered_mask(frac):
    n_mask = int(round(N_BANDS * frac))
    return rng.choice(N_BANDS, size=n_mask, replace=False)

def reconstruct(tile, masked_idx):
    unmasked_idx = np.setdiff1d(np.arange(N_BANDS), masked_idx)
    order = np.argsort(wave_axis[unmasked_idx])
    sorted_idx = unmasked_idx[order]
    f = interp1d(wave_axis[sorted_idx], tile[:, :, sorted_idx], axis=-1,
                 kind="linear", fill_value="extrapolate")
    recon = tile.copy()
    recon[:, :, masked_idx] = f(wave_axis[masked_idx])
    return recon

def psnr_of(gt, recon, masked_idx):
    err = recon[:, :, masked_idx] - gt[:, :, masked_idx]
    rmse = np.sqrt(np.nanmean(err ** 2))
    return 10 * np.log10(1.0 / (rmse ** 2)) if rmse > 0 else np.inf

def false_color(tile):
    targets_nm = [650, 550, 470]  # R, G, B
    idxs = [int(np.argmin(np.abs(wave_axis - t))) for t in targets_nm]
    channels = []
    for i in idxs:
        band = tile[:, :, i]
        lo, hi = np.nanpercentile(band, [2, 98])
        band = np.clip((band - lo) / (hi - lo + 1e-9), 0, 1)
        channels.append(band)
    return np.stack(channels, axis=-1)

def patch_error_table(gt, recon, masked_idx):
    err = np.nanmean(np.abs(recon[:, :, masked_idx] - gt[:, :, masked_idx]), axis=-1)
    n = 96 // PATCH
    table = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            table[i, j] = np.nanmean(err[i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH])
    return table

tile = np.load(TILE_PATH).astype(np.float64)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#0d0d0d")
fig.suptitle(f"Spectral masking — mask ratio {int(RATIO*100)}% | Random (top) vs Block (bottom)\n"
             "No spatial boundary to draw -- masking hides WAVELENGTHS, not a pixel region. "
             "Bottom strip on each error table shows which of the 291 bands were masked.",
             color="white", fontsize=10)

for row, (mask_type, mask_fn) in enumerate([("scattered", scattered_mask), ("contiguous", contiguous_mask)]):
    masked_idx = mask_fn(RATIO)
    recon = reconstruct(tile, masked_idx)
    psnr = psnr_of(tile, recon, masked_idx)

    ax_gt, ax_recon, ax_err = axes[row]
    for ax in (ax_gt, ax_recon, ax_err):
        ax.set_facecolor("#0d0d0d")

    ax_gt.imshow(false_color(tile))
    ax_gt.set_title("Ground truth (false color, R=650/G=550/B=470nm)", color="white", fontsize=9)
    ax_gt.axis("off")

    ax_recon.imshow(false_color(recon))
    ax_recon.set_title(f"Reconstructed | {mask_type.capitalize()} — PSNR: {psnr:.1f} dB",
                        color="#FFD54A", fontsize=9)
    ax_recon.axis("off")

    table = patch_error_table(tile, recon, masked_idx)
    im = ax_err.imshow(table, cmap="hot", vmin=0, vmax=0.08)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax_err.text(j, i, f"{table[i,j]:.3f}", ha="center", va="center",
                        color="white" if table[i, j] < 0.05 else "black", fontsize=6)
    ax_err.set_title(f"Per-patch mean |GT-Recon| ({PATCH}x{PATCH}px cells)", color="white", fontsize=9)
    ax_err.set_xticks([]); ax_err.set_yticks([])
    fig.colorbar(im, ax=ax_err, shrink=0.8)

    # spectral mask strip beneath the error table, in place of a spatial boundary box
    strip = np.zeros((1, N_BANDS))
    strip[0, masked_idx] = 1
    strip_ax = fig.add_axes([ax_err.get_position().x0, ax_err.get_position().y0 - 0.035,
                              ax_err.get_position().width, 0.02])
    strip_ax.imshow(strip, cmap="hot", aspect="auto", vmin=0, vmax=1)
    strip_ax.set_xticks([]); strip_ax.set_yticks([])
    strip_ax.set_facecolor("#0d0d0d")

fig.savefig("spectral_gt_recon_errortable.png", dpi=150, facecolor="#0d0d0d", bbox_inches="tight")
print("Saved spectral_gt_recon_errortable.png")
