"""
Masks 20/40/60/80% of the 291 bands, contiguous vs. scattered, across
the 20 diverse tiles, reconstructs via wavelength-based linear
interpolation, and records MAE/RMSE/PSNR per run. Also saves per-pixel
error maps (one representative tile, all 8 ratio x mask_type
combinations) for later visualization.
"""
import csv
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from pace_band_wavelengths import BAND_WAVELENGTH_NM

HSI_LIST = Path("hsi_diverse_20.txt")
SCORES_CSV = Path("hsi_diverse_20_scores.csv")
N_BANDS = 291
RATIOS = [0.2, 0.4, 0.6, 0.8]
N_TRIALS = 10
ERRORMAP_DIR = Path("errormaps")
ERRORMAP_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
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

def metrics(gt, recon, masked_idx):
    err = recon[:, :, masked_idx] - gt[:, :, masked_idx]
    mae = np.nanmean(np.abs(err))
    rmse = np.sqrt(np.nanmean(err ** 2))
    psnr = 10 * np.log10(1.0 / (rmse ** 2)) if rmse > 0 else np.inf
    return mae, rmse, psnr

tile_files = [l.strip() for l in HSI_LIST.read_text().splitlines() if l.strip()]

# pick the representative (median-variance) tile for saved error maps
scores = []
with open(SCORES_CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        scores.append((row["tile"], float(row["variance_score"])))
scores.sort(key=lambda x: x[1])
representative_tile = scores[len(scores) // 2][0]
print(f"Representative tile for error maps: {representative_tile}")

rows = []
for tf in tile_files:
    tile = np.load(Path(tf)).astype(np.float64)
    tile_id = Path(tf).stem
    is_representative = (tf == representative_tile)

    for ratio in RATIOS:
        for mask_type, mask_fn in [("contiguous", contiguous_mask), ("scattered", scattered_mask)]:
            for trial in range(N_TRIALS):
                masked_idx = mask_fn(ratio)
                recon = reconstruct(tile, masked_idx)
                mae, rmse, psnr = metrics(tile, recon, masked_idx)
                rows.append([tile_id, mask_type, ratio, trial, mae, rmse, psnr])

                if is_representative and trial == 0:
                    err_map = np.nanmean(np.abs(recon[:, :, masked_idx] - tile[:, :, masked_idx]), axis=-1)
                    np.save(ERRORMAP_DIR / f"{mask_type}_ratio{int(ratio*100)}.npy", err_map)

with open("spectral_ratio_sweep_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tile", "mask_type", "ratio", "trial", "mae", "rmse", "psnr"])
    w.writerows(rows)

print("\n=== Median PSNR / MAE by ratio and mask type ===")
for ratio in RATIOS:
    for mask_type in ["contiguous", "scattered"]:
        vals = [r for r in rows if r[1] == mask_type and r[2] == ratio]
        psnrs = np.array([r[6] for r in vals])
        maes = np.array([r[4] for r in vals])
        print(f"ratio={ratio:.1f}  {mask_type:10s}  median PSNR={np.nanmedian(psnrs):6.2f}dB"
              f"  median MAE={np.nanmedian(maes):.4f}")

print(f"\nSaved: spectral_ratio_sweep_results.csv, {len(list(ERRORMAP_DIR.glob('*.npy')))} error maps in errormaps/")
