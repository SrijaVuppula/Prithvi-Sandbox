"""
Redoes reconstruction using REAL wavelength distance instead of band
index, and compares against index-based results.
"""
import csv
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from pace_band_wavelengths import BAND_WAVELENGTH_NM

HSI_DIR = Path("cvpr_dataset/hsi")
SAMPLE_LIST = Path("hsi_sample_20.txt")
N_TRIALS = 10
MASK_FRAC = 0.20
N_BANDS = 291
rng = np.random.default_rng(42)

def contiguous_mask():
    n_mask = int(round(N_BANDS * MASK_FRAC))
    start = rng.integers(0, N_BANDS - n_mask + 1)
    return np.arange(start, start + n_mask)

def scattered_mask():
    n_mask = int(round(N_BANDS * MASK_FRAC))
    return rng.choice(N_BANDS, size=n_mask, replace=False)

def reconstruct(tile, masked_idx, x_axis):
    unmasked_idx = np.setdiff1d(np.arange(N_BANDS), masked_idx)
    order = np.argsort(x_axis[unmasked_idx])  # required: wavelength axis is NOT globally monotonic
    sorted_idx = unmasked_idx[order]
    f = interp1d(x_axis[sorted_idx], tile[:, :, sorted_idx], axis=-1,
                 kind="linear", fill_value="extrapolate")
    recon = tile.copy()
    recon[:, :, masked_idx] = f(x_axis[masked_idx])
    return recon

def metrics(gt, recon, masked_idx):
    err = recon[:, :, masked_idx] - gt[:, :, masked_idx]
    mae = np.nanmean(np.abs(err))
    rmse = np.sqrt(np.nanmean(err ** 2))
    return mae, rmse

index_axis = np.arange(N_BANDS, dtype=np.float64)
wave_axis = BAND_WAVELENGTH_NM

tile_files = [l.strip() for l in SAMPLE_LIST.read_text().splitlines() if l.strip()]
rows = []

for tf in tile_files:
    tile = np.load(Path(tf)).astype(np.float64)
    tile_id = Path(tf).stem
    for mask_type, mask_fn in [("contiguous", contiguous_mask), ("scattered", scattered_mask)]:
        for trial in range(N_TRIALS):
            masked_idx = mask_fn()
            for method, x_axis in [("index", index_axis), ("wavelength", wave_axis)]:
                recon = reconstruct(tile, masked_idx, x_axis)
                mae, rmse = metrics(tile, recon, masked_idx)
                o2a_mae = None
                hit = masked_idx[(masked_idx >= 221) & (masked_idx <= 227)]
                if len(hit):
                    o2a_mae = float(np.nanmean(np.abs(recon[:, :, hit] - tile[:, :, hit])))
                rows.append([tile_id, mask_type, trial, method, mae, rmse, o2a_mae])

with open("wavelength_vs_index_interp_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tile", "mask_type", "trial", "method", "mae", "rmse", "o2a_zone_mae"])
    w.writerows(rows)

def summarize(mask_type, method):
    vals = [r for r in rows if r[1] == mask_type and r[3] == method]
    maes = np.array([r[4] for r in vals])
    o2a = np.array([r[6] for r in vals if r[6] is not None])
    extra = f"  O2A-zone median MAE={np.nanmedian(o2a):.4f} (n={len(o2a)})" if len(o2a) else ""
    print(f"{mask_type:10s} {method:10s}  median MAE={np.nanmedian(maes):.4f}  mean MAE={np.nanmean(maes):.4f}{extra}")

print("=== Summary (compare to original: contiguous median MAE 0.0134, scattered 0.0041) ===")
for mt in ["contiguous", "scattered"]:
    for method in ["index", "wavelength"]:
        summarize(mt, method)
