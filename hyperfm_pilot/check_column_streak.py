"""
Tests the pushbroom column-defect hypothesis across tiles: does a
fixed column show elevated error under contiguous masking, and does
it recur at the SAME column across tiles (systematic defect) vs.
random columns (tile-specific/random)? Scattered masking is the
control -- it should NOT show the same effect.
"""
import csv
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

HSI_DIR = Path("cvpr_dataset/hsi")
SAMPLE_LIST = Path("hsi_sample_20.txt")
N_TRIALS = 8
MASK_FRAC = 0.20
N_BANDS = 291
STREAK_Z_THRESH = 3.0
rng = np.random.default_rng(7)

def contiguous_mask():
    n_mask = int(round(N_BANDS * MASK_FRAC))
    start = rng.integers(0, N_BANDS - n_mask + 1)
    return np.arange(start, start + n_mask)

def scattered_mask():
    n_mask = int(round(N_BANDS * MASK_FRAC))
    return rng.choice(N_BANDS, size=n_mask, replace=False)

def reconstruct(tile, masked_idx):
    unmasked_idx = np.setdiff1d(np.arange(N_BANDS), masked_idx)
    f = interp1d(unmasked_idx, tile[:, :, unmasked_idx], axis=-1,
                 kind="linear", fill_value="extrapolate")
    recon = tile.copy()
    recon[:, :, masked_idx] = f(masked_idx)
    return recon

def column_zscores(err_map):
    col_mean = np.nanmean(err_map, axis=0)
    mu, sigma = np.nanmean(col_mean), np.nanstd(col_mean)
    return np.zeros_like(col_mean) if sigma == 0 else (col_mean - mu) / sigma

rows = []
tile_files = [l.strip() for l in SAMPLE_LIST.read_text().splitlines() if l.strip()]

for tf in tile_files:
    tile = np.load(Path(tf)).astype(np.float64)
    tile_id = Path(tf).stem
    for mask_type, mask_fn in [("contiguous", contiguous_mask), ("scattered", scattered_mask)]:
        err_maps = []
        for _ in range(N_TRIALS):
            masked_idx = mask_fn()
            recon = reconstruct(tile, masked_idx)
            err = np.abs(recon[:, :, masked_idx] - tile[:, :, masked_idx])
            err_maps.append(np.nanmean(err, axis=-1))
        mean_err_map = np.nanmean(np.stack(err_maps, axis=0), axis=0)
        z = column_zscores(mean_err_map)
        max_col = int(np.nanargmax(z))
        max_z = float(z[max_col])
        flagged = max_z > STREAK_Z_THRESH
        rows.append([tile_id, mask_type, max_col if flagged else "", max_z, flagged])
        print(f"{tile_id:20s} {mask_type:10s} max col z={max_z:6.2f}" + ("  <-- STREAK" if flagged else ""))

with open("column_streak_check_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tile", "mask_type", "streak_column", "streak_zscore", "flagged"])
    w.writerows(rows)

contig_flagged = [r for r in rows if r[1] == "contiguous" and r[4]]
scatter_flagged = [r for r in rows if r[1] == "scattered" and r[4]]
n_tiles = len(set(r[0] for r in rows))
print(f"\n{len(contig_flagged)}/{n_tiles} tiles flagged under contiguous masking")
print(f"{len(scatter_flagged)}/{n_tiles} tiles flagged under scattered masking")
if contig_flagged:
    print(f"Flagged columns (contiguous): {sorted(r[2] for r in contig_flagged)}")
    print("(repeated column index across tiles -> systematic defect; scattered/no-repeat -> tile noise)")
