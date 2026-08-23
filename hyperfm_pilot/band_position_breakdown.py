import numpy as np
import glob
import os
import csv
from scipy.interpolate import interp1d

N_BANDS = 291
MASK_RATIO = 0.20
N_TRIALS = 10

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

files = sorted(glob.glob('cvpr_dataset/**/*.npy', recursive=True))
all_idx = np.arange(N_BANDS)
rows = []

for f in files:
    arr = np.load(f)
    tile_name = os.path.basename(f)
    for trial in range(N_TRIALS):
        for mask_type, mask_fn in [('contiguous', build_contiguous_mask), ('scattered', build_scattered_mask)]:
            masked_idx = mask_fn(N_BANDS, MASK_RATIO, trial_seed=trial)
            gt_vals = arr[..., masked_idx]
            recon_vals = reconstruct_interp(arr, masked_idx, all_idx)
            # per-band error, averaged over the 96x96 spatial grid (nan-safe)
            err = np.abs(gt_vals - recon_vals)
            per_band_mae = np.nanmean(err, axis=(0, 1))  # shape (n_masked,)
            for band_idx, band_mae in zip(masked_idx, per_band_mae):
                rows.append({
                    'tile': tile_name,
                    'trial': trial,
                    'mask_type': mask_type,
                    'band_idx': int(band_idx),
                    'band_mae': float(band_mae),
                })
    print(f"done: {tile_name}")

out_path = 'band_position_breakdown.csv'
with open(out_path, 'w', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nWrote {len(rows)} rows to {out_path}")

# quick summary: worst 15 bands by mean error, across both mask types
import collections
band_errs = collections.defaultdict(list)
for r in rows:
    band_errs[r['band_idx']].append(r['band_mae'])

band_means = [(b, np.mean(v)) for b, v in band_errs.items()]
band_means.sort(key=lambda x: -x[1])
print("\nWorst 15 bands by mean reconstruction error (across all tiles/trials/mask types):")
for b, m in band_means[:15]:
    print(f"  band {b}: mean MAE = {m:.5f}")
