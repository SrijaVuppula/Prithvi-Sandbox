import numpy as np
import glob
import os
import csv
from scipy.interpolate import interp1d

MASK_RATIO = 0.20
N_BANDS = 291
N_TRIALS = 10
DATA_RANGE = 1.0

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

def compute_metrics(gt_masked_vals, recon_vals):
    err = gt_masked_vals - recon_vals
    total = err.size
    n_valid = np.sum(~np.isnan(err))
    n_excluded = total - n_valid
    mae = np.nanmean(np.abs(err))
    mse = np.nanmean(err ** 2)
    rmse = np.sqrt(mse)
    psnr = 10 * np.log10((DATA_RANGE ** 2) / mse) if mse > 0 else float('inf')
    return mae, rmse, psnr, int(n_excluded), total

def main():
    files = sorted(glob.glob('cvpr_dataset/**/*.npy', recursive=True))
    print(f"Found {len(files)} tiles")
    all_idx = np.arange(N_BANDS)
    rows = []

    for f in files:
        arr = np.load(f)
        tile_name = os.path.basename(f)
        tile_had_nan = np.isnan(arr).any()
        for trial in range(N_TRIALS):
            for mask_type, mask_fn in [('contiguous', build_contiguous_mask), ('scattered', build_scattered_mask)]:
                masked_idx = mask_fn(N_BANDS, MASK_RATIO, trial_seed=trial)
                gt_vals = arr[..., masked_idx]
                recon_vals = reconstruct_interp(arr, masked_idx, all_idx)
                mae, rmse, psnr, n_excluded, total = compute_metrics(gt_vals, recon_vals)
                rows.append({
                    'tile': tile_name,
                    'tile_had_nan': tile_had_nan,
                    'trial': trial,
                    'mask_type': mask_type,
                    'mask_ratio': MASK_RATIO,
                    'n_bands_masked': len(masked_idx),
                    'mae': mae,
                    'rmse': rmse,
                    'psnr_db': psnr,
                    'n_values_excluded_nan': n_excluded,
                    'n_values_total': total,
                })
        print(f"done: {tile_name} (had_nan={tile_had_nan})")

    out_path = 'spectral_masking_pilot_results_v2.csv'
    with open(out_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    for mask_type in ('contiguous', 'scattered'):
        maes = [r['mae'] for r in rows if r['mask_type'] == mask_type]
        rmses = [r['rmse'] for r in rows if r['mask_type'] == mask_type]
        psnrs_finite = [r['psnr_db'] for r in rows if r['mask_type'] == mask_type and r['psnr_db'] != float('inf')]
        print(f"\n{mask_type}:")
        print(f"  MAE:  mean={np.mean(maes):.5f}  median={np.median(maes):.5f}")
        print(f"  RMSE: mean={np.mean(rmses):.5f}  median={np.median(rmses):.5f}")
        print(f"  PSNR (finite, n={len(psnrs_finite)}/{len(maes)}): mean={np.mean(psnrs_finite):.2f} dB  median={np.median(psnrs_finite):.2f} dB")

if __name__ == '__main__':
    main()
