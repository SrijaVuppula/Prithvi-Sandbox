"""
spectral_ratio_sweep_cubic.py

Reuses the EXACT same rng stream/mask draws as spectral_ratio_sweep_v2.py
(seed=42, same 100-tile list/order) so cubic and linear are compared on
identical masks -- only the interpolation kind differs. Self-checks the
linear reconstruction against the committed CSV first (confirms the mask
stream is in sync) before computing cubic on the same masked_idx.

Known from a synthetic-data test: cubic spline extrapolation blows up in
the SWIR tail (bands 283-290, no data beyond the last unmasked band to
anchor the curve) -- expect huge/negative-dB all-band PSNR. The
tail-excluded columns are the ones expected to be usable.
"""
import csv
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from pace_band_wavelengths import BAND_WAVELENGTH_NM

HSI_LIST = Path("hsi_diverse_100.txt")
N_BANDS = 291
RATIOS = [0.2, 0.4, 0.6, 0.8]
N_TRIALS = 50
TAIL_BANDS = np.arange(283, 291)
O2A_BANDS = np.arange(221, 228)

ORIGINAL_CSV = Path("results_zeroshot/ratio_sweep_results.csv")
OUTPUT_CSV = Path("results_zeroshot/ratio_sweep_results_cubic.csv")

rng = np.random.default_rng(42)  # same seed/order as v1 and v2 -- same mask stream
wave_axis = BAND_WAVELENGTH_NM


def contiguous_mask(frac):
    n_mask = int(round(N_BANDS * frac))
    start = rng.integers(0, N_BANDS - n_mask + 1)
    return np.arange(start, start + n_mask)


def scattered_mask(frac):
    n_mask = int(round(N_BANDS * frac))
    return rng.choice(N_BANDS, size=n_mask, replace=False)


def reconstruct(tile, masked_idx, kind):
    unmasked_idx = np.setdiff1d(np.arange(N_BANDS), masked_idx)
    order = np.argsort(wave_axis[unmasked_idx])
    sorted_idx = unmasked_idx[order]
    f = interp1d(wave_axis[sorted_idx], tile[:, :, sorted_idx], axis=-1,
                 kind=kind, fill_value="extrapolate")
    recon = tile.copy()
    recon[:, :, masked_idx] = f(wave_axis[masked_idx])
    return recon


def band_metrics(gt, recon, band_idx):
    if len(band_idx) == 0:
        return np.nan, np.nan, np.nan
    err = recon[:, :, band_idx] - gt[:, :, band_idx]
    mae = np.nanmean(np.abs(err))
    rmse = np.sqrt(np.nanmean(err ** 2))
    psnr = 10 * np.log10(1.0 / (rmse ** 2)) if rmse > 0 else np.inf
    return mae, rmse, psnr


def main():
    tile_files = [l.strip() for l in HSI_LIST.read_text().splitlines() if l.strip()]

    orig_rows = {}
    with open(ORIGINAL_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["tile"], row["mask_type"], row["ratio"], row["trial"])
            orig_rows[key] = (float(row["mae"]), float(row["rmse"]))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fh = open(OUTPUT_CSV, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["tile", "mask_type", "ratio", "trial",
                "cubic_mae", "cubic_rmse", "cubic_psnr",
                "cubic_mae_tailexcl", "cubic_rmse_tailexcl", "cubic_psnr_tailexcl",
                "linear_mae_forcompare", "linear_rmse_forcompare"])

    mismatches = 0
    n_checked = 0
    extreme_count = 0

    for tf in tile_files:
        tile = np.load(Path(tf)).astype(np.float64)
        tile_id = Path(tf).stem

        for ratio in RATIOS:
            for mask_type, mask_fn in [("contiguous", contiguous_mask), ("scattered", scattered_mask)]:
                for trial in range(N_TRIALS):
                    masked_idx = mask_fn(ratio)  # same draw as v1/v2 -- keeps stream in sync

                    key = (tile_id, mask_type, f"{ratio}", str(trial))

                    # self-check: linear on this same mask must match committed CSV
                    recon_lin = reconstruct(tile, masked_idx, "linear")
                    mae_lin, rmse_lin, _ = band_metrics(tile, recon_lin, masked_idx)
                    if key in orig_rows:
                        o_mae, o_rmse = orig_rows[key]
                        n_checked += 1
                        if not (np.isclose(mae_lin, o_mae, rtol=1e-5, atol=1e-6) and
                                np.isclose(rmse_lin, o_rmse, rtol=1e-5, atol=1e-6)):
                            mismatches += 1
                            print(f"MISMATCH (mask stream desynced) {key}")

                    # cubic on the SAME masked_idx
                    recon_cub = reconstruct(tile, masked_idx, "cubic")
                    mae_c, rmse_c, psnr_c = band_metrics(tile, recon_cub, masked_idx)
                    excl_idx = np.setdiff1d(masked_idx, TAIL_BANDS)
                    mae_ct, rmse_ct, psnr_ct = band_metrics(tile, recon_cub, excl_idx)

                    if not np.isfinite(psnr_c) or psnr_c < 0 or rmse_c > 1.0:
                        extreme_count += 1

                    w.writerow([tile_id, mask_type, ratio, trial,
                                mae_c, rmse_c, psnr_c,
                                mae_ct, rmse_ct, psnr_ct,
                                mae_lin, rmse_lin])

        fh.flush()
        print(f"done: {tile_id}")

    fh.close()

    print(f"\nMask-stream self-check: {n_checked} rows, {mismatches} mismatches "
          f"({'PASS' if mismatches==0 else 'FAIL -- masks not aligned with linear run'})")
    print(f"Extreme/blown-up all-band cubic results (PSNR<0, non-finite, or RMSE>1.0): "
          f"{extreme_count}/{n_checked} rows")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
