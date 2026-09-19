"""
spectral_ratio_sweep_v2.py

Reproduces spectral_ratio_sweep.py's exact rng stream (same seed, same
mask_fn call order) so all-band MAE/RMSE/PSNR match results_zeroshot/
ratio_sweep_results.csv row for row -- self-checked as it runs.

Adds, from the same masks/reconstructions (zero extra rng draws):
  - tail-excluded MAE/RMSE/PSNR (SWIR tail, bands 283-290, dropped from
    the masked-index error set before aggregating)
  - o2a_mae / o2a_n_masked: error on O2-A band(s) (221-227) when masked
  - tail_mae / tail_n_masked: error on tail band(s) (283-290) when masked
  - per-tile ground-truth radiance min/max

--resume: always redraws every mask (to keep the rng stream in sync)
but skips recon/metrics/writing for rows already in the output CSV.
"""
import csv
import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from pace_band_wavelengths import BAND_WAVELENGTH_NM

HSI_LIST = Path("hsi_diverse_100.txt")
N_BANDS = 291
RATIOS = [0.2, 0.4, 0.6, 0.8]
N_TRIALS = 50
TAIL_BANDS = np.arange(283, 291)   # SWIR tail -- extrapolation-only zone
O2A_BANDS = np.arange(221, 228)    # O2-A absorption feature, 759-767nm

ORIGINAL_CSV = Path("results_zeroshot/ratio_sweep_results.csv")
OUTPUT_CSV = Path("results_zeroshot/ratio_sweep_results_v2.csv")

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


def band_metrics(gt, recon, band_idx):
    if len(band_idx) == 0:
        return np.nan, np.nan, np.nan
    err = recon[:, :, band_idx] - gt[:, :, band_idx]
    mae = np.nanmean(np.abs(err))
    rmse = np.sqrt(np.nanmean(err ** 2))
    psnr = 10 * np.log10(1.0 / (rmse ** 2)) if rmse > 0 else np.inf
    return mae, rmse, psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tile_files = [l.strip() for l in HSI_LIST.read_text().splitlines() if l.strip()]

    orig_rows = {}
    with open(ORIGINAL_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["tile"], row["mask_type"], row["ratio"], row["trial"])
            orig_rows[key] = (float(row["mae"]), float(row["rmse"]))

    done_keys = set()
    write_mode = "w"
    if args.resume and OUTPUT_CSV.exists():
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["tile"], row["mask_type"], row["ratio"], row["trial"]))
        write_mode = "a"
        print(f"Resuming: {len(done_keys)} rows already present in {OUTPUT_CSV}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fh = open(OUTPUT_CSV, write_mode, newline="")
    w = csv.writer(fh)
    if write_mode == "w":
        w.writerow(["tile", "mask_type", "ratio", "trial",
                    "mae", "rmse", "psnr",
                    "mae_tailexcl", "rmse_tailexcl", "psnr_tailexcl",
                    "o2a_mae", "o2a_n_masked",
                    "tail_mae", "tail_n_masked",
                    "tile_radiance_min", "tile_radiance_max"])

    mismatches = 0
    n_checked = 0
    radiance_range_seen = {}

    for tf in tile_files:
        tile = np.load(Path(tf)).astype(np.float64)
        tile_id = Path(tf).stem
        rmin, rmax = float(np.nanmin(tile)), float(np.nanmax(tile))
        radiance_range_seen[tile_id] = (rmin, rmax)

        for ratio in RATIOS:
            for mask_type, mask_fn in [("contiguous", contiguous_mask), ("scattered", scattered_mask)]:
                for trial in range(N_TRIALS):
                    masked_idx = mask_fn(ratio)  # always draw -- keeps rng stream in sync

                    key = (tile_id, mask_type, f"{ratio}", str(trial))
                    if key in done_keys:
                        continue

                    recon = reconstruct(tile, masked_idx)
                    mae, rmse, psnr = band_metrics(tile, recon, masked_idx)

                    if key in orig_rows:
                        o_mae, o_rmse = orig_rows[key]
                        n_checked += 1
                        if not (np.isclose(mae, o_mae, rtol=1e-5, atol=1e-6) and
                                np.isclose(rmse, o_rmse, rtol=1e-5, atol=1e-6)):
                            mismatches += 1
                            print(f"MISMATCH {key}: got mae={mae},rmse={rmse} "
                                  f"vs committed mae={o_mae},rmse={o_rmse}")

                    tail_in_mask = np.intersect1d(masked_idx, TAIL_BANDS)
                    o2a_in_mask = np.intersect1d(masked_idx, O2A_BANDS)
                    excl_idx = np.setdiff1d(masked_idx, TAIL_BANDS)

                    mae_t, rmse_t, psnr_t = band_metrics(tile, recon, excl_idx)
                    o2a_mae, _, _ = band_metrics(tile, recon, o2a_in_mask)
                    tail_mae, _, _ = band_metrics(tile, recon, tail_in_mask)

                    w.writerow([tile_id, mask_type, ratio, trial,
                                mae, rmse, psnr,
                                mae_t, rmse_t, psnr_t,
                                o2a_mae, len(o2a_in_mask),
                                tail_mae, len(tail_in_mask),
                                rmin, rmax])

        fh.flush()
        print(f"done: {tile_id}  radiance range=[{rmin:.3f}, {rmax:.3f}]")

    fh.close()

    print(f"\nSelf-check vs committed results_zeroshot/ratio_sweep_results.csv: "
          f"{n_checked} rows checked, {mismatches} mismatches.")
    if n_checked == 0:
        print("WARNING: no overlapping rows found -- verify tile list / CSV path.")
    elif mismatches == 0:
        print("PASS: all-band MAE/RMSE reproduce the committed sweep exactly.")
    else:
        print("FAIL: reproduction does not match committed results. Do not trust output.")

    all_min = min(v[0] for v in radiance_range_seen.values())
    all_max = max(v[1] for v in radiance_range_seen.values())
    print(f"\nRadiance range across all {len(radiance_range_seen)} tiles: "
          f"[{all_min:.4f}, {all_max:.4f}]")


if __name__ == "__main__":
    main()
