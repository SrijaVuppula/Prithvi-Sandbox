"""
spectral_ratio_sweep_400_new.py

Same reconstruction/metrics logic as spectral_ratio_sweep_v2.py, run only
on the 300 NEW tiles from hsi_diverse_400.txt (the original 100 already
have full results in results_zeroshot/ratio_sweep_results_v2.csv -- no
need to recompute those). Independent rng stream (fresh seed) since there
is no original to reproduce for these tiles -- just needs to be internally
reproducible and recorded.
"""
import csv
import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from pace_band_wavelengths import BAND_WAVELENGTH_NM

FULL_LIST = Path("hsi_diverse_400.txt")
EXISTING_LIST = Path("hsi_diverse_100.txt")
N_BANDS = 291
RATIOS = [0.2, 0.4, 0.6, 0.8]
N_TRIALS = 50
TAIL_BANDS = np.arange(283, 291)
O2A_BANDS = np.arange(221, 228)

OUTPUT_CSV = Path("results_zeroshot/ratio_sweep_results_400_new.csv")

rng = np.random.default_rng(777)  # independent stream, new tiles only
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

    full = [l.strip() for l in FULL_LIST.read_text().splitlines() if l.strip()]
    existing = set(l.strip() for l in EXISTING_LIST.read_text().splitlines() if l.strip())
    new_tiles = [t for t in full if t not in existing]
    print(f"400-tile list: {len(full)}  already-done (skip): {len(existing)}  "
          f"new to process: {len(new_tiles)}")

    done_keys = set()
    write_mode = "w"
    if args.resume and OUTPUT_CSV.exists():
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["tile"], row["mask_type"], row["ratio"], row["trial"]))
        write_mode = "a"
        print(f"Resuming: {len(done_keys)} rows already present")

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

    for tf in new_tiles:
        tile = np.load(Path(tf)).astype(np.float64)
        tile_id = Path(tf).stem
        rmin, rmax = float(np.nanmin(tile)), float(np.nanmax(tile))

        for ratio in RATIOS:
            for mask_type, mask_fn in [("contiguous", contiguous_mask), ("scattered", scattered_mask)]:
                for trial in range(N_TRIALS):
                    masked_idx = mask_fn(ratio)

                    key = (tile_id, mask_type, f"{ratio}", str(trial))
                    if key in done_keys:
                        continue

                    recon = reconstruct(tile, masked_idx)
                    mae, rmse, psnr = band_metrics(tile, recon, masked_idx)

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
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
