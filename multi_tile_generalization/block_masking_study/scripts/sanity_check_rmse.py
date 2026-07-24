"""
Task 3 sanity check: derive RMSE from the existing post-fix PSNR results
(masked_psnr uses MSE on [0,1]-scale masked pixels, so RMSE = 10^(-PSNR/20)
is exact -- no rerun needed). Flags any rows that look broken.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("multi_tile_generalization/block_masking_study/outputs")
BACKBONES = ["tiny", "100M", "300M", "600M"]

def psnr_to_rmse(psnr):
    # masked_psnr caps at 99.0 dB when MSE <= 0 (near-perfect reconstruction)
    return np.where(psnr >= 99.0, 0.0, 10.0 ** (-psnr / 20.0))

all_rows = []
for bb in BACKBONES:
    path = RESULTS_DIR / f"results_{bb}.csv"
    df = pd.read_csv(path)
    df["block_rmse"] = psnr_to_rmse(df["block_psnr"].values)
    df["random_rmse"] = psnr_to_rmse(df["random_psnr"].values)
    all_rows.append(df)

full = pd.concat(all_rows, ignore_index=True)
print(f"Total rows: {len(full)}  (expect 40,000 across 4 backbones)")

# --- sanity flags ---
n_nan = full[["block_psnr", "random_psnr"]].isna().sum().sum()
n_neg = ((full["block_psnr"] < 0) | (full["random_psnr"] < 0)).sum()
n_capped = ((full["block_psnr"] >= 99.0) | (full["random_psnr"] >= 99.0)).sum()
n_extreme_rmse = ((full["block_rmse"] > 0.5) | (full["random_rmse"] > 0.5)).sum()

print(f"\nSanity flags:")
print(f"  NaN PSNR values:        {n_nan}")
print(f"  Negative PSNR values:   {n_neg}")
print(f"  Capped at 99.0 dB:      {n_capped}")
print(f"  RMSE > 0.5 (extreme):   {n_extreme_rmse}")

# --- summary table: mean/std RMSE per backbone x ratio x masker ---
summary = full.groupby(["backbone", "mask_ratio"]).agg(
    block_rmse_mean=("block_rmse", "mean"),
    block_rmse_std=("block_rmse", "std"),
    random_rmse_mean=("random_rmse", "mean"),
    random_rmse_std=("random_rmse", "std"),
).reset_index()

print("\nRMSE summary (mean +/- std), by backbone and ratio:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

out_path = RESULTS_DIR / "sanity_check_rmse_summary.csv"
summary.to_csv(out_path, index=False)
print(f"\nSaved summary -> {out_path}")

out_full_path = RESULTS_DIR / "sanity_check_rmse_full.csv"
full.to_csv(out_full_path, index=False)
print(f"Saved full per-row RMSE (all 40,000 rows) -> {out_full_path}")
