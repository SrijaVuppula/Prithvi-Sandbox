"""
build_sanity_check.py
---------------------
Merges per-chip block masking results with random masking results,
computes delta = random_psnr - block_psnr per chip per ratio,
and exports a structured Excel workbook for the professor sanity check.

Run from repo root:
    python multi_tile_generalization/block_masking_study/scripts/build_sanity_check.py
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[3]
CHIPS_DIR    = REPO_ROOT / "multi_tile_generalization" / "training_chips"
BLOCK_DIR    = REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "outputs"
RANDOM_DIR   = REPO_ROOT / "multi_tile_generalization" / "outputs" / "per_tile"
OUT_DIR      = BLOCK_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES    = ["tiny", "100M", "300M", "600M"]
SEED         = 42
N_CHIPS      = 500
BLOCK_RATIOS = [0.2, 0.4, 0.6, 0.8]

# ── Step 1: Reconstruct chip_idx → filename mapping (same seed=42 sample) ───
all_files   = sorted(CHIPS_DIR.glob("chip_*_merged.tif"))
rng         = random.Random(SEED)
sampled     = rng.sample(all_files, min(N_CHIPS, len(all_files)))
idx_to_chip = {i: f.name for i, f in enumerate(sampled)}
chip_to_idx = {v: k for k, v in idx_to_chip.items()}

unmapped_check = None  # filled after block load

# ── Step 2: Load block results, average across 5 trials per chip/ratio ───────
block_frames = []
for bb in BACKBONES:
    csv = BLOCK_DIR / f"results_{bb}.csv"
    df  = pd.read_csv(csv)
    agg = (df.groupby(["backbone", "chip", "mask_ratio"])["block_psnr"]
             .mean()
             .reset_index()
             .rename(columns={"block_psnr": "block_psnr_mean"}))
    block_frames.append(agg)

block_df = pd.concat(block_frames, ignore_index=True)
block_df["chip_idx"] = block_df["chip"].map(chip_to_idx)

unmapped = block_df["chip_idx"].isna().sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} block rows could not be mapped to a chip_idx — check seed consistency")

# ── Step 3: Load random results, keep only overlapping ratios ────────────────
random_frames = []
for bb in BACKBONES:
    csv = RANDOM_DIR / f"{bb}_results.csv"
    df  = pd.read_csv(csv)
    df  = df[df["mask_ratio"].isin(BLOCK_RATIOS)][["chip_idx", "backbone", "mask_ratio", "masked_psnr"]]
    df.rename(columns={"masked_psnr": "random_psnr"}, inplace=True)
    random_frames.append(df)

random_df = pd.concat(random_frames, ignore_index=True)

# ── Step 4: Merge on chip_idx + backbone + mask_ratio ────────────────────────
merged = pd.merge(
    block_df,
    random_df,
    on=["chip_idx", "backbone", "mask_ratio"],
    how="inner"
)

# Professor's sign convention: delta = random - block
# Positive = random is harder (block PSNR higher)
# Negative = block is harder (random PSNR higher)
merged["delta_random_minus_block"] = merged["random_psnr"] - merged["block_psnr_mean"]
merged["winner"] = merged["delta_random_minus_block"].apply(
    lambda x: "random_harder" if x > 0 else "block_harder"
)

merged = merged[[
    "backbone", "chip_idx", "chip", "mask_ratio",
    "random_psnr", "block_psnr_mean", "delta_random_minus_block", "winner"
]].sort_values(["backbone", "mask_ratio", "chip_idx"]).reset_index(drop=True)

print(f"Total rows: {len(merged)}")
print(merged.head(10).to_string())

# ── Step 5: Build summary with mean / median / trimmed mean ──────────────────
summary_rows = []
for bb in BACKBONES:
    for ratio in BLOCK_RATIOS:
        sub = merged[
            (merged["backbone"] == bb) & (merged["mask_ratio"] == ratio)
        ]["delta_random_minus_block"]
        n       = len(sub)
        trim_n  = int(np.floor(0.05 * n))
        trimmed = np.sort(sub.values)[trim_n: n - trim_n]
        summary_rows.append({
            "backbone":            bb,
            "mask_ratio":          ratio,
            "n_chips":             n,
            "mean_delta":          round(sub.mean(),     4),
            "median_delta":        round(sub.median(),   4),
            "trimmed_mean_delta":  round(trimmed.mean(), 4),
            "pct_block_harder":    round((sub < 0).mean() * 100, 1),
            "pct_random_harder":   round((sub > 0).mean() * 100, 1),
        })

summary_df = pd.DataFrame(summary_rows)
print("\n── Summary ──────────────────────────────────────────────────────────")
print(summary_df.to_string(index=False))

# ── Step 6: Export Excel ──────────────────────────────────────────────────────
out_path = OUT_DIR / "sanity_check_block_vs_random.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    for bb in BACKBONES:
        sub = merged[merged["backbone"] == bb].copy()
        sub.to_excel(writer, sheet_name=bb, index=False)
    merged.to_excel(writer, sheet_name="All", index=False)

print(f"\nSaved: {out_path}")
