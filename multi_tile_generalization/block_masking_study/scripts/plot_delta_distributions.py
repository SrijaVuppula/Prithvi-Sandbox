"""
plot_delta_distributions.py
---------------------------
For each mask ratio (20/40/60/80%), plots the distribution of
delta = random_psnr - block_psnr across all chips.

One figure, 4 subplots (one per ratio), all 4 backbones overlaid.
x-axis: delta (random - block). Positive = random harder, Negative = block harder.
Vertical dashed line at x=0 separates the two regimes.

Run from repo root:
    python multi_tile_generalization/block_masking_study/scripts/plot_delta_distributions.py
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[3]
CHIPS_DIR    = REPO_ROOT / "multi_tile_generalization" / "training_chips"
BLOCK_DIR    = REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "outputs"
RANDOM_DIR   = REPO_ROOT / "multi_tile_generalization" / "outputs" / "per_tile"
OUT_DIR      = BLOCK_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES    = ["tiny", "100M", "300M", "600M"]
COLORS       = {"tiny": "#4878CF", "100M": "#D65F5F", "300M": "#6ACC65", "600M": "#B47CC7"}
SEED         = 42
N_CHIPS      = 500
BLOCK_RATIOS = [0.2, 0.4, 0.6, 0.8]
RATIO_LABELS = {0.2: "20% masked", 0.4: "40% masked",
                0.6: "60% masked", 0.8: "80% masked"}

# ── Reconstruct chip_idx → filename mapping ───────────────────────────────────
all_files   = sorted(CHIPS_DIR.glob("chip_*_merged.tif"))
rng         = random.Random(SEED)
sampled     = rng.sample(all_files, min(N_CHIPS, len(all_files)))
idx_to_chip = {i: f.name for i, f in enumerate(sampled)}
chip_to_idx = {v: k for k, v in idx_to_chip.items()}

# ── Load block results (mean across 5 trials) ─────────────────────────────────
block_frames = []
for bb in BACKBONES:
    df  = pd.read_csv(BLOCK_DIR / f"results_{bb}.csv")
    agg = (df.groupby(["backbone", "chip", "mask_ratio"])["block_psnr"]
             .mean().reset_index()
             .rename(columns={"block_psnr": "block_psnr_mean"}))
    block_frames.append(agg)
block_df = pd.concat(block_frames, ignore_index=True)
block_df["chip_idx"] = block_df["chip"].map(chip_to_idx)

# ── Load random results ────────────────────────────────────────────────────────
random_frames = []
for bb in BACKBONES:
    df = pd.read_csv(RANDOM_DIR / f"{bb}_results.csv")
    df = df[df["mask_ratio"].isin(BLOCK_RATIOS)][["chip_idx","backbone","mask_ratio","masked_psnr"]]
    df.rename(columns={"masked_psnr": "random_psnr"}, inplace=True)
    random_frames.append(df)
random_df = pd.concat(random_frames, ignore_index=True)

# ── Merge and compute delta ────────────────────────────────────────────────────
merged = pd.merge(block_df, random_df, on=["chip_idx","backbone","mask_ratio"], how="inner")
merged["delta"] = merged["random_psnr"] - merged["block_psnr_mean"]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
fig.suptitle("Distribution of  Δ = Random PSNR − Block PSNR  per Chip\n"
             "Peak left of 0 → block harder   |   Peak right of 0 → random harder",
             fontsize=13, y=1.02)

for ax, ratio in zip(axes, BLOCK_RATIOS):
    sub = merged[merged["mask_ratio"] == ratio]

    x_all  = sub["delta"].values
    x_min  = np.percentile(x_all, 1) - 0.5
    x_max  = np.percentile(x_all, 99) + 0.5
    x_grid = np.linspace(x_min, x_max, 400)

    for bb in BACKBONES:
        vals = sub[sub["backbone"] == bb]["delta"].values
        if len(vals) < 10:
            continue
        kde = gaussian_kde(vals, bw_method=0.35)
        y   = kde(x_grid)
        ax.plot(x_grid, y, color=COLORS[bb], linewidth=2, label=bb)
        ax.fill_between(x_grid, y, alpha=0.10, color=COLORS[bb])

        # Mark mean and median
        ax.axvline(vals.mean(),      color=COLORS[bb], linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axvline(np.median(vals),  color=COLORS[bb], linewidth=1.2, linestyle=":",  alpha=0.8)

    # x=0 divider
    ax.axvline(0, color="black", linewidth=1.8, linestyle="-")

    ax.set_xlim(x_min, x_max)
    ax.set_title(RATIO_LABELS[ratio], fontsize=12, fontweight="bold")
    ax.set_xlabel("Δ = Random PSNR − Block PSNR (dB)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.text(0.02, 0.97, "← Block harder", transform=ax.transAxes,
            fontsize=8, color="grey", va="top", ha="left")
    ax.text(0.98, 0.97, "Random harder →", transform=ax.transAxes,
            fontsize=8, color="grey", va="top", ha="right")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

# Legend
backbone_handles = [mpatches.Patch(color=COLORS[bb], label=bb) for bb in BACKBONES]
line_handles = [
    plt.Line2D([0],[0], color="grey", linewidth=1.5, linestyle="--", label="Mean"),
    plt.Line2D([0],[0], color="grey", linewidth=1.5, linestyle=":",  label="Median"),
    plt.Line2D([0],[0], color="black", linewidth=2,  linestyle="-",  label="Δ = 0"),
]
fig.legend(handles=backbone_handles + line_handles,
           loc="lower center", ncol=7, fontsize=10,
           bbox_to_anchor=(0.5, -0.08), frameon=False)

plt.tight_layout()
out_path = OUT_DIR / "fig_delta_distributions.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
