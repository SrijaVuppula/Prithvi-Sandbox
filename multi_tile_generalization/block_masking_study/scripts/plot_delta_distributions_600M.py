import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[3]
CHIPS_DIR    = REPO_ROOT / "multi_tile_generalization" / "training_chips"
BLOCK_DIR    = REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "outputs"
RANDOM_DIR   = REPO_ROOT / "multi_tile_generalization" / "outputs" / "per_tile"
OUT_DIR      = BLOCK_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED=42; N_CHIPS=500; BLOCK_RATIOS=[0.2,0.4,0.6,0.8]; COLOR="#B47CC7"

all_files   = sorted(CHIPS_DIR.glob("chip_*_merged.tif"))
rng         = random.Random(SEED)
sampled     = rng.sample(all_files, min(N_CHIPS, len(all_files)))
chip_to_idx = {f.name: i for i, f in enumerate(sampled)}

df_block = pd.read_csv(BLOCK_DIR / "results_600M.csv")
df_block = (df_block.groupby(["chip","mask_ratio"])["block_psnr"]
              .mean().reset_index()
              .rename(columns={"block_psnr": "block_psnr_mean"}))
df_block["chip_idx"] = df_block["chip"].map(chip_to_idx)

df_random = pd.read_csv(RANDOM_DIR / "600M_results.csv")
df_random = df_random[df_random["mask_ratio"].isin(BLOCK_RATIOS)][["chip_idx","mask_ratio","masked_psnr"]]
df_random.rename(columns={"masked_psnr": "random_psnr"}, inplace=True)

merged = pd.merge(df_block, df_random, on=["chip_idx","mask_ratio"], how="inner")
merged["delta"] = merged["random_psnr"] - merged["block_psnr_mean"]

all_deltas = merged["delta"].values
x_min  = np.percentile(all_deltas, 1) - 0.5
x_max  = np.percentile(all_deltas, 99) + 0.5
x_grid = np.linspace(x_min, x_max, 400)

y_max_global = 0
for ratio in BLOCK_RATIOS:
    vals = merged[merged["mask_ratio"]==ratio]["delta"].values
    y_max_global = max(y_max_global, gaussian_kde(vals, bw_method=0.35)(x_grid).max())
y_max_global *= 1.15

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("600M Backbone — Distribution of Δ = Random PSNR − Block PSNR per Chip\n"
             "Positive Δ → block masking is harder   |   Negative Δ → random masking is harder",
             fontsize=13, y=1.02)

for ax, ratio in zip(axes, BLOCK_RATIOS):
    vals     = merged[merged["mask_ratio"]==ratio]["delta"].values
    kde      = gaussian_kde(vals, bw_method=0.35)
    y        = kde(x_grid)

    mean_v   = vals.mean()
    median_v = np.median(vals)
    trim_n   = int(np.floor(0.05 * len(vals)))
    trim_v   = np.sort(vals)[trim_n: len(vals)-trim_n].mean()
    pct_block  = (vals < 0).mean() * 100
    pct_random = (vals > 0).mean() * 100

    ax.plot(x_grid, y, color=COLOR, linewidth=2.5)
    ax.fill_between(x_grid, y, alpha=0.25, color=COLOR)

    l0 = ax.axvline(0,        color="black", linewidth=2,   linestyle="-")
    l1 = ax.axvline(mean_v,   color="red",   linewidth=1.8, linestyle="--")
    l2 = ax.axvline(median_v, color="blue",  linewidth=1.8, linestyle=":")
    l3 = ax.axvline(trim_v,   color="green", linewidth=1.8, linestyle="-.")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max_global)
    ax.set_title(f"{int(ratio*100)}% masked", fontsize=12, fontweight="bold")
    ax.set_xlabel("Δ = Random PSNR − Block PSNR (dB)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)

    ax.text(0.02, 0.08, f"← Random harder\n({pct_block:.0f}% of chips)",
            transform=ax.transAxes, fontsize=8, color="dimgrey", va="bottom", ha="left")
    ax.text(0.98, 0.08, f"Block harder →\n({pct_random:.0f}% of chips)",
            transform=ax.transAxes, fontsize=8, color="dimgrey", va="bottom", ha="right")

    # Legend with line style + value clearly labelled
    ax.legend(
        handles=[l0, l1, l2, l3],
        labels=[
            "Δ = 0  (reference)",
            f"Mean = {mean_v:+.2f} dB",
            f"Median = {median_v:+.2f} dB",
            f"Trimmed mean = {trim_v:+.2f} dB",
        ],
        fontsize=8, frameon=True, loc="upper right",
        framealpha=0.9, edgecolor="lightgrey"
    )

    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out_path = OUT_DIR / "fig_delta_distributions_600M.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
