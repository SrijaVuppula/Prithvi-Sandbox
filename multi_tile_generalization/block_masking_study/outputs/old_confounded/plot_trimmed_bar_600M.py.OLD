"""
plot_trimmed_bar_600M.py
------------------------
Replots block vs random PSNR bar chart for 600M backbone only,
comparing original mean vs 5% trimmed mean side by side.

Run from repo root:
    python multi_tile_generalization/block_masking_study/scripts/plot_trimmed_bar_600M.py
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[3]
CHIPS_DIR    = REPO_ROOT / "multi_tile_generalization" / "training_chips"
BLOCK_DIR    = REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "outputs"
RANDOM_DIR   = REPO_ROOT / "multi_tile_generalization" / "outputs" / "per_tile"
OUT_DIR      = BLOCK_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
N_CHIPS      = 500
BLOCK_RATIOS = [0.2, 0.4, 0.6, 0.8]
COLOR_BLOCK  = "#B47CC7"
COLOR_RANDOM = "#7FB3D3"

all_files   = sorted(CHIPS_DIR.glob("chip_*_merged.tif"))
rng         = random.Random(SEED)
sampled     = rng.sample(all_files, min(N_CHIPS, len(all_files)))
chip_to_idx = {f.name: i for i, f in enumerate(sampled)}

# Load block
df_block = pd.read_csv(BLOCK_DIR / "results_600M.csv")
df_block = (df_block.groupby(["chip","mask_ratio"])["block_psnr"]
              .mean().reset_index()
              .rename(columns={"block_psnr": "block_psnr_mean"}))
df_block["chip_idx"] = df_block["chip"].map(chip_to_idx)

# Load random
df_random = pd.read_csv(RANDOM_DIR / "600M_results.csv")
df_random = df_random[df_random["mask_ratio"].isin(BLOCK_RATIOS)][["chip_idx","mask_ratio","masked_psnr"]]
df_random.rename(columns={"masked_psnr": "random_psnr"}, inplace=True)

merged = pd.merge(df_block, df_random, on=["chip_idx","mask_ratio"], how="inner")
merged["delta"] = merged["random_psnr"] - merged["block_psnr_mean"]

def trimmed_mean(vals, pct=0.05):
    n = len(vals)
    k = int(np.floor(pct * n))
    return np.sort(vals)[k: n-k].mean()

# Build stats per ratio
stats = []
for ratio in BLOCK_RATIOS:
    sub = merged[merged["mask_ratio"] == ratio]
    b   = sub["block_psnr_mean"].values
    r   = sub["random_psnr"].values
    stats.append({
        "ratio":          ratio,
        "block_mean":     b.mean(),
        "random_mean":    r.mean(),
        "block_trimmed":  trimmed_mean(b),
        "random_trimmed": trimmed_mean(r),
    })
stats = pd.DataFrame(stats)
print(stats.to_string(index=False))

# ── Plot: 1 row, 4 subplots (one per ratio) ───────────────────────────────────
# Each subplot: 4 bars — block mean, random mean, block trimmed, random trimmed
fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)
fig.suptitle("600M Backbone — Block vs Random PSNR: Original Mean vs 5% Trimmed Mean\n"
             "Taller bar = higher PSNR = easier reconstruction",
             fontsize=13, fontweight="bold", y=1.02)

x      = np.array([0, 1, 2.2, 3.2])   # slight gap between mean and trimmed pairs
labels = ["Block\n(mean)", "Random\n(mean)", "Block\n(trimmed)", "Random\n(trimmed)"]
colors = [COLOR_BLOCK, COLOR_RANDOM, COLOR_BLOCK, COLOR_RANDOM]
alphas = [0.95, 0.95, 0.55, 0.55]
hatches= ["", "", "//", "//"]

for ax, (_, row) in zip(axes, stats.iterrows()):
    vals = [row["block_mean"], row["random_mean"],
            row["block_trimmed"], row["random_trimmed"]]

    bars = ax.bar(x, vals, width=0.7,
                  color=colors, alpha=1.0,
                  hatch=["", "", "//", "//"],
                  edgecolor="white", linewidth=0.8)

    # Apply alpha manually
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    # Annotate bar tops
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="500")

    # Delta annotations between block and random pairs
    d_mean    = row["random_mean"]    - row["block_mean"]
    d_trimmed = row["random_trimmed"] - row["block_trimmed"]
    sign_m = "+" if d_mean    >= 0 else ""
    sign_t = "+" if d_trimmed >= 0 else ""

    y_bracket = max(vals[:2]) + 0.45
    ax.annotate("", xy=(x[1], y_bracket), xytext=(x[0], y_bracket),
                arrowprops=dict(arrowstyle="<->", color="dimgrey", lw=1.2))
    ax.text((x[0]+x[1])/2, y_bracket + 0.1,
            f"Δ={sign_m}{d_mean:.2f} dB", ha="center", fontsize=8, color="dimgrey")

    y_bracket2 = max(vals[2:]) + 0.45
    ax.annotate("", xy=(x[3], y_bracket2), xytext=(x[2], y_bracket2),
                arrowprops=dict(arrowstyle="<->", color="dimgrey", lw=1.2))
    ax.text((x[2]+x[3])/2, y_bracket2 + 0.1,
            f"Δ={sign_t}{d_trimmed:.2f} dB", ha="center", fontsize=8, color="dimgrey")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{int(row['ratio']*100)}% masked", fontsize=12, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.set_ylim(28, 40)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Vertical separator between mean and trimmed groups
    ax.axvline(1.6, color="lightgrey", linewidth=1, linestyle="--")
    ax.text(1.6, 28.2, "original  |  trimmed", ha="center",
            fontsize=7.5, color="lightgrey")

# Legend
import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(color=COLOR_BLOCK,  alpha=0.95, label="Block masking"),
    mpatches.Patch(color=COLOR_RANDOM, alpha=0.95, label="Random masking"),
    mpatches.Patch(color="grey",       alpha=0.55, hatch="//", label="5% trimmed"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.06), frameon=False)

plt.tight_layout()
out_path = OUT_DIR / "fig_trimmed_bar_600M.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close()
