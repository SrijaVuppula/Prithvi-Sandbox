import sys, os
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from plot_style import apply_style, style_ax, COLORS, BACKBONE_COLORS, FIGSIZE, FONT

import pandas as pd
import matplotlib.pyplot as plt

apply_style()

OUT_DIR = "multi_tile_generalization/block_masking_study/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_CSV   = "multi_tile_generalization/block_masking_study/outputs/energy_FIXED_warmup_order_10_20_40_60_80_allBB.csv"
UNFIXED_CSV = "multi_tile_generalization/block_masking_study/outputs/energy_UNFIXED_order_10_20_40_60_80_allBB.csv"

BACKBONES = ["tiny", "100M", "300M", "600M"]

# Hardcoded, fixed y-axis ranges per backbone — all start at 0 (honest baseline,
# no exaggeration of drop size), round numbers, chosen from max observed value
# across BOTH the fixed and unfixed datasets so before/after stays comparable.
YLIM = {
    "tiny": (0, 2000),
    "100M": (0, 4000),
    "300M": (0, 8000),
    "600M": (0, 20000),
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: FIXED data — energy vs ratio, random vs block overlaid, 2x2 grid
# ─────────────────────────────────────────────────────────────────────────────
df_fixed = pd.read_csv(FIXED_CSV)

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
axes = axes.flatten()

for i, bb in enumerate(BACKBONES):
    ax = axes[i]
    sub = df_fixed[df_fixed["backbone"] == bb].sort_values("mask_ratio")
    color = BACKBONE_COLORS[bb]

    rand = sub[sub["mask_type"] == "random"]
    block = sub[sub["mask_type"] == "block"]

    ax.plot(rand["mask_ratio"] * 100, rand["energy_mean"],
            marker="o", linestyle="-", color=color, label="random", linewidth=1.8)
    ax.plot(block["mask_ratio"] * 100, block["energy_mean"],
            marker="s", linestyle="--", color=color, label="block", linewidth=1.8,
            alpha=0.7)

    ax.set_ylim(*YLIM[bb])
    style_ax(ax, xlabel="Mask ratio (%)", ylabel="Energy (mJ)", title=f"{bb}")
    ax.legend(fontsize=FONT["legend"])

fig.suptitle("Energy vs Mask Ratio — Random vs Block (Fixed Measurement)",
             fontsize=FONT["title"] + 1, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_energy_vs_ratio_FIXED.png")
plt.close(fig)
print(f"Saved {OUT_DIR}/fig_energy_vs_ratio_FIXED.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Before/after — 600M random vs block, unfixed vs fixed side by side
# Fixed shared y-axis, starting at 0, hardcoded (not computed from data).
# ─────────────────────────────────────────────────────────────────────────────
df_unfixed = pd.read_csv(UNFIXED_CSV)

Y_LIM_600M = (0, 20000)  # hardcoded, matches Figure 1's 600M range for consistency

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["wide"], sharey=True)

for ax, df_src, title in [
    (axes[0], df_unfixed, "Before fix (order-confounded)"),
    (axes[1], df_fixed,   "After fix (per-condition warmup)"),
]:
    sub = df_src[df_src["backbone"] == "600M"].sort_values("mask_ratio")
    rand = sub[sub["mask_type"] == "random"]
    block = sub[sub["mask_type"] == "block"]

    ax.plot(rand["mask_ratio"] * 100, rand["energy_mean"],
            marker="o", linestyle="-", color=COLORS["blue"], label="random", linewidth=1.8)
    ax.plot(block["mask_ratio"] * 100, block["energy_mean"],
            marker="s", linestyle="--", color=COLORS["vermillion"], label="block", linewidth=1.8)

    ax.set_ylim(*Y_LIM_600M)
    style_ax(ax, xlabel="Mask ratio (%)", ylabel="Energy (mJ)", title=title)
    ax.legend(fontsize=FONT["legend"])

fig.suptitle("600M Energy: Random vs Block — Before vs After Clock-Ramp Fix",
             fontsize=FONT["title"] + 1, y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_energy_before_after_fix_600M.png")
plt.close(fig)
print(f"Saved {OUT_DIR}/fig_energy_before_after_fix_600M.png")
