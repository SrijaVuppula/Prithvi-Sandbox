"""
plot_difficulty_gap.py
----------------------
Corrected difficulty gap: mean delta (random_psnr - block_psnr) vs mask ratio,
per backbone. Positive => block harder. The dashed zero line is where the old
study claimed a crossover; the corrected curves stay entirely above it.

Reads outputs/results_{bb}.csv -> outputs/figures/fig_difficulty_gap.png
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _style import get_style, BACKBONES, RATIOS, MARKERS

OUT_DIR = SCRIPT_DIR.parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YLIM = (0, 4.0)   # zero is meaningful here: it is the crossover line

apply_style, COLORS = get_style()
apply_style()

fig, ax = plt.subplots(figsize=(7.0, 4.8))
x = np.array([int(r * 100) for r in RATIOS])

for bb in BACKBONES:
    df = pd.read_csv(OUT_DIR / f"results_{bb}.csv")
    g = df.groupby("mask_ratio")["delta_rand_minus_block"]
    d = g.mean().loc[RATIOS].values
    se = (g.std() / np.sqrt(g.count())).loc[RATIOS].values
    c = COLORS[bb]
    ax.fill_between(x, d - 1.96 * se, d + 1.96 * se, color=c, alpha=0.15, linewidth=0)
    ax.plot(x, d, color=c, marker=MARKERS[bb], ms=6, lw=2.0, label=bb, zorder=3)

ax.axhline(0, color="0.25", lw=1.1, ls="--", zorder=2)
ax.set_title("Masking Difficulty Gap: Random vs. Block Occlusion", fontsize=11, pad=10)

ax.set_xlabel("Mask Ratio (%)")
ax.set_ylabel("Δ PSNR  (random − block, dB)")
ax.set_xticks(x)
ax.set_xlim(14, 86)
ax.set_ylim(*YLIM)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.grid(axis="x", visible=False)
ax.legend(title="Backbone", loc="upper right", frameon=False)

fig.tight_layout()
out = FIG_DIR / "fig_difficulty_gap.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
