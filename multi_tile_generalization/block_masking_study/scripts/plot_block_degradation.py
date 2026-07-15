"""
plot_block_degradation.py
-------------------------
Block-masking PSNR degradation by backbone, corrected data.
500 chips x 5 trials, contiguous rectangular block on the summer frame,
spring/fall context frames fully visible.

Reads outputs/results_{bb}.csv -> outputs/figures/fig_block_degradation.png
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

YLIM = (30, 35.5)

apply_style, COLORS = get_style()
apply_style()

fig, ax = plt.subplots(figsize=(7.4, 4.8))
x = np.array([int(r * 100) for r in RATIOS])

for bb in BACKBONES:
    df = pd.read_csv(OUT_DIR / f"results_{bb}.csv")
    g = df.groupby("mask_ratio")["block_psnr"]
    m = g.mean().loc[RATIOS].values
    se = (g.std() / np.sqrt(g.count())).loc[RATIOS].values
    c = COLORS[bb]
    ax.fill_between(x, m - 1.96*se, m + 1.96*se, color=c, alpha=0.15, lw=0)
    ax.plot(x, m, color=c, marker=MARKERS[bb], ms=6, lw=2.0, zorder=3)
    # label at line end instead of a legend box
    ax.text(x[-1] + 2.0, m[-1], bb, color=c, fontsize=10,
            va="center", ha="left", fontweight="medium")

ax.set_xlabel("Block mask ratio (% of summer frame occluded)")
ax.set_ylabel("Block PSNR (dB)")
ax.set_xticks(x)
ax.set_xticklabels([f"{v}%" for v in x])
ax.set_xlim(14, 92)
ax.set_ylim(*YLIM)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.grid(axis="x", visible=False)

fig.tight_layout()
out = FIG_DIR / "fig_block_degradation.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
