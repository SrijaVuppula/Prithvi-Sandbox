"""
plot_delta_distributions.py
---------------------------
Density of chip-level delta (random - block), faceted by mask ratio, one curve
per backbone. Matches the layout of the old study's delta distribution figure,
but built on the corrected paired results.

Convention: delta = random_psnr - block_psnr.  delta > 0  =>  BLOCK HARDER.
Chip-level delta = mean over the 5 trials per (chip, ratio).

Reads outputs/results_{bb}.csv -> outputs/figures/fig_delta_distributions.png
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _style import get_style, BACKBONES, RATIOS

OUT_DIR = SCRIPT_DIR.parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

XLIM = (-2.0, 8.0)   # shared across panels for comparability

apply_style, COLORS = get_style()
apply_style()

fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharex=True)

for ax, r in zip(axes, RATIOS):
    for bb in BACKBONES:
        df = pd.read_csv(OUT_DIR / f"results_{bb}.csv")
        d = (df[df["mask_ratio"] == r]
               .groupby("chip")["delta_rand_minus_block"].mean().values)
        xs = np.linspace(XLIM[0], XLIM[1], 400)
        kde = gaussian_kde(d)
        c = COLORS[bb]
        ax.plot(xs, kde(xs), color=c, lw=1.8, zorder=3)
        ax.fill_between(xs, kde(xs), color=c, alpha=0.10, linewidth=0)
        ax.axvline(d.mean(), color=c, lw=1.2, ls="--", alpha=0.9, zorder=2)

    ax.axvline(0, color="black", lw=1.6, zorder=4)
    ax.set_title(f"{int(r*100)}% masked", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Δ = Random PSNR − Block PSNR (dB)")
    ax.set_xlim(*XLIM)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.grid(axis="x", visible=False)

    yt = ax.get_ylim()[1]
    ax.text(-1.85, yt * 0.96, "← Random harder", fontsize=8.5, color="0.45", va="top")
    ax.text(7.85, yt * 0.96, "Block harder →", fontsize=8.5, color="0.45",
            va="top", ha="right")

axes[0].set_ylabel("Density")

handles = [Line2D([0], [0], color=COLORS[b], lw=2.5, label=b) for b in BACKBONES]
handles += [Line2D([0], [0], color="0.4", lw=1.2, ls="--", label="Mean"),
            Line2D([0], [0], color="black", lw=1.6, label="Δ = 0")]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Distribution of  Δ = Random PSNR − Block PSNR  per Chip\n"
             "Peak right of 0 → block harder   |   Peak left of 0 → random harder",
             y=1.04, fontsize=12)
fig.tight_layout()
out = FIG_DIR / "fig_delta_distributions.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
