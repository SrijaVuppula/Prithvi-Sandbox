"""
plot_random_vs_block_finetuned.py
------------------
Grouped bars: block vs random PSNR by backbone, faceted by mask ratio.
Fine-tuned checkpoints (leakage-free 2004-chip split). Same visual encoding
as the zero-shot figure for direct side-by-side comparison.

Reads outputs_finetuned/results_{bb}.csv -> outputs_finetuned/figures/fig_random_vs_block_finetuned.png
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _style import get_style, BACKBONES, RATIOS

OUT_DIR = SCRIPT_DIR.parent / "outputs_finetuned"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YLIM = (0, 46)

apply_style, COLORS = get_style()
apply_style()

means = {}
for bb in BACKBONES:
    df = pd.read_csv(OUT_DIR / f"results_{bb}.csv")
    means[bb] = df.groupby("mask_ratio")[["block_psnr", "random_psnr"]].mean()

fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
x = np.arange(len(BACKBONES))
w = 0.36

for ax, r in zip(axes, RATIOS):
    blk = np.array([means[b].loc[r, "block_psnr"] for b in BACKBONES])
    rnd = np.array([means[b].loc[r, "random_psnr"] for b in BACKBONES])
    cols = [COLORS[b] for b in BACKBONES]

    ax.bar(x - w/2, blk, w, color=cols, edgecolor="white", lw=0.8, zorder=2)
    ax.bar(x + w/2, rnd, w, color=cols, alpha=0.42, edgecolor="white",
           lw=0.8, hatch="///", zorder=2)

    for xi, (b, rr) in enumerate(zip(blk, rnd)):
        ax.text(xi, max(b, rr) + 1.0, f"Δ{rr-b:.1f}", ha="center", va="bottom",
                fontsize=8.5, color="0.3")

    ax.set_title(f"{int(r*100)}% masked", fontsize=11, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(BACKBONES)
    ax.set_ylim(*YLIM)
    ax.grid(axis="y", alpha=0.22, lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Reconstruction PSNR (dB)")

handles = [Patch(facecolor="0.45", edgecolor="white", label="block (contiguous)"),
           Patch(facecolor="0.45", alpha=0.42, hatch="///", edgecolor="white",
                 label="random (scattered)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Fine-tuned: block vs random masking PSNR",
             y=1.02, fontsize=12)
fig.tight_layout()
out = FIG_DIR / "fig_random_vs_block_finetuned.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
