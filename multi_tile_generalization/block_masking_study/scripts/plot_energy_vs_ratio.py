"""
plot_energy_vs_ratio.py
------------------------
Energy vs mask ratio, random vs block overlaid, one panel per backbone.
Data: inference_energy.csv -- corrected on BOTH axes (temporal_gap_masker's
fixed block/random maskers, and the GPU_WARMUP_S=3.0 per-condition clock-ramp
fix). Supersedes the archived energy_FIXED_warmup_...csv, which used the old
(pre-2026-07-16) masking convention despite the "FIXED" name -- that name only
referred to the GPU-timing fix, not the masking fix.

Y-axis: log-scale, SHARED across all four backbones (sharey=True, DELIBERATE
override of the zero-start convention, by request 2026-07-16). Backbone
energy spans ~10x (tiny ~1.8k mJ vs 600M ~20k mJ) -- a literal linear shared
range would flatten the smaller backbones to near-invisible lines. Log scale
keeps proportional change visible for every backbone on one consistent axis.
ScalarFormatter keeps tick labels as plain numbers, not powers of ten.

Reads outputs/inference_energy.csv -> outputs/figures/fig_energy_vs_ratio.png
"""
import os, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from plot_style import apply_style, style_ax, BACKBONE_COLORS, FONT

apply_style()

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR.parent / "outputs"
CSV = OUT_ROOT / "inference_energy.csv"
OUT_DIR = OUT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = ["tiny", "100M", "300M", "600M"]

df = pd.read_csv(CSV)

# Shared log-scale range across all four panels, ~12% padded beyond the
# global min/max (computed from real data, not hand-picked).
lo, hi = df["energy_mean"].min(), df["energy_mean"].max()
YLIM = (lo / 1.12, hi * 1.12)

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True)
axes = axes.flatten()

for i, bb in enumerate(BACKBONES):
    ax = axes[i]
    sub = df[df["backbone"] == bb].sort_values("mask_ratio")
    color = BACKBONE_COLORS[bb]

    rand = sub[sub["mask_type"] == "random"]
    block = sub[sub["mask_type"] == "block"]

    ax.plot(rand["mask_ratio"] * 100, rand["energy_mean"],
            marker="o", linestyle="-", color=color, label="random", linewidth=1.8)
    ax.plot(block["mask_ratio"] * 100, block["energy_mean"],
            marker="s", linestyle="--", color=color, label="block", linewidth=1.8,
            alpha=0.7)

    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y)):,}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=()))  # no minor ticks
    style_ax(ax, xlabel="Mask ratio (%)", ylabel="Energy (mJ)", title=f"{bb}")
    if i == 0:
        ax.legend(fontsize=FONT["legend"])

fig.suptitle("Energy vs Mask Ratio — Random vs Block (log scale, shared y-axis)",
             fontsize=FONT["title"] + 1, y=1.02)
fig.tight_layout()
out = OUT_DIR / "fig_energy_vs_ratio.png"
fig.savefig(out)
plt.close(fig)
print("Wrote", out)
