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
MIN_LABEL_GAP = 0.35  # dB -- minimum vertical spacing between end-of-line labels

apply_style, COLORS = get_style()
apply_style()

fig, ax = plt.subplots(figsize=(7.4, 4.8))
x = np.array([int(r * 100) for r in RATIOS])

end_labels = []  # (y_true, backbone, color)
for bb in BACKBONES:
    df = pd.read_csv(OUT_DIR / f"results_{bb}.csv")
    g = df.groupby("mask_ratio")["block_psnr"]
    m = g.mean().loc[RATIOS].values
    se = (g.std() / np.sqrt(g.count())).loc[RATIOS].values
    c = COLORS[bb]
    ax.fill_between(x, m - 1.96*se, m + 1.96*se, color=c, alpha=0.15, lw=0)
    ax.plot(x, m, color=c, marker=MARKERS[bb], ms=6, lw=2.0, zorder=3)
    end_labels.append((m[-1], bb, c))

# Stagger end-of-line labels so close backbones (e.g. 600M/300M at high
# ratio) don't overlap: sort by value, push apart to a minimum gap, and
# draw a thin leader line back to the true data point if nudged.
end_labels.sort(key=lambda t: t[0])
adjusted_y = []
for y_true, bb, c in end_labels:
    y = y_true if not adjusted_y else max(y_true, adjusted_y[-1] + MIN_LABEL_GAP)
    adjusted_y.append(y)

for (y_true, bb, c), y_label in zip(end_labels, adjusted_y):
    if abs(y_label - y_true) > 0.02:
        ax.plot([x[-1], x[-1] + 1.6], [y_true, y_label], color=c, lw=0.7, alpha=0.5, zorder=2)
    ax.text(x[-1] + 2.0, y_label, bb, color=c, fontsize=10,
            va="center", ha="left", fontweight="medium")

ax.set_title("Block Masking PSNR Degradation by Backbone", fontsize=11, pad=10)
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
