"""
plot_zeroshot_vs_finetuned_psnr.py
-----------------------------------
Zero-shot (dashed, hollow markers) vs fine-tuned (solid, filled markers)
PSNR curves, one panel per backbone, block (top row) and random (bottom row).
Reads the paired summary already computed by compare_zeroshot_finetuned.py.

Reads outputs_finetuned/zeroshot_vs_finetuned_summary.csv
-> outputs_finetuned/figures/fig_zeroshot_vs_finetuned_psnr.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _style import get_style, BACKBONES, RATIOS, MARKERS

OUT_DIR = SCRIPT_DIR.parent / "outputs_finetuned"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

apply_style, COLORS = get_style()
apply_style()

df = pd.read_csv(OUT_DIR / "zeroshot_vs_finetuned_summary.csv")
df["mask_ratio"] = df["mask_ratio"].astype(float)

fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True)
x = np.array([int(r * 100) for r in RATIOS])

for row, geom in enumerate(["block", "random"]):
    for col, bb in enumerate(BACKBONES):
        ax = axes[row, col]
        sub = df[(df["backbone"] == bb) & (df["geometry"] == geom)].sort_values("mask_ratio")
        c = COLORS[bb]

        ax.plot(x, sub["zeroshot_mean_psnr"], color=c, marker=MARKERS[bb],
                mfc="white", mec=c, ms=7, lw=1.6, ls="--", alpha=0.85,
                label="zero-shot", zorder=3)
        ax.plot(x, sub["finetuned_mean_psnr"], color=c, marker=MARKERS[bb],
                ms=7, lw=2.2, ls="-", label="fine-tuned", zorder=4)

        for xi, (z, f) in enumerate(zip(sub["zeroshot_mean_psnr"], sub["finetuned_mean_psnr"])):
            d = f - z
            sign = "+" if d >= 0 else "\u2212"
            ax.text(x[xi], max(z, f) + 0.35, f"{sign}{abs(d):.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="0.35")

        if row == 0:
            ax.set_title(bb, fontsize=11.5, pad=6)
        if col == 0:
            ax.set_ylabel(f"{geom.capitalize()} PSNR (dB)")
        ax.set_xticks(x)
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)

for ax in axes[-1, :]:
    ax.set_xlabel("Mask Ratio (%)")

handles = [Line2D([0], [0], color="0.35", ls="--", marker="o", mfc="white",
                   mec="0.35", ms=6, label="zero-shot"),
           Line2D([0], [0], color="0.35", ls="-", marker="o", ms=6,
                   label="fine-tuned")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Fine-tuning effect on reconstruction PSNR, by backbone and masking geometry",
             y=1.01, fontsize=13)
fig.tight_layout()
out = FIG_DIR / "fig_zeroshot_vs_finetuned_psnr.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
