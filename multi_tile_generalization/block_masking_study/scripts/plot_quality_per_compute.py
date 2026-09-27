"""
plot_quality_per_compute.py
----------------------------
Headline frontier figure: fine-tuned tiny (solid) vs zero-shot tiny (dashed,
faint) vs zero-shot 100M (dashed) on block-masked PSNR. Annotated with the
one-time fine-tuning energy cost and the fraction of the tiny->100M gap
that fine-tuning closes at each ratio.

Reads outputs_finetuned/zeroshot_vs_finetuned_summary.csv
-> outputs_finetuned/figures/fig_quality_per_compute.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _style import get_style, RATIOS

OUT_DIR = SCRIPT_DIR.parent / "outputs_finetuned"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

apply_style, COLORS = get_style()
apply_style()

df = pd.read_csv(OUT_DIR / "zeroshot_vs_finetuned_summary.csv")
df["mask_ratio"] = df["mask_ratio"].astype(float)
block = df[df["geometry"] == "block"]

tiny_zs = block[block["backbone"] == "tiny"].sort_values("mask_ratio")["zeroshot_mean_psnr"].values
tiny_ft = block[block["backbone"] == "tiny"].sort_values("mask_ratio")["finetuned_mean_psnr"].values
m100_zs = block[block["backbone"] == "100M"].sort_values("mask_ratio")["zeroshot_mean_psnr"].values

x = np.array([int(r * 100) for r in RATIOS])
c_tiny, c_100m = COLORS["tiny"], COLORS["100M"]

fig, ax = plt.subplots(figsize=(8.0, 5.4))

ax.plot(x, tiny_zs, color=c_tiny, ls="--", marker="o", mfc="white", mec=c_tiny,
        ms=7, lw=1.6, alpha=0.8, label="tiny, zero-shot", zorder=3)
ax.plot(x, tiny_ft, color=c_tiny, ls="-", marker="o", ms=7.5, lw=2.4,
        label="tiny, fine-tuned (332.7 kJ)", zorder=5)
ax.plot(x, m100_zs, color=c_100m, ls="--", marker="s", mfc="white", mec=c_100m,
        ms=7, lw=1.6, alpha=0.8, label="100M, zero-shot", zorder=4)

for xi in range(len(x)):
    total_gap = m100_zs[xi] - tiny_zs[xi]
    remaining_gap = m100_zs[xi] - tiny_ft[xi]
    pct_closed = 100 * (1 - remaining_gap / total_gap)
    ax.annotate("", xy=(x[xi], tiny_ft[xi]), xytext=(x[xi], m100_zs[xi]),
                arrowprops=dict(arrowstyle="-", color="0.6", lw=1.0, ls=":"))
    ax.text(x[xi] + 1.5, (tiny_ft[xi] + m100_zs[xi]) / 2,
            f"{pct_closed:.0f}% closed", fontsize=8, color="0.35", va="center")

ax.set_xlabel("Mask Ratio (%)")
ax.set_ylabel("Block-masked PSNR (dB)")
ax.set_xticks(x)
ax.grid(alpha=0.22, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=9.5)
ax.set_title("Fine-tuned tiny closes most of its quality gap to zero-shot 100M\n"
             "at unchanged per-inference energy", fontsize=12, pad=12)

fig.tight_layout()
out = FIG_DIR / "fig_quality_per_compute.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
