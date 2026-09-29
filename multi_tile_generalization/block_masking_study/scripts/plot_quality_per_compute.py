"""
plot_quality_per_compute.py
----------------------------
Quality-per-compute figure: fine-tuned tiny (solid) vs zero-shot tiny (dashed) vs
zero-shot 100M (dashed), contiguous-masked PSNR. Each dotted connector is annotated
with the share of the tiny -> zero-shot-100M gap that fine-tuning closes.

All values are medians over 500 chips (per-chip mean over 5 trials), read from
outputs_finetuned/paper_stats.csv, so the annotations match the text and tables.
One-time fine-tuning energy is read from outputs/finetune_logs/tiny_finetune_cost.csv.

Single-column figure (3.35 in), paper fonts via ~/Prithvi/paper_style.py.

-> outputs_finetuned/figures/fig_quality_per_compute.pdf  (LaTeX)
   outputs_finetuned/figures/fig_quality_per_compute.png  (preview)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from _style import get_style                                   # noqa: E402
from paper_style import apply_paper_style, PAPER_FONT, COL_W   # noqa: E402

BASE = SCRIPT_DIR.parent
OUT_DIR = BASE / "outputs_finetuned"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_, COLORS = get_style()
apply_paper_style()

stats = pd.read_csv(OUT_DIR / "paper_stats.csv").sort_values(["backbone", "ratio"])
tiny = stats[stats["backbone"] == "tiny"]
m100 = stats[stats["backbone"] == "100M"]
assert list(tiny["ratio"]) == list(m100["ratio"]) == [0.2, 0.4, 0.6, 0.8]

x = (tiny["ratio"].values * 100).round().astype(int)
tiny_zs = tiny["zs_contig_med"].values
tiny_ft = tiny["ft_contig_med"].values
m100_zs = m100["zs_contig_med"].values

cost_kj = pd.read_csv(BASE / "outputs" / "finetune_logs" / "tiny_finetune_cost.csv")["energy_kj"].iloc[0]

Y_LO, Y_HI = 30.0, 34.5
assert min(tiny_zs.min(), tiny_ft.min()) > Y_LO and max(m100_zs.max(), tiny_ft.max()) < Y_HI

c_tiny, c_100m = COLORS["tiny"], COLORS["100M"]
F = PAPER_FONT

fig, ax = plt.subplots(figsize=(COL_W, 2.35))

ax.plot(x, tiny_zs, color=c_tiny, ls="--", marker="o", mfc="white", mec=c_tiny,
        ms=3.5, lw=1.0, alpha=0.85, label="tiny, zero-shot", zorder=3)
ax.plot(x, tiny_ft, color=c_tiny, ls="-", marker="o", ms=3.8, lw=1.7,
        label=f"tiny, fine-tuned ({cost_kj:.0f} kJ one-time)", zorder=5)
ax.plot(x, m100_zs, color=c_100m, ls="--", marker="s", mfc="white", mec=c_100m,
        ms=3.5, lw=1.0, alpha=0.85, label="100M, zero-shot", zorder=4)

pct = 100 * (tiny_ft - tiny_zs) / (m100_zs - tiny_zs)   # share of the tiny -> 100M gap closed
for xi in range(len(x)):
    ax.plot([x[xi], x[xi]], [tiny_ft[xi], m100_zs[xi]], color="0.6", lw=0.8, ls=":", zorder=2)
    ax.text(x[xi] + 1.6, (tiny_ft[xi] + m100_zs[xi]) / 2, f"{pct[xi]:.0f}%",
            fontsize=F["annot"], color="0.3", va="center", ha="left")

ax.set_xlabel("Mask ratio (%)")
ax.set_ylabel("Contiguous PSNR (dB)")
ax.set_xticks(x)
ax.set_xlim(15, 88)
ax.set_ylim(Y_LO, Y_HI)
ax.set_yticks(np.arange(30, 34.6, 1))
ax.grid(alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=0.5)
for s in ax.spines.values():
    s.set_linewidth(0.6)
ax.legend(loc="lower left", frameon=False, fontsize=F["legend"], handlelength=2.0,
          borderaxespad=0.3, labelspacing=0.3)

fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    out = FIG_DIR / f"fig_quality_per_compute.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")
