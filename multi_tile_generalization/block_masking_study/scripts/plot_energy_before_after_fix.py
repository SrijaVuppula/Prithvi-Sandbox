"""
plot_energy_before_after_fix.py
--------------------------------
600M energy, random vs block, before vs after the GPU clock-ramp fix
(per-condition GPU_WARMUP_S=3.0 warmup loop, see measure_inference_energy.py).
"Before" is order-confounded (whichever condition ran first read artificially
low power); "after" uses a fixed warmup so order no longer matters.

Data:
  before -> outputs/old_confounded/energy_UNFIXED_order_10_20_40_60_80_allBB.csv
  after  -> outputs/inference_energy.csv (also has the corrected masker applied,
            but that's orthogonal to what this figure demonstrates -- this panel
            is about the GPU-timing fix specifically, not the masking fix)

Y-axis: DELIBERATE OVERRIDE of the zero-start convention used elsewhere in this
project (2026-07-16, by request). Both panels' 600M energy values sit in a
narrow high band (~7.5k-20k mJ); a zero-anchored axis squeezed the actual
curve shape into the top ~15% of the plot. Range is computed from the real
combined min/max of both datasets (not hand-picked), padded ~15%, and shared
identically across both panels via sharey=True.

Reads -> outputs/figures/fig_energy_before_after_fix_600M.png
"""
import os, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from plot_style import apply_style, style_ax, COLORS, FIGSIZE, FONT

apply_style()

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR.parent / "outputs"
BEFORE_CSV = OUT_ROOT / "old_confounded" / "energy_UNFIXED_order_10_20_40_60_80_allBB.csv"
AFTER_CSV  = OUT_ROOT / "inference_energy.csv"
OUT_DIR = OUT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_before = pd.read_csv(BEFORE_CSV)
df_after  = pd.read_csv(AFTER_CSV)

b600 = df_before[df_before["backbone"] == "600M"]["energy_mean"]
a600 = df_after[df_after["backbone"] == "600M"]["energy_mean"]
combined = pd.concat([b600, a600])
lo, hi = combined.min(), combined.max()
pad = (hi - lo) * 0.15
Y_LIM_600M = (lo - pad, hi + pad)

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["wide"], sharey=True)

for ax, df_src, title in [
    (axes[0], df_before, "Before fix (order-confounded)"),
    (axes[1], df_after,  "After fix (per-condition warmup)"),
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
out = OUT_DIR / "fig_energy_before_after_fix_600M.png"
fig.savefig(out)
plt.close(fig)
print("Wrote", out)
