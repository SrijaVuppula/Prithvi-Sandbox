"""
Inference time breakdown per backbone: encoder vs decoder, by mask ratio.

Block and random encode identical token counts at a matched summer count, so
their cost is identical by construction (verified: <0.5% apart in all 40
conditions). mask_type is therefore averaged over.

The stack shows why energy savings are sub-proportional to masking: the decoder
reconstructs all positions at every ratio, so its cost is a fixed floor; only
the encoder shrinks.

Reads outputs/inference_energy.csv -> outputs/figures/fig_inference_metrics.png
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _style import get_style, BACKBONES

OUT_DIR = SCRIPT_DIR.parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

apply_style, COLORS = get_style()
apply_style()

df = pd.read_csv(OUT_DIR / "inference_energy.csv")
g = (df.groupby(["backbone", "mask_ratio"])
       .agg(enc=("enc_mean", "mean"), dec=("dec_mean", "mean"),
            energy=("energy_mean", "mean"), tokens=("tokens_encoded", "first"))
       .reset_index())

ratios = sorted(df["mask_ratio"].unique())
x = np.arange(len(ratios))

fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))

for ax, bb in zip(axes, BACKBONES):
    sub = g[g["backbone"] == bb].sort_values("mask_ratio")
    enc, dec = sub["enc"].values, sub["dec"].values
    c = COLORS[bb]
    ax.bar(x, enc, 0.62, color=c, label="encoder", zorder=2)
    ax.bar(x, dec, 0.62, bottom=enc, color=c, alpha=0.35, hatch="///",
           edgecolor="white", lw=0.6, label="decoder", zorder=2)

    total = enc + dec
    drop = (total[0] - total[-1]) / total[0] * 100
    ax.text(0.5, 0.94, f"total −{drop:.0f}% over sweep\n(tokens −24%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="0.3")

    ax.set_title(bb, fontsize=11, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r*100)}%" for r in ratios])
    ax.set_xlabel("Mask ratio")
    ax.set_ylim(0, total.max() * 1.35)   # per-backbone scale: ranges differ 10x
    ax.grid(axis="y", alpha=0.22, lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Inference time (ms)")

handles = [Patch(facecolor="0.45", label="encoder (scales with visible tokens)"),
           Patch(facecolor="0.45", alpha=0.35, hatch="///", edgecolor="white",
                 label="decoder (fixed: reconstructs all positions)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.05))

fig.suptitle("Masking more saves less than expected: decoder cost is independent of mask ratio",
             y=1.02, fontsize=12)
fig.tight_layout()
out = FIG_DIR / "fig_inference_metrics.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
