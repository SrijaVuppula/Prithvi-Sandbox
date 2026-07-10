"""Grid of all inference metrics per backbone (block vs random) with min/max whiskers, fixed y per row."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/Prithvi"))
try:
    from plot_style import apply_style; apply_style()
except Exception: pass

CSV = "multi_tile_generalization/block_masking_study/outputs/inference_energy.csv"
OUT = "multi_tile_generalization/block_masking_study/outputs/figures/fig_inference_metrics.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(CSV)
backbones = ["tiny", "100M", "300M", "600M"]
ratios = sorted(df["mask_ratio"].unique())
metrics = [("time", "time (ms)"), ("enc", "encoder (ms)"), ("dec", "decoder (ms)"),
           ("power", "power (W)"), ("energy", "energy (mJ)")]
x = np.arange(len(ratios)); w = 0.38

ylims = {}
for mkey, _ in metrics:
    ymax = df[f"{mkey}_max"].max()
    ylims[mkey] = (0, ymax * 1.08)

fig, axes = plt.subplots(len(metrics), len(backbones), figsize=(15, 4 * len(metrics)))
for row, (mkey, mlabel) in enumerate(metrics):
    for col, b in enumerate(backbones):
        ax = axes[row, col]
        for mt, off, c in [("random", -w/2, "#0072B2"), ("block", +w/2, "#D55E00")]:
            sub = df[(df.backbone == b) & (df.mask_type == mt)].set_index("mask_ratio").loc[ratios]
            mean = sub[f"{mkey}_mean"].values
            lo = mean - sub[f"{mkey}_min"].values
            hi = sub[f"{mkey}_max"].values - mean
            ax.bar(x + off, mean, w, label=mt, color=c, yerr=[lo, hi], capsize=3, error_kw=dict(lw=1))
        ax.set_xlim(-0.5, len(ratios) - 0.5); ax.set_ylim(*ylims[mkey])
        ax.set_xticks(x); ax.set_xticklabels([f"{int(r*100)}%" for r in ratios])
        if row == 0: ax.set_title(b)
        if col == 0: ax.set_ylabel(mlabel)
        else: ax.set_yticklabels([])
        if row == len(metrics) - 1: ax.set_xlabel("mask ratio")
        if row == 0 and col == 0: ax.legend(fontsize=7)

fig.suptitle("Inference metrics: block vs random (whiskers = min/max over 20 passes, continuous power sampling)", y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Wrote", OUT)
