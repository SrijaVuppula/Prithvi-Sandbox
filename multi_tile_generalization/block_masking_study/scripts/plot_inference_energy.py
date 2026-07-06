"""One clean figure per metric: block vs random, min/max whiskers, mean±std labels."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/Prithvi"))
try:
    from plot_style import apply_style
    apply_style()
except Exception:
    pass

CSV = "multi_tile_generalization/block_masking_study/outputs/inference_energy.csv"
OUTDIR = "multi_tile_generalization/block_masking_study/outputs/figures"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(CSV)
backbones = ["tiny", "100M", "300M", "600M"]
ratios = sorted(df["mask_ratio"].unique())
metrics = [("energy", "energy (mJ)", "%.0f"),
           ("power", "power (W)", "%.0f"),
           ("enc", "encoder (ms)", "%.1f")]
x = np.arange(len(ratios)); w = 0.38

for mkey, mlabel, fmt in metrics:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    ymax = df[f"{mkey}_max"].max() * 1.18
    for ax, b in zip(axes, backbones):
        for mt, off, c in [("random", -w/2, "#0072B2"), ("block", +w/2, "#D55E00")]:
            sub = df[(df.backbone == b) & (df.mask_type == mt)].set_index("mask_ratio").loc[ratios]
            mean = sub[f"{mkey}_mean"].values
            std  = sub[f"{mkey}_std"].values
            lo = mean - sub[f"{mkey}_min"].values
            hi = sub[f"{mkey}_max"].values - mean
            ax.bar(x + off, mean, w, label=mt, color=c,
                   yerr=[lo, hi], capsize=4, error_kw=dict(lw=1.2))
            for xi, m, s, h in zip(x + off, mean, std, sub[f"{mkey}_max"].values):
                ax.text(xi, h + ymax*0.01, f"{fmt % m}\n±{fmt % s}",
                        ha="center", va="bottom", fontsize=6.5, rotation=0)
        ax.set_xlim(-0.5, len(ratios) - 0.5); ax.set_ylim(0, ymax)
        ax.set_xticks(x); ax.set_xticklabels([f"{int(r*100)}%" for r in ratios])
        ax.set_title(b); ax.set_xlabel("mask ratio")
        if b == "tiny": ax.set_ylabel(mlabel); ax.legend(fontsize=8)
        else: ax.set_yticklabels([])
    fig.suptitle(f"Inference {mlabel} — block vs random (whiskers = min/max, label = mean±std over 20 passes)")
    fig.tight_layout()
    out = f"{OUTDIR}/fig_inference_{mkey}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("Wrote", out)
