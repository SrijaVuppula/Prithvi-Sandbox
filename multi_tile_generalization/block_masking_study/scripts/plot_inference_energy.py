"""
plot_inference_energy.py
-------------------------
One clean figure per metric: block vs random, min/max whiskers, mean+-std labels.
Reads outputs/inference_energy.csv -> outputs/figures/fig_inference_{energy,power,enc}.png

Colors: block uses each backbone's own accent color (BACKBONE_COLORS, same
identity used throughout the project); random uses a neutral gray with a
hatch pattern. Replaces the prior fixed blue (#0072B2) vs vermillion (#D55E00)
pairing -- a bright, high-contrast complementary clash, avoided per project
convention (primary accent + neutral gray for two-series comparisons).
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from plot_style import apply_style, BACKBONE_COLORS
apply_style()

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR.parent / "outputs"
CSV = OUT_ROOT / "inference_energy.csv"
OUTDIR = OUT_ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

NEUTRAL_GRAY = "#7F7F7F"

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
        accent = BACKBONE_COLORS[b]
        for mt, off, c, hatch in [("random", -w/2, NEUTRAL_GRAY, "///"),
                                   ("block",  +w/2, accent,      None)]:
            sub = df[(df.backbone == b) & (df.mask_type == mt)].set_index("mask_ratio").loc[ratios]
            mean = sub[f"{mkey}_mean"].values
            std  = sub[f"{mkey}_std"].values
            lo = mean - sub[f"{mkey}_min"].values
            hi = sub[f"{mkey}_max"].values - mean
            ax.bar(x + off, mean, w, label=mt, color=c, hatch=hatch,
                   edgecolor="white", linewidth=0.8,
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
    out = OUTDIR / f"fig_inference_{mkey}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("Wrote", out)
