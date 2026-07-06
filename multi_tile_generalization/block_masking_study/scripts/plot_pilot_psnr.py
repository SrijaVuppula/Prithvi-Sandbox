"""Trial-averaged block PSNR vs mask ratio, per chip."""
import os, sys
import pandas as pd, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
try:
    from plot_style import apply_style; apply_style()
except Exception: pass

CSV = "multi_tile_generalization/block_masking_study/outputs/pilot_trials.csv"
OUT = "multi_tile_generalization/block_masking_study/outputs/figures/fig_pilot_psnr.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(CSV)
g = df.groupby(["chip","ratio"]).psnr.agg(["mean","std"]).reset_index()
labels = {"chip_105_452_merged.tif":"simple (105_452)",
          "chip_268_410_merged.tif":"mid (268_410)",
          "chip_217_425_merged.tif":"complex (217_425)"}

fig, ax = plt.subplots(figsize=(7,5))
for c in labels:
    s = g[g.chip==c].sort_values("ratio")
    ax.errorbar(s.ratio*100, s["mean"], yerr=s["std"], marker="o",
                capsize=4, label=labels[c])
ax.set_xlabel("mask ratio (%)"); ax.set_ylabel("block PSNR (dB)")
ax.set_xticks([20,40,60,80])
ax.set_title("Trial-averaged block PSNR (50 trials, 600M)")
ax.legend()
fig.tight_layout(); fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Wrote", OUT)
