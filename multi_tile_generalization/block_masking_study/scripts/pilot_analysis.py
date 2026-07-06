"""Required trials via n=(z*sigma/E)^2 + running-mean convergence plot."""
import os, sys, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
try:
    from plot_style import apply_style; apply_style()
except Exception: pass

CSV = "multi_tile_generalization/block_masking_study/outputs/pilot_trials.csv"
OUT = "multi_tile_generalization/block_masking_study/outputs/figures/fig_pilot_convergence.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Z, E = 1.96, 0.25   # margin = +/-0.25 dB

df = pd.read_csv(CSV)
chips = df.chip.unique(); ratios = sorted(df.ratio.unique())

print(f"{'chip':<28}{'ratio':>6}{'sigma':>8}{'req_n(E=0.25)':>14}")
for c in chips:
    for r in ratios:
        sd = df[(df.chip==c)&(df.ratio==r)].psnr.std()
        n = math.ceil((Z*sd/E)**2) if sd>0 else 1
        print(f"{c:<28}{int(r*100):>5}%{sd:>8.3f}{n:>14}")

fig, axes = plt.subplots(1, len(ratios), figsize=(16,4))
for ax, r in zip(axes, ratios):
    for c in chips:
        v = df[(df.chip==c)&(df.ratio==r)].sort_values("trial").psnr.values
        run = np.cumsum(v)/np.arange(1,len(v)+1)
        ax.plot(range(1,len(v)+1), run, label=c.replace("_merged.tif",""))
    ax.set_title(f"{int(r*100)}% mask"); ax.set_xlabel("trials averaged")
    if r==ratios[0]: ax.set_ylabel("running mean PSNR (dB)"); ax.legend(fontsize=7)
fig.suptitle("Convergence of block-PSNR vs number of trials (600M)")
fig.tight_layout(); fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Wrote", OUT)
