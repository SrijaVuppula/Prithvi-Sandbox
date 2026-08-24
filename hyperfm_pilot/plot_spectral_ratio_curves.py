import sys
sys.path.insert(0, str(__import__('pathlib').Path.home() / "Prithvi"))
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_style import apply_style

apply_style()

RATIOS = [0.2, 0.4, 0.6, 0.8]
ERRORMAP_DIR = Path("errormaps")

rows = []
with open("spectral_ratio_sweep_results.csv") as f:
    for row in csv.DictReader(f):
        rows.append(row)

def median_by(mask_type, metric):
    return [np.median([float(r[metric]) for r in rows
             if r["mask_type"] == mask_type and float(r["ratio"]) == ratio])
            for ratio in RATIOS]

contig_psnr = median_by("contiguous", "psnr")
scatter_psnr = median_by("scattered", "psnr")
ratios_pct = [r * 100 for r in RATIOS]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(ratios_pct, contig_psnr, marker="o", label="Contiguous", color="#D55E00", linewidth=2)
ax.plot(ratios_pct, scatter_psnr, marker="o", label="Scattered", color="#888888", linewidth=2)
ax.set_xlabel("Mask ratio (%)")
ax.set_ylabel("Median PSNR (dB)")
ax.set_ylim(0, max(scatter_psnr) * 1.15)
ax.set_xticks(ratios_pct)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("spectral_psnr_vs_ratio.png", dpi=150)
print("Saved spectral_psnr_vs_ratio.png")

gap_db = [s - c for s, c in zip(scatter_psnr, contig_psnr)]
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.plot(ratios_pct, gap_db, marker="o", color="#D55E00", linewidth=2)
ax2.axhline(0, color="#888888", linewidth=1)
ax2.set_xlabel("Mask ratio (%)")
ax2.set_ylabel("PSNR gap: scattered - contiguous (dB)")
ax2.set_xticks(ratios_pct)
ax2.set_ylim(0, max(gap_db) * 1.3)
fig2.tight_layout()
fig2.savefig("spectral_gap_vs_ratio.png", dpi=150)
print("Saved spectral_gap_vs_ratio.png")

for mask_type in ["contiguous", "scattered"]:
    maps = [np.load(ERRORMAP_DIR / f"{mask_type}_ratio{int(r*100)}.npy") for r in RATIOS]
    vmax = max(np.nanmax(m) for m in maps)
    fig3, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, ratio, m in zip(axes, RATIOS, maps):
        im = ax.imshow(m, cmap="gray_r", vmin=0, vmax=vmax)
        ax.set_title(f"{int(ratio*100)}% masked")
        ax.axis("off")
    fig3.colorbar(im, ax=axes, shrink=0.8, label="Mean abs error (masked bands)")
    fig3.savefig(f"spectral_errormap_grid_{mask_type}.png", dpi=150, bbox_inches="tight")
    print(f"Saved spectral_errormap_grid_{mask_type}.png")
