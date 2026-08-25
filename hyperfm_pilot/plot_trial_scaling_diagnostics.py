import sys, os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~/Prithvi'))
try:
    from plot_style import apply_style
    apply_style()
    ACCENT = "#0072B2"   # Okabe-Ito blue
    ACCENT2 = "#D55E00"  # Okabe-Ito vermillion, muted use only
    GRAY = "#999999"
except ImportError:
    print("WARNING: plot_style.py not found on path, using matplotlib defaults")
    ACCENT = "#0072B2"
    ACCENT2 = "#D55E00"
    GRAY = "#999999"

df = pd.read_csv('spectral_ratio_sweep_results.csv')
df10 = pd.read_csv('spectral_ratio_sweep_results_10trials_backup.csv')

# ---- Figure 1: per-tile median PSNR spread at 20% scattered ----
sub = df[(df['ratio'] == 0.2) & (df['mask_type'] == 'scattered')]
per_tile = sub.groupby('tile')['psnr'].median().sort_values()
labels = [t.split('_image_')[-1] for t in per_tile.index]  # short tile id

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(labels, per_tile.values, color=ACCENT)
ax.set_xlim(0, per_tile.values.max() * 1.1)
ax.set_xlabel('Median PSNR (dB)')
ax.set_ylabel('Tile ID')
ax.set_title('Tile-to-Tile PSNR Spread — 20% Scattered Masking (50 trials/tile)')
ax.axvline(per_tile.values.mean(), color=GRAY, linestyle='--', linewidth=1,
           label=f'mean across tiles: {per_tile.values.mean():.1f} dB')
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig('fig_tile_variance_20pct_scattered.png', dpi=150)
print(f"Saved fig_tile_variance_20pct_scattered.png — range: {per_tile.values.max()-per_tile.values.min():.2f} dB")

# ---- Figure 2: gap (contiguous - scattered) at 10 vs 50 trials, across ratios ----
def gap_by_ratio(data):
    out = []
    for r in sorted(data['ratio'].unique()):
        c = data[(data['ratio'] == r) & (data['mask_type'] == 'contiguous')]['psnr'].median()
        s = data[(data['ratio'] == r) & (data['mask_type'] == 'scattered')]['psnr'].median()
        out.append(c - s)
    return out

ratios = sorted(df['ratio'].unique())
gap10 = gap_by_ratio(df10)
gap50 = gap_by_ratio(df)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([r*100 for r in ratios], gap10, marker='o', color=GRAY, linewidth=2, label='10 trials')
ax.plot([r*100 for r in ratios], gap50, marker='o', color=ACCENT, linewidth=2, label='50 trials')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mask Ratio (%)')
ax.set_ylabel('Gap: Contiguous − Scattered PSNR (dB)')
ax.set_title('Spectral Masking Gap vs. Ratio — Trial Count Sensitivity')
ax.set_xticks([r*100 for r in ratios])
ax.legend()
fig.tight_layout()
fig.savefig('fig_gap_stability_10v50trials.png', dpi=150)
print("Saved fig_gap_stability_10v50trials.png")
