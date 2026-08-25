import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.expanduser('~/Prithvi'))
try:
    from plot_style import apply_style
    apply_style()
except ImportError:
    print("WARNING: plot_style.py not found on path, using matplotlib defaults")

GRAY_LIGHT = "#BEBEBE"
GRAY_MED   = "#7A7A7A"
ACCENT     = "#0072B2"  # Okabe-Ito blue

def gap_by_ratio(data):
    out = []
    for r in sorted(data['ratio'].unique()):
        c = data[(data['ratio'] == r) & (data['mask_type'] == 'contiguous')]['psnr'].median()
        s = data[(data['ratio'] == r) & (data['mask_type'] == 'scattered')]['psnr'].median()
        out.append(c - s)
    return out

df_10t_20tile  = pd.read_csv('spectral_ratio_sweep_results_10trials_backup.csv')
df_50t_20tile  = pd.read_csv('spectral_ratio_sweep_results_20tiles_50trials_backup.csv')
df_50t_100tile = pd.read_csv('spectral_ratio_sweep_results.csv')

ratios = sorted(df_50t_100tile['ratio'].unique())
x = [r*100 for r in ratios]

gap1 = gap_by_ratio(df_10t_20tile)
gap2 = gap_by_ratio(df_50t_20tile)
gap3 = gap_by_ratio(df_50t_100tile)

fig, ax = plt.subplots(figsize=(8.5, 6))

ax.plot(x, gap1, marker='o', markersize=6, color=GRAY_LIGHT, linewidth=1.5,
        linestyle=(0, (4, 2)), label='10 trials, 20 tiles  (n=1,600)', zorder=2)
ax.plot(x, gap2, marker='o', markersize=6, color=GRAY_MED, linewidth=1.8,
        linestyle='-', label='50 trials, 20 tiles  (n=8,000)', zorder=3)
ax.plot(x, gap3, marker='o', markersize=8, color=ACCENT, linewidth=2.8,
        linestyle='-', label='50 trials, 100 tiles  (n=40,000, current best)', zorder=4)

# label the current-best line directly with its values
for xi, gi in zip(x, gap3):
    ax.annotate(f'{gi:+.2f}', (xi, gi), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9.5, color=ACCENT, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.9, zorder=1)

# clean up spines and ticks
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#444444')
ax.spines['bottom'].set_color('#444444')
ax.tick_params(colors='#444444')

ax.set_xlabel('Mask Ratio (%)', fontsize=11.5, labelpad=8)
ax.set_ylabel('Gap: Contiguous \u2212 Scattered PSNR (dB)', fontsize=11.5, labelpad=8)
ax.set_title('Spectral Masking Gap \u2014 Estimate Stabilization Across Sample Sizes',
             fontsize=13.5, fontweight='bold', pad=14)

ax.set_xticks(x)
ax.set_xlim(min(x) - 5, max(x) + 5)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.grid(axis='y', color='#E5E5E5', linewidth=0.8, zorder=0)
ax.grid(axis='x', visible=False)

# legend below the plot, out of the way of data
legend = ax.legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.14),
    ncol=1, frameon=False, fontsize=10, handlelength=2.5,
    title='Sample size', title_fontsize=10.5, alignment='left'
)
legend.get_title().set_fontweight('bold')

fig.text(0.5, -0.02,
         'Gap computed as median PSNR, contiguous minus scattered masking, per mask ratio.',
         ha='center', fontsize=8.5, color='#666666', style='italic')

fig.tight_layout()
fig.savefig('fig_gap_stability_evolution.png', dpi=150, bbox_inches='tight')
print("Saved fig_gap_stability_evolution.png")
