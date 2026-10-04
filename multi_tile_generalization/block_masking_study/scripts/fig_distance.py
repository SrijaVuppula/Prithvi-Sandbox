"""Figure 3: reconstruction error vs distance to the nearest visible summer patch.

(a) Contiguous patch-level PSNR by distance d at r = 40% (zero-shot dashed/hollow,
    fine-tuned seed 1 solid/filled); "S" = scattered hidden patches at d = 1.
(b) Geometry gap (scattered minus contiguous PSNR, zero-shot) over all hidden
    patches vs over the edge ring only (contiguous d = 1 vs scattered d = 1).
Reads outputs_finetuned/distance/distance_stats.csv (from scripts/distance_stats.py).
Writes fig_distance.pdf to the current directory.
Usage: python scripts/fig_distance.py [path/to/distance_stats.csv]
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'outputs_finetuned', 'distance', 'distance_stats.csv')

# ---- style (true print size: 8 pt here = 8 pt on the page) ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})
COL_W = 3.35
BBS = ['tiny', '100M', '300M', '600M']
CMAP = plt.get_cmap('plasma')
COL = {b: CMAP(v) for b, v in zip(BBS, (0.05, 0.34, 0.60, 0.80))}
EDGE = {b: (COL[b] if b != '600M' else '#B5651D') for b in BBS}   # darker edge for gold
INK, MUTED, GRID, BANNER = '#222222', '#666666', '#C8C8C8', '#EDEDED'
R_A = 0.4

s = pd.read_csv(CSV)
def get(**k):
    q = s
    for a, b in k.items():
        q = q[np.isclose(q[a], b)] if isinstance(b, float) else q[q[a] == b]
    return q

def banner(ax, text):
    ax.add_patch(Rectangle((0, 1.0), 1, 0.13, transform=ax.transAxes, clip_on=False,
                           facecolor=BANNER, edgecolor='none'))
    ax.text(0.012, 1.065, text, transform=ax.transAxes, ha='left', va='center',
            fontsize=7.5, fontweight='bold', color=INK)

def style(ax):
    ax.grid(True, ls=':', lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

fig, (axa, axb) = plt.subplots(2, 1, figsize=(COL_W, 3.55),
                               gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.62))
fig.subplots_adjust(left=0.13, right=0.985, top=0.935, bottom=0.10)

# ---------------- (a) PSNR vs distance, r = 40% ----------------
print(f'(a) median contiguous patch PSNR by distance, r = {int(R_A*100)}%   [S = scattered d=1]')
JIT = {'zs': -0.07, 'ft': 0.07}
for bb in BBS:
    for cond in ('zs', 'ft'):
        q = get(stat='psnr_d', backbone=bb, cond=cond, ratio=R_A, geometry='contiguous').sort_values('dist')
        sc = get(stat='psnr_d', backbone=bb, cond=cond, ratio=R_A, geometry='scattered', dist=1).iloc[0]
        filled = cond == 'ft'
        kw = dict(color=COL[bb], lw=1.1 if filled else 0.9, ls='-' if filled else (0, (3, 1.6)),
                  marker='o', ms=3.4, mew=0.8, mec=EDGE[bb],
                  mfc=COL[bb] if filled else 'white', zorder=3 if filled else 2)
        ax_x = q.dist.values + JIT[cond]
        axa.plot(ax_x, q.med.values, **kw)
        axa.plot([0 + JIT[cond]], [sc.med], **{**kw, 'ls': 'none', 'marker': 'D', 'ms': 3.2})
        print(f'  {bb:5s} {cond}: S {sc.med:5.2f} | ' +
              ' '.join(f'd{int(r.dist)} {r.med:5.2f}(n={int(r.n_chips)})' for r in q.itertuples()))
# distances reached only by blocks touching the image border (tiny/100M/300M grid)
nsub = int(get(stat='psnr_d', backbone='100M', cond='zs', ratio=R_A, geometry='contiguous', dist=4).n_chips.iloc[0])
axa.axvspan(3.5, 6.45, color='#F4F4F4', zorder=0, lw=0)
axa.text(4.97, 30.62, f'tiny–300M: {nsub} of 500 chips', ha='center', va='center', fontsize=6.3, color=MUTED)
axa.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
axa.set_xticks(range(0, 7)); axa.set_xticklabels(['S'] + [str(i) for i in range(1, 7)])
axa.set_xlim(-0.45, 6.45); axa.set_ylim(29.3, 37.9)
axa.set_xlabel('Distance $d$ to the nearest visible patch (patches)', labelpad=2)
axa.set_ylabel('PSNR (dB)', labelpad=2)
style(axa)
banner(axa, f'(a) Contiguous PSNR by distance, $r = {int(R_A*100)}\\%$')
h = [Line2D([], [], color=COL[b], lw=1.1, marker='o', ms=3.2, mec=EDGE[b], mfc=COL[b], label=b) for b in BBS]
h += [Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 1.6)), marker='o', ms=3.2, mfc='white', mec=MUTED, label='zero-shot'),
      Line2D([], [], color=MUTED, lw=1.1, marker='o', ms=3.2, mfc=MUTED, mec=MUTED, label='fine-tuned'),
      Line2D([], [], color=MUTED, ls='none', marker='D', ms=3.0, mfc='white', mec=MUTED, label='scattered')]
axa.legend(handles=h, ncol=2, loc='upper right', frameon=False, handlelength=2.2,
           columnspacing=0.9, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)

# ---------------- (b) gap over all hidden patches vs at d = 1 ----------------
print('(b) zero-shot geometry gap (dB): all hidden patches | edge ring d=1 [95% CI]')
RS = [0.2, 0.4, 0.6, 0.8]
OFF = dict(zip(BBS, (-1.8, -0.6, 0.6, 1.8)))
axb.axhline(0, color=MUTED, lw=0.6)
for bb in BBS:
    a = pd.concat([get(stat='gap_all', backbone=bb, cond='zs', ratio=r) for r in RS])
    d = pd.concat([get(stat='gap_d1', backbone=bb, cond='zs', ratio=r) for r in RS])
    x = np.array(RS) * 100 + OFF[bb]
    axb.plot(x, a.med, color=COL[bb], lw=0.8, alpha=0.75, marker='s', ms=2.8,
             mfc='white', mec=EDGE[bb], mew=0.7, zorder=2)
    axb.errorbar(x, d.med, yerr=[d.med - d.lo, d.hi - d.med], color=COL[bb], lw=1.1,
                 marker='o', ms=3.4, mfc=COL[bb], mec=EDGE[bb], mew=0.8,
                 elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
    print(f'  {bb:5s} ' + ' | '.join(f'r{int(r*100)} {ai:5.2f} {di:5.2f} [{lo:5.2f},{hi:5.2f}]'
          for r, ai, di, lo, hi in zip(RS, a.med, d.med, d.lo, d.hi)))
axb.text(63, 2.78, 'all hidden patches', ha='left', va='bottom', fontsize=6.5, color=INK)
axb.text(21, 0.45, 'edge ring ($d = 1$)', ha='left', va='bottom', fontsize=6.5, color=INK)
axb.set_xticks([20, 40, 60, 80]); axb.set_xlim(13, 87); axb.set_ylim(-0.45, 3.45)
axb.set_xlabel('Ratio $r$ (%)', labelpad=2)
axb.set_ylabel('$\\Delta$ (dB)', labelpad=2)
style(axb)
banner(axb, '(b) Geometry gap $\\Delta$: all hidden patches vs. edge ring (zero-shot)')

out = 'fig_distance.pdf'
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=300)
print('wrote', os.path.abspath(out))
