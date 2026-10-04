"""Figure 3: reconstruction error vs distance to the nearest visible summer patch.

Visual language matches Figure 1 (fig1_overview.py) and Figure 2 (fig_breakeven.py):
backbone hue = plasma at 0.05/0.34/0.60/0.80 (#2a0593, #9e199d, #e16462, #fca636),
zero-shot = hollow square + dashed line, fine-tuned = filled circle + solid line,
#EDEDED banners with centered bold titles, horizontal dotted grid, Liberation Serif.

(a) Contiguous patch-level PSNR by distance d at r = 40%; "S" = scattered hidden
    patches (all at d = 1), shown with the same markers.
(b) Zero-shot geometry gap: outlined light bar = all hidden patches, solid inner bar
    = edge ring only (contiguous d = 1 vs scattered d = 1), with 95% bootstrap CI.
Reads outputs_finetuned/distance/distance_stats.csv (from scripts/distance_stats.py).
Writes fig_distance.pdf (and a .png preview) to the current directory.
Usage: python scripts/fig_distance.py [path/to/distance_stats.csv]
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'outputs_finetuned', 'distance', 'distance_stats.csv')

# ---- style: sizes measured from Figure 1 in the compiled paper ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.4, 'axes.labelsize': 7.4,
    'xtick.labelsize': 6.9, 'ytick.labelsize': 6.9, 'legend.fontsize': 6.9,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.edgecolor': '#444444', 'axes.linewidth': 0.7,
    'xtick.color': '#444444', 'ytick.color': '#444444',
    'xtick.labelcolor': '#222222', 'ytick.labelcolor': '#222222',
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})
COL_W = 3.35
BBS = ['tiny', '100M', '300M', '600M']
COL = dict(zip(BBS, ('#2a0593', '#9e199d', '#e16462', '#fca636')))   # = Figure 1
INK, NOTE, GRID, BANNER, SHADE = '#222222', '#7a7a7a', '#C8C8C8', '#EDEDED', '#F3F3F3'
ZS = dict(marker='s', ms=3.6, mfc='white', mew=1.0, ls=(0, (3.2, 1.8)), lw=0.9)   # zero-shot
FT = dict(marker='o', ms=3.6, mew=0.0, ls='-', lw=1.1)                             # fine-tuned
R_A = 0.4

s = pd.read_csv(CSV)
def get(**k):
    q = s
    for a, b in k.items():
        q = q[np.isclose(q[a], b)] if isinstance(b, float) else q[q[a] == b]
    return q

def tint(c, f=0.72):
    return tuple(v + (1 - v) * f for v in to_rgb(c))

def banner(ax, text):
    ax.add_patch(Rectangle((0, 1.035), 1, 0.15, transform=ax.transAxes, clip_on=False,
                           facecolor=BANNER, edgecolor='none'))
    ax.text(0.5, 1.11, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=7.9, fontweight='bold', color=INK)

def style(ax):
    ax.grid(True, axis='y', ls=':', lw=0.6, color=GRID)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

fig, (axa, axb) = plt.subplots(2, 1, figsize=(COL_W, 3.75),
                               gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.66))
fig.subplots_adjust(left=0.125, right=0.985, top=0.92, bottom=0.09)

# ---------------- (a) PSNR vs distance, r = 40% ----------------
print(f'(a) median contiguous patch PSNR by distance, r = {int(R_A*100)}%   [S = scattered, d=1]')
JIT = {'zs': -0.08, 'ft': 0.08}
for bb in BBS:
    for cond, st in (('zs', ZS), ('ft', FT)):
        q = get(stat='psnr_d', backbone=bb, cond=cond, ratio=R_A, geometry='contiguous').sort_values('dist')
        sc = get(stat='psnr_d', backbone=bb, cond=cond, ratio=R_A, geometry='scattered', dist=1).iloc[0]
        kw = dict(st, color=COL[bb], mec=COL[bb], zorder=3 if cond == 'ft' else 2)
        if cond == 'ft':
            kw['mfc'] = COL[bb]
        axa.plot(q.dist.values + JIT[cond], q.med.values, **kw)
        axa.plot([JIT[cond]], [sc.med], **{**kw, 'ls': 'none'})
        print(f'  {bb:5s} {cond}: S {sc.med:5.2f} | ' +
              ' '.join(f'd{int(r.dist)} {r.med:5.2f}(n={int(r.n_chips)})' for r in q.itertuples()))
nsub = int(get(stat='psnr_d', backbone='100M', cond='zs', ratio=R_A,
               geometry='contiguous', dist=4).n_chips.iloc[0])
axa.axvspan(3.5, 6.45, color=SHADE, zorder=0, lw=0)
axa.text(6.4, 28.72, f'tiny–300M: {nsub} of 500 chips',
         ha='right', va='bottom', fontsize=6.3, style='italic', color=NOTE)
axa.axvline(0.5, color=NOTE, lw=0.6, ls=(0, (1, 1.5)))
axa.text(0, 28.72, 'scattered', ha='center', va='bottom', fontsize=6.3, style='italic', color=NOTE)
axa.set_xticks(range(0, 7)); axa.set_xticklabels(['S'] + [str(i) for i in range(1, 7)])
axa.set_xlim(-0.5, 6.45); axa.set_ylim(28.6, 37.9)
axa.set_xlabel('Distance $d$ to the nearest visible patch (patches)', labelpad=2)
axa.set_ylabel('PSNR (dB)', labelpad=2)
style(axa)
banner(axa, f'(a) Contiguous PSNR by distance, r = {int(R_A*100)}%')
h = [Line2D([], [], ls='none', marker='o', ms=4.2, mfc=COL[b], mec=COL[b], label=b) for b in BBS]
h += [Line2D([], [], color='#7a7a7a', mec='#7a7a7a', **{**ZS, 'ms': 3.8}, label='zero-shot'),
      Line2D([], [], color='#7a7a7a', mec='#7a7a7a', mfc='#7a7a7a', **{**FT, 'ms': 3.8}, label='fine-tuned')]
axa.legend(handles=h, ncol=2, loc='upper right', frameon=False, handlelength=2.0,
           columnspacing=0.8, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.1)

# ---------------- (b) zero-shot gap: all hidden patches vs edge ring ----------------
print('(b) zero-shot geometry gap (dB): all hidden patches | edge ring d=1 [95% CI]')
RS = [0.2, 0.4, 0.6, 0.8]
W_OUT, W_IN, STEP = 0.19, 0.095, 0.205
for i, bb in enumerate(BBS):
    a = pd.concat([get(stat='gap_all', backbone=bb, cond='zs', ratio=r) for r in RS])
    d = pd.concat([get(stat='gap_d1', backbone=bb, cond='zs', ratio=r) for r in RS])
    x = np.arange(len(RS)) + (i - 1.5) * STEP
    axb.bar(x, a.med, width=W_OUT, color=tint(COL[bb]), edgecolor=COL[bb], linewidth=0.6, zorder=2)
    axb.bar(x, d.med, width=W_IN, color=COL[bb], linewidth=0, zorder=3)
    axb.errorbar(x, d.med, yerr=[d.med - d.lo, d.hi - d.med], fmt='none', color=INK, ecolor=INK,
                 elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=4)
    print(f'  {bb:5s} ' + ' | '.join(f'r{int(r*100)} {ai:5.2f} {di:5.2f} [{lo:5.2f},{hi:5.2f}]'
          for r, ai, di, lo, hi in zip(RS, a.med, d.med, d.lo, d.hi)))
axb.axhline(0, color='#444444', lw=0.7, zorder=5)
axb.set_xticks(range(len(RS))); axb.set_xticklabels([str(int(r * 100)) for r in RS])
axb.set_xlim(-0.5, len(RS) - 0.5); axb.set_ylim(-0.35, 3.8)
axb.set_xlabel('Mask ratio $r$ (%)', labelpad=2)
axb.set_ylabel('$\\Delta$ (dB)', labelpad=2)
style(axb)
banner(axb, '(b) Geometry gap \u0394, zero-shot')
gray = '#7a7a7a'
hb = [Patch(facecolor=tint(gray), edgecolor=gray, linewidth=0.6, label='all hidden patches'),
      Patch(facecolor=gray, edgecolor='none', label='edge ring ($d = 1$)')]
axb.legend(handles=hb, ncol=2, loc='upper right', frameon=False, handlelength=1.4,
           handleheight=0.9, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1)

out = 'fig_distance.pdf'
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=300)
print('wrote', os.path.abspath(out))
