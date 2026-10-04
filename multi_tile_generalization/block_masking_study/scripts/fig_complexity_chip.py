"""Figure: zero-shot vs fine-tuned reconstructions of one chip for all four backbones.

Chip chosen by a fixed rule: complexity score (outputs/chip_complexity_v2.csv) closest to
the median of the 500 evaluation chips. r = 40%, trial 0 (fixed in advance), both geometries.
Per backbone: zero-shot patch PSNR (gray tiles, dark = low) and the fine-tuning gain per patch,
PSNR(fine-tuned) - PSNR(zero-shot), with the fine-tuned patch MSE averaged over the ten runs
(teal = gain, magenta = loss, the colors of Figure 1c). Above the panels: masked-region PSNR
of trial 0 (zero-shot) and the gain G of the ten-run mean over it (fine-tuned PSNR = zero-shot + G).
Visual language matches Figures 1-3.
Reads outputs_finetuned/distance/cchip_{bb}.npz and cchip_psnr.csv (qual_complexity_chip.py).
Writes fig_complexity_chip.pdf (+ .png). Usage: python scripts/fig_complexity_chip.py [dir]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'outputs_finetuned', 'distance')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
FULL_W = 6.875
BBS = ['tiny', '100M', '300M', '600M']
COL = {'tiny': '#2a0593', '100M': '#9e199d', '300M': '#e16462', '600M': '#fca636'}   # = Figure 1
INK, NOTE, BANNER = '#222222', '#7a7a7a', '#EDEDED'
GEOS = ['contiguous', 'scattered']
PMIN, PMAX = 26.0, 42.0
CMAP = LinearSegmentedColormap.from_list('psnr', ['#111111', '#4A4A4A', '#8C8C8C', '#CFCFCF', '#FFFFFF'])
GMAP = LinearSegmentedColormap.from_list('gain', ['#C23B75', '#FFFFFF', '#159AA8'])   # = Figure 1c
GLIM = 2.0                                    # gain scale (dB), symmetric

Z = {b: np.load(os.path.join(D, f'cchip_{b}.npz')) for b in BBS}
P_ = pd.read_csv(os.path.join(D, 'cchip_psnr.csv'))
P_['trial'] = P_.trial.astype(int)
T0 = int(Z['tiny']['trials'][0])
chip = str(Z['tiny']['chip']); rank = int(Z['tiny']['rank'])

def rgb(x, lo, hi):
    return np.clip((x[[2, 1, 0]].transpose(1, 2, 0) - lo) / (hi - lo), 0, 1)

def tiles(pm, m, ps):
    pp = np.kron(10 * np.log10(1 / np.maximum(pm, 1e-12)), np.ones((ps, ps)))
    return np.ma.masked_where(~m, pp)

# ---- layout (inches) ----
LEFT, RIGHT_CB, TOP, BOT = 0.30, 0.98, 0.33, 0.04
GAP, GGAP, ROWGAP = 0.035, 0.10, 0.17
ncol = 1 + 2 * len(BBS)
P = (FULL_W - LEFT - RIGHT_CB - (ncol - 1) * GAP - len(BBS) * GGAP) / ncol
H = TOP + 2 * P + ROWGAP + ROWGAP + BOT
fig = plt.figure(figsize=(FULL_W, H))
def box(x, y, w, h):
    return fig.add_axes([x / FULL_W, 1 - (y + h) / H, w / FULL_W, h / H])
def col_x(j):
    return LEFT + j * (P + GAP) + GGAP * ((j + 1) // 2)
def fig_text(x, y, s, **kw):
    return fig.text(x / FULL_W, 1 - y / H, s, **kw)
def fig_rect(x, y, w, h, **kw):
    fig.add_artist(Rectangle((x / FULL_W, 1 - (y + h) / H), w / FULL_W, h / H,
                             transform=fig.transFigure, **kw))

# headers
fig_rect(col_x(0), 0.02, P, 0.17, facecolor=BANNER, edgecolor='none')
fig_text(col_x(0) + P / 2, 0.105, 'Summer', ha='center', va='center', fontsize=7.9, fontweight='bold', color=INK)
fig_text(col_x(0) + P / 2, 0.29, 'ground truth', ha='center', va='center', fontsize=6.9, color=INK)
HEAD = []
for k, b in enumerate(BBS):
    x0 = col_x(1 + 2 * k); x1 = col_x(2 + 2 * k) + P
    fig_rect(x0, 0.02, x1 - x0, 0.17, facecolor=BANNER, edgecolor='none')
    HEAD.append((fig_text((x0 + x1) / 2 + 0.045, 0.105, b, ha='center', va='center', fontsize=7.9,
                          fontweight='bold', color=INK), COL[b]))
    for j, lab in enumerate(['zero-shot', 'fine-tuning gain']):
        fig_text(col_x(1 + 2 * k + j) + P / 2, 0.29, lab, ha='center', va='center', fontsize=6.9, color=INK)
fig.canvas.draw(); rend = fig.canvas.get_renderer()
for t, c in HEAD:
    bb_ = t.get_window_extent(rend).transformed(fig.transFigure.inverted())
    fig.add_artist(plt.Line2D([bb_.x0 - 0.06 / FULL_W], [1 - 0.105 / H], marker='o', ms=4.2, color=c,
                              transform=fig.transFigure))

gt = Z['100M']['gt'].astype(np.float64)
lo, hi = np.percentile(gt[[2, 1, 0]], [2, 98])
img = rgb(gt, lo, hi)
faded = 0.30 * img + 0.70
y0 = TOP + ROWGAP
ax = box(col_x(0), y0 + (P + ROWGAP) / 2, P, P); ax.imshow(img, interpolation='nearest'); ax.set_axis_off()

print(f'{chip} (complexity rank {rank}/500), r = 40%, trial {T0}')
for gi, geo in enumerate(GEOS):
    y = y0 + gi * (P + ROWGAP)
    fig_rect(0.02, y, 0.17, P, facecolor=BANNER, edgecolor='none')
    fig_text(0.105, y + P / 2, geo.capitalize(), ha='center', va='center', rotation=90, fontsize=7.4,
             fontweight='bold', color=INK)
    for k, b in enumerate(BBS):
        z = Z[b]; ps = int(z['patch']); m = z[f'{geo}|mask']
        q = P_[(P_.backbone == b) & (P_.geometry == geo) & (P_.trial == T0)]
        p_zs = float(q[q.cond == 'zs'].psnr.iloc[0]); p_ft = float(q[q.cond == 'ft'].psnr.mean())
        pz, pf = z['pm_zs'][0, gi], z['pm_ft'][:, 0, gi].mean(0)
        gain = np.ma.masked_where(~m, np.kron(10 * np.log10(np.maximum(pz, 1e-12) / np.maximum(pf, 1e-12)),
                                              np.ones((ps, ps))))
        for j, lab in enumerate([f'{p_zs:.2f} dB', f'G {p_ft - p_zs:+.2f} dB'.replace('-', '\u2212')]):
            ax = box(col_x(1 + 2 * k + j), y, P, P)
            ax.imshow(faded, interpolation='nearest')
            if j == 0:
                ax.imshow(tiles(pz, m, ps), cmap=CMAP, vmin=PMIN, vmax=PMAX, interpolation='nearest')
            else:
                ax.imshow(gain, cmap=GMAP, vmin=-GLIM, vmax=GLIM, interpolation='nearest')
            if geo == 'contiguous':
                ax.contour(m.astype(float), levels=[0.5], colors=[NOTE], linewidths=0.4)
            ax.set_axis_off()
            fig_text(col_x(1 + 2 * k + j) + P / 2, y - 0.012, lab, ha='center', va='bottom', fontsize=6.3, color=INK)
        print(f'  {b:5s} {geo:10s} zero-shot {p_zs:6.2f} | fine-tuned (mean of 10 runs) {p_ft:6.2f} | G {p_ft - p_zs:+.2f}')

for c, (cm, lo_, hi_, ticks, lab) in enumerate([(CMAP, PMIN, PMAX, [26, 30, 34, 38, 42], 'zero-shot patch PSNR (dB)'),
                                                (GMAP, -GLIM, GLIM, [-2, -1, 0, 1, 2], 'fine-tuning gain (dB)')]):
    cax = box(FULL_W - RIGHT_CB + 0.10 + c * 0.47, y0, 0.07, 2 * P + ROWGAP)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(lo_, hi_))
    cb = fig.colorbar(sm, cax=cax, extend='both')
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t:g}'.replace('-', '\u2212') for t in ticks])
    cb.ax.tick_params(labelsize=6.3, width=0.6, length=2, color='#444444')
    cb.outline.set_linewidth(0.5)
    cb.set_label(lab, fontsize=6.9, labelpad=2)

out = 'fig_complexity_chip.pdf'
fig.savefig(out, dpi=600)
fig.savefig(out.replace('.pdf', '.png'), dpi=300)
print('wrote', os.path.abspath(out), f'({FULL_W:.3f} x {H:.2f} in)')
