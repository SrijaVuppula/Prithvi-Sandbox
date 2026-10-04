"""Figure 4: qualitative reconstructions on real HLS chips (zero-shot, r = 40%, one trial).

Rows: three evaluation chips chosen by rule (25th / 50th / 75th percentile of zero-shot
100M contiguous PSNR at r = 40%, mean over the chip's five trials).
Columns: summer ground truth (RGB) | contiguous: masked input, then patch-level PSNR of
tiny, 100M, 300M | scattered: the same. Each hidden 16-px patch is a gray tile (PSNR over
its pixels and six bands, reflectance; dark = low), drawn on a pale copy of the scene; the
masked-region PSNR of the trial is printed above each panel (as in Tables 1-2).
Visual language matches Figures 1-3 (hidden = #23233A, #EDEDED banners, backbone dots).
Reads outputs_finetuned/distance/qual_{bb}_zs.npz (tiny, 100M, 300M; 600M uses a 14-px
grid, so its hidden region differs and it is left out). Writes fig_qualitative.pdf (+ .png).
Usage: python scripts/fig_qualitative.py [distance_dir]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'outputs_finetuned', 'distance')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
FULL_W = 6.875
BBS = ['tiny', '100M', '300M']        # same 16-px patch grid, so all share the input mask
COL = {'tiny': '#2a0593', '100M': '#9e199d', '300M': '#e16462', '600M': '#fca636'}   # = Figure 1
HIDDEN, INK, NOTE, BANNER = '#23233A', '#222222', '#7a7a7a', '#EDEDED'
CHIPS = [('chip_218_423_merged.tif', 'harder (P25)'),
         ('chip_167_411_merged.tif', 'median (P50)'),
         ('chip_200_441_merged.tif', 'easier (P75)')]
GEOS = ['contiguous', 'scattered']
PMIN, PMAX = 26.0, 42.0                       # shared patch-PSNR scale (dB)
PATCH = 16                                    # tiny/100M/300M patch size (px)
CMAP = LinearSegmentedColormap.from_list('psnr', ['#111111', '#4A4A4A', '#8C8C8C', '#CFCFCF', '#FFFFFF'])  # dark = low PSNR

Z = {b: np.load(os.path.join(D, f'qual_{b}_zs.npz')) for b in BBS}

def rgb(x, lo, hi):
    return np.clip((x[[2, 1, 0]].transpose(1, 2, 0) - lo) / (hi - lo), 0, 1)

# ---- layout (inches) ----
LEFT, RIGHT_CB, TOP, BOT = 0.30, 0.50, 0.33, 0.04
GAP, GGAP, ROWGAP = 0.035, 0.12, 0.17
ncol = 1 + 4 * len(GEOS)
P = (FULL_W - LEFT - RIGHT_CB - (ncol - 1) * GAP - 2 * GGAP) / ncol    # panel size
H = TOP + len(CHIPS) * P + (len(CHIPS) - 1) * ROWGAP + ROWGAP + BOT
fig = plt.figure(figsize=(FULL_W, H))
def box(x, y, w, h):        # inches from top-left -> figure fraction axes
    return fig.add_axes([x / FULL_W, 1 - (y + h) / H, w / FULL_W, h / H])
def col_x(j):
    return LEFT + j * (P + GAP) + (GGAP if j >= 1 else 0) + (GGAP if j >= 5 else 0)

def fig_text(x, y, s, **kw):
    fig.text(x / FULL_W, 1 - y / H, s, **kw)

def fig_rect(x, y, w, h, **kw):
    fig.add_artist(Rectangle((x / FULL_W, 1 - (y + h) / H), w / FULL_W, h / H,
                             transform=fig.transFigure, **kw))

# group banners and column titles
HEAD = []
fig_rect(col_x(0), 0.02, P, 0.17, facecolor=BANNER, edgecolor='none')
fig_text(col_x(0) + P / 2, 0.105, 'Summer', ha='center', va='center', fontsize=7.9, fontweight='bold', color=INK)
for g, geo in enumerate(GEOS):
    x0 = col_x(1 + 4 * g); x1 = col_x(4 + 4 * g) + P
    fig_rect(x0, 0.02, x1 - x0, 0.17, facecolor=BANNER, edgecolor='none')
    fig_text((x0 + x1) / 2, 0.105, f'{geo.capitalize()}, r = 40%', ha='center', va='center',
             fontsize=7.9, fontweight='bold', color=INK)
    for k, lab in enumerate(['masked input'] + BBS):
        xc = col_x(1 + 4 * g + k) + P / 2
        t = fig.text((xc + (0.045 if k else 0)) / FULL_W, 1 - 0.29 / H, lab, ha='center', va='center',
                     fontsize=7.4 if k else 6.9, color=INK, fontweight='bold' if k else 'normal')
        if k:
            HEAD.append((t, COL[BBS[k - 1]]))
fig_text(col_x(0) + P / 2, 0.29, 'ground truth', ha='center', va='center', fontsize=6.9, color=INK)
# backbone dots just left of each name (measured), as in Figure 1(c)
fig.canvas.draw(); rend = fig.canvas.get_renderer()
for t, c in HEAD:
    bb_ = t.get_window_extent(rend).transformed(fig.transFigure.inverted())
    fig.add_artist(plt.Line2D([bb_.x0 - 0.06 / FULL_W], [1 - 0.29 / H], marker='o', ms=4.2, color=c,
                              transform=fig.transFigure))

print('chip, geometry, backbone, PSNR (dB) of the stored trial')
for i, (ch, rowlab) in enumerate(CHIPS):
    y = TOP + i * (P + ROWGAP) + ROWGAP
    gt = Z['100M'][f'{ch}|gt'].astype(np.float64)
    lo, hi = np.percentile(gt[[2, 1, 0]], [2, 98])          # fixed stretch per chip
    img = rgb(gt, lo, hi)
    faded = 0.30 * img + 0.70                                        # pale color context; tiles are gray
    # row label (vertical banner, as in Figure 1a)
    fig_rect(0.02, y, 0.17, P, facecolor=BANNER, edgecolor='none')
    fig_text(0.105, y + P / 2, rowlab, ha='center', va='center', rotation=90, fontsize=7.4,
             fontweight='bold', color=INK)
    ax = box(col_x(0), y, P, P); ax.imshow(img, interpolation='nearest'); ax.set_axis_off()
    for g, geo in enumerate(GEOS):
        m_in = Z['100M'][f'{ch}|{geo}|mask']
        inp = img.copy(); inp[m_in] = matplotlib.colors.to_rgb(HIDDEN)
        ax = box(col_x(1 + 4 * g), y, P, P); ax.imshow(inp, interpolation='nearest'); ax.set_axis_off()
        for k, b in enumerate(BBS):
            z = Z[b]
            m = z[f'{ch}|{geo}|mask']
            se = ((z[f'{ch}|{geo}|rec'].astype(np.float64) - gt) ** 2).mean(0)
            g_ = se.reshape(224 // PATCH, PATCH, 224 // PATCH, PATCH).mean((1, 3))     # patch MSE
            pp = np.kron(10 * np.log10(1 / np.maximum(g_, 1e-12)), np.ones((PATCH, PATCH)))  # patch PSNR map (visible patches: error 0, not drawn)
            ax = box(col_x(2 + 4 * g + k), y, P, P)
            ax.imshow(faded, interpolation='nearest')
            ax.imshow(np.ma.masked_where(~m, pp), cmap=CMAP, vmin=PMIN, vmax=PMAX, interpolation='nearest')
            if geo == 'contiguous':
                ax.contour(m.astype(float), levels=[0.5], colors=[NOTE], linewidths=0.4)
            ax.set_axis_off()
            p = float(z[f'{ch}|{geo}|psnr'])
            fig_text(col_x(2 + 4 * g + k) + P / 2, y - 0.012, f'{p:.2f} dB', ha='center', va='bottom',
                     fontsize=6.3, color=INK)
            print(f'  {ch[5:12]} {geo:10s} {b:5s} {p:6.2f}')

# shared colorbar
cy = TOP + ROWGAP; ch_ = len(CHIPS) * P + (len(CHIPS) - 1) * ROWGAP
cax = box(FULL_W - RIGHT_CB + 0.10, cy, 0.07, ch_)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(PMIN, PMAX))
cb = fig.colorbar(sm, cax=cax, extend='both')
cb.set_ticks([26, 30, 34, 38, 42])
cb.ax.tick_params(labelsize=6.3, width=0.6, length=2, color='#444444')
cb.outline.set_linewidth(0.5)
cb.set_label('patch PSNR (dB)', fontsize=6.9, labelpad=2)

out = 'fig_qualitative.pdf'
fig.savefig(out, dpi=600)
fig.savefig(out.replace('.pdf', '.png'), dpi=300)
print('wrote', os.path.abspath(out), f'({FULL_W:.3f} x {H:.2f} in)')
