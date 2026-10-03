"""
fig1_overview.py -- Figure 1 (full width, 7.0 in). Usage:  python fig1_overview.py [classic|viridis|plasma|all]

(a) same fields in three seasons (schematic false-colour scene), summer frame masked, two geometries
(b) quality vs. energy: zero-shot scaling curve, fine-tuning moves a model straight up (same energy)
(c) fine-tuning gain G, both geometries
Colour roles: hue = backbone size, hollow = zero-shot, filled = fine-tuned (palettes 'viridis'/'plasma');
'classic' keeps blue = fine-tuned, gray = zero-shot, vermillion = masked.
All numbers: 10-seed means (DATA block printed by scripts/fig1_data_block.py).
"""
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.4,
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
})
BANNER, INK, LGRAY = "#EDEDED", "#222222", "#E4E4E4"

# ------------------------------------------------------------------ DATA (10-seed means)
BB = ["tiny", "100M", "300M", "600M"]
ENERGY_J = {"tiny": 1.7436, "100M": 3.2142, "300M": 7.2190, "600M": 19.6145}   # J / pass, r = 40%
PSNR_ZS = {"tiny": 31.32, "100M": 33.02, "300M": 33.22, "600M": 33.93}
PSNR_FT = {"tiny": 32.25, "100M": 32.98, "300M": 33.38, "600M": 33.89}
G_CONT = np.array([[1.055, .863, .804, .777], [.016, .029, .059, .141], [.308, .158, .108, .181], [.062, .019, .056, .173]])
G_SCAT = np.array([[.499, .253, .153, .198], [-.140, -.201, -.210, -.117], [-.038, -.183, -.253, -.187], [.008, -.199, -.319, -.231]])
M = "\u2212"
T_CONT = [["1.05", "0.86", "0.80", "0.78"], ["0.02", "0.03", "0.06", "0.14"], ["0.31", "0.16", "0.11", "0.18"], ["0.06", "0.02", "0.06", "0.17"]]
T_SCAT = [["0.50", "0.25", "0.15", "0.20"], [M + "0.14", M + "0.20", M + "0.21", M + "0.12"], [M + "0.04", M + "0.18", M + "0.25", M + "0.19"], ["0.01", M + "0.20", M + "0.32", M + "0.23"]]
D_CONT = [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
D_SCAT = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]

# ------------------------------------------------------------------ palettes
def cm_slice(name, a, b):
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(name + "_s", base(np.linspace(a, b, 256)))
def hexs(name, vals):
    return [mpl.colors.to_hex(plt.get_cmap(name)(v)) for v in vals]

PALETTES = {
    "classic": dict(mode="semantic", mask="#D55E00", ft="#0072B2", zs="#8A8A8A", bb=None,
                    pos="#0072B2", neg="#CC79A7", shade="#0072B2",
                    scene=LinearSegmentedColormap.from_list("nat", ["#8C6D46", "#D9C28F", "#B9C45A", "#5FAE46", "#1E6B36"])),
    "viridis": dict(mode="backbone", mask="#E8563A", zs="#8A8A8A", bb=hexs("viridis", (0.06, 0.36, 0.60, 0.80)),
                    pos="#2F6DB5", neg="#D1495B", shade=None, scene=cm_slice("viridis", 0.04, 0.96)),
    "plasma": dict(mode="backbone", mask="#23233A", zs="#8A8A8A", bb=hexs("plasma", (0.05, 0.34, 0.60, 0.80)),
                   pos="#159AA8", neg="#C23B75", shade=None, scene=cm_slice("plasma", 0.04, 0.96)),
}

# ------------------------------------------------------------------ schematic scene + masks
N = 14
def make_scene(n=N, px=8, nf=30, seed=11):
    rng = np.random.default_rng(seed)
    S = n * px
    pts = rng.random((nf, 2)) * S
    yy, xx = np.mgrid[0:S, 0:S]
    fid = ((yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2).argmin(-1)
    ctype = rng.choice(4, nf, p=[.25, .35, .25, .15])
    jit = rng.normal(0, .07, (nf, 3))
    Gs = np.array([[.80, .30, .06], [.10, .96, .40], [.55, .68, .44], [.08, .18, .10]])   # greenness: spring, summer, fall
    noise = rng.normal(0, .035, (3, S, S))
    return [np.clip(Gs[ctype[fid], k] + jit[fid, k] + noise[k], 0, 1) for k in range(3)]

def example_masks(seed=7, n=N, h=3, w=13):
    rng = np.random.default_rng(seed)
    cont = np.zeros((n, n), bool); cont[7:7 + h, 1:1 + w] = True
    sc = np.zeros(n * n, bool); sc[rng.choice(n * n, cont.sum(), replace=False)] = True
    return cont, sc.reshape(n, n)

# ------------------------------------------------------------------ figure
def build(name):
    P = PALETTES[name]
    bbmode = P["mode"] == "backbone"
    W, H = 7.0, 2.72
    fig = plt.figure(figsize=(W, H))
    ax_in = lambda x, y, w, h, **kw: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)
    def rect(x, y, w, h, **kw):
        fig.patches.append(Rectangle((x / W, 1 - (y + h) / H), w / W, h / H, transform=fig.transFigure, **kw))
    def text(x, y, s, **kw):
        fig.text(x / W, 1 - y / H, s, **kw)
    def banner(x, y, w, s, h=0.22):
        rect(x, y, w, h, facecolor=BANNER, edgecolor="none")
        text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    col_bb = (lambda b: P["bb"][BB.index(b)]) if bbmode else (lambda b: P["ft"])
    col_zs = (lambda b: P["bb"][BB.index(b)]) if bbmode else (lambda b: P["zs"])
    hero = col_bb("tiny")

    # ============================================================ (a) seasons + masks
    PA_X, PA_W = 0.06, 2.10
    banner(PA_X, 0.05, PA_W, "(a) Occlusion, r = 20%")
    scene = make_scene()
    cont, scat = example_masks()
    TS, GAP, X0 = 0.60, 0.045, PA_X + 0.26
    heads = ["Spring", "Summer (masked)", "Fall"]
    for k, hname in enumerate(heads):
        text(X0 + k * (TS + GAP) + TS / 2, 0.385, hname, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
    for i, (mask, label) in enumerate([(cont, "Contiguous"), (scat, "Scattered")]):
        y0 = 0.49 + i * (TS + 0.09)
        rect(PA_X, y0, 0.20, TS, facecolor=BANNER, edgecolor="none")
        text(PA_X + 0.10, y0 + TS / 2, label, rotation=90, ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK)
        for k in range(3):
            ax = ax_in(X0 + k * (TS + GAP), y0, TS, TS)
            ax.imshow(P["scene"](scene[k]), extent=(0, N, 0, N), origin="upper", interpolation="nearest")
            if k == 1:
                for r in range(N):
                    for c in range(N):
                        if mask[r, c]:
                            ax.add_patch(Rectangle((c, N - 1 - r), 1, 1, facecolor=P["mask"], edgecolor="none"))
            lines = [[(j, 0), (j, N)] for j in range(1, N)] + [[(0, j), (N, j)] for j in range(1, N)]
            ax.add_collection(LineCollection(lines, colors="white", linewidths=0.22, alpha=0.6))
            ax.add_patch(Rectangle((0, 0), N, N, fill=False, edgecolor="#444444", linewidth=0.7))
            ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect("equal"); ax.axis("off")
    yl = 0.49 + 2 * TS + 0.09 + 0.22
    rect(PA_X + 0.26, yl - 0.055, 0.11, 0.11, facecolor=P["mask"], edgecolor="#777777", linewidth=0.4)
    text(PA_X + 0.26 + 0.16, yl, "hidden: 39 of 196 summer patches", ha="left", va="center", fontsize=6.6, color=INK)
    text(PA_X + 0.26, yl + 0.19, "spring and fall stay fully visible", ha="left", va="center", fontsize=6.4,
         style="italic", color="#555555")
    text(PA_X + 0.26, yl + 0.40, "schematic scene; same fields, three seasons", ha="left", va="center", fontsize=6.0,
         style="italic", color="#888888")

    # ============================================================ (b) quality vs energy
    PB_X, PB_W = 2.30, 2.16
    banner(PB_X, 0.05, PB_W, "(b) Quality vs. energy, r = 40%")
    ax = ax_in(PB_X + 0.43, 0.40, PB_W - 0.50, 1.78)
    ax.set_xscale("log"); ax.set_xlim(1.35, 27); ax.set_ylim(30.85, 34.45)
    ax.set_xticks([2, 5, 10, 20]); ax.set_xticklabels(["2", "5", "10", "20"]); ax.set_yticks([31, 32, 33, 34])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle=(0, (1, 2))); ax.set_axisbelow(True)
    ax.set_xlabel("Energy per forward pass (J, log scale)", labelpad=2)
    ax.set_ylabel("Contiguous PSNR (dB)", labelpad=2)
    xs_ = np.array([ENERGY_J[b] for b in BB]); ys_ = np.array([PSNR_ZS[b] for b in BB])
    xx = np.geomspace(xs_[0], xs_[-1], 300)
    yy = np.interp(np.log(xx), np.log(xs_), ys_)
    ax.fill_between(xx, yy, 34.45, color=P["shade"] or hero, alpha=0.085, lw=0, zorder=0)
    ax.plot(xs_, ys_, color="#A0A0A0", lw=0.9, ls=(0, (3, 2)), zorder=1)
    e_t, e_h = ENERGY_J["tiny"], ENERGY_J["100M"]
    ax.plot([e_t, e_h], [PSNR_ZS["100M"]] * 2, color="#9A9A9A", lw=0.7, ls=(0, (1.5, 2)), zorder=1)
    ax.plot([e_t, e_t], [PSNR_FT["tiny"], PSNR_ZS["100M"]], color="#9A9A9A", lw=0.7, ls=(0, (1.5, 2)), zorder=1)
    for b in BB:
        e = ENERGY_J[b]
        if b != "tiny":
            ax.plot([e, e], [PSNR_ZS[b], PSNR_FT[b]], color="#B5B5B5", lw=1.0, zorder=2)
        ax.scatter([e], [PSNR_ZS[b]], marker="s", s=40, facecolor="white", edgecolor=col_zs(b), linewidth=1.5, zorder=3)
        ax.scatter([e], [PSNR_FT[b]], marker="o", s=30, facecolor=col_bb(b), edgecolor="white", linewidth=0.6, zorder=4)
    ax.annotate("", xy=(e_t, PSNR_FT["tiny"] - 0.03), xytext=(e_t, PSNR_ZS["tiny"] + 0.12),
                arrowprops=dict(arrowstyle="-|>", color=hero, lw=1.8, mutation_scale=9, shrinkA=0, shrinkB=0), zorder=3)
    ax.text(e_t, PSNR_ZS["tiny"] - 0.20, "tiny", ha="center", va="top", fontsize=7.5, color=INK)
    ax.text(e_h, PSNR_ZS["100M"] + 0.20, "100M", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.text(ENERGY_J["300M"], PSNR_FT["300M"] + 0.20, "300M", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.text(ENERGY_J["600M"], PSNR_ZS["600M"] + 0.20, "600M", ha="center", va="bottom", fontsize=7.5, color=INK)
    closed = (PSNR_FT["tiny"] - PSNR_ZS["tiny"]) / (PSNR_ZS["100M"] - PSNR_ZS["tiny"]) * 100
    saved = (1 - e_t / e_h) * 100
    ax.text(e_t * 1.13, 31.95, f"{closed:.0f}% of the gap to\n100M closed at\n{saved:.0f}% less energy", ha="left", va="center",
            fontsize=7, color=hero, fontweight="bold", linespacing=1.15)
    ax.text(1.5, 34.28, "above the zero-shot curve", ha="left", va="center", fontsize=6.5, style="italic",
            color=P["shade"] or hero, alpha=0.95)
    ax.text(4.6, 32.66, "zero-shot\nscaling", ha="center", va="center", fontsize=6.3, style="italic", color="#8A8A8A", linespacing=1.1)
    big = max(abs(PSNR_FT[b] - PSNR_ZS[b]) for b in ("100M", "300M", "600M"))
    ax.text(24.5, 32.30, f"larger backbones:\nwithin \u00b1{big:.2f} dB", ha="right", va="center", fontsize=6.5,
            style="italic", color="#666666", linespacing=1.15)
    lz = P["zs"] if not bbmode else "#7A7A7A"
    h1 = Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=lz, markeredgewidth=1.4, markersize=5.5)
    h2 = Line2D([], [], marker="o", ls="", markerfacecolor=(P["ft"] if not bbmode else "#7A7A7A"), markeredgecolor="white", markersize=5)
    ax.legend([h1, h2], ["zero-shot", "fine-tuned"], loc="lower right", frameon=False, fontsize=7, handletextpad=0.3,
              labelspacing=0.3, borderaxespad=0.1, bbox_to_anchor=(1.02, 0.0))

    # ============================================================ (c) gain heat map
    PC_X, PC_W = 4.60, 2.36
    banner(PC_X, 0.05, PC_W, "(c) Fine-tuning gain G (dB)")
    cmap = LinearSegmentedColormap.from_list("gain", [(0, P["neg"]), (0.5, "white"), (1, P["pos"])])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.40, vmax=1.10)
    CW, CH = 0.225, 0.30
    HX = [PC_X + 0.46, PC_X + 0.46 + 4 * CW + 0.08]
    HY = 0.66
    for k, (G, T, D, title) in enumerate([(G_CONT, T_CONT, D_CONT, "Contiguous"), (G_SCAT, T_SCAT, D_SCAT, "Scattered")]):
        x0 = HX[k]
        rect(x0, 0.36, 4 * CW, 0.20, facecolor=BANNER, edgecolor="none")
        text(x0 + 2 * CW, 0.46, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
        a = ax_in(x0, HY, 4 * CW, 4 * CH)
        a.imshow(G, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", extent=(0, 4, 4, 0))
        for i in range(5):
            a.axhline(i, color="white", lw=1.3); a.axvline(i, color="white", lw=1.3)
        for i in range(4):
            for j in range(4):
                v = G[i, j]; c = "white" if v > 0.62 or v < -0.30 else INK
                a.text(j + 0.5, i + 0.53, T[i][j], ha="center", va="center", fontsize=6.4, color=c)
                if D[i][j]:
                    a.text(j + 0.94, i + 0.20, "\u2020", ha="right", va="center", fontsize=5.2, color=c)
        a.set_xticks(np.arange(4) + 0.5); a.set_xticklabels(["20", "40", "60", "80"]); a.tick_params(length=0, pad=2)
        a.set_yticks([]) if k else (a.set_yticks(np.arange(4) + 0.5), a.set_yticklabels(BB), a.tick_params(axis="y", pad=2))
        for s in a.spines.values():
            s.set_visible(False)
    if bbmode:     # colour dots tie the rows to panel (b)
        for i, b in enumerate(BB):
            yc = HY + (i + 0.5) * CH
            fig.patches.append(Ellipse(((HX[0] - 0.39) / W, 1 - yc / H), 0.085 / W, 0.085 / H, transform=fig.transFigure,
                                       facecolor=col_bb(b), edgecolor="white", linewidth=0.4))
    text(HX[0] + 4 * CW + 0.04, HY + 4 * CH + 0.28, "Mask ratio (%)", ha="center", va="center", fontsize=7.5)
    cb_y = HY + 4 * CH + 0.46
    cw_ = 2 * 4 * CW + 0.08 - 0.30
    cax = ax_in(HX[0] + 0.15, cb_y, cw_, 0.07)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-0.4, 0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5); cb.outline.set_linewidth(0.4)
    cb.ax.set_xticklabels([M + "0.4", "0", "0.5", "1.0"])
    text(HX[0] + 0.15 + cw_ / 2, cb_y + 0.31, "\u2020 = 95% bootstrap interval includes zero", ha="center", va="center",
         fontsize=6.3, style="italic", color="#444444")

    fig.savefig(f"fig1_overview_{name}.pdf", dpi=300)
    fig.savefig(f"fig1_overview_{name}.png", dpi=240)
    plt.close(fig)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in (PALETTES if arg == "all" else [arg]):
        build(n); print("built", n)
