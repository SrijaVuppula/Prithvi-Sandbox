"""
fig_breakeven.py -- break-even curve (single column, 3.35 in), paper style matching Figure 1.
Cumulative inference energy vs number of inferences at r = 40%:
  fine-tuned tiny  = E_train(tiny) + N * E_pass(tiny)     (E_ft set equal to zero-shot tiny, Sec. 3.5)
  zero-shot 100M   = N * E_pass(100M)
Shaded band: break-even N* over r = 20-80% (Table 4). Dashed line: one pass over the contiguous US.
Usage: python fig_breakeven.py
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
})
INK = "#222222"
PLASMA = [mpl.colors.to_hex(plt.get_cmap("plasma")(v)) for v in (0.05, 0.34, 0.60, 0.80)]   # tiny, 100M, 300M, 600M (as Fig. 1)

# ------------------------------------------------------------------ DATA
E_TRAIN_J = 333761.0          # tiny, mean of ten runs (outputs/finetune_logs/seeds/*/tiny_finetune_cost.csv)
E_PASS_J = {"tiny": 1.7436, "100M": 3.2142}   # zero-shot, contiguous, r = 40% (outputs/inference_energy.csv)
NSTAR_RANGE = (188e3, 243e3)  # Table 4, tiny, r = 20-80%
CHIP_KM2 = (224 * 30 / 1000) ** 2      # 45.16 km^2 per non-overlapping chip
CONUS_KM2 = 8.08e6

nstar = E_TRAIN_J / (E_PASS_J["100M"] - E_PASS_J["tiny"])
conus_chips = CONUS_KM2 / CHIP_KM2

W, H = 3.35, 2.05
fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(left=0.135, right=0.97, bottom=0.19, top=0.80)
N = np.linspace(0, 320e3, 400)
ft = (E_TRAIN_J + N * E_PASS_J["tiny"]) / 1e6
zs = N * E_PASS_J["100M"] / 1e6

ax.axvspan(*[v / 1e3 for v in NSTAR_RANGE], color=PLASMA[0], alpha=0.10, lw=0, zorder=0)
ax.plot(N / 1e3, zs, color=PLASMA[1], lw=1.4, ls=(0, (4, 2)), zorder=2, label="zero-shot 100M")
ax.plot(N / 1e3, ft, color=PLASMA[0], lw=1.6, zorder=3, label="fine-tuned tiny (incl. training)")
ax.scatter([nstar / 1e3], [E_TRAIN_J / 1e6 + nstar * E_PASS_J["tiny"] / 1e6], s=16, color=PLASMA[0], zorder=4)
ax.annotate(f"$N^*$ = {nstar / 1e3:.0f}k", xy=(nstar / 1e3, (E_TRAIN_J + nstar * E_PASS_J["tiny"]) / 1e6),
            xytext=(nstar / 1e3 + 14, 0.47), fontsize=7, color=PLASMA[0], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=PLASMA[0], lw=0.6))
ax.axvline(conus_chips / 1e3, color="#8A8A8A", lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
ax.text(conus_chips / 1e3 - 4, 0.05, "one pass over\nthe contiguous US", ha="right", va="bottom", fontsize=6.3,
        style="italic", color="#555555", linespacing=1.1)
ax.text(sum(NSTAR_RANGE) / 2e3, 0.97, "$N^*$, $r$ = 20–80%", ha="center", va="top", fontsize=6.3, color=PLASMA[0])
ax.text(8, E_TRAIN_J / 1e6 - 0.035, "training energy", ha="left", va="top", fontsize=6.3, style="italic", color="#555555")

ax.set_xlim(0, 320); ax.set_ylim(0, 1.0)
ax.set_xlabel("Inferences (thousands)", labelpad=2)
ax.set_ylabel("Cumulative energy (MJ)", labelpad=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle=(0, (1, 2))); ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=2.2, borderaxespad=0.2)

top = ax.secondary_xaxis("top", functions=(lambda k: k * 1e3 * CHIP_KM2 / 1e6, lambda m: m * 1e6 / CHIP_KM2 / 1e3))
top.set_xlabel("Area at one pass per chip (million km$^2$)", labelpad=3)
top.set_xticks([0, 4, 8, 12])
top.tick_params(labelsize=7)

fig.savefig("fig_breakeven.pdf")
fig.savefig("fig_breakeven.png", dpi=300)
print(f"N* at r=40%: {nstar:,.0f}; CONUS chips: {conus_chips:,.0f}")
