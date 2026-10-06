#!/usr/bin/env python3
"""
Zero-shot vs fine-tuned side-by-side grids, shared y-axis within each row.

Figure 1  fig_zeroshot_vs_finetuned_psnr_sharedy.png
          rows: block PSNR, random PSNR      cols: tiny / 100M / 300M / 600M
Figure 2  fig_zeroshot_vs_finetuned_cost.png
          rows: inference energy (J), power (W), time (ms)   (block masking)
          Per-row y is shared by default; --free-cost-y gives each panel its own y.

Dashed + hollow marker = zero-shot, solid + filled marker = fine-tuned.
Labels above points = fine-tuned minus zero-shot (dB for PSNR, % for cost).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BACKBONES = ["tiny", "100M", "300M", "600M"]
RATIOS = [20, 40, 60, 80]
COLORS = {"tiny": "#56B4E9", "100M": "#009E73", "300M": "#E69F00", "600M": "#0072B2"}
MARKERS = {"tiny": "o", "100M": "s", "300M": "^", "600M": "D"}
MINUS = "\u2212"


# ---------- column helpers (robust to naming differences) ----------
def find_col(df, *must, exclude=()):
    for c in df.columns:
        lc = c.lower()
        if all(m in lc for m in must) and not any(e in lc for e in exclude):
            return c
    raise KeyError(f"No column containing {must} in {list(df.columns)}")


def ratio_series(df):
    col = "mask_ratio" if "mask_ratio" in df.columns else find_col(df, "ratio", exclude=("global",))
    s = df[col].astype(float)
    if s.max() <= 1.0:
        s = s * 100
    return s.round().astype(int)


def metric_col(df, key):
    try:
        return find_col(df, key, "mean")
    except KeyError:
        return find_col(df, key, exclude=("std", "min", "max"))


# ---------- loaders ----------
def load_psnr(path, geom):
    df = pd.read_csv(path)
    df["_r"] = ratio_series(df)
    col = find_col(df, geom, "psnr")
    return df.groupby("_r")[col].mean().reindex(RATIOS)


def load_cost(path, backbone, key, geom="block"):
    df = pd.read_csv(path)
    df = df[df[find_col(df, "backbone")].astype(str) == backbone]
    if any("mask_type" == c.lower() for c in df.columns):
        mt = [c for c in df.columns if c.lower() == "mask_type"][0]
        df = df[df[mt].astype(str).str.lower() == geom]
    df = df.assign(_r=ratio_series(df))
    s = df.groupby("_r")[metric_col(df, key)].mean().reindex(RATIOS)
    return s


# ---------- plotting ----------
def fmt_delta(d, kind):
    if np.isnan(d):
        return ""
    sign = "+" if d >= 0 else MINUS
    return f"{sign}{abs(d):.2f}" if kind == "abs" else f"{sign}{abs(d):.1f}%"


def draw_grid(rows, out_png, title, share_y=True):
    n = len(rows)
    fig, axes = plt.subplots(
        n, 4, figsize=(11.5, 2.7 * n + 1.0), sharex=True,
        sharey="row" if share_y else False, squeeze=False,
    )
    for i, row in enumerate(rows):
        for j, bb in enumerate(BACKBONES):
            ax, c, m = axes[i][j], COLORS[bb], MARKERS[bb]
            zs, ft = row["zs"][bb], row["ft"][bb]
            ax.plot(RATIOS, zs.values, ls="--", color=c, marker=m, mfc="white", mec=c, ms=5.5, lw=1.4)
            ax.plot(RATIOS, ft.values, ls="-", color=c, marker=m, mfc=c, mec=c, ms=5.5, lw=1.9)
            for r, z, f in zip(RATIOS, zs.values, ft.values):
                d = (f - z) if row["delta"] == "abs" else (100 * (f - z) / z)
                ax.annotate(fmt_delta(d, row["delta"]), (r, f), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7, color="0.25")
            ax.grid(alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(RATIOS)
            if i == 0:
                ax.set_title(bb, fontsize=11)
            if j == 0:
                ax.set_ylabel(row["ylabel"])
            if i == n - 1:
                ax.set_xlabel("Mask Ratio (%)")
    # headroom for the delta labels, applied after all panels are drawn
    for i in range(n):
        for j in range(4):
            ax = axes[i][j]
            ax.relim()
            ax.autoscale_view()
    for i in range(n):
        for j in range(4):
            ax = axes[i][j]
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.08 * (hi - lo))
            if share_y:
                break  # shared row: setting one panel sets them all
    handles = [
        Line2D([], [], color="0.3", ls="--", marker="o", mfc="white", label="zero-shot"),
        Line2D([], [], color="0.3", ls="-", marker="o", mfc="0.3", label="fine-tuned"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print("saved", out_png)


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs-dir", default=base / "outputs")
    ap.add_argument("--ft-dir", default=base / "outputs_finetuned")
    ap.add_argument("--zs-energy", default=base / "outputs" / "inference_energy.csv")
    ap.add_argument("--ft-energy", default=base / "outputs_finetuned" / "inference_energy_finetuned.csv")
    ap.add_argument("--out", default=base / "outputs_finetuned" / "figures")
    ap.add_argument("--free-cost-y", action="store_true", help="independent y-axis per panel in the cost figure")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: PSNR, shared y per row ----
    psnr_rows = []
    for geom, label in (("block", "Block PSNR (dB)"), ("random", "Random PSNR (dB)")):
        zs = {bb: load_psnr(Path(a.zs_dir) / f"results_{bb}.csv", geom) for bb in BACKBONES}
        ft = {bb: load_psnr(Path(a.ft_dir) / f"results_{bb}.csv", geom) for bb in BACKBONES}
        psnr_rows.append(dict(zs=zs, ft=ft, ylabel=label, delta="abs"))
    draw_grid(psnr_rows, out / "fig_zeroshot_vs_finetuned_psnr_sharedy.png",
              "Fine-tuning effect on reconstruction PSNR (same y-axis across backbones)")

    # ---- Figure 2: energy / power / time (block), shared y per row ----
    cost_rows = []
    for key, label in (("energy", "Energy per pass (J)"),
                       ("power", "Board power (W)"),
                       ("time", "Time per pass (ms)")):
        zs = {bb: load_cost(a.zs_energy, bb, key) for bb in BACKBONES}
        ft = {bb: load_cost(a.ft_energy, bb, key) for bb in BACKBONES}
        # energy may be stored in mJ; convert to J if values look like mJ
        if key == "energy" and np.nanmedian(np.concatenate([s.values for s in zs.values()])) > 100:
            zs = {k: v / 1000 for k, v in zs.items()}
            ft = {k: v / 1000 for k, v in ft.items()}
        cost_rows.append(dict(zs=zs, ft=ft, ylabel=label, delta="pct"))
    draw_grid(cost_rows, out / "fig_zeroshot_vs_finetuned_cost.png",
              "Inference cost, zero-shot vs fine-tuned (block masking, batch size 1)",
              share_y=not a.free_cost_y)


if __name__ == "__main__":
    main()
