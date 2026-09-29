"""Paper-protocol stats: per-chip mean over trials, median over chips,
95% bootstrap CI over chips (10,000 resamples), share of chips with diff > 0.
G = finetuned - zero-shot; Delta = scattered(random) - contiguous(block)."""
import numpy as np
import pandas as pd
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
BACKBONES = ["tiny", "100M", "300M", "600M"]
KEYS = ["chip", "mask_ratio", "trial_seed"]
rng = np.random.default_rng(0)

def load(p):
    return pd.read_csv(p, dtype={k: str for k in KEYS})

def summarize(x):
    n = len(x)
    idx = rng.integers(0, n, (10000, n))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return round(float(np.median(x)), 3), round(float(lo), 3), round(float(hi), 3), round(float((x > 0).mean()), 3)

rows = []
for bb in BACKBONES:
    zs = load(STUDY / "outputs" / f"results_{bb}.csv")
    ft = load(STUDY / "outputs_finetuned" / f"results_{bb}.csv")
    m = zs.merge(ft, on=KEYS, suffixes=("_zs", "_ft"))
    assert len(m) == len(zs) == len(ft), (bb, len(m), len(zs), len(ft))
    for c in ["block_psnr_zs", "random_psnr_zs", "block_psnr_ft", "random_psnr_ft"]:
        m[c] = m[c].astype(float)
    chip = m.groupby(["mask_ratio", "chip"])[["block_psnr_zs", "random_psnr_zs",
                                             "block_psnr_ft", "random_psnr_ft"]].mean().reset_index()
    for ratio, g in chip.groupby("mask_ratio"):
        d = {
            "G_contiguous": (g.block_psnr_ft - g.block_psnr_zs).values,
            "G_scattered": (g.random_psnr_ft - g.random_psnr_zs).values,
            "gap_zeroshot": (g.random_psnr_zs - g.block_psnr_zs).values,
            "gap_finetuned": (g.random_psnr_ft - g.block_psnr_ft).values,
        }
        row = {"backbone": bb, "ratio": ratio,
               "zs_contig_med": round(g.block_psnr_zs.median(), 2),
               "ft_contig_med": round(g.block_psnr_ft.median(), 2),
               "zs_scatt_med": round(g.random_psnr_zs.median(), 2),
               "ft_scatt_med": round(g.random_psnr_ft.median(), 2)}
        for name, x in d.items():
            med, lo, hi, pos = summarize(x)
            row.update({f"{name}_med": med, f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_pos": pos})
        rows.append(row)

out = pd.DataFrame(rows).sort_values(["backbone", "ratio"])
out.to_csv(STUDY / "outputs_finetuned" / "paper_stats.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
print(out.to_string(index=False))
