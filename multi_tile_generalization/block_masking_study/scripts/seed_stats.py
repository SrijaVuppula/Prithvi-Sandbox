"""Seed-aggregated paper stats (10 fine-tuning seeds per backbone).
Chip-level value = mean over the 5 trials and the 10 seeds; median over chips;
95% bootstrap CI over chips (10,000 resamples); share of chips > 0.
Also the run-to-run spread of the per-seed medians. Writes outputs_finetuned/seed_stats.csv."""
import numpy as np
import pandas as pd
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
BACKBONES = ["tiny", "100M", "300M", "600M"]
SEEDS = range(1, 11)
KEYS = ["chip", "mask_ratio", "trial_seed"]
COLS = ["block_psnr", "random_psnr"]
rng = np.random.default_rng(0)

def load(p):
    d = pd.read_csv(p, dtype={k: str for k in KEYS})
    for c in COLS:
        d[c] = d[c].astype(float)
    return d

def chip_means(d):
    return d.groupby(["mask_ratio", "chip"])[COLS].mean()

def boot(x):
    n = len(x)
    meds = np.median(x[rng.integers(0, n, (10000, n))], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi), float((x > 0).mean())

def quantities(z, f):
    gc = (f.block_psnr - z.block_psnr).values
    gs = (f.random_psnr - z.random_psnr).values
    return {"G_contiguous": gc, "G_scattered": gs,
            "gap_zeroshot": (z.random_psnr - z.block_psnr).values,
            "gap_finetuned": (f.random_psnr - f.block_psnr).values,
            "dgap": gs - gc}

rows = []
for bb in BACKBONES:
    zs = load(STUDY / "outputs" / f"results_{bb}.csv")
    zc = chip_means(zs)
    per_seed = []
    for s in SEEDS:
        ft = load(STUDY / "outputs_finetuned" / "seeds" / f"seed{s}" / f"results_{bb}.csv")
        assert len(ft) == len(zs), (bb, s, len(ft), len(zs))
        assert len(zs.merge(ft, on=KEYS)) == len(zs), (bb, s, "trial keys differ")
        fc = chip_means(ft)
        assert fc.index.equals(zc.index), (bb, s)
        per_seed.append(fc)
    fmean = sum(per_seed) / len(per_seed)
    for ratio in sorted(zc.index.get_level_values(0).unique()):
        z = zc.xs(ratio, level=0)
        f = fmean.xs(ratio, level=0)
        seed_q = [quantities(z, p.xs(ratio, level=0)) for p in per_seed]
        row = {"backbone": bb, "ratio": float(ratio),
               "zs_contig_med": round(float(z.block_psnr.median()), 2),
               "ft_contig_med": round(float(f.block_psnr.median()), 2),
               "zs_scatt_med": round(float(z.random_psnr.median()), 2),
               "ft_scatt_med": round(float(f.random_psnr.median()), 2)}
        for name, x in quantities(z, f).items():
            med, lo, hi, pos = boot(x)
            sm = np.array([np.median(q[name]) for q in seed_q])
            row.update({f"{name}_med": round(med, 6), f"{name}_lo": round(lo, 6),
                        f"{name}_hi": round(hi, 6), f"{name}_pos": round(pos, 6),
                        f"{name}_seedmean": round(float(sm.mean()), 6),
                        f"{name}_seedsd": round(float(sm.std(ddof=1)), 6),
                        f"{name}_seedmin": round(float(sm.min()), 6),
                        f"{name}_seedmax": round(float(sm.max()), 6)})
        rows.append(row)

out = pd.DataFrame(rows).sort_values(["backbone", "ratio"])
out.to_csv(STUDY / "outputs_finetuned" / "seed_stats.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
show = ["backbone", "ratio", "G_contiguous_med", "G_contiguous_lo", "G_contiguous_hi", "G_contiguous_seedsd",
        "G_scattered_med", "G_scattered_lo", "G_scattered_hi", "dgap_med", "dgap_lo", "dgap_hi"]
print(out[show].to_string(index=False))
