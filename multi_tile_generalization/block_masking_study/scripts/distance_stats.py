"""Per-patch reconstruction error vs distance to the nearest visible summer patch.

Reads outputs_finetuned/distance/patch_mse_{bb}_{zs,ft}.csv (trial 0, fine-tuned = seed 1)
and writes outputs_finetuned/distance/distance_stats.csv.
PSNR of a bin = 10 log10(1 / (sum mse_sum / sum n_patches)), per chip; medians over chips,
95% bootstrap CI (10,000 resamples over chips, rng seed 0).
Usage: python scripts/distance_stats.py [distance_dir]
"""
import sys, os
import numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), '..', 'outputs_finetuned', 'distance')
BBS = ['tiny', '100M', '300M', '600M']
NBOOT, MIN_CHIPS = 10_000, 50
rng = np.random.default_rng(0)

def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), (NBOOT, len(x)))
    m = np.median(x[idx], axis=1)
    return np.median(x), np.percentile(m, 2.5), np.percentile(m, 97.5), (x > 0).mean(), len(x)

def psnr(mse_sum, n):
    return 10 * np.log10(n / mse_sum)

def load(bb):
    out = []
    for c in ['zs', 'ft']:
        d = pd.read_csv(os.path.join(D, f'patch_mse_{bb}_{c}.csv'), dtype={'chip': str})
        out.append(d)
    return pd.concat(out)

rows = []
def add(**kw):
    med, lo, hi, pos, n = boot(kw.pop('x'))
    rows.append(dict(**kw, med=med, lo=lo, hi=hi, pos=pos, n_chips=n))

for bb in BBS:
    d = load(bb)
    key = ['cond', 'geometry', 'mask_ratio', 'chip']
    # per-chip PSNR at each distance, and over all hidden patches
    pd_ = d.groupby(key + ['dist'])[['mse_sum', 'n_patches']].sum()
    pd_['psnr'] = psnr(pd_.mse_sum, pd_.n_patches)
    pa = d.groupby(key)[['mse_sum', 'n_patches']].sum()
    pa['psnr'] = psnr(pa.mse_sum, pa.n_patches)
    P = pd_.psnr.unstack('dist')            # rows: cond, geometry, ratio, chip
    A = pa.psnr
    for cond in ['zs', 'ft']:
        for r in sorted(d.mask_ratio.unique()):
            Pc, Ps = P.loc[(cond, 'contiguous', r)], P.loc[(cond, 'scattered', r)]
            Ac, As = A.loc[(cond, 'contiguous', r)], A.loc[(cond, 'scattered', r)]
            add(backbone=bb, cond=cond, ratio=r, stat='psnr_all', geometry='contiguous', dist=0, x=Ac)
            add(backbone=bb, cond=cond, ratio=r, stat='psnr_all', geometry='scattered', dist=0, x=As)
            add(backbone=bb, cond=cond, ratio=r, stat='gap_all', geometry='-', dist=0, x=(As - Ac))
            for k in P.columns:
                for g, M in [('contiguous', Pc), ('scattered', Ps)]:
                    x = M[k].dropna()
                    if len(x) >= MIN_CHIPS:
                        add(backbone=bb, cond=cond, ratio=r, stat='psnr_d', geometry=g, dist=k, x=x)
                        # within-chip drop relative to that chip's d=1 bin
                        if k > 1:
                            add(backbone=bb, cond=cond, ratio=r, stat='drop_vs_d1', geometry=g,
                                dist=k, x=(M[k] - M[1]).dropna())
            # matched-distance gap at d=1: scattered d=1 minus contiguous d=1
            add(backbone=bb, cond=cond, ratio=r, stat='gap_d1', geometry='-', dist=1, x=(Ps[1] - Pc[1]))
    # fine-tuning gain per distance (paired per chip)
    for g in ['contiguous', 'scattered']:
        for r in sorted(d.mask_ratio.unique()):
            Z, F = P.loc[('zs', g, r)], P.loc[('ft', g, r)]
            add(backbone=bb, cond='ft-zs', ratio=r, stat='G_all', geometry=g, dist=0,
                x=A.loc[('ft', g, r)] - A.loc[('zs', g, r)])
            for k in P.columns:
                x = (F[k] - Z[k]).dropna()
                if len(x) >= MIN_CHIPS:
                    add(backbone=bb, cond='ft-zs', ratio=r, stat='G_d', geometry=g, dist=k, x=x)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(D, 'distance_stats.csv'), index=False, float_format='%.6f')
print('wrote', len(out), 'rows')
