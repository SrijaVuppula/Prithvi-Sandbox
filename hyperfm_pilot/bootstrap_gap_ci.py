"""Scene-level bootstrap 95% CI on the scattered-minus-contiguous PSNR gap.
Delta = median(scattered) - median(contiguous), pooled over trials; scenes resampled."""
import sys, numpy as np, pandas as pd
N_BOOT, SEED = 10000, 0
rng = np.random.default_rng(SEED)

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    df["psnr"] = df["psnr"].replace([np.inf, -np.inf], np.nan)
    print(f"\n=== {path}  (n_rows={len(df)}, n_tiles={df.tile.nunique()}) ===")
    print(f"{'ratio':>6} {'cont_med':>9} {'scat_med':>9} {'delta_med':>10} {'95% CI':>18} "
          f"{'cont_mean':>10} {'scat_mean':>10} {'delta_mean':>11}")
    out = []
    for ratio in sorted(df.ratio.unique()):
        sub = df[df.ratio == ratio]
        tiles = np.sort(sub.tile.unique())
        # (n_tiles, n_trials) matrix per geometry
        mats = {g: sub[sub.mask_type == g].pivot(index="tile", columns="trial", values="psnr")
                   .reindex(tiles).to_numpy() for g in ("contiguous", "scattered")}
        c, s = mats["contiguous"], mats["scattered"]
        d_med = np.nanmedian(s) - np.nanmedian(c)
        d_mean = np.nanmean(s) - np.nanmean(c)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, len(tiles), len(tiles))
            boots[b] = np.nanmedian(s[idx]) - np.nanmedian(c[idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{ratio:>6.2f} {np.nanmedian(c):9.2f} {np.nanmedian(s):9.2f} {d_med:10.2f} "
              f"   [{lo:5.2f}, {hi:5.2f}] {np.nanmean(c):10.2f} {np.nanmean(s):10.2f} {d_mean:11.2f}")
        out.append([ratio, np.nanmedian(c), np.nanmedian(s), d_med, lo, hi,
                    np.nanmean(c), np.nanmean(s), d_mean, int(np.isnan(c).sum() + np.isnan(s).sum())])
    pd.DataFrame(out, columns=["ratio", "cont_median", "scat_median", "delta_median",
                               "ci_lo", "ci_hi", "cont_mean", "scat_mean", "delta_mean",
                               "n_nonfinite"]).to_csv(path.replace(".csv", "_gap_ci.csv"), index=False)
