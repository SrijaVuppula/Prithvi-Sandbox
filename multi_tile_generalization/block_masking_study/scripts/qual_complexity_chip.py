"""
qual_complexity_chip.py
-----------------------
Zero-shot vs fine-tuned reconstructions for ONE evaluation chip chosen by a fixed rule:
the chip whose complexity score (outputs/chip_complexity_v2.csv: mean of normalised edge
density, spatial std and entropy of the summer RGB) is closest to the median over the 500
study chips. The rule was fixed before any reconstruction of this chip was inspected.
For each backbone: zero-shot and all ten fine-tuned runs, r = 0.4, the chip's five logged
trials, contiguous and scattered masks (replayed from the logged trial_seed, as in the
paper). Every PSNR is checked against the logged evaluation value.
Pipeline imported from qualitative_and_distance.py (and through it the paper's eval module).
Outputs in outputs_finetuned/distance/ (prefix cchip_smoke_ with --smoke):
  cchip_psnr.csv   backbone, cond, seed, trial, geometry, psnr, logged
  cchip_{bb}.npz   gt; trial-0 masks; trial-0 reconstructions (zero-shot, fine-tuned seed 1);
                   patch-MSE grids: pm_zs [trial, geometry, g, g], pm_ft [seed, trial, geometry, g, g]
Usage: python -u scripts/qual_complexity_chip.py [--backbones tiny 100M 300M 600M] [--smoke]
"""
import csv, argparse, time
import numpy as np
import pandas as pd
import qualitative_and_distance as qd

ev = qd.ev
GEOS = ("contiguous", "scattered")


def pick_chip():
    c = pd.read_csv(qd.STUDY / "outputs" / "chip_complexity_v2.csv")
    assert len(c) == 500, len(c)
    i = (c.complexity - c.complexity.median()).abs().idxmin()
    return str(c.chip[i]), float(c.complexity[i]), int(c["rank"][i])


def logged(path, chip, rk):
    d = pd.read_csv(path, dtype={"chip": str, "mask_ratio": str, "trial": str})
    d = d[(d.chip == chip) & (d.mask_ratio == rk)]
    return {t: (float(b), float(s)) for t, b, s in zip(d.trial, d.block_psnr, d.random_psnr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=qd.BB_ALL)
    ap.add_argument("--smoke", action="store_true", help="tiny only, zero-shot + seed 1")
    a = ap.parse_args()
    bbs, seeds_ft = (["tiny"], [1]) if a.smoke else (a.backbones, list(range(1, 11)))
    tag = "cchip_smoke_" if a.smoke else "cchip_"
    qd.OUT.mkdir(parents=True, exist_ok=True)

    cfg = ev.load_cfg()
    chips = ev.get_chips(cfg)
    name, cx, rank = pick_chip()
    chip = [c for c in chips if c.name == name]
    assert len(chip) == 1, name
    chip = chip[0]
    r = [x for x in qd.RATIOS if abs(x - 0.4) < 1e-9][0]
    rk = f"{r}"
    print(f"chip (complexity closest to the median): {name}  complexity={cx:.4f}  rank={rank}/500")

    rows, maxdiff = [], 0.0
    for bb in bbs:
        t0 = time.time()
        patch = cfg["backbones"][bb]["patch_size"]
        g = qd.IMG // patch
        lookup = ev.load_seed_lookup(qd.STUDY / "outputs" / f"results_{bb}.csv")
        trials = sorted(t for (c, rr, t) in lookup if c == name and rr == rk)
        assert len(trials) == 5, trials
        masks = {}
        for t in trials:
            s = lookup[(name, rk, t)]
            nb, grb, idxb = qd.fixed_block(r, patch, qd.IMG, qd.T, qd.FRAME, trial_seed=s)
            nr, grr, idxr = qd.fixed_random(r, patch, qd.IMG, qd.T, qd.FRAME, trial_seed=s,
                                            n_summer_masked=len(idxb))
            masks[t] = (("contiguous", nb, grb, idxb), ("scattered", nr, grr, idxr))
        pm_zs = np.zeros((len(trials), 2, g, g))
        pm_ft = np.zeros((len(seeds_ft), len(trials), 2, g, g))
        out = {}
        for cond, sd in [("zs", 0)] + [("ft", s) for s in seeds_ft]:
            lg = logged(qd.STUDY / "outputs" / f"results_{bb}.csv" if cond == "zs" else
                        qd.STUDY / "outputs_finetuned" / "seeds" / f"seed{sd}" / f"results_{bb}.csv", name, rk)
            model, _, mean, std, sp = qd.load_model(cfg, bb, cond, sd if cond == "ft" else 1)
            assert sp == patch
            x, gt = ev.load_norm_and_gt(chip, mean, std)
            out["gt"] = gt.astype(np.float16)
            for ti, t in enumerate(trials):
                for gi, (geo, nz, gr, idx) in enumerate(masks[t]):
                    rec = ev.recon_unit(model, x, nz, gr, mean, std)
                    pmap = qd.pixel_map(idx, patch, qd.IMG, qd.T, qd.FRAME)
                    ps = ev.masked_psnr(rec, gt, pmap)
                    maxdiff = max(maxdiff, abs(ps - lg[t][gi]))
                    rows.append([bb, cond, sd, t, geo, repr(float(ps)), lg[t][gi]])
                    pm = qd.patch_mse(rec, gt, patch)
                    if cond == "zs":
                        pm_zs[ti, gi] = pm
                    else:
                        pm_ft[seeds_ft.index(sd), ti, gi] = pm
                    if t == trials[0]:
                        out[f"{geo}|mask"] = pmap.numpy()
                        if cond == "zs":
                            out[f"{geo}|rec_zs"] = rec.astype(np.float16)
                        elif sd == 1:
                            out[f"{geo}|rec_ft1"] = rec.astype(np.float16)
            del model
            ev.torch.cuda.empty_cache()
        out["pm_zs"], out["pm_ft"] = pm_zs, pm_ft
        out["seeds_ft"], out["trials"] = np.array(seeds_ft), np.array([int(t) for t in trials])
        out["chip"], out["complexity"], out["rank"], out["patch"] = np.array(name), cx, rank, patch
        np.savez_compressed(qd.OUT / f"{tag}{bb}.npz", **out)
        print(f"DONE {bb}: {time.time() - t0:.0f}s; max |PSNR - logged| so far = {maxdiff:.4f} dB")

    with open(qd.OUT / f"{tag}psnr.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone", "cond", "seed", "trial", "geometry", "psnr", "logged"])
        w.writerows(rows)

    d = pd.DataFrame(rows, columns=["backbone", "cond", "seed", "trial", "geometry", "psnr", "logged"])
    d["psnr"] = d.psnr.astype(float)
    print(f"\nchip {name} (complexity rank {rank}/500), r = 0.4; mean PSNR over trials (dB)")
    for bb in bbs:
        for geo in GEOS:
            q = d[(d.backbone == bb) & (d.geometry == geo)]
            zs = q[q.cond == "zs"].psnr.mean()
            per_seed = q[q.cond == "ft"].groupby("seed").psnr.mean()
            print(f"  {bb:5s} {geo:10s} zero-shot {zs:6.2f} | fine-tuned mean {per_seed.mean():6.2f} "
                  f"(runs {per_seed.min():.2f}-{per_seed.max():.2f}) | G {per_seed.mean() - zs:+.2f}")
    print(f"max |PSNR - logged| = {maxdiff:.4f} dB")
    if maxdiff > 0.01:
        print("WARNING: reconstructions do not match the logged evaluation; stop and check")


if __name__ == "__main__":
    main()
