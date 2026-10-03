"""
qualitative_and_distance.py
---------------------------
Part B: per-patch reconstruction error vs. distance (in patches, chessboard metric) to the
        nearest visible summer patch. All 500 study chips, 4 ratios, trial 0, both geometries,
        zero-shot and fine-tuned (seed 1) for each backbone.
Part A: full reconstructions at r = 0.4 for 3 chips chosen by rule (25th/50th/75th percentile
        of zero-shot 100M contiguous PSNR at r = 0.4, mean over its 5 trials) for the
        qualitative figure.
Masks replay trial 0's logged trial_seed from the zero-shot results, exactly as in the paper.
The pipeline (chip list, normalisation, forward pass, PSNR) is imported from
run_paired_block_random_finetuned_seeded.py, and every PSNR is checked against the logged value.
Outputs (resume-safe, one file per backbone and condition):
  outputs_finetuned/distance/patch_mse_{bb}_{cond}.csv
  outputs_finetuned/distance/qual_{bb}_{cond}.npz
Usage: python scripts/qualitative_and_distance.py [--smoke] [--backbones tiny 100M ...] [--seed 1]
"""
import sys, csv, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_cdt

REPO = Path("~/Prithvi/Prithvi-Sandbox").expanduser()
STUDY = REPO / "multi_tile_generalization" / "block_masking_study"
for p in (REPO, REPO / "patch_masking_study", STUDY / "masking", STUDY / "scripts"):
    sys.path.insert(0, str(p))

import run_paired_block_random_finetuned_seeded as ev          # same pipeline as the paper's eval
from terratorch_loader import load_prithvi_from_terratorch
from temporal_gap_masker import (build_block_noise_mask as fixed_block,
                                 build_random_noise_mask as fixed_random, pixel_map)

IMG, T, FRAME, RATIOS = ev.IMG, ev.T, ev.FRAME, ev.RATIOS
BB_ALL = ["tiny", "100M", "300M", "600M"]
OUT = STUDY / "outputs_finetuned" / "distance"
FIELDS = ["backbone", "cond", "geometry", "chip", "mask_ratio", "dist", "n_patches", "mse_sum"]


def hidden_grid(idx, patch):
    g = IMG // patch
    local = idx.numpy() - FRAME * g * g
    local = local[(local >= 0) & (local < g * g)]
    H = np.zeros(g * g, bool)
    H[local] = True
    return H.reshape(g, g)


def patch_mse(rec, gt, patch):
    g = IMG // patch
    e = ((rec.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(0)
    return e.reshape(g, patch, g, patch).mean((1, 3))


def pick_chips():
    d = pd.read_csv(STUDY / "outputs" / "results_100M.csv")
    d = d[d.mask_ratio.astype(float).round(2) == 0.4]
    m = d.groupby("chip").block_psnr.mean()
    return [str((m - m.quantile(q)).abs().idxmin()) for q in (0.25, 0.5, 0.75)]


def logged_psnr(path):
    d = pd.read_csv(path, dtype={"chip": str, "mask_ratio": str, "trial": str})
    d = d[d.trial == "0"]
    return {(c, r): (float(b), float(s)) for c, r, b, s in zip(d.chip, d.mask_ratio, d.block_psnr, d.random_psnr)}


def load_model(cfg, bb, cond, seed):
    if cond == "zs":
        ck = Path(cfg["backbones"][bb]["checkpoint"]).expanduser()
        base, name = ck.parent, ck.name
    else:
        base, name = STUDY / "checkpoints" / "seeds" / f"seed{seed}" / bb, "best.pt"
    return load_prithvi_from_terratorch(backbone_name=bb, base_dir=base, checkpoint_filename=name,
                                        num_frames=T, device=ev.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 chips, writes smoke_* files")
    ap.add_argument("--backbones", nargs="+", default=BB_ALL)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ev.load_cfg()
    chips = ev.get_chips(cfg)
    assert len(chips) == ev.N_CHIPS, len(chips)
    picks = pick_chips()
    print("qualitative chips (25th/50th/75th pct of zero-shot 100M contiguous PSNR, r=0.4):", picks)
    if a.smoke:
        chips = [c for c in chips if c.name in picks][:3]
    tag = "smoke_" if a.smoke else ""

    for bb in a.backbones:
        patch = cfg["backbones"][bb]["patch_size"]
        seeds = ev.load_seed_lookup(STUDY / "outputs" / f"results_{bb}.csv")
        for cond in ("zs", "ft"):
            csv_path = OUT / f"{tag}patch_mse_{bb}_{cond}.csv"
            npz_path = OUT / f"{tag}qual_{bb}_{cond}.npz"
            if csv_path.exists() and npz_path.exists():
                print(f"SKIP {bb} {cond}: done"); continue
            logged = logged_psnr(STUDY / "outputs" / f"results_{bb}.csv" if cond == "zs" else
                                 STUDY / "outputs_finetuned" / "seeds" / f"seed{a.seed}" / f"results_{bb}.csv")
            model, _, mean, std, sp = load_model(cfg, bb, cond, a.seed)
            assert sp == patch
            qual, maxdiff, t0 = {}, 0.0, time.time()
            tmp = csv_path.with_suffix(".part")
            with open(tmp, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(FIELDS)
                for i, chip in enumerate(chips):
                    x, gt = ev.load_norm_and_gt(chip, mean, std)
                    for r in RATIOS:
                        key = (chip.name, f"{r}")
                        seed = seeds[(chip.name, f"{r}", "0")]
                        nb, grb, idxb = fixed_block(r, patch, IMG, T, FRAME, trial_seed=seed)
                        nr, grr, idxr = fixed_random(r, patch, IMG, T, FRAME, trial_seed=seed, n_summer_masked=len(idxb))
                        for gi, (geo, nz, gr, idx) in enumerate((("contiguous", nb, grb, idxb), ("scattered", nr, grr, idxr))):
                            rec = ev.recon_unit(model, x, nz, gr, mean, std)
                            ps = ev.masked_psnr(rec, gt, pixel_map(idx, patch, IMG, T, FRAME))
                            maxdiff = max(maxdiff, abs(ps - logged[key][gi]))
                            H = hidden_grid(idx, patch)
                            assert H.sum() == len(idx)
                            pm = patch_mse(rec, gt, patch)
                            dist = distance_transform_cdt(H, metric="chessboard")
                            for dv in np.unique(dist[H]):
                                sel = H & (dist == dv)
                                w.writerow([bb, cond, geo, chip.name, r, int(dv), int(sel.sum()), repr(float(pm[sel].sum()))])
                            if chip.name in picks and abs(r - 0.4) < 1e-9:
                                k = f"{chip.name}|{geo}"
                                qual[f"{k}|rec"] = rec.astype(np.float16)
                                qual[f"{k}|mask"] = pixel_map(idx, patch, IMG, T, FRAME).numpy()
                                qual[f"{k}|psnr"] = np.float64(ps)
                                qual[f"{chip.name}|gt"] = gt.astype(np.float16)
                    if (i + 1) % 50 == 0:
                        print(f"  [{bb} {cond}] {i + 1}/{len(chips)} chips  {time.time() - t0:.0f}s  max|PSNR-logged|={maxdiff:.4f}")
            np.savez_compressed(npz_path, **qual)
            tmp.rename(csv_path)
            print(f"DONE {bb} {cond}: {len(chips)} chips in {time.time() - t0:.0f}s; max |PSNR - logged| = {maxdiff:.4f} dB")
            if maxdiff > 0.01:
                print("  WARNING: reconstructions do not match the logged evaluation; stop and check")
            del model
            ev.torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
