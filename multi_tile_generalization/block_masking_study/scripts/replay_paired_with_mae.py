"""
replay_paired_with_mae.py
--------------------------
Replays run_paired_block_random.py row-for-row from the logged trial_seed
in results_{backbone}.csv (no re-hashing -- the seed is already recorded).
Rebuilds both maskers exactly, reruns the forward pass, checks replayed
PSNR against the logged PSNR (tolerance 0.01 dB), and writes true MAE
(mean abs error over the masked summer pixels) to a NEW file. Never
touches the original results_{backbone}.csv -- no published PSNR changes.

Usage:
  python replay_paired_with_mae.py --backbones tiny --limit 200   # quick check
  python replay_paired_with_mae.py                                # full replay, all backbones
  python replay_paired_with_mae.py --resume                       # continue after crash
"""

import sys, csv, argparse
from pathlib import Path
import numpy as np
import torch
import rasterio
import yaml

REPO  = Path("~/Prithvi/Prithvi-Sandbox").expanduser()
STUDY = REPO / "multi_tile_generalization" / "block_masking_study"
MASK  = STUDY / "masking"
CHIPS_DIR = REPO / "multi_tile_generalization" / "study_chips_500"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "patch_masking_study"))
sys.path.insert(0, str(MASK))

from terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from temporal_gap_masker import (
    build_block_noise_mask as fixed_block,
    build_random_noise_mask as fixed_random,
    pixel_map,
)

CFG_PATH = STUDY / "config" / "block_masking_config.yaml"
IMG, BANDS, T, FRAME = 224, 6, 3, 1
PSNR_TOL = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def load_norm_and_gt(path, mean, std):
    with rasterio.open(path) as src:
        data = src.read()
    raw = torch.tensor(data.reshape(T, BANDS, IMG, IMG), dtype=torch.float32)
    gt = np.clip(raw[FRAME].numpy() / 10000.0, 0.0, 1.0)
    m = torch.tensor(mean[:BANDS]).reshape(1, -1, 1, 1)
    s = torch.tensor(std[:BANDS]).reshape(1, -1, 1, 1)
    x = ((raw - m) / s).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    return x, gt


@torch.no_grad()
def recon_unit(model, x, noise, ratio_global, mean, std):
    _, _, rec_img, _, _ = run_masked_forward(
        model=model, x=x, temporal_coords=None, location_coords=None,
        mask_ratio=ratio_global, noise=noise.unsqueeze(0).to(device),
    )
    pred = rec_img[0, :, FRAME, :, :]
    m = torch.tensor(mean[:BANDS]).reshape(-1, 1, 1)
    s = torch.tensor(std[:BANDS]).reshape(-1, 1, 1)
    return torch.clamp((pred * s + m) / 10000.0, 0.0, 1.0).numpy()


def masked_psnr(recon, gt, pmask):
    m = pmask.numpy() if isinstance(pmask, torch.Tensor) else pmask
    if m.sum() == 0:
        return float("nan")
    d = recon[:, m] - gt[:, m]
    mse = float(np.mean(d ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def masked_mae(recon, gt, pmask):
    m = pmask.numpy() if isinstance(pmask, torch.Tensor) else pmask
    if m.sum() == 0:
        return float("nan")
    d = recon[:, m] - gt[:, m]
    return float(np.mean(np.abs(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", type=str, default=None,
                     help="comma-separated subset, e.g. tiny,100M")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap rows replayed per backbone (quick test)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    out_dir = Path(cfg["output"]["dir"]).expanduser()
    backbones = list(cfg["backbones"].items())
    if args.backbones:
        wanted = set(args.backbones.split(","))
        backbones = [(bb, c) for bb, c in backbones if bb in wanted]

    for bb, bcfg in backbones:
        src_csv = out_dir / f"results_{bb}.csv"
        if not src_csv.exists():
            print(f"SKIP {bb}: {src_csv} not found")
            continue
        rows = list(csv.DictReader(open(src_csv)))
        if args.limit:
            rows = rows[:args.limit]
        print(f"\n{'='*60}\n{bb}  replaying {len(rows)} rows from {src_csv.name}\n{'='*60}")

        out_csv = out_dir / f"results_{bb}_mae_replay.csv"
        done_keys = set()
        write_mode = "w"
        if args.resume and out_csv.exists():
            with open(out_csv) as fh:
                for r in csv.DictReader(fh):
                    done_keys.add((r["chip"], r["mask_ratio"], r["trial"]))
            write_mode = "a"
            print(f"  resuming: {len(done_keys)} rows already done")

        ckpt = Path(bcfg["checkpoint"]).expanduser()
        patch = bcfg["patch_size"]
        model, _, mean, std, sp = load_prithvi_from_terratorch(
            backbone_name=bb, base_dir=ckpt.parent,
            checkpoint_filename=ckpt.name, num_frames=T, device=device)
        assert sp == patch

        fields = ["backbone", "chip", "mask_ratio", "trial", "trial_seed",
                  "summer_masked",
                  "block_psnr_logged", "block_psnr_replay", "block_psnr_diff",
                  "random_psnr_logged", "random_psnr_replay", "random_psnr_diff",
                  "block_mae", "random_mae"]

        fh = open(out_csv, write_mode, newline="")
        w = csv.DictWriter(fh, fieldnames=fields)
        if write_mode == "w":
            w.writeheader()

        n_checked = 0
        n_mismatch = 0
        last_chip_name, last_xgt = None, None

        for row in rows:
            key = (row["chip"], row["mask_ratio"], row["trial"])
            if key in done_keys:
                continue

            chip_name = row["chip"]
            r = float(row["mask_ratio"])
            t = int(row["trial"])
            seed = int(row["trial_seed"])
            n_summer_masked = int(row["summer_masked"])
            logged_block = float(row["block_psnr"])
            logged_random = float(row["random_psnr"])

            if chip_name != last_chip_name:
                chip_path = CHIPS_DIR / chip_name
                if not chip_path.exists():
                    chip_path = Path(cfg["data"]["chips_dir"]).expanduser() / chip_name
                last_xgt = load_norm_and_gt(chip_path, mean, std)
                last_chip_name = chip_name
            x, gt = last_xgt

            nb, grb, idxb = fixed_block(r, patch, IMG, T, FRAME, trial_seed=seed)
            if len(idxb) != n_summer_masked:
                print(f"  WARNING {chip_name} r={r} t={t}: block count mismatch "
                      f"(replay={len(idxb)} vs logged={n_summer_masked})")
            rb = recon_unit(model, x, nb, grb, mean, std)
            pmb = pixel_map(idxb, patch, IMG, T, FRAME)
            bp_replay = masked_psnr(rb, gt, pmb)
            bp_mae = masked_mae(rb, gt, pmb)

            nr, grr, idxr = fixed_random(r, patch, IMG, T, FRAME,
                                          trial_seed=seed, n_summer_masked=n_summer_masked)
            rr = recon_unit(model, x, nr, grr, mean, std)
            pmr = pixel_map(idxr, patch, IMG, T, FRAME)
            rp_replay = masked_psnr(rr, gt, pmr)
            rp_mae = masked_mae(rr, gt, pmr)

            b_diff = bp_replay - logged_block
            r_diff = rp_replay - logged_random
            n_checked += 1
            if abs(b_diff) > PSNR_TOL or abs(r_diff) > PSNR_TOL:
                n_mismatch += 1
                print(f"  MISMATCH {chip_name} r={r} t={t}: "
                      f"block {logged_block}->{bp_replay:.4f} (D{b_diff:+.4f}), "
                      f"random {logged_random}->{rp_replay:.4f} (D{r_diff:+.4f})")

            w.writerow({
                "backbone": bb, "chip": chip_name, "mask_ratio": r, "trial": t,
                "trial_seed": seed, "summer_masked": n_summer_masked,
                "block_psnr_logged": logged_block, "block_psnr_replay": round(bp_replay, 4),
                "block_psnr_diff": round(b_diff, 4),
                "random_psnr_logged": logged_random, "random_psnr_replay": round(rp_replay, 4),
                "random_psnr_diff": round(r_diff, 4),
                "block_mae": round(bp_mae, 6), "random_mae": round(rp_mae, 6),
            })
            fh.flush()

            if n_checked % 50 == 0:
                print(f"  [{bb}] {n_checked}/{len(rows)} checked, {n_mismatch} mismatches so far")

        fh.close()
        print(f"\n{bb}: {n_checked} rows replayed, {n_mismatch} PSNR mismatches "
              f"(tolerance +/-{PSNR_TOL}dB)")
        if n_checked > 0 and n_mismatch == 0:
            print(f"  PASS: replay reproduces logged PSNR within tolerance.")
        elif n_checked == 0:
            print(f"  Nothing new to check (already done or empty).")
        else:
            print(f"  FAIL: replay does not reproduce logged PSNR. Do not trust MAE output.")


if __name__ == "__main__":
    main()
