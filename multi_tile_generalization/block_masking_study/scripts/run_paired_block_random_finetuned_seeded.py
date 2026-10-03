"""
run_paired_block_random_finetuned.py
-------------------------------------
Identical protocol to run_paired_block_random.py (same 500 study chips, same
4 ratios, 5 trials, paired block/random maskers, matched summer-patch count)
but loads the FINE-TUNED checkpoint (checkpoints/{backbone}/best.pt) instead
of the zero-shot weights. Writes to outputs_finetuned/ so the existing
zero-shot results_*.csv files are never touched -- both sets share the same
500 chips, so they're directly comparable row-for-row.
"""
import sys, csv, random, argparse
from pathlib import Path
import numpy as np
import torch
import rasterio
import yaml

REPO  = Path("~/Prithvi/Prithvi-Sandbox").expanduser()
STUDY = REPO / "multi_tile_generalization" / "block_masking_study"
MASK  = STUDY / "masking"
CKPT_DIR = STUDY / "checkpoints"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "patch_masking_study"))
sys.path.insert(0, str(MASK))

from terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from temporal_gap_masker import (
    build_block_noise_mask  as fixed_block,
    build_random_noise_mask as fixed_random,
    pixel_map,
)

CFG_PATH = STUDY / "config" / "block_masking_config.yaml"
IMG, BANDS, T, FRAME = 224, 6, 3, 1
N_TRIALS = 5
RATIOS   = [0.20, 0.40, 0.60, 0.80]
N_CHIPS, CHIP_SEED = 500, 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def expand(p):
    return Path(p).expanduser()


def get_chips(cfg):
    """Same canonical 500 study chips as the zero-shot eval -- unchanged,
    so results are chip-for-chip comparable."""
    d = REPO / "multi_tile_generalization" / "study_chips_500"
    files = sorted(p for p in d.glob("chip_*_merged.tif")
                   if not p.name.startswith("._")) if d.exists() else []
    if len(files) >= N_CHIPS:
        return files[:N_CHIPS] if len(files) > N_CHIPS else files
    d2 = expand(cfg["data"]["chips_dir"])
    allf = sorted(p for p in d2.glob("chip_*_merged.tif")
                  if not p.name.startswith("._"))
    rng = random.Random(CHIP_SEED)
    return rng.sample(allf, min(N_CHIPS, len(allf)))


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


def load_seed_lookup(zeroshot_csv_path):
    """Read trial_seed from the zero-shot results file so the fine-tuned
    eval replays the EXACT same masks. Do not recompute hash((chip, r, t))
    here -- Python's str hash is randomized per-process (PYTHONHASHSEED),
    so an independent process launch gets different seeds for the same
    key, silently breaking the zero-shot vs. fine-tuned pairing."""
    lookup = {}
    with open(zeroshot_csv_path) as fh:
        for row in csv.DictReader(fh):
            key = (row["chip"], row["mask_ratio"], row["trial"])
            lookup[key] = int(row["trial_seed"])
    return lookup


def load_done(csv_path):
    done = set()
    if csv_path.exists():
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                done.add((row["chip"], row["mask_ratio"], row["trial"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_cfg()
    cfg["backbones"] = {k: v for k, v in cfg["backbones"].items() if k == args.backbone}
    assert cfg["backbones"], f"unknown backbone {args.backbone}"
    out_dir = STUDY / "outputs_finetuned" / "seeds" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chips = get_chips(cfg)
    print(f"Chips: {len(chips)}  device: {device}")
    if len(chips) != N_CHIPS:
        print(f"  WARNING: expected {N_CHIPS} chips, got {len(chips)}")

    fields = ["backbone", "chip", "mask_ratio", "trial", "trial_seed",
              "summer_masked", "block_psnr", "random_psnr", "delta_rand_minus_block"]

    for bb, bcfg in cfg["backbones"].items():
        patch = bcfg["patch_size"]
        ckpt_dir = CKPT_DIR / "seeds" / f"seed{args.seed}" / bb
        ckpt_file = ckpt_dir / "best.pt"
        if not ckpt_file.exists():
            print(f"  SKIP {bb}: no finetuned checkpoint at {ckpt_file}")
            continue

        csv_path = out_dir / f"results_{bb}.csv"
        zeroshot_csv = expand(cfg["output"]["dir"]) / f"results_{bb}.csv"
        if not zeroshot_csv.exists():
            print(f"  SKIP {bb}: no zero-shot results at {zeroshot_csv} to pair against")
            continue
        seed_lookup = load_seed_lookup(zeroshot_csv)
        done = load_done(csv_path)
        print(f"\n{'='*60}\n{bb} (FINETUNED)  patch={patch}  (already done: {len(done)})\n{'='*60}")

        model, _, mean, std, sp = load_prithvi_from_terratorch(
            backbone_name=bb, base_dir=ckpt_dir,
            checkpoint_filename="best.pt", num_frames=T, device=device)
        assert sp == patch

        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                w.writeheader()

            total = len(chips) * len(RATIOS) * N_TRIALS
            n = 0
            for chip in chips:
                try:
                    x, gt = load_norm_and_gt(chip, mean, std)
                except Exception as e:
                    print(f"  SKIP {chip.name}: {e}")
                    continue
                for r in RATIOS:
                    for t in range(N_TRIALS):
                        n += 1
                        key = (chip.name, f"{r}", f"{t}")
                        if key in done:
                            continue
                        seed_key = (chip.name, f"{r}", f"{t}")
                        if seed_key not in seed_lookup:
                            print(f"  SKIP {seed_key}: no matching zero-shot seed found")
                            continue
                        seed = seed_lookup[seed_key]
                        try:
                            nb, grb, idxb = fixed_block(r, patch, IMG, T, FRAME, trial_seed=seed)
                            rb = recon_unit(model, x, nb, grb, mean, std)
                            bp = masked_psnr(rb, gt, pixel_map(idxb, patch, IMG, T, FRAME))

                            nr, grr, idxr = fixed_random(r, patch, IMG, T, FRAME,
                                                         trial_seed=seed, n_summer_masked=len(idxb))
                            rr = recon_unit(model, x, nr, grr, mean, std)
                            rp = masked_psnr(rr, gt, pixel_map(idxr, patch, IMG, T, FRAME))

                            w.writerow({
                                "backbone": bb, "chip": chip.name, "mask_ratio": r,
                                "trial": t, "trial_seed": seed,
                                "summer_masked": len(idxb),
                                "block_psnr": round(bp, 4), "random_psnr": round(rp, 4),
                                "delta_rand_minus_block": round(rp - bp, 4),
                            })
                            fh.flush()
                        except Exception as e:
                            print(f"  ERROR {chip.name} r={r} t={t}: {e}")
                        if n % 200 == 0:
                            print(f"  [{bb}] {n}/{total}")
        print(f"  Saved: {csv_path}")

    print("\nAll backbones complete.")


if __name__ == "__main__":
    main()
