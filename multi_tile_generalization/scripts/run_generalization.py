import argparse, csv, sys, time, random
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(ROOT))

from data.hf_chip_loader import load_chips
from metrics.evaluate_masked import compute_metrics
from terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from masking.patch_masker import build_noise_for_mask_ratio, get_masked_token_indices


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize_chip(chip_tensor, mean, std):
    """
    chip_tensor: (T, C, H, W) in raw HLS scale [0, 10000]
    mean, std:   lists of length C in HLS scale
    Returns:     (T, C, H, W) normalized for model input
    """
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1)
    return (chip_tensor - mean_t) / std_t

def to_01(chip_tensor):
    """Convert raw HLS [0, 10000] to [0, 1] for metric computation."""
    return torch.clamp(chip_tensor / 10000.0, 0.0, 1.0)

def run_backbone(backbone_cfg, chips, config, out_dir, resume):
    name       = backbone_cfg["name"]
    base_dir   = Path(backbone_cfg["base_dir"])
    checkpoint = backbone_cfg["checkpoint"]
    csv_path   = out_dir / f"{name}_results.csv"

    if resume and csv_path.exists():
        with open(csv_path) as f:
            lines = f.readlines()
        expected = len(chips) * len(config["masking"]["ratios"])
        if len(lines) - 1 >= expected:
            print(f"[run] SKIP {name} — already complete")
            return

    print(f"\n{'='*60}\n[run] Backbone: {name}\n{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bands, mean, std, patch_size = load_prithvi_from_terratorch(
        backbone_name=name,
        base_dir=base_dir,
        checkpoint_filename=checkpoint,
        num_frames=3,
        device=device,
    )
    model.eval()
    print(f"[run] Model loaded on {device}, patch_size={patch_size}")

    frame_idx = config["masking"]["frame_idx"]
    ratios    = config["masking"]["ratios"]
    seed      = config["experiment"]["seed"]

    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)

    fieldnames = ["chip_idx","backbone","mask_ratio",
                  "mae","psnr","ssim",
                  "masked_mae","masked_psnr","masked_ssim","time_sec"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for chip_i, chip_tensor in enumerate(chips):
            # chip_tensor: (T, C, H, W) raw HLS [0, 10000]
            H, W = chip_tensor.shape[-2], chip_tensor.shape[-1]

            # Normalize for model input → (1, C, T, H, W)
            chip_norm = normalize_chip(chip_tensor, mean, std)
            x = chip_norm.permute(1, 0, 2, 3).unsqueeze(0).to(device)

            # Ground truth in [0, 1] for metrics
            target_frame = to_01(chip_tensor[frame_idx]).numpy()  # (C, H, W)

            for ratio in ratios:
                t0 = time.time()
                noise, tokens_per_frame, seq_length = build_noise_for_mask_ratio(
                    x, ratio, patch_size, device,
                    seed=seed + chip_i * 1000 + int(ratio * 100)
                )

                try:
                    loss, pred_img, rec_img, mask_img, x_cpu = run_masked_forward(
                        model=model,
                        x=x,
                        temporal_coords=None,
                        location_coords=None,
                        mask_ratio=ratio,
                        noise=noise,
                    )

                    # rec_img: (1, C, T, H, W) in normalized space
                    pred_norm = rec_img[0, :, frame_idx, :, :]  # (C, H, W)

                    # Denormalize → raw HLS scale → [0, 1]
                    pred_frame = (pred_norm * std_t + mean_t)
                    pred_frame = torch.clamp(pred_frame / 10000.0, 0.0, 1.0).numpy()

                    # Build pixel mask for target frame only
                    masked_indices = get_masked_token_indices(noise)
                    nh, nw = H // patch_size, W // patch_size
                    frame_start = frame_idx * tokens_per_frame
                    frame_end   = frame_start + tokens_per_frame
                    frame_indices = masked_indices[
                        (masked_indices >= frame_start) & (masked_indices < frame_end)
                    ] - frame_start

                    pixel_mask = np.zeros((nh * nw,), dtype=bool)
                    if len(frame_indices) > 0:
                        pixel_mask[frame_indices.cpu().numpy()] = True
                    pixel_mask = pixel_mask.reshape(nh, nw)
                    pixel_mask = np.kron(pixel_mask, np.ones((patch_size, patch_size), dtype=bool))

                    metrics = compute_metrics(pred_frame, target_frame, pixel_mask)
                    writer.writerow({
                        "chip_idx": chip_i, "backbone": name, "mask_ratio": ratio,
                        **metrics, "time_sec": round(time.time() - t0, 2)
                    })
                    f.flush()
                    print(f"  chip={chip_i} ratio={ratio} masked_PSNR={metrics['masked_psnr']:.2f}")

                except Exception as e:
                    print(f"  [WARN] chip={chip_i} ratio={ratio}: {e}")

            if (chip_i + 1) % 25 == 0:
                print(f"  [run] {name}: {chip_i+1}/{len(chips)} chips done")

    del model
    torch.cuda.empty_cache()
    print(f"[run] {name} complete → {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/generalization_config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    per_tile_dir = ROOT / config["output"]["per_tile_dir"]
    per_tile_dir.mkdir(parents=True, exist_ok=True)

    chips, chip_indices = load_chips(
        hf_repo=config["dataset"]["hf_repo"],
        split=config["dataset"]["split"],
        num_chips=config["experiment"]["num_chips"],
        seed=config["experiment"]["seed"],
        num_workers=config["dataset"]["num_workers"],
    )

    with open(per_tile_dir / "sampled_chip_indices.txt", "w") as f:
        f.write("\n".join(map(str, chip_indices)))

    for backbone_cfg in config["backbones"]:
        run_backbone(backbone_cfg, chips, config, per_tile_dir, resume=args.resume)

    print("\n[main] All backbones complete. Run aggregate_results.py next.")

if __name__ == "__main__":
    main()
