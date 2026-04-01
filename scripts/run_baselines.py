"""
Phase 1 baseline sweep — all four experimental dimensions.

Usage:
    python scripts/run_baselines.py --config config/experiment_config.yaml

Produces:
    outputs/results.csv          — one row per (tile × backbone × position × T × gap)
    outputs/runs/<run_id>.json   — full sidecar per run
    outputs/plots/<run_id>.png   — comparison plot (up to save_plots_limit)
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path
from itertools import product

# Local imports — adjust sys.path if running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.hls_loader import select_files_for_condition, load_sample_from_inference_module
from masking.temporal_masker import get_masked_frame_index, build_noise_from_frame_idx
from metrics.evaluate import evaluate_reconstruction, compute_gap_days
from logging_utils.experiment_logger import ExperimentLogger


def load_model(base_dir: Path, checkpoint_filename: str,
               num_frames: int, device: torch.device):
    """Load a Prithvi backbone. Reuses your existing load_model logic."""
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from prithvi_mae import PrithviMAE  # type: ignore
    import yaml as _yaml

    config_path     = base_dir / "config.json"
    checkpoint_path = base_dir / checkpoint_filename

    with config_path.open() as f:
        config = _yaml.safe_load(f)["pretrained_cfg"]

    bands = config["bands"]
    mean  = config["mean"]
    std   = config["std"]
    config.update(num_frames=num_frames, in_chans=len(bands))

    model = PrithviMAE(**config).to(device)

    try:
        state_dict = torch.load(
            checkpoint_path, weights_only=True, map_location=device
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    # Swap positional embeddings to match runtime num_frames
    for key in list(state_dict.keys()):
        if key == "encoder.pos_embed":
            state_dict[key] = model.encoder.pos_embed
        elif key == "decoder.decoder_pos_embed":
            state_dict[key] = model.decoder.decoder_pos_embed

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, bands, mean, std


@torch.no_grad()
def run_one_condition(
    model,
    x: torch.Tensor,
    temporal_coords: torch.Tensor,
    location_coords: torch.Tensor,
    frame_idx: int,
    patch_size: int,
    device: torch.device,
) -> dict:
    """Single forward pass for one masked frame. Returns raw outputs."""
    noise, tokens_per_frame, seq_length = build_noise_from_frame_idx(
        x, frame_idx, patch_size, device
    )
    n_frames   = x.shape[2]
    mask_ratio = 1.0 / n_frames   # one fully-masked frame out of T

    loss, pred, mask = model(
        x,
        temporal_coords=temporal_coords,
        location_coords=location_coords,
        mask_ratio=mask_ratio,
        noise=noise,
    )

    orig_h, orig_w = x.shape[-2], x.shape[-1]
    mask_img = model.unpatchify(
        mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]).detach().cpu(),
        image_size=(orig_h, orig_w),
    )
    pred_img = model.unpatchify(pred.detach().cpu(), image_size=(orig_h, orig_w))

    x_cpu   = x.detach().cpu()
    rec_img = x_cpu.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]

    return {
        "loss":       float(loss),
        "mask_ratio": mask_ratio,
        "x_cpu":      x_cpu,
        "rec_img":    rec_img,
    }


def save_plot(out_path: Path, x_cpu, rec_img, frame_idx, bands, mean, std):
    """Saves a side-by-side original vs. reconstruction plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def to_rgb(tensor_cthw, t):
        rgb_idx = [bands.index("B04"), bands.index("B03"), bands.index("B02")]
        mean_t = torch.tensor(np.asarray(mean)[:, None, None], dtype=torch.float32)
        std_t  = torch.tensor(np.asarray(std)[:, None, None],  dtype=torch.float32)
        img = tensor_cthw[:, t].clone() * std_t + mean_t
        return (img[rgb_idx] / 3000.0).clamp(0, 1).permute(1, 2, 0).numpy()

    n = x_cpu.shape[2]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for t in range(n):
        axes[0, t].imshow(to_rgb(x_cpu[0], t))
        axes[0, t].set_title(f"original T{t}")
        axes[0, t].axis("off")
        label = f"recon T{t}" + (" <-- masked" if t == frame_idx else "")
        axes[1, t].imshow(to_rgb(rec_img[0], t))
        axes[1, t].set_title(label)
        axes[1, t].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(cfg: dict):
    base_dir   = Path(cfg["model"]["base_dir"])
    device     = torch.device(cfg["model"]["device"]
                               if torch.cuda.is_available() else "cpu")
    patch_size = cfg["data"]["patch_size"]
    output_dir = Path(cfg["experiment"]["output_dir"])
    save_plots = cfg["experiment"].get("save_plots", True)
    plot_limit = cfg["experiment"].get("save_plots_limit", 20)
    plot_count = 0

    # Apply source patches once at startup
    from inference.runner import apply_all_patches, load_model, run_one_condition
    apply_all_patches(base_dir)

    logger = ExperimentLogger(output_dir=str(output_dir))

    tiles       = cfg["data"]["tiles"]
    backbones   = cfg["model"]["backbones"]
    positions   = cfg["experiment"]["mask_positions"]
    seq_lengths = cfg["experiment"]["sequence_lengths"]
    gap_types   = cfg["experiment"]["gap_types"]

    total = (len(tiles) * len(backbones) * len(positions)
             * len(seq_lengths) * len(gap_types))
    done  = 0

    for tile_cfg in tiles:
        tile_id   = tile_cfg["id"]
        all_files = tile_cfg["files"]

        for bb, pos, n_frames, gap_cfg in product(
            backbones, positions, seq_lengths, gap_types
        ):
            done += 1
            gap_name = gap_cfg["name"]
            print(f"\n[{done}/{total}] tile={tile_id} backbone={bb['name']} "
                  f"pos={pos} T={n_frames} gap={gap_name}")

            # --- select files for this (n_frames, gap_type) condition ---
            try:
                selected_files = select_files_for_condition(
                    all_files, n_frames, gap_cfg
                )
            except ValueError as e:
                print(f"  SKIP: {e}")
                continue

            gap_days = compute_gap_days(selected_files)

            # --- load model (reload each backbone to avoid VRAM accumulation) ---
            try:
                model, bands, mean, std = load_model(
                    base_dir=base_dir,
                    checkpoint_filename=bb["checkpoint"],
                    num_frames=n_frames,
                    device=device,
                )
            except Exception as e:
                print(f"  ERROR loading backbone {bb['name']}: {e}")
                continue

            # --- load data ---
            x, temporal_coords, location_coords, _ = load_sample_from_inference_module(
                base_dir=base_dir,
                file_paths=selected_files,
                mean=mean,
                std=std,
                device=device,
            )

            # --- run inference ---
            frame_idx = get_masked_frame_index(pos, n_frames)
            try:
                result = run_one_condition(
                    model=model,
                    x=x,
                    temporal_coords=temporal_coords,
                    location_coords=location_coords,
                    frame_idx=frame_idx,
                    patch_size=patch_size,
                    device=device,
                )
            except Exception as e:
                print(f"  ERROR during inference: {e}")
                del model
                torch.cuda.empty_cache()
                continue

            # --- metrics ---
            pred_frame = result["rec_img"][0, :, frame_idx]  # (C, H, W)
            gt_frame   = result["x_cpu"][0, :, frame_idx]
            metrics    = evaluate_reconstruction(pred_frame, gt_frame, mean, std)

            # --- log ---
            run_id = logger.log(
                backbone=bb["name"],
                mask_position=pos,
                n_frames=n_frames,
                gap_type=gap_name,
                tile_id=tile_id,
                masked_frame_idx=frame_idx,
                gap_days=gap_days,
                metrics=metrics,
                loss=result["loss"],
                mask_ratio=result["mask_ratio"],
                checkpoint=bb["checkpoint"],
            )

            # --- optional plot ---
            if save_plots and plot_count < plot_limit:
                plot_path = output_dir / "plots" / f"{run_id}.png"
                save_plot(plot_path, result["x_cpu"], result["rec_img"],
                          frame_idx, bands, mean, std)
                plot_count += 1

            del model
            torch.cuda.empty_cache()

    print(f"\nSweep complete. {done} conditions run.")
    print(f"Results: {output_dir / 'results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)