"""
Training loop for the ControlNet spectral band adapter, one backbone at a
time. Only hint_encoder (input side) and spectral_head (output side) are
trained -- everything else frozen, confirmed via verify_full_integration.py
and verify_training_step.py. Mask ratio and geometry are randomized per
sample (pace_dataset.py) so one trained model generalizes across the full
eval grid (20/40/60/80% x contiguous/scattered) instead of needing 8
separately-trained models.

Usage:
    python train_band_adapter.py --backbone 100M --epochs 5
    python train_band_adapter.py --backbone 600M --epochs 5 --lr 5e-5

batch_size defaults to 1: every verification so far (including the real
end-to-end training-step check) has only been run at batch_size=1, and
291-band tiles through a full transformer forward+backward may be tight on
24GB VRAM at higher batch sizes -- raise it and watch nvidia-smi rather
than assuming it fits.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402
from controlnet_encoder_adapter import ControlNetEncoderAdapter  # noqa: E402
from spectral_output_head import SpectralOutputAdapter  # noqa: E402
from trainable_forward import run_masked_forward_trainable  # noqa: E402
from masked_reconstruction import unpatchify_bands, masked_reconstruction_loss  # noqa: E402
from pace_dataset import PaceMaskedDataset, train_val_split  # noqa: E402

BACKBONE_SPECS = {
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16, embed_dim=768),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", patch_size=14, embed_dim=1280),
}


def run_one_batch(model, enc_adapter, dec_adapter, batch, spec, device):
    masked_cube = batch["masked_cube"].to(device)
    target_cube = batch["target_cube"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    band_mask = batch["band_mask"].to(device)  # (B, 291), per-sample -- see pace_dataset.py

    B, _, H, W = masked_cube.shape
    placeholder = torch.zeros(B, 6, 1, H, W, device=device)
    enc_adapter.set_pace_cube(masked_cube)

    run_masked_forward_trainable(
        model, placeholder, temporal_coords=None, location_coords=None,
        mask_ratio=0.0, noise=None,  # spatial masking disabled -- band masking is the task
    )

    patch_tokens = dec_adapter.last_output[:, 1:, :]  # drop CLS
    grid = H // spec["patch_size"]
    prediction = unpatchify_bands(
        patch_tokens, num_channels=291, patch_size=spec["patch_size"],
        num_patches_h=grid, num_patches_w=grid,
    ).squeeze(2)  # drop size-1 temporal dim

    return masked_reconstruction_loss(prediction, target_cube, band_mask, valid_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["100M", "600M"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--eval_exclude_path", type=str, default="hsi_diverse_100.txt",
                         help="tile IDs to exclude from train/val -- keeps the zero-shot "
                              "pilot's eval set disjoint from fine-tuning data")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    spec = BACKBONE_SPECS[args.backbone]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"backbone: {args.backbone}, device: {device}")

    checkpoints = list(spec["base_dir"].glob("*.pt"))
    assert len(checkpoints) == 1, f"expected one .pt in {spec['base_dir']}, found: {checkpoints}"

    model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
        backbone_name=args.backbone, base_dir=str(spec["base_dir"]),
        checkpoint_filename=checkpoints[0].name, num_frames=1, device=device,
    )
    print(f"loaded {args.backbone}, spatial_patch={spatial_patch}")

    enc_adapter = ControlNetEncoderAdapter(
        model.encoder, embed_dim=spec["embed_dim"], patch_size=spec["patch_size"],
    ).to(device)
    dec_adapter = SpectralOutputAdapter(model.decoder, patch_size=spec["patch_size"]).to(device)

    trainable_params = (
        list(enc_adapter.band_adapter.hint_encoder.parameters())
        + list(dec_adapter.spectral_head.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    print(f"excluding eval tiles listed in: {args.eval_exclude_path}")
    train_tiles, val_tiles = train_val_split(
        val_fraction=args.val_fraction, exclude_ids_path=args.eval_exclude_path,
    )
    print(f"train tiles: {len(train_tiles)}, val tiles: {len(val_tiles)}")
    if len(train_tiles) == 0 or len(val_tiles) == 0:
        raise SystemExit("not enough tiles for a train/val split -- extract more from the tar, "
                          "or lower --val_fraction")

    train_ds = PaceMaskedDataset(train_tiles, target_size=112, seed=0)
    val_ds = PaceMaskedDataset(val_tiles, target_size=112, seed=1, deterministic=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    checkpoint_dir = (
        Path(args.checkpoint_dir) if args.checkpoint_dir
        else Path.home() / "Prithvi" / "hyperfm_pilot" / "checkpoints" / args.backbone
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints -> {checkpoint_dir}")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.eval()  # frozen backbone; dropout is p=0.0 throughout anyway (confirmed via repr)
        enc_adapter.train()
        dec_adapter.train()

        epoch_start = time.time()
        train_losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, n_scored = run_one_batch(model, enc_adapter, dec_adapter, batch, spec, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.6f}")

        mean_train_loss = sum(train_losses) / len(train_losses)

        enc_adapter.eval()
        dec_adapter.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                loss, n_scored = run_one_batch(model, enc_adapter, dec_adapter, batch, spec, device)
                val_losses.append(loss.item())
        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")

        elapsed = time.time() - epoch_start
        print(f"epoch {epoch}: train_loss={mean_train_loss:.6f} val_loss={mean_val_loss:.6f} "
              f"({elapsed:.1f}s)")

        ckpt = {
            "epoch": epoch,
            "backbone": args.backbone,
            "hint_encoder_state": enc_adapter.band_adapter.hint_encoder.state_dict(),
            "spectral_head_state": dec_adapter.spectral_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
        }
        torch.save(ckpt, checkpoint_dir / "latest.pt")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(ckpt, checkpoint_dir / "best.pt")
            print(f"  new best val_loss: {best_val_loss:.6f} -> {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
