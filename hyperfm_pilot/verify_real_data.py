"""
Verification with REAL PACE data (not random dummy tensors), at the real
112x112 padded tile size, through the full ControlNet-adapter pipeline.
Same checks as verify_full_integration.py, but the actual thing this is
all for: does a real HyperFM250K tile flow through both adapters and the
frozen model correctly.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402
from controlnet_encoder_adapter import ControlNetEncoderAdapter  # noqa: E402
from spectral_output_head import SpectralOutputAdapter  # noqa: E402
from trainable_forward import run_masked_forward_trainable  # noqa: E402
from pace_tile_loader import load_pace_tile, list_pace_tiles  # noqa: E402

BACKBONE_SPECS = {
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16, embed_dim=768),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", patch_size=14, embed_dim=1280),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

tiles = list_pace_tiles()
if not tiles:
    raise SystemExit("no .npy tiles found -- check the data directory")
tile_path = tiles[0]
print(f"using real tile: {tile_path.name}")

pace_cube, valid_mask = load_pace_tile(tile_path, target_size=112)
print(f"loaded tile shape: {tuple(pace_cube.shape)}, valid_mask shape: {tuple(valid_mask.shape)}")
print(f"valid fraction: {valid_mask.float().mean().item():.4f}")
pace_cube = pace_cube.to(device)

for name, spec in BACKBONE_SPECS.items():
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    base_dir = spec["base_dir"]
    checkpoints = list(base_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        print(f"Expected exactly one .pt file in {base_dir}, found: {checkpoints}")
        continue

    model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
        backbone_name=name, base_dir=str(base_dir),
        checkpoint_filename=checkpoints[0].name, num_frames=1, device=device,
    )
    print(f"loaded OK -- spatial_patch={spatial_patch}, bands={bands}")

    try:
        enc_adapter = ControlNetEncoderAdapter(
            model.encoder, embed_dim=spec["embed_dim"], patch_size=spec["patch_size"],
        ).to(device)
        dec_adapter = SpectralOutputAdapter(model.decoder, patch_size=spec["patch_size"]).to(device)
    except Exception as e:
        print(f"adapter init FAILED: {type(e).__name__}: {e}")
        continue
    print("both adapters attached OK")

    placeholder = torch.zeros(1, 6, 1, 112, 112, device=device)
    enc_adapter.set_pace_cube(pace_cube)

    try:
        run_masked_forward_trainable(
            model, placeholder, temporal_coords=None, location_coords=None,
            mask_ratio=0.2, noise=None,
        )
    except Exception as e:
        print(f"run_masked_forward_trainable FAILED: {type(e).__name__}: {e}")
        continue

    if dec_adapter.last_output is None:
        print("decoder hook never fired.")
        continue
    print(f"real-data 291-band output shape: {tuple(dec_adapter.last_output.shape)}")

    grid = 112 // spec["patch_size"]
    n_pos = grid ** 2
    expected_tokens = n_pos + 1  # +CLS
    actual_tokens = dec_adapter.last_native_output.shape[1]
    print(f"expected patch grid: {grid}x{grid} = {n_pos} + 1 CLS = {expected_tokens} tokens; "
          f"got {actual_tokens}")
    if actual_tokens != expected_tokens:
        print("MISMATCH -- padding math or patch grid assumption is wrong somewhere")

    # --- gradient check with real data ---
    model.zero_grad()
    enc_adapter.zero_grad()
    dec_adapter.zero_grad()
    enc_adapter.set_pace_cube(pace_cube)
    run_masked_forward_trainable(model, placeholder, temporal_coords=None, location_coords=None,
                                  mask_ratio=0.2, noise=None)
    loss = dec_adapter.last_output.sum()
    loss.backward()

    hint_grad = sum(p.grad.abs().sum().item() for p in enc_adapter.band_adapter.hint_encoder.parameters()
                     if p.grad is not None)
    head_grad = sum(p.grad.abs().sum().item() for p in dec_adapter.spectral_head.parameters()
                     if p.grad is not None)
    native_grads_nonzero = [n for n, p in model.named_parameters() if p.grad is not None]

    print(f"hint_encoder grad norm: {hint_grad:.4f}")
    print(f"spectral_head grad norm: {head_grad:.4f}")
    if native_grads_nonzero:
        print(f"WARNING -- {len(native_grads_nonzero)} native model params got gradients")
    else:
        print("native model fully frozen -- no native param received a gradient")
