"""
Verification for SpectralOutputAdapter against the real 100M/600M decoders.

Deliberately isolated from the input-side ControlNetBandAdapter: this test
runs the model's own native 6-band forward path (via the existing
run_masked_forward, unmodified) purely to make decoder_pred fire once, so
the output-side hook can be checked on its own. Wiring both sides together
into one full forward pass is the next step, after this one is confirmed.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from terratorch_loader import load_prithvi_from_terratorch, run_masked_forward  # noqa: E402
from spectral_output_head import SpectralOutputAdapter  # noqa: E402

BACKBONE_SPECS = {
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", patch_size=14),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

for name, spec in BACKBONE_SPECS.items():
    print(f"{'=' * 60}\n{name}\n{'=' * 60}")
    base_dir = spec["base_dir"]

    checkpoints = list(base_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        print(f"Expected exactly one .pt file in {base_dir}, found: {checkpoints}")
        continue

    model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
        backbone_name=name,
        base_dir=str(base_dir),
        checkpoint_filename=checkpoints[0].name,
        num_frames=1,
        device=device,
    )
    print(f"loaded OK -- spatial_patch={spatial_patch}, bands={bands}")

    try:
        adapter = SpectralOutputAdapter(model.decoder, patch_size=spec["patch_size"]).to(device)
    except Exception as e:
        print(f"SpectralOutputAdapter init FAILED: {type(e).__name__}: {e}")
        continue
    print("SpectralOutputAdapter attached OK")

    x_norm = torch.randn(1, 6, 1, 224, 224, device=device)
    try:
        result = run_masked_forward(
            model, x_norm, temporal_coords=None, location_coords=None,
            mask_ratio=0.2, noise=None,
        )
    except Exception as e:
        print(f"run_masked_forward FAILED: {type(e).__name__}: {e}")
        continue

    if adapter.last_output is None:
        print("Hook never fired -- decoder_pred was not called during this forward pass.")
        continue

    print(f"native decoder_pred output shape: {tuple(adapter.last_native_output.shape)}")
    print(f"291-band head output shape:       {tuple(adapter.last_output.shape)}")

    n_pos = spec["patch_size"] ** 2
    expected = (adapter.last_native_output.shape[0], adapter.last_native_output.shape[1], n_pos * 291)
    if tuple(adapter.last_output.shape) == expected:
        print("shape check OK")
    else:
        print(f"shape MISMATCH -- expected {expected}")
