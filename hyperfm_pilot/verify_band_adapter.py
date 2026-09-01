"""
Verification script for the ControlNet spectral band adapter, 100M + 600M only.

Does three things per backbone, in order, and stops with a clear message if
any step fails rather than crashing opaquely:
  1. Reads patch_size/embed_dim straight from that backbone's config.json
     (600M nests some keys under "pretrained_cfg" -- handled).
  2. Loads the real model via terratorch_loader.load_prithvi_from_terratorch,
     then finds the actual patch_embed submodule by name rather than
     assuming an attribute path.
  3. Runs ControlNetBandAdapter on a random PACE-shaped tensor (1, 291, 224, 224)
     to confirm the shapes line up end-to-end.

num_frames=1 throughout, matching the real PACE data (single-date only, no
temporal dimension yet) -- not the num_frames=3 used in the original
multispectral spring/summer/fall study.
"""

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402
from spectral_band_adapter import ControlNetBandAdapter  # noqa: E402

BACKBONE_DIRS = {
    "100M": Path.home() / "Prithvi" / "prithvi_100M",
    "600M": Path.home() / "Prithvi" / "prithvi_600M",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

for name, base_dir in BACKBONE_DIRS.items():
    print(f"{'=' * 60}\n{name}\n{'=' * 60}")

    # --- 1. Real config values ---
    cfg_path = base_dir / "config.json"
    if not cfg_path.exists():
        print(f"No config.json at {cfg_path} -- skipping {name}")
        continue
    with open(cfg_path) as f:
        cfg = json.load(f)
    pcfg = cfg.get("pretrained_cfg", {})
    raw_patch_size = pcfg.get("patch_size", cfg.get("patch_size"))
    embed_dim = pcfg.get("embed_dim", cfg.get("embed_dim"))
    # config.json's patch_size is [T, H, W] (e.g. [1, 16, 16]), not a scalar --
    # confirmed via diagnose_patch_embed.py. Take the spatial patch only
    # (assumes square patches, true for both 100M and 600M).
    if isinstance(raw_patch_size, (list, tuple)):
        assert raw_patch_size[-2] == raw_patch_size[-1], f"non-square patch: {raw_patch_size}"
        patch_size = raw_patch_size[-1]
    else:
        patch_size = raw_patch_size
    print(f"config.json -> raw patch_size={raw_patch_size} -> using spatial patch_size={patch_size}, embed_dim={embed_dim}")
    if patch_size is None or embed_dim is None:
        print("Could not determine patch_size/embed_dim from config.json:")
        print(json.dumps(cfg, indent=2)[:2000])
        continue

    # --- 2. Find the checkpoint file rather than guessing its exact name ---
    checkpoints = list(base_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        print(f"Expected exactly one .pt file in {base_dir}, found: {checkpoints}")
        continue
    checkpoint_filename = checkpoints[0].name
    print(f"checkpoint: {checkpoint_filename}")

    # --- 3. Load the real model ---
    try:
        model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
            backbone_name=name,
            base_dir=str(base_dir),
            checkpoint_filename=checkpoint_filename,
            num_frames=1,
            device=device,
        )
    except Exception as e:
        print(f"load_prithvi_from_terratorch FAILED: {type(e).__name__}: {e}")
        continue
    print(f"loaded OK -- loader reports spatial_patch={spatial_patch}, bands={bands}")

    # --- 4. Find the patch-embed module by name, don't assume the path ---
    candidates = [(n, m) for n, m in model.encoder.named_modules() if "patch" in n.lower()]
    print("submodules with 'patch' in the name:")
    for n, m in candidates:
        print(f"  encoder.{n}: {type(m).__name__}")
    if not candidates:
        print("No patch_embed-like submodule found. Full encoder repr:")
        print(model.encoder)
        continue
    patch_embed_name, patch_embed = min(candidates, key=lambda nm: nm[0].count("."))
    print(f"using: model.encoder.{patch_embed_name}")

    # --- 5. Wire up the adapter and run one dummy PACE tile through it ---
    try:
        adapter = ControlNetBandAdapter(
            patch_embed=patch_embed, embed_dim=embed_dim, patch_size=patch_size,
        ).to(device)
        dummy_pace = torch.randn(1, 291, 224, 224, device=device)
        out = adapter(dummy_pace)
        print(f"adapter(dummy_pace) OK -- output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"adapter(dummy_pace) FAILED: {type(e).__name__}: {e}")
