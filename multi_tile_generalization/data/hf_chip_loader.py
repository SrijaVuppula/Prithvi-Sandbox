import numpy as np
import torch
import random
import rasterio
from pathlib import Path


def load_chips(hf_repo, split, num_chips, seed, num_workers=4):
    root = Path(__file__).resolve().parent.parent
    chips_dir = root / "training_chips"

    if not chips_dir.exists():
        raise FileNotFoundError(f"training_chips/ not found at {chips_dir}")

    all_files = sorted(chips_dir.glob("chip_*_merged.tif"))
    total = len(all_files)
    print(f"[chip_loader] Found {total} merged tif chips")

    rng = random.Random(seed)
    selected = rng.sample(all_files, min(num_chips, total))
    print(f"[chip_loader] Sampled {len(selected)} chips (seed={seed})")

    chips, valid_indices = [], []
    for i, fpath in enumerate(selected):
        try:
            tensor = _load_tif(fpath)
            if tensor is not None:
                chips.append(tensor)
                valid_indices.append(i)
        except Exception as e:
            print(f"[chip_loader] WARNING: Skipping {fpath.name} — {e}")
        if (i + 1) % 50 == 0:
            print(f"[chip_loader] Loaded {i+1}/{len(selected)} ({len(chips)} valid)...")

    print(f"[chip_loader] Done: {len(chips)} valid chips.")
    return chips, valid_indices


def _load_tif(fpath: Path):
    with rasterio.open(fpath) as src:
        data = src.read()              # (18, 224, 224) int16
    C_total, H, W = data.shape
    T, C = 3, 6
    assert C_total == T * C, f"Expected 18 bands, got {C_total}"
    # Keep in raw HLS reflectance scale [0, 10000] — normalization done per model
    data = data.reshape(T, C, H, W).astype(np.float32)
    return torch.tensor(data, dtype=torch.float32)
