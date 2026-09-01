"""
Real PACE tile loader.

Two things confirmed via diagnose_real_data.py / diagnose_wavelengths_and_nan.py
(Sep 1) that the dummy-tensor tests so far didn't have to deal with:

  1. Tiles are 96x96, not Prithvi's native 224x224. 600M's patch_size=14
     doesn't divide 96 evenly (96/14=6.86) -- patch_embed would silently
     crop ~12 edge pixels (confirmed via PatchEmbed.forward's own source:
     it warns, doesn't error, and drops the remainder). Padding to 112x112
     fixes this cleanly for BOTH backbones at once: 112/16=7, 112/14=8,
     both exact -- no cropping, no backbone-specific special-casing.

  2. Real tiles contain scattered per-band NaN values: up to ~0.8% of
     pixels affected in the tiles checked, NEVER a whole pixel (a few bad
     detector bands per pixel, not cloud-masked regions -- that fraction
     is far below HyperFM's stated >60% cloud coverage, so this is very
     likely per-detector bad-pixel flags in the raw L1B data, not the
     cloud mask, which HyperFM ships separately in target/). NaNs are
     replaced with 0.0 for the model input; a validity mask is returned
     alongside so a loss function can exclude both NaN pixels AND the
     padded border from training, rather than learning against fabricated
     values.

Padding mode is 'reflect', not zero-padding: zeros aren't a physically
plausible radiance value (real range runs roughly -0.02 to ~2, per checked
tiles) and would hand the frozen pretrained backbone a jarring, obviously
out-of-distribution border. Reflected edge pixels are still not real
independent measurements, so the padded region is marked INVALID in the
mask regardless -- it's context for the model to look at, not something to
train against.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_pace_tile(npy_path, target_size: int = 112):
    """
    npy_path: path to a HyperFM250K .npy file, shape (96, 96, 291) float32.
    target_size: pad to this size (both spatial dims). Default 112, chosen
    so both 100M's patch_size=16 (112/16=7) and 600M's patch_size=14
    (112/14=8) divide evenly with no cropped remainder.

    Returns:
        tile: (1, 291, target_size, target_size) float32 tensor.
              NaN bad-pixels -> 0.0; border -> reflected edge values.
        valid_mask: (1, 291, target_size, target_size) bool tensor. True
              only where the value is BOTH real data (not NaN) AND not
              padding. Use this to mask any reconstruction loss.
    """
    arr = np.load(npy_path)  # (H, W, C) = (96, 96, 291)
    h, w, c = arr.shape

    nan_mask = np.isnan(arr)
    arr = np.nan_to_num(arr, nan=0.0)

    tile = torch.from_numpy(arr).permute(2, 0, 1).float()        # (291, H, W)
    valid = torch.from_numpy(~nan_mask).permute(2, 0, 1)         # (291, H, W), True=real data

    if h != target_size or w != target_size:
        pad_h = target_size - h
        pad_w = target_size - w
        assert pad_h >= 0 and pad_w >= 0, (
            f"tile {h}x{w} is larger than target_size={target_size}"
        )
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        tile = F.pad(tile, pad, mode="reflect")
        # padded border is context only, never counts toward loss --
        # F.pad has no bool support, pad as float then threshold back to bool
        valid = F.pad(valid.float(), pad, mode="constant", value=0.0) > 0.5

    return tile.unsqueeze(0), valid.unsqueeze(0)  # add batch dim -> (1, 291, T, T)


def list_pace_tiles(data_dir=None):
    """Convenience: all .npy tile paths under the standard hyperfm_pilot data dir."""
    if data_dir is None:
        data_dir = Path.home() / "Prithvi" / "hyperfm_pilot" / "cvpr_dataset" / "hsi"
    return sorted(Path(data_dir).glob("*.npy"))
