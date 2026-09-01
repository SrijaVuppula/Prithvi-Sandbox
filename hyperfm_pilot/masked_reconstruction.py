"""
Spectral-axis masked reconstruction: band masking, unpatchify, and loss.

Deliberately mirrors the Step 1 zero-shot pilot's masking design (contiguous
vs. scattered spectral bands hidden, same two geometries) rather than
Prithvi's native spatial-patch masking -- Phase 2's research question is
about the SPECTRAL axis specifically, so masking happens at the input (hide
K% of the 291 bands before the adapter sees them) with spatial masking
disabled (mask_ratio=0 in run_masked_forward_trainable, nothing spatially
hidden). Loss is scored only on the hidden bands, at valid (non-NaN,
non-padding) positions -- the direct model-based successor to the pilot's
linear-interpolation baseline, same task, different reconstruction method.
"""

import numpy as np
import torch
from einops import rearrange


def apply_band_mask(pace_cube, mask_ratio, geometry="contiguous", rng=None, num_bands=291):
    """
    pace_cube: (B, num_bands, H, W)
    geometry: "contiguous" (one random contiguous block) or "scattered"
    (random individual bands, no replacement) -- same two geometries as the
    Step 1 pilot's spectral_masking_pilot.py.

    Returns:
        masked_cube: pace_cube with hidden bands zeroed out (same tensor
                     the frozen 6-band resampling path AND the hint branch
                     both see -- genuinely hidden, not just down-weighted)
        band_mask: (num_bands,) bool tensor, True = HIDDEN from the model
    """
    if rng is None:
        rng = np.random.default_rng()
    n_hidden = int(round(mask_ratio * num_bands))
    band_mask = np.zeros(num_bands, dtype=bool)

    if geometry == "contiguous":
        start = rng.integers(0, num_bands - n_hidden + 1)
        band_mask[start:start + n_hidden] = True
    elif geometry == "scattered":
        hidden_idx = rng.choice(num_bands, size=n_hidden, replace=False)
        band_mask[hidden_idx] = True
    else:
        raise ValueError(f"unknown geometry: {geometry!r}, expected 'contiguous' or 'scattered'")

    band_mask_t = torch.from_numpy(band_mask).to(pace_cube.device)
    masked_cube = pace_cube.clone()
    masked_cube[:, band_mask_t, :, :] = 0.0
    return masked_cube, band_mask_t


def unpatchify_bands(patchified, num_channels, patch_size, num_patches_h, num_patches_w,
                      patch_size_t=1, num_patches_t=1):
    """
    Generalized version of the real model.unpatchify (confirmed via
    inspect.getsource, Sep 1): identical rearrange pattern, just
    parameterized for num_channels=291 instead of the native 6.

    patchified: (B, num_patches_t*num_patches_h*num_patches_w,
                 patch_size_t*patch_size*patch_size*num_channels)
    returns: (B, num_channels, num_patches_t*patch_size_t, H, W)
    """
    return rearrange(
        patchified,
        "b (t h w) (s p q c) -> b c (t s) (h p) (w q)",
        t=num_patches_t, h=num_patches_h, w=num_patches_w,
        s=patch_size_t, p=patch_size, q=patch_size, c=num_channels,
    )


def masked_reconstruction_loss(prediction, target, band_mask, valid_mask):
    """
    prediction, target: (B, 291, H, W)
    band_mask: (291,) bool -- True = band was hidden from the model. Only
               these bands are scored; visible bands aren't part of the task.
    valid_mask: (B, 291, H, W) bool -- True = real data (not NaN, not
                padding). From pace_tile_loader.load_pace_tile.

    Returns: scalar MSE over (hidden bands) x (valid positions) only.
    """
    band_mask_expanded = band_mask.view(1, -1, 1, 1).expand_as(target)
    score_mask = band_mask_expanded & valid_mask
    n_scored = score_mask.sum().item()
    if n_scored == 0:
        raise RuntimeError(
            "no valid masked-band pixels to score -- band_mask and valid_mask "
            "don't overlap anywhere, check both are being built correctly"
        )
    diff_sq = (prediction - target) ** 2
    return diff_sq[score_mask].mean(), n_scored
