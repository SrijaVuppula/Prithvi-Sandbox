"""
block_masker.py
---------------
Generates a contiguous rectangular block mask in PATCH space to simulate
realistic cloud cover. Unlike random scattered masking, real clouds occupy
a solid connected region of the sky.

Patch grid:
  - Image: 224 x 224 pixels
  - Patch size: 16x16 (tiny / 100M / 300M) OR 14x14 (600M)
  - Grid:  224/16 = 14 x 14 = 196 patches per frame  (16px models)
           224/14 = 16 x 16 = 256 patches per frame  (600M)

The model's noise mask is 1-D over ALL frames concatenated:
  T=3, 16px model -> total tokens = 3 * 196 = 588
  T=3, 14px model -> total tokens = 3 * 256 = 768

We mask ONLY the middle frame (frame_idx=1). The block is a random
axis-aligned rectangle of patches inside that frame's token slice.
"""

import math
import torch


def build_block_noise_mask(
    mask_ratio: float,
    patch_size: int = 16,
    img_size: int = 224,
    num_frames: int = 3,
    frame_idx: int = 1,
    trial_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = torch.Generator()
    if trial_seed is not None:
        rng.manual_seed(trial_seed)

    grid_h, grid_w = _grid_size(img_size, patch_size)
    patches_per_frame = grid_h * grid_w
    total_tokens = num_frames * patches_per_frame

    n_mask_target = int(round(mask_ratio * patches_per_frame))
    n_mask_target = max(1, min(n_mask_target, patches_per_frame - 1))

    block_h, block_w = _find_block_dims(n_mask_target, grid_h, grid_w)

    max_r = grid_h - block_h
    max_c = grid_w - block_w
    r0 = int(torch.randint(0, max(max_r, 1) + 1, (1,), generator=rng).item())
    c0 = int(torch.randint(0, max(max_c, 1) + 1, (1,), generator=rng).item())

    frame_mask_grid = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    frame_mask_grid[r0:r0 + block_h, c0:c0 + block_w] = True

    masked_local = frame_mask_grid.flatten().nonzero(as_tuple=True)[0]
    frame_offset = frame_idx * patches_per_frame
    masked_global = masked_local + frame_offset

    noise = torch.rand(total_tokens, generator=rng)
    noise[masked_global] = 2.0

    ids_restore = torch.argsort(torch.argsort(noise))
    n_keep = total_tokens - len(masked_global)
    ids_keep = torch.argsort(noise)[:n_keep]

    return noise, ids_keep, ids_restore


def block_mask_to_pixel_map(
    noise: torch.Tensor,
    patch_size: int = 16,
    img_size: int = 224,
    num_frames: int = 3,
    frame_idx: int = 1,
) -> torch.Tensor:
    grid_h, grid_w = _grid_size(img_size, patch_size)
    patches_per_frame = grid_h * grid_w
    frame_offset = frame_idx * patches_per_frame

    frame_noise = noise[frame_offset: frame_offset + patches_per_frame]
    masked_patches = (frame_noise >= 2.0).reshape(grid_h, grid_w)

    pixel_mask = masked_patches.repeat_interleave(patch_size, dim=0)\
                               .repeat_interleave(patch_size, dim=1)
    return pixel_mask[:img_size, :img_size]


def _grid_size(img_size: int, patch_size: int) -> tuple[int, int]:
    g = img_size // patch_size
    return g, g


def _find_block_dims(n_target: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    best_h, best_w = 1, min(n_target, grid_w)
    best_diff = abs(best_h * best_w - n_target)
    for h in range(1, grid_h + 1):
        w = round(n_target / h)
        w = max(1, min(w, grid_w))
        diff = abs(h * w - n_target)
        if diff < best_diff or (diff == best_diff and abs(h - w) < abs(best_h - best_w)):
            best_h, best_w = h, w
            best_diff = diff
    return best_h, best_w
