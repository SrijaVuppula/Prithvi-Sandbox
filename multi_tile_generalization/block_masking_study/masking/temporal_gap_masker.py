"""
temporal_gap_masker.py
----------------------------
Corrected masking for the block-vs-random temporal-gap-filling comparison.

Fixes two issues found on 2026-07-12:
  1. TerraTorch's random_masking computes len_keep GLOBALLY over all frames,
     so the old build_block_noise_mask let spring/fall visibility degrade with
     ratio (they were NEVER full context, contrary to the session docs).
  2. The old block masker used mask_ratio = fraction-of-SUMMER while the old
     random masker (build_noise_for_mask_ratio) used fraction-of-ALL-tokens,
     so at the same nominal ratio they masked different amounts of summer.

Corrected convention (identical for both maskers):
  * spring + fall + summer-KEEP patches -> noise in [0, 0.9)  (guaranteed kept)
  * summer patches to mask              -> noise = 1.0         (guaranteed masked)
  * CALLER passes mask_ratio = n_summer_masked / total_tokens (the GLOBAL ratio)
    so TerraTorch's len_keep = total - n_summer_masked keeps everything except
    the intended summer patches.

Result: spring & fall 100% visible; summer masked at exactly the nominal ratio;
block and random mask the SAME NUMBER of summer patches -> the ONLY difference
is geometry (contiguous block vs scattered). That is the comparison the paper
intends to make.
"""

import torch


def _grid(img_size, patch_size):
    g = img_size // patch_size
    return g, g


def _find_block_dims(n_target, grid_h, grid_w):
    best_h, best_w = 1, min(n_target, grid_w)
    best_diff = abs(best_h * best_w - n_target)
    for h in range(1, grid_h + 1):
        w = max(1, min(round(n_target / h), grid_w))
        diff = abs(h * w - n_target)
        if diff < best_diff or (diff == best_diff and abs(h - w) < abs(best_h - best_w)):
            best_h, best_w, best_diff = h, w, diff
    return best_h, best_w


def _base_noise(total_tokens, rng):
    # everything starts guaranteed-KEEP: strictly below the 1.0 mask value
    return torch.rand(total_tokens, generator=rng) * 0.9


def _summer_slice(patch_size, img_size, num_frames, frame_idx):
    grid_h, grid_w = _grid(img_size, patch_size)
    ppf = grid_h * grid_w
    total = num_frames * ppf
    offset = frame_idx * ppf
    return grid_h, grid_w, ppf, total, offset


def build_block_noise_mask(mask_ratio, patch_size=16, img_size=224,
                           num_frames=3, frame_idx=1, trial_seed=None):
    """Contiguous rectangular block over the summer frame only.

    Returns (noise, global_mask_ratio, masked_global_idx).
    Pass `global_mask_ratio` (NOT mask_ratio) to run_masked_forward.
    """
    rng = torch.Generator()
    if trial_seed is not None:
        rng.manual_seed(trial_seed)

    grid_h, grid_w, ppf, total, offset = _summer_slice(
        patch_size, img_size, num_frames, frame_idx)

    n_target = max(1, min(int(round(mask_ratio * ppf)), ppf - 1))
    bh, bw = _find_block_dims(n_target, grid_h, grid_w)
    r0 = int(torch.randint(0, grid_h - bh + 1, (1,), generator=rng).item())
    c0 = int(torch.randint(0, grid_w - bw + 1, (1,), generator=rng).item())

    grid = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    grid[r0:r0 + bh, c0:c0 + bw] = True
    masked_local = grid.flatten().nonzero(as_tuple=True)[0]
    masked_global = masked_local + offset

    noise = _base_noise(total, rng)
    noise[masked_global] = 1.0

    global_ratio = len(masked_global) / total
    return noise, global_ratio, masked_global


def build_random_noise_mask(mask_ratio, patch_size=16, img_size=224,
                            num_frames=3, frame_idx=1, trial_seed=None,
                            n_summer_masked=None):
    """Scattered patches over the summer frame only.

    If n_summer_masked is given, masks EXACTLY that many summer patches so the
    count is matched to a paired block trial. Otherwise uses round(ratio*ppf).
    Returns (noise, global_mask_ratio, masked_global_idx).
    """
    rng = torch.Generator()
    if trial_seed is not None:
        rng.manual_seed(trial_seed)

    grid_h, grid_w, ppf, total, offset = _summer_slice(
        patch_size, img_size, num_frames, frame_idx)

    n = n_summer_masked if n_summer_masked is not None \
        else max(1, min(int(round(mask_ratio * ppf)), ppf - 1))

    perm = torch.randperm(ppf, generator=rng)[:n]
    masked_global = perm + offset

    noise = _base_noise(total, rng)
    noise[masked_global] = 1.0

    global_ratio = len(masked_global) / total
    return noise, global_ratio, masked_global


def pixel_map(masked_global, patch_size=16, img_size=224,
              num_frames=3, frame_idx=1):
    """Boolean (img_size, img_size) mask of the summer masked region."""
    grid_h, grid_w = _grid(img_size, patch_size)
    ppf = grid_h * grid_w
    offset = frame_idx * ppf
    local = (masked_global - offset)
    local = local[(local >= 0) & (local < ppf)]
    flat = torch.zeros(ppf, dtype=torch.bool)
    flat[local] = True
    grid = flat.reshape(grid_h, grid_w)
    return grid.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)[:img_size, :img_size]
