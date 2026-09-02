"""
PyTorch Dataset wrapping real PACE tiles + random spectral-band masking for
training. Mask ratio and geometry are randomized PER SAMPLE, mirroring how
MAE pretraining itself uses stochastic masking as the training signal --
this trains one model that generalizes across the full eval grid
(20/40/60/80% x contiguous/scattered) instead of needing 8 separately
trained models, one per condition.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from pace_tile_loader import load_pace_tile, list_pace_tiles
from masked_reconstruction import apply_band_mask


class PaceMaskedDataset(Dataset):
    def __init__(self, tile_paths, target_size=112, mask_ratios=(0.2, 0.4, 0.6, 0.8),
                 geometries=("contiguous", "scattered"), seed=None, deterministic=False):
        """
        deterministic=False (default, for training): masking draws from one
        shared RNG that advances across every __getitem__ call, so the same
        tile gets a fresh random mask each epoch -- ordinary augmentation.

        deterministic=True (for validation): each tile's mask is derived
        from (seed, idx) instead, so the SAME tile always gets the SAME
        mask no matter how many times or in what order it's been called --
        validation compares against a fixed task every epoch, so train/val
        loss trends are actually comparable epoch to epoch.
        """
        self.tile_paths = list(tile_paths)
        self.target_size = target_size
        self.mask_ratios = mask_ratios
        self.geometries = geometries
        self.deterministic = deterministic
        self._base_seed = seed if seed is not None else 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        target_cube, valid_mask = load_pace_tile(self.tile_paths[idx], target_size=self.target_size)
        item_rng = np.random.default_rng((self._base_seed, idx)) if self.deterministic else self.rng
        mask_ratio = float(item_rng.choice(self.mask_ratios))
        geometry = str(item_rng.choice(self.geometries))
        masked_cube, band_mask = apply_band_mask(
            target_cube, mask_ratio=mask_ratio, geometry=geometry, rng=item_rng
        )
        # load_pace_tile adds a batch dim (for standalone use); squeeze it
        # back off here so the DataLoader's default collate re-adds it
        # correctly as the real batch dimension.
        return {
            "masked_cube": masked_cube.squeeze(0),
            "target_cube": target_cube.squeeze(0),
            "valid_mask": valid_mask.squeeze(0),
            "band_mask": band_mask,
            "mask_ratio": mask_ratio,
            "geometry": geometry,
        }


def train_val_split(data_dir=None, val_fraction=0.1, seed=42, exclude_ids_path=None):
    """
    Pragmatic default split of whatever .npy files are currently extracted
    on disk. Fixed seed for reproducibility.

    exclude_ids_path: optional path to a newline-delimited file of tile IDs
    (e.g. hsi_diverse_100.txt) to remove from the pool before splitting --
    keeps the zero-shot pilot's eval set disjoint from fine-tuning data.
    """
    tiles = list_pace_tiles(data_dir)

    if exclude_ids_path is not None:
        with open(exclude_ids_path) as f:
            exclude_ids = {Path(line.strip()).stem for line in f if line.strip()}
        tiles = [t for t in tiles if Path(t).stem not in exclude_ids]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tiles))
    n_val = max(1, int(len(tiles) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train_tiles = [t for i, t in enumerate(tiles) if i not in val_idx]
    val_tiles = [t for i, t in enumerate(tiles) if i in val_idx]
    return train_tiles, val_tiles
