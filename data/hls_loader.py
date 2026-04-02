import numpy as np
import torch
from pathlib import Path
import sys


def select_files_for_condition(
    all_files: list[str],
    n_frames: int,
    gap_type_cfg: dict,
) -> list[str]:
    """
    Pick the subset of files that realise a given (n_frames, gap_type) condition.

    Args:
        all_files:     ordered list of all available HLS filenames for this tile
        n_frames:      how many frames to use (sequence_length)
        gap_type_cfg:  the gap_type dict from config, e.g.:
                       {"name": "regular", "index_maps": {3:[0,2,5], 4:[0,1,3,5], 6:[...]}}

    Returns:
        List of n_frames filenames in temporal order.
    """
    index_map = gap_type_cfg["index_maps"]
    if n_frames not in index_map:
        raise ValueError(
            f"No index_map entry for n_frames={n_frames} "
            f"in gap_type '{gap_type_cfg['name']}'. "
            f"Available: {list(index_map.keys())}"
        )
    indices = index_map[n_frames]
    if max(indices) >= len(all_files):
        raise ValueError(
            f"Index {max(indices)} out of range for tile with {len(all_files)} files."
        )
    return [all_files[i] for i in indices]


def load_sample_from_inference_module(
    base_dir: Path,
    file_paths: list[str],
    mean: list[float],
    std: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    import importlib.util, sys

    base_dir = Path(base_dir)
    base_str = str(base_dir)

    # Add backbone dir to path so prithvi_mae.py resolves when inference.py loads
    if base_str not in sys.path:
        sys.path.insert(0, base_str)

    # Load Prithvi's inference.py directly by file path
    inference_path = base_dir / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "prithvi_inference", inference_path
    )
    prithvi_inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prithvi_inference)
    load_example = prithvi_inference.load_example

    input_data, temporal_coords, location_coords, meta_data = load_example(
        file_paths=[str(base_dir / "examples" / f) for f in file_paths],
        mean=mean,
        std=std,
    )

    x = torch.tensor(input_data, dtype=torch.float32, device=device)
    temporal_coords = torch.tensor(
        temporal_coords, dtype=torch.float32, device=device
    ).unsqueeze(0)
    location_coords = torch.tensor(
        location_coords[0], dtype=torch.float32, device=device
    ).unsqueeze(0)

    return x, temporal_coords, location_coords, meta_data