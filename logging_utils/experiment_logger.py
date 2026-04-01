import csv
import json
from datetime import datetime
from pathlib import Path

FIELDNAMES = [
    "run_id", "timestamp",
    # four dimensions
    "backbone", "mask_position", "n_frames", "gap_type",
    # data
    "tile_id", "masked_frame_idx", "gap_days",
    # metrics
    "mae", "psnr", "ssim", "loss", "mask_ratio",
    # bookkeeping
    "checkpoint",
]


class ExperimentLogger:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.runs_dir   = self.output_dir / "runs"
        self.csv_path   = self.output_dir / "results.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def log(
        self,
        backbone: str,
        mask_position: str,
        n_frames: int,
        gap_type: str,
        tile_id: str,
        masked_frame_idx: int,
        gap_days: list[int],
        metrics: dict,           # keys: mae, psnr, ssim
        loss: float,
        mask_ratio: float,
        checkpoint: str,
        extra: dict | None = None,
    ) -> str:
        ts     = datetime.now()
        run_id = (
            ts.strftime("run_%Y%m%d_%H%M%S")
            + f"_{backbone}_{mask_position}_T{n_frames}_{gap_type}"
        )

        row = {
            "run_id":           run_id,
            "timestamp":        ts.isoformat(),
            "backbone":         backbone,
            "mask_position":    mask_position,
            "n_frames":         n_frames,
            "gap_type":         gap_type,
            "tile_id":          tile_id,
            "masked_frame_idx": masked_frame_idx,
            "gap_days":         json.dumps(gap_days),   # e.g. "[80, 95, 65]"
            "mae":              metrics["mae"],
            "psnr":             metrics["psnr"],
            "ssim":             metrics["ssim"],
            "loss":             round(loss, 6),
            "mask_ratio":       round(mask_ratio, 4),
            "checkpoint":       checkpoint,
        }

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        json_path = self.runs_dir / f"{run_id}.json"
        with open(json_path, "w") as f:
            json.dump({**row, **(extra or {})}, f, indent=2)

        print(
            f"[{backbone:>5}] pos={mask_position:<9} T={n_frames} "
            f"gap={gap_type:<10} "
            f"MAE={metrics['mae']:.4f} PSNR={metrics['psnr']:6.2f} "
            f"SSIM={metrics['ssim']:.4f}"
        )
        return run_id