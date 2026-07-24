"""
Re-runs the bleeding check and writes a structured CSV to outputs/,
alongside the existing text log.
"""
import sys
import csv
import inspect

sys.path.insert(0, "multi_tile_generalization/block_masking_study/masking")
from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask

RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
IMG_SIZE = 224
NUM_FRAMES = 3
TARGET_FRAME = 1
MASK_THRESHOLD = 0.95

def frame_masked_counts(noise, patch_size):
    noise = noise.squeeze()
    grid = IMG_SIZE // patch_size
    tokens_per_frame = grid * grid
    counts = []
    for f in range(NUM_FRAMES):
        start = f * tokens_per_frame
        end = start + tokens_per_frame
        frame_noise = noise[start:end]
        masked = (frame_noise >= MASK_THRESHOLD).sum().item()
        counts.append((masked, tokens_per_frame))
    return counts

rows = []
for masker_fn, masker_name in [(build_block_noise_mask, "block"), (build_random_noise_mask, "random")]:
    for patch_size in [16, 14]:
        for ratio in RATIOS:
            out = masker_fn(
                mask_ratio=ratio, patch_size=patch_size, img_size=IMG_SIZE,
                num_frames=NUM_FRAMES, frame_idx=TARGET_FRAME, trial_seed=0,
            )
            noise = out[0] if isinstance(out, tuple) else out
            counts = frame_masked_counts(noise, patch_size)
            total_tokens = sum(c[1] for c in counts)
            total_masked = sum(c[0] for c in counts)
            spring_pct, summer_pct, fall_pct = [100.0 * m / t for m, t in counts]
            global_pct = 100.0 * total_masked / total_tokens
            rows.append({
                "masker": masker_name,
                "patch_size": patch_size,
                "nominal_pct": ratio * 100,
                "spring_pct": round(spring_pct, 4),
                "summer_pct": round(summer_pct, 4),
                "fall_pct": round(fall_pct, 4),
                "global_pct": round(global_pct, 4),
                "spring_bleed": spring_pct > 0.01,
                "fall_bleed": fall_pct > 0.01,
            })

out_path = "multi_tile_generalization/block_masking_study/outputs/masking_bleeding_check.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_path}")
print(f"Any bleeding detected: {any(r['spring_bleed'] or r['fall_bleed'] for r in rows)}")
