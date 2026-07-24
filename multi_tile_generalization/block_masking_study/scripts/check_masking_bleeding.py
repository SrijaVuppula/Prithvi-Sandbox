"""
Verifies: for ANY requested mask_ratio, does the masker hide exactly that
proportion of the summer (target) frame, with zero bleeding into spring/fall?
"""
import sys
import inspect

sys.path.insert(0, "multi_tile_generalization/block_masking_study/masking")
from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask

print("build_block_noise_mask signature: ", inspect.signature(build_block_noise_mask))
print("build_random_noise_mask signature:", inspect.signature(build_random_noise_mask))
print()

RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
IMG_SIZE = 224
NUM_FRAMES = 3
TARGET_FRAME = 1  # summer
MASK_THRESHOLD = 0.95  # flagged patches use noise=1.0, background is [0, 0.9)

BACKBONES_BY_PATCH_SIZE = {
    16: ["tiny", "100M", "300M"],
    14: ["600M"],
}

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
        counts.append((masked, tokens_per_frame, frame_noise.min().item(), frame_noise.max().item()))
    return counts

def run_check(masker_fn, masker_name, patch_size):
    grid = IMG_SIZE // patch_size
    backbones = BACKBONES_BY_PATCH_SIZE[patch_size]
    print(f"\n=== {masker_name} | patch_size={patch_size} | backbones: {', '.join(backbones)} "
          f"| grid {grid}x{grid}, {grid*grid}/frame ===")
    print(f"{'nominal':>8} | {'spring%':>8} {'summer%':>8} {'fall%':>8} | {'global%':>8} | flag")
    for ratio in RATIOS:
        out = masker_fn(
            mask_ratio=ratio,
            patch_size=patch_size,
            img_size=IMG_SIZE,
            num_frames=NUM_FRAMES,
            frame_idx=TARGET_FRAME,
            trial_seed=0,
        )
        noise = out[0] if isinstance(out, tuple) else out

        counts = frame_masked_counts(noise, patch_size)
        total_tokens = sum(c[1] for c in counts)
        total_masked = sum(c[0] for c in counts)
        pct = [100.0 * m / t for m, t, _, _ in counts]

        flags = []
        if pct[0] > 0.01:
            flags.append("SPRING BLEED")
        if pct[2] > 0.01:
            flags.append("FALL BLEED")
        target_pct = ratio * 100
        if abs(pct[1] - target_pct) > 2.0:
            flags.append(f"SUMMER MISMATCH (want {target_pct:.1f}%)")

        flag_str = ", ".join(flags) if flags else "OK"
        global_pct = 100.0 * total_masked / total_tokens
        print(f"{ratio*100:7.1f}% | {pct[0]:7.2f}% {pct[1]:7.2f}% {pct[2]:7.2f}% | {global_pct:7.2f}% | {flag_str}")

    out = masker_fn(mask_ratio=0.4, patch_size=patch_size, img_size=IMG_SIZE,
                     num_frames=NUM_FRAMES, frame_idx=TARGET_FRAME, trial_seed=0)
    noise = out[0] if isinstance(out, tuple) else out
    counts = frame_masked_counts(noise, patch_size)
    labels = ["spring", "summer", "fall"]
    print("  raw min/max per frame @ 40% nominal:")
    for lbl, (m, t, mn, mx) in zip(labels, counts):
        print(f"    {lbl}: min={mn:.4f} max={mx:.4f} masked_count={m}/{t}")

for patch_size in [16, 14]:
    run_check(build_block_noise_mask, "BLOCK", patch_size)
    run_check(build_random_noise_mask, "RANDOM", patch_size)
