"""
Measure inference time and GPU energy per reconstruction.
Backbone x mask ratio x masking type (random + block).
20 timed forward passes per condition after 3 warmup passes.
"""
import os, sys, csv, time, warnings
import numpy as np
import torch
import rasterio
import pynvml
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
from patch_masking_study.terratorch_loader import load_prithvi_from_terratorch, run_masked_forward

# ── config ────────────────────────────────────────────────────────────────────
CHIP_PATH  = "multi_tile_generalization/study_chips_500/chip_217_425_merged.tif"
OUT_CSV    = "multi_tile_generalization/block_masking_study/outputs/inference_energy.csv"
BACKBONES  = {
    "tiny":  ("~/Prithvi/prithvi_tiny",  "Prithvi_EO_V2_tiny_TL.pt"),
    "100M":  ("~/Prithvi/prithvi_100M",  "Prithvi_EO_V2_100M_TL.pt"),
    "300M":  ("~/Prithvi/prithvi_300M",  "Prithvi_EO_V2_300M_TL.pt"),
    "600M":  ("~/Prithvi/prithvi_600M",  "Prithvi_EO_V2_600M_TL.pt"),
}
MASK_RATIOS   = [0.2, 0.4, 0.6, 0.8]
N_WARMUP      = 3
N_TIMED       = 20
SUMMER_OFFSET = 6
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── helpers ───────────────────────────────────────────────────────────────────
def load_chip(path):
    with rasterio.open(os.path.expanduser(path)) as src:
        arr = src.read()
    return arr.astype(np.float32)   # (18, 224, 224) raw HLS

def build_random_noise(n_tokens, ratio, patch_size, grid):
    target_frame_tokens = list(range(grid * grid, 2 * grid * grid))
    n_mask = int(ratio * len(target_frame_tokens))
    noise = np.random.rand(n_tokens)
    noise[target_frame_tokens] = 0.9 + 0.1 * np.random.rand(len(target_frame_tokens))
    idx = sorted(target_frame_tokens, key=lambda i: noise[i], reverse=True)[:n_mask]
    noise[:] = 0.0
    noise[idx] = 1.0
    return noise

def build_block_noise(n_tokens, ratio, patch_size, grid):
    n_target = grid * grid
    offset   = grid * grid   # middle frame starts here
    target_h = target_w = grid
    bh = max(1, int(round((ratio * n_target) ** 0.5)))
    bw = max(1, int(round(ratio * n_target / bh)))
    bh = min(bh, target_h); bw = min(bw, target_w)
    r0 = np.random.randint(0, target_h - bh + 1)
    c0 = np.random.randint(0, target_w - bw + 1)
    block_idx = [offset + (r0 + dr) * target_w + (c0 + dc)
                 for dr in range(bh) for dc in range(bw)]
    noise = np.zeros(n_tokens)
    noise[block_idx] = 1.0
    return noise

def get_gpu_power_mw(handle):
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle)   # milliwatts
    except:
        return 0

def _stats(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.min()), float(a.max()), float(a.std())

def measure(model, chip_tensor, mean, std, noise_fn,
            n_tokens, ratio, patch_size, grid, handle):
    times_ms, enc_ms, dec_ms, powers_w, energy_mj = [], [], [], [], []
    for _ in range(N_WARMUP + N_TIMED):
        noise = noise_fn(n_tokens, ratio, patch_size, grid)
        noise_t = torch.tensor(noise).unsqueeze(0).to(DEVICE)
        # --- encoder ---
        torch.cuda.synchronize()
        te0 = time.perf_counter()
        with torch.no_grad():
            from patch_masking_study.terratorch_loader import _encode_with_noise
            latent, mask, ids_restore = _encode_with_noise(
                model, chip_tensor, None, None, ratio, noise_t)
        torch.cuda.synchronize()
        te1 = time.perf_counter()
        # --- decoder ---
        p0 = get_gpu_power_mw(handle)
        torch.cuda.synchronize()
        td0 = time.perf_counter()
        with torch.no_grad():
            pred = model.decoder(latent, ids_restore, None, None,
                                 input_size=chip_tensor.shape)
        torch.cuda.synchronize()
        td1 = time.perf_counter()
        p1 = get_gpu_power_mw(handle)
        if _ >= N_WARMUP:
            t_ms = (td1 - td0 + te1 - te0) * 1000
            p_w  = ((p0 + p1) / 2) / 1000.0
            times_ms.append(t_ms)
            enc_ms.append((te1 - te0) * 1000)
            dec_ms.append((td1 - td0) * 1000)
            powers_w.append(p_w)
            energy_mj.append(p_w * (t_ms / 1000.0) * 1000)   # W·s·1000 = mJ
    return {"time": _stats(times_ms), "enc": _stats(enc_ms), "dec": _stats(dec_ms),
            "power": _stats(powers_w), "energy": _stats(energy_mj)}

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"GPU: {gpu_name}  |  Device: {DEVICE}")

    raw = load_chip(CHIP_PATH)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    for bname, (bdir, ckpt) in BACKBONES.items():
        print(f"\nLoading {bname}...")
        model, _, mean, std, patch_size = load_prithvi_from_terratorch(
            backbone_name=f"prithvi_eo_v2_{bname.lower().replace('m','')}",
            base_dir=os.path.expanduser(bdir),
            checkpoint_filename=ckpt,
            num_frames=3, device=DEVICE
        )
        model.eval()

        # build chip tensor (normalise to model scale)
        summer = raw[SUMMER_OFFSET:SUMMER_OFFSET+6]
        grid   = 224 // patch_size
        n_tokens = 3 * grid * grid

        # normalise chip using model mean/std
        chip_norm = raw.copy()
        m = np.tile(np.array(mean, dtype=np.float32), 3).reshape(-1,1,1)
        s = np.tile(np.array(std,  dtype=np.float32), 3).reshape(-1,1,1)
        chip_norm = (chip_norm - m) / (s + 1e-6)
        chip_tensor = torch.tensor(chip_norm).reshape(3, 6, 224, 224).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

        for ratio in MASK_RATIOS:
            for mask_type, noise_fn in [("random", build_random_noise),
                                         ("block",  build_block_noise)]:
                st = measure(model, chip_tensor, mean, std,
                             noise_fn, n_tokens, ratio,
                             patch_size, grid, handle)
                row = {"backbone": bname, "mask_ratio": ratio, "mask_type": mask_type}
                for metric in ("time", "enc", "dec", "power", "energy"):
                    mn, mi, mx, sd = st[metric]
                    row[f"{metric}_mean"] = round(mn, 4)
                    row[f"{metric}_min"]  = round(mi, 4)
                    row[f"{metric}_max"]  = round(mx, 4)
                    row[f"{metric}_std"]  = round(sd, 4)
                rows.append(row)
                tm, pw, en = st["time"], st["power"], st["energy"]
                print(f"  {bname} | {int(ratio*100)}% {mask_type:6s} | "
                      f"time {tm[0]:6.2f}±{tm[3]:.2f} ms | "
                      f"pow {pw[0]:5.1f}±{pw[3]:.1f} W | "
                      f"E {en[0]:7.1f} [{en[1]:.0f},{en[2]:.0f}] mJ")

        del model; torch.cuda.empty_cache()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"\nWrote {OUT_CSV}")
    print(f"\n{'Backbone':<8}{'Ratio':>6}{'Type':>8}"
          f"{'E mean':>9}{'E min':>9}{'E max':>9}{'E std':>8}"
          f"{'P mean':>9}{'P std':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r['backbone']:<8}{int(r['mask_ratio']*100):>5}%{r['mask_type']:>8}"
              f"{r['energy_mean']:>9.1f}{r['energy_min']:>9.1f}"
              f"{r['energy_max']:>9.1f}{r['energy_std']:>8.2f}"
              f"{r['power_mean']:>9.1f}{r['power_std']:>8.2f}")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
