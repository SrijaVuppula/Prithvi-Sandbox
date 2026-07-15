"""
Measure inference time and GPU energy per reconstruction.
Backbone x mask ratio x masking type (random + block).
20 timed forward passes per condition after 3 warmup passes.

Power is sampled continuously in a background thread (not point-sampled per pass)
because the GPU power sensor refreshes slower than a single forward pass.

Masking: only summer patches are occluded; spring/fall stay 100% visible; block
and random mask the SAME number of summer patches at each ratio. The GLOBAL
ratio (n_summer_masked / total_tokens) is passed to the encoder so TerraTorch's
global len_keep keeps everything except the intended summer patches.
"""
import os, sys, csv, time, threading, warnings
import numpy as np
import torch
import rasterio
import pynvml
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "multi_tile_generalization",
                                "block_masking_study", "masking"))
from patch_masking_study.terratorch_loader import (
    load_prithvi_from_terratorch, _encode_with_noise)
from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask

# ── config ────────────────────────────────────────────────────────────────────
CHIP_PATH  = "multi_tile_generalization/study_chips_500/chip_217_425_merged.tif"
OUT_CSV    = "multi_tile_generalization/block_masking_study/outputs/inference_energy.csv"
BACKBONES  = {
    "tiny":  ("~/Prithvi/prithvi_tiny",  "Prithvi_EO_V2_tiny_TL.pt"),
    "100M":  ("~/Prithvi/prithvi_100M",  "Prithvi_EO_V2_100M_TL.pt"),
    "300M":  ("~/Prithvi/prithvi_300M",  "Prithvi_EO_V2_300M_TL.pt"),
    "600M":  ("~/Prithvi/prithvi_600M",  "Prithvi_EO_V2_600M_TL.pt"),
}
MASK_RATIOS   = [0.1, 0.2, 0.4, 0.6, 0.8]
N_WARMUP      = 3
N_TIMED       = 20
POWER_POLL_S  = 0.02   # poll every 20 ms
GPU_WARMUP_S  = 3.0    # dummy passes before EACH condition, to reach steady
                       # clock-boost state before measuring (prevents
                       # order-dependent power readings)
FRAME_IDX     = 1      # summer
N_FRAMES      = 3
IMG           = 224
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── helpers ───────────────────────────────────────────────────────────────────
def load_chip(path):
    with rasterio.open(os.path.expanduser(path)) as src:
        arr = src.read()
    return arr.astype(np.float32)

def get_gpu_power_w(handle):
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except:
        return 0.0

def _stats(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.min()), float(a.max()), float(a.std())

class PowerSampler:
    """Background thread that continuously polls GPU power, decoupled from per-pass timing."""
    def __init__(self, handle, interval_s=POWER_POLL_S):
        self.handle = handle
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
    def _run(self):
        while not self._stop.is_set():
            self.samples.append(get_gpu_power_w(self.handle))
            time.sleep(self.interval_s)
    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    def stop_and_get_mean(self):
        self._stop.set()
        self._thread.join()
        if not self.samples:
            return 0.0
        return float(np.mean(self.samples))

def make_noise_fn(mask_type, ratio, patch_size, n_match):
    """Return f(seed) -> (noise_tensor_1d, global_ratio). Matched summer count."""
    def f(seed):
        if mask_type == "block":
            noise, gr, idx = build_block_noise_mask(
                ratio, patch_size, IMG, N_FRAMES, FRAME_IDX, trial_seed=seed)
        else:
            noise, gr, idx = build_random_noise_mask(
                ratio, patch_size, IMG, N_FRAMES, FRAME_IDX, trial_seed=seed,
                n_summer_masked=n_match)
        return noise, gr, len(idx)
    return f

def measure(model, chip_tensor, noise_fn, handle):
    times_ms, enc_ms, dec_ms = [], [], []

    # --- per-condition GPU warmup: dummy passes for a fixed wall-clock duration
    # so the GPU reaches steady clock-boost state BEFORE this condition is
    # measured, regardless of run order. ---
    warmup_start = time.perf_counter()
    wseed = 0
    while time.perf_counter() - warmup_start < GPU_WARMUP_S:
        noise_w, gr_w, _ = noise_fn(wseed); wseed += 1
        noise_wt = noise_w.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            latent_w, mask_w, ids_restore_w = _encode_with_noise(
                model, chip_tensor, None, None, gr_w, noise_wt)
            _ = model.decoder(latent_w, ids_restore_w, None, None,
                              input_size=chip_tensor.shape)
        torch.cuda.synchronize()

    sampler = PowerSampler(handle)
    sampler.start()
    tokens_kept = None
    for i in range(N_WARMUP + N_TIMED):
        noise, gr, n_masked = noise_fn(1000 + i)
        noise_t = noise.unsqueeze(0).to(DEVICE)
        torch.cuda.synchronize()
        te0 = time.perf_counter()
        with torch.no_grad():
            latent, mask, ids_restore = _encode_with_noise(
                model, chip_tensor, None, None, gr, noise_t)
        torch.cuda.synchronize()
        te1 = time.perf_counter()
        torch.cuda.synchronize()
        td0 = time.perf_counter()
        with torch.no_grad():
            pred = model.decoder(latent, ids_restore, None, None,
                                 input_size=chip_tensor.shape)
        torch.cuda.synchronize()
        td1 = time.perf_counter()
        if tokens_kept is None:
            tokens_kept = int(latent.shape[1])
        if i >= N_WARMUP:
            times_ms.append((td1 - td0 + te1 - te0) * 1000)
            enc_ms.append((te1 - te0) * 1000)
            dec_ms.append((td1 - td0) * 1000)
    avg_power_w = sampler.stop_and_get_mean()
    energy_mj = [avg_power_w * (t_ms / 1000.0) * 1000 for t_ms in times_ms]
    powers_w = [avg_power_w] * len(times_ms)
    return ({"time": _stats(times_ms), "enc": _stats(enc_ms), "dec": _stats(dec_ms),
             "power": _stats(powers_w), "energy": _stats(energy_mj)},
            n_masked, tokens_kept)

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

        grid = IMG // patch_size
        ppf = grid * grid
        n_tokens = N_FRAMES * ppf

        chip_norm = raw.copy()
        m = np.tile(np.array(mean, dtype=np.float32), 3).reshape(-1,1,1)
        s = np.tile(np.array(std,  dtype=np.float32), 3).reshape(-1,1,1)
        chip_norm = (chip_norm - m) / (s + 1e-6)
        chip_tensor = torch.tensor(chip_norm).reshape(3, 6, 224, 224).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

        for ratio in MASK_RATIOS:
            # block first, to fix the summer patch count random must match
            _, _, idx_b = build_block_noise_mask(
                ratio, patch_size, IMG, N_FRAMES, FRAME_IDX, trial_seed=0)
            n_match = len(idx_b)

            for mask_type in ("random", "block"):
                fn = make_noise_fn(mask_type, ratio, patch_size, n_match)
                st, n_masked, tokens_kept = measure(model, chip_tensor, fn, handle)
                row = {"backbone": bname, "mask_ratio": ratio, "mask_type": mask_type,
                       "summer_masked": n_masked, "total_tokens": n_tokens,
                       "tokens_encoded": tokens_kept,
                       "global_ratio": round(n_masked / n_tokens, 4)}
                for metric in ("time", "enc", "dec", "power", "energy"):
                    mn, mi, mx, sd = st[metric]
                    row[f"{metric}_mean"] = round(mn, 4)
                    row[f"{metric}_min"]  = round(mi, 4)
                    row[f"{metric}_max"]  = round(mx, 4)
                    row[f"{metric}_std"]  = round(sd, 4)
                rows.append(row)
                tm, pw, en = st["time"], st["power"], st["energy"]
                print(f"  {bname} | {int(ratio*100)}% {mask_type:6s} | "
                      f"summer {n_masked:3d}/{ppf} | enc {tokens_kept:3d} tok | "
                      f"time {tm[0]:6.2f}±{tm[3]:.2f} ms | "
                      f"pow {pw[0]:5.1f}W | E {en[0]:7.1f}±{en[3]:.1f} mJ")

        del model; torch.cuda.empty_cache()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"\nWrote {OUT_CSV}")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
