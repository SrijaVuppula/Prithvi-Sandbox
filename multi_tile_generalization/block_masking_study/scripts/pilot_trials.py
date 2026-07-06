"""Pilot: 30 random block positions per chip x ratio. Saves PSNR per trial for power analysis."""
import os, sys, csv, warnings
import numpy as np, torch, rasterio
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "patch_masking_study"))
from patch_masking_study.terratorch_loader import load_prithvi_from_terratorch, run_masked_forward

CHIPS = ["chip_217_425_merged.tif", "chip_268_410_merged.tif", "chip_105_452_merged.tif"]
CHIP_DIR = "multi_tile_generalization/study_chips_500"
OUT = "multi_tile_generalization/block_masking_study/outputs/pilot_trials.csv"
BDIR, CKPT = "~/Prithvi/prithvi_600M", "Prithvi_EO_V2_600M_TL.pt"
RATIOS = [0.2, 0.4, 0.6, 0.8]
N_TRIALS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def load_chip(path):
    with rasterio.open(os.path.expanduser(path)) as src:
        return src.read().astype(np.float32)

def block_noise(n_tokens, ratio, grid, seed):
    rng = np.random.default_rng(seed)
    n_t = grid*grid; off = grid*grid
    bh = max(1, int(round((ratio*n_t)**0.5)))
    bw = max(1, int(round(ratio*n_t/bh)))
    bh, bw = min(bh, grid), min(bw, grid)
    r0 = rng.integers(0, grid-bh+1); c0 = rng.integers(0, grid-bw+1)
    idx = [off+(r0+dr)*grid+(c0+dc) for dr in range(bh) for dc in range(bw)]
    noise = np.zeros(n_tokens); noise[idx] = 1.0
    return noise, idx

def psnr(a, b):
    mse = np.mean((a-b)**2)
    return 99.0 if mse == 0 else 20*np.log10(1.0/np.sqrt(mse))

model, _, mean, std, ps = load_prithvi_from_terratorch(
    backbone_name="prithvi_eo_v2_600", base_dir=os.path.expanduser(BDIR),
    checkpoint_filename=CKPT, num_frames=3, device=DEVICE)
model.eval()
grid = 224//ps; n_tokens = 3*grid*grid
m = np.tile(np.array(mean,dtype=np.float32),3).reshape(-1,1,1)
s = np.tile(np.array(std,dtype=np.float32),3).reshape(-1,1,1)
mc = np.array(mean,dtype=np.float32).reshape(-1,1,1)
sc = np.array(std,dtype=np.float32).reshape(-1,1,1)

rows = []
for chip in CHIPS:
    raw = load_chip(f"{CHIP_DIR}/{chip}")
    norm = (raw - m)/(s+1e-6)
    x = torch.tensor(norm).reshape(3,6,224,224).permute(1,0,2,3).unsqueeze(0).to(DEVICE)
    gt = raw[6:12]  # summer, raw HLS
    for ratio in RATIOS:
        for t in range(N_TRIALS):
            noise, idx = block_noise(n_tokens, ratio, grid, seed=t)
            nt = torch.tensor(noise).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = run_masked_forward(model, x, None, None, ratio, nt)
            rec = out[2].squeeze(0).cpu().numpy()          # (C,T,H,W) norm
            rec_summer = rec[:,1]*sc + mc                   # denorm summer
            # block-region PSNR
            gtn = gt/10000.0; rcn = np.clip(rec_summer/10000.0, 0, 1)
            rows.append(dict(chip=chip, ratio=ratio, trial=t,
                             psnr=round(psnr(gtn, rcn), 4)))
        print(f"{chip} {int(ratio*100)}% done")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("Wrote", OUT)
