"""
train_block_finetune.py
------------------------
Fine-tunes Prithvi EO 2.0 (one backbone at a time) on the SAME block-masking
temporal-gap-filling task used for the zero-shot spatial-track eval, so the
resulting checkpoint is directly comparable to the existing zero-shot numbers.

Only the masking geometry trained on is BLOCK (contiguous) -- mask ratio is
randomized per step over the same 20-80% range used at eval, so one
checkpoint covers the whole eval grid instead of needing 4 separate runs
per backbone.

Train/val chips come from finetune_{train,val}_chips.txt (disjoint from the
500-chip zero-shot eval set in study_chips_500/ -- eval always uses that set
unchanged, so zero-shot vs finetuned stays apples-to-apples).

Usage:
    python train_block_finetune.py --backbone tiny --smoke_test
    python train_block_finetune.py --backbone 100M --epochs 20
    python train_block_finetune.py --backbone 600M --epochs 20 --lr 1e-5

batch_size defaults to 1 -- untested above that here; HLS's 6-band input is
much lighter than the 291-band PACE track that required batch_size=1, but
raise cautiously and watch nvidia-smi rather than assuming it fits.
"""
import argparse, csv, math, shutil, sys, threading, time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

REPO = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "patch_masking_study"))
sys.path.insert(0, str(REPO / "multi_tile_generalization" / "block_masking_study" / "masking"))

from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402
from temporal_gap_masker import build_block_noise_mask       # noqa: E402

STUDY_DIR = REPO / "multi_tile_generalization" / "block_masking_study"
POOL_DIR  = REPO / "multi_tile_generalization" / "training_chips"
CKPT_DIR  = STUDY_DIR / "checkpoints"
LOG_DIR   = STUDY_DIR / "outputs" / "finetune_logs"

BACKBONE_SPECS = {
    "tiny": dict(base_dir=Path.home() / "Prithvi" / "prithvi_tiny", ckpt="Prithvi_EO_V2_tiny_TL.pt"),
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", ckpt="Prithvi_EO_V2_100M_TL.pt"),
    "300M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_300M", ckpt="Prithvi_EO_V2_300M_TL.pt"),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", ckpt="Prithvi_EO_V2_600M_TL.pt"),
}

IMG, BANDS, T, FRAME = 224, 6, 3, 1
RATIO_LO, RATIO_HI = 0.20, 0.80


class ChipDataset(Dataset):
    """Raw (T, BANDS, IMG, IMG) float32 tensors, unnormalized -- normalization
    is backbone-specific (different mean/std per config.json) and applied in
    the training loop instead."""
    def __init__(self, chip_names, pool_dir):
        self.paths = [pool_dir / name for name in chip_names]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} chips missing, e.g. {missing[0]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with rasterio.open(self.paths[idx]) as src:
            data = src.read()
        raw = data.reshape(T, BANDS, IMG, IMG).astype(np.float32)
        return torch.from_numpy(raw)


def normalize(raw, mean, std):
    """raw: (B, T, BANDS, IMG, IMG) -> model input (B, BANDS, T, IMG, IMG).
    Same convention as run_paired_block_random.py / measure_inference_energy.py:
    raw HLS scale, NOT divided by 10000 -- mean/std in config.json are
    calibrated to this raw scale."""
    m = torch.tensor(mean[:BANDS], dtype=torch.float32).reshape(1, 1, -1, 1, 1)
    s = torch.tensor(std[:BANDS],  dtype=torch.float32).reshape(1, 1, -1, 1, 1)
    x = (raw - m) / s
    return x.permute(0, 2, 1, 3, 4)


def encode_with_noise_trainable(model, x, mask_ratio, noise):
    """Copy of terratorch_loader._encode_with_noise WITHOUT @torch.no_grad()
    -- reusing the eval version directly would silently zero every gradient
    through the encoder."""
    enc = model.encoder
    sample_shape = x.shape[-3:]
    x_enc = enc.patch_embed(x)
    pos_embed = enc.interpolate_pos_encoding(sample_shape)
    x_enc = x_enc + pos_embed[:, 1:, :]
    x_enc, mask, ids_restore = enc.random_masking(x_enc, mask_ratio, noise=noise)
    cls_token  = enc.cls_token + pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x_enc.shape[0], -1, -1)
    x_enc      = torch.cat((cls_tokens, x_enc), dim=1)
    for block in enc.blocks:
        x_enc = block(x_enc)
    x_enc = enc.norm(x_enc)
    return x_enc, mask, ids_restore


def forward_loss_trainable(model, x, mask_ratio, noise):
    latent, mask, ids_restore = encode_with_noise_trainable(model, x, mask_ratio, noise)
    pred = model.decoder(latent, ids_restore, None, None, input_size=x.shape)
    return model.forward_loss(x, pred, mask)


def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class PowerSampler:
    """Background NVML poll, decoupled from step timing -- same pattern as
    measure_inference_energy.py, reused here to log total fine-tuning energy
    (the one-time adaptation cost)."""
    def __init__(self, handle, interval_s=0.02):
        self.handle, self.interval_s = handle, interval_s
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_get_mean(self):
        self._stop.set(); self._thread.join()
        return float(np.mean(self.samples)) if self.samples else 0.0


def evaluate(model, val_raw, mean, std, patch_size, device, ratios=(0.2, 0.4, 0.6, 0.8)):
    model.eval()
    losses = []
    with torch.no_grad():
        for raw in val_raw:
            x = normalize(raw.unsqueeze(0), mean, std).to(device)
            for r in ratios:
                noise, gr, _ = build_block_noise_mask(r, patch_size, IMG, T, FRAME,
                                                       trial_seed=hash((r,)) % (2**31))
                loss = forward_loss_trainable(model, x, gr, noise.unsqueeze(0).to(device))
                losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONE_SPECS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3, help="stop after N epochs with no val_loss improvement")
    ap.add_argument("--sched_epochs", type=int, default=None,
                    help="epochs the cosine schedule anneals over, independent of --epochs/early stop; "
                         "defaults to --epochs if not set. Set explicitly so all backbones anneal over "
                         "the same horizon regardless of when patience fires.")
    ap.add_argument("--smoke_test", action="store_true",
                     help="1 epoch, 20 train chips, 5 val chips -- sanity check before a real run")
    ap.add_argument("--seed", type=int, required=True,
                    help="training seed: data order + mask/ratio stream")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_names = (STUDY_DIR / "config" / "finetune_train_chips.txt").read_text().split()
    val_names   = (STUDY_DIR / "config" / "finetune_val_chips.txt").read_text().split()
    if args.smoke_test:
        train_names, val_names, args.epochs = train_names[:20], val_names[:5], 1
        print("SMOKE TEST: 20 train chips, 5 val chips, 1 epoch")

    train_ds = ChipDataset(train_names, POOL_DIR)
    val_ds   = ChipDataset(val_names, POOL_DIR)
    print(f"Train chips: {len(train_ds)}  Val chips: {len(val_ds)}")
    val_raw = [val_ds[i] for i in range(len(val_ds))]

    spec = BACKBONE_SPECS[args.backbone]
    model, _, mean, std, patch_size = load_prithvi_from_terratorch(
        backbone_name=f"prithvi_eo_v2_{args.backbone.lower().replace('m','')}",
        base_dir=spec["base_dir"], checkpoint_filename=spec["ckpt"],
        num_frames=T, device=device)
    model.train()

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched_epochs = args.sched_epochs if args.sched_epochs is not None else args.epochs
    total_steps = max(1, sched_epochs * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, args.warmup_steps, total_steps))

    ckpt_dir = CKPT_DIR / "seeds" / f"seed{args.seed}" / args.backbone
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_src = spec["base_dir"] / "config.json"
    if cfg_src.exists() and not (ckpt_dir / "config.json").exists():
        shutil.copy(cfg_src, ckpt_dir / "config.json")  # load_prithvi_from_terratorch needs
                                                          # config.json next to the checkpoint

    log_dir = LOG_DIR / "seeds" / f"seed{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / f"{args.backbone}_train_log.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "step", "train_loss", "val_loss", "lr", "elapsed_s"])

    sampler = None
    if _HAS_NVML:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        sampler = PowerSampler(handle)
        sampler.start()

    best_val = float("inf")
    no_improve = 0
    t_start = time.perf_counter()
    step = 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        epoch_losses = []
        for raw in loader:
            ratio = float(rng.uniform(RATIO_LO, RATIO_HI))
            x = normalize(raw, mean, std).to(device)
            noises, grs = [], []
            for _ in range(x.shape[0]):
                noise, gr, _ = build_block_noise_mask(
                    ratio, patch_size, IMG, T, FRAME, trial_seed=int(rng.integers(0, 2**31)))
                noises.append(noise); grs.append(gr)
            noise_batch = torch.stack(noises).to(device)
            global_ratio = float(np.mean(grs))  # identical across the batch for a fixed nominal ratio

            loss = forward_loss_trainable(model, x, global_ratio, noise_batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            epoch_losses.append(loss.item())
            step += 1

            if step % 50 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  [{args.backbone}] epoch {epoch} step {step}/{total_steps} "
                      f"loss {float(loss):.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"elapsed {elapsed/60:.1f}m")

        train_loss = float(np.mean(epoch_losses))
        val_loss = None
        if (epoch + 1) % args.val_every == 0:
            val_loss = evaluate(model, val_raw, mean, std, patch_size, device)
            print(f"  [{args.backbone}] epoch {epoch} DONE train_loss {train_loss:.4f} val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                print(f"    -> new best, saved {ckpt_dir/'best.pt'}")
            else:
                no_improve += 1

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        elapsed = time.perf_counter() - t_start
        log_w.writerow([epoch, step, train_loss, val_loss, sched.get_last_lr()[0], round(elapsed, 1)])
        log_f.flush()

        if val_loss is not None and no_improve >= args.patience:
            print(f"  [{args.backbone}] early stop: no val_loss improvement for "
                  f"{args.patience} epochs (best={best_val:.4f})")
            break

    log_f.close()

    if sampler is not None:
        avg_power = sampler.stop_and_get_mean()
        total_elapsed = time.perf_counter() - t_start
        energy_kj = avg_power * total_elapsed / 1000.0
        with open(log_dir / f"{args.backbone}_finetune_cost.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["backbone", "epochs", "steps", "elapsed_s", "avg_power_w", "energy_kj"])
            w.writerow([args.backbone, args.epochs, step, round(total_elapsed, 1),
                        round(avg_power, 2), round(energy_kj, 2)])
        print(f"Fine-tuning cost: {total_elapsed/3600:.2f}h, avg {avg_power:.1f}W, ~{energy_kj:.1f} kJ")
        pynvml.nvmlShutdown()

    print(f"Done. Best val_loss={best_val:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
