# Prithvi-Sandbox

Experimental code for evaluating the **Prithvi EO 2.0** satellite-image foundation model (IBM/NASA) on **temporal gap filling** — reconstructing a missing time step in a multi-temporal satellite image sequence to simulate cloud removal.

This repo contains the evaluation pipeline, masking utilities, and analysis scripts for a systematic study of the model under different masking conditions and backbone sizes.

---

## The problem

Clouds constantly block satellite views, and an entire observation can be lost when a satellite passes over a region under cloud cover. Given surrounding cloud-free acquisitions (e.g. spring and fall), can a foundation model pretrained with masked autoencoding reconstruct the occluded frame (summer) — without any additional training?

Prithvi was pretrained to reconstruct scattered patches. Reconstructing a whole missing frame, or a contiguous cloud-shaped block, is a different task. This repo provides the tooling to characterize that.

---

## Repository structure
Prithvi-Sandbox/
├── baseline_study/                  # Zero-shot temporal gap filling across conditions (single tile)
│   ├── data/                        #   hls_loader.py
│   ├── inference/                   #   runner.py
│   ├── masking/                     #   temporal_masker.py (masks a full frame)
│   ├── metrics/                     #   evaluate.py (MAE, PSNR, SSIM)
│   ├── logging_utils/               #   experiment_logger.py
│   ├── scripts/                     #   run_baselines.py, plot_results.py
│   └── outputs/
│
├── patch_masking_study/             # Scattered patch masking across mask ratios (single tile)
│   ├── config/                      #   patch_experiment_config.yaml
│   ├── masking/                     #   patch_masker.py
│   ├── metrics/                     #   evaluate_masked.py (global + masked-region-only)
│   ├── scripts/                     #   run_patch_experiment.py, save_reconstructions.py, plot_degradation_curve.py
│   ├── terratorch_loader.py         #   TerraTorch model loader
│   └── outputs/
│
├── multi_tile_generalization/       # Multi-chip generalization across backbones and ratios
│   ├── config/                      #   generalization_config.yaml
│   ├── data/                        #   hf_chip_loader.py
│   ├── metrics/                     #   evaluate_masked.py
│   ├── scripts/                     #   run_generalization.py, aggregate_results.py,
│   │                                #   plot_generalization.py, analyze_degradation.py
│   ├── training_chips/              #   dataset chips (gitignored)
│   ├── study_chips_500/             #   sampled study set (gitignored)
│   └── block_masking_study/         # Contiguous block vs. scattered random masking, paired comparison
│       ├── config/                  #   block_masking_config.yaml
│       ├── masking/                 #   block_masker.py (legacy, superseded — see note below),
│       │                            #   temporal_gap_masker.py (corrected, canonical — use this)
│       ├── metrics/                 #   evaluate_block_masked.py
│       ├── scripts/                 #   run_paired_block_random.py (main entry point), build_sanity_check.py,
│       │                            #   plot_block_degradation.py, plot_delta_distributions_600M.py,
│       │                            #   plot_trimmed_bar_600M.py, plot_random_vs_block.py, plot_difficulty_gap.py,
│       │                            #   error-map visualization, complexity scoring, inference-energy measurement,
│       │                            #   power-analysis / trial-count scripts
│       └── outputs/                 #   results_{backbone}.csv (canonical), archived_duplicate_runs/, old_confounded/
│
├── exploratory/                     # Self-initiated follow-up experiments
│   ├── 01_complexity_stratified/
│   ├── 02_distance_to_context/
│   ├── 03_fragmentation_spectrum/   #   (planned)
│   └── 04_real_cloud_masks/         #   (planned)
│
└── docs_and_reference/              # Original Prithvi inference scripts and reference material
---

## Data

**IBM-NASA Multi-Temporal Crop Classification Dataset** ([HuggingFace](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)) — chips across the continental US, 224×224 px at 30 m, 6 HLS bands, 3 time steps (spring/summer/fall). The baseline study uses HLS acquisitions over a single site.

A chip loads as `src.read().reshape(3, 6, 224, 224)` (T×C×H×W, raw HLS reflectance) and is reshaped to `(1, 6, 3, 224, 224)` for the model. Normalize with per-model mean/std from each backbone's `config.json`. Chip data is gitignored.

---

## Models

Four Prithvi EO 2.0 backbones (tiny, 100M, 300M, 600M), checkpoints named `Prithvi_EO_V2_*_TL.pt`. Note that the 600M backbone uses 14×14 patches (256 tokens/frame) while the others use 16×16 (196 tokens/frame). Model weights are stored outside the repo.

---

## Setup

```bash
# GPU server (NVIDIA RTX 3090, 24 GB)
python3.11 -m venv ~/.venv
source ~/.venv/bin/activate
pip install torch terratorch==1.2.6 scikit-image rasterio einops timm PyYAML matplotlib numpy pynvml
```

---

## Metrics

- **PSNR (dB)** — primary metric, higher is better.
- **MAE** — lower is better.
- **SSIM** — higher is better; structure/texture preservation.

Masked-region-only metrics measure what the model reconstructed, not what it copied from visible context.

---

## Masking convention (block masking study)

Both maskers in `temporal_gap_masker.py` mask **only the summer (middle) frame**; spring and fall are guaranteed 100% visible at every ratio. Block and random maskers mask the exact same number of summer patches (matched via `n_summer_masked`), so at a given ratio the only difference between the two conditions is masking *geometry* (contiguous block vs. scattered patches), not how much context is available. This corrects an earlier masking convention where context-frame visibility silently degraded with ratio; `block_masker.py` holds the earlier (superseded) version, kept for reference only.

Because TerraTorch's `random_masking()` computes its keep-budget globally across all three frames, callers must pass the *returned* `global_mask_ratio` into `run_masked_forward` — not the nominal `mask_ratio` argument.

---

## Running an experiment

Always run from the repo root:

```bash
cd Prithvi-Sandbox

# Long jobs: use tmux to survive disconnects
tmux new -s block_masking
python multi_tile_generalization/block_masking_study/scripts/run_paired_block_random.py
# rejoin later: tmux attach -t block_masking

python multi_tile_generalization/block_masking_study/scripts/build_sanity_check.py
python multi_tile_generalization/block_masking_study/scripts/plot_block_degradation.py
```

Key function signatures:

- `build_block_noise_mask(mask_ratio, patch_size=16, img_size=224, num_frames=3, frame_idx=1, trial_seed=None)` → `(noise, global_mask_ratio, masked_global_idx)`
- `build_random_noise_mask(mask_ratio, patch_size=16, img_size=224, num_frames=3, frame_idx=1, trial_seed=None, n_summer_masked=None)` → `(noise, global_mask_ratio, masked_global_idx)`; pass `n_summer_masked` to match a paired block trial's masked count exactly.
- `run_masked_forward(model, x, temporal_coords, location_coords, mask_ratio, noise)` → 5-tuple `(loss, pred_img, rec_img, mask_img, x_cpu)`; use index `[2]` for the composite reconstruction (already on CPU). Pass the masker's returned `global_mask_ratio` here, not the nominal ratio.

---

## Status

Code and experimental pipeline are complete for the studies above. Results and analysis are being written up for publication and will be linked here once released.

---

## Links

- Prithvi models: https://huggingface.co/ibm-nasa-geospatial
- Crop dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification
- NASA HLS: https://hls.gsfc.nasa.gov
- TerraTorch: https://github.com/IBM/terratorch
