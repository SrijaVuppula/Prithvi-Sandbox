#!/usr/bin/env bash
#
# migrate_to_archive.sh
# ---------------------
# Archives the OLD confounded block-masking outputs into outputs/old_confounded/,
# then promotes the corrected results_fixed_{bb}.csv -> results_{bb}.csv.
#
# NON-DESTRUCTIVE: everything is MOVED into old_confounded/, nothing is deleted.
# DRY-RUN by default. Review the printed plan, then re-run with --go to execute.

set -uo pipefail

OUT="multi_tile_generalization/block_masking_study/outputs"
ARCH="$OUT/old_confounded"
FIGARCH="$ARCH/figures"

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

run() {
  echo "  $*"
  [[ $GO -eq 1 ]] && eval "$*"
}

if [[ ! -d "$OUT" ]]; then
  echo "ERROR: $OUT not found. Run from the repo root (~/Prithvi/Prithvi-Sandbox)."
  exit 1
fi

echo "=============================================================="
if [[ $GO -eq 1 ]]; then echo "EXECUTING migration"; else echo "DRY RUN (no changes). Re-run with --go to apply."; fi
echo "=============================================================="

if [[ -d "$ARCH" && $GO -eq 1 ]]; then
  echo "NOTE: $ARCH already exists — new moves will be added to it."
fi

run "mkdir -p '$ARCH' '$FIGARCH'"

echo
echo "--- 1. Archive OLD block-only results (confounded random comparison) ---"
for bb in tiny 100M 300M 600M; do
  [[ -f "$OUT/results_${bb}.csv" ]] && run "mv '$OUT/results_${bb}.csv' '$ARCH/results_${bb}.csv'"
done

echo
echo "--- 2. Archive OLD aggregates / sanity check built on old comparison ---"
for f in block_summary.csv sanity_check_block_vs_random.xlsx; do
  [[ -f "$OUT/$f" ]] && run "mv '$OUT/$f' '$ARCH/$f'"
done

echo
echo "--- 3. Archive OLD crossover-dependent figures + stale plot script ---"
for f in fig2_block_vs_random.png fig3_difficulty_gap.png \
         fig_delta_distributions.png fig_delta_distributions_600M.png \
         fig_trimmed_bar_600M.png fig1_block_degradation.png \
         fig1a_block_degradation_clean.png; do
  [[ -f "$OUT/figures/$f" ]] && run "mv '$OUT/figures/$f' '$FIGARCH/$f'"
done
# plot_block_results.py carries the OLD hardcoded RANDOM_PSNR dict -> archive it
SCR="multi_tile_generalization/block_masking_study/scripts"
[[ -f "$SCR/plot_block_results.py" ]] && run "mv '$SCR/plot_block_results.py' '$ARCH/plot_block_results.py.OLD'"

echo
echo "--- 4. Promote corrected results_fixed_{bb}.csv -> results_{bb}.csv ---"
missing=0
for bb in tiny 100M 300M 600M; do
  if [[ -f "$OUT/results_fixed_${bb}.csv" ]]; then
    run "mv '$OUT/results_fixed_${bb}.csv' '$OUT/results_${bb}.csv'"
  else
    echo "  WARNING: $OUT/results_fixed_${bb}.csv not found"
    missing=1
  fi
done

echo
echo "=============================================================="
if [[ $GO -eq 1 ]]; then
  echo "Done. Old files are in: $ARCH"
  echo "Canonical corrected results now at: $OUT/results_{tiny,100M,300M,600M}.csv"
else
  echo "Preview only. Re-run with --go to apply."
fi
[[ $missing -eq 1 ]] && echo "NOTE: some results_fixed_*.csv were missing — check before --go."
echo "=============================================================="
