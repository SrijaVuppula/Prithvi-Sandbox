#!/bin/bash
# usage: BACKBONES="tiny 100M" SEEDS="1 2 3" bash scripts/run_seeds.sh
export PYTHONUNBUFFERED=1
cd ~/Prithvi/Prithvi-Sandbox/multi_tile_generalization/block_masking_study
source ~/.venv/bin/activate
BACKBONES=${BACKBONES:-"tiny 100M"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
for bb in $BACKBONES; do
  for s in $SEEDS; do
    cost=outputs/finetune_logs/seeds/seed$s/${bb}_finetune_cost.csv
    res=outputs_finetuned/seeds/seed$s/results_${bb}.csv
    if [ ! -f "$cost" ]; then
      echo "[$(date +%F_%T)] TRAIN $bb seed=$s"
      python scripts/train_block_finetune_seeded.py --backbone $bb --epochs 20 --patience 99 --seed $s \
        || { echo "TRAIN FAILED $bb seed=$s"; continue; }
      rm -f checkpoints/seeds/seed$s/$bb/last.pt
    fi
    rows=$( [ -f "$res" ] && wc -l < "$res" || echo 0 )
    if [ "$rows" -lt 10001 ]; then
      echo "[$(date +%F_%T)] EVAL  $bb seed=$s"
      python scripts/run_paired_block_random_finetuned_seeded.py --backbone $bb --seed $s \
        || echo "EVAL FAILED $bb seed=$s"
    fi
    echo "[$(date +%F_%T)] DONE  $bb seed=$s"
  done
done
echo "ALL DONE"
