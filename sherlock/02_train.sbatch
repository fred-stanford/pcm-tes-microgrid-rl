#!/bin/bash
# STAGE 2 — multi-seed x multi-hyperparameter training as a SLURM ARRAY.
# The array runs its tasks IN PARALLEL (subject to your account limits) — one task per
# (seed, layers, gamma) combo, 10 seeds x 6 configs = 60 tasks. They do NOT run in series:
# wall-clock is ~one training run, not 30. Confirm the count: python -m microgrid.study jobs --scenario B --count
#
#SBATCH --job-name=mg-train
#SBATCH --array=0-59                 # 0..(N-1) = 10 seeds x 6 configs; add %N (e.g. 0-59%12) to cap concurrency if your account requires
#SBATCH --partition=normal          # <-- adjust for Sherlock (normal / owners / <your PI>)
#SBATCH --time=1-00:00:00           # 24 h/task — generous buffer; a 500k-step run is usually a few hours.
##SBATCH --qos=long                 # UNCOMMENT only if you raise --time above 48 h (normal QOS caps at 2 days)
#SBATCH --cpus-per-task=4           # CPU-bound (tiny MLP + physics env); no GPU needed
#SBATCH --mem=8G
#SBATCH --output=sherlock/logs/train_%A_%a.out
#SBATCH --error=sherlock/logs/train_%A_%a.err
##SBATCH --mail-type=END,FAIL       # UNCOMMENT for email (NOTE: an array sends one email PER task = 60)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source sherlock/env.sh

SCENARIO="${SCENARIO:-B}"           # override: sbatch --export=ALL,SCENARIO=D sherlock/02_train.sbatch
TIMESTEPS="${TIMESTEPS:-500000}"

echo "=== train start $(date) | host=$(hostname) task=$SLURM_ARRAY_TASK_ID scenario=$SCENARIO steps=$TIMESTEPS ==="
python -u -m microgrid.study train \
    --scenario "$SCENARIO" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --timesteps "$TIMESTEPS"
echo "=== train done $(date) ==="
