#!/bin/bash
# STAGE 3 — per-seed validation as a SLURM ARRAY (parallel; one task per seed).
# Each task ranks that seed's 6 agents on the validation year (2023) by total training reward,
# writes runs/study_B/seed_<s>/winner.json, and saves the winner's validation timeseries.
# 6 full-year rollouts + 1 winner re-eval per task = ~7 rollouts.
#
#SBATCH --job-name=mg-validate
#SBATCH --array=0-9                  # one task per seed (parallel) — 10 seeds
#SBATCH --partition=normal          # <-- adjust for Sherlock
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=sherlock/logs/val_%A_%a.out
#SBATCH --error=sherlock/logs/val_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source sherlock/env.sh

SCENARIO="${SCENARIO:-B}"

echo "=== validate start $(date) | host=$(hostname) seed-index=$SLURM_ARRAY_TASK_ID scenario=$SCENARIO ==="
python -u -m microgrid.study validate \
    --scenario "$SCENARIO" \
    --seed-index "$SLURM_ARRAY_TASK_ID"
echo "=== validate done $(date) ==="
