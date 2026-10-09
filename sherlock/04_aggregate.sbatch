#!/bin/bash
# STAGE 4 — aggregate + TEST vs RBC (single task).
# Evaluates the per-seed winners on the test year (2024), reports RL mean +/- std vs the
# deterministic RBC baseline, saves per-seed 5-min traces + learning curves for offline plotting.
# ~7 full-year rollouts (5 winners + RBC + median re-eval).
#
#SBATCH --job-name=mg-aggregate
#SBATCH --partition=normal          # <-- adjust for Sherlock
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=sherlock/logs/agg_%j.out
#SBATCH --error=sherlock/logs/agg_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source sherlock/env.sh

SCENARIO="${SCENARIO:-B}"

echo "=== aggregate start $(date) | host=$(hostname) scenario=$SCENARIO ==="
python -u -m microgrid.study aggregate --scenario "$SCENARIO"

# Optional: seed-aware figures (safe to remove if you plot locally instead).
if python -c "import microgrid.plots_study" 2>/dev/null; then
    echo "generating study figures ..."
    python -u -m microgrid.plots_study --scenario "$SCENARIO" || echo "(plots step skipped/failed - non-fatal)"
fi
echo "=== aggregate done $(date) ==="
