#!/bin/bash
# STAGE 6 (post-Monte-Carlo) — merge the shard files from 05_montecarlo.sbatch and write the summary.
# Reads runs/study_<scn>/montecarlo_metrics_*.csv, concatenates -> montecarlo_metrics.csv, and writes
# montecarlo_summary.json (RL vs RBC mean/std, bootstrap CIs, P(RL better), worst-case). No rollouts.
# Submit with a dependency on the array:  sbatch --dependency=afterok:<mc_array_jobid> sherlock/06_mc_merge.sbatch
#
#SBATCH --job-name=mg-mc-merge
#SBATCH --partition=normal          # <-- adjust for Sherlock
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=sherlock/logs/mcmerge_%j.out
#SBATCH --error=sherlock/logs/mcmerge_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source sherlock/env.sh

SCENARIO="${SCENARIO:-B}"

echo "=== mc merge start $(date) | scenario=$SCENARIO ==="
python -u -m microgrid.study montecarlo --scenario "$SCENARIO" --merge
echo "result -> runs/study_${SCENARIO}/montecarlo_summary.json"
echo "=== mc merge done $(date) ==="
