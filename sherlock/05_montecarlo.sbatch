#!/bin/bash
# STAGE 5 (optional, post-training) — M12 Monte-Carlo RESILIENCE, SHARDED as a SLURM ARRAY.
# Paired RL(median winner) vs RBC over N_TOTAL event realizations at the TRUE rate on the fixed 2024
# world. A full-year paired rollout is ~5-10 min, so this is sharded across the array to run in
# parallel: each of the 10 tasks does N_TOTAL/10 realizations and writes a shard file. Then run
# 06_mc_merge.sbatch (afterok) to concatenate the shards + write the summary.
# Run AFTER 04_aggregate (needs runs/study_<scn>/study_summary.json).
#
#SBATCH --job-name=mg-mc
#SBATCH --array=0-9                  # 10 shards, in parallel  (keep in sync with SHARDS below)
#SBATCH --partition=normal          # <-- adjust for Sherlock
#SBATCH --time=12:00:00             # per shard: ~(N_TOTAL/10)*2 rollouts * ~6-10 min
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=sherlock/logs/mc_%A_%a.out
#SBATCH --error=sherlock/logs/mc_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source sherlock/env.sh

SCENARIO="${SCENARIO:-B}"
N_TOTAL="${N_TOTAL:-200}"
SHARDS=10                            # MUST equal the array size (0-9)
PER=$(( N_TOTAL / SHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER ))

echo "=== mc shard start $(date) | task=$SLURM_ARRAY_TASK_ID start=$START per=$PER scenario=$SCENARIO ==="
python -u -m microgrid.study montecarlo \
    --scenario "$SCENARIO" --start "$START" --n "$PER" --shards "$SHARDS"
echo "=== mc shard done $(date) ==="
