#!/bin/bash
# sherlock/submit_everything.sh
# ============================================================================
# ONE command runs the ENTIRE study end-to-end on Sherlock as a SLURM
# dependency chain, then returns immediately. Nothing else to babysit.
#
#   bash sherlock/submit_everything.sh
#
# What it does (all via afterok dependencies, so each stage waits for the last):
#   0. archive any previous runs/ results (so the fresh run never mixes with old)
#   1. regenerate the datasets (data/{train,validation,test}_new_0605.csv)
#   2. for EACH arm  — B (TES + RL)  and  B_nopcm (PCM-disabled ablation) —
#        train (60-task array: 10 seeds x 6 configs)
#        -> validate (10-task array: pick each seed's winner)
#        -> aggregate (test vs the deployed RBC baseline A)
#        -> Monte-Carlo resilience (10-shard array, paired RL vs RBC)
#        -> merge (CIs / P(RL better) / worst-case)
#   train/validate/montecarlo of the two arms run in PARALLEL after data-gen; the
#   two AGGREGATE steps are SERIALIZED (B_nopcm's aggregate waits for B's) because
#   both write the shared RBC-baseline cache (runs/baselines/A) — serializing avoids
#   a torn write of that shared cache. The aggregate step is quick, so the cost is small.
#
# Optional overrides:
#   DR_CASE=moderate|severe   (default moderate)
#   N_MC=200                  (Monte-Carlo realizations per arm; MUST be a multiple of 10)
#   CAP=12                    (max concurrent array tasks per stage; default none)
#   Example:  N_MC=200 CAP=16 bash sherlock/submit_everything.sh
#
# Watch:   squeue -u $USER      |      tail -f sherlock/logs/*.out
# Results: runs/study_B/{study_summary.json, montecarlo_summary.json}
#          runs/study_B_nopcm/{study_summary.json, montecarlo_summary.json}
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root
source sherlock/env.sh                  # activate the python env (also needed for the count calls below)

DR_CASE="${DR_CASE:-moderate}"
N_MC="${N_MC:-200}"
CAP="${CAP:-}"                          # e.g. CAP=12 -> append %12 to arrays if your account caps concurrency
PCT=""; if [ -n "$CAP" ]; then PCT="%$CAP"; fi
mkdir -p sherlock/logs

# N_MC must divide evenly across the 10 Monte-Carlo shards, else realizations are silently dropped.
if [ $(( N_MC % 10 )) -ne 0 ]; then
    NEW=$(( (N_MC / 10) * 10 )); echo "WARNING: N_MC=$N_MC is not a multiple of 10; rounding down to $NEW"; N_MC=$NEW
fi

# 0. Archive previous results so a re-run never mixes with old (e.g. old-physics) outputs.
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCH="runs/_archive_$STAMP"
for p in runs/study_B runs/study_B_nopcm runs/baselines runs/B_g* runs/B_nopcm_g* runs/D_g*; do
    [ -e "$p" ] || continue
    mkdir -p "$ARCH"; mv "$p" "$ARCH/"; echo "archived $p -> $ARCH/"
done

# Array sizes derived from the study itself so they can't drift out of sync with study.SEEDS.
NJOBS="$(python -m microgrid.study jobs --scenario B --count)"          # 60 = seeds x layers x gammas
NSEEDS="$(python -c 'import microgrid.study as s; print(len(s.SEEDS))')" # 10
echo "=== full study: arms [B, B_nopcm] | train=$NJOBS tasks/arm | seeds=$NSEEDS | MC=$N_MC/arm | dr=$DR_CASE ${CAP:+| cap=$CAP} ==="

# 1. Data generation (ONCE; shared by both arms).
DATA="$(sbatch --parsable --export=ALL,DR_CASE="$DR_CASE" sherlock/01_generate_data.sbatch)"
echo "[1] datagen                       -> job $DATA"

# One full arm: train -> validate -> aggregate -> montecarlo -> mc-merge, all gated on datagen.
# $2 (optional) = an extra afterok jobid the AGGREGATE must also wait for (used to serialize the
# two arms' aggregates so they don't race on the shared RBC-baseline cache). Sets LAST_AGG on return.
LAST_AGG=""
submit_arm () {
    local SCN="$1" EXTRA_AGG="${2:-}" TR VA AG MC MM AGG_DEP
    TR="$(sbatch --parsable --dependency=afterok:"$DATA" --array=0-$((NJOBS-1))$PCT \
         --export=ALL,SCENARIO="$SCN" sherlock/02_train.sbatch)"
    VA="$(sbatch --parsable --dependency=afterok:"$TR" --array=0-$((NSEEDS-1))$PCT \
         --export=ALL,SCENARIO="$SCN" sherlock/03_validate.sbatch)"
    AGG_DEP="afterok:$VA"; if [ -n "$EXTRA_AGG" ]; then AGG_DEP="afterok:$VA:$EXTRA_AGG"; fi
    AG="$(sbatch --parsable --dependency="$AGG_DEP" \
         --export=ALL,SCENARIO="$SCN" sherlock/04_aggregate.sbatch)"
    MC="$(sbatch --parsable --dependency=afterok:"$AG" --array=0-9$PCT \
         --export=ALL,SCENARIO="$SCN",N_TOTAL="$N_MC" sherlock/05_montecarlo.sbatch)"
    MM="$(sbatch --parsable --dependency=afterok:"$MC" \
         --export=ALL,SCENARIO="$SCN" sherlock/06_mc_merge.sbatch)"
    echo "[$SCN] train=$TR  validate=$VA  aggregate=$AG  montecarlo=$MC  merge=$MM"
    LAST_AGG="$AG"
}

submit_arm B                       # arm B: computes + caches the RBC-A baseline
submit_arm B_nopcm "$LAST_AGG"     # arm B_nopcm: its aggregate waits for B's -> reads the cached A, no race

echo
echo "=== all submitted. the whole study runs unattended via afterok dependencies. ==="
echo "watch:    squeue -u \$USER      |      tail -f sherlock/logs/*.out"
echo "results:  runs/study_B/{study_summary.json, montecarlo_summary.json}"
echo "          runs/study_B_nopcm/{study_summary.json, montecarlo_summary.json}"
echo "headline: RBC(A)  vs  DQN-nopcm(B_nopcm)  vs  DQN-full(B)   + paired Monte-Carlo resilience CIs"
