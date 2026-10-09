#!/bin/bash
# submit_arm.sh — add ONE study arm (train -> validate -> aggregate) for a scenario, WITHOUT
# regenerating data (assumes data/{train,validation,test}_new_0605.csv already exist from the main
# run). Use this to add the A1 ablation arm alongside the main B run.
#   SCENARIO=B_nopcm bash sherlock/submit_arm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SCENARIO="${SCENARIO:-B_nopcm}"
mkdir -p sherlock/logs

NJOBS="$(python -m microgrid.study jobs --scenario "$SCENARIO" --count)"
NSEEDS=10
echo "arm scenario=$SCENARIO  train tasks=$NJOBS  seeds=$NSEEDS"

TRAIN=$(sbatch --parsable --array=0-$((NJOBS-1)) \
    --export=ALL,SCENARIO="$SCENARIO" sherlock/02_train.sbatch)
echo "  train     -> job $TRAIN"

VAL=$(sbatch --parsable --dependency=afterok:"$TRAIN" --array=0-$((NSEEDS-1)) \
    --export=ALL,SCENARIO="$SCENARIO" sherlock/03_validate.sbatch)
echo "  validate  -> job $VAL  (afterok:$TRAIN)"

AGG=$(sbatch --parsable --dependency=afterok:"$VAL" \
    --export=ALL,SCENARIO="$SCENARIO" sherlock/04_aggregate.sbatch)
echo "  aggregate -> job $AGG  (afterok:$VAL)"

echo "result -> runs/study_${SCENARIO}/study_summary.json"
