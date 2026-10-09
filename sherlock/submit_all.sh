#!/bin/bash
# sherlock/submit_all.sh — submit the WHOLE pipeline as a dependency chain and return.
#   data-gen  ->  train (array)  ->  validate (array)  ->  aggregate+test-vs-RBC
# Each stage starts only if the previous one finished OK (afterok). Run from the repo root:
#     bash sherlock/submit_all.sh
# Override the scenario / DR case via env, e.g.:  SCENARIO=B DR_CASE=moderate bash sherlock/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

SCENARIO="${SCENARIO:-B}"
DR_CASE="${DR_CASE:-moderate}"
mkdir -p sherlock/logs

# Keep the training array size in sync with the study's job map.
NJOBS="$(python -m microgrid.study jobs --scenario "$SCENARIO" --count)"
NSEEDS=10
echo "scenario=$SCENARIO  dr_case=$DR_CASE  train tasks=$NJOBS  seeds=$NSEEDS"

DATA=$(sbatch --parsable \
    --export=ALL,DR_CASE="$DR_CASE" \
    sherlock/01_generate_data.sbatch)
echo "  [1] datagen      -> job $DATA"

TRAIN=$(sbatch --parsable --dependency=afterok:"$DATA" \
    --array=0-$((NJOBS-1)) \
    --export=ALL,SCENARIO="$SCENARIO" \
    sherlock/02_train.sbatch)
echo "  [2] train        -> job $TRAIN  (afterok:$DATA)"

# afterok on an array job waits for EVERY task to succeed.
VAL=$(sbatch --parsable --dependency=afterok:"$TRAIN" \
    --array=0-$((NSEEDS-1)) \
    --export=ALL,SCENARIO="$SCENARIO" \
    sherlock/03_validate.sbatch)
echo "  [3] validate     -> job $VAL  (afterok:$TRAIN)"

AGG=$(sbatch --parsable --dependency=afterok:"$VAL" \
    --export=ALL,SCENARIO="$SCENARIO" \
    sherlock/04_aggregate.sbatch)
echo "  [4] aggregate    -> job $AGG  (afterok:$VAL)"

echo
echo "submitted. watch with:  squeue -u \$USER   |   tail -f sherlock/logs/*.out"
echo "final result:           runs/study_${SCENARIO}/study_summary.json"
