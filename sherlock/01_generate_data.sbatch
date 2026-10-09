#!/bin/bash
# STAGE 1 — generate the train/validation/test datasets (offline, no internet needed).
# Runs data_generation_pipeline/generate_dataset.py (forecast-free, vectorized) and writes
#   data/{train,validation,test}_new_0605.csv  (2018-2022 / 2023 / 2024).
# Single job; fast (data-gen for 1 year is ~1 min, 5-year train a few min).
#
#SBATCH --job-name=mg-datagen
#SBATCH --partition=normal          # <-- adjust for Sherlock
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=sherlock/logs/datagen_%j.out
#SBATCH --error=sherlock/logs/datagen_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"              # repo root
source sherlock/env.sh

DR_CASE="${DR_CASE:-moderate}"     # override: sbatch --export=ALL,DR_CASE=severe sherlock/01_generate_data.sbatch

echo "=== datagen start $(date) | host=$(hostname) dr_case=$DR_CASE ==="
cd data_generation_pipeline
python -u generate_dataset.py --split all --out-dir ../data --dr-case "$DR_CASE"
cd ..

echo "== datasets in data/ =="
ls -lh data/train_new_0605.csv data/validation_new_0605.csv data/test_new_0605.csv
# Sanity: train should be 525,600 rows (5 x 105,120), val/test 105,120 each (leap Feb-29 stripped).
python -u -c "import pandas as pd; [print(f, len(pd.read_csv('data/'+f))) for f in ['train_new_0605.csv','validation_new_0605.csv','test_new_0605.csv']]"
echo "=== datagen done $(date) ==="
