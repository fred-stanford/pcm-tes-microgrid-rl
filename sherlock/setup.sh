#!/bin/bash
# sherlock/setup.sh — one-time setup, run ONCE on a Sherlock LOGIN node (has internet):
#     cd $HOME/framework/RL        # repo root
#     bash sherlock/setup.sh
#
# Creates the `microgrid` conda env and prepares the log dir. After this, edit
# sherlock/env.sh to match how you activate conda, then use sherlock/submit_all.sh.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

echo "== microgrid Sherlock setup =="
echo "repo root: $(pwd)"

# --- conda ---------------------------------------------------------------------
# If conda isn't on PATH yet, load it (adjust to your install / Sherlock module):
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif command -v module >/dev/null 2>&1; then
        ml python || true          # or: ml anaconda
    fi
fi

if conda env list | grep -qE '^microgrid\s'; then
    echo "env 'microgrid' already exists — updating from sherlock/environment.yml"
    conda env update -n microgrid -f sherlock/environment.yml --prune
else
    echo "creating env 'microgrid' from sherlock/environment.yml"
    conda env create -f sherlock/environment.yml
fi

mkdir -p sherlock/logs

echo
echo "== setup done =="
echo "Next:"
echo "  1) edit sherlock/env.sh so 'conda activate microgrid' works on compute nodes"
echo "  2) sanity check the job map:  conda activate microgrid && python -m microgrid.study jobs --scenario B"
echo "  3) submit the whole pipeline: bash sherlock/submit_all.sh"
