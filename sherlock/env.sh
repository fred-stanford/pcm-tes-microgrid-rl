#!/bin/bash
# sherlock/env.sh — EDIT ONCE. Sourced by every .sbatch script to activate the Python env.
# This is the single place that knows how to enter your conda/venv on Sherlock.
#
# --- Option A: conda (recommended; matches sherlock/setup.sh) -----------------
# Point this at wherever you installed miniconda/anaconda on Sherlock.
source "$HOME/miniconda3/etc/profile.d/conda.sh"   # <-- adjust: miniconda3 / anaconda3 / $GROUP_HOME/...
conda activate microgrid
#
# --- Option B: Sherlock module + virtualenv (comment out A, use this) ---------
# ml python/3.10
# source "$HOME/envs/microgrid/bin/activate"
#
# Keep the physics/RL and data-gen imports deterministic + single-threaded per task
# (SLURM allocates the cores; let the array give you parallelism, not BLAS threads).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# Report the failing line on any error (sourced into each job; aids cluster debugging).
trap 'echo "FAILED at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR
