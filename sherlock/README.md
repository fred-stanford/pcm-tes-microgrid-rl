# Sherlock end-to-end pipeline

One folder to run the **whole study** on Stanford Sherlock (SLURM), as a dependency chain:

```
data generation  →  training (multi-seed × multi-hyperparameter)  →  validation  →  testing vs RBC
   Stage 1               Stage 2 (array)                              Stage 3 (array)   Stage 4
```

Everything is offline on the compute nodes (the Open-Meteo forecast step was removed — the
controllers are forecast-free), so no internet is needed except for the one-time env setup.

## Files
| File | What it does |
|---|---|
| `environment.yml` | Unified Linux conda env (RL + data-gen deps; no Mac `apple` channel). |
| `setup.sh` | **Run once on a login node.** Creates/updates the `microgrid` env + `logs/`. |
| `env.sh` | **Edit once.** How every job activates the env (conda path / module). Sourced by all `.sbatch`. |
| `01_generate_data.sbatch` | Stage 1 — `generate_dataset.py` → `data/{train,validation,test}_new_0605.csv`. |
| `02_train.sbatch` | Stage 2 — SLURM array, one task per (seed, layers, γ) = 60 tasks (10 seeds × 6 configs). |
| `03_validate.sbatch` | Stage 3 — SLURM array, one task per seed = 10 tasks; picks each seed's winner. |
| `04_aggregate.sbatch` | Stage 4 — test winners on 2024, report mean ± std vs RBC, save figures. |
| `05_montecarlo.sbatch` | Stage 5 (optional, post-training) — resilience Monte-Carlo, **sharded as a 10-task array** (paired RL vs RBC over N event realizations). |
| `06_mc_merge.sbatch` | Stage 6 — merge the MC shards + write the resilience summary (CIs, P(RL better), worst-case). |
| `submit_all.sh` | Submit stages 1–4 chained with `--dependency=afterok`. |

## Parallelism & walltime
- **Training runs in parallel, not in series.** Stage 2 is a SLURM **array** (`--array=0-59`), so all 60
  (seed × layers × γ) agents train concurrently — wall-clock ≈ *one* training run. Stage 3 is a
  10-task array (one per seed). If your account caps concurrent jobs, add `%N` (e.g. `--array=0-59%12`).
- **Walltimes** are set generously (train 24 h/task, validate/aggregate 8 h, data-gen 4 h) and all stay
  **under the 48 h `normal`-QOS cap** — so no `--qos=long` is needed. **Calibrate from the smoke run:**
  time the 2 k-step `--smoke` train, ×250 for 500 k steps, add ~5 full-year eval rollouts, add buffer.
  If a real run exceeds ~20 h, raise `--time` (up to `2-00:00:00`) or, beyond 48 h, uncomment
  `#SBATCH --qos=long` in `02_train.sbatch`.
- **Monte-Carlo** is the expensive part (a full-year paired rollout is ~5–10 min, so N=200 serial ≈
  30–70 h). Stage 5 shards it across a 10-task array (each ~N/10 realizations), then Stage 6 merges:
  ```bash
  MC=$(sbatch --parsable --export=ALL,SCENARIO=B,N_TOTAL=200 sherlock/05_montecarlo.sbatch)
  sbatch --dependency=afterok:$MC --export=ALL,SCENARIO=B sherlock/06_mc_merge.sbatch
  ```

## Quick start
```bash
cd <repo root>                        # wherever you cloned it
bash sherlock/setup.sh                # 1) one-time: build the conda env
$EDITOR sherlock/env.sh               # 2) one-time: make `conda activate microgrid` work on nodes
conda activate microgrid
python -m microgrid.study jobs --scenario B   # 3) sanity-check the 60-job map
bash sherlock/submit_all.sh           # 4) submit the whole pipeline
```
Watch progress: `squeue -u $USER` and `tail -f sherlock/logs/*.out`.
Final result: `runs/study_B/study_summary.json` (+ `aggregate_metrics.csv`, figures).

## Do I generate data locally first? (No — but you can, to inspect it)
Data generation is **Stage 1 on Sherlock** (it's offline = no internet needed, so it runs on a
compute node). `submit_all.sh` does data → train → validate → test in one shot; you don't have to
produce anything here first. Generation is deterministic (fixed seeds), so re-running it on the
cluster reproduces the exact same CSVs.

If you'd rather eyeball the data before spending cluster hours, generate it locally:
```bash
cd data_generation_pipeline
python generate_dataset.py --split all --out-dir ../data   # writes data/{train,validation,test}_new_0605.csv
```
Then either commit/scp those CSVs and still run `submit_all.sh` (Stage 1 just recreates them
identically), or skip Stage 1 by submitting the rest by hand:
```bash
NJOBS=$(python -m microgrid.study jobs --scenario B --count)
T=$(sbatch --parsable --array=0-$((NJOBS-1)) --export=ALL,SCENARIO=B sherlock/02_train.sbatch)
V=$(sbatch --parsable --dependency=afterok:$T --array=0-9 --export=ALL,SCENARIO=B sherlock/03_validate.sbatch)
sbatch --dependency=afterok:$V --export=ALL,SCENARIO=B sherlock/04_aggregate.sbatch
```

## Outputs — every result as CSV (for offline plotting)
After a full run, pull `runs/study_B/` back and plot locally. Everything is CSV.

**Test — the headline agent comparison** (`runs/study_B/aggregate/`)
- `detailed_seed{0..9}.csv` — full **5-min timeseries for each seed winner** (temps, SOCs, HP mode,
  PCM, battery, grid import/export, prices, PV, all physics channels).
- `DQN.csv` — the **median** winner's 5-min timeseries (canonical; = `detailed_seed<median>`).
- `RBC.csv` — the RBC baseline's 5-min timeseries (same schema) → direct DQN-vs-RBC overlays.
- `trace_seed{0..9}.csv` — compact **per-15-min-decision** trace per winner (temps, SOCs, prices,
  actions, import/export kW, DR/outage flags) → good for the seed-spread band.

**Test — aggregate metrics** (`runs/study_B/`)
- `aggregate_metrics.csv` — one row per seed winner + `mean/std/min/max` + an `RBC` row
  (net cost, import/export kWh, discomfort, loss-of-load, peaks, throughputs, …).
- `study_summary.json` — headline: RL mean ± std vs RBC, seeds beating RBC, the median winner.

**Validation** (`runs/study_B/`)
- `validation_all.csv` — all seeds' 6-agent rankings concatenated (which agent won and why).
- `seed_{s}/validation_metrics.csv` — that seed's 6 agents ranked; `seed_{s}/winner.json`.
- `seed_{s}/winner_val_detailed.csv` + `winner_val_trace.csv` — the selected agent's **2023 timeseries**.

**Training** (`runs/study_B/`)
- `learning_curves.csv` — validation reward vs timestep for **all 60 runs** (columns: seed, layers,
  gamma, run_name, timestep, val_reward_mean/std/min/max) → convergence and γ-effect plots.
- `training_summary.csv` — per run: best vs final validation reward + best timestep.
- per run `runs/<run_name>/`: `learning_curve.csv`, `episode_log.csv`, `best_model.zip`, `model.zip`,
  `config.json`.

## Knobs
- **Scenario:** `SCENARIO=B bash sherlock/submit_all.sh` (B = TES + RL, the main case; D = 2×battery, no-TES).
- **DR stress case:** `DR_CASE=severe bash sherlock/submit_all.sh` (default `moderate`).
- **Training budget:** edit `TIMESTEPS` in `02_train.sbatch` (default 500k). Check the validation
  curves (`best_model` vs final) before increasing.
- **Seeds / hyperparameters:** `SEEDS` in `microgrid/study.py`; `SWEEP_LAYERS`, `SWEEP_GAMMAS` in
  `microgrid/config.py`. If you change these, the array sizes in `02/03` update automatically via
  `submit_all.sh` (it reads `study jobs --count`), but the hard-coded `#SBATCH --array` defaults
  are for the 10×6 grid — `submit_all.sh` overrides them, so prefer submitting through it.

## Cluster placeholders to set (once)
Every `.sbatch` has `--partition=normal` — change to your Sherlock partition (`normal`, `owners`,
or your PI's). Adjust `--time`/`--mem`/`--cpus-per-task` if your quotas differ. `env.sh` holds the
conda activation. Nothing else is Sherlock-specific.

## Notes / caveats
- **Smoke test first:** `python -m microgrid.study train --scenario B --index 0 --smoke` (a 2k-step run)
  after Stage 1 checks the paths before committing cluster hours.
- **`afterok` on arrays** waits for *every* task to succeed; if one training task dies, validation
  won't start. Re-run just the failed array index (`sbatch --array=<i> sherlock/02_train.sbatch`),
  then submit Stage 3/4 manually.
- **Grid events are sampled at run time:** the env draws a fresh realization per training episode
  (event-stratified for learnability), DR is a critical-peak *price* event (the grid stays up), and the
  CSV event columns are ignored. Resilience is evaluated by the Stage-5 Monte-Carlo at the true event
  rate; the paper uses `N_TOTAL=100`. The observation has 19 values, including the DR and PSPS
  advance-notice signals.
