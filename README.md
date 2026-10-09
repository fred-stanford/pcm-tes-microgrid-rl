# Deep reinforcement learning control of a grid-tied microgrid with a heat pump and PCM thermal storage

Code and results for the paper *Deep reinforcement learning control of a grid-tied microgrid with a heat
pump and PCM thermal storage: a comparison with rule-based control in an experimentally calibrated
simulation* (under review).

The study compares a deep Q-network (DQN) controller with a rule-based controller (RBC) on a simulated
grid-tied microgrid in Half Moon Bay, California: a 3.84 kW PV array, a 7.2 kWh battery, a reversible
air-to-water heat pump, and hot and cold phase-change-material thermal energy storage (PCM TES). Both
controllers choose one of 20 actions (an HP-TES mode and a battery command) every 15 minutes under a
time-of-use tariff, an export credit, demand-response events, and grid outages. On the 2024 test year,
the RL controller reduced operating cost by 47% and thermal discomfort by 27%.

## Repository layout

| Path | Contents |
|---|---|
| `microgrid/` | Gymnasium environment, observation and reward, DQN and RBC agents, training, validation, evaluation, and the multi-seed study driver (`study.py`) |
| `physics/` | Simulation models: building envelope, HP-TES system (`hp_tes/`), battery and grid balance (`loadmatch.py`), grid events, and the RBC rules (`rbc_rules.py`) |
| `Input_Parameters.py` | Site, equipment, and reward constants |
| `data_generation_pipeline/` | Builds the train/validation/test datasets from NSRDB weather, the PV and load models, and the tariffs |
| `data/` | Calibrated building-model coefficients; the generated datasets are written here |
| `sherlock/` | SLURM scripts used to run the study on Stanford's Sherlock cluster |
| `results/study_B/` | Summary results behind the paper's tables and figures (see `results/README.md`) |
| `Fig/` | Figure scripts and the paper's figures (PDF) |
| `tests/` | Unit tests |

## Setup

```bash
conda env create -f environment.yml
conda activate microgrid
python -m pytest tests/ -q
```

## Reproducing the study

All commands run from the repository root.

**1. Data.** Download the NSRDB 5-minute data for the site (location 57357, 37.46° N, 122.43° W, years
2018 to 2024) into `data_generation_pipeline/NSRDB_data/`, keeping the NSRDB file names
(`57357_37.46_-122.43_<year>.csv`). Then generate the datasets:

```bash
cd data_generation_pipeline
python generate_dataset.py --split all --out-dir ../data   # writes data/{train,validation,test}_new_0605.csv
cd ..
```

The prepared datasets are also in the data archive (https://doi.org/10.5281/zenodo.23252129).

**2. Train.** The study trains 60 agents: 10 seeds × discount factor {0.98, 0.99, 0.9965} × {2, 3}
hidden layers, 500,000 decisions each. Each job is independent:

```bash
python -m microgrid.study jobs --scenario B                    # list the 60 jobs
python -m microgrid.study train --scenario B --index 0         # train job 0 (repeat for 0..59)
```

**3. Select and test.** Pick each seed's configuration on the 2023 validation year, then evaluate the ten
selected controllers and the RBC on 2024:

```bash
python -m microgrid.study validate --scenario B --seed 0       # repeat for seeds 0..9
python -m microgrid.study aggregate --scenario B
```

**4. Monte-Carlo.** Evaluate the median controller and the RBC over 100 grid-event schedules:

```bash
python -m microgrid.study montecarlo --scenario B --n 100
```

**5. Figures.** See `Fig/README.md`.

On a SLURM cluster, `sherlock/` runs steps 1 to 4 as a dependency chain (`sherlock/README.md`).
Outputs go to `runs/` (not tracked).

## Data and results

- `results/study_B/` contains the summary outputs reported in the paper.
- The data archive (https://doi.org/10.5281/zenodo.23252129) contains the generated datasets, the 5-minute test-year traces
  used for the time-series figures, and the trained models.
- The weather data are from the U.S. National Solar Radiation Database (NSRDB). Tariffs follow PG&E
  Schedule E-ELEC and PG&E's Energy Export Credit values (`data_generation_pipeline/Data/`).

## License

The code is released under the MIT License (`LICENSE`). The data archive (datasets, traces, results,
and trained models) is released under CC BY 4.0.

## Citation

The paper is under review; its citation will be added on publication. Until then, please cite the data archive: https://doi.org/10.5281/zenodo.23252129.
