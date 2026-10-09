# results/ — study outputs behind the paper

`study_B/` holds the summary outputs of the 60-run study (10 training seeds × 6 hyperparameter
configurations) and its evaluation on the 2024 test year. "B" is the scenario with PCM TES controlled
by RL; its rule-based baseline is scenario "A". Costs are in US$ per year and energies in kWh.

| File | Contents | Used for |
|---|---|---|
| `aggregate_metrics.csv` | Test-year outcomes for each seed's selected controller, plus `mean`, `std`, `min`, `max`, and `RBC` rows | Table 3, Fig. 3 |
| `montecarlo_metrics.csv` | Paired RBC and RL outcomes for each of the 100 event schedules | Fig. 4 |
| `montecarlo_summary.json` | Monte-Carlo means, standard deviations, bootstrap 95% CIs, win shares, worst cases | Table 4 |
| `study_summary.json` | Selected controller per seed, mean ± s.d. per outcome, and the median controller | §4.1 |
| `learning_curves.csv` | Validation reward every 100,000 decisions for all 60 runs | Fig. S2 |
| `training_summary.csv` | Best and final validation reward and best checkpoint for each run | §4.1 |
| `validation_all.csv` | Validation-year (2023) metrics of all six configurations for each seed | Fig. S3 |
| `seed_*/winner.json` | The configuration selected for each seed | §3.4 |

The 5-minute test-year traces (`aggregate/RBC.csv`, `aggregate/DQN.csv`, `aggregate/detailed_seed*.csv`)
behind Figs. 5, 6, S4, and S5 total 1.04 GB and are in the data archive (see the top-level README).
`Fig/make_figs.py` loads them at start-up, so to redraw the figures, place them in `aggregate/` next to
these files and point `STUDY_DIR` to that folder.
