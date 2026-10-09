# Fig/ — paper figures

`make_figs.py` draws every data figure from the study outputs, and `make_schematic.py` draws the system
schematic (Fig. 1). Each figure is written as `.pdf` (vector, in the repo) and `.png` (300 dpi, not tracked)
to `Fig/main` and `Fig/SI`. No simulation is re-run.

## Regenerate
```
# STUDY_DIR: the study output folder (default runs/study_B). It needs the summary files in
# results/study_B plus the aggregate/ traces from the data archive (see the top-level README).
STUDY_DIR=runs/study_B python Fig/make_figs.py     # Fig. 2-6 and Fig. S1-S5
python Fig/make_schematic.py                       # Fig. 1 (no data needed)
```
On Windows PowerShell, set the variable with `$env:STUDY_DIR = "runs/study_B"`. Optionally set
`FIG_MIRROR` to a second folder that should receive copies of the figures.

## Contents

**`main/`**
| File | Figure |
|---|---|
| `Fig1_system_schematic` | System topology and control interface |
| `Fig2_annual_ledger` | (a) cash flows, (b) discomfort and loss of load at estimated prices |
| `Fig3_robustness` | (a) net grid cost by training seed, (b) outcomes, RL relative to RBC |
| `Fig4_montecarlo_resilience` | (a) net grid cost saving and (b) loss of load over 100 event schedules |
| `Fig5_diurnal_dispatch` | Year-averaged daily (a) grid import and (b) HP-TES electricity |
| `Fig6_summer_operation` | Average summer day: (a) battery, (b) import, (c) export and credit, (d) cold PCM TES charging |

**`SI/`**
| File | Figure |
|---|---|
| `FigS1_calibration_PLACEHOLDER` | Calibration against measured data (placeholder) |
| `FigS2_learning_curves` | Validation reward during training, by discount factor |
| `FigS3_validation_selection` | Per-seed configuration selection on the validation year |
| `FigS4_cumulative_cost` | Cumulative net grid cost over 2024 (median RL controller vs RBC) |
| `FigS5_dispatch_window` | Three days in April 2024 at 15-minute resolution |

Colors follow the Okabe-Ito colorblind-safe palette: RBC grey `#7F7F7F`, RL blue `#0072B2`,
favors RL green `#009E73`, favors RBC vermillion `#D55E00`, PV and export credit orange `#E69F00`.
