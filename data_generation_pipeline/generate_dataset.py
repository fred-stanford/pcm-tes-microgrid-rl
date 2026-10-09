"""
generate_dataset.py — offline, forecast-free dataset generation for the RL microgrid study.

Script version of Data_Preprocess_open_meteo_prices.ipynb, refactored for cluster (Sherlock) use:
  * OFFLINE: no Open-Meteo forecast API call (compute nodes have no internet). The 4 `*Forecasted`
    columns are set to 0.0 — the controllers are forecast-free, so they are unused (kept only for
    the RL loader's schema/REQUIRED_COLUMNS compatibility).
  * VECTORIZED: no per-step Python loop (the loop existed only to build the forecast columns).
  * PARAMETERIZED SPLITS: generates train / validation / test in one run, writing to the RL repo's
    data/ folder with the names microgrid.config expects.
  * Independent stochastic events per split (distinct event seed) -> no cross-split event leak (C3).

Run from THIS folder (data_generation_pipeline/), which holds the local modules + NSRDB_data/ + Data/:
    python generate_dataset.py --split all --out-dir ../data
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

import Input_Parameters
import Data_Conversion
import Electrical_Load
import Solar_Generation
import grid_electricity_prices
import grid_events

# Chronological splits over the available NSRDB weather (2018-2024). Prices/load/PV are
# year-agnostic, so weather is the only availability constraint.
SPLITS = {
    "train": list(range(2018, 2023)),   # 2018-2022 (5 years)
    "validation": [2023],
    "test": [2024],
}
# Output filenames microgrid.config reads (base names; set USE_INDEPENDENT_EVENTS=False so these
# are used directly — the events below are already independent per split).
OUT_NAMES = {
    "train": "train_new_0605.csv",
    "validation": "validation_new_0605.csv",
    "test": "test_new_0605.csv",
}
# Fixed seed for the synthetic-load stochasticity (traceability); events use a distinct per-split seed.
LOAD_SEED = 1
EVENT_SEED = {"train": 20180, "validation": 20230, "test": 20240}

FORECAST_COLS = [
    "Total PV Capacity Factor Forecasted", "Total Electrical Load Forecasted",
    "Total Heating Load Forecasted", "Total Cooling Load Forecasted",
]


def build_split(years, load_seed, event_seed, dr_case="moderate", nsrdb_dir="NSRDB_data"):
    """Assemble the 25-column exogenous CSV (no forecasts computed) for `years`."""
    res_min = Input_Parameters.δt / 60.0
    start = f"{min(years)}-01-01"
    end = f"{max(years)}-12-31"

    np.random.seed(load_seed)
    random.seed(load_seed)

    weather = Data_Conversion.fetch_historical_weather_nsrdb(
        start, end, res_min, nsrdb_data_dir=nsrdb_dir
    ).reset_index(drop=True)

    load = Electrical_Load.generate_schedules("complex_bayesian", weather, load_seed).reset_index(drop=True)
    pv_cf = np.asarray(Solar_Generation.pv_capacity_factor(weather.copy()), dtype=float)
    prices = grid_electricity_prices.grid_prices_for_series(weather["DateTime"]).reset_index(drop=True)
    events = grid_events.generate_for_datetimes(
        weather["DateTime"], random_seed=event_seed, dr_case=dr_case
    ).reset_index(drop=True)

    # Align lengths: generate_schedules keeps FULL DAYS ONLY, so `load` can be one boundary row
    # shorter than `weather`. Truncate every series to the common length (no-op when all equal).
    n = min(len(weather), len(load), len(pv_cf), len(prices), len(events))

    df = pd.DataFrame({
        "Iteration": np.arange(1, n + 1),
        "DateTime": pd.to_datetime(weather["DateTime"]).values[:n],
        "Ta": weather["Ta"].to_numpy()[:n], "WS": weather["WS"].to_numpy()[:n], "RH": weather["RH"].to_numpy()[:n],
        "GHI": weather["GHI"].to_numpy()[:n], "DHI": weather["DHI"].to_numpy()[:n], "DNI": weather["DNI"].to_numpy()[:n],
        "k_t": weather["k_t"].to_numpy()[:n], "SolarTime": weather["SolarTime"].to_numpy()[:n],
        "Cloud": weather["Cloud"].to_numpy()[:n], "DayStatus": weather["DayStatus"].to_numpy()[:n],
        "PV Capacity Factor": pv_cf[:n],
        "Grid Electricity Buy Price": prices["Grid Electricity Buy Price"].to_numpy()[:n],
        "Grid Electricity Sell Price": prices["Grid Electricity Sell Price"].to_numpy()[:n],
        "Grid Outage Event": events["Grid Outage Event"].to_numpy()[:n],
        "DR Event": events["DR Event"].to_numpy()[:n],
        "Available Grid Import Capacity": events["Available Grid Import Capacity"].to_numpy()[:n],
        "E_Load": load["Total Load"].to_numpy()[:n],
        "Occupancy": load["Total Occupancy"].to_numpy()[:n],
        "Internal Heat Gain (Electrical Load)": load["Total Heat Gain"].to_numpy()[:n],
    })
    for c in FORECAST_COLS:      # forecast-free controllers -> unused; 0.0 for schema compatibility
        df[c] = 0.0
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    ap.add_argument("--out-dir", default="../data", help="RL repo data/ folder")
    ap.add_argument("--dr-case", default="moderate", choices=["moderate", "severe"])
    ap.add_argument("--nsrdb-dir", default="NSRDB_data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    todo = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    for name, years in todo.items():
        print(f"[{name}] years={years} dr_case={args.dr_case} ...", flush=True)
        df = build_split(years, LOAD_SEED, EVENT_SEED[name], dr_case=args.dr_case, nsrdb_dir=args.nsrdb_dir)
        out = os.path.join(args.out_dir, OUT_NAMES[name])
        df.to_csv(out, index=False)
        n_dr = int((df["DR Event"] > 0).sum()); n_out = int((df["Grid Outage Event"] > 0).sum())
        print(f"[{name}] wrote {out}  rows={len(df)}  DR steps={n_dr}  outage steps={n_out}", flush=True)


if __name__ == "__main__":
    main()
