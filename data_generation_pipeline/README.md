# Data generation pipeline

`generate_dataset.py` builds the 5-minute exogenous time series that drive the simulation and writes
`data/train_new_0605.csv` (2018–2022), `data/validation_new_0605.csv` (2023), and
`data/test_new_0605.csv` (2024). Generation is offline and deterministic (fixed random seeds).

```bash
cd data_generation_pipeline
python generate_dataset.py --split all --out-dir ../data
```

## Inputs
- `NSRDB_data/` (not tracked): NSRDB 5-minute weather for location 57357 (37.46° N, 122.43° W),
  one file per year, named `57357_37.46_-122.43_<year>.csv`, for 2018–2024.
- `Data/TOU.xlsx`: PG&E Schedule E-ELEC time-of-use import rates.
- `Data/EEC.xlsx`: PG&E Energy Export Credit values by month, hour, and day type.
- The tables come from PG&E's Electric Schedule E-ELEC
  (https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_E-ELEC.pdf) and its Solar Billing Plan
  Energy Export Credit price sheets
  (https://www.pge.com/en/clean-energy/solar/getting-started-with-solar/solar-billing-plan.html).

## Modules
- `Data_Conversion.py`: reads the NSRDB files and adds solar time, clearness index, cloud cover, and the
  day/night flag.
- `Solar_Generation.py`: PV capacity factor (pvlib plane-of-array irradiance, Fuentes cell temperature).
- `Electrical_Load.py`: stochastic occupancy and non-HVAC electrical load.
- `grid_electricity_prices.py`: import price and export credit for each time step.
- `grid_events.py`: event columns kept for schema compatibility. The simulation samples grid events at
  run time and ignores these columns.
- `Input_Parameters.py`, `Utility_functions.py`: site constants and solar-geometry helpers.

The datasets carry no forecasts; the four `*Forecasted` columns are zero and unused.
