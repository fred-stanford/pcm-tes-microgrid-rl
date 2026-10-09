"""
Environment_GridTiedV2.py
-------------------------
Grid-tied V2 blackbox (single battery, DR/outage aware) for the 2x2 study.

Promotes the colleague's `changes/Environment_GridTied (1).py` into the repo, with:
  * UNIT FIX (critical): Gen is now energy [kWh] = PV Capacity Factor * pv_size * δt_hr.
    The original wrote `Gen = CF * IP.PVSize` (power [kW]) and passed it to
    gridtrading_events, which expects kWh (E_Load and HP_TES_energy are already
    converted). Without the `* δt_hr`, PV was overestimated by 12x (δt = 5 min).
  * Per-scenario capacity args `battery_capacity` / `pv_size` / `inverter_size`
    (no global IP mutation), threaded into LoadMatchV2.gridtrading_events.
  * Uses LoadMatchV2 (leaves legacy LoadMatch.py untouched).

Buy/sell prices are NOT used here (the electrical balance is price-agnostic); the
environment reads them from the row and hands them to the reward.

Returns (single battery):
  new_states, fmu_result_timeseries, fmu_states,
  grid_purchase [kWh], grid_sale [kWh], curtailment [kWh], loss_of_load [kWh]
"""

import os
import sys
import numpy as np
import pandas as pd

import Input_Parameters as IP
from physics import Passive_Model
from physics import loadmatch as LoadMatch
from physics import hp_tes
from physics.passive_single_step import passive_heat_transfer_single_step


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout.close()
        finally:
            sys.stdout = self._original_stdout


def _row_to_weather_row(row) -> pd.DataFrame:
    Ta = float(row["Ta"])
    return pd.DataFrame({
        "Ta": [Ta],
        "WS": [float(row.get("WS", 0.0))],
        "RH": [float(row.get("RH", 50.0))],
        "GHI": [float(row.get("GHI", 0.0))],
        "DateTime": [row.get("DateTime")],
        "k_t": [float(row.get("k_t", 0.0))],
        "SolarTime": [float(row.get("SolarTime", 12.0))],
        "Cloud": [float(row.get("Cloud", 0.0))],
        "DayStatus": [float(row.get("DayStatus", 1.0))]
    })


def Fast_Backbox_GridTied(iteration, fmu_action, battery_action, row, model,
                          past_states, model_coefficients,
                          battery_capacity=None, pv_size=None, inverter_size=None):
    """Grid-tied V2 blackbox with DR/outage via LoadMatchV2.gridtrading_events.

    Parameters
    ----------
    battery_action : int
        Suggested action in {-2, -1, 0, 1, 2}.
    row : Series-like
        ``PV Capacity Factor`` [-] and ``E_Load`` [kW]; ``DR Event`` /
        ``Grid Outage Event`` (0/1). Buy/sell prices are read by the environment.
    battery_capacity : float, optional
        Pack size [kWh]; defaults to IP.BatterySize. Per-scenario override.
    pv_size : float, optional
        PV nameplate [kW]; defaults to IP.PVSize. Per-scenario override.
    inverter_size : float, optional
        Inverter power [kW]; defaults to IP.InverterSize.

    Returns
    -------
    new_states : dict (single "Battery SOC")
    fmu_result_timeseries
    fmu_states : dict
    grid_purchase : float [kWh]
    grid_sale : float [kWh]
    curtailment : float [kWh]
    loss_of_load : float [kWh]
    """
    FCU_mode = past_states["FCU Mode"]
    Ti = past_states["Indoor Temperature"]
    B_SOC = past_states["Battery SOC"]

    delta_t_hr = IP.δt / 3600.0

    if pv_size is None:
        pv_size = IP.PVSize

    E_Load = float(row["E_Load"]) * delta_t_hr  # [kW] -> [kWh]
    Ta = float(row["Ta"])
    Internal_Heat_Gain = float(row.get("Internal Heat Gain (Electrical Load)", 0.0))

    weather_row = _row_to_weather_row(row)
    passive_dict = passive_heat_transfer_single_step(
        model_coefficients, weather_row, float(Ti), Internal_Heat_Gain
    )
    Q_Passive = passive_dict["Passive Heat Transfer"]

    # UNIT FIX: PV Capacity Factor [-] * nameplate [kW] * δt_hr [h] -> energy [kWh]
    Gen = float(row["PV Capacity Factor"]) * pv_size * delta_t_hr
    dr_event, grid_outage = row["DR Event"], row["Grid Outage Event"]

    T_start = IP.δt * (iteration - 1)
    T_end = IP.δt * iteration
    time_points = np.linspace(T_start, T_end, IP.T_interval)

    input_variables_dict = {
        "schedule_input": int(fmu_action),
        "mpc_enable": 1,
        "room_temperature": 273.15 + float(Ti),
        "outside_air_temperature": 273.15 + float(Ta),
    }

    with HiddenPrints():
        fmu_result_timeseries, fmu_states = hp_tes.system.run_python_hp_tes(
            iteration, input_variables_dict, time_points, model
        )

    Active_Heating = fmu_states["FCU Heat Delivered"]
    Q_active = float(Active_Heating) + float(Internal_Heat_Gain)

    # fmu power outputs are [kW] (cleaned_output divides W by 1000) -> * δt_hr = [kWh]
    HP_TES_energy = (
        float(fmu_states["HP Electric Power"])
        + float(fmu_states["Pump 1 Electric Power"])
        + float(fmu_states["Pump 2 Electric Power"])
        + float(fmu_states["Fan Coil Fan Electric Power"])
    ) * delta_t_hr

    Ti_new, FCU_mode_new = Passive_Model.passive_model_reduced(
        float(Q_Passive), float(Ti), float(Q_active), int(FCU_mode)
    )

    B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load = LoadMatch.gridtrading_events(
        Gen,
        E_Load,
        HP_TES_energy,
        B_SOC,
        int(battery_action),
        dr_event,
        grid_outage,
        battery_capacity=battery_capacity,
        inverter_size=inverter_size,
    )

    new_states = {
        "Indoor Temperature": float(Ti_new),
        "T_HP": float(fmu_states["Heat Pump Supply Water Temperature"]),
        "compressor_speed": float(fmu_states["Heat Pump Compressor Speed"]),
        "FCU Mode": int(FCU_mode_new),
        "Battery SOC": float(B_SOC_new)
    }

    return (
        new_states,
        fmu_result_timeseries,
        fmu_states,
        float(grid_purchase),
        float(grid_sale),
        float(curtailment),
        float(loss_of_load),
        passive_dict,
    )
