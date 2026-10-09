"""
passive_single_step.py
----------------------
Self-contained implementation of passive_heat_transfer_single_step for dynamic passive mode.
Bundled in RL-microgrid_framework so dynamic passive works when the parent Passive_Model
does not include this function (e.g. older versions).

Requires: Utility_functions, Input_Parameters (same as the rest of the framework).
"""

import pandas as pd
import numpy as np

try:
    from physics import Utility_functions
    import Input_Parameters
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

_cols_for_clustering_SHGC = ["WS", "RH", "Ta", "Cloud", "GHI", "k_t"]
_cols_for_clustering_RCC = ["WS", "RH", "Ta", "Cloud"]
_bounds = {
    "WS": (0.0, 10.0),
    "RH": (0.0, 100.0),
    "Ta": (0.0, 40.0),
    "GHI": (0.0, 1000.0),
    "Cloud": (0.0, 1.0),
    "k_t": (0.0, 1.0),
}


def passive_heat_transfer_single_step(model_coefficients, weather_row, Ti, internal_heat_gain=0.0):
    """
    Compute passive heat transfer for a single timestep using actual indoor temperature Ti.
    Use this for step-by-step simulation where Ti evolves; avoids constant-Ti assumption.

    Inputs:
        model_coefficients: dict of model coefficients (from Calibration_Model_Input.xlsx)
        weather_row: single-row DataFrame with Ta, WS, RH, GHI, DateTime, k_t, SolarTime, Cloud, DayStatus
        Ti: current indoor temperature [°C]
        internal_heat_gain: internal heat gain [kW], default 0

    Returns:
        dict with keys: Conduction, Convection, Radiative Heat Gain, Radiative Cooling,
        Passive Heat Transfer, Internal Heat Gain
    """
    if not _DEPS_OK:
        raise ImportError(
            "passive_single_step requires Utility_functions and Input_Parameters. "
            "Add the parent RL folder to sys.path or PYTHONPATH."
        )

    SHGCs = model_coefficients["Tuned Parameters"]["_SHGC_"]
    RCCs_day = model_coefficients["Tuned Parameters"]["_RCC_day_"]
    RCCs_night = model_coefficients["Tuned Parameters"]["_RCC_night_"]

    day_centroids_SHGC_df = model_coefficients["SHGC Day Centroids"]
    day_centroids_RCC_df = model_coefficients["RCC Day Centroids"]
    night_centroids_RCC_df = model_coefficients["RCC Night Centroids"]

    day_centroids_SHGC = day_centroids_SHGC_df.to_numpy().T
    day_centroids_RCC = day_centroids_RCC_df.to_numpy().T
    night_centroids_RCC = night_centroids_RCC_df.to_numpy().T

    Ta = float(weather_row["Ta"].iloc[0])
    WS = float(weather_row["WS"].iloc[0])
    RH = float(weather_row["RH"].iloc[0])
    GHI = float(weather_row["GHI"].iloc[0])
    local_time = weather_row["DateTime"].iloc[0]
    # ensure datetime (CSV may store DateTime as string)
    if not hasattr(local_time, "timetuple"):
        local_time = pd.to_datetime(local_time).to_pydatetime()
    k_t = float(weather_row["k_t"].iloc[0])
    solar_time = float(weather_row["SolarTime"].iloc[0])
    day_status = float(weather_row["DayStatus"].iloc[0])

    q_cd = Utility_functions.Q_conduction(Ta, Ti) / 1000
    q_cv = Utility_functions.Q_convection(WS, Ta, Ti) / 1000
    q_rc = Utility_functions.Q_radiativeCooling(Ti, Ta, RH, GHI, solar_time, day_status) / 1000
    q_rh = Utility_functions.calculate_solarheatgain(solar_time, GHI, k_t, local_time) / 1000

    new_row = pd.DataFrame({
        "WS": [WS], "RH": [RH], "Ta": [Ta], "GHI": [GHI],
        "Cloud": [weather_row["Cloud"].iloc[0]], "k_t": [k_t],
    })

    if day_status == 0.0:
        normalized_new_row_rcc = Utility_functions.normalize_row(
            new_row, _cols_for_clustering_RCC, _bounds
        )
        nearest_rcc = Utility_functions.find_nearest_centroid(
            normalized_new_row_rcc, night_centroids_RCC, _cols_for_clustering_RCC
        )
        q_radcool = q_rc * RCCs_night[nearest_rcc]
        q_radheat = 0.0
    else:
        normalized_new_row_shgc = Utility_functions.normalize_row(
            new_row, _cols_for_clustering_SHGC, _bounds
        )
        normalized_new_row_rcc = Utility_functions.normalize_row(
            new_row, _cols_for_clustering_RCC, _bounds
        )
        nearest_shgc = Utility_functions.find_nearest_centroid(
            normalized_new_row_shgc, day_centroids_SHGC, _cols_for_clustering_SHGC
        )
        nearest_rcc = Utility_functions.find_nearest_centroid(
            normalized_new_row_rcc, day_centroids_RCC, _cols_for_clustering_RCC
        )
        q_radcool = q_rc * RCCs_day[nearest_rcc]
        q_radheat = q_rh * SHGCs[nearest_shgc]

    Q_passive = q_cd + q_cv + q_radheat - q_radcool
    return {
        "Conduction": q_cd,
        "Convection": q_cv,
        "Radiative Heat Gain": q_radheat,
        "Radiative Cooling": q_radcool,
        "Passive Heat Transfer": Q_passive,
        "Internal Heat Gain": internal_heat_gain,
    }
