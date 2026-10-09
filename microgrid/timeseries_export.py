"""
Detailed per-timestep CSV export for test/validation evaluation.

Builds one row per 5-min physics substep with a fixed column schema shared by
RBC and DQN runs (e.g. runs/<run>/tests/RBC.csv and DQN.csv).
"""
from __future__ import annotations

import Input_Parameters as IP
import pandas as pd
from physics.Data_Conversion import calculate_urgency_scores

SUBSTEP_HOURS = IP.δt / 3600.0
FORECAST_HORIZON = 96  # 8 h of 5-min steps

# Fixed column order for RBC and DQN CSVs (same schema).
DETAILED_TRACE_COLUMNS = [
    # ── index / time ──
    "Iteration",
    "DateTime",
    # ── agent actions & grid dispatch ──
    "Action",
    "hp_action",
    "batt_action",
    "imp_kW",
    "exp_kW",
    "grid_import_kWh",
    "grid_export_kWh",
    "p_buy",
    "p_sell",
    "pv",
    "pv_gen_kWh",
    "dr",
    "outage",
    "dr_notice",
    "psps_warning",
    "dr_price_adder",
    # ── reward (5-min substep) ──
    "reward_step",
    "rew_grid_purchase",
    "rew_grid_sale",
    "rew_comfort",
    "rew_loss_of_load",
    "rew_infeasible",
    # ── comfort / storage state ──
    "Indoor Temperature (C)",
    "Ta",
    "Battery SOC (kWh)",
    "battery_soc_frac",
    "FCU Mode",
    "call_for_heating",
    "call_for_cooling",
    "Loss_of_Load (kWh)",
    "Curtailment (kWh)",
    "Battery action (kWh)",
    "hp_elec_kWh",
    # ── loads / urgency ──
    "Net Total Heat Gain (kW)",
    "Total HP-TES Electrical Load (kW)",
    "Electricity Urgency Score",
    "Heating Urgency Score",
    "Cooling Urgency Score",
    "Remaining Heating Energy PCM (kWh)",
    "Remaining Cooling Energy PCM (kWh)",
    "PV Power (kW)",
    "E Load (kW)",
    "Occupancy",
    "Internal Heat Gain (Electrical Load) (kW)",
    # ── passive model ──
    "Conduction (kW)",
    "Convection (kW)",
    "Radiative Heat Gain (kW)",
    "Radiative Cooling (kW)",
    "Passive Heat Transfer (kW)",
    "Internal Heat Gain (kW)",
    # ── HP-TES / FMU ──
    "Time",
    "FMU mode from MPC",
    "Exact System Mode",
    "HP Mode",
    "HP ON OFF",
    "hot pcm not fully discharged",
    "cold pcm not fully discharged",
    "PCM_Cold_SOC",
    "PCM_Hot_SOC",
    "PCM_Cold_Temp",
    "PCM_Hot_Temp",
    "FCU Entering Water Temperature",
    "FCU Leaving Water Temperature",
    "Heat Pump Supply Water Temperature",
    "Heat Pump Return Water Temperature",
    "Heat Pump Compressor Speed",
    "COP Heating",
    "COP Cooling",
    "Fan Coil Fan Mass Flow",
    "Pump 1 Mass Flow",
    "Pump 2 Mass Flow",
    "HP Electric Power",
    "HP Useful Thermal Power",
    "Pump 1 Electric Power",
    "Pump 2 Electric Power",
    "Heat Transfer Cold PCM",
    "Heat Transfer Hot PCM",
    "FCU Heat Delivered",
    "Fan Coil Fan Electric Power",
    "Discharge Water Temp Hot",
    "PCM Water Inlet Hot",
]

_FMU_KEYS = [
    "Time",
    "FMU mode from MPC",
    "Exact System Mode",
    "HP Mode",
    "HP ON OFF",
    "hot pcm not fully discharged",
    "cold pcm not fully discharged",
    "PCM_Cold_SOC",
    "PCM_Hot_SOC",
    "PCM_Cold_Temp",
    "PCM_Hot_Temp",
    "FCU Entering Water Temperature",
    "FCU Leaving Water Temperature",
    "Heat Pump Supply Water Temperature",
    "Heat Pump Return Water Temperature",
    "Heat Pump Compressor Speed",
    "COP Heating",
    "COP Cooling",
    "Fan Coil Fan Mass Flow",
    "Pump 1 Mass Flow",
    "Pump 2 Mass Flow",
    "HP Electric Power",
    "HP Useful Thermal Power",
    "Pump 1 Electric Power",
    "Pump 2 Electric Power",
    "Heat Transfer Cold PCM",
    "Heat Transfer Hot PCM",
    "FCU Heat Delivered",
    "Fan Coil Fan Electric Power",
    "Discharge Water Temp Hot",
    "PCM Water Inlet Hot",
]


def _urgency_scores(base, info):
    """Electricity / heating / cooling urgency from CSV forecast columns."""
    idx = int(info.get("data_idx", max(0, base._idx() - 1)))
    sl = base.data_all.iloc[idx: idx + FORECAST_HORIZON]
    _fcast = ["Total Cooling Load Forecasted", "Total Heating Load Forecasted",
              "Total Electrical Load Forecasted"]
    # Forecast-free operation: the *Forecasted columns are 0, so these legacy urgency scores are
    # undefined -> NaN placeholders (kept only for schema compatibility). Avoids feeding zeros
    # into calculate_urgency_scores.
    if len(sl) < 2 or float(sl[_fcast].abs().to_numpy().sum()) == 0.0:
        return float("nan"), float("nan"), float("nan")

    cooling = sl["Total Cooling Load Forecasted"].astype(float).values
    heating = sl["Total Heating Load Forecasted"].astype(float).values
    passive_fc = {"Total Net Heat Gain": cooling - heating}
    passive_fc = pd.DataFrame(passive_fc)
    e_fc = pd.DataFrame({"Total Load": sl["Total Electrical Load Forecasted"].astype(float).values})
    socs = [
        float(info.get("battery_soc_kWh", info.get("Battery SOC", 0.0))),
        float(info.get("PCM_Hot_E_kWh", 0.0)),
        float(info.get("PCM_Cold_E_kWh", 0.0)),
    ]
    scores = calculate_urgency_scores(passive_fc, e_fc, socs)
    return (
        scores["Elctricity Urgency Score"],
        scores["Heating Urgency Score"],
        scores["Cooling Urgency Score"],
    )


def build_detailed_row(base, macro_action: int, info: dict) -> dict:
    """Assemble one detailed trace row from a single 5-min substep ``info`` dict."""
    fmu = getattr(base, "last_fmu_states", None) or {}
    pv_cf = float(info.get("pv_capacity_factor", 0.0))
    pv_kw = pv_cf * float(base.pv_size)
    gp = float(info.get("grid_purchase_kWh", 0.0))
    gs = float(info.get("grid_sale_kWh", 0.0))
    imp_kw = gp / SUBSTEP_HOURS if SUBSTEP_HOURS > 0 else 0.0
    exp_kw = gs / SUBSTEP_HOURS if SUBSTEP_HOURS > 0 else 0.0
    hp_kwh = float(info.get("hp_elec_kWh", 0.0))
    hp_kw = hp_kwh / SUBSTEP_HOURS if SUBSTEP_HOURS > 0 else 0.0
    urg_e, urg_h, urg_c = _urgency_scores(base, info)
    soc_kwh = float(info.get("battery_soc_kWh", 0.0))

    out = {
        "Iteration": int(info.get("Iteration", 0)),
        "DateTime": info.get("DateTime"),
        "Action": int(macro_action),
        "hp_action": int(info.get("fmu_action", 0)),
        "batt_action": int(info.get("battery_action", 0)),
        "imp_kW": imp_kw,
        "exp_kW": exp_kw,
        "grid_import_kWh": gp,
        "grid_export_kWh": gs,
        "p_buy": float(info.get("price_buy", 0.0)),
        "p_sell": float(info.get("price_sell", 0.0)),
        "pv": pv_cf,
        "pv_gen_kWh": float(info.get("pv_gen_kWh", 0.0)),
        "dr": int(info.get("DR Event", 0)),
        "outage": int(info.get("Grid Outage Event", 0)),
        "dr_notice": float(info.get("DR Notice", 0.0)),
        "psps_warning": float(info.get("PSPS Warning", 0.0)),
        "dr_price_adder": float(info.get("dr_price_adder", 0.0)),
        "reward_step": float(info.get("reward_total", 0.0)),
        "rew_grid_purchase": float(info.get("rew_grid_purchase", 0.0)),
        "rew_grid_sale": float(info.get("rew_grid_sale", 0.0)),
        "rew_comfort": float(info.get("rew_comfort", 0.0)),
        "rew_loss_of_load": float(info.get("rew_loss_of_load", 0.0)),
        "rew_infeasible": float(info.get("rew_infeasible", 0.0)),
        "Indoor Temperature (C)": float(info.get("Indoor Temperature", 0.0)),
        "Ta": float(info.get("ambient_temp_C", 0.0)),
        "Battery SOC (kWh)": soc_kwh,
        "battery_soc_frac": float(info.get("battery_soc_frac", IP.battery_physical_soc_frac(
            soc_kwh, base.battery_capacity))),
        "FCU Mode": int(info.get("FCU Mode", 0)),
        "call_for_heating": int(bool(info.get("call_for_heating", False))),
        "call_for_cooling": int(bool(info.get("call_for_cooling", False))),
        "Loss_of_Load (kWh)": float(info.get("loss_of_load_kWh", 0.0)),
        "Curtailment (kWh)": float(info.get("curtailment_kWh", 0.0)),
        "Battery action (kWh)": float(info.get("battery_throughput_kWh", 0.0)),
        "hp_elec_kWh": hp_kwh,
        "Net Total Heat Gain (kW)": float(info.get("Net Total Heat Gain", 0.0)),
        "Total HP-TES Electrical Load (kW)": hp_kw,
        "Electricity Urgency Score": urg_e,
        "Heating Urgency Score": urg_h,
        "Cooling Urgency Score": urg_c,
        "Remaining Heating Energy PCM (kWh)": float(info.get("PCM_Hot_E_kWh", 0.0)),
        "Remaining Cooling Energy PCM (kWh)": float(info.get("PCM_Cold_E_kWh", 0.0)),
        "PV Power (kW)": pv_kw,
        "E Load (kW)": float(info.get("e_load_kw", 0.0)),
        "Occupancy": float(info.get("Occupancy", 0.0)),
        "Internal Heat Gain (Electrical Load) (kW)": float(
            info.get("internal_heat_gain_elec_kw", 0.0)
        ),
        "Conduction (kW)": float(info.get("Conduction", 0.0)),
        "Convection (kW)": float(info.get("Convection", 0.0)),
        "Radiative Heat Gain (kW)": float(info.get("Radiative Heat Gain", 0.0)),
        "Radiative Cooling (kW)": float(info.get("Radiative Cooling", 0.0)),
        "Passive Heat Transfer (kW)": float(info.get("Passive Heat Transfer", 0.0)),
        "Internal Heat Gain (kW)": float(info.get("Internal Heat Gain", 0.0)),
    }
    for key in _FMU_KEYS:
        out[key] = fmu.get(key, float("nan"))
    return out
