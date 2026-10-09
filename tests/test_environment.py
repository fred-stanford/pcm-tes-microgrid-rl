"""
Integration tests for GridTiedEnv (runs the real hp_tes model).

Covers: obs shape, info keys, single-battery SOC bounds, no-TES forces hp_action=0,
and that DR/outage rows never produce grid import (resp. import+export).
"""
import os
import numpy as np
import pandas as pd
import pytest

import Input_Parameters as IP
from microgrid.observation import ObsBuilder
from microgrid.reward import GridTiedReward
from microgrid.environment import GridTiedEnv

DATA = "data/test_new_0605.csv"
COEFFS = "data/Calibration_Model_Input.xlsx"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(DATA) and os.path.exists(COEFFS)),
    reason="grid-tied data / coefficients not available",
)

_INFO_KEYS = [
    "grid_purchase_kWh", "grid_sale_kWh", "curtailment_kWh", "loss_of_load_kWh",
    "price_buy", "price_sell", "hp_elec_kWh", "fmu_action", "battery_action",
    "call_for_heating", "call_for_cooling", "pcm_hot_soc_pre", "pcm_cold_soc_pre",
]


def _make_env(enable_tes=True, cap=IP.BATTERY_TES_KWH, seed=0, system_type="hp_tes"):
    df = pd.read_csv(DATA)
    coeffs = pd.read_excel(COEFFS, sheet_name=None)
    return GridTiedEnv(
        data_df=df, model_coefficients=coeffs, episode_days=1, seed=seed,
        obs_builder=ObsBuilder(), reward_class=GridTiedReward,
        battery_capacity=cap, pv_size=IP.PVSize, enable_tes=enable_tes, system_type=system_type,
    )


def test_reset_and_obs_shape():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == (19,)   # ObsBuilder.N: 17 state values + DR notice + PSPS warning
    assert np.all(np.isfinite(obs))


def test_step_info_and_soc_bounds():
    env = _make_env()
    env.reset()
    for _ in range(4):
        a = int(env.action_space.sample())
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (19,)
        assert np.isfinite(r)
        assert all(k in info for k in _INFO_KEYS)
        soc_min = IP.battery_soc_min_kwh(IP.BATTERY_TES_KWH)
        assert soc_min - 1e-6 <= env.state["Battery SOC"] <= IP.BATTERY_TES_KWH + 1e-6


def test_no_tes_forces_hp_standby():
    env = _make_env(enable_tes=False, cap=IP.BATTERY_NO_TES_KWH, seed=1, system_type="hp_only")
    env.reset()
    for _ in range(6):
        _, _, _, _, info = env.step(int(env.action_space.sample()))
        assert info["fmu_action"] == 0


def test_outage_blocks_grid_and_dr_is_price_event():
    # M6/M12: events are sampled at RUNTIME. An OUTAGE hard-blocks import+export; a DR event keeps
    # the grid UP but raises the buy price (critical-peak), so import is allowed (not blocked) during DR.
    env = _make_env()
    env.reset()

    # Force an outage realization and step at an active-outage index.
    ev = env.sample_events(seed=0, force="outage")
    assert np.any(ev["outage"] > 0)
    env.t = int(np.argmax(ev["outage"] > 0))
    _, _, _, _, info = env.step(int(env.action_space.sample()))
    assert info["grid_purchase_kWh"] == 0.0 and info["grid_sale_kWh"] == 0.0  # outage blocks both

    # Put the episode window in summer, force a DR, and step at an active-DR index.
    dts = pd.to_datetime(env.data_all["DateTime"])
    july = np.flatnonzero((dts.dt.month == 7).to_numpy())
    if july.size == 0:
        pytest.skip("no summer rows in data")
    env.reset()
    env.t0 = int(july[0]); env.t = 0; env.iteration = 1
    ev = env.sample_events(seed=1, force="dr")
    if not np.any(ev["dr"] > 0):
        pytest.skip("no DR placed in the sampled window")
    env.t = int(np.argmax(ev["dr"] > 0))
    _, _, _, _, info = env.step(int(env.action_space.sample()))
    assert int(info["Grid Outage Event"]) == 0   # grid stays UP during DR
    assert int(info["DR Event"]) == 1
    assert info["dr_price_adder"] > 0.0          # DR is a critical-peak PRICE event
