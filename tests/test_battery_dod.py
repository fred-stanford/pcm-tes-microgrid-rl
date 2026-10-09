"""Unit tests for BatteryDoD limits in loadmatch and SOC helpers."""
import pytest

import Input_Parameters as IP
from physics.loadmatch import gridtrading_events


def test_battery_soc_helpers():
    cap = 10.0
    assert IP.battery_soc_min_kwh(cap) == pytest.approx(2.0)
    assert IP.battery_usable_capacity_kwh(cap) == pytest.approx(8.0)
    assert IP.battery_usable_soc_frac(2.0, cap) == pytest.approx(0.0)
    assert IP.battery_usable_soc_frac(10.0, cap) == pytest.approx(1.0)
    assert IP.battery_usable_soc_frac(6.0, cap) == pytest.approx(0.5)
    assert IP.battery_physical_soc_frac(2.0, cap) == pytest.approx(0.2)
    assert IP.battery_physical_soc_frac(10.0, cap) == pytest.approx(1.0)


def test_discharge_stops_at_dod_floor():
    cap = 10.0
    soc_min = IP.battery_soc_min_kwh(cap)
    delta_t_hr = IP.δt / 3600.0
    max_step = IP.InverterSize * delta_t_hr

    b_new, _, _, _, _ = gridtrading_events(
        Gen=0.0,
        E_Load=max_step * 2.0,
        M_Load=0.0,
        B_SOC=soc_min,
        battery_action=-2,
        DR_event=0,
        Grid_outage=0,
        battery_capacity=cap,
    )
    assert b_new == pytest.approx(soc_min, abs=1e-3)


def test_discharge_from_full_usable_range():
    cap = 10.0
    soc_min = IP.battery_soc_min_kwh(cap)
    delta_t_hr = IP.δt / 3600.0
    max_step = IP.InverterSize * delta_t_hr

    b_new, _, _, _, _ = gridtrading_events(
        Gen=0.0,
        E_Load=max_step * IP.η_B2H,
        M_Load=0.0,
        B_SOC=cap,
        battery_action=-2,
        DR_event=0,
        Grid_outage=0,
        battery_capacity=cap,
    )
    assert b_new >= soc_min
    assert b_new < cap
