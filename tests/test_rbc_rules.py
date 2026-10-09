"""Unit tests for RBC battery / HP decisions (no data files required)."""
import Input_Parameters as IP
from physics.rbc_rules import (
    BAT_FULL_CHARGE,
    BAT_FULL_DISCHARGE,
    BAT_MEDIUM_CHARGE,
    BAT_NATURAL,
    BAT_MEDIUM_DISCHARGE,
    FMU_CHARGE_HOT,
    FMU_STANDBY,
    FMU_USE_TES,
    SystemType,
    decide_battery_action,
    decide_fmu_action,
)


def _grid_tied(cap, f, **kw):
    """decide_battery_action for a grid-tied step at usable-SOC fraction f.

    Defaults: off-peak summer, night (no current PV) -> the forecast-free pre-charge can fire.
    """
    soc = IP.battery_soc_min_kwh(cap) + f * IP.battery_usable_capacity_kwh(cap)
    base = dict(pv_kw=0.0, e_load_kw=0.5, hp_kw=0.0, b_soc=soc,
                allow_import=True, allow_export=True, battery_capacity=cap,
                price_buy=0.323, price_sell=0.089, is_peak_season=True, solar_now=False)
    base.update(kw)
    return decide_battery_action(**base)


def test_grid_tied_never_grid_charges_at_peak():
    """Key fix (C4): at the TOU peak price the RBC must NOT force grid charging (old strawman)."""
    assert _grid_tied(10.0, 0.2, price_buy=0.527) == BAT_NATURAL          # low SOC, peak
    assert _grid_tied(10.0, 0.5, price_buy=0.527) == BAT_NATURAL          # mid SOC, peak


def test_grid_tied_offpeak_precharge_at_night():
    """Cheap off-peak in the peak season + low SOC + no current PV (night) -> pre-charge."""
    assert _grid_tied(10.0, 0.2, price_buy=0.323, solar_now=False) == BAT_MEDIUM_CHARGE


def test_grid_tied_no_precharge_when_solar_present():
    """Current PV available (daytime) -> rely on solar (forecast-free), do not grid pre-charge."""
    assert _grid_tied(10.0, 0.2, price_buy=0.323, solar_now=True) == BAT_NATURAL


def test_grid_tied_no_precharge_off_peak_season():
    """Flat-price season (no evening peak) -> no arbitrage value -> natural."""
    assert _grid_tied(10.0, 0.2, price_buy=0.30, is_peak_season=False) == BAT_NATURAL


def test_grid_tied_export_arbitrage_when_sell_exceeds_buy():
    """Rare EEC spike (sell > buy) with SOC above reserve -> discharge to export."""
    assert _grid_tied(10.0, 0.7, price_sell=0.50) in (BAT_FULL_DISCHARGE, BAT_MEDIUM_DISCHARGE)
    # ...but keep the reserve: below the reserve fraction, do not drain for export
    assert _grid_tied(10.0, 0.15, price_sell=0.50, is_peak_season=False, price_buy=0.40) == BAT_NATURAL


def test_islanded_deficit_discharges_even_when_full():
    cap = 10.0
    assert decide_battery_action(
        pv_kw=0.5, e_load_kw=2.0, hp_kw=1.5, b_soc=cap * 0.99,
        allow_import=False, allow_export=False, battery_capacity=cap,
        heating_call=True,
    ) == BAT_FULL_DISCHARGE


def test_islanded_no_charge_on_deficit():
    cap = 10.0
    action = decide_battery_action(
        pv_kw=0.0, e_load_kw=2.0, hp_kw=1.0, b_soc=5.0,
        allow_import=False, allow_export=False, battery_capacity=cap,
        heating_call=True,
    )
    assert action in (BAT_FULL_DISCHARGE, BAT_MEDIUM_DISCHARGE)


def test_islanded_skips_pcm_charge_on_thin_surplus():
    cap = 10.0
    hot_max, _ = 5.0, 5.0
    action = decide_fmu_action(
        SystemType.HP_TES,
        heating_call=True, cooling_call=False, season="heating",
        pv_kw=1.5, e_load_kw=1.0, b_soc=cap, pcm_h_kwh=0.0, pcm_c_kwh=0.0,
        allow_import=False, battery_capacity=cap,
    )
    assert action == FMU_STANDBY


def test_islanded_uses_tes_when_available():
    cap = 10.0
    hot_max, _ = sum(IP.PCM_H_Energy_Split), sum(IP.PCM_C_Energy_Split)
    action = decide_fmu_action(
        SystemType.HP_TES,
        heating_call=True, cooling_call=False, season="heating",
        pv_kw=0.2, e_load_kw=1.0, b_soc=cap * 0.5,
        pcm_h_kwh=0.5 * hot_max, pcm_c_kwh=0.0,
        allow_import=False, battery_capacity=cap,
    )
    assert action == FMU_USE_TES
