"""
Unit tests for GridTiedReward (pure, no hp_tes).

reward = price_sell*sale - price_buy*purchase - discomfort - loss_penalty - infeasible_penalty
"""
import Input_Parameters as IP
from microgrid.reward import GridTiedReward

EPS = 1e-6
COMFORT_TI = 23.5  # inside [T_low, T_high] -> no discomfort


def _rew(state, info):
    return GridTiedReward(state, info=info, dt_s=IP.δt).compute()


def test_pure_sale():
    assert abs(_rew({"Indoor Temperature": COMFORT_TI},
                    {"price_sell": 0.2, "grid_sale_kWh": 1.0}) - 0.2) < EPS


def test_pure_purchase():
    assert abs(_rew({"Indoor Temperature": COMFORT_TI},
                    {"price_buy": 0.3, "grid_purchase_kWh": 1.0}) + 0.3) < EPS


def test_asymmetric_buy_sell():
    v = _rew({"Indoor Temperature": COMFORT_TI},
             {"price_buy": 0.3, "grid_purchase_kWh": 2.0,
              "price_sell": 0.1, "grid_sale_kWh": 1.0})
    assert abs(v - (0.1 * 1.0 - 0.3 * 2.0)) < EPS


def test_discomfort_above():
    dt_h = IP.δt / 3600.0
    Ti = IP.COMFORT_BAND_HIGH + 1.5   # 1.5 C above the comfort-band ceiling
    v = _rew({"Indoor Temperature": Ti}, {})
    assert abs(v - (-IP.DISCOMFORT_USD_PER_DEGC_PER_H * 1.5 * dt_h)) < 1e-5


def test_loss_of_load():
    v = _rew({"Indoor Temperature": COMFORT_TI}, {"loss_of_load_kWh": 0.5})
    assert abs(v - (-IP.LOSS_OF_LOAD_PENALTY_USD_PER_KWH * 0.5)) < EPS


def test_infeasible_heating_charge_cold():
    v = _rew({"Indoor Temperature": COMFORT_TI},
             {"call_for_heating": True, "fmu_action": 2})
    assert abs(v + IP.INFEASIBLE_ACTION_PENALTY) < EPS


def test_infeasible_cooling_charge_hot():
    v = _rew({"Indoor Temperature": COMFORT_TI},
             {"call_for_cooling": True, "fmu_action": 1})
    assert abs(v + IP.INFEASIBLE_ACTION_PENALTY) < EPS


def test_infeasible_use_tes_empty_pcm():
    v = _rew({"Indoor Temperature": COMFORT_TI},
             {"call_for_heating": True, "fmu_action": 3, "pcm_hot_soc_pre": 0.0})
    assert abs(v + IP.INFEASIBLE_NO_PCM_PENALTY) < EPS


def test_use_tes_with_full_pcm_no_penalty():
    v = _rew({"Indoor Temperature": COMFORT_TI},
             {"call_for_heating": True, "fmu_action": 3, "pcm_hot_soc_pre": 0.8})
    assert abs(v) < EPS


def test_components_sum_to_total():
    info = {"price_buy": 0.3, "grid_purchase_kWh": 1.0,
            "price_sell": 0.1, "grid_sale_kWh": 0.5,
            "loss_of_load_kWh": 0.2, "call_for_heating": True, "fmu_action": 2}
    total = _rew({"Indoor Temperature": 26.0}, info)
    comp = (info["rew_grid_sale"] + info["rew_grid_purchase"] + info["rew_comfort"]
            + info["rew_loss_of_load"] + info["rew_infeasible"])
    assert abs(total - comp) < EPS
    assert abs(info["reward_total"] - total) < EPS
