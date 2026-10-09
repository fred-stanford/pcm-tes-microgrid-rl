"""
Tests for the grid-tied agents: RBCAgent (encode + decide) and the SB3 DQNAgent
(smoke train + predict). The integration tests need the data/coeffs.
"""
import os
import warnings
import pandas as pd
import pytest

import Input_Parameters as IP
from microgrid.agents.rbc import RBCAgent

warnings.filterwarnings("ignore")

DATA = "data/test_new_0605.csv"
COEFFS = "data/Calibration_Model_Input.xlsx"
_has_data = os.path.exists(DATA) and os.path.exists(COEFFS)


def test_rbc_encode_decode_roundtrip():
    ag = RBCAgent()
    for hp in range(4):
        for batt in range(-2, 3):
            a = ag._encode(hp, batt)
            assert 0 <= a <= 19
            assert a // 5 == hp and (a % 5) - 2 == batt


def _wrapped_env(enable_tes=True, cap=IP.BATTERY_TES_KWH, st="hp_tes", seed=0):
    from microgrid.observation import ObsBuilder
    from microgrid.reward import GridTiedReward
    from microgrid.environment import GridTiedEnv
    from microgrid.wrappers import ActionRepeat
    df = pd.read_csv(DATA)
    coeffs = pd.read_excel(COEFFS, sheet_name=None)
    e = GridTiedEnv(
        data_df=df, model_coefficients=coeffs, episode_days=1, seed=seed,
        obs_builder=ObsBuilder(), reward_class=GridTiedReward,
        battery_capacity=cap, pv_size=IP.PVSize, enable_tes=enable_tes, system_type=st,
    )
    e.episode_steps = 6
    return ActionRepeat(e, repeat=3, switching_cost=0.0)


@pytest.mark.skipif(not _has_data, reason="data not available")
def test_rbc_decide_action_valid_on_wrapper():
    env = _wrapped_env()
    env.reset()
    agent = RBCAgent(num_actions=20)
    for _ in range(3):
        a = agent.decide_action(env)
        assert 0 <= a <= 19
        env.step(a)


@pytest.mark.skipif(not _has_data, reason="data not available")
def test_rbc_no_tes_forces_hp_standby():
    env = _wrapped_env(enable_tes=False, cap=IP.BATTERY_NO_TES_KWH, st="hp_only", seed=1)
    env.reset()
    agent = RBCAgent(num_actions=20)
    for _ in range(3):
        a = agent.decide_action(env)
        assert a // 5 == 0  # hp_action forced to standby in hp_only
        env.step(a)


@pytest.mark.skipif(not _has_data, reason="data not available")
def test_dqn_sb3_smoke_train_and_predict(tmp_path):
    from microgrid.agents.dqn import DQNAgent, DQNConfig
    env = _wrapped_env()
    cfg = DQNConfig(run_name="smoke_pytest", base_dir=str(tmp_path),
                    learning_starts=4, buffer_size=300, batch_size=8,
                    train_freq=1, target_update_interval=10)
    agent = DQNAgent(cfg, num_actions=20)
    agent.train(env, total_timesteps=12)
    assert os.path.exists(cfg.model_path)
    env.reset()
    a = agent.decide_action(env)
    assert 0 <= a <= 19
