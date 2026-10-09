"""
microgrid.config — scenarios, env/agent factories, run naming + settings dump.

The 2x2 study:

         | TES system        | no-TES (2x cost-equivalent battery)
  -------+-------------------+------------------------------------
  RBC    | A                 | C
  RL     | B                 | D

Shared by microgrid.train (trains B, D) and microgrid.eval (evaluates A-D).
"""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import Input_Parameters as IP
from microgrid.agents.dqn import DQNConfig

# ── Data (5-min resolution, provided by the PI) ───────────────────────────────
DATA_DIR = "data"

# C3 fix: each split now carries an INDEPENDENT stochastic event calendar (distinct seed per
# split), generated directly by `data_generation_pipeline/generate_dataset.py` into the base
# filenames — so 2023-selection no longer leaks the 2024 event schedule. This supersedes the
# `*_ev1.csv` + `physics/event_regenerator.py` stopgap (retired). Set USE_INDEPENDENT_EVENTS=True
# only to fall back to any legacy `*_ev1.csv` files.
USE_INDEPENDENT_EVENTS = False


def _data_csv(base):
    ev = f"{DATA_DIR}/{base}_ev1.csv"
    orig = f"{DATA_DIR}/{base}.csv"
    return ev if (USE_INDEPENDENT_EVENTS and os.path.exists(ev)) else orig


TRAIN_CSV = _data_csv("train_new_0605")        # train years (2018-2022)
VAL_CSV = _data_csv("validation_new_0605")     # validation year (2023)
TEST_CSV = _data_csv("test_new_0605")          # test year (2024)
COEFFS = f"{DATA_DIR}/Calibration_Model_Input.xlsx"

# ── Outputs ───────────────────────────────────────────────────────────────────
RUNS_DIR = "runs"          # runs/<run_name>/{model.zip, episode_log.csv, config.json}
TESTS_DIR = "tests"        # runs/<run_name>/tests/ — test-year eval outputs (metrics, traces, RBC/DQN CSVs)

# RL acts every 15 min (3 substeps of 5 min); 20 discrete actions.
ACTION_REPEAT = 3
NUM_ACTIONS = 20

# Hyperparameter-sweep grid (single source of truth for train_sweep + study).
# Discount factors chosen by effective planning horizon ~ 1/(1-gamma) decisions x 15 min:
#   0.98  -> ~12.5 h (~half day: sub-daily, below the full diurnal peak cycle)
#   0.99  -> ~25 h   (~1 day: the dominant diurnal TOU/PV/PCM/battery cycle; primary value)
#   0.9965-> ~71 h   (~3 days: multi-day pre-positioning ahead of DR / a PSPS + its day-ahead notice)
# (Brackets the diurnal cycle from below (half day) and above (multi-day). Earlier grids
#  [0.9,0.99,0.995] and [0.7,0.95,0.99] showed too-short horizons under-exploit load shifting.)
SWEEP_LAYERS = [2, 3]
SWEEP_GAMMAS = [0.98, 0.99, 0.9965]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str                     # "A".."D", "B_nopcm"
    label: str
    system_type: str              # "hp_tes" | "hp_only"
    enable_tes: bool
    control: str                  # "RL" | "RBC"
    battery_capacity_kwh: float
    pv_size_kw: float = IP.PVSize
    rbc_baseline: str = None      # override the RBC scenario to compare against (A1 ablation)


SCENARIOS = {
    "A": ScenarioConfig("A", "TES + RBC",    "hp_tes",  True,  "RBC", IP.BATTERY_TES_KWH),
    "B": ScenarioConfig("B", "TES + RL",     "hp_tes",  True,  "RL",  IP.BATTERY_TES_KWH),
    "C": ScenarioConfig("C", "no-TES + RBC", "hp_only", False, "RBC", IP.BATTERY_NO_TES_KWH),
    "D": ScenarioConfig("D", "no-TES + RL",  "hp_only", False, "RL",  IP.BATTERY_NO_TES_KWH),
    # A1 ablation: SAME plant + SAME battery as B, but the controller cannot operate the PCM
    # (HP mode locked to standby via enable_tes=False; the HP still serves comfort directly through
    # the safety override, so thermal load is met — the agent just can't pre-store thermal energy).
    # Isolates the value of RL-operated PCM from battery arbitrage. Compared vs the deployed RBC (A).
    "B_nopcm": ScenarioConfig("B_nopcm", "TES-disabled + RL (ablation)", "hp_tes", False, "RL",
                              IP.BATTERY_TES_KWH, rbc_baseline="A"),
}
RL_SCENARIOS = [s for s in SCENARIOS.values() if s.control == "RL"]    # B, D
RBC_SCENARIOS = [s for s in SCENARIOS.values() if s.control == "RBC"]  # A, C


# ── Env / agent factories ─────────────────────────────────────────────────────
def build_env(scn, data_df, coeffs, *, episode_days=7, seed=0,
              action_repeat=ACTION_REPEAT, switching_cost=0.0, on_substep=None):
    """Build the wrapped (15-min macro-step) env for a scenario."""
    from microgrid.environment import GridTiedEnv
    from microgrid.observation import ObsBuilder
    from microgrid.reward import GridTiedReward
    from microgrid.wrappers import ActionRepeat

    env = GridTiedEnv(
        data_df=data_df, model_coefficients=coeffs, episode_days=episode_days, seed=seed,
        obs_builder=ObsBuilder(), reward_class=GridTiedReward,
        battery_capacity=scn.battery_capacity_kwh, pv_size=scn.pv_size_kw,
        enable_tes=scn.enable_tes, system_type=scn.system_type,
    )
    return ActionRepeat(env, repeat=action_repeat, switching_cost=switching_cost,
                        on_substep=on_substep)


def build_agent(scn, *, num_actions=NUM_ACTIONS, dqn_config=None):
    """Build the RBC or DQN agent for a scenario (both are microgrid.agents.Agent)."""
    if scn.control == "RBC":
        from microgrid.agents.rbc import RBCAgent
        return RBCAgent(num_actions=num_actions, system_type=scn.system_type)

    from microgrid.agents.dqn import DQNAgent
    return DQNAgent(config=dqn_config or DQNConfig(), num_actions=num_actions)


# ── Run naming + settings dump ────────────────────────────────────────────────
def run_name(scn, dqn: DQNConfig, episode_days: int, tag: str = None) -> str:
    """e.g. B_g0.99_lr0.0001_3x128_500k_ep7_s0  (+ _<tag> if given)."""
    name = (f"{scn.name}_g{dqn.gamma:g}_lr{dqn.learning_rate:g}"
            f"_{dqn.hidden_layers}x{dqn.hidden_units}_{dqn.total_timesteps // 1000}k"
            f"_ep{episode_days}_s{dqn.seed}")
    return f"{name}_{tag}" if tag else name


def run_dir(name: str) -> str:
    return os.path.join(RUNS_DIR, name)


def dump_config(out_dir: str, scn, dqn: DQNConfig, episode_days: int, *, timestamp=None):
    """Write config.json capturing everything needed to reproduce a run."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(),
        "scenario": {
            "name": scn.name, "label": scn.label, "system_type": scn.system_type,
            "enable_tes": scn.enable_tes, "control": scn.control,
            "battery_capacity_kwh": scn.battery_capacity_kwh,
            "pv_size_kw": scn.pv_size_kw,
            "inverter_size_kw": scn.battery_capacity_kwh / 2.0,
        },
        "dqn": asdict(dqn),
        "rollout": {
            "episode_days": episode_days, "action_repeat": ACTION_REPEAT,
            "num_actions": NUM_ACTIONS, "physics_dt_s": IP.δt,
        },
        "data": {"train": TRAIN_CSV, "validation": VAL_CSV, "test": TEST_CSV, "coeffs": COEFFS},
        "reward_params": {
            "DISCOMFORT_USD_PER_DEGC_PER_H": IP.DISCOMFORT_USD_PER_DEGC_PER_H,
            "LOSS_OF_LOAD_PENALTY_USD_PER_KWH": IP.LOSS_OF_LOAD_PENALTY_USD_PER_KWH,
            "BATTERY_DEGRADATION_USD_PER_KWH": IP.BATTERY_DEGRADATION_USD_PER_KWH,
            "INFEASIBLE_ACTION_PENALTY": IP.INFEASIBLE_ACTION_PENALTY,
            "INFEASIBLE_NO_PCM_PENALTY": IP.INFEASIBLE_NO_PCM_PENALTY,
            "T_low": IP.T_low, "T_high": IP.T_high,
            "COMFORT_BAND_LOW": IP.COMFORT_BAND_LOW, "COMFORT_BAND_HIGH": IP.COMFORT_BAND_HIGH,
        },
        "physical": {
            "PVSize_kw": IP.PVSize, "BatteryLoss_per_hr": IP.BatteryLoss,
            "eta_PV2B": IP.η_PV2B, "eta_B2H": IP.η_B2H,
        },
        # ObsBuilder normalization constants (hardcoded in microgrid/observation.py) —
        # snapshotted here for reproducibility.
        "obs_normalization": {
            "e_load_divisor_kw": IP.InverterSize,     # obs[4] = E_Load / InverterSize (global 5.4)
            "price_buy_divisor": 1.15, "price_sell_divisor": 1.05,
            "Ti_center": 23.5, "Ti_scale": 5.0, "Ta_center": 15.0, "Ta_scale": 15.0,
            "last_action_divisor": 19.0,
        },
    }
    path = os.path.join(out_dir, "config.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
