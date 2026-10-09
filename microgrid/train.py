"""
microgrid.train — train an RL agent (scenario B = TES, D = no-TES) with SB3 DQN.

The agent acts every 15 min (3-substep rollout over the 5-min physics). Unspecified
hyperparameters use defaults; pass a flag to override. Each run gets its own folder
runs/<run_name>/ containing model.zip, episode_log.csv and config.json (full settings).

Examples:
  python -m microgrid.train --scenario B
  python -m microgrid.train --scenario B --gamma 0.99 --lr 1e-4 --layers 3 --units 128 --tag exp1
  python -m microgrid.train --scenario D --timesteps 800000 --seed 1
  python -m microgrid.train --scenario B --smoke
"""
import argparse
import warnings

import pandas as pd

import microgrid.config as C
from microgrid.agents.dqn import DQNConfig, DQNAgent
from microgrid.agents.logger import EpisodeCsvLogger

warnings.filterwarnings("ignore")

# FR5: fixed event seed for the validation env used to pick the best checkpoint. A single fixed
# realization (same every evaluation, at the TRUE event rate) makes the selection metric stable and
# representative — see train_one() where the eval env is configured.
VAL_EVENT_SEED = 2023


def train_one(
    scenario: str,
    *,
    gamma: float = 0.99,
    lr: float = 1e-4,
    layers: int = 3,
    units: int = 128,
    timesteps: int = 500_000,
    episode_days: int = 7,
    seed: int = 0,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    learning_starts: int = 5_000,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.1,
    tag: str = None,
    smoke: bool = False,
    base_dir: str = None,
) -> str:
    """Train a single RL agent. Returns the run name (subfolder under runs/)."""
    scn = C.SCENARIOS[scenario]
    timesteps = 2_000 if smoke else timesteps
    episode_days = 1 if smoke else episode_days
    if smoke:
        tag = f"{tag}_smoke" if tag else "smoke"

    dqn = DQNConfig(
        base_dir=base_dir or C.RUNS_DIR,
        gamma=gamma, learning_rate=lr,
        hidden_layers=layers, hidden_units=units,
        total_timesteps=timesteps, seed=seed,
        eval_freq=(500 if smoke else 100_000),
        buffer_size=buffer_size, batch_size=batch_size,
        learning_starts=learning_starts,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
    )
    dqn.run_name = C.run_name(scn, dqn, episode_days, tag=tag)

    print(f"\n{'='*64}\n  TRAIN {scn.name} ({scn.label}) -> {dqn.run_dir}\n"
          f"  gamma={gamma}  layers={layers}  battery={scn.battery_capacity_kwh} kWh  "
          f"enable_tes={scn.enable_tes}  steps={timesteps:,}\n{'='*64}")

    cfg_path = C.dump_config(dqn.run_dir, scn, dqn, episode_days)
    print(f"  settings -> {cfg_path}")

    df = pd.read_csv(C.TRAIN_CSV)
    coeffs = pd.read_excel(C.COEFFS, sheet_name=None)
    env = C.build_env(scn, df, coeffs, episode_days=episode_days, seed=seed)

    # Held-out validation env for EvalCallback best-model selection (M2). Full validation year
    # (episode_days=365 -> deterministic t0=0 rollout, so the callback's metric matches
    # eval_scenario's total_reward). Tiny for smoke to keep the plumbing check fast.
    val_df = pd.read_csv(C.VAL_CSV)
    eval_env = C.build_env(scn, val_df, coeffs, episode_days=(1 if smoke else 365), seed=seed)
    # FR5: pick the best checkpoint on the SAME distribution we ultimately report on. The *training*
    # env deliberately over-samples grid events (enrichment) so the agent sees enough of them to
    # learn; but the *validation* env used to choose the best checkpoint must use the TRUE event rate
    # (force=None) with a FIXED realization (VAL_EVENT_SEED) — so a higher validation score reflects a
    # genuinely better policy, not a luckier/rarer event draw, and matches how eval_scenario scores.
    _eval_base = eval_env.unwrapped
    _eval_base.event_force = None
    _eval_base.event_reset_seed = VAL_EVENT_SEED

    agent = DQNAgent(config=dqn, num_actions=C.NUM_ACTIONS)
    agent.build(env)
    logger = EpisodeCsvLogger(dqn.episode_log_path)
    agent.train(env, callback=logger, total_timesteps=timesteps, eval_env=eval_env)
    print(f"  done -> {dqn.model_path} (best-on-val -> {dqn.best_model_path})")
    return dqn.run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["B", "D"], help="RL scenario to train")
    # hyperparameter overrides (defaults match DQNConfig)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=500_000, help="15-min decisions")
    ap.add_argument("--episode-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer-size", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-starts", type=int, default=5_000)
    ap.add_argument("--exploration-fraction", type=float, default=0.3)
    ap.add_argument("--exploration-final-eps", type=float, default=0.1)
    ap.add_argument("--tag", default=None, help="suffix appended to the run name")
    ap.add_argument("--smoke", action="store_true", help="tiny run (2k steps, 1-day episodes)")
    args = ap.parse_args()

    train_one(
        args.scenario,
        gamma=args.gamma, lr=args.lr, layers=args.layers, units=args.units,
        timesteps=args.timesteps, episode_days=args.episode_days, seed=args.seed,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tag=args.tag, smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
