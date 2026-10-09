"""
microgrid.train_sweep — train a grid of RL agents with varying hyperparameters.

Runs 6 agents sequentially (layers ∈ {2, 3} × gamma ∈ {0.7, 0.95, 0.99}), each saved
to its own subfolder under runs/. All other hyperparameters match microgrid.train defaults.

Examples:
  python -m microgrid.train_sweep --scenario B
  python -m microgrid.train_sweep --scenario B --smoke
  python -m microgrid.train_sweep --scenario D --tag mystudy
"""
import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import microgrid.config as C
from microgrid.train import train_one

warnings.filterwarnings("ignore")

# Single source of truth lives in microgrid.config (kept importable without SB3).
LAYERS = C.SWEEP_LAYERS
GAMMAS = C.SWEEP_GAMMAS


def main():
    ap = argparse.ArgumentParser(
        description="Train 6 DQN agents (layers 2/3 × gamma 0.7/0.95/0.99) sequentially",
    )
    ap.add_argument("--scenario", required=True, choices=["B", "D"], help="RL scenario to train")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=500_000, help="15-min decisions per agent")
    ap.add_argument("--episode-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer-size", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-starts", type=int, default=5_000)
    ap.add_argument("--exploration-fraction", type=float, default=0.3)
    ap.add_argument("--exploration-final-eps", type=float, default=0.1)
    ap.add_argument("--tag", default=None, help="suffix appended to every run name")
    ap.add_argument("--smoke", action="store_true", help="tiny run (2k steps, 1-day episodes)")
    args = ap.parse_args()

    combos = [(layers, gamma) for layers in LAYERS for gamma in GAMMAS]

    print(f"\n{'#'*64}")
    print(f"  HYPERPARAMETER SWEEP — {len(combos)} agents")
    print(f"  scenario={args.scenario}  layers={LAYERS}  gamma={GAMMAS}")
    if args.tag:
        print(f"  tag={args.tag}")
    if args.smoke:
        print("  mode=smoke")
    print(f"{'#'*64}")

    completed = []
    for i, (layers, gamma) in enumerate(combos, 1):
        print(f"\n>>> Agent {i}/{len(combos)}: layers={layers}, gamma={gamma}")
        run_name = train_one(
            args.scenario,
            gamma=gamma, lr=args.lr, layers=layers, units=args.units,
            timesteps=args.timesteps, episode_days=args.episode_days, seed=args.seed,
            buffer_size=args.buffer_size, batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            tag=args.tag, smoke=args.smoke,
        )
        completed.append({"layers": layers, "gamma": gamma, "run_name": run_name})

    summary_dir = os.path.join(C.RUNS_DIR, f"sweep_{args.scenario}")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "sweep_summary.json")
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "layers": LAYERS,
        "gammas": GAMMAS,
        "tag": args.tag,
        "smoke": args.smoke,
        "runs": completed,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*64}")
    print(f"  SWEEP COMPLETE — {len(completed)} agents trained")
    for entry in completed:
        print(f"    layers={entry['layers']}  gamma={entry['gamma']}  -> runs/{entry['run_name']}")
    print(f"  summary -> {summary_path}")
    print(f"{'#'*64}\n")


if __name__ == "__main__":
    main()
