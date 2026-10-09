"""
microgrid.validate — evaluate all 6 hyperparameter-sweep agents on the validation year.

Loads the trained models from train_sweep, runs each on the validation CSV (2023),
ranks them by net grid cost, and writes a comparison table + best-agent summary.

Examples:
  python -m microgrid.validate --scenario B
  python -m microgrid.validate --scenario B --days 30
  python -m microgrid.validate --scenario B --tag mystudy
  python -m microgrid.validate --scenario B --smoke   # match smoke-trained runs
"""
import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import pandas as pd

import microgrid.config as C
from microgrid.agents.dqn import DQNConfig
from microgrid.eval import METRIC_COLS, REWARD_COLS, eval_scenario
from microgrid.train_sweep import GAMMAS, LAYERS

warnings.filterwarnings("ignore")

VAL_METRIC_COLS = ["run_name", "layers", "gamma", "rank"] + [
    c for c in METRIC_COLS if c not in ("scenario", "label")
] + REWARD_COLS


def _log(msg=""):
    print(msg, flush=True)


def _expected_run_name(scenario, *, layers, gamma, episode_days, timesteps, seed, lr, units, tag):
    dqn = DQNConfig(
        gamma=gamma, learning_rate=lr,
        hidden_layers=layers, hidden_units=units,
        total_timesteps=timesteps, seed=seed,
    )
    scn = C.SCENARIOS[scenario]
    return C.run_name(scn, dqn, episode_days, tag=tag)


def discover_runs(scenario, *, episode_days, timesteps, seed, lr, units, tag, smoke):
    """Return list of {run_name, layers, gamma} from sweep_summary.json or expected names."""
    summary_path = os.path.join(C.RUNS_DIR, f"sweep_{scenario}", "sweep_summary.json")
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))
        if summary.get("scenario") != scenario:
            raise ValueError(f"sweep summary scenario {summary['scenario']!r} != {scenario!r}")
        if tag and summary.get("tag") != tag:
            raise ValueError(f"--tag {tag!r} does not match sweep summary tag {summary.get('tag')!r}")
        if smoke and not summary.get("smoke"):
            raise ValueError("sweep summary was not a smoke run; drop --smoke or retrain")
        return summary["runs"]

    if smoke:
        timesteps = 2_000
        episode_days = 1
        tag = f"{tag}_smoke" if tag else "smoke"

    runs = []
    for layers in LAYERS:
        for gamma in GAMMAS:
            run_name = _expected_run_name(
                scenario, layers=layers, gamma=gamma,
                episode_days=episode_days, timesteps=timesteps,
                seed=seed, lr=lr, units=units, tag=tag,
            )
            runs.append({"run_name": run_name, "layers": layers, "gamma": gamma})
    return runs


def _resolve_model(run_name, prefer_best=True):
    """Path to the model to evaluate. Prefer the best-on-validation model (EvalCallback, M2);
    fall back to the final-step model.zip if no best_model.zip exists (e.g. eval disabled)."""
    d = os.path.join(C.RUNS_DIR, run_name)
    best = os.path.join(d, "best_model.zip")
    final = os.path.join(d, "model.zip")
    if prefer_best and os.path.exists(best):
        return best
    if os.path.exists(final):
        return final
    raise FileNotFoundError(f"missing model for run {run_name!r}: {best} / {final}")


def val_best_final_gap(run_name):
    """Best-vs-final validation gap from EvalCallback's evaluations.npz (M2). Zero extra compute
    (the validation curve was logged during training). Returns None if the file is absent."""
    import numpy as np
    p = os.path.join(C.RUNS_DIR, run_name, "evaluations.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    r = d["results"].mean(axis=1)          # per-eval validation reward (mean over eval episodes)
    ts = d["timesteps"]
    i = int(np.argmax(r))
    return {"best": float(r[i]), "final": float(r[-1]), "gap": float(r[i] - r[-1]),
            "best_step": int(ts[i]), "final_step": int(ts[-1])}


def print_reward_summary(rows):
    """Compact reward breakdown per agent (sorted by rank)."""
    _log("\n  Reward breakdown (validation year):")
    hdr = (f"  {'rank':<5}{'run':<36}{'total':>10}"
           f"{'import':>10}{'export':>10}{'comfort':>10}{'lol':>8}{'infeas':>10}")
    _log(hdr)
    _log("  " + "-" * (len(hdr) - 2))
    for r in rows:
        _log(f"  {r['rank']:<5}{r['run_name']:<36}"
             f"{r['total_reward']:>10.1f}"
             f"{r['rew_grid_purchase']:>10.1f}"
             f"{r['rew_grid_sale']:>10.1f}"
             f"{r['rew_comfort']:>10.1f}"
             f"{r['rew_loss_of_load']:>8.1f}"
             f"{r['rew_infeasible']:>10.1f}")


def print_ranking(rows):
    """Print a compact ranked table (higher total_reward is better = the training objective)."""
    metrics = [
        ("total_reward", "reward ($, train obj)", "{:.1f}"),
        ("net_grid_cost", "net grid cost ($)", "{:.2f}"),
        ("import_cost", "import cost ($)", "{:.2f}"),
        ("export_revenue", "export revenue ($)", "{:.2f}"),
        ("grid_import_kWh", "grid import (kWh)", "{:.1f}"),
        ("grid_export_kWh", "grid export (kWh)", "{:.1f}"),
        ("discomfort_degC_h", "discomfort (degC.h)", "{:.1f}"),
        ("loss_of_load_kWh", "loss of load (kWh)", "{:.2f}"),
        ("battery_throughput_kWh", "battery throughput (kWh)", "{:.1f}"),
        ("peak_import_kW", "peak import (kW)", "{:.2f}"),
    ]
    W = 30
    head = f"\n{'rank':<5}{'run':<36}{'layers':>6}{'gamma':>6}"
    for _, label, _ in metrics:
        head += f"{label:>16}"
    _log(head)
    _log("-" * len(head.strip("\n")))
    for r in rows:
        line = f"{r['rank']:<5}{r['run_name']:<36}{r['layers']:>6}{r['gamma']:>6.2f}"
        for key, _, fmt in metrics:
            line += f"{fmt.format(float(r[key])):>16}"
        _log(line)
    best = rows[0]
    _log(f"\n  BEST: {best['run_name']}  (layers={best['layers']}, gamma={best['gamma']}, "
         f"reward={best['total_reward']:.1f}, net_grid_cost=${best['net_grid_cost']:.2f})\n")


def main():
    ap = argparse.ArgumentParser(
        description="Validate all 6 sweep agents on the validation year and rank them",
    )
    ap.add_argument("--scenario", required=True, choices=["B", "D"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--episode-days", type=int, default=7,
                    help="episode length used during training (for run-name discovery)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="must match the tag used in train_sweep")
    ap.add_argument("--days", type=int, default=None, help="evaluate only the first N days")
    ap.add_argument("--trace", action="store_true", help="also dump per-decision trace CSVs")
    ap.add_argument("--smoke", action="store_true", help="match smoke-trained sweep runs")
    args = ap.parse_args()

    _log(f"Starting validation (scenario {args.scenario})...")

    runs = discover_runs(
        args.scenario,
        episode_days=args.episode_days, timesteps=args.timesteps,
        seed=args.seed, lr=args.lr, units=args.units,
        tag=args.tag, smoke=args.smoke,
    )

    _log(f"Loading {C.VAL_CSV} and model coefficients...")
    df = pd.read_csv(C.VAL_CSV)
    coeffs = pd.read_excel(C.COEFFS, sheet_name=None)
    max_dec = None if args.days is None else args.days * (288 // C.ACTION_REPEAT)
    out_dir = os.path.join(C.RUNS_DIR, f"sweep_{args.scenario}", "validation")
    os.makedirs(out_dir, exist_ok=True)

    _log(f"\n{'#'*64}")
    _log(f"  VALIDATION - {len(runs)} agents on {C.VAL_CSV}")
    if args.days:
        _log(f"  first {args.days} days only")
    else:
        _log(f"  full validation year (~{365 * (288 // C.ACTION_REPEAT)} decisions per agent)")
    _log(f"  output -> {out_dir}")
    for entry in runs:
        _log(f"    - {entry['run_name']}  (layers={entry['layers']}, gamma={entry['gamma']})")
    _log(f"{'#'*64}")

    scn = C.SCENARIOS[args.scenario]
    rows = []
    for i, entry in enumerate(runs, 1):
        run_name = entry["run_name"]
        _log(f"\n>>> [{i}/{len(runs)}] {run_name}  (layers={entry['layers']}, gamma={entry['gamma']})")
        _log("  loading model...")
        model_path = _resolve_model(run_name)
        m, tr, _ = eval_scenario(
            scn, df, coeffs, model_path=model_path,
            max_decisions=max_dec, trace=args.trace, seed=args.seed,
            progress=True,
        )
        row = {
            "run_name": run_name,
            "layers": entry["layers"],
            "gamma": entry["gamma"],
            **{k: m[k] for k in m if k not in ("scenario", "label")},
        }
        rows.append(row)
        _log(f"  done - net grid cost=${m['net_grid_cost']:.2f} | "
             f"import {m['grid_import_kWh']:.1f} kWh | export {m['grid_export_kWh']:.1f} kWh | "
             f"discomfort {m['discomfort_degC_h']:.1f} degC.h")
        _log(f"  reward={m['total_reward']:.1f}  "
             f"(import {m['rew_grid_purchase']:.1f}, export +{m['rew_grid_sale']:.1f}, "
             f"comfort {m['rew_comfort']:.1f}, lol {m['rew_loss_of_load']:.1f}, "
             f"infeas {m['rew_infeasible']:.1f})")
        if args.trace and tr is not None:
            tr.to_csv(os.path.join(out_dir, f"trace_{run_name}.csv"), index=False)
            _log(f"  trace saved -> trace_{run_name}.csv")

    # Select the agent ALIGNED WITH TRAINING: highest total reward (the full cost objective the
    # DQN optimizes = grid profit - comfort/productivity cost - loss-of-load - infeasibility
    # shaping). net_grid_cost is only a tiebreaker. (Previously ranked on net_grid_cost alone,
    # which mismatched the training objective — see reviewer concern M1.)
    rows.sort(key=lambda r: (-r["total_reward"], r["net_grid_cost"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    print_ranking(rows)
    print_reward_summary(rows)

    # Best-vs-final validation gap (M2): quantifies how much the missing selection step mattered.
    _log("\n  Best-vs-final validation gap (from EvalCallback curve; positive = final overtrains):")
    for r in rows:
        g = val_best_final_gap(r["run_name"])
        if g is not None:
            r["val_best_reward"] = g["best"]
            r["val_final_reward"] = g["final"]
            r["val_best_final_gap"] = g["gap"]
            _log(f"    {r['run_name']:<34} best={g['best']:>9.1f} @{g['best_step']//1000}k  "
                 f"final={g['final']:>9.1f}  gap={g['gap']:>8.1f}")
        else:
            _log(f"    {r['run_name']:<34} (no evaluations.npz — trained without eval callback)")

    metrics_path = os.path.join(out_dir, "validation_metrics.csv")
    pd.DataFrame(rows)[VAL_METRIC_COLS].to_csv(metrics_path, index=False)

    best = rows[0]
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "validation_csv": C.VAL_CSV,
        "days": args.days,
        "ranking_metric": "total_reward (descending; aligned with the training objective)",
        "best_run": {
            "run_name": best["run_name"],
            "layers": best["layers"],
            "gamma": best["gamma"],
            "total_reward": best["total_reward"],
            "net_grid_cost": best["net_grid_cost"],
        },
        "runs": rows,
    }
    summary_path = os.path.join(out_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"Metrics saved to {metrics_path}")
    _log(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
