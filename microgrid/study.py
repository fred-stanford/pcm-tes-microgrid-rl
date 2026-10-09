"""
microgrid.study — multi-seed study for a robust RL-vs-RBC comparison.

Why: a single trained agent (one seed) can't tell you whether the RL advantage is a
property of the *method* or of *that one lucky/unlucky training run*. This module runs
the whole pipeline across several independent training seeds and reports the RL
performance as **mean ± std across the per-seed winners** vs the deterministic RBC.

Design (built for a SLURM cluster, e.g. Stanford Sherlock):
  * The atomic unit is ONE (seed, layers, gamma) training run — see `microgrid.train`.
  * SEEDS × LAYERS × GAMMAS training runs are launched as a SLURM **array** (all parallel).
  * `validate` picks ONE winner per seed (best on the validation year, 2023),
    exactly the same select-then-report pipeline as the single-seed study — repeated
    per seed so the selection step itself is inside the randomness (gold standard).
  * `aggregate` evaluates the per-seed winners on the test year (2024), reports
    mean ± std (+ min/max) vs the cached RBC baseline, and saves the **median** winner's
    detailed trace for the timeseries figures.

Data seed stays fixed (same CSVs / same world for every run); only the DQN *training*
seed varies, so the whole error bar comes cleanly from the learner.

Subcommands
-----------
  jobs      print the (index -> seed,layers,gamma) map + the array size for SLURM
  train     train ONE combo by array index:   --scenario B --index $SLURM_ARRAY_TASK_ID
  validate  pick the winner for ONE seed:      --scenario B --seed-index $SLURM_ARRAY_TASK_ID
  aggregate collect per-seed winners -> test-year mean ± std vs RBC

See sherlock/README.md for the end-to-end Sherlock workflow.
"""
import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import microgrid.config as C
from microgrid.agents.dqn import DQNConfig
# NOTE: microgrid.train / stable_baselines3 are imported lazily inside cmd_train so that
# `jobs`, `validate` and `aggregate` (which only need names / pandas) work without SB3.

warnings.filterwarnings("ignore")

# Independent DQN training seeds. Data seed is fixed (see module docstring).
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 10 seeds (bumped from 5 for statistical power — see final review)
LAYERS = C.SWEEP_LAYERS
GAMMAS = C.SWEEP_GAMMAS

# Training defaults — MUST match microgrid.train so run names line up for discovery.
DEF_TIMESTEPS = 500_000
DEF_EPISODE_DAYS = 7
DEF_LR = 1e-4
DEF_UNITS = 128


# ── job map ───────────────────────────────────────────────────────────────────
def combos(seeds=SEEDS):
    """Ordered (seed-major) list of training combos for SLURM array indexing."""
    out, idx = [], 0
    for seed in seeds:
        for layers in LAYERS:
            for gamma in GAMMAS:
                out.append({"index": idx, "seed": seed, "layers": layers, "gamma": gamma})
                idx += 1
    return out


def _smoke_params(smoke, timesteps, episode_days, tag):
    """Replicate microgrid.train's smoke transform so expected run names match."""
    if smoke:
        return 2_000, 1, (f"{tag}_smoke" if tag else "smoke")
    return timesteps, episode_days, tag


def expected_run_name(scenario, *, layers, gamma, seed,
                      timesteps=DEF_TIMESTEPS, episode_days=DEF_EPISODE_DAYS,
                      lr=DEF_LR, units=DEF_UNITS, tag=None, smoke=False):
    timesteps, episode_days, tag = _smoke_params(smoke, timesteps, episode_days, tag)
    dqn = DQNConfig(gamma=gamma, learning_rate=lr, hidden_layers=layers,
                    hidden_units=units, total_timesteps=timesteps, seed=seed)
    return C.run_name(C.SCENARIOS[scenario], dqn, episode_days, tag=tag)


def study_dir(scenario):
    return os.path.join(C.RUNS_DIR, f"study_{scenario}")


def export_learning_curve(run_name):
    """Convert a run's evaluations.npz (M2 validation curve) to a plot-friendly
    runs/<run>/learning_curve.csv and return it (None if the run wasn't eval-logged)."""
    d = os.path.join(C.RUNS_DIR, run_name)
    npz = os.path.join(d, "evaluations.npz")
    if not os.path.exists(npz):
        return None
    data = np.load(npz)
    res = data["results"]  # (n_evals, n_eval_episodes)
    curve = pd.DataFrame({
        "timestep": data["timesteps"],
        "val_reward_mean": res.mean(axis=1),
        "val_reward_std": res.std(axis=1),
        "val_reward_min": res.min(axis=1),
        "val_reward_max": res.max(axis=1),
    })
    if "ep_lengths" in data:
        curve["ep_len_mean"] = data["ep_lengths"].mean(axis=1)
    curve.to_csv(os.path.join(d, "learning_curve.csv"), index=False)
    return curve


def _seeds_from_args(args):
    if getattr(args, "seeds", None):
        return [int(s) for s in str(args.seeds).split(",")]
    return SEEDS


# ── subcommand: jobs ────────────────────────────────────────────────────────────
def cmd_jobs(args):
    seeds = _seeds_from_args(args)
    cs = combos(seeds)
    if args.count:
        print(len(cs))
        return
    print(f"# scenario={args.scenario}  seeds={seeds}  layers={LAYERS}  gammas={GAMMAS}")
    print(f"# {len(cs)} training jobs  ->  SLURM: --array=0-{len(cs) - 1}")
    print(f"{'idx':>4}  {'seed':>4}  {'layers':>6}  {'gamma':>5}   run_name")
    for c in cs:
        rn = expected_run_name(args.scenario, layers=c["layers"], gamma=c["gamma"],
                               seed=c["seed"], timesteps=args.timesteps,
                               episode_days=args.episode_days, tag=args.tag, smoke=args.smoke)
        print(f"{c['index']:>4}  {c['seed']:>4}  {c['layers']:>6}  {c['gamma']:>5}   {rn}")


# ── subcommand: train (one combo by array index) ────────────────────────────────
def cmd_train(args):
    seeds = _seeds_from_args(args)
    cs = combos(seeds)
    if not (0 <= args.index < len(cs)):
        raise SystemExit(f"--index {args.index} out of range 0..{len(cs) - 1} (for seeds={seeds})")
    c = cs[args.index]
    print(f">>> study train job {args.index}: seed={c['seed']} layers={c['layers']} gamma={c['gamma']}")
    if args.dry_run:
        print("   (dry run - not training)")
        return
    from microgrid.train import train_one  # lazy: pulls stable_baselines3
    train_one(
        args.scenario,
        gamma=c["gamma"], lr=args.lr, layers=c["layers"], units=args.units,
        timesteps=args.timesteps, episode_days=args.episode_days, seed=c["seed"],
        tag=args.tag, smoke=args.smoke,
    )


# ── subcommand: validate (winner for one seed) ──────────────────────────────────
def _resolve_model(run_name, prefer_best=True):
    """Prefer the best-on-validation model (EvalCallback, M2); fall back to final model.zip."""
    d = os.path.join(C.RUNS_DIR, run_name)
    best = os.path.join(d, "best_model.zip")
    final = os.path.join(d, "model.zip")
    if prefer_best and os.path.exists(best):
        return best
    if os.path.exists(final):
        return final
    raise FileNotFoundError(f"missing model for {run_name!r}: {best} / {final} - has training finished?")


def validate_seed(scenario, seed, df_val, coeffs, *, timesteps, episode_days,
                  lr, units, tag, smoke, max_decisions=None):
    """Rank the 6 agents of one seed on the validation year; return ranked rows + out dir."""
    from microgrid.eval import eval_scenario  # local import: pulls env/SB3 deps
    scn = C.SCENARIOS[scenario]
    runs = [{"run_name": expected_run_name(scenario, layers=l, gamma=g, seed=seed,
                                           timesteps=timesteps, episode_days=episode_days,
                                           lr=lr, units=units, tag=tag, smoke=smoke),
             "layers": l, "gamma": g}
            for l in LAYERS for g in GAMMAS]
    rows = []
    for i, e in enumerate(runs, 1):
        print(f"  [{i}/{len(runs)}] seed {seed}: {e['run_name']}", flush=True)
        m, _, _ = eval_scenario(scn, df_val, coeffs, model_path=_resolve_model(e["run_name"]),
                                max_decisions=max_decisions, seed=seed, progress=False)
        rows.append({"seed": seed, "run_name": e["run_name"], "layers": e["layers"],
                     "gamma": e["gamma"],
                     **{k: v for k, v in m.items() if k not in ("scenario", "label")}})
    # Select ALIGNED WITH TRAINING: highest total reward (net_grid_cost is a tiebreaker only).
    rows.sort(key=lambda r: (-r["total_reward"], r["net_grid_cost"]))
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    out = os.path.join(study_dir(scenario), f"seed_{seed}")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "validation_metrics.csv"), index=False)
    winner = rows[0]
    json.dump({"seed": seed, "run_name": winner["run_name"], "layers": winner["layers"],
               "gamma": winner["gamma"], "total_reward": winner["total_reward"],
               "net_grid_cost": winner["net_grid_cost"],
               "timestamp_utc": datetime.now(timezone.utc).isoformat()},
              open(os.path.join(out, "winner.json"), "w"), indent=2)
    print(f"  seed {seed} winner: {winner['run_name']} "
          f"(layers={winner['layers']}, gamma={winner['gamma']}, "
          f"val reward={winner['total_reward']:.1f}, net cost=${winner['net_grid_cost']:.2f})")
    # Save the winning agent's validation-year timeseries (5-min detailed + per-decision) so the
    # selected agent's behaviour on 2023 can be plotted offline (one extra rollout).
    _wm, wtr, wdet = eval_scenario(scn, df_val, coeffs, model_path=_resolve_model(winner["run_name"]),
                                   max_decisions=max_decisions, trace=True, detailed_trace=True,
                                   seed=seed, progress=False)
    if wdet is not None:
        wdet.to_csv(os.path.join(out, "winner_val_detailed.csv"), index=False)
    if wtr is not None:
        wtr.to_csv(os.path.join(out, "winner_val_trace.csv"), index=False)
    return rows, out


def cmd_validate(args):
    seeds = _seeds_from_args(args)
    if args.seed is not None:
        seed = args.seed
    elif args.seed_index is not None:
        if not (0 <= args.seed_index < len(seeds)):
            raise SystemExit(f"--seed-index {args.seed_index} out of range 0..{len(seeds) - 1}")
        seed = seeds[args.seed_index]
    else:
        raise SystemExit("pass --seed S or --seed-index N")
    df = pd.read_csv(C.VAL_CSV)
    coeffs = pd.read_excel(C.COEFFS, sheet_name=None)
    max_dec = None if args.days is None else args.days * (288 // C.ACTION_REPEAT)
    validate_seed(args.scenario, seed, df, coeffs,
                  timesteps=args.timesteps, episode_days=args.episode_days,
                  lr=args.lr, units=args.units, tag=args.tag, smoke=args.smoke,
                  max_decisions=max_dec)


# ── subcommand: aggregate (per-seed winners -> test-year mean ± std vs RBC) ──────
AGG_METRICS = [c for c in [
    "net_grid_cost", "import_cost", "export_revenue", "grid_import_kWh", "grid_export_kWh",
    "curtailment_kWh", "loss_of_load_kWh", "loss_during_outage_kWh", "import_during_dr_kWh",
    "discomfort_degC_h", "battery_throughput_kWh", "battery_degradation_usd",
    "pcm_throughput_kWh", "hp_elec_kWh",
    "hp_elec_peak_price_kWh", "peak_import_kW", "peak_export_kW",
]]


def cmd_aggregate(args):
    from microgrid.eval import eval_scenario, load_cached_baseline, save_cached_baseline
    seeds = _seeds_from_args(args)
    scn = C.SCENARIOS[args.scenario]
    sdir = study_dir(args.scenario)

    # 1) collect per-seed winners
    winners = []
    for s in seeds:
        wp = os.path.join(sdir, f"seed_{s}", "winner.json")
        if not os.path.exists(wp):
            raise SystemExit(f"missing winner for seed {s}: {wp} - run `study validate` first")
        winners.append(json.load(open(wp)))

    df = pd.read_csv(C.TEST_CSV)
    coeffs = pd.read_excel(C.COEFFS, sheet_name=None)
    max_dec = None if args.days is None else args.days * (288 // C.ACTION_REPEAT)

    # 2) RBC baseline on the test year (deterministic — reuse the eval cache).
    #    `rbc_baseline` lets an ablation (e.g. B_nopcm) compare against the deployed RBC (A).
    rbc_scn = C.SCENARIOS.get(scn.rbc_baseline) or (C.SCENARIOS["A"] if scn.enable_tes else C.SCENARIOS["C"])
    cached = None if (max_dec is not None or args.refresh_baseline) else load_cached_baseline(rbc_scn)
    if cached is not None:
        rbc_m = cached[0]
        print(f"RBC baseline (cached): net cost=${rbc_m['net_grid_cost']:.2f}")
    else:
        print("Evaluating RBC baseline on test year...")
        rbc_m, rbc_tr, rbc_det = eval_scenario(rbc_scn, df, coeffs, max_decisions=max_dec,
                                               trace=True, detailed_trace=True, progress=True)
        if max_dec is None:
            save_cached_baseline(rbc_scn, rbc_m, rbc_tr, detailed_df=rbc_det)

    # 3) evaluate each winner on the test year — save BOTH the per-decision trace (seed-spread band)
    #    and the full 5-min detailed timeseries, so EVERY agent can be plotted offline.
    agg_out = os.path.join(sdir, "aggregate")
    os.makedirs(agg_out, exist_ok=True)
    per_seed = []
    detailed_by_seed = {}
    for w in winners:
        print(f"\n=== seed {w['seed']} winner {w['run_name']} on test year ===")
        m, tr, det = eval_scenario(scn, df, coeffs,
                                   model_path=_resolve_model(w["run_name"]),
                                   max_decisions=max_dec, trace=True, detailed_trace=True,
                                   progress=True)
        per_seed.append({"seed": w["seed"], "run_name": w["run_name"],
                         "layers": w["layers"], "gamma": w["gamma"],
                         **{k: m[k] for k in AGG_METRICS}})
        if tr is not None:
            tr.to_csv(os.path.join(agg_out, f"trace_seed{w['seed']}.csv"), index=False)
        if det is not None:
            det.to_csv(os.path.join(agg_out, f"detailed_seed{w['seed']}.csv"), index=False)
            detailed_by_seed[int(w["seed"])] = det
        print(f"  net cost=${m['net_grid_cost']:.2f}")

    ps = pd.DataFrame(per_seed).sort_values("net_grid_cost").reset_index(drop=True)

    # 4) median winner -> canonical DQN.csv (reuse its already-saved detailed trace, no re-eval)
    #    + RBC.csv, so plots_study's DQN-vs-RBC figures work out of the box.
    med = ps.iloc[len(ps) // 2]
    med_seed = int(med["seed"])
    print(f"\nMedian winner (by net cost): seed {med_seed} — {med['run_name']}")
    if med_seed in detailed_by_seed:
        detailed_by_seed[med_seed].to_csv(os.path.join(agg_out, "DQN.csv"), index=False)
    rbc_det_src = os.path.join(C.RUNS_DIR, "baselines", rbc_scn.name, "RBC.csv")
    if os.path.exists(rbc_det_src):
        import shutil
        shutil.copy(rbc_det_src, os.path.join(agg_out, "RBC.csv"))

    # 5) aggregate mean/std/min/max across the per-seed winners, vs RBC
    stats = {m_: {"mean": float(ps[m_].mean()), "std": float(ps[m_].std(ddof=1)),
                  "min": float(ps[m_].min()), "max": float(ps[m_].max()),
                  "rbc": float(rbc_m[m_])} for m_ in AGG_METRICS}

    # write a tidy CSV: one row per seed winner, then mean/std/min/max/RBC rows
    rows = per_seed.copy()
    for tag, fn in [("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")]:
        rows.append({"seed": tag, "run_name": "", "layers": "", "gamma": "",
                     **{m_: stats[m_][fn] for m_ in AGG_METRICS}})
    rows.append({"seed": "RBC", "run_name": rbc_scn.name, "layers": "", "gamma": "",
                 **{m_: stats[m_]["rbc"] for m_ in AGG_METRICS}})
    pd.DataFrame(rows).to_csv(os.path.join(sdir, "aggregate_metrics.csv"), index=False)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario, "seeds": seeds, "n_seeds": len(seeds),
        "test_csv": C.TEST_CSV, "days": args.days,
        "rbc_scenario": rbc_scn.name,
        "winners": [{"seed": int(r["seed"]), "run_name": r["run_name"],
                     "layers": int(r["layers"]), "gamma": float(r["gamma"]),
                     "net_grid_cost": float(r["net_grid_cost"])} for r in per_seed],
        "median": {"seed": int(med["seed"]), "run_name": med["run_name"]},
        "n_seeds_beating_rbc": int((ps["net_grid_cost"] < stats["net_grid_cost"]["rbc"]).sum()),
        "aggregate": stats,
    }
    json.dump(summary, open(os.path.join(sdir, "study_summary.json"), "w"), indent=2)

    # 6) offline-plotting bundle: training learning curves (all runs) + per-run training summary
    #    + consolidated validation rankings. Reads existing logs only (no extra rollouts).
    curve_frames, train_rows = [], []
    for c in combos(seeds):
        rn = expected_run_name(args.scenario, layers=c["layers"], gamma=c["gamma"], seed=c["seed"],
                               timesteps=args.timesteps, episode_days=args.episode_days,
                               lr=args.lr, units=args.units, tag=args.tag, smoke=args.smoke)
        cur = export_learning_curve(rn)
        if cur is None:
            continue
        tagged = cur.copy()
        tagged.insert(0, "run_name", rn)
        tagged.insert(0, "gamma", c["gamma"]); tagged.insert(0, "layers", c["layers"])
        tagged.insert(0, "seed", c["seed"])
        curve_frames.append(tagged)
        bi = int(cur["val_reward_mean"].idxmax())
        train_rows.append({"seed": c["seed"], "layers": c["layers"], "gamma": c["gamma"],
                           "run_name": rn, "best_val_reward": float(cur["val_reward_mean"].max()),
                           "final_val_reward": float(cur["val_reward_mean"].iloc[-1]),
                           "best_timestep": int(cur["timestep"].iloc[bi]),
                           "final_timestep": int(cur["timestep"].iloc[-1]), "n_evals": len(cur)})
    if curve_frames:
        pd.concat(curve_frames, ignore_index=True).to_csv(
            os.path.join(sdir, "learning_curves.csv"), index=False)
    if train_rows:
        pd.DataFrame(train_rows).to_csv(os.path.join(sdir, "training_summary.csv"), index=False)
    val_frames = [pd.read_csv(vp) for s in seeds
                  if os.path.exists(vp := os.path.join(sdir, f"seed_{s}", "validation_metrics.csv"))]
    if val_frames:
        pd.concat(val_frames, ignore_index=True).to_csv(
            os.path.join(sdir, "validation_all.csv"), index=False)

    # headline
    nc = stats["net_grid_cost"]
    print(f"\n{'='*64}")
    print(f"  STUDY AGGREGATE - scenario {args.scenario}, {len(seeds)} seeds")
    print(f"  RL net grid cost: ${nc['mean']:.2f} +/- {nc['std']:.2f} "
          f"(min ${nc['min']:.2f}, max ${nc['max']:.2f})")
    print(f"  RBC net grid cost: ${nc['rbc']:.2f}")
    print(f"  RL improvement: {100*(nc['rbc']-nc['mean'])/nc['rbc']:.1f}% "
          f"| seeds beating RBC: {summary['n_seeds_beating_rbc']}/{len(seeds)}")
    print(f"  outputs -> {sdir}")
    print(f"{'='*64}")


# ── subcommand: montecarlo (M12 resilience power — paired RL/RBC over N event realizations) ──
MC_METRICS = ["net_grid_cost", "loss_of_load_kWh", "loss_during_outage_kWh",
              "import_during_dr_kWh", "discomfort_degC_h", "grid_import_kWh"]


def _bootstrap_ci(diffs, n_boot=2000, alpha=0.05, rng=None):
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = rng if rng is not None else np.random.default_rng(0)
    means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(n_boot)])
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def _mc_summarize(sdir, mc, scenario, rl_run, rbc_name):
    """Write the paired RL-vs-RBC Monte-Carlo summary (per-realization CSV + CIs / P(RL better) / worst-case)."""
    mc = mc.sort_values("realization").drop_duplicates("realization").reset_index(drop=True)
    mc.to_csv(os.path.join(sdir, "montecarlo_metrics.csv"), index=False)
    rng = np.random.default_rng(12345)
    summary = {"n": int(len(mc)), "rl_run": rl_run, "rbc_scenario": rbc_name, "metrics": {}}
    for k in MC_METRICS:
        d = mc[f"diff_{k}"].to_numpy()
        summary["metrics"][k] = {
            "rl_mean": float(mc[f"rl_{k}"].mean()), "rl_std": float(mc[f"rl_{k}"].std(ddof=1)),
            "rbc_mean": float(mc[f"rbc_{k}"].mean()), "rbc_std": float(mc[f"rbc_{k}"].std(ddof=1)),
            "diff_mean": float(d.mean()), "diff_ci95": list(_bootstrap_ci(d, rng=rng)),
            "frac_rl_better": float((d > 0).mean()),
            "rl_worst": float(mc[f"rl_{k}"].max()), "rbc_worst": float(mc[f"rbc_{k}"].max()),
        }
    json.dump(summary, open(os.path.join(sdir, "montecarlo_summary.json"), "w"), indent=2)
    r = summary["metrics"]["loss_of_load_kWh"]; nc = summary["metrics"]["net_grid_cost"]
    print(f"\n{'='*64}")
    print(f"  MONTE-CARLO RESILIENCE - scenario {scenario}, {len(mc)} realizations")
    print(f"  emergency backup energy (kWh): RL {r['rl_mean']:.2f} vs RBC {r['rbc_mean']:.2f}  "
          f"| RL better in {100*r['frac_rl_better']:.0f}% of realizations")
    print(f"  net grid cost ($): RL {nc['rl_mean']:.2f} vs RBC {nc['rbc_mean']:.2f}  "
          f"| mean diff ${nc['diff_mean']:.2f} CI95 [{nc['diff_ci95'][0]:.2f},{nc['diff_ci95'][1]:.2f}]")
    print(f"  outputs -> {sdir}/montecarlo_*")
    print(f"{'='*64}")


def cmd_montecarlo(args):
    """Fix the test-year world; draw event realizations at the TRUE marginal rate; run the RL winner
    and RBC PAIRED on each (same event_seed -> identical events); report CIs / P(RL better) / worst-case.

    A full-year paired rollout is ~5-10 min, so N=200 serial is ~30-70 h. Instead SHARD it as a SLURM
    array (each task: --start k*P --n P), then run one --merge job to concatenate shards + summarize."""
    scn = C.SCENARIOS[args.scenario]
    sdir = study_dir(args.scenario)
    rbc_scn = C.SCENARIOS.get(scn.rbc_baseline) or (C.SCENARIOS["A"] if scn.enable_tes else C.SCENARIOS["C"])
    summ_p = os.path.join(sdir, "study_summary.json")
    rl_run = args.run or (json.load(open(summ_p))["median"]["run_name"] if os.path.exists(summ_p) else None)

    # ── merge mode: concatenate shard files + summarize (no rollouts) ──
    if args.merge:
        shards = [f for f in os.listdir(sdir)
                  if f.startswith("montecarlo_metrics_") and f.endswith(".csv")]
        if not shards:
            raise SystemExit(f"no shard files (montecarlo_metrics_*.csv) in {sdir}")
        mc = pd.concat([pd.read_csv(os.path.join(sdir, f)) for f in shards], ignore_index=True)
        print(f"merging {len(shards)} shards -> {len(mc)} realizations")
        _mc_summarize(sdir, mc, args.scenario, rl_run or "(unknown)", rbc_scn.name)
        return

    # ── run mode: paired RL/RBC over realizations [start, start+n) ──
    from microgrid.eval import eval_scenario
    if rl_run is None:
        raise SystemExit("pass --run <name>, or run `study aggregate` first (need the median winner)")
    df = pd.read_csv(C.TEST_CSV)
    coeffs = pd.read_excel(C.COEFFS, sheet_name=None)
    lo, hi = args.start, args.start + args.n
    print(f"Monte-Carlo: realizations [{lo},{hi}) paired  RL={rl_run}  vs  RBC={rbc_scn.name}", flush=True)
    rows = []
    for i in range(lo, hi):
        rbc_m, _, _ = eval_scenario(rbc_scn, df, coeffs, event_seed=i, progress=False)
        rl_m, _, _ = eval_scenario(scn, df, coeffs, model_path=_resolve_model(rl_run),
                                   event_seed=i, progress=False)
        row = {"realization": i}
        for k in MC_METRICS:
            row[f"rbc_{k}"] = float(rbc_m[k]); row[f"rl_{k}"] = float(rl_m[k])
            row[f"diff_{k}"] = float(rbc_m[k] - rl_m[k])   # >0 => RL lower (better)
        rows.append(row)
        print(f"  [{i - lo + 1}/{args.n}] realization {i}: "
              f"net cost RL ${rl_m['net_grid_cost']:.2f} vs RBC ${rbc_m['net_grid_cost']:.2f}", flush=True)

    mc = pd.DataFrame(rows)
    os.makedirs(sdir, exist_ok=True)
    if args.shards != 1 or args.start != 0:      # sharded run -> write a shard file, defer the summary to --merge
        mc.to_csv(os.path.join(sdir, f"montecarlo_metrics_{args.start}_{hi}.csv"), index=False)
        print(f"shard written -> montecarlo_metrics_{args.start}_{hi}.csv  (run `montecarlo --merge` after all shards)")
        return
    _mc_summarize(sdir, mc, args.scenario, rl_run, rbc_scn.name)


# ── CLI ─────────────────────────────────────────────────────────────────────────
def _common(sp):
    sp.add_argument("--scenario", required=True, choices=["B", "D", "B_nopcm"])
    sp.add_argument("--seeds", default=None, help="override seed list, e.g. 0,1,2,3,4")
    sp.add_argument("--timesteps", type=int, default=DEF_TIMESTEPS)
    sp.add_argument("--episode-days", type=int, default=DEF_EPISODE_DAYS)
    sp.add_argument("--lr", type=float, default=DEF_LR)
    sp.add_argument("--units", type=int, default=DEF_UNITS)
    sp.add_argument("--tag", default=None)
    sp.add_argument("--smoke", action="store_true")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("jobs", help="print job map / array size"); _common(p)
    p.add_argument("--count", action="store_true", help="print only the number of jobs")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("train", help="train ONE combo by array index"); _common(p)
    p.add_argument("--index", type=int, required=True, help="SLURM_ARRAY_TASK_ID (0..N-1)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("validate", help="pick the winner for ONE seed"); _common(p)
    p.add_argument("--seed", type=int, default=None, help="seed value to validate")
    p.add_argument("--seed-index", type=int, default=None, help="index into the seed list (SLURM array)")
    p.add_argument("--days", type=int, default=None, help="validate only the first N days")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("aggregate", help="per-seed winners -> test-year mean ± std vs RBC"); _common(p)
    p.add_argument("--days", type=int, default=None, help="evaluate only the first N days")
    p.add_argument("--refresh-baseline", action="store_true")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("montecarlo", help="M12 resilience: paired RL/RBC over N event realizations"); _common(p)
    p.add_argument("--run", default=None, help="RL run to stress-test (default: study median winner)")
    p.add_argument("--n", type=int, default=200, help="number of event realizations")
    p.add_argument("--start", type=int, default=0, help="first realization index (for sharding)")
    p.add_argument("--shards", type=int, default=1, help="if >1 (or --start>0), write a shard file; summary via --merge")
    p.add_argument("--merge", action="store_true", help="concatenate shard files + write the summary (no rollouts)")
    p.set_defaults(func=cmd_montecarlo)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
