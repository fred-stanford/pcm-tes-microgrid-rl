"""
microgrid.eval — evaluate scenarios on the test year (2024) through the common agent
interface (RBC for A/C, trained SB3-DQN for B/D) and report the study metrics.

Examples:
  python -m microgrid.eval --run B_g0.99_lr0.0001_3x128_500k_ep7_s0   # that RL run + its RBC baseline
  python -m microgrid.eval --scenarios A,C                            # RBC scenarios (no model)
  python -m microgrid.eval --run <name> --days 30 --trace             # 30 days + per-decision trace.csv

Writes runs/<run>/tests/eval_metrics.csv (+ trace_<scn>.csv with --trace, RBC/DQN.csv with --detailed-trace).
"""
import argparse
import json
import os
import shutil
import time
import warnings

import pandas as pd

import Input_Parameters as IP
import microgrid.config as C
from microgrid.agents.dqn import DQNConfig, DQNAgent
from microgrid.timeseries_export import DETAILED_TRACE_COLUMNS, build_detailed_row

warnings.filterwarnings("ignore")

MACRO_HOURS = C.ACTION_REPEAT * IP.δt / 3600.0   # 0.25 h

# RBC baselines are deterministic (fixed rules + fixed test data), so a full-year
# baseline eval is cached here and reused across RL runs instead of recomputed.
BASELINES_DIR = os.path.join(C.RUNS_DIR, "baselines")

REWARD_COLS = [
    "total_reward", "rew_grid_purchase", "rew_grid_sale", "rew_comfort",
    "rew_loss_of_load", "rew_degradation", "rew_infeasible", "rew_switching",
]

METRIC_COLS = ["scenario", "label", "net_grid_cost", "import_cost", "export_revenue",
               "grid_import_kWh", "grid_export_kWh", "curtailment_kWh", "loss_of_load_kWh",
               "loss_during_outage_kWh", "import_during_dr_kWh", "discomfort_degC_h",
               "battery_throughput_kWh", "battery_degradation_usd", "pcm_throughput_kWh", "hp_elec_kWh",
               "hp_elec_peak_price_kWh", "peak_import_kW", "peak_export_kW", "decisions"]


# Bump this whenever the RBC rules (physics/rbc_rules.py) or any RBC-relevant Input_Parameters
# constant changes, so cached baselines auto-invalidate instead of silently going stale (m9).
BASELINE_VERSION = 8  # v8: reward/physics changes — B1 battery degradation cost, FR4 RTE 0.87,
                      #     FR7 comfort band aligned to [21.5, 24.5]. Forces RBC baseline recompute so
                      #     RBC and RL are scored under the SAME reward (v7 = HP-TES model swap).


def _baseline_meta(scn):
    """Identity of a baseline cache — must match for the cache to be valid."""
    return {"version": BASELINE_VERSION, "scenario": scn.name,
            "battery_capacity_kwh": scn.battery_capacity_kwh,
            "pv_size_kw": scn.pv_size_kw, "test_csv": C.TEST_CSV}


def load_cached_baseline(scn):
    """Return (metrics_dict, trace_path|None, detailed_path|None) if cache exists."""
    d = os.path.join(BASELINES_DIR, scn.name)
    meta_p, metrics_p = os.path.join(d, "meta.json"), os.path.join(d, "eval_metrics.csv")
    if not (os.path.exists(meta_p) and os.path.exists(metrics_p)):
        return None
    meta = json.load(open(meta_p))
    if any(meta.get(k) != v for k, v in _baseline_meta(scn).items()):
        return None  # stale: battery / PV / data changed
    m = pd.read_csv(metrics_p).iloc[0].to_dict()
    trace_p = os.path.join(d, "trace.csv")
    detailed_p = os.path.join(d, "RBC.csv")
    return m, (
        trace_p if os.path.exists(trace_p) else None,
        detailed_p if os.path.exists(detailed_p) else None,
    )


def save_cached_baseline(scn, m, trace_df, detailed_df=None):
    d = os.path.join(BASELINES_DIR, scn.name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([m])[METRIC_COLS].to_csv(os.path.join(d, "eval_metrics.csv"), index=False)
    json.dump(_baseline_meta(scn), open(os.path.join(d, "meta.json"), "w"), indent=2)
    if trace_df is not None:
        trace_df.to_csv(os.path.join(d, "trace.csv"), index=False)
    if detailed_df is not None:
        os.makedirs(d, exist_ok=True)
        detailed_df.to_csv(os.path.join(d, "RBC.csv"), index=False)


def _agent_for(scn, model_path=None, env=None):
    if scn.control == "RL":
        cfg = DQNConfig(base_dir=C.RUNS_DIR)
        agent = DQNAgent(config=cfg, num_actions=C.NUM_ACTIONS)
        agent.load(model_path, env=env)
        return agent
    return C.build_agent(scn)


def eval_scenario(scn, df, coeffs, *, model_path=None, max_decisions=None, trace=False,
                  detailed_trace=False, seed=0, progress=False, event_seed=None):
    detailed_rows = []

    def on_substep(macro_action, info, base_env):
        detailed_rows.append(build_detailed_row(base_env, macro_action, info))

    env = C.build_env(
        scn, df, coeffs, episode_days=365, seed=seed,
        on_substep=on_substep if detailed_trace else None,
    )
    env.reset()
    base = env.unwrapped
    base.t0 = 0
    base.t = 0
    base.iteration = 1
    # M6/M12: reset() sampled events for a random window; re-sample aligned to the t0=0 eval window.
    # force=None -> TRUE marginal event rate (not the stratified training enrichment). event_seed
    # gives a reproducible realization; the Monte-Carlo harness varies it for a paired RL/RBC study.
    if getattr(base, "enable_events", False):
        base.sample_events(seed=(event_seed if event_seed is not None else seed), force=None)
    agent = _agent_for(scn, model_path=model_path, env=env)

    m = dict(net_grid_cost=0.0, import_cost=0.0, export_revenue=0.0,
             grid_import_kWh=0.0, grid_export_kWh=0.0, curtailment_kWh=0.0,
             loss_of_load_kWh=0.0, discomfort_degC_h=0.0, battery_throughput_kWh=0.0,
             hp_elec_kWh=0.0, import_during_dr_kWh=0.0, loss_during_outage_kWh=0.0,
             pcm_throughput_kWh=0.0, peak_import_kW=0.0, peak_export_kW=0.0,
             hp_elec_peak_price_kWh=0.0, battery_degradation_usd=0.0, decisions=0,
             **{k: 0.0 for k in REWARD_COLS})
    prev_pcm = float(base.state.get("PCM_Hot_E_kWh", 0.0)) + float(base.state.get("PCM_Cold_E_kWh", 0.0))
    peak_buy = float(df["Grid Electricity Buy Price"].max())
    rows = []

    macro_steps = base.episode_steps // C.ACTION_REPEAT
    total_expected = macro_steps if max_decisions is None else min(macro_steps, max_decisions)
    progress_interval = max(1, total_expected // 100)  # ~1% updates
    t0 = time.perf_counter()
    if progress:
        print(f"  simulating {total_expected} decisions...", flush=True)

    done, n = False, 0
    while not done:
        a = agent.decide_action(env)
        obs, r, term, trunc, info = env.step(a)
        m["total_reward"] += r
        m["rew_grid_purchase"] += info.get("rew_grid_purchase", 0.0)
        m["rew_grid_sale"] += info.get("rew_grid_sale", 0.0)
        m["rew_comfort"] += info.get("rew_comfort", 0.0)
        m["rew_loss_of_load"] += info.get("rew_loss_of_load", 0.0)
        m["rew_infeasible"] += info.get("rew_infeasible", 0.0)
        m["rew_degradation"] += info.get("rew_degradation", 0.0)
        m["battery_degradation_usd"] += -info.get("rew_degradation", 0.0)  # $ wear cost (positive)
        m["rew_switching"] += info.get("rew_switching", 0.0)
        gp, gs = info.get("grid_purchase_kWh", 0.0), info.get("grid_sale_kWh", 0.0)
        m["grid_import_kWh"] += gp
        m["grid_export_kWh"] += gs
        m["import_cost"] += -info.get("rew_grid_purchase", 0.0)
        m["export_revenue"] += info.get("rew_grid_sale", 0.0)
        m["curtailment_kWh"] += info.get("curtailment_kWh", 0.0)
        m["loss_of_load_kWh"] += info.get("loss_of_load_kWh", 0.0)
        m["discomfort_degC_h"] += -info.get("rew_comfort", 0.0) / IP.DISCOMFORT_USD_PER_DEGC_PER_H
        m["battery_throughput_kWh"] += info.get("battery_throughput_kWh", 0.0)
        m["hp_elec_kWh"] += info.get("hp_elec_kWh", 0.0)
        if int(info.get("DR Event", 0)) == 1:
            m["import_during_dr_kWh"] += gp
        if int(info.get("Grid Outage Event", 0)) == 1:
            m["loss_during_outage_kWh"] += info.get("loss_of_load_kWh", 0.0)
        if float(info.get("price_buy", 0.0)) >= peak_buy - 1e-9:
            m["hp_elec_peak_price_kWh"] += info.get("hp_elec_kWh", 0.0)
        cur_pcm = float(base.state.get("PCM_Hot_E_kWh", 0.0)) + float(base.state.get("PCM_Cold_E_kWh", 0.0))
        m["pcm_throughput_kWh"] += abs(cur_pcm - prev_pcm)
        prev_pcm = cur_pcm
        m["peak_import_kW"] = max(m["peak_import_kW"], gp / MACRO_HOURS)
        m["peak_export_kW"] = max(m["peak_export_kW"], gs / MACRO_HOURS)

        if trace:
            rows.append(dict(
                t=info.get("DateTime"), Ti=info.get("Indoor Temperature"),
                Ta=info.get("ambient_temp_C"),
                soc_kWh=info.get("battery_soc_kWh"),
                soc=info.get("battery_soc_frac"),
                pcm_h=info.get("PCM_Hot_SOC"), pcm_c=info.get("PCM_Cold_SOC"),
                imp_kW=gp / MACRO_HOURS, exp_kW=gs / MACRO_HOURS,
                p_buy=info.get("price_buy"), p_sell=info.get("price_sell"),
                pv=info.get("pv_capacity_factor"),
                hp_action=int(info.get("fmu_action", 0)),
                batt_action=int(info.get("battery_action", 0)),
                dr=int(info.get("DR Event", 0)), outage=int(info.get("Grid Outage Event", 0)),
            ))

        n += 1
        done = term or trunc or (max_decisions is not None and n >= max_decisions)
        if progress and (n == 1 or n % progress_interval == 0 or done):
            elapsed = time.perf_counter() - t0
            pct = 100.0 * n / total_expected
            rate = n / elapsed if elapsed > 0 else 0.0
            eta_s = (total_expected - n) / rate if rate > 0 else 0.0
            eta = f"ETA {eta_s / 60:.0f}m" if eta_s >= 60 else f"ETA {eta_s:.0f}s"
            print(f"  [{n}/{total_expected}] {pct:5.1f}%  "
                  f"reward={m['total_reward']:.1f}  net cost=${m['import_cost'] - m['export_revenue']:.2f}  "
                  f"({elapsed / 60:.1f}m elapsed, {eta})",
                  flush=True)

    m["net_grid_cost"] = m["import_cost"] - m["export_revenue"]
    m["decisions"] = n
    m["scenario"], m["label"] = scn.name, scn.label
    trace_df = pd.DataFrame(rows) if trace else None
    detailed_df = pd.DataFrame(detailed_rows)[DETAILED_TRACE_COLUMNS] if detailed_trace else None
    return m, trace_df, detailed_df


def print_comparison(rows):
    """Compact comparison table to the terminal (RBC vs RL, with Δ when both are present)."""
    metrics = [
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
    cols = sorted(rows, key=lambda r: 0 if "RBC" in r["label"] else 1)
    hdr = lambda r: f"{r['scenario']}.{'RBC' if 'RBC' in r['label'] else 'RL'}"
    rbc = next((r for r in cols if "RBC" in r["label"]), None)
    rl = next((r for r in cols if "RBC" not in r["label"]), None)
    show_delta = rbc is not None and rl is not None and len(cols) == 2
    W = 26
    head = f"\n{'metric':<{W}}" + "".join(f"{hdr(r):>13}" for r in cols) + (f"{'D RL/RBC':>12}" if show_delta else "")
    print(head)
    print("-" * len(head.strip("\n")))
    for key, label, fmt in metrics:
        line = f"{label:<{W}}" + "".join(f"{fmt.format(float(r[key])):>13}" for r in cols)
        if show_delta:
            base = float(rbc[key])
            d = 100.0 * (float(rl[key]) - base) / abs(base) if base else 0.0
            line += f"{d:>+11.0f}%"
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="trained RL run name under runs/ (evals it + its RBC baseline)")
    ap.add_argument("--scenarios", default=None, help="comma list, e.g. A,C (RBC scenarios)")
    ap.add_argument("--days", type=int, default=None, help="evaluate only the first N days")
    ap.add_argument("--trace", action="store_true", help="also dump per-decision trace_<scn>.csv")
    ap.add_argument("--detailed-trace", action="store_true",
                    help="dump 5-min timeseries CSVs (RBC.csv + DQN.csv) under runs/<run>/tests/")
    ap.add_argument("--refresh-baseline", action="store_true",
                    help="recompute the RBC baseline instead of using the cache")
    ap.add_argument("--smoke", action="store_true", help="quick run (2 days)")
    args = ap.parse_args()

    df = pd.read_csv(C.TEST_CSV)
    coeffs = pd.read_excel(C.COEFFS, sheet_name=None)
    days = 2 if args.smoke else args.days
    max_dec = None if days is None else days * (288 // C.ACTION_REPEAT)

    # Build the (scenario, model_path) work list.
    work = []
    if args.run:
        cfg = json.load(open(os.path.join(C.RUNS_DIR, args.run, "config.json")))
        rl = C.SCENARIOS[cfg["scenario"]["name"]]
        model_path = os.path.join(C.RUNS_DIR, args.run, "model.zip")
        baseline = C.SCENARIOS["A"] if rl.enable_tes else C.SCENARIOS["C"]
        work = [(rl, model_path), (baseline, None)]
        out_dir = os.path.join(C.RUNS_DIR, args.run, C.TESTS_DIR)
    elif args.scenarios:
        names = [s.strip() for s in args.scenarios.split(",")]
        for nm in names:
            scn = C.SCENARIOS[nm]
            if scn.control == "RL":
                print(f"  skip {nm}: RL scenario needs --run <trained run>")
                continue
            work.append((scn, None))
        out_dir = os.path.join(C.RUNS_DIR, f"{C.TESTS_DIR}_{'_'.join(names)}")
    else:
        ap.error("pass --run <name> and/or --scenarios A,C")

    full_year = max_dec is None
    rows = []
    for scn, model_path in work:
        is_baseline = (scn.control == "RBC" and model_path is None)
        is_rl = not is_baseline
        ts_name = "DQN.csv" if is_rl else "RBC.csv"

        # Reuse a cached full-year RBC baseline instead of recomputing it.
        if is_baseline and full_year and not args.refresh_baseline:
            cached = load_cached_baseline(scn)
            if cached is not None:
                m, cached_trace, cached_detailed = cached
                need_detailed = args.detailed_trace and cached_detailed is None
                if not need_detailed:
                    rows.append(m)
                    print(f"\n=== {scn.name} ({scn.label}) - cached baseline ===")
                    if args.trace and cached_trace:
                        os.makedirs(out_dir, exist_ok=True)
                        shutil.copy(cached_trace, os.path.join(out_dir, f"trace_{scn.name}.csv"))
                    if args.detailed_trace and cached_detailed:
                        os.makedirs(out_dir, exist_ok=True)
                        shutil.copy(cached_detailed, os.path.join(out_dir, ts_name))
                        print(f"  detailed timeseries -> {out_dir}/{ts_name} (cached)")
                    if not need_detailed:
                        continue

        print(f"\n=== Evaluating {scn.name} ({scn.label}) ===")
        want_trace = args.trace or (is_baseline and full_year)
        want_detailed = args.detailed_trace
        m, tr, detailed = eval_scenario(
            scn, df, coeffs, model_path=model_path,
            max_decisions=max_dec, trace=want_trace, detailed_trace=want_detailed,
            progress=args.detailed_trace,
        )
        rows.append(m)
        print(f"  net grid cost = ${m['net_grid_cost']:.2f} | import {m['grid_import_kWh']:.1f} kWh | "
              f"export {m['grid_export_kWh']:.1f} kWh | discomfort {m['discomfort_degC_h']:.1f} degC.h | "
              f"LoL {m['loss_of_load_kWh']:.2f} kWh")
        if args.trace and tr is not None:
            os.makedirs(out_dir, exist_ok=True)
            tr.to_csv(os.path.join(out_dir, f"trace_{scn.name}.csv"), index=False)
        if args.detailed_trace and detailed is not None:
            os.makedirs(out_dir, exist_ok=True)
            out_ts = os.path.join(out_dir, ts_name)
            detailed.to_csv(out_ts, index=False)
            print(f"  detailed timeseries -> {out_ts} ({len(detailed)} rows)")
        if is_baseline and full_year:
            save_cached_baseline(scn, m, tr, detailed_df=detailed if want_detailed else None)

    print_comparison(rows)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "eval_metrics.csv")
    pd.DataFrame(rows)[METRIC_COLS].to_csv(out, index=False)
    print(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
