"""
microgrid.plots — figures for a trained/evaluated run.

Reads runs/<run>/ (episode_log.csv) and runs/<run>/tests/ (eval_metrics.csv, trace_*.csv)
and writes PNGs to runs/<run>/tests/plots/:
  1_cost.png             annual grid economics (RL vs RBC baseline)
  2_energy.png           grid import/export + spilled/unserved
  3_rl_vs_rbc_pct.png    one-glance % change (green = RL lower)
  4_training_curve.png   episode reward + reward components
  5_actions.png          action distribution (heat-pump + battery)        [needs --trace]
  6_seasonal.png         3-day physical snapshots at Jan/Apr/Aug/Dec       [needs --trace]
  7_daily_profile.png    average daily cycle (grid + battery by hour)      [needs --trace]
  8_cumulative_cost.png  cumulative net grid cost over the year            [needs --trace]

Usage:  python -m microgrid.plots --run <run_name>
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import Input_Parameters as IP
import microgrid.config as C

C_RL, C_RBC = "#C44E52", "#4C72B0"   # RL red, RBC blue
MACRO_H = C.ACTION_REPEAT * IP.δt / 3600.0   # 0.25 h
SEASON_MONTHS = [(1, "January"), (4, "April"), (8, "August"), (12, "December")]


def _paired(ax, m_rbc, m_rl, keys, labels, title, ylabel, fmt="{:.0f}"):
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w/2, [float(m_rbc[k]) for k in keys], w, label=f"{m_rbc['scenario']} · RBC", color=C_RBC)
    ax.bar(x + w/2, [float(m_rl[k]) for k in keys], w, label=f"{m_rl['scenario']} · RL", color=C_RL)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(axis="y", alpha=0.3)


def _action_plot(tA, tB, rbc, rl, out):
    """5 — distribution of the decoded action (heat-pump + battery), RBC vs RL."""
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hp_vals = [0, 1, 2, 3]; hp_lab = ["0 standby", "1 hot", "2 cold", "3 useTES"]
    a = tA["hp_action"].value_counts(normalize=True).reindex(hp_vals, fill_value=0) * 100
    b = tB["hp_action"].value_counts(normalize=True).reindex(hp_vals, fill_value=0) * 100
    x = np.arange(4)
    axes[0].bar(x - w/2, a.values, w, color=C_RBC, label=f"{rbc['scenario']} · RBC")
    axes[0].bar(x + w/2, b.values, w, color=C_RL, label=f"{rl['scenario']} · RL")
    axes[0].set_xticks(x); axes[0].set_xticklabels(hp_lab, fontsize=8)
    axes[0].set_ylabel("% of decisions"); axes[0].set_title("Heat-pump action", fontweight="bold"); axes[0].legend()
    bt_vals = [-2, -1, 0, 1, 2]; bt_lab = ["−2", "−1", "0", "+1", "+2"]
    a2 = tA["batt_action"].value_counts(normalize=True).reindex(bt_vals, fill_value=0) * 100
    b2 = tB["batt_action"].value_counts(normalize=True).reindex(bt_vals, fill_value=0) * 100
    x2 = np.arange(5)
    axes[1].bar(x2 - w/2, a2.values, w, color=C_RBC); axes[1].bar(x2 + w/2, b2.values, w, color=C_RL)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(bt_lab)
    axes[1].set_xlabel("battery action (− discharge … + charge)")
    axes[1].set_ylabel("% of decisions"); axes[1].set_title("Battery action", fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Action distribution over the eval", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(f"{out}/5_actions.png", dpi=130); plt.close(fig)


def _seasonal_plot(tA, tB, out, days=3):
    """6 — 3-day physical windows at 4 points in the year (Jan/Apr/Aug/Dec), RBC vs RL."""
    npts = days * 288 // C.ACTION_REPEAT
    panels = []
    for m, name in SEASON_MONTHS:
        idx = tA.index[tA["t"].dt.month == m]
        if len(idx) == 0:
            continue
        s = idx[0]
        panels.append((name, tA.loc[s:s + npts - 1], tB.loc[s:s + npts - 1]))
    if len(panels) < 2:
        return

    nrows, ncols = 5, len(panels)
    fig, ax = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 11), squeeze=False)
    for j, (name, a, b) in enumerate(panels):
        x = a["t"]
        ax[0][j].axhspan(IP.COMFORT_BAND_LOW, IP.COMFORT_BAND_HIGH, color="green", alpha=0.08)
        ax[0][j].plot(x, a["Ti"], color=C_RBC, lw=1); ax[0][j].plot(x, b["Ti"], color=C_RL, lw=1)
        ax[0][j].set_title(name, fontweight="bold")
        ax[1][j].plot(x, a["soc"], color=C_RBC, lw=1); ax[1][j].plot(x, b["soc"], color=C_RL, lw=1)
        ax[1][j].axhline(1.0 - IP.BatteryDoD, color="gray", ls=":", lw=0.8)
        ax[1][j].set_ylim(0, 1)
        ax[2][j].plot(x, a["pcm_h"], color=C_RBC, lw=1); ax[2][j].plot(x, b["pcm_h"], color=C_RL, lw=1)
        ax[2][j].plot(x, a["pcm_c"], color=C_RBC, ls="--", lw=1); ax[2][j].plot(x, b["pcm_c"], color=C_RL, ls="--", lw=1)
        ax[2][j].set_ylim(0, 1)
        ax[3][j].axhline(0, color="k", lw=0.5)
        ax[3][j].plot(x, a["imp_kW"] - a["exp_kW"], color=C_RBC, lw=1)
        ax[3][j].plot(x, b["imp_kW"] - b["exp_kW"], color=C_RL, lw=1)
        axpv = ax[3][j].twinx(); axpv.fill_between(x, a["pv"], color="gold", alpha=0.2, lw=0)
        axpv.set_ylim(0, 1); axpv.set_yticks([])
        ax[4][j].plot(x, a["p_buy"], color="firebrick", lw=1)
        ax[4][j].plot(x, a["p_sell"], color="seagreen", lw=1)
        for i in range(nrows):
            ax[i][j].grid(alpha=0.25); ax[i][j].tick_params(labelsize=7)
        ax[4][j].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax[0][0].set_ylabel("Indoor °C"); ax[1][0].set_ylabel("Battery SoC\n(frac of pack)")
    ax[2][0].set_ylabel("PCM SoC\n(hot —, cold --)"); ax[3][0].set_ylabel("Grid kW\n(+imp/−exp)")
    ax[4][0].set_ylabel("$/kWh  (buy/sell)")
    ax[0][-1].plot([], [], color=C_RBC, label="RBC"); ax[0][-1].plot([], [], color=C_RL, label="RL")
    ax[0][-1].legend(fontsize=7, loc="upper right")
    fig.autofmt_xdate()
    fig.suptitle(f"Physical signals over {days} days, by season (gold = PV)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(f"{out}/6_seasonal.png", dpi=130); plt.close(fig)


def _daily_profile(tA, tB, rbc, rl, out):
    """7 — average daily cycle (grid exchange + battery SoC by hour of day), RBC vs RL."""
    h = np.arange(24)
    def by_hour(t, col): return t.groupby(t["t"].dt.hour)[col].mean().reindex(h)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].plot(h, by_hour(tA, "imp_kW") - by_hour(tA, "exp_kW"), color=C_RBC, lw=1.8, label=f"{rbc['scenario']} · RBC")
    axes[0].plot(h, by_hour(tB, "imp_kW") - by_hour(tB, "exp_kW"), color=C_RL, lw=1.8, label=f"{rl['scenario']} · RL")
    axpv = axes[0].twinx(); axpv.fill_between(h, by_hour(tA, "pv"), color="gold", alpha=0.2, lw=0)
    axpv.set_ylim(0, 1); axpv.set_yticks([])
    axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("avg grid kW (+imp/−exp)")
    axes[0].set_title("Average daily grid exchange (gold = PV)", fontweight="bold"); axes[0].legend()
    axes[1].plot(h, by_hour(tA, "soc"), color=C_RBC, lw=1.8); axes[1].plot(h, by_hour(tB, "soc"), color=C_RL, lw=1.8)
    axes[1].axhline(1.0 - IP.BatteryDoD, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("hour of day"); axes[1].set_ylabel("avg battery SoC (frac of pack)"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Average daily battery cycle", fontweight="bold")
    for ax in axes:
        ax.grid(alpha=0.3); ax.set_xticks(range(0, 24, 3))
    fig.tight_layout(); fig.savefig(f"{out}/7_daily_profile.png", dpi=130); plt.close(fig)


def _cumulative_cost(tA, tB, rbc, rl, out):
    """8 — running cumulative net grid cost over the year, RBC vs RL."""
    def cum(t): return ((t["p_buy"] * t["imp_kW"] - t["p_sell"] * t["exp_kW"]) * MACRO_H).cumsum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tA["t"], cum(tA), color=C_RBC, lw=1.6, label=f"{rbc['scenario']} · RBC")
    ax.plot(tB["t"], cum(tB), color=C_RL, lw=1.6, label=f"{rl['scenario']} · RL")
    ax.set_ylabel("Cumulative net grid cost ($)")
    ax.set_title("Cumulative grid cost over the year", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(); fig.savefig(f"{out}/8_cumulative_cost.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/")
    args = ap.parse_args()

    rdir = os.path.join(C.RUNS_DIR, args.run)
    edir = os.path.join(rdir, C.TESTS_DIR)
    out = os.path.join(edir, "plots")
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "*.png")):   # clear stale figures
        os.remove(f)

    metrics = pd.read_csv(os.path.join(edir, "eval_metrics.csv")).set_index("scenario", drop=False)
    rl = metrics[metrics["label"].str.contains("RL")].iloc[0]
    rbc = metrics[metrics["label"].str.contains("RBC")].iloc[0]

    # 1 — cost
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _paired(ax, rbc, rl, ["net_grid_cost", "import_cost", "export_revenue"],
            ["Net grid cost", "Import cost", "Export revenue"],
            "Annual grid economics (test year)", "USD", "${:.0f}")
    fig.tight_layout(); fig.savefig(f"{out}/1_cost.png", dpi=130); plt.close(fig)

    # 2 — energy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _paired(axes[0], rbc, rl, ["grid_import_kWh", "grid_export_kWh"],
            ["Grid import", "Grid export"], "Annual energy exchange", "kWh")
    _paired(axes[1], rbc, rl, ["curtailment_kWh", "loss_of_load_kWh"],
            ["Curtailment", "Loss of load"], "Spilled / unserved", "kWh")
    fig.tight_layout(); fig.savefig(f"{out}/2_energy.png", dpi=130); plt.close(fig)

    # 3 — % change RL vs RBC
    keys = ["net_grid_cost", "import_cost", "grid_import_kWh", "curtailment_kWh",
            "loss_of_load_kWh", "discomfort_degC_h", "hp_elec_kWh", "hp_elec_peak_price_kWh",
            "peak_import_kW", "export_revenue", "grid_export_kWh", "peak_export_kW"]
    nice = ["Net grid cost", "Import cost", "Grid import", "Curtailment", "Loss of load",
            "Discomfort", "HP electricity", "HP @ peak price", "Peak import",
            "Export revenue", "Grid export", "Peak export"]
    pct = [100.0 * (float(rl[k]) - float(rbc[k])) / abs(float(rbc[k])) if float(rbc[k]) else 0.0 for k in keys]
    order = np.argsort(pct)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(order))
    ax.barh(y, [pct[i] for i in order], color=["#55A868" if pct[i] < 0 else "#C44E52" for i in order])
    ax.set_yticks(y); ax.set_yticklabels([nice[i] for i in order])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("RL relative to RBC  [%]")
    ax.set_title("Where RL beats / trails the RBC\n(green = RL lower, red = RL higher)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/3_rl_vs_rbc_pct.png", dpi=130); plt.close(fig)

    # 4 — training curve
    log_path = os.path.join(rdir, "episode_log.csv")
    if os.path.exists(log_path):
        t = pd.read_csv(log_path)
        xs = t["timesteps"] / 1000.0
        roll = lambda s: s.rolling(30, min_periods=1, center=True).mean()
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(xs, t["episode_reward"], color="0.7", lw=0.7, label="per episode")
        axes[0].plot(xs, roll(t["episode_reward"]), color=C_RL, lw=2, label="rolling mean")
        axes[0].set_ylabel("Episode reward ($)"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_title(f"{args.run} — training", fontweight="bold")
        for col, lab, c in [("rew_grid_purchase", "import cost", "#C44E52"),
                            ("rew_grid_sale", "export revenue", "#55A868"),
                            ("rew_comfort", "comfort", "#4C72B0"),
                            ("rew_infeasible", "infeasible", "#8172B3")]:
            if col in t:
                axes[1].plot(xs, roll(t[col]), label=lab, lw=1.8, color=c)
        axes[1].set_ylabel("Reward component ($)"); axes[1].set_xlabel("Training decisions [×1000]")
        axes[1].legend(ncol=2); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{out}/4_training_curve.png", dpi=130); plt.close(fig)

    # 5–8 — from the per-decision traces (need --trace)
    ta_p = os.path.join(edir, f"trace_{rbc['scenario']}.csv")
    tb_p = os.path.join(edir, f"trace_{rl['scenario']}.csv")
    if os.path.exists(ta_p) and os.path.exists(tb_p):
        tA, tB = pd.read_csv(ta_p, parse_dates=["t"]), pd.read_csv(tb_p, parse_dates=["t"])
        if "hp_action" in tA.columns and "hp_action" in tB.columns:
            _action_plot(tA, tB, rbc, rl, out)
        _seasonal_plot(tA, tB, out)
        _daily_profile(tA, tB, rbc, rl, out)
        _cumulative_cost(tA, tB, rbc, rl, out)

    print("Wrote plots to", out)
    for f in sorted(os.listdir(out)):
        print("  ", os.path.join(out, f))


if __name__ == "__main__":
    main()
