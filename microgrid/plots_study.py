"""
microgrid.plots_study — figures + tables for the multi-seed study (microgrid.study).

Consumes runs/study_<scn>/ :
  aggregate_metrics.csv          per-seed winners + mean/std/min/max/RBC rows
  aggregate/DQN.csv, RBC.csv     median winner + RBC 5-min detailed traces (mechanism figures)
  aggregate/trace_seed*.csv      per-decision series per winner (seed-spread band)

Produces (into runs/study_<scn>/plots/):
  summary_table.md / .csv        RL mean ± std vs RBC (+ improvement)
  1_performance_summary.png      cost breakdown (RL mean ± std vs RBC) + % improvement
  2_load_shift_window.png        representative window (median winner) — the mechanism
  3_load_shift_annual.png        HP load / grid import by hour + HP by price tier (median winner)
  4_thermal_load_shift.png       monthly load shift (supporting metric) + peak targeting (median)
  5_cumulative_cost.png          per-seed cumulative cost + mean ± min–max band vs RBC

Run:  python -m microgrid.plots_study --scenario B
Single-run mode (one agent, no seeds):  python -m microgrid.plots_study --dir runs/<run>/tests
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import microgrid.config as C
import Input_Parameters as IP

C_RL, C_RBC, C_PV, C_INK, C_DR, C_OUT, COMFORT = (
    "#0072B2", "#D55E00", "#E69F00", "#333333", "#CC79A7", "#555555", "#009E73")
LABEL_RL, LABEL_RBC = "TES + RL (DQN)", "TES + RBC"
DT_H = 5.0 / 60.0
MACRO_H = C.ACTION_REPEAT * 5.0 / 60.0  # 0.25 h
HP = "Total HP-TES Electrical Load (kW)"
PV = "PV Power (kW)"
TI = "Indoor Temperature (C)"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.axisbelow": True})


# ── shared helpers ──────────────────────────────────────────────────────────────
def _shade(ax, times, mask, color, alpha, hatch=None):
    m, tv = np.asarray(mask), np.asarray(times)
    started, start = False, None
    for i in range(len(m)):
        if m[i] and not started:
            start, started = tv[i], True
        elif not m[i] and started:
            ax.axvspan(start, tv[i], color=color, alpha=alpha, lw=0, zorder=0, hatch=hatch); started = False
    if started:
        ax.axvspan(start, tv[-1], color=color, alpha=alpha, lw=0, zorder=0, hatch=hatch)


def _prep(d):
    d = d.copy()
    d["t"] = pd.to_datetime(d["DateTime"])
    d["net_cost_step"] = d["p_buy"] * d["grid_import_kWh"] - d["p_sell"] * d["grid_export_kWh"]
    return d


# ── mechanism figures (median winner) — ported from the validated scripts ────────
def fig_window(dqn, rbc, out):
    thr = np.quantile(rbc["p_buy"], 0.80)
    rb = rbc.copy(); rb["day"] = rb["t"].dt.normalize()
    rb["hp_hi"] = np.where(rb["p_buy"] >= thr, rb["hp_elec_kWh"], 0.0)
    roll = rb.groupby("day")["hp_hi"].sum().rolling(4).sum()
    end = roll.idxmax(); w0 = end - pd.Timedelta(days=3); w1 = w0 + pd.Timedelta(days=4)
    wd = dqn[(dqn["t"] >= w0) & (dqn["t"] < w1)].reset_index(drop=True)
    wr = rbc[(rbc["t"] >= w0) & (rbc["t"] < w1)].reset_index(drop=True)
    tt = wd["t"]; peak = (wd["p_buy"] >= 0.9 * wd["p_buy"].max()).values
    dr = (wd["dr"] > 0).values; og = (wd["outage"] > 0).values
    fig, ax = plt.subplots(5, 1, figsize=(12.5, 12.5), sharex=True)
    for a in ax:
        _shade(a, tt, peak, C_PV, 0.14); _shade(a, tt, dr, C_DR, 0.22, "//"); _shade(a, tt, og, C_OUT, 0.28)
        a.margins(x=0.005)
    ax[0].step(tt, wd["p_buy"], where="post", color=C_INK, lw=1.6, label="Buy price (TOU)")
    ax[0].step(tt, wd["p_sell"], where="post", color="#888", lw=1.1, ls=":", label="Sell price (EEC)")
    ax[0].set_ylabel("$/kWh"); ax[0].set_title("Electricity price  (amber=high-price, hatched=DR, gray=outage)")
    ax[0].legend(loc="upper right", ncol=2, fontsize=9)
    ax[1].fill_between(tt, wd[PV], color=C_PV, alpha=0.55, step="mid"); ax[1].set_ylabel("kW")
    ax[1].set_title("On-site PV generation")
    ax[2].plot(tt, wr[HP].rolling(12, center=True, min_periods=1).mean(), color=C_RBC, lw=1.8, label=LABEL_RBC)
    ax[2].plot(tt, wd[HP].rolling(12, center=True, min_periods=1).mean(), color=C_RL, lw=1.9, label=LABEL_RL)
    ax[2].set_ylabel("kW (1-h avg)"); ax[2].legend(loc="upper right", ncol=2, fontsize=9)
    ax[2].set_title("Heat-pump electrical load (1-h avg) — RL runs off-peak/on PV; RBC into the peak")
    ax[3].plot(tt, wr["PCM_Cold_SOC"], color=C_RBC, lw=1.4, label="Cold PCM - RBC")
    ax[3].plot(tt, wd["PCM_Cold_SOC"], color=C_RL, lw=1.7, label="Cold PCM - RL")
    ax[3].plot(tt, wr["PCM_Hot_SOC"], color=C_RBC, lw=1.2, ls="--", label="Hot PCM - RBC")
    ax[3].plot(tt, wd["PCM_Hot_SOC"], color=C_RL, lw=1.4, ls="--", label="Hot PCM - RL")
    ax[3].set_ylabel("PCM SOC"); ax[3].set_ylim(-0.03, 1.05); ax[3].legend(loc="upper right", ncol=2, fontsize=8)
    ax[3].set_title("Thermal storage (PCM) SOC — RL pre-charges, discharges through the peak")
    ax[4].axhspan(IP.COMFORT_BAND_LOW, IP.COMFORT_BAND_HIGH, color=COMFORT, alpha=0.12,
                  label=f"Comfort band {IP.COMFORT_BAND_LOW:g}-{IP.COMFORT_BAND_HIGH:g} C")
    ax[4].plot(tt, wr[TI], color=C_RBC, lw=1.4, label=LABEL_RBC)
    ax[4].plot(tt, wd[TI], color=C_RL, lw=1.6, label=LABEL_RL)
    ax[4].set_ylabel("C"); ax[4].legend(loc="upper right", ncol=3, fontsize=8)
    ax[4].set_title("Indoor temperature — comfort maintained while shifting the load")
    ax[-1].xaxis.set_major_locator(mdates.DayLocator())
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax[-1].set_xlabel(f"{w0.date()} to {(w1 - pd.Timedelta(days=1)).date()}")
    fig.suptitle("Thermal load shifting (median winner): charge PCM with cheap/solar power, discharge through the peak",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(os.path.join(out, "2_load_shift_window.png")); plt.close(fig)


def fig_annual(dqn, rbc, out):
    for d in (dqn, rbc):
        d["hour"] = d["t"].dt.hour
    price_hour = rbc.groupby("hour")["p_buy"].mean()
    peak_hours = price_hour.index[price_hour >= price_hour.quantile(0.75)].tolist()
    q1, q2 = np.quantile(rbc["p_buy"], [1 / 3, 2 / 3])
    tiers = ["Low", "Mid", "High"]
    tier = lambda p: np.where(p <= q1, "Low", np.where(p <= q2, "Mid", "High"))
    fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(15.5, 4.8)); hrs = np.arange(24)
    def shp(ax):
        for h in peak_hours:
            ax.axvspan(h - 0.5, h + 0.5, color=C_PV, alpha=0.13, lw=0)
    shp(b1)
    b1.plot(hrs, rbc.groupby("hour")[HP].mean().reindex(hrs), color=C_RBC, lw=2, marker="o", ms=3, label=LABEL_RBC)
    b1.plot(hrs, dqn.groupby("hour")[HP].mean().reindex(hrs), color=C_RL, lw=2, marker="o", ms=3, label=LABEL_RL)
    b1.set_title("Heat-pump load by hour of day\n(whole test year; amber=high-price)")
    b1.set_xlabel("hour of day"); b1.set_ylabel("mean HP load (kW)"); b1.set_xticks(range(0, 24, 3)); b1.legend()
    shp(b2)
    b2.plot(hrs, rbc.groupby("hour")["imp_kW"].mean().reindex(hrs), color=C_RBC, lw=2, marker="o", ms=3, label=LABEL_RBC)
    b2.plot(hrs, dqn.groupby("hour")["imp_kW"].mean().reindex(hrs), color=C_RL, lw=2, marker="o", ms=3, label=LABEL_RL)
    b2.set_title("Grid import by hour of day\n(peak shaving)")
    b2.set_xlabel("hour of day"); b2.set_ylabel("mean grid import (kW)"); b2.set_xticks(range(0, 24, 3)); b2.legend()
    hr = dqn.assign(tr=tier(dqn["p_buy"])).groupby("tr")["hp_elec_kWh"].sum().reindex(tiers)
    hb = rbc.assign(tr=tier(rbc["p_buy"])).groupby("tr")["hp_elec_kWh"].sum().reindex(tiers)
    xt = np.arange(3); w = 0.38
    b3.bar(xt - w / 2, hr.values, w, color=C_RL, label=LABEL_RL); b3.bar(xt + w / 2, hb.values, w, color=C_RBC, label=LABEL_RBC)
    for xi, v in zip(xt - w / 2, hr.values): b3.text(xi, v + 8, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, color=C_RL, fontweight="bold")
    for xi, v in zip(xt + w / 2, hb.values): b3.text(xi, v + 8, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, color=C_RBC, fontweight="bold")
    b3.set_xticks(xt); b3.set_xticklabels([f"{t}\nprice tier" for t in tiers]); b3.set_ylabel("HP electricity (kWh/yr)")
    b3.set_ylim(0, max(hb.max(), hr.max()) * 1.42); b3.legend(loc="upper right"); b3.grid(axis="x", visible=False)
    b3.set_title("HP electricity by price tier\n(RL moves energy to cheaper hours)")
    fig.suptitle("Load shifting across the full test year (median winner)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(out, "3_load_shift_annual.png")); plt.close(fig)


def fig_thermal_shift(dqn, rbc, out):
    thr = np.quantile(rbc["p_buy"], 0.8)
    def ls(d):
        d = d.copy(); d["m"] = d["t"].dt.month
        d["fcu"] = d["FCU Heat Delivered"].abs() * DT_H
        d["sto"] = np.where(d["HP Electric Power"] <= 0.01, d["fcu"], 0.0)
        d["peak"] = d["p_buy"] >= thr
        return d
    dq, rb = ls(dqn), ls(rbc)
    def monthly(d):
        g = d.groupby("m").agg(f=("fcu", "sum"), s=("sto", "sum")); return 100 * g["s"] / g["f"]
    def scal(d, pk=None):
        x = d if pk is None else d[d["peak"] == pk]; return 100 * x["sto"].sum() / x["fcu"].sum()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.7, 1]})
    mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]; x = np.arange(12); w = 0.4
    a.bar(x - w / 2, monthly(dq).reindex(range(1, 13)).values, w, color=C_RL, label=LABEL_RL)
    a.bar(x + w / 2, monthly(rb).reindex(range(1, 13)).values, w, color=C_RBC, label=LABEL_RBC)
    a.set_xticks(x); a.set_xticklabels(mn); a.set_ylabel("thermal load shift (% of demand from storage)")
    a.set_title("Monthly load shift — RL steady; RBC swings", fontsize=11); a.legend(loc="upper center", ncol=2); a.set_ylim(0, 55)
    grp = ["All hours", "Peak-price\nhours"]; xt = np.arange(2)
    rl_v = [scal(dq), scal(dq, True)]; rb_v = [scal(rb), scal(rb, True)]
    b.bar(xt - w / 2, rl_v, w, color=C_RL, label="RL (DQN)"); b.bar(xt + w / 2, rb_v, w, color=C_RBC, label="RBC")
    for xi, v in zip(xt - w / 2, rl_v): b.text(xi, v + 0.6, f"{v:.0f}%", ha="center", va="bottom", color=C_RL, fontweight="bold", fontsize=10)
    for xi, v in zip(xt + w / 2, rb_v): b.text(xi, v + 0.6, f"{v:.0f}%", ha="center", va="bottom", color=C_RBC, fontweight="bold", fontsize=10)
    b.set_xticks(xt); b.set_xticklabels(grp); b.set_ylabel("load shift (%)"); b.set_ylim(0, max(rl_v + rb_v) * 1.35)
    b.legend(loc="upper right"); b.grid(axis="x", visible=False); b.set_title("RL targets the peak")
    fig.suptitle("Thermal load shift (supporting metric): RL uses storage consistently and at the peak",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(out, "4_thermal_load_shift.png")); plt.close(fig)


# ── seed-aware figures ───────────────────────────────────────────────────────────
def _read_agg(sdir):
    df = pd.read_csv(os.path.join(sdir, "aggregate_metrics.csv"))
    df["seed"] = df["seed"].astype(str)
    seeds = df[df["seed"].str.fullmatch(r"\d+")]
    stat = {r["seed"]: r for _, r in df[df["seed"].isin(["mean", "std", "min", "max", "RBC"])].iterrows()}
    return seeds, stat


def study_table(sdir, out):
    seeds, stat = _read_agg(sdir)
    rows = [("Net grid cost", "net_grid_cost", "$", "down"),
            ("Import cost", "import_cost", "$", "down"),
            ("Export revenue", "export_revenue", "$", "up"),
            ("Grid import", "grid_import_kWh", "kWh", "down"),
            ("HP elec. at peak price", "hp_elec_peak_price_kWh", "kWh", "down"),
            ("Peak import", "peak_import_kW", "kW", "down"),
            ("Thermal discomfort", "discomfort_degC_h", "degC.h", "down"),
            ("Loss of load", "loss_of_load_kWh", "kWh", "down")]
    n = len(seeds)
    beat = int((seeds["net_grid_cost"].astype(float) < float(stat["RBC"]["net_grid_cost"])).sum())
    lines = [f"# Study results — RL (mean ± std over {n} seeds) vs RBC\n",
             f"Seeds beating RBC on net cost: **{beat}/{n}**\n",
             "| Metric | Unit | RL mean ± std | RL [min, max] | RBC | Improvement |",
             "|---|---|--:|--:|--:|--:|"]
    for name, col, unit, direction in rows:
        mean, std = float(stat["mean"][col]), float(stat["std"][col])
        lo, hi, rbc = float(stat["min"][col]), float(stat["max"][col]), float(stat["RBC"][col])
        imp = (rbc - mean) / abs(rbc) * 100 if direction == "down" else (mean - rbc) / abs(rbc) * 100
        lines.append(f"| {name} | {unit} | {mean:,.1f} ± {std:,.1f} | [{lo:,.1f}, {hi:,.1f}] | {rbc:,.1f} | **{imp:+.0f}%** |")
    open(os.path.join(out, "summary_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def fig_performance(sdir, out):
    seeds, stat = _read_agg(sdir)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    cats = ["Import\ncost", "Export\nrevenue", "Net grid\ncost"]
    cols = ["import_cost", "export_revenue", "net_grid_cost"]
    rl = [float(stat["mean"][c]) for c in cols]; err = [float(stat["std"][c]) for c in cols]
    rb = [float(stat["RBC"][c]) for c in cols]
    x = np.arange(3); w = 0.38
    a1.bar(x - w / 2, rl, w, yerr=err, capsize=4, color=C_RL, label=LABEL_RL + " (mean±std)")
    a1.bar(x + w / 2, rb, w, color=C_RBC, label=LABEL_RBC)
    a1.set_xticks(x); a1.set_xticklabels(cats); a1.set_ylabel("US$ over test year")
    a1.set_title("Annual grid economics"); a1.legend(loc="upper right"); a1.set_ylim(0, max(rb) * 1.2)
    metrics = [("Net grid cost", "net_grid_cost", "down"), ("Import cost", "import_cost", "down"),
               ("Export revenue", "export_revenue", "up"), ("Grid import", "grid_import_kWh", "down"),
               ("HP elec. at peak price", "hp_elec_peak_price_kWh", "down"),
               ("Peak import", "peak_import_kW", "down"), ("Thermal discomfort", "discomfort_degC_h", "down"),
               ("Loss of load", "loss_of_load_kWh", "down")]
    names, vals = [], []
    for nm, col, d in metrics:
        m, r = float(stat["mean"][col]), float(stat["RBC"][col])
        if r == 0:
            continue
        names.append(nm); vals.append((r - m) / abs(r) * 100 if d == "down" else (m - r) / abs(r) * 100)
    order = np.argsort(vals); names = [names[i] for i in order]; vals = [vals[i] for i in order]
    yy = np.arange(len(names))
    a2.barh(yy, vals, color=[C_RL if v >= 0 else C_RBC for v in vals]); a2.axvline(0, color=C_INK, lw=1)
    for y, v in zip(yy, vals):
        a2.text(v + (2 if v >= 0 else -2), y, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, fontweight="bold", color=C_RL if v >= 0 else C_RBC)
    a2.set_yticks(yy); a2.set_yticklabels(names); a2.set_xlabel("RL improvement vs RBC (%)")
    a2.set_title("Where RL helps (mean over seeds)"); a2.grid(axis="y", visible=False)
    a2.set_xlim(min(vals) * 1.25 - 5, max(vals) * 1.2 + 8)
    fig.suptitle("RL vs RBC — test-year performance (mean ± std over seeds; identical TES system)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(out, "1_performance_summary.png")); plt.close(fig)


def _cum_from_trace(path):
    d = pd.read_csv(path); d["t"] = pd.to_datetime(d["t"] if "t" in d else d["DateTime"])
    if "grid_import_kWh" in d and "grid_export_kWh" in d:
        gi, ge = d["grid_import_kWh"], d["grid_export_kWh"]
    else:  # per-decision power trace -> energy over the macro-step
        gi, ge = d["imp_kW"] * MACRO_H, d["exp_kW"] * MACRO_H
    return d["t"], (d["p_buy"] * gi - d["p_sell"] * ge).cumsum()


def fig_cumulative(sdir, out):
    traces = sorted(glob.glob(os.path.join(sdir, "aggregate", "trace_seed*.csv")))
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(11, 5.2))
    curves = []
    for p in traces:
        t, c = _cum_from_trace(p); curves.append(c.values)
        ax.plot(t, c, color=C_RL, lw=0.8, alpha=0.35)
    L = min(len(c) for c in curves); arr = np.vstack([c[:L] for c in curves]); tt = t[:L]
    ax.plot(tt, arr.mean(0), color=C_RL, lw=2.4, label=LABEL_RL + " (mean of seeds)")
    ax.fill_between(tt, arr.min(0), arr.max(0), color=C_RL, alpha=0.15, label="seed min–max")
    rbc_p = os.path.join(sdir, "aggregate", "RBC.csv")
    if os.path.exists(rbc_p):
        tr, cr = _cum_from_trace(rbc_p); ax.plot(tr, cr, color=C_RBC, lw=2.4, label=LABEL_RBC)
    ax.axhline(0, color=C_INK, lw=0.8); ax.set_ylabel("cumulative net grid cost (US$)")
    ax.set_title("Cumulative operating cost over the test year — per-seed spread")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "5_cumulative_cost.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="B", choices=["B", "D"])
    ap.add_argument("--study-dir", default=None, help="override runs/study_<scn>")
    ap.add_argument("--dir", default=None, help="single-run mode: a tests/ dir with DQN.csv/RBC.csv (mechanism figs only)")
    args = ap.parse_args()

    if args.dir:  # single-run: just the mechanism figures from one agent
        out = os.path.join(args.dir, "study_plots"); os.makedirs(out, exist_ok=True)
        dqn = _prep(pd.read_csv(os.path.join(args.dir, "DQN.csv")))
        rbc = _prep(pd.read_csv(os.path.join(args.dir, "RBC.csv")))
        fig_window(dqn, rbc, out); fig_annual(dqn, rbc, out); fig_thermal_shift(dqn, rbc, out)
        print(f"mechanism figures -> {out}"); return

    sdir = args.study_dir or os.path.join(C.RUNS_DIR, f"study_{args.scenario}")
    out = os.path.join(sdir, "plots"); os.makedirs(out, exist_ok=True)
    study_table(sdir, out)
    fig_performance(sdir, out)
    fig_cumulative(sdir, out)
    agg = os.path.join(sdir, "aggregate")
    if os.path.exists(os.path.join(agg, "DQN.csv")) and os.path.exists(os.path.join(agg, "RBC.csv")):
        dqn = _prep(pd.read_csv(os.path.join(agg, "DQN.csv")))
        rbc = _prep(pd.read_csv(os.path.join(agg, "RBC.csv")))
        fig_window(dqn, rbc, out); fig_annual(dqn, rbc, out); fig_thermal_shift(dqn, rbc, out)
    print(f"study figures + table -> {out}")


if __name__ == "__main__":
    main()
