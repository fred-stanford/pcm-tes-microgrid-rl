# -*- coding: utf-8 -*-
"""Generate the paper figure suite (main Fig 2-6 + SI S1-S5) from runs/study_B/.

Main text:  Fig 2 annual ledger | Fig 3 seeds + outcome ratios | Fig 4 Monte-Carlo |
            Fig 5 year-averaged daily profiles | Fig 6 average summer day
            (Fig 1, the system schematic, is drawn by make_schematic.py.)
SI:         S1 calibration (placeholder) | S2 learning curves | S3 validation selection |
            S4 cumulative net cost | S5 3-day window

Figures carry no titles; the captions in the manuscript describe them. Panels are labeled (a), (b), ...
Outputs go to Fig/main and Fig/SI next to this script (and to $FIG_MIRROR, if set).

Inputs: the study outputs in $STUDY_DIR (default: runs/study_B). The summary files are in
results/study_B; the 5-minute traces in aggregate/ come from the data archive (see README.md).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SB = os.environ.get("STUDY_DIR", os.path.join(REPO, "runs", "study_B"))   # study outputs
AGG = os.path.join(SB, "aggregate")                                       # 5-min traces
_MIRROR = os.environ.get("FIG_MIRROR")                                    # optional extra output folder
OUT_ROOTS = [HERE] + ([_MIRROR] if _MIRROR and os.path.isdir(_MIRROR) else [])
for _r in OUT_ROOTS:
    for _s in ("main", "SI"):
        os.makedirs(os.path.join(_r, _s), exist_ok=True)

# ---- Okabe-Ito colorblind-safe palette ----
RBC_C   = "#7F7F7F"   # grey  (RBC)
RL_C    = "#0072B2"   # blue  (RL)
FAVOR   = "#009E73"   # green (favors RL)
UNFAVOR = "#D55E00"   # vermillion (favors RBC)
PV_C    = "#E69F00"   # orange
COLD_C  = "#56B4E9"   # sky blue
HOT_C   = "#D55E00"   # vermillion
BATT_C  = "#009E73"   # green

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def save(fig, sub, name):
    for root in OUT_ROOTS:
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(root, sub, f"{name}.{ext}"))
    plt.close(fig)
    print("  wrote", sub, name)


def ptitle(ax, s):
    ax.set_title(s, loc="left", fontsize=11, fontweight="bold")


# ---------- load tabular data ----------
am = pd.read_csv(os.path.join(SB, "aggregate_metrics.csv"), dtype={"seed": str})
def row(tag):
    return am[am["seed"] == tag].iloc[0]
mean_r, std_r, rbc_r = row("mean"), row("std"), row("RBC")
seeds = am[am["seed"].str.fullmatch(r"\d+")].copy()
for c in seeds.columns:
    if c not in ("run_name",):
        seeds[c] = pd.to_numeric(seeds[c], errors="coerce")

mc = pd.read_csv(os.path.join(SB, "montecarlo_metrics.csv"))
mc = mc[pd.to_numeric(mc["realization"], errors="coerce").notna()]

# =====================================================================
# SI FIG S1 — calibration placeholder (author-supplied data)
# =====================================================================
def figS1_calibration():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    titles = ["(a) Heat-pump COP / capacity\nmeasured vs simulated",
              "(b) PCM TES charge/discharge\nmeasured vs simulated",
              "(c) Indoor temperature\nmeasured vs simulated"]
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linestyle((0, (4, 4))); s.set_edgecolor("#B0B0B0")
        ax.text(0.5, 0.5, "AUTHOR-SUPPLIED\ncalibration data\n[CITE]",
                ha="center", va="center", fontsize=11, color="#888888",
                style="italic", transform=ax.transAxes)
    save(fig, "SI", "FigS1_calibration_PLACEHOLDER")

# =====================================================================
# FIG 2 — annual economic ledger
# =====================================================================
def fig2():
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [3, 1.4]})
    labels = ["Import cost", "Export\nrevenue", "Net grid\ncost", "Battery\nwear",
              "Operating cost\n(net grid + wear)"]
    rbc = [rbc_r.import_cost, rbc_r.export_revenue, rbc_r.net_grid_cost,
           rbc_r.battery_degradation_usd, rbc_r.net_grid_cost + rbc_r.battery_degradation_usd]
    rl  = [mean_r.import_cost, mean_r.export_revenue, mean_r.net_grid_cost,
           mean_r.battery_degradation_usd, mean_r.net_grid_cost + mean_r.battery_degradation_usd]
    hc_std = float((seeds.net_grid_cost + seeds.battery_degradation_usd).std(ddof=1))   # per-seed sum, not hypot
    err = [std_r.import_cost, std_r.export_revenue, std_r.net_grid_cost,
           std_r.battery_degradation_usd, hc_std]
    x = np.arange(len(labels)); w = 0.38
    axc.bar(x - w/2, rbc, w, label="RBC", color=RBC_C)
    axc.bar(x + w/2, rl, w, yerr=err, label="RL (10-seed mean \u00b1 s.d.)", color=RL_C,
            error_kw=dict(ecolor="#333333", capsize=3, lw=1))
    for xi, (a, b) in enumerate(zip(rbc, rl)):
        axc.text(xi - w/2, a + 12, f"${a:.0f}", ha="center", fontsize=8.5, color="#555")
        axc.text(xi + w/2, b + err[xi] + 12, f"${b:.0f}", ha="center", fontsize=8.5,
                 color=RL_C, fontweight="bold")
    axc.set_xticks(x); axc.set_xticklabels(labels, fontsize=9)
    axc.set_ylabel("US$ per year")
    ptitle(axc, "(a) Cash flows")
    axc.legend(loc="upper right")
    pct = 100 * (rl[4] - rbc[4]) / rbc[4]
    axc.annotate(f"\u2212{abs(pct):.0f}%", xy=(4 + w/2, rl[4] + err[4] + 32), xytext=(4 + w/2, rl[4] + 210),
                 ha="center", color=FAVOR, fontweight="bold", fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=FAVOR, lw=1.6))
    axc.set_ylim(0, max(rbc) * 1.25)

    labs2 = ["Discomfort\n(\u00d7 $2.423/\u00b0C\u00b7h)", "Loss of load\n(\u00d7 $10/kWh)"]
    rbc2 = [rbc_r.discomfort_degC_h * 2.423, rbc_r.loss_of_load_kWh * 10]
    rl2  = [mean_r.discomfort_degC_h * 2.423, mean_r.loss_of_load_kWh * 10]
    x2 = np.arange(len(labs2))
    axs.bar(x2 - w/2, rbc2, w, color=RBC_C, hatch="//", edgecolor="white")
    axs.bar(x2 + w/2, rl2, w, color=RL_C, hatch="//", edgecolor="white")
    for xi, (a, b) in enumerate(zip(rbc2, rl2)):
        axs.text(xi - w/2, a + 15, f"${a:.0f}", ha="center", fontsize=8.5, color="#555")
        axs.text(xi + w/2, b + 15, f"${b:.0f}", ha="center", fontsize=8.5, color=RL_C, fontweight="bold")
    axs.set_xticks(x2); axs.set_xticklabels(labs2, fontsize=9)
    axs.set_ylabel("US$ per year")
    ptitle(axs, "(b) Terms at estimated prices")
    axs.set_ylim(0, max(rbc2) * 1.25)
    fig.tight_layout()
    save(fig, "main", "Fig2_annual_ledger")

# =====================================================================
# FIG 3 — per-seed net grid cost + outcome ratios
# =====================================================================
def fig3():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.35]}, constrained_layout=True)
    nc = seeds["net_grid_cost"].values
    m = float(mean_r.net_grid_cost); sd = float(std_r.net_grid_cost); n = len(nc)
    ci = 2.262 * sd / np.sqrt(n)
    xs = np.arange(n)
    axa.axhspan(m - ci, m + ci, color=RL_C, alpha=0.15, lw=0, label=f"RL mean ${m:.0f}, 95% CI")
    axa.axhline(m, color="#003f5c", lw=1.5)
    axa.axhline(rbc_r.net_grid_cost, color=RBC_C, ls="--", lw=1.6, label=f"RBC ${rbc_r.net_grid_cost:.0f}")
    axa.scatter(xs, nc, s=55, color=RL_C, zorder=3, edgecolor="white", linewidth=0.8, label="RL, one seed")
    axa.set_xticks(xs); axa.set_xticklabels([str(i) for i in range(n)], fontsize=8)
    axa.set_xlim(-0.6, n - 0.4)
    axa.set_xlabel("Training seed"); axa.set_ylabel("Net grid cost (US$ per year)")
    axa.set_ylim(0, rbc_r.net_grid_cost * 1.15)
    ptitle(axa, "(a) Net grid cost by training seed")
    axa.text(0.5, 0.93, f"{int((nc < rbc_r.net_grid_cost).sum())} of {n} seeds below the RBC",
             transform=axa.transAxes, ha="center", color=FAVOR, fontweight="bold", fontsize=10)
    axa.legend(loc="lower center", fontsize=8.2)

    comps = [
        ("Net grid cost", mean_r.net_grid_cost / rbc_r.net_grid_cost),
        ("HP-TES electricity, summer peak hours", mean_r.hp_elec_peak_price_kWh / rbc_r.hp_elec_peak_price_kWh),
        ("Import cost", mean_r.import_cost / rbc_r.import_cost),
        ("Grid import energy", mean_r.grid_import_kWh / rbc_r.grid_import_kWh),
        ("Discomfort", mean_r.discomfort_degC_h / rbc_r.discomfort_degC_h),
        ("Battery wear", mean_r.battery_degradation_usd / rbc_r.battery_degradation_usd),
        ("HP-TES electricity", mean_r.hp_elec_kWh / rbc_r.hp_elec_kWh),
        ("Peak grid import", mean_r.peak_import_kW / rbc_r.peak_import_kW),
        ("Export revenue", mean_r.export_revenue / rbc_r.export_revenue),
        ("Loss of load", mean_r.loss_of_load_kWh / rbc_r.loss_of_load_kWh),
    ]
    comps_sorted = sorted(comps, key=lambda t: t[1])
    names = [c[0] for c in comps_sorted]
    vals = [float(c[1]) for c in comps_sorted]
    fav = lambda name, v: (v > 1) if name == "Export revenue" else (v < 1)
    colors = [FAVOR if fav(n_, v) else UNFAVOR for n_, v in zip(names, vals)]
    y = np.arange(len(names))
    axb.barh(y, vals, color=colors, height=0.66)
    axb.axvline(1.0, color="#333333", lw=1.4)
    axb.text(0.98, -0.6, "parity", fontsize=8, color="#333", ha="right", va="center")
    for yi, v in zip(y, vals):
        axb.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=8.5, color="#333")
    axb.set_yticks(y); axb.set_yticklabels(names, fontsize=9)
    axb.set_xlim(0, 1.55)
    axb.set_ylim(-0.8, 12.6)
    axb.set_xlabel("RL \u00f7 RBC  (below 1 favors RL; for export revenue, above 1 favors RL)")
    ptitle(axb, "(b) Outcomes, RL relative to RBC")
    axb.legend(handles=[Patch(color=FAVOR, label="favors RL"), Patch(color=UNFAVOR, label="favors RBC")],
               loc="lower right", fontsize=8.5)
    dr_rl, dr_rb = float(mean_r.import_during_dr_kWh), float(rbc_r.import_during_dr_kWh)
    dr_cost = (dr_rl - dr_rb) * (0.55214 + 0.60)   # summer peak rate + DR adder, US$/kWh
    axb.text(0.02, 12.45,
             f"Also favoring the RBC: grid import during\n"
             f"DR events (RL {dr_rl:.2f} vs RBC {dr_rb:.2f} kWh,\n"
             f"${dr_cost:.2f} per year). Loss of load reverses\n"
             f"over 100 event schedules (Fig. 4).",
             fontsize=7.8, color=UNFAVOR, va="top")
    save(fig, "main", "Fig3_robustness")

# =====================================================================
# FIG 4 — Monte-Carlo over 100 event schedules
# =====================================================================
def fig4_mc():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.55, 1]})
    rng = np.random.default_rng(1)
    dn = mc["diff_net_grid_cost"].astype(float)
    axA.hist(dn, bins=15, color=RL_C, alpha=0.85, edgecolor="white")
    axA.axvline(dn.mean(), color="#003f5c", lw=1.8)
    axA.set_ylim(0, 16)
    axA.set_xlabel("Net grid cost saving, RBC \u2212 RL (US$ per year)")
    axA.set_ylabel("Event schedules")
    ptitle(axA, "(a) Net grid cost saving")
    axA.text(0.03, 0.97, f"RL cheaper in {int((dn > 0).sum())} of {len(dn)} schedules\nmean saving ${dn.mean():.0f}",
             transform=axA.transAxes, fontsize=9.5, color=FAVOR, va="top", fontweight="bold")

    llr = mc["rbc_loss_of_load_kWh"].astype(float); lll = mc["rl_loss_of_load_kWh"].astype(float)
    bp = axB.boxplot([llr, lll], positions=[1, 2], widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [RBC_C, RL_C]):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for med in bp["medians"]:
        med.set_color("#222"); med.set_linewidth(1.4)
    axB.scatter(1 + (rng.random(len(llr)) - 0.5) * 0.25, llr, s=14, color=RBC_C, alpha=0.55, zorder=3)
    axB.scatter(2 + (rng.random(len(lll)) - 0.5) * 0.25, lll, s=14, color=RL_C, alpha=0.55, zorder=3)
    axB.scatter([1, 2], [llr.max(), lll.max()], marker="v", s=64, color=UNFAVOR, zorder=5, label="worst case")
    axB.text(1.14, llr.max(), f"{llr.max():.1f}", fontsize=8.5, color=UNFAVOR, va="center", fontweight="bold")
    axB.text(2.14, lll.max(), f"{lll.max():.1f}", fontsize=8.5, color=UNFAVOR, va="center", fontweight="bold")
    axB.set_xticks([1, 2]); axB.set_xticklabels(["RBC", "RL"])
    axB.set_ylabel("Loss of load (kWh per year)")
    ptitle(axB, "(b) Loss of load")
    axB.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "main", "Fig4_montecarlo_resilience")
    print(f"    [fig4] saving range ${dn.min():.1f}-${dn.max():.1f}, mean ${dn.mean():.2f}; "
          f"worst LoL RBC {llr.max():.2f} RL {lll.max():.2f}")

# =====================================================================
# load traces
# =====================================================================
usecols = ["DateTime", "imp_kW", "exp_kW", "grid_import_kWh", "grid_export_kWh", "p_buy", "p_sell",
           "PV Power (kW)", "Indoor Temperature (C)", "hp_elec_kWh", "dr", "outage",
           "rew_grid_purchase", "rew_grid_sale", "Battery SOC (kWh)", "Exact System Mode",
           "Remaining Heating Energy PCM (kWh)"]
def load_trace(name):
    df = pd.read_csv(os.path.join(AGG, f"{name}.csv"), usecols=lambda c: c in usecols)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["summer"] = df["DateTime"].dt.month.between(6, 9)
    return df
print("loading traces...")
dqn = load_trace("DQN"); rbc = load_trace("RBC")
seed_traces = [load_trace(f"detailed_seed{k}") for k in range(10)]

def tou(a):
    a.axvspan(15.5, 20.5, color=UNFAVOR, alpha=0.08, lw=0)   # peak 4-9 pm

# =====================================================================
# FIG 5 — year-averaged daily profiles (grid import, HP-TES electricity)
# =====================================================================
def fig5_diurnal():
    rl_imp = np.mean([s.groupby("hour")["imp_kW"].mean().reindex(range(24)).values for s in seed_traces], axis=0)
    rl_hp = np.mean([(s.groupby("hour")["hp_elec_kWh"].mean() * 12).reindex(range(24)).values for s in seed_traces], axis=0)
    rbc_imp = rbc.groupby("hour")["imp_kW"].mean().reindex(range(24)).values
    rbc_hp = (rbc.groupby("hour")["hp_elec_kWh"].mean() * 12).reindex(range(24)).values
    print(f"    [fig5] import RL={rl_imp.sum()*365:.0f}/RBC={rbc_imp.sum()*365:.0f} | "
          f"HP-TES RL={rl_hp.sum()*365:.0f}/RBC={rbc_hp.sum()*365:.0f} | "
          f"night(0-6) import RL={rl_imp[0:7].sum()*365:.0f}/RBC={rbc_imp[0:7].sum()*365:.0f}, "
          f"HP-TES RL={rl_hp[0:7].sum()*365:.0f}/RBC={rbc_hp[0:7].sum()*365:.0f}")
    hrs = np.arange(24)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for a, r_, l_, ylab, t in [(ax[0], rbc_imp, rl_imp, "Mean grid import (kW)", "(a) Grid import"),
                               (ax[1], rbc_hp, rl_hp, "Mean HP-TES electricity (kW)", "(b) HP-TES electricity")]:
        tou(a)
        a.plot(hrs, r_, color=RBC_C, lw=2, marker="o", ms=4, label="RBC")
        a.plot(hrs, l_, color=RL_C, lw=2, marker="o", ms=4, label="RL (10-seed mean)")
        a.set_xlabel("Hour of day"); a.set_ylabel(ylab)
        a.set_xlim(-0.5, 23.5); a.set_ylim(bottom=0); a.set_xticks(range(0, 24, 3))
        ptitle(a, t); a.legend(loc="upper center", fontsize=8.8)
    save(fig, "main", "Fig5_diurnal_dispatch")

# =====================================================================
# FIG 6 — average summer day (June-September): battery, import, export, cold PCM TES charging
# =====================================================================
def fig6_summer():
    def prof(d):
        s = d[d["summer"]]
        g = s.groupby("hour")
        return pd.DataFrame({
            "soc": g["Battery SOC (kWh)"].mean(),
            "imp": g["grid_import_kWh"].mean() * 12,
            "exp": g["grid_export_kWh"].mean() * 12,
            "cold": g["Exact System Mode"].apply(lambda x: x.round().isin([5, 7]).mean()) * 100,
            "credit": g["p_sell"].mean(),
        }).reindex(range(24))
    rb = prof(rbc)
    rl = [prof(s) for s in seed_traces]
    rl_mean = sum(rl) / len(rl)
    rl_min = pd.concat(rl).groupby(level=0).min()
    rl_max = pd.concat(rl).groupby(level=0).max()
    hrs = np.arange(24)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.4), sharex=True)
    specs = [(ax[0, 0], "soc", "Battery state of charge (kWh)", "(a) Battery state of charge"),
             (ax[0, 1], "imp", "Grid import (kW)", "(b) Grid import"),
             (ax[1, 0], "exp", "Grid export (kW)", "(c) Grid export and export credit"),
             (ax[1, 1], "cold", "Time spent charging (%)", "(d) Charging of the cold PCM TES")]
    for a, col, ylab, t in specs:
        tou(a)
        a.plot(hrs, rb[col], color=RBC_C, lw=2, marker="o", ms=3.5, label="RBC")
        a.fill_between(hrs, rl_min[col], rl_max[col], color=RL_C, alpha=0.15, lw=0, label="RL, range over 10 seeds")
        a.plot(hrs, rl_mean[col], color=RL_C, lw=2, marker="o", ms=3.5, label="RL, 10-seed mean")
        a.set_ylabel(ylab)
        ptitle(a, t)
    cap = float(rbc_r.get("battery_capacity_kwh", 7.2)) if "battery_capacity_kwh" in rbc_r else 7.2
    ax[0, 0].axhline(cap, color="#999", lw=0.8, ls=":"); ax[0, 0].text(0.2, cap + 0.06, "full", fontsize=8.5, color="#777")
    ax[0, 0].axhline(0.2 * cap, color="#999", lw=0.8, ls=":"); ax[0, 0].text(0.2, 0.2 * cap + 0.06, "reserve", fontsize=8.5, color="#777")
    ax[0, 1].legend(loc="upper right", fontsize=9)
    tw = ax[1, 0].twinx()
    tw.plot(hrs, rb["credit"], color=PV_C, lw=1.6, ls="--")
    tw.set_ylabel("Export credit (US$/kWh)", color="#8a5a00")
    tw.spines["right"].set_visible(True); tw.grid(False)
    ax[1, 0].legend(handles=[Line2D([], [], color=PV_C, ls="--", lw=1.6, label="export credit (right axis)")],
                    loc="upper left", fontsize=8.5)
    for a in ax[1]:
        a.set_xlabel("Hour of day"); a.set_xticks(range(0, 24, 3)); a.set_xlim(-0.5, 23.5)
    fig.tight_layout()
    save(fig, "main", "Fig6_summer_operation")
    print(f"    [fig6] RBC soc 0h/3h/10h = {rb.soc[0]:.2f}/{rb.soc[3]:.2f}/{rb.soc[10]:.2f}; "
          f"RL soc 5h/8h/13h = {rl_mean.soc[5]:.2f}/{rl_mean.soc[8]:.2f}/{rl_mean.soc[13]:.2f}; "
          f"import 0h RBC {rb.imp[0]:.2f} RL {rl_mean.imp[0]:.2f}; credit 17-19h {rb.credit[17]:.2f}/{rb.credit[18]:.2f}/{rb.credit[19]:.2f}")

# =====================================================================
# SI FIG S2 — validation reward during training
# =====================================================================
def figS2():
    lc = pd.read_csv(os.path.join(SB, "learning_curves.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    gammas = sorted(lc["gamma"].unique())
    gcolors = {gammas[0]: "#0072B2", gammas[1]: "#009E73", gammas[2]: "#CC79A7"}
    for g in gammas:
        sub = lc[lc["gamma"] == g]
        grp = sub.groupby("timestep")["val_reward_mean"]
        ts = np.array(sorted(sub["timestep"].unique()))
        m = grp.mean().reindex(ts).values
        sd = grp.std().reindex(ts).values
        ax.plot(ts / 1000, m, color=gcolors[g], lw=2, marker="o", ms=4, label=f"\u03b3 = {g}")
        ax.fill_between(ts / 1000, m - sd, m + sd, color=gcolors[g], alpha=0.15, lw=0)
    ax.set_xlabel("Training decisions (thousands)")
    ax.set_ylabel("Validation reward, 2023 (US$; higher is better)")
    ax.legend(title="Discount factor (mean \u00b1 s.d. over 20 runs)")
    save(fig, "SI", "FigS2_learning_curves")

# =====================================================================
# SI FIG S3 — validation-based selection
# =====================================================================
def figS3():
    va = pd.read_csv(os.path.join(SB, "validation_all.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(2)
    for _, r in va.iterrows():
        s = int(r["seed"]); win = int(r["rank"]) == 1
        xj = s + (rng.random() - 0.5) * 0.35
        ax.scatter(xj, r["net_grid_cost"], s=(90 if win else 34),
                   color=(FAVOR if win else RL_C), alpha=(1.0 if win else 0.45),
                   marker=("*" if win else "o"), zorder=(4 if win else 3),
                   edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(10)); ax.set_xlabel("Training seed")
    ax.set_ylabel("Validation-year (2023) net grid cost (US$)")
    ax.legend(handles=[Line2D([], [], marker="*", color=FAVOR, ls="", ms=12, label="selected configuration"),
                       Line2D([], [], marker="o", color=RL_C, ls="", ms=7, label="other configurations (5 per seed)")],
              loc="upper right")
    save(fig, "SI", "FigS3_validation_selection")

# =====================================================================
# SI FIG S4 — cumulative net grid cost over the year
# =====================================================================
def figS4():
    def cum(df):
        net = -(df["rew_grid_purchase"].values) - (df["rew_grid_sale"].values)
        return df["DateTime"].values, np.cumsum(net)
    tr, cr = cum(rbc); td, cd = cum(dqn)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tr, cr, color=RBC_C, lw=2, label=f"RBC (${cr[-1]:.0f})")
    ax.plot(td, cd, color=RL_C, lw=2, label=f"RL, median controller (${cd[-1]:.0f})")
    ax.fill_between(td, cd, cr, color=FAVOR, alpha=0.12, lw=0)
    ax.set_ylabel("Cumulative net grid cost (US$)")
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    save(fig, "SI", "FigS4_cumulative_cost")

# =====================================================================
# SI FIG S5 — representative 3-day window (15-min)
# =====================================================================
def figS5():
    best = None
    for start in pd.date_range("2024-02-05", "2024-04-25", freq="1D"):
        end = start + pd.Timedelta(days=3)
        w = dqn[(dqn.DateTime >= start) & (dqn.DateTime < end)]
        if len(w) < 800:
            continue
        if w["outage"].sum() == 0 and w["dr"].sum() == 0:
            pv = w["PV Power (kW)"].sum()
            if best is None or pv > best[1]:
                best = (start, pv)
    start = best[0] if best else pd.Timestamp("2024-03-11")
    end = start + pd.Timedelta(days=3)
    dw = dqn[(dqn.DateTime >= start) & (dqn.DateTime < end)].iloc[::3].reset_index(drop=True)
    rw = rbc[(rbc.DateTime >= start) & (rbc.DateTime < end)].iloc[::3].reset_index(drop=True)
    t = dw["DateTime"].values

    def shade_peaks(a):
        day = start.normalize()
        for k in range(3):
            p0 = day + pd.Timedelta(days=k, hours=16); p1 = day + pd.Timedelta(days=k, hours=21)
            a.axvspan(p0, p1, color=UNFAVOR, alpha=0.07, lw=0)

    fig, ax = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)
    shade_peaks(ax[0])
    ax[0].step(t, dw["p_buy"], where="post", color="#444444", lw=1.6)
    ax[0].set_ylabel("Import price\n(US$/kWh)")
    ptitle(ax[0], "(a) Import price")
    shade_peaks(ax[1])
    ax[1].fill_between(t, dw["PV Power (kW)"], color=PV_C, alpha=0.30, label="PV output", lw=0)
    ax[1].fill_between(t, dw["imp_kW"], color=RL_C, alpha=0.28, lw=0)
    ax[1].plot(t, dw["imp_kW"], color=RL_C, lw=1.7, label="RL grid import")
    ax[1].plot(t, rw["imp_kW"], color=RBC_C, lw=1.4, label="RBC grid import")
    ax[1].set_ylabel("Power (kW)")
    ptitle(ax[1], "(b) Grid import and PV output")
    ax[1].legend(ncol=3, loc="upper right", fontsize=8.5)
    shade_peaks(ax[2])
    def _norm(s):
        s = s.values.astype(float); lo, hi = np.nanmin(s), np.nanmax(s)
        return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
    ax[2].plot(t, _norm(dw["Battery SOC (kWh)"]), color=BATT_C, lw=1.7, label="Battery")
    ax[2].plot(t, _norm(dw["Remaining Heating Energy PCM (kWh)"]), color=HOT_C, lw=1.7, label="Hot PCM TES")
    ax[2].set_ylabel("Relative charge\n(0 = window min,\n1 = window max)")
    ax[2].set_ylim(-0.05, 1.18)
    ptitle(ax[2], "(c) RL controller: battery and hot PCM TES")
    ax[2].legend(ncol=2, loc="upper right", fontsize=8.5)
    ax[3].axhspan(21.5, 24.5, color=FAVOR, alpha=0.10, lw=0)
    ax[3].plot(t, rw["Indoor Temperature (C)"], color=RBC_C, lw=1.3, label="RBC")
    ax[3].plot(t, dw["Indoor Temperature (C)"], color=RL_C, lw=1.3, label="RL")
    ax[3].set_ylabel("Indoor temperature\n(\u00b0C)")
    ptitle(ax[3], "(d) Indoor temperature (shaded: 21.5 to 24.5 \u00b0C comfort band)")
    ax[3].legend(ncol=2, loc="upper right", fontsize=8.5)
    last = end - pd.Timedelta(minutes=5)
    ax[3].set_xlabel(f"{start:%B} {start.day} to {last.day}, 2024 (shaded columns: 4 to 9 pm peak price)")
    for a in ax:
        a.margins(x=0.01)
    save(fig, "SI", "FigS5_dispatch_window")
    print(f"    [figS5] window {start:%Y-%m-%d} to {end:%Y-%m-%d}")


if __name__ == "__main__":
    print("MAIN figures:")
    fig2(); fig3(); fig4_mc(); fig5_diurnal(); fig6_summer()
    print("SI figures:")
    figS1_calibration(); figS2(); figS3(); figS4(); figS5()
    print("DONE")
