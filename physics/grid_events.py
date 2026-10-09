"""
Runtime grid-service event sampler for the RL microgrid env (M6/M12).

Unlike the data-generation `grid_events` (which bakes one fixed annual calendar into the CSV), this
samples a FRESH realization per training episode (domain randomization) and per Monte-Carlo test
draw. Event semantics follow California (PG&E) practice:

  * DR     -> announced critical-peak PRICE event. Grid stays UP; the buy price is raised by an
              adder over the 16:00-21:00 summer window (PG&E Peak Day Pricing). NOT an import block.
  * outage -> ordinary fault: hard disconnect (no import/export), NO advance signal.
  * PSPS   -> long announced disconnect (24-48 h), with an advance-warning signal.

`sample_episode` returns per-step arrays aligned to a datetime index:
  outage (0/1), dr (0/1), dr_price_adder ($/kWh), dr_notice (0..1 countdown to DR onset),
  psps_warning (0..1 countdown to PSPS onset).

Event-stratified sampling: `force` guarantees an event of a given type inside a short training window
so the agent experiences the prepare->event->recover arc often enough to learn (rare-event
learnability). Evaluation draws at the true marginal rate (`force=None`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── rates / durations (match data_generation_pipeline/grid_events.py + PG&E) ──
ORDINARY_PER_YEAR = 2.0            # Poisson mean, ordinary outages/yr
ORDINARY_DUR_H = (2.0, 3.0)
PSPS_ANNUAL_PROB = (0.10, 0.25)   # P(>=1 PSPS in a year)
PSPS_DUR_H = (24.0, 48.0)
DR_PER_SUMMER = {"moderate": (8, 10), "severe": (12, 16)}
DR_DUR_H = (4.0, 5.0)
DR_HOUR_WINDOW = (16, 21)         # 4-9 pm
DR_MONTHS = (6, 9)                # Jun-Sep inclusive
SUMMER_DAYS = 122                 # Jun 1 - Sep 30

# PG&E Peak Day Pricing: ~+0.60 $/kWh adder over the event window (≈2x the peak TOU rate).
DR_PRICE_ADDER_USD_PER_KWH = 0.60
DR_NOTICE_H = 4.0                 # same-day DR notice
PSPS_NOTICE_H = 24.0             # day-ahead PSPS warning

FORCE_TYPES = ("dr", "outage", "psps")


def _step_hours(dt: pd.DatetimeIndex) -> float:
    if len(dt) < 2:
        return 5.0 / 60.0
    return (dt[1] - dt[0]).total_seconds() / 3600.0


def _mark_block(arr, s, e, value=1.0):
    """Set arr[s:e] = value (index-space, clipped)."""
    s = max(0, s); e = min(len(arr), e)
    if e > s:
        arr[s:e] = value


def _set_notice(notice, onset_idx, lead_steps):
    """Ramp a 0->1 countdown over [onset-lead, onset): notice[i] = 1 - (onset-i)/lead."""
    if lead_steps <= 0:
        return
    s = max(0, onset_idx - lead_steps)
    for i in range(s, min(onset_idx, len(notice))):
        notice[i] = max(notice[i], 1.0 - (onset_idx - i) / float(lead_steps))


def sample_episode(datetimes, rng, *, force=None, dr_case="moderate",
                   dr_price_adder=DR_PRICE_ADDER_USD_PER_KWH,
                   dr_notice_h=DR_NOTICE_H, psps_notice_h=PSPS_NOTICE_H):
    """Sample a grid-service event realization over `datetimes` (a 5-min window).

    force: None (true marginal rate), or one of "dr"/"outage"/"psps" to guarantee >=1 such event
           inside the window (event-stratified training).
    Returns a dict of float arrays, each len == len(datetimes).
    """
    dt = pd.DatetimeIndex(pd.to_datetime(datetimes))
    n = len(dt)
    outage = np.zeros(n, dtype=np.float32)
    dr = np.zeros(n, dtype=np.float32)
    dr_adder = np.zeros(n, dtype=np.float32)
    dr_notice = np.zeros(n, dtype=np.float32)
    psps_warning = np.zeros(n, dtype=np.float32)
    if n == 0:
        return {"outage": outage, "dr": dr, "dr_price_adder": dr_adder,
                "dr_notice": dr_notice, "psps_warning": psps_warning}

    step_h = _step_hours(dt)
    steps_per_h = 1.0 / step_h
    window_days = n * step_h / 24.0
    hours = dt.hour + dt.minute / 60.0
    months = dt.month
    summer_mask = (months >= DR_MONTHS[0]) & (months <= DR_MONTHS[1])

    def dur_steps(dur_h):
        return max(1, int(round(dur_h * steps_per_h)))

    # ── ordinary outages: Poisson over the window at the annual rate ──
    n_ord = int(rng.poisson(ORDINARY_PER_YEAR * window_days / 365.0))
    if force == "outage":
        n_ord = max(1, n_ord)
    for _ in range(n_ord):
        s = int(rng.integers(0, n))
        _mark_block(outage, s, s + dur_steps(rng.uniform(*ORDINARY_DUR_H)))

    # ── PSPS: Bernoulli over the window, long duration, day-ahead warning ──
    p_psps = float(rng.uniform(*PSPS_ANNUAL_PROB)) * window_days / 365.0
    if force == "psps" or rng.random() < min(1.0, p_psps):
        s = int(rng.integers(0, n))
        d = dur_steps(rng.uniform(*PSPS_DUR_H))
        _mark_block(outage, s, s + d)
        _set_notice(psps_warning, s, dur_steps(psps_notice_h))

    # ── DR: summer 4-9pm critical-peak-price events with same-day notice ──
    if summer_mask.any():
        lo, hi = DR_PER_SUMMER.get(dr_case, DR_PER_SUMMER["moderate"])
        summer_days_in_window = float(summer_mask.sum()) * step_h / 24.0
        expected_dr = rng.uniform(lo, hi) * summer_days_in_window / SUMMER_DAYS
        n_dr = int(rng.poisson(max(0.0, expected_dr)))
        if force == "dr":
            n_dr = max(1, n_dr)
        # candidate onsets: summer steps at the window start hour (16:00)
        onset_ok = summer_mask & (hours >= DR_HOUR_WINDOW[0]) & (hours < DR_HOUR_WINDOW[0] + 1)
        cand = np.flatnonzero(onset_ok)
        if cand.size == 0:  # window has summer but no 16:00 boundary -> fall back to any summer step
            cand = np.flatnonzero(summer_mask & (hours >= DR_HOUR_WINDOW[0]) & (hours < DR_HOUR_WINDOW[1]))
        for _ in range(n_dr):
            if cand.size == 0:
                break
            s = int(rng.choice(cand))
            # clamp the event to the 4-9pm window
            end_h_cap = int(round((DR_HOUR_WINDOW[1] - hours[s]) * steps_per_h))
            d = min(dur_steps(rng.uniform(*DR_DUR_H)), max(1, end_h_cap))
            e = min(n, s + d)
            _mark_block(dr, s, e)
            dr_adder[s:e] = np.float32(dr_price_adder)
            _set_notice(dr_notice, s, dur_steps(dr_notice_h))

    # outage overrides DR (grid down -> no price event / no import)
    down = outage > 0
    dr[down] = 0.0
    dr_adder[down] = 0.0
    return {"outage": outage, "dr": dr, "dr_price_adder": dr_adder,
            "dr_notice": dr_notice, "psps_warning": psps_warning}


def choose_force(rng, event_episode_frac=0.5):
    """For event-stratified training: with prob `event_episode_frac` return a forced event type
    (DR/outage/PSPS, weighted toward the commoner DR/outage), else None (true-rate episode)."""
    if rng.random() >= event_episode_frac:
        return None
    return str(rng.choice(FORCE_TYPES, p=[0.5, 0.4, 0.1]))
