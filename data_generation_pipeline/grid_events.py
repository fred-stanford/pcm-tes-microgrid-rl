"""
Stochastic grid outage and demand-response (DR) event timelines for preprocessing.

Event definitions follow PG&E-style ordinary/long outages and summer DR windows.
All series are aligned to a ``DateTime`` index at the simulation time resolution.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pandas as pd

DrCase = Literal["moderate", "severe"]

# Ordinary outage: higher weight in spring & autumn (storm / low-PV season)
_ORDINARY_MONTH_WEIGHTS = {
    1: 1.0,
    2: 1.0,
    3: 3.0,
    4: 3.0,
    5: 3.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 3.0,
    10: 3.0,
    11: 3.0,
    12: 1.0,
}

# Long outage / PSPS: winter storms + wildfire season
_LONG_MONTH_WEIGHTS = {
    1: 2.5,
    2: 2.5,
    3: 1.5,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 2.5,
    9: 3.0,
    10: 3.0,
    11: 3.0,
    12: 2.5,
}

DR_CASE_PARAMS = {
    "moderate": {
        "events_per_summer_range": (8, 10),
        "spacing_weeks": 2.0,
        "import_capacity_pct": 80.0,
    },
    "severe": {
        "events_per_summer_range": (12, 16),
        "spacing_weeks": 1.0,
        "import_capacity_pct_range": (50.0, 70.0),
    },
}


def _weighted_month(rng: np.random.Generator, weights: dict[int, float]) -> int:
    months = np.array(list(weights.keys()), dtype=int)
    w = np.array([weights[m] for m in months], dtype=float)
    w /= w.sum()
    return int(rng.choice(months, p=w))


def _mask_interval(
    datetimes: pd.DatetimeIndex,
    start: pd.Timestamp,
    duration_hours: float,
) -> np.ndarray:
    """Contiguous [start, start + duration) in simulation time — half-open interval."""
    if duration_hours <= 0:
        return np.zeros(len(datetimes), dtype=bool)
    end = start + pd.Timedelta(hours=duration_hours)
    return (datetimes >= start) & (datetimes < end)


def _clip_event_to_simulation(
    datetimes: pd.DatetimeIndex,
    start: pd.Timestamp,
    duration_hours: float,
) -> tuple[pd.Timestamp, float] | None:
    """Clip event to simulation window; return None if no overlap."""
    sim_start = datetimes[0]
    sim_end = datetimes[-1] + (datetimes[1] - datetimes[0] if len(datetimes) > 1 else pd.Timedelta(minutes=5))
    end = start + pd.Timedelta(hours=duration_hours)
    if end <= sim_start or start >= sim_end:
        return None
    clipped_start = max(start, sim_start)
    clipped_end = min(end, sim_end)
    clipped_dur = (clipped_end - clipped_start).total_seconds() / 3600.0
    if clipped_dur <= 0:
        return None
    return clipped_start, clipped_dur


def _apply_outage(
    dt_index: pd.DatetimeIndex,
    outage_mask: np.ndarray,
    start: pd.Timestamp,
    duration_hours: float,
) -> None:
    clipped = _clip_event_to_simulation(dt_index, start, duration_hours)
    if clipped is None:
        return
    start_c, dur_c = clipped
    outage_mask |= _mask_interval(dt_index, start_c, dur_c)


def _apply_dr(
    dt_index: pd.DatetimeIndex,
    outage_mask: np.ndarray,
    dr_mask: np.ndarray,
    dr_capacity_at_step: np.ndarray,
    start: pd.Timestamp,
    duration_hours: float,
    cap_pct: float,
) -> None:
    clipped = _clip_event_to_simulation(dt_index, start, duration_hours)
    if clipped is None:
        return
    start_c, dur_c = clipped
    m = _mask_interval(dt_index, start_c, dur_c)
    dr_mask |= m  # DR event flag is continuous for the scheduled window
    active_cap = m & ~outage_mask  # import limit only applies when grid is up
    if not active_cap.any():
        return
    existing = dr_capacity_at_step[active_cap]
    dr_capacity_at_step[active_cap] = np.where(
        np.isnan(existing),
        cap_pct,
        np.minimum(existing, cap_pct),
    )


def _random_time_in_month(
    rng: np.random.Generator,
    year: int,
    month: int,
) -> pd.Timestamp:
    """Uniform random instant within calendar month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1)
    offset_sec = rng.uniform(0, (end - start).total_seconds())
    return start + pd.Timedelta(seconds=offset_sec)


def _schedule_ordinary_outages(
    rng: np.random.Generator,
    year: int,
    events_per_year: float = 2.0,
) -> list[tuple[pd.Timestamp, float]]:
    """~2 events/year, ~2.5 h each; Poisson(2) => ~86% chance of >=1 outage/year."""
    n_events = int(rng.poisson(events_per_year))
    scheduled = []
    for _ in range(n_events):
        month = _weighted_month(rng, _ORDINARY_MONTH_WEIGHTS)
        start = _random_time_in_month(rng, year, month)
        duration_h = float(rng.uniform(2.0, 3.0))
        scheduled.append((start, duration_h))
    return scheduled


def _schedule_long_outage(
    rng: np.random.Generator,
    year: int,
    annual_chance_range: tuple[float, float] = (0.10, 0.25),
) -> tuple[pd.Timestamp, float] | None:
    """At most one long outage per year; annual probability in ``annual_chance_range``."""
    p = float(rng.uniform(*annual_chance_range))
    if rng.random() >= p:
        return None
    month = _weighted_month(rng, _LONG_MONTH_WEIGHTS)
    start = _random_time_in_month(rng, year, month)
    duration_h = float(rng.uniform(24.0, 48.0))
    return start, duration_h


def _schedule_dr_events(
    rng: np.random.Generator,
    year: int,
    dr_case: DrCase,
) -> list[tuple[pd.Timestamp, float, float]]:
    """Summer DR (Jun–Sep): 4–9 p.m., 4–5 h per event."""
    params = DR_CASE_PARAMS[dr_case]
    lo, hi = params["events_per_summer_range"]
    n_events = int(rng.integers(lo, hi + 1))

    summer_start = pd.Timestamp(year=year, month=6, day=1)
    summer_end = pd.Timestamp(year=year, month=10, day=1)
    summer_days = (summer_end - summer_start).days
    spacing_days = params["spacing_weeks"] * 7.0
    capacity_pct = params.get("import_capacity_pct")

    first_offset = rng.uniform(0, spacing_days)
    scheduled: list[tuple[pd.Timestamp, float, float]] = []
    for k in range(n_events):
        day_offset = first_offset + k * spacing_days + rng.uniform(-1.5, 1.5)
        day_offset = float(np.clip(day_offset, 0, max(summer_days - 1, 0)))
        day = summer_start + pd.Timedelta(days=day_offset)
        start = day.normalize() + pd.Timedelta(
            hours=16, minutes=int(rng.integers(0, 60))
        )
        if start >= summer_end:
            continue
        duration_h = float(rng.uniform(4.0, 5.0))
        window_end = day.normalize() + pd.Timedelta(hours=21)
        end = min(start + pd.Timedelta(hours=duration_h), window_end)
        duration_h = (end - start).total_seconds() / 3600.0
        if duration_h < 5 / 60:
            continue
        if capacity_pct is None:
            cap_lo, cap_hi = params["import_capacity_pct_range"]
            cap = float(rng.uniform(cap_lo, cap_hi))
        else:
            cap = float(capacity_pct)
        scheduled.append((start, duration_h, cap))
    return scheduled


def generate_for_datetimes(
    datetimes: Union[pd.Series, pd.DatetimeIndex, list],
    random_seed: int = 1,
    dr_case: DrCase = "moderate",
    ordinary_events_per_year: float = 2.0,
    long_outage_annual_chance_range: tuple[float, float] = (0.10, 0.25),
) -> pd.DataFrame:
    """
    Build per-timestep grid outage / DR flags and available import capacity.

    Returns
    -------
    DataFrame indexed like ``datetimes`` with:
        Grid Outage Event (0/1)
        DR Event (0/1)
        Available Grid Import Capacity (%) — 0 in outage; DR cap otherwise; 100 default
    """
    dt_index = pd.DatetimeIndex(pd.to_datetime(datetimes))
    n = len(dt_index)
    outage_mask = np.zeros(n, dtype=bool)
    dr_mask = np.zeros(n, dtype=bool)
    capacity = np.full(n, 100.0, dtype=float)
    dr_capacity_at_step = np.full(n, np.nan, dtype=float)

    rng = np.random.default_rng(random_seed)

    for year in sorted(dt_index.year.unique()):
        for start, dur in _schedule_ordinary_outages(
            rng, year, events_per_year=ordinary_events_per_year
        ):
            _apply_outage(dt_index, outage_mask, start, dur)

        long_ev = _schedule_long_outage(rng, year, long_outage_annual_chance_range)
        if long_ev is not None:
            _apply_outage(dt_index, outage_mask, long_ev[0], long_ev[1])

        for start, dur, cap_pct in _schedule_dr_events(rng, year, dr_case):
            _apply_dr(dt_index, outage_mask, dr_mask, dr_capacity_at_step, start, dur, cap_pct)

    capacity[outage_mask] = 0.0
    has_dr_cap = ~np.isnan(dr_capacity_at_step)
    capacity[has_dr_cap & ~outage_mask] = dr_capacity_at_step[has_dr_cap & ~outage_mask]

    return pd.DataFrame(
        {
            "Grid Outage Event": outage_mask.astype(np.int8),
            "DR Event": dr_mask.astype(np.int8),
            "Available Grid Import Capacity": capacity,
        },
        index=dt_index,
    )
