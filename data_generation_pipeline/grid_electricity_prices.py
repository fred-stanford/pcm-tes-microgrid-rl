"""
Grid buy (TOU) and sell (EEC export credit) electricity prices for preprocessing.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

TOU_PATH = "Data/TOU.xlsx"
EEC_PATH = "Data/EEC.xlsx"

# User-specified seasons (override TOU sheet winter label Oct–May)
SUMMER_MONTHS = {6, 7, 8, 9}
WINTER_MONTHS = {1, 2, 3, 4, 5, 10, 11, 12}

# PG&E Schedule E-ELEC "Total Bundled Rates" (Total Energy Rates $/kWh), from the E-ELEC tariff
# Sheet 2 (Cal. P.U.C. Sheet 61116-E), verified against the PG&E tariff PDF 2026-08-19.
# NOTE: live data reads Data/TOU.xlsx if present — keep that workbook in
# sync with these values (or remove it to use this default) so the regenerated dataset carries the
# current tariff. (Prior vintage was summer 0.527/0.375/0.323, winter 0.312/0.291/0.279.)
_DEFAULT_TOU_RATES = {
    "summer": {"Peak": 0.55214, "Part-Peak": 0.39026, "Off-Peak": 0.33358},
    "winter": {"Peak": 0.32063, "Part-Peak": 0.29854, "Off-Peak": 0.28468},
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """weekday: Monday=0; n=1 first, n=-1 last."""
    if n > 0:
        d = date(year, month, 1)
        shift = (weekday - d.weekday()) % 7
        d = d.replace(day=1 + shift + 7 * (n - 1))
        return d
    if month == 12:
        d = date(year + 1, 1, 1)
    else:
        d = date(year, month + 1, 1)
    from datetime import timedelta

    d = d - timedelta(days=1)
    shift = (d.weekday() - weekday) % 7
    return d.replace(day=d.day - shift)


def _federal_holidays(year: int) -> set[date]:
    """US federal holidays (no 'observed' shift)."""
    return {
        date(year, 1, 1),
        date(year, 6, 19),
        date(year, 7, 4),
        date(year, 11, 11),
        date(year, 12, 25),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _nth_weekday(year, 5, 0, -1),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _nth_weekday(year, 11, 3, 4),
    }


def is_weekend_or_holiday(ts: pd.Timestamp) -> bool:
    ts = pd.Timestamp(ts)
    if ts.dayofweek >= 5:
        return True
    return ts.date() in _federal_holidays(ts.year)


def tou_season(ts: pd.Timestamp) -> Literal["summer", "winter"]:
    m = pd.Timestamp(ts).month
    if m in SUMMER_MONTHS:
        return "summer"
    if m in WINTER_MONTHS:
        return "winter"
    raise ValueError(f"Month {m} not in TOU summer/winter calendar")


def tou_period(ts: pd.Timestamp) -> Literal["Peak", "Part-Peak", "Off-Peak"]:
    """
    PG&E-style TOU periods (from Data/TOU.xlsx):
    Peak 4–9 p.m.; Part-Peak 3–4 p.m. and 9 p.m.–12 a.m.; else Off-Peak.
    Uses clock hour + fraction so 5-minute steps map consistently.
    """
    ts = pd.Timestamp(ts)
    h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    if 16 <= h < 21:
        return "Peak"
    if 15 <= h < 16 or 21 <= h < 24:
        return "Part-Peak"
    return "Off-Peak"


def _parse_tou_rate(cell) -> float:
    text = str(cell).strip()
    return float(text.replace("$", "").replace("/kWh", ""))


@lru_cache(maxsize=1)
def load_tou_rates() -> Dict[str, Dict[str, float]]:
    """Read summer/winter Peak, Part-Peak, Off-Peak $/kWh from Data/TOU.xlsx."""
    try:
        df = pd.read_excel(TOU_PATH, sheet_name=0, header=None, engine="openpyxl")
        rates: Dict[str, Dict[str, float]] = {}
        for i in range(len(df)):
            season = str(df.iloc[i, 0]).strip().lower()
            if season not in ("summer", "winter") or season in rates:
                continue
            try:
                row = {
                    "Peak": _parse_tou_rate(df.iloc[i, 1]),
                    "Part-Peak": _parse_tou_rate(df.iloc[i, 2]),
                    "Off-Peak": _parse_tou_rate(df.iloc[i, 3]),
                }
            except (ValueError, TypeError):
                # A season-labelled row whose cells aren't rates (e.g. the season
                # date-range definition rows further down the sheet) — skip it, don't
                # let it abort the whole read and silently fall back to the default.
                continue
            rates[season] = row
        if set(rates) == {"summer", "winter"}:
            return rates
    except Exception:
        pass
    return {k: v.copy() for k, v in _DEFAULT_TOU_RATES.items()}


def grid_buy_price(ts: pd.Timestamp) -> float:
    rates = load_tou_rates()
    season = tou_season(ts)
    period = tou_period(ts)
    return rates[season][period]


def _eec_block_bounds(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Return (weekend_start_col, workday_start_col, n_cols)."""
    row1 = df.iloc[1].astype(str).str.strip()
    weekend_start = workday_start = None
    for c, v in enumerate(row1):
        if v == "Weekend/Holiday":
            weekend_start = c
        elif v == "Workday":
            workday_start = c
    if weekend_start is None or workday_start is None:
        raise ValueError("EEC sheet must contain 'Weekend/Holiday' and 'Workday' headers in row 2")
    return weekend_start, workday_start, df.shape[1]


def _parse_eec_block(
    df: pd.DataFrame,
    start_col: int,
    end_col: int,
) -> Dict[int, Dict[int, float]]:
    """
    Parse one EEC block between ``start_col`` (month label column) and ``end_col`` (exclusive).

    Layout per month pair: row 3 = Delivery, Generation; rows 4–27 = Hour-Beginning 0–23.
    Sell price = Delivery + Generation ($/kWh).
    """
    months: Dict[int, Dict[int, float]] = {}
    row2 = df.iloc[2]
    c = start_col
    while c < end_col - 1:
        m = row2.iloc[c]
        if pd.isna(m) or str(m).strip() == "":
            break
        month = int(float(m))
        del_col, gen_col = c, c + 1
        hours = {}
        for hour in range(24):
            r = 4 + hour
            del_v, gen_v = df.iloc[r, del_col], df.iloc[r, gen_col]
            if pd.isna(del_v) or pd.isna(gen_v):
                hours[hour] = np.nan
            else:
                hours[hour] = float(del_v) + float(gen_v)
        months[month] = hours
        c += 2
    if len(months) != 12:
        raise ValueError(f"EEC block at col {start_col}: expected 12 months, got {sorted(months.keys())}")
    return months


@lru_cache(maxsize=1)
def load_eec_sell_tables() -> tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    df = pd.read_excel(EEC_PATH, sheet_name=0, header=None, engine="openpyxl")
    weekend_start, workday_start, n_cols = _eec_block_bounds(df)
    weekend = _parse_eec_block(df, weekend_start, workday_start)
    workday = _parse_eec_block(df, workday_start, n_cols)
    return workday, weekend


def eec_hour_beginning(ts: pd.Timestamp) -> int:
    """Hour-Beginning index 0–23 for EEC row lookup."""
    return int(pd.Timestamp(ts).hour)


def grid_sell_price(ts: pd.Timestamp) -> float:
    """Sell price = Delivery + Generation (EEC), by schedule, month, and Hour-Beginning."""
    ts = pd.Timestamp(ts)
    workday, weekend = load_eec_sell_tables()
    table = weekend if is_weekend_or_holiday(ts) else workday
    price = table[int(ts.month)][eec_hour_beginning(ts)]
    if np.isnan(price):
        return 0.0
    return float(price)


def grid_prices_for_series(datetimes: pd.Series) -> pd.DataFrame:
    dts = pd.DatetimeIndex(pd.to_datetime(datetimes))
    buy = np.array([grid_buy_price(t) for t in dts], dtype=float)
    sell = np.array([grid_sell_price(t) for t in dts], dtype=float)
    return pd.DataFrame(
        {
            "Grid Electricity Buy Price": buy,
            "Grid Electricity Sell Price": sell,
        },
        index=dts,
    )
