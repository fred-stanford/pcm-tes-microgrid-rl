import pandas as pd
import numpy as np
import os
import glob
import random
import requests
from datetime import datetime, timedelta

import Input_Parameters
from physics import Utility_functions
from physics import RuleBasedControl

# Open-Meteo 15-min archived forecast API (used by archived_forecast_for_passive_load)
_OPENMETEO_MINUTELY_15_VARS = (
    "temperature_2m,relative_humidity_2m,cloud_cover,"
    "wind_speed_10m,"
    "shortwave_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant"
)


def _wind_speed_2m(wind_10m_ms, z0=0.03):
    """Convert 10 m wind speed to 2 m using log profile, to match NSRDB (wind at 2 m)."""
    z2, z10 = 2.0, 10.0
    return wind_10m_ms * (np.log(z2 / z0) / np.log(z10 / z0))


def _fetch_openmeteo_archived_15min(start_date, end_date, latitude, longitude, timezone="auto"):
    """Fetch 15-min archived forecast from Open-Meteo. start_date/end_date as 'YYYY-MM-DD'."""
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&start_minutely_15={start_date}T00:00&end_minutely_15={end_date}T23:45"
        f"&minutely_15={_OPENMETEO_MINUTELY_15_VARS}&timezone={timezone}&wind_speed_unit=ms"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _openmeteo_json_to_raw_df(data):
    """Convert Open-Meteo minutely_15 JSON to NSRDB-like raw DataFrame (index = time)."""
    key = "minutely_15"
    if key not in data:
        return None
    df = pd.DataFrame(data[key])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    out = pd.DataFrame(index=df.index)
    out["Year"] = df.index.year
    out["Month"] = df.index.month
    out["Day"] = df.index.day
    out["Hour"] = df.index.hour
    out["Minute"] = df.index.minute
    out["Temperature"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
    w10 = pd.to_numeric(df["wind_speed_10m"], errors="coerce")
    out["Wind Speed"] = _wind_speed_2m(w10)  # 2 m to match NSRDB
    out["GHI"] = pd.to_numeric(df.get("shortwave_radiation_instant", 0), errors="coerce").fillna(0)
    out["DHI"] = pd.to_numeric(df.get("diffuse_radiation_instant", 0), errors="coerce").fillna(0)
    out["DNI"] = pd.to_numeric(df.get("direct_normal_irradiance_instant", 0), errors="coerce").fillna(0)
    out["Relative Humidity"] = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
    # API cloud_cover is 0-100 (%); stored as "Cloud Type" for prepare_archived_forecast (which converts to Cloud 0-1)
    out["Cloud Type"] = pd.to_numeric(df["cloud_cover"], errors="coerce")
    out["dhi"] = out["DHI"].copy()
    out["dni"] = out["DNI"].copy()
    return out


# Open-Meteo Historical Weather API (archive): hourly reanalysis (ERA5/IFS)
_OPENMETEO_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,"
    "shortwave_radiation,diffuse_radiation,direct_normal_irradiance"
)


def _fetch_openmeteo_archive_hourly(start_date, end_date, latitude, longitude, timezone="auto"):
    """Fetch hourly historical weather from Open-Meteo Archive API (reanalysis)."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={_OPENMETEO_HOURLY_VARS}"
        f"&timezone={timezone}&wind_speed_unit=ms"
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def _openmeteo_hourly_json_to_raw_df(data):
    """Convert Open-Meteo hourly (archive) JSON to raw DataFrame for prepare_archived_forecast."""
    if "hourly" not in data:
        return None
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    out = pd.DataFrame(index=df.index)
    out["Year"] = df.index.year
    out["Month"] = df.index.month
    out["Day"] = df.index.day
    out["Hour"] = df.index.hour
    out["Minute"] = df.index.minute  # 0 for hourly
    out["Temperature"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
    w10 = pd.to_numeric(df["wind_speed_10m"], errors="coerce")
    out["Wind Speed"] = _wind_speed_2m(w10)
    out["GHI"] = pd.to_numeric(df.get("shortwave_radiation", 0), errors="coerce").fillna(0)
    out["DHI"] = pd.to_numeric(df.get("diffuse_radiation", 0), errors="coerce").fillna(0)
    out["DNI"] = pd.to_numeric(df.get("direct_normal_irradiance", 0), errors="coerce").fillna(0)
    out["Relative Humidity"] = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
    out["Cloud Type"] = pd.to_numeric(df["cloud_cover"], errors="coerce")
    out["dhi"] = out["DHI"].copy()
    out["dni"] = out["DNI"].copy()
    return out


def fetch_historical_weather_openmeteo(
    start_date,
    end_date,
    time_resolution,
    latitude=None,
    longitude=None,
    timezone_offset=None,
    timezone="auto",
):
    """
    Fetch historical weather from Open-Meteo Archive API (hourly) and return a DataFrame
    in the same format as prepare_NSRDB + upsample: same columns (Ta, WS, RH, GHI, DHI, DNI,
    Cloud, k_t, SolarTime, DayStatus, DateTime) at the requested time_resolution (e.g. 5 min).

    Same transformation chain as archived forecast: cloud % -> Cloud 0-1, clearsky GHI -> k_t,
    then upsampled from hourly to time_resolution via reindex + ffill.

    Parameters
    ----------
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD' (inclusive).
    time_resolution : float
        Target interval in minutes (e.g. 5).
    latitude, longitude, timezone_offset : optional
        Default from Input_Parameters.
    timezone : str
        Open-Meteo timezone (default "auto").

    Returns
    -------
    pd.DataFrame
        Weather at time_resolution with DateTime column; same columns as sampled_weather.
    """
    if latitude is None:
        latitude = Input_Parameters.latitude
    if longitude is None:
        longitude = Input_Parameters.longitude
    if timezone_offset is None:
        timezone_offset = Input_Parameters.timezone_offset

    data = _fetch_openmeteo_archive_hourly(start_date, end_date, latitude, longitude, timezone)
    raw = _openmeteo_hourly_json_to_raw_df(data)
    if raw is None or raw.empty:
        raise ValueError(f"No hourly data from Open-Meteo for {start_date} to {end_date}")

    df_prepared = prepare_archived_forecast(raw, latitude, longitude, timezone_offset)

    # Upsample from hourly to time_resolution (e.g. 5 min): reindex to fine grid, then linear interpolation between hourly values
    df_prepared = df_prepared.set_index("DateTime")
    t_start = df_prepared.index.min()
    t_end = df_prepared.index.max()
    new_index = pd.date_range(
        start=t_start,
        end=t_end,
        freq=pd.Timedelta(minutes=float(time_resolution)),
    )
    df_upsampled = df_prepared.reindex(new_index)
    # Linear interpolation between hourly points for numeric columns
    num_cols = df_upsampled.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df_upsampled[num_cols] = df_upsampled[num_cols].interpolate(method="linear")
    # Fill any remaining NaN (edges, or non-numeric columns) with ffill/bfill
    df_upsampled = df_upsampled.ffill().bfill()
    df_upsampled = df_upsampled.reset_index().rename(columns={"index": "DateTime"})

    return df_upsampled


def prepare_live_forecast(df_raw):

    # Rename columns
    df_raw.rename(columns={
        'temp_c': 'Ta',
        'wind_speed_mph': 'WS',
        'humidity_%': 'RH',
        'dew_point_c': 'Td',
        'ghi': 'GHI',
        'dni': 'DNI',
        'dhi': 'DHI',
    }, inplace=True)

    # convert to m/s
    df_raw['WS'] = df_raw['WS'] * 0.44704

    # Convert string to datetime with timezone awareness
    df_raw["DateTime"] = df_raw["timestamp_utc"].dt.tz_localize(None)

    current_time = df_raw["DateTime"].iloc[0]
    timezone_offset = Input_Parameters.timezone_offset  # Default: -8 (PST)

    if Utility_functions.is_dst("America/Los_Angeles", current_time):
        timezone_offset += 1  # Adjust to -7 for DST
    
    # Add additional columns
    df_raw = Utility_functions.add_clearness_index(df_raw, Input_Parameters.latitude, Input_Parameters.longitude, timezone_offset)
    df_raw = Utility_functions.add_solar_time_NSRDB(df_raw, timezone_offset, Input_Parameters.longitude)
    df_raw = Utility_functions.add_day_status(df_raw, Input_Parameters.latitude)
    df_raw = Utility_functions.add_cloudcover_epw(df_raw)
    
    return df_raw

def read_NSRDB_HMB(data_dir, weather_year):
    """
    Read NSRDB CSV file for a specific year.
    
    Parameters:
    -----------
    data_dir : str
        Directory path containing the NSRDB CSV files
    weather_year : int
        Year for which to read the data (e.g., 2024, 2023, etc.)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the NSRDB weather data
        
    Raises:
    -------
    FileNotFoundError
        If the file for the specified year is not found
    """
    # Construct the file pattern based on the year
    # Files follow pattern: 57357_37.46_-122.43_YYYY.csv
    file_pattern = os.path.join(data_dir, f"*_{weather_year}.csv")
    
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(
            f"No file found for year '{weather_year}' in folder '{data_dir}'. "
            f"Tried pattern: {file_pattern}"
        )
    # If multiple files match, take the first one.
    file_path = files[0]

    # Read the CSV file; data starts at row 3 (header=2)
    df = pd.read_csv(file_path, header=2)
    return df

def prepare_NSRDB(df_raw):
    # Rename columns
    df_raw = df_raw.rename(columns={
        "Temperature": "Ta",
        "Wind Speed": "WS",
        "Relative Humidity": "RH"
    })

    # Create the DateTime column
    df_raw["DateTime"] = pd.to_datetime(df_raw[["Year", "Month", "Day", "Hour", "Minute"]])

    # Apply additional transformations (assuming these are defined functions)
    df_raw = Utility_functions.add_clearness_index(df_raw, Input_Parameters.latitude, Input_Parameters.longitude, Input_Parameters.timezone_offset)
    df_raw = Utility_functions.add_solar_time_NSRDB(df_raw, Input_Parameters.timezone_offset, Input_Parameters.longitude)
    df_raw = Utility_functions.add_day_status(df_raw, Input_Parameters.latitude)
    df_raw = Utility_functions.add_cloudcover(df_raw)

    df_raw.drop(columns=['dhi'], inplace=True)
    df_raw.drop(columns=['dni'], inplace=True)
    
    return df_raw


def prepare_archived_forecast(df_raw, latitude=None, longitude=None, timezone_offset=None):
    """
    Prepare 15-min archived forecast (e.g. Open-Meteo) into the same format as prepare_NSRDB
    for use with passive_load_forecast and downstream pipelines.

    Differences from NSRDB:
    - Cloud: archived forecast has cloud cover in percentage (0-100), not cloud type (0-9).
      We set Cloud = cloud_cover_pct / 100 (fraction 0-1).
    - k_t: clear-sky GHI is computed from DateTime and lat/lon (pvlib Ineichen); then
      k_t = min(observed_GHI / clear_sky_GHI, 1) when sun is up, 0 at night. GHI is kept
      as the API’s observed shortwave radiation.

    Expects df_raw with columns: Year, Month, Day, Hour, Minute, Temperature, Wind Speed,
    Relative Humidity, GHI, DHI, DNI, Cloud Type (0-100, from API cloud_cover %). Optional: dhi, dni (dropped).
    """
    if latitude is None:
        latitude = Input_Parameters.latitude
    if longitude is None:
        longitude = Input_Parameters.longitude
    if timezone_offset is None:
        timezone_offset = Input_Parameters.timezone_offset

    df_raw = df_raw.rename(columns={
        "Temperature": "Ta",
        "Wind Speed": "WS",
        "Relative Humidity": "RH",
    })

    df_raw["DateTime"] = pd.to_datetime(df_raw[["Year", "Month", "Day", "Hour", "Minute"]])

    # Cloud: archived forecast API gives cloud_cover in percentage (0-100). Convert to fraction 0-1
    # to match NSRDB output (NSRDB uses add_cloudcover to convert Cloud Type 0-9 -> Cloud fraction 0-1).
    # Do not use add_cloudcover here—that expects integer cloud type 0-9, not percentage.
    if "Cloud Type" in df_raw.columns:
        cloud_pct = pd.to_numeric(df_raw["Cloud Type"], errors="coerce").fillna(0)
        df_raw["Cloud"] = (cloud_pct / 100.0).clip(0.0, 1.0)
        df_raw.drop(columns=["Cloud Type"], inplace=True)

    # k_t = GHI (observed) / clearsky GHI, same as NSRDB (add_clearness_index). NSRDB has clearsky in data;
    # archived forecast does not, so we compute clearsky GHI first, then k_t.
    clearsky_ghi = Utility_functions.compute_clearsky_ghi(df_raw, latitude, longitude, timezone_offset)
    observed_ghi = np.asarray(pd.to_numeric(df_raw["GHI"], errors="coerce").fillna(0).values, dtype=float)
    # Avoid divide-by-zero and invalid value warnings: only divide where clearsky is finite and > 10
    valid = np.isfinite(clearsky_ghi) & (clearsky_ghi > 10)
    ratio = np.where(valid, observed_ghi / np.maximum(clearsky_ghi, 1e-10), 0.0)
    df_raw["k_t"] = np.minimum(ratio, 1.0)
    # GHI stays as observed (API shortwave radiation); do not replace with clearsky

    df_raw = Utility_functions.add_solar_time_NSRDB(df_raw, timezone_offset, longitude)
    df_raw = Utility_functions.add_day_status(df_raw, latitude)

    for col in ["dhi", "dni"]:
        if col in df_raw.columns:
            df_raw.drop(columns=[col], inplace=True)

    return df_raw


def archived_forecast_for_passive_load(
    start_time,
    horizon_days=6,
    latitude=None,
    longitude=None,
    timezone_offset=None,
    time_resolution=None,
):
    """
    Fetch the 15-min archived forecast from Open-Meteo for the given start time and return
    a DataFrame in the same format as the weather input for passive_load_forecast.

    Single entry point: only start_time is required; location and resolution default to
    Input_Parameters. The forecast is issued for the calendar day of start_time and covers
    horizon_days (default 6) from that day. Data is expanded to 5-min, prepared with
    prepare_archived_forecast (cloud % and clear-sky GHI), then upsampled to time_resolution.

    Parameters
    ----------
    start_time : datetime, date, or str
        Issue date of the forecast. Can be datetime (e.g. 2024-01-15 00:00), date, or
        "YYYY-MM-DD". Only the date part is used to request the archived forecast.
    horizon_days : int, default 6
        Number of days of forecast to fetch (e.g. 6 so that "next 5 days from now" fits
        when "now" is mid-day).
    latitude : float, optional
        Site latitude. Default: Input_Parameters.latitude.
    longitude : float, optional
        Site longitude. Default: Input_Parameters.longitude.
    timezone_offset : float, optional
        Local timezone offset from UTC in hours. Default: Input_Parameters.timezone_offset.
    time_resolution : float, optional
        Target interval in minutes for the output (e.g. 5). Default: Input_Parameters.δt / 60.

    Returns
    -------
    pd.DataFrame
        Weather DataFrame with DateTime column and columns expected by passive_load_forecast
        (Ta, WS, RH, GHI, k_t, SolarTime, DayStatus, Cloud, etc.), at time_resolution.
    """
    if latitude is None:
        latitude = Input_Parameters.latitude
    if longitude is None:
        longitude = Input_Parameters.longitude
    if timezone_offset is None:
        timezone_offset = Input_Parameters.timezone_offset
    if time_resolution is None:
        time_resolution = Input_Parameters.δt / 60

    issue_dt = pd.Timestamp(start_time)
    issue_date = issue_dt.date() if hasattr(issue_dt, "date") else issue_dt
    if hasattr(issue_date, "strftime"):
        start_date = issue_date.strftime("%Y-%m-%d")
    else:
        start_date = pd.Timestamp(issue_date).strftime("%Y-%m-%d")
    end_dt = issue_dt + timedelta(days=horizon_days - 1)
    end_date = end_dt.strftime("%Y-%m-%d") if hasattr(end_dt, "strftime") else pd.Timestamp(end_dt).strftime("%Y-%m-%d")

    raw_json = _fetch_openmeteo_archived_15min(start_date, end_date, latitude, longitude, timezone="auto")
    df_raw = _openmeteo_json_to_raw_df(raw_json)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"No 15-min archived forecast data returned for {start_date}–{end_date}")

    idx_5min = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq="5min")
    df_5min = df_raw.reindex(idx_5min).ffill()
    df_5min = df_5min.reset_index(drop=True)

    df_prepared = prepare_archived_forecast(
        df_5min.copy(),
        latitude=latitude,
        longitude=longitude,
        timezone_offset=timezone_offset,
    )
    df_sampled = upsample_dataframe(
        df_prepared,
        target_interval_minutes=time_resolution,
        datetime_col="DateTime",
        method="mean",
    )
    # Linearly fill any NaN from 15-min -> 5-min (or other) resampling so downstream plots and models get continuous series
    num_cols = df_sampled.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df_sampled[num_cols] = df_sampled[num_cols].interpolate(method="linear").ffill().bfill()
    return df_sampled


def archived_forecast_and_nsrdb_at_start(
    start_time,
    nsrdb_data_dir="NSRDB_data",
    forecast_hours=120,
    time_resolution=5,
    latitude=None,
    longitude=None,
    timezone_offset=None,
):
    """
    Read start time and return (1) archived forecast at time_resolution (min) sliced from
    start_time for the next forecast_hours (default 120 h = 5 days), and (2) the single
    row of NSRDB data at that same start time. Both have the same columns.

    Example: start_time = X day Y hour → forecast from X day Y hour for the next 120 h;
    NSRDB row is the observation at X day Y hour.

    Parameters
    ----------
    start_time : datetime, or str
        Start instant (e.g. 2022-06-01 14:30:00). Parsed with pd.Timestamp.
    nsrdb_data_dir : str, default "NSRDB_data"
        Directory containing NSRDB CSV files (e.g. *_2022.csv).
    forecast_hours : int, default 120
        Length of forecast window in hours (120 = 5 days).
    time_resolution : float, default 5
        Time step in minutes for forecast and NSRDB output (e.g. 5 = 5 min).
    latitude, longitude, timezone_offset : optional
        Passed to archived forecast; default from Input_Parameters.

    Returns
    -------
    forecast_df : pd.DataFrame
        Archived forecast at time_resolution (min) from start_time for the next forecast_hours; index = DateTime.
    nsrdb_start_row : pd.DataFrame
        One row of NSRDB data at start_time (same columns as forecast_df).
    """
    start_dt = pd.Timestamp(start_time)
    end_dt = start_dt + timedelta(hours=forecast_hours)

    # --- Archived forecast: fetch 6 days from issue date, slice to [start_dt, start_dt + forecast_hours) ---
    df_forecast = archived_forecast_for_passive_load(
        start_dt,
        horizon_days=6,
        time_resolution=time_resolution,
        latitude=latitude,
        longitude=longitude,
        timezone_offset=timezone_offset,
    )
    df_forecast = df_forecast.set_index("DateTime")
    if df_forecast.index.tz is not None:
        df_forecast.index = df_forecast.index.tz_localize(None)
    df_forecast = df_forecast.loc[(df_forecast.index >= start_dt) & (df_forecast.index < end_dt)]

    # --- NSRDB: load years covering [start_dt, end_dt), prepare, 5 min, then take row at start_time ---
    years_needed = list(range(start_dt.year, end_dt.year + 1))
    chunks = []
    for yr in years_needed:
        try:
            chunks.append(read_NSRDB_HMB(nsrdb_data_dir, yr))
        except FileNotFoundError:
            continue
    if not chunks:
        raise FileNotFoundError(f"No NSRDB data in {nsrdb_data_dir} for years {years_needed}")
    nsrdb_raw = pd.concat(chunks, ignore_index=True)
    nsrdb_raw["DateTime"] = pd.to_datetime(nsrdb_raw[["Year", "Month", "Day", "Hour", "Minute"]])
    mask = (nsrdb_raw["DateTime"] >= start_dt) & (nsrdb_raw["DateTime"] < end_dt)
    nsrdb_raw = nsrdb_raw.loc[mask].copy()
    nsrdb_prepared = prepare_NSRDB(nsrdb_raw)
    df_nsrdb = upsample_dataframe(nsrdb_prepared, time_resolution, datetime_col="DateTime", method="mean")
    df_nsrdb = df_nsrdb.set_index("DateTime")
    if "Clearsky GHI" in df_nsrdb.columns:
        df_nsrdb = df_nsrdb.drop(columns=["Clearsky GHI"])
    if df_nsrdb.index.tz is not None:
        df_nsrdb.index = df_nsrdb.index.tz_localize(None)
    # Row at start_time (nearest 5-min if no exact match)
    idx_nearest = df_nsrdb.index.get_indexer([start_dt], method="nearest")[0]
    nsrdb_start_row = df_nsrdb.iloc[idx_nearest : idx_nearest + 1].copy()

    # --- Same columns for both ---
    common_cols = sorted(set(df_forecast.columns) & set(nsrdb_start_row.columns))
    if not common_cols:
        raise ValueError("No common columns between archived forecast and NSRDB.")
    forecast_df = df_forecast[common_cols].copy()
    nsrdb_start_row = nsrdb_start_row[common_cols].copy()

    # Return with DateTime as column so downstream (e.g. Solar_Generation.generate_pv_onsite) can use weather["DateTime"]
    forecast_df = forecast_df.reset_index()
    nsrdb_start_row = nsrdb_start_row.reset_index()

    return forecast_df, nsrdb_start_row


def get_nsrdb_row_at_start(
    start_time,
    nsrdb_data_dir="NSRDB_data",
    time_resolution=5,
):
    """
    Return a single row of NSRDB data at start_time (no API call).
    Useful when you already have a 6-day forecast and only need the observed row at the current time.

    Parameters
    ----------
    start_time : datetime or str
        Time for which to get the NSRDB row.
    nsrdb_data_dir : str
        Directory containing NSRDB CSV files.
    time_resolution : float
        Time step in minutes (must match the resolution of your forecast).

    Returns
    -------
    pd.DataFrame
        One row with DateTime as column; same column set as prepared NSRDB (Clearsky GHI dropped if present).
    """
    start_dt = pd.Timestamp(start_time)
    years_needed = [start_dt.year]
    if start_dt.month == 12 and start_dt.day == 31:
        years_needed.append(start_dt.year + 1)
    chunks = []
    for yr in years_needed:
        try:
            chunks.append(read_NSRDB_HMB(nsrdb_data_dir, yr))
        except FileNotFoundError:
            continue
    if not chunks:
        raise FileNotFoundError(f"No NSRDB data in {nsrdb_data_dir} for years {years_needed}")
    nsrdb_raw = pd.concat(chunks, ignore_index=True)
    nsrdb_raw["DateTime"] = pd.to_datetime(nsrdb_raw[["Year", "Month", "Day", "Hour", "Minute"]])
    nsrdb_prepared = prepare_NSRDB(nsrdb_raw)
    df_nsrdb = upsample_dataframe(
        nsrdb_prepared, time_resolution, datetime_col="DateTime", method="mean"
    )
    df_nsrdb = df_nsrdb.set_index("DateTime")
    if "Clearsky GHI" in df_nsrdb.columns:
        df_nsrdb = df_nsrdb.drop(columns=["Clearsky GHI"])
    if df_nsrdb.index.tz is not None:
        df_nsrdb.index = df_nsrdb.index.tz_localize(None)
    idx_nearest = df_nsrdb.index.get_indexer([start_dt], method="nearest")[0]
    row = df_nsrdb.iloc[idx_nearest : idx_nearest + 1].copy()
    return row.reset_index()


def slice_forecast_to_horizon(
    forecast_6day, start_time, forecast_hours, columns=None, num_steps=None, time_resolution=None
):
    """
    Slice a 6-day forecast DataFrame to [start_time, start_time + forecast_hours).
    Optionally reindex to exactly num_steps rows so it matches E_load_forecast length.

    Parameters
    ----------
    forecast_6day : pd.DataFrame
        Full 6-day forecast with DateTime column.
    start_time : datetime or str
        Start of the desired window.
    forecast_hours : int or float
        Length of the window in hours (e.g. 120 for 5 days).
    columns : list, optional
        If provided, return only these columns (one copy, less memory).
    num_steps : int, optional
        If provided with time_resolution, reindex to exactly this many rows (fixes 1439 vs 1440 mismatch).
    time_resolution : float, optional
        Time step in minutes; required if num_steps is set.

    Returns
    -------
    pd.DataFrame
        Sliced forecast; same columns as forecast_6day or columns if given.
    """
    start_dt = pd.Timestamp(start_time)
    end_dt = start_dt + timedelta(hours=float(forecast_hours))
    mask = (forecast_6day["DateTime"] >= start_dt) & (forecast_6day["DateTime"] < end_dt)
    out = forecast_6day.loc[mask]
    if columns is not None:
        out = out[columns].copy()
    else:
        out = out.copy()
    if num_steps is not None and time_resolution is not None and len(out) != num_steps:
        out = out.set_index("DateTime")
        expected_index = pd.date_range(
            start=start_dt, periods=num_steps, freq=pd.Timedelta(minutes=float(time_resolution))
        )
        out = out.reindex(expected_index).ffill().bfill()
        out = out.reset_index().rename(columns={"index": "DateTime"})
        if columns is not None:
            out = out[columns]
    return out


def p2e(M_States):
    """
    Calculate energy consumption and thermal energy delivery using trapezoidal integration.
    Handles uneven time intervals correctly.
    
    Parameters:
    M_States: pandas DataFrame with time-series data
    
    Returns:
    E: list of energy values [kWh] in order:
       [Fan, FCU, Heat Pump, Cold PCM, Hot PCM, Pump 1, Pump 2, Thermal Power Delivered]
    """
    
    msize = len(M_States)
    
    # Calculate time intervals in hours (handles uneven spacing)
    time_intervals = np.zeros(msize - 1)
    for i in range(msize - 1):
        time_intervals[i] = (M_States.iloc[i + 1]['Time'] - M_States.iloc[i]['Time']) / 3600  # [hr]

    # Energy calculations using trapezoidal integration with actual time intervals
    E_Pump_1_0 = sum(((M_States.iloc[i + 1]['Pump 1 Electric Power'] + 
                       M_States.iloc[i]['Pump 1 Electric Power']) / 2) * time_intervals[i] 
                     for i in range(msize - 1)) / 1000  # [kWh]

    E_Pump_2_0 = sum(((M_States.iloc[i + 1]['Pump 2 Electric Power'] + 
                       M_States.iloc[i]['Pump 2 Electric Power']) / 2) * time_intervals[i] 
                     for i in range(msize - 1)) / 1000  # [kWh]

    E_HP_0 = sum(((M_States.iloc[i + 1]['HP Electric Power'] + 
                   M_States.iloc[i]['HP Electric Power']) / 2) * time_intervals[i] 
                 for i in range(msize - 1)) / 1000  # [kWh]

    Thermal_Power_Delivered_0 = sum(((M_States.iloc[i + 1]['HP Useful Thermal Power'] + 
                                      M_States.iloc[i]['HP Useful Thermal Power']) / 2) * time_intervals[i] 
                                    for i in range(msize - 1)) / 1000  # [kWh]

    E_Fan_0 = sum(((M_States.iloc[i + 1]['Fan Coil Fan Electric Power'] + 
                    M_States.iloc[i]['Fan Coil Fan Electric Power']) / 2) * time_intervals[i] 
                  for i in range(msize - 1)) / 1000  # [kWh]

    E_PCM_C_0 = sum(((M_States.iloc[i + 1]['Heat Transfer Cold PCM'] + 
                      M_States.iloc[i]['Heat Transfer Cold PCM']) / 2) * time_intervals[i] 
                    for i in range(msize - 1)) / 1000  # [kWh]

    E_PCM_H_0 = sum(((M_States.iloc[i + 1]['Heat Transfer Hot PCM'] + 
                      M_States.iloc[i]['Heat Transfer Hot PCM']) / 2) * time_intervals[i] 
                    for i in range(msize - 1)) / 1000  # [kWh]

    E_FCU_0 = sum(((M_States.iloc[i + 1]['FCU Heat Delivered'] + 
                    M_States.iloc[i]['FCU Heat Delivered']) / 2) * time_intervals[i] 
                  for i in range(msize - 1)) / 1000  # [kWh]
    
    E_dict = {
        'Fan Coil Fan Electric Power': E_Fan_0,
        'FCU Heat Delivered': E_FCU_0, 
        'HP Electric Power': E_HP_0,
        'Heat Transfer Cold PCM': E_PCM_C_0,
        'Heat Transfer Hot PCM': E_PCM_H_0,
        'Pump 1 Electric Power': E_Pump_1_0,
        'Pump 2 Electric Power': E_Pump_2_0,
        'HP Useful Thermal Power': Thermal_Power_Delivered_0
    }
            
    return E_dict

def find_average(M_States: pd.DataFrame, col_name: str):
    """
    Compute the time-weighted average of a column in M_States
    using the trapezoidal rule based on the 'Time' column.
    
    Parameters
    ----------
    M_States : pd.DataFrame
        DataFrame with a 'Time' column (in seconds) and the target column.
    col_name : str
        Column name whose time-weighted average will be calculated.
    
    Returns
    -------
    float
        The time-weighted average value.
    """
    # Ensure sorted by time
    M_States = M_States.sort_values("Time").reset_index(drop=True)
    
    # Time differences between rows [s]
    time_intervals = M_States["Time"].diff().iloc[1:].to_numpy()
    
    # Time-weighted average using trapezoidal integration
    avg_value = np.sum(
        ((M_States[col_name].iloc[1:].to_numpy() + M_States[col_name].iloc[:-1].to_numpy()) / 2) 
        * time_intervals
    ) / (M_States["Time"].iloc[-1] - M_States["Time"].iloc[0])
    
    return avg_value

def prepare_site_data(df_raw):
    # Convert temperature from Fahrenheit to Celsius
    df_raw['Ti'] = (df_raw['Ti'] - 32) * 5/9  # [°C]
    df_raw['Ta'] = (df_raw['Ta'] - 32) * 5/9  # [°C]
    
    # Convert wind speed from mph to m/s
    df_raw['WS'] = df_raw['WS'] * 0.44704  # [m/s]
    
    # Convert DateTime to datetime object FIRST
    df_raw['DateTime'] = pd.to_datetime(df_raw['Time'])

    # Calculate solar parameters
    result_df = Utility_functions.calculate_solar_parameters(
        df=df_raw,
        datetime_col='Time',
        latitude=Input_Parameters.latitude,  
        longitude=Input_Parameters.longitude,
        timezone='America/Los_Angeles',
        ghi_col='GHI',
        cloud_cover_col='Cloud'
    )

    result_df.rename(columns={'clearness_index_kt': 'k_t'}, inplace=True)
    result_df.rename(columns={'dni': 'DNI'}, inplace=True)
    result_df.rename(columns={'dhi': 'DHI'}, inplace=True)

    # Remove timezone info but keep the same time
    result_df['T_solar'] = result_df['solar_time'].dt.tz_localize(None)

    # Then calculate omega using the time component
    result_df['SolarTime'] = (
        result_df['T_solar'].dt.hour + 
        result_df['T_solar'].dt.minute / 60 + 
        result_df['T_solar'].dt.second / 3600
    )

    result_df = Utility_functions.add_day_status(result_df, Input_Parameters.latitude)
    
    return result_df

def prepare_rbc_states(weather, passive_heat_transfer_forecast, pv_forecast, e_load_forecast, fmu_states, system_states):
    # potentially add LMP forecast??
    # 
    # Create an empty DataFrame
    selected_df = pd.DataFrame()

    # Assign columns properly
    selected_df['DateTime'] = weather['DateTime']
    selected_df['Ta'] = weather['Ta']  # [°C]
    selected_df['PV'] = pv_forecast  # [kW]
    selected_df['DayStatus'] = weather['DayStatus']  # (1 = daytime, 0 = nighttime)

    selected_df['Cooling_Load'] = passive_heat_transfer_forecast['Total Net Heat Gain'].where(passive_heat_transfer_forecast['Total Net Heat Gain'] > 0, 0)  # [kW]
    selected_df['Heating_Load'] = (-passive_heat_transfer_forecast['Total Net Heat Gain']).where(passive_heat_transfer_forecast['Total Net Heat Gain'] < 0, 0)  # [kW]
    selected_df['E_Load'] = e_load_forecast['Total Load']  # [kW]

    # Total HP-TES system electrical power consumption over the interval
    # HP_TES_power =  fmu_states['HP Electric Power'] +  fmu_states['Pump 1 Electric Power'] +  fmu_states['Pump 2 Electric Power'] +  fmu_states['Fan Coil Fan Electric Power']  # [kW]
    
    Supervisor_CallHot = 0
    Supervisor_CallCold = 0

    if fmu_states['FCU Mode'] == 1: 
        Supervisor_CallHot = 1
    elif fmu_states['FCU Mode'] == -1:
        Supervisor_CallCold = 1
    
    states = {
        'OC_Mod':  fmu_states['Exact System Mode'],
        'xPCMTemperatureCold' : fmu_states['PCM_Cold_Temp'],  # [°C]
        'xPCMTemperatureHot' : fmu_states['PCM_Hot_Temp'],   # [°C]
        'Ambient Temperature' : weather['Ta'].iloc[0],  # [°C]
        'xSupplyTemperatureHP' : fmu_states['Heat Pump Supply Water Temperature'],  # [°C]
        'Heat Pump Compressor Speed' : fmu_states['Heat Pump Compressor Speed'],  # [0-1]
        'Supervisor_CallHot': Supervisor_CallHot,
        'Supervisor_CallCold': Supervisor_CallCold,
        'PV': pv_forecast[0],  # [kW]
        # 'E_Modelica_0': HP_TES_power,  # [kW] Does not need E_Modelica_0, B_SOC_0, and PV_Gen_0 anymore because we can get B_SOC directly from loadmatch
        'Battery SOC': system_states['Battery SOC'],  # [kWh]
        'E_Load': e_load_forecast['Total Load'].iloc[0]  # [kW]
    }

    return selected_df, states

def upsample_dataframe(df, target_interval_minutes, datetime_col='DateTime', method='mean'):
    """
    Resample a DataFrame from 5-minute intervals to a target interval (must be multiple of 5 minutes).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a DateTime column and data at 5-minute intervals
    target_interval_minutes : int
        Target time interval in minutes (must be a multiple of 5, e.g., 10, 15, 30, 60)
    datetime_col : str, default 'DateTime'
        Name of the DateTime column in the DataFrame
    method : str, default 'mean'
        Aggregation method for resampling. Options:
        - 'mean': Average values over the interval
        - 'first': Take the first value in each interval
        - 'last': Take the last value in each interval
        - 'sum': Sum values over the interval (for cumulative or rate-based data)
    
    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame with the target interval
        
    Raises:
    -------
    ValueError
        If target_interval_minutes is not a multiple of 5 or is less than 5
    """
    # Validate target interval
    if target_interval_minutes < 5:
        raise ValueError(f"Target interval must be at least 5 minutes. Got {target_interval_minutes} minutes.")
    
    if target_interval_minutes % 5 != 0:
        raise ValueError(f"Target interval must be a multiple of 5 minutes. Got {target_interval_minutes} minutes.")
    
    # Make a copy to avoid modifying the original
    df_resampled = df.copy()
    
    # Ensure DateTime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_resampled[datetime_col]):
        df_resampled[datetime_col] = pd.to_datetime(df_resampled[datetime_col])
    
    # Set DateTime as index for resampling
    df_resampled.set_index(datetime_col, inplace=True)
    
    # Sort by datetime to ensure proper resampling
    df_resampled.sort_index(inplace=True)
    
    # Create the frequency string for pandas resample
    freq_str = f'{target_interval_minutes}min'  # 'min' stands for minutes
    
    # Resample based on method
    if method == 'mean':
        df_resampled = df_resampled.resample(freq_str).mean()
    elif method == 'first':
        df_resampled = df_resampled.resample(freq_str).first()
    elif method == 'last':
        df_resampled = df_resampled.resample(freq_str).last()
    elif method == 'sum':
        df_resampled = df_resampled.resample(freq_str).sum()
    else:
        raise ValueError(f"Unknown method '{method}'. Must be one of: 'mean', 'first', 'last', 'sum'")
    
    # Remove leap day (February 29) if it exists
    # Filter out rows where month is 2 and day is 29
    leap_day_mask = (df_resampled.index.month == 2) & (df_resampled.index.day == 29)
    df_resampled = df_resampled[~leap_day_mask]
    
    # Reset index to make DateTime a column again
    df_resampled.reset_index(inplace=True)
    
    return df_resampled

def calculate_urgency_scores(passive_heat_transfer_forecast, e_load_forecast, socs):
    '''
    Inputs: 
        passive_heat_transfer_forecast: pandas DataFrame containing the forecasted passive heat transfer
        e_load_forecast: pandas DataFrame containing the forecasted electrical load

        A list of SOC values for the defined horizon, each SOC is a float (remaining energy in the storage).
        socs[0] is the SOC value for the electrical battery
        socs[1] is the SOC value for the heating PCM
        socs[2] is the SOC value for the cooling PCM

    Outputs:
        urgency_scores: dictionary containing the urgency scores for the defined horizon
        urgency_scores['Elctricity Urgency Score'] is the urgency score for the electrical load
        urgency_scores['Heating Urgency Score'] is the urgency score for the heating load
        urgency_scores['Cooling Urgency Score'] is the urgency score for the cooling load

    Need to normalize it

    The method to calculate the urgency scores is to calculate the ratio of the weighted (discounted to present value) forecasted load to the available energy in the storage. 
    '''

    PCM_C_Energy = max(0.01, socs[2])
    PCM_H_Energy = max(0.01, socs[1])
    Battery_SOC = max(0.01, socs[0])

    Cooling_Load = passive_heat_transfer_forecast['Total Net Heat Gain'].where(passive_heat_transfer_forecast['Total Net Heat Gain'] > 0, 0)  # [kW]
    Heating_Load = (-passive_heat_transfer_forecast['Total Net Heat Gain']).where(passive_heat_transfer_forecast['Total Net Heat Gain'] < 0, 0)  # [kW]
    E_Load = e_load_forecast['Total Load']  # [kW]

    E_Load = E_Load.reset_index(drop=True)
    Heating_Load = Heating_Load.reset_index(drop=True)
    Cooling_Load = Cooling_Load.reset_index(drop=True)

    NumTime = len(Cooling_Load)
    
    urgency_e = sum(E_Load[i] * np.exp(-i * Input_Parameters.discount_rate) for i in range(NumTime))/Battery_SOC
    urgency_h = sum(Heating_Load[i] * np.exp(-i * Input_Parameters.discount_rate) for i in range(NumTime))/PCM_H_Energy
    urgency_c = sum(Cooling_Load[i] * np.exp(-i * Input_Parameters.discount_rate) for i in range(NumTime))/PCM_C_Energy

    urgency_scores = {
        'Elctricity Urgency Score': urgency_e,
        'Heating Urgency Score': urgency_h,
        'Cooling Urgency Score': urgency_c
    }
    return urgency_scores


def weighted_aggregated_forecast(passive_heat_transfer_forecast, e_load_forecast, pv_generation_forecast):
    '''
    Calculate the urgency score for heating and cooling based on the forecasted loads.
    The urgency score is calculated based on the forecasted loads and the available energy in the PCM storage. 
    The higher the urgency score, the more urgent the heating/cooling call is.
    The urgency score is calculated as the ratio of the forecasted load to the available energy in the PCM storage.
    The urgency score is normalized to be between 0 and 1.

    Inputs:
    input_df: input DataFrame
    PCM_C_Energy: available energy in the PCM cold storage [kWh]
    PCM_H_Energy: available energy in the PCM hot storage [kWh]

    Outputs:
    urgency_H: urgency score for heating [0-1]
    urgency_C: urgency score for cooling [0-1]

    '''
    Cooling_Load = passive_heat_transfer_forecast['Total Net Heat Gain'].where(passive_heat_transfer_forecast['Total Net Heat Gain'] > 0, 0)  # [kW]
    Heating_Load = (-passive_heat_transfer_forecast['Total Net Heat Gain']).where(passive_heat_transfer_forecast['Total Net Heat Gain'] < 0, 0)  # [kW]
    E_Load = e_load_forecast['Total Load']  # [kW]

    E_Load = E_Load.reset_index(drop=True)
    Heating_Load = Heating_Load.reset_index(drop=True)
    Cooling_Load = Cooling_Load.reset_index(drop=True)

    NumTime = len(Cooling_Load)
    
    weighted_e_load_forecast = sum(E_Load[i] * np.exp(-i * Input_Parameters.discount_rate) for i in range(NumTime))
    weighted_h_load_forecast = sum(Heating_Load[i] * np.exp(-i * Input_Parameters.discount_rate) for i in range(NumTime))
    weighted_c_load_forecast = sum(Cooling_Load[i] * np.exp(-i * Input_Parameters.discount_rate) for i in range(NumTime))
    weighted_pv_forecast = sum(pv_generation_forecast[i] * np.exp(-i * Input_Parameters.discount_rate) for i in range(NumTime))

    return weighted_e_load_forecast, weighted_h_load_forecast, weighted_c_load_forecast, weighted_pv_forecast

def apply_forecast_error(percentage, weighted_aggregated_forecast, seed=None):
    """
    Apply forecast error by generating a random value within ±percentage of the forecast.
    
    Parameters:
    percentage: Error percentage as decimal (e.g., 0.1 for 10%)
    weighted_aggregated_forecast: The base forecast value
    seed: Optional random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    lower_bound = (1 - percentage) * weighted_aggregated_forecast
    upper_bound = (1 + percentage) * weighted_aggregated_forecast
    
    weighted_aggregated_forecast_with_error = max(random.uniform(lower_bound, upper_bound), 0)

    return weighted_aggregated_forecast_with_error