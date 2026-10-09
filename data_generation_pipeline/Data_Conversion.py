import pandas as pd
import numpy as np
import os
import glob
import time
import requests
from datetime import datetime, timedelta

# Retry config for Open-Meteo API (connection resets / rate limits)
_OPENMETEO_MAX_RETRIES = 4
_OPENMETEO_BACKOFF_BASE = 2.0  # seconds

import Input_Parameters
import Utility_functions


def _wind_speed_2m(wind_10m_ms, z0=0.03):
    """Convert 10 m wind speed to 2 m using log profile, to match NSRDB (wind at 2 m)."""
    z2, z10 = 2.0, 10.0
    return wind_10m_ms * (np.log(z2 / z0) / np.log(z10 / z0))


def _fetch_openmeteo_archived_hourly_forecast(start_date, end_date, latitude, longitude, timezone="auto", models="ecmwf_ifs"):
    """Fetch hourly archived forecast from Open-Meteo IFS HRES. 6-day horizon, treated as midnight run.
    Retries on connection errors with exponential backoff."""
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={_OPENMETEO_HOURLY_VARS}&timezone={timezone}&wind_speed_unit=ms"
    )
    if models:
        url += f"&models={models}"
    last_err = None
    for attempt in range(_OPENMETEO_MAX_RETRIES):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError, OSError) as e:
            last_err = e
            if attempt < _OPENMETEO_MAX_RETRIES - 1:
                delay = _OPENMETEO_BACKOFF_BASE ** (attempt + 1)
                time.sleep(delay)
    raise last_err


# Open-Meteo Historical Weather API (archive): hourly reanalysis (ERA5/IFS)
_OPENMETEO_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,"
    "shortwave_radiation,diffuse_radiation,direct_normal_irradiance"
)


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


def _nsrdb_lst_to_local(df, datetime_col="DateTime", lst_offset=None, local_tz="America/Los_Angeles"):
    """Convert NSRDB DateTime from LST (no DST) to local clock for alignment with Open-Meteo forecast."""
    try:
        import pytz
        offset = lst_offset if lst_offset is not None else Input_Parameters.timezone_offset
        lst_tz = pytz.FixedOffset(int(offset * 60))
        tz = pytz.timezone(local_tz)
        dt = pd.to_datetime(df[datetime_col])
        local = dt.dt.tz_localize(lst_tz, ambiguous="infer").dt.tz_convert(tz).dt.tz_localize(None)
        df = df.copy()
        df[datetime_col] = local
        return df
    except Exception:
        return df


def fetch_historical_weather_nsrdb(
    start_date,
    end_date,
    time_resolution,
    nsrdb_data_dir="NSRDB_data",
    lst_to_local=True,
):
    """
    Load historical weather from NSRDB CSVs and return a DataFrame with columns
    Ta, WS, RH, GHI, DHI, DNI, Cloud, k_t, SolarTime, DayStatus, DateTime at time_resolution.

    NSRDB uses Local Standard Time (LST). If lst_to_local=True (default), DateTime is converted
    to local clock time (with DST) so it aligns with Open-Meteo archived forecast when merged.
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    years = list(range(start_dt.year, end_dt.year + 1))
    chunks = []
    for yr in years:
        df_yr = read_NSRDB_HMB(nsrdb_data_dir, yr)
        df_yr["DateTime"] = pd.to_datetime(df_yr[["Year", "Month", "Day", "Hour", "Minute"]])
        chunks.append(df_yr)
    raw = pd.concat(chunks, ignore_index=True)
    mask = (raw["DateTime"] >= start_dt) & (raw["DateTime"] <= end_dt + pd.Timedelta(days=1))
    raw = raw.loc[mask].copy()
    if raw.empty:
        raise ValueError(f"No NSRDB data for {start_date} to {end_date} in {nsrdb_data_dir}")
    prepared = prepare_NSRDB(raw)
    df_sampled = upsample_dataframe(
        prepared,
        target_interval_minutes=time_resolution,
        datetime_col="DateTime",
        method="mean",
    )
    if lst_to_local:
        df_sampled = _nsrdb_lst_to_local(df_sampled)
    return df_sampled


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

    # Open-Meteo returns local clock time (with DST). During DST we need -7 (PDT), not -8 (PST).
    if Utility_functions.is_dst("America/Los_Angeles", df_raw["DateTime"].iloc[0]):
        timezone_offset = timezone_offset + 1

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

    raw_json = _fetch_openmeteo_archived_hourly_forecast(start_date, end_date, latitude, longitude, timezone="auto")
    df_raw = _openmeteo_hourly_json_to_raw_df(raw_json)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"No hourly archived forecast (IFS HRES) returned for {start_date}–{end_date}")

    idx_fine = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq=pd.Timedelta(minutes=time_resolution))
    df_fine = df_raw.reindex(idx_fine).interpolate(method="linear").ffill().bfill()
    df_fine["Year"] = df_fine.index.year
    df_fine["Month"] = df_fine.index.month
    df_fine["Day"] = df_fine.index.day
    df_fine["Hour"] = df_fine.index.hour
    df_fine["Minute"] = df_fine.index.minute
    df_fine = df_fine.reset_index(drop=True)

    df_prepared = prepare_archived_forecast(
        df_fine.copy(),
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
