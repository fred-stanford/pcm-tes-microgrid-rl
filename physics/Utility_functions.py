import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime, timedelta
import time
from math import sin, cos, acos, asin, pi, exp, log, radians, degrees
import functools

# pvlib (and scipy) may fail on some Windows setups (e.g. Application Control blocking scipy DLL)
try:
    import pvlib
    from pvlib.location import Location
    from pvlib.clearsky import ineichen
    from pvlib import solarposition, irradiance, atmosphere
    _PVLIB_AVAILABLE = True
except (ImportError, OSError) as _e:
    pvlib = None
    Location = None
    ineichen = None
    solarposition = irradiance = atmosphere = None
    _PVLIB_AVAILABLE = False
    _PVLIB_IMPORT_ERROR = _e
else:
    _PVLIB_IMPORT_ERROR = None

try:
    from pysolar.solar import get_altitude
except (ImportError, OSError):
    get_altitude = None

import pytz
import Input_Parameters 
# import Input_Parameters 
# Utility Functions
def retry(tries=5, delay=30, backoff=1):
    """
    Retry calling the decorated function using an exponential backoff.
    
    :param exceptions: The exception(s) to check. May be a tuple of exceptions to check.
    :param tries: Number of times to try (not retry) before giving up.
    :param delay: Initial delay between retries in seconds.
    :param backoff: Backoff multiplier (e.g. value of 2 will double the delay each retry).
    """
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"{e}, Retrying in {_delay} seconds...")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry

def add_cloudcover(df):
    df['Cloud'] = df['Cloud Type'].apply(cloud_type_to_percent)
    return df

def cloud_type_to_percent(cloud_type):
    if cloud_type < 0 or cloud_type > 9:
        raise ValueError("Invalid cloud type. Must be an integer between 0 and 9.")

    cloud_cover = {
        0: 0,   # Clear sky
        1: 0.2,  # Cumulus - scattered clouds
        2: 0.4,  # Altocumulus - partial cloud cover
        3: 0.1,  # Cirrus - minimal obstruction
        4: 0.25,  # Cirrostratus - thin clouds, partial coverage
        5: 0.7,  # Stratus - thick cloud cover
        6: 0.9,  # Nimbostratus - very thick, overcast, precipitation likely
        7: 0.6,  # Stratocumulus - broken but significant cloud cover
        8: 1,    # Cumulonimbus - full cloud cover with storm potential
        9: 0.5   # Indeterminate - assumed average cloud cover
    }
    return cloud_cover[cloud_type]

def add_cloudcover_epw(df):
    df['Cloud'] = df.apply(lambda row: estimate_cloudcover(row['Ta'], row['Td'], row['RH']), axis=1)
    return df

'''
def calculate_k_t(df, latitude=34.0522, longitude=-118.2437, timezone="America/Los_Angeles"):
    
    # Set the index to the local timezone-aware datetime
    df = df.set_index('DateTime')
    
    # Define the location (Los Angeles)
    location = Location(latitude=latitude, longitude=longitude, tz=timezone)
    
    # Calculate solar zenith angle for each time step
    solar_zenith = []
    for date in df.index:
        solar_altitude = get_altitude(location.latitude, location.longitude, date)
        zenith = 90 - solar_altitude
        solar_zenith.append(zenith)
    df['solar_zenith'] = solar_zenith
    
    # Calculate clear-sky GHI using the index (DatetimeIndex)
    clearsky = location.get_clearsky(df.index, model='ineichen')
    df['clearsky_ghi'] = clearsky['ghi']
    
    # Calculate k_t using vectorized operations
    df['k_t'] = np.where(
        df['clearsky_ghi'] > 10,
        np.minimum(df['ghi'] / df['clearsky_ghi'], 1),
        0
    )
    
    return df.reset_index()
'''

def is_dst(timezone_name="America/Los_Angeles", dt=None):
    """
    Check if a datetime is in Daylight Saving Time (DST) for the given timezone.
    
    Args:
        timezone_name (str): Timezone string (e.g., "America/Los_Angeles").
        dt (datetime): Naive datetime object. If None, uses current time.
    
    Returns:
        bool: True if in DST, False otherwise.
    """
    tz = pytz.timezone(timezone_name)
    if dt is None:
        dt = datetime.datetime.now()
    localized_dt = tz.localize(dt, is_dst=None)
    return localized_dt.dst() != timedelta(0)  # Fixed: Use timedelta directly

def add_clearness_index(df, latitude, longitude, timezone_offset):
    if not _PVLIB_AVAILABLE:
        raise ImportError(
            "pvlib (or its dependency scipy) could not be loaded; clearness index needs it. "
            "On Windows, 'DLL load failed' is often caused by Application Control blocking scipy. "
            "Try: run Python as administrator, add an exclusion for the scipy folder, or use a different environment. "
            f"Original error: {_PVLIB_IMPORT_ERROR}"
        )
    # Ensure latitude and longitude are floats
    latitude = float(latitude)
    longitude = float(longitude)

    # Create a Location object with UTC timezone (will be adjusted by the offset)
    site = Location(latitude, longitude, tz="UTC")

    # Create a datetime index for the DataFrame
    df['datetime'] = df['DateTime']
    df.set_index('datetime', inplace=True)

    # Create the correct Etc/GMT timezone string
    if timezone_offset <= 0:
        tz_str = f"Etc/GMT+{int(-timezone_offset)}"  # For negative offsets (e.g., UTC-8 -> Etc/GMT+8)
    else:
        tz_str = f"Etc/GMT-{int(timezone_offset)}"  # For positive offsets (e.g., UTC+7 -> Etc/GMT-7)

    # Localize the datetime index to the local timezone
    df.index = df.index.tz_localize(tz_str)

    # Convert the datetime index to UTC for solar calculations
    df_utc = df.tz_convert("UTC")

    # Calculate the theoretical clearsky GHI
    clearsky_ghi = site.get_clearsky(df_utc.index, model='ineichen')

    # Merge the clearsky GHI with the original DataFrame
    df = df.join(clearsky_ghi)

    # Calculate k_t using vectorized operations
    df['k_t'] = np.where(
        df['ghi'] > 10,  # Only calculate k_t if ghi > 10 to avoid division by small numbers
        np.minimum(df['GHI'] / df['ghi'], 1),  # Cap k_t at 1
        0  # Set k_t to 0 if ghi <= 10
    )

    # Drop the 'ghi' column as it's no longer needed
    df.drop(columns=['ghi'], inplace=True)

    # Reset the index to return the DataFrame to its original form
    df.reset_index(drop=True, inplace=True)
    return df


def compute_clearsky_ghi(df, latitude, longitude, timezone_offset):
    """
    Compute clear-sky GHI for each row using datetime and location (pvlib Ineichen).
    Used when observed GHI is not available or not reliable (e.g. archived forecast).

    Parameters
    ----------
    df : pd.DataFrame
        Must have column 'DateTime' (timezone-naive local time).
    latitude : float
    longitude : float
    timezone_offset : float
        Local offset from UTC in hours (e.g. -8 for PST).

    Returns
    -------
    np.ndarray
        Clear-sky GHI [W/m^2] in same row order as df.
    """
    if not _PVLIB_AVAILABLE:
        raise ImportError(
            "pvlib (or its dependency scipy) could not be loaded; clear-sky GHI needs it. "
            "On Windows, 'DLL load failed' is often caused by Application Control blocking scipy. "
            "Try: run Python as administrator, add an exclusion for the scipy folder, or use a different environment. "
            f"Original error: {_PVLIB_IMPORT_ERROR}"
        )
    latitude = float(latitude)
    longitude = float(longitude)
    site = Location(latitude, longitude, tz="UTC")
    df_work = df[["DateTime"]].copy()
    df_work["datetime"] = df_work["DateTime"]
    df_work = df_work.set_index("datetime")
    if timezone_offset <= 0:
        tz_str = f"Etc/GMT+{int(-timezone_offset)}"
    else:
        tz_str = f"Etc/GMT-{int(timezone_offset)}"
    df_work.index = df_work.index.tz_localize(tz_str)
    df_utc = df_work.tz_convert("UTC")
    clearsky = site.get_clearsky(df_utc.index, model="ineichen")
    return clearsky["ghi"].values


def correct_daylightsaving(df):
    for i in range(len(df)):
        if Input_Parameters.dls_start <= df.iloc[i]['DateTime'] <= Input_Parameters.dls_end:
            df.at[i, 'DateTime'] = df.iloc[i]['DateTime'] - timedelta(hours=1)
    return df

def add_day_status(df, latitude):
    df['DayStatus'] = df.apply(lambda row: IsDay(row['SolarTime'], row['DateTime'], latitude), axis=1)
    return df

def calculate_dew_point(T, RH):
    es = 6.112 * exp((17.67 * T) / (T + 243.5))
    e = (RH / 100) * es
    Td = (243.5 * log(e / 6.112)) / (17.67 - log(e / 6.112))
    return Td

def Q_convection(WS, Ta, Ti):
    # NOTE (effective, calibrated term — intentional, not an unnoticed bug): `delta_T` below is an
    # EFFECTIVE stack-effect driver, not a literal |Ti - Ta|. T_indoor is in K while Ta is in C, so
    # delta_T ~= 273 + (Ti - Ta), which inflates the infiltration airflow (~2-3x). The envelope's
    # radiative (SHGC/RCC) coefficients and the lumped thermal capacitance were CALIBRATED end-to-end
    # to measured indoor temperature WITH this term in place (train/test RMSE ~0.73/0.72 C), so the
    # calibrated grey-box reproduces real Ti and the same plant is used identically for both
    # controllers (does not affect the RL-vs-RBC comparison). A units-consistent form,
    # `delta_T = abs(Ti - Ta)`, would require RE-RUNNING the envelope calibration; kept
    # as-calibrated for consistency.
    T_indoor = Ti + 273.15
    delta_T = abs(T_indoor - Ta)
    Infiltration = Input_Parameters.Al * np.sqrt(Input_Parameters.Cs * delta_T + Input_Parameters.Cw * WS**2)
    CFM = Infiltration * 0.001
    Q_inf_sen = 1005 * 1.25 * CFM * (Ta - Ti)
    return Q_inf_sen

def Q_conduction(Ta, Ti):
    Q_cond = Input_Parameters.UA * (Ta - Ti)
    return Q_cond

def add_solar_time(df, timezone="America/Los_Angeles"):
    # Ensure the DateTime column is timezone-aware
    if df['DateTime'].dt.tz is None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(timezone)
    
    solartimes = []
    for i in range(len(df)):
        local_time = df.iloc[i]['DateTime']
        
        # Check if the local time is in DST
        is_dst = local_time.dst() != pd.Timedelta(0)
        
        # Adjust for DST (subtract 1 hour if DST is active)
        if is_dst:
            local_time = local_time - pd.Timedelta(hours=1)
        
        # Calculate solar time
        LT = local_time.hour + local_time.minute / 60 + local_time.second / 3600
        N = local_time.timetuple().tm_yday
        Eqt = calculate_equation_of_time(N)
        T_solar = LT + Eqt / 60 + (4 * (Input_Parameters.standard_meridian_longitude - Input_Parameters.longitude)) / 60
        solartimes.append(T_solar)
    
    df['SolarTime'] = solartimes
    return df

def add_solar_time_NSRDB(df, timezone_offset, longitude):
    """
    Add a solar time column to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'DateTime' column.
        timezone_offset (float): Timezone offset from UTC in hours (e.g., -7 for MST in Arizona).
        longitude (float): Longitude of the location.

    Returns:
        pd.DataFrame: DataFrame with an additional 'SolarTime' column.
    """
    # Ensure the DateTime column is timezone-naive
    if df['DateTime'].dt.tz is not None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    
    # Calculate standard meridian longitude for the timezone
    standard_meridian_longitude = timezone_offset * 15
    
    # Calculate solar time for each row
    solartimes = []
    for i in range(len(df)):
        local_time = df.iloc[i]['DateTime']
        
        # Calculate solar time
        LT = local_time.hour + local_time.minute / 60 + local_time.second / 3600
        N = local_time.timetuple().tm_yday
        Eqt = calculate_equation_of_time(N)
        longitudinal_correction = (4 * (standard_meridian_longitude - float(longitude))) / 60  # Fixed sign
        T_solar = LT + longitudinal_correction + Eqt / 60
        
        # Ensure solar time is within 0-24 hours
        if T_solar < 0:
            T_solar += 24
        elif T_solar >= 24:
            T_solar -= 24
        
        solartimes.append(T_solar)
    
    # Add the SolarTime column to the DataFrame
    df['SolarTime'] = solartimes
    return df

def calculate_equation_of_time(N):
    if 1 <= N <= 106:
        return -14.2 * sin(pi * (N + 7) / 111)
    elif 107 <= N <= 166:
        return 4 * sin(pi * (N - 106) / 59)
    elif 167 <= N <= 246:
        return 6.5 * sin(pi * (N - 166) / 80)
    elif 247 <= N <= 366:
        return 16.4 * sin(pi * (N - 247) / 113)
    return 0

def calculate_irradiance_new(solar_time, pyranometer, surface_azimuth, tilt_angle, k_t, local_time):
    N = local_time.timetuple().tm_yday
    T_solar = solar_time
    delta = 23.45 * sin(radians((360 / 365) * (284 + N)))
    omega = 15 * (T_solar - 12)
    phi = radians(Input_Parameters.latitude)
    alpha = asin(sin(phi) * sin(radians(delta)) + cos(phi) * cos(radians(delta)) * cos(radians(omega)))
    if omega < 0:
        Psi = acos((sin(radians(delta)) - sin(alpha) * sin(phi)) / (cos(alpha) * cos(phi)))
    else:
        Psi = 2 * pi - acos((sin(radians(delta)) - sin(alpha) * sin(phi)) / (cos(alpha) * cos(phi)))
    beta = radians(tilt_angle)
    gamma = radians(surface_azimuth)
    theta = acos(sin(alpha) * cos(beta) + cos(alpha) * sin(beta) * cos(gamma - Psi))
    if k_t <= 0.22:
        k_d = 1 - 0.09 * k_t
    elif 0.22 < k_t <= 0.8:
        k_d = 0.9511 - 0.1604 * k_t + 4.388 * k_t**2 - 16.638 * k_t**3 + 12.336 * k_t**4
    else:
        k_d = 0.165
    DHI = pyranometer * k_d
    theta_z = radians(0)
    DNI = min((pyranometer - DHI) / cos(pi / 2 - alpha - theta_z), pyranometer)
    if alpha < 0:
        DNI = 0
        DHI = pyranometer
    if cos(theta) < 0:
        I_b = 0
    else:
        I_b = DNI * cos(theta)
    GHI = pyranometer
    I_d = max(DHI * (1 + cos(beta)) / 2 + 0.5 * GHI * (0.012 * (pi / 2 - alpha) - 0.04) * (1 - cos(beta)), 0)
    albedo = 0.18
    I_r = max(GHI * albedo * (1 - cos(beta)) / 2, 0)
    I_total = I_d + I_r + I_b
    return I_total

def calculate_solarheatgain(solar_time, GHI, k_t, local_time):
    Berg_tilt = 15
    GHI = max(GHI, 0)
    I_t = calculate_irradiance_new(solar_time, GHI, 0, 0, k_t, local_time)
    I_e = calculate_irradiance_new(solar_time, GHI, 90 - Berg_tilt, 90, k_t, local_time)
    I_s = calculate_irradiance_new(solar_time, GHI, 180 - Berg_tilt, 90, k_t, local_time)
    I_w = calculate_irradiance_new(solar_time, GHI, 270 - Berg_tilt, 90, k_t, local_time)
    I_n = calculate_irradiance_new(solar_time, GHI, 0 - Berg_tilt, 90, k_t, local_time)
    TotalSolarHeatGain = I_t * Input_Parameters.L_wall * Input_Parameters.L_wall + (I_e + I_s + I_w + I_n) * Input_Parameters.L_wall * Input_Parameters.H_wall
    return TotalSolarHeatGain

def estimate_cloudcover(Ta, Td, rh):
    temp_dew_diff = Ta - Td
    if temp_dew_diff < 2 and rh > 90:
        return 0.95
    elif temp_dew_diff < 4 and rh > 80:
        return 0.8
    elif temp_dew_diff < 6 and rh > 70:
        return 0.5
    else:
        return 0.15

def estimate_Ts(Ti, Ta, solartime, day_status, GHI):
    if day_status == 0:
        Ts = (Ta + Ti) / 2
        return [Ts, Ts, Ts, Ts, Ts]
    Ts_E = max(Ti, Ti * ((GHI + 1000) / 2000) * (-13 / 7) * cos(2 * pi * (solartime + 2) / 24))
    Ts_S = max(Ti, Ti * ((GHI + 1000) / 2000) * (-13 / 7) * cos(2 * pi * (solartime - 2) / 24))
    Ts_W = max(Ti, Ti * ((GHI + 1000) / 2000) * (-13 / 7) * cos(2 * pi * (solartime - 4.5) / 24))
    Ts_N = max(Ti, Ti * (-8 / 7) * ((GHI + 1000) / 2000) * cos(2 * pi * (solartime - 48) / 72))
    Ts_R = max(Ti, Ti * ((GHI + 3000) / 4000) * (-12 / 7) * cos(2 * pi * (solartime - 52) / 72))
    return [Ts_E, Ts_S, Ts_W, Ts_N, Ts_R]

def Q_radiativeCooling(Ti, Ta, RH, GHI, solartime, day_status):
    temp_dewpoint = calculate_dew_point(Ta, RH)
    e_sky = 0.741 + 0.0062 * temp_dewpoint
    sigma = 5.67e-8
    exposed_area = [Input_Parameters.L_wall * Input_Parameters.H_wall, Input_Parameters.L_wall * Input_Parameters.H_wall, Input_Parameters.L_wall * Input_Parameters.H_wall, 
                    Input_Parameters.L_wall * Input_Parameters.H_wall, Input_Parameters.L_wall * Input_Parameters.L_wall]
    total_exposed_area = sum(exposed_area)
    Ts_list = estimate_Ts(Ti, Ta, solartime, day_status, GHI)
    Ts = [T + 273.15 for T in Ts_list]
    T_sky = (Ta + 273.15)* e_sky**0.25
    Q_radcool = sigma * (Input_Parameters.e_Berg * sum((T**4 * area for T, area in zip(Ts, exposed_area))) - e_sky * total_exposed_area * (T_sky)**4)
    return Q_radcool

def IsDay(solar_time, local_time, latitude):
    N = local_time.timetuple().tm_yday
    delta = 23.45 * sin(radians((360 / 365) * (284 + N)))
    omega = 15 * (solar_time - 12)
    phi = radians(float(latitude))
    alpha = asin(sin(phi) * sin(radians(delta)) + cos(phi) * cos(radians(delta)) * cos(radians(omega)))
    return 1 if alpha > 0 else 0

def normalize_row(row, cols, bounds):
    normalized_row = row.copy()
    for col in cols:
        if col in bounds:
            lower_bound, upper_bound = bounds[col]
            normalized_row[col] = (row[col] - lower_bound) / (upper_bound - lower_bound)
    return normalized_row

def find_nearest_centroid(normalized_row, centroids, cols):
    row_vector = np.array([normalized_row[col] for col in cols])
    distances = [np.linalg.norm(row_vector - centroid) for centroid in centroids.T]
    return np.argmin(distances)



def calculate_ghi_python(lat, lon, start_date, end_date, stepsize):
    
    timezone = "America/Los_Angeles"
    # Create a date range in Python
    dates = pd.date_range(start=start_date, end=end_date, freq=stepsize)
    
    # Convert to DateTimeIndex with the specified timezone
    times = dates.tz_localize("UTC").tz_convert(timezone)
    
    # Create a location object
    location = pvlib.location.Location(lat, lon, tz=timezone)
    
    # Calculate the solar position and clear sky data
    solar_position = location.get_solarposition(times)
    clearsky = location.get_clearsky(times)
    
    # Filter out nighttime values based on solar elevation
    clearsky["ghi"] = clearsky["ghi"].where(solar_position["elevation"] > 0, 0)
    
    # Format the index in the desired "YYYY-MM-DD HH:MM:SS" format
    formatted_index = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in clearsky.index]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Time': formatted_index,
        'GHI': clearsky["ghi"].values
    })
    
    return df


def create_ghi_ratio(df):
    from pandas.tseries.offsets import Hour
    
    timezone_offset = - Input_Parameters.timezone_offset  # Default: -8 (PST)
    # Calculate start and end dates (adding 8 hours)
    start_date = df.iloc[0]['DateTime'] + Hour(timezone_offset)
    end_date = df.iloc[-1]['DateTime'] + Hour(timezone_offset)
    
    # Calculate stepsize
    stepsize = df.iloc[1]['DateTime'] - df.iloc[0]['DateTime']
    
    # Calculate GHI (assuming calculate_ghi_python is defined)
    ghi = calculate_ghi_python(Input_Parameters.latitude, Input_Parameters.longitude, start_date, end_date, stepsize)
    
    # Ensure GHI is non-negative
    df['GHI'] = df['GHI'].clip(lower=0)
    
    # Add GHI columns from calculated data
    df['GHI_time'] = ghi.iloc[:, 0]  # Assuming first column is time
    df['GHI_perfect'] = ghi.iloc[:, 1]  # Assuming second column is GHI values
    
    # Initialize k_t column with zeros
    df['k_t'] = 0.0
    
    # Calculate clearness index k_t
    for i in range(len(df)):
        if df.iloc[i]['GHI_perfect'] > 10:
            ratio = df.iloc[i]['GHI'] / df.iloc[i]['GHI_perfect']
            df.loc[df.index[i], 'k_t'] = min(ratio, 1)
    
    return df



'''

NEW

'''

def calculate_solar_parameters(df, datetime_col, latitude, longitude, timezone,
                               ghi_col='GHI', cloud_cover_col='cloud_cover'):
    """
    Calculate comprehensive solar parameters using pvlib (vectorized, tz-aware).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    datetime_col : str
        Name of the datetime column
    latitude : float
        Latitude in decimal degrees
    longitude : float
        Longitude in decimal degrees
    timezone : str
        Timezone string (e.g., 'UTC', 'US/Eastern', 'US/Pacific')
    ghi_col : str
        Name of the GHI column in df
    cloud_cover_col : str
        Name of cloud cover column (0-1). Not used directly here but kept for API consistency.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added solar parameters (tz-aware datetimes preserved)
    """

    # Work on a copy
    result_df = df.copy()

    # Ensure datetime column exists and is parsed
    result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    # Make timezone-aware: if naive, localize to provided timezone; if tz-aware, convert to that timezone
    if result_df[datetime_col].dt.tz is None:
        result_df[datetime_col] = result_df[datetime_col].dt.tz_localize(timezone)
    else:
        result_df[datetime_col] = result_df[datetime_col].dt.tz_convert(timezone)

    times = result_df[datetime_col]

    # Create pvlib Location
    location = Location(latitude=latitude, longitude=longitude, tz=timezone)

    # Solar position (pvlib accepts a tz-aware series)
    solar_pos = solarposition.get_solarposition(times, latitude, longitude, altitude=0, pressure=None)

    # Prefer apparent_zenith if available (better for irradiance), otherwise use zenith
    zenith_col = 'apparent_zenith' if 'apparent_zenith' in solar_pos.columns else 'zenith'
    elevation_col = 'apparent_elevation' if 'apparent_elevation' in solar_pos.columns else 'elevation'

    # Compute day of year as numeric array
    doy = times.dt.dayofyear.values
    
    # Compute extraterrestrial radiation manually
    dni_extra = irradiance.get_extra_radiation(doy)

    # times = pd.DatetimeIndex(pd.to_datetime(df[datetime_col]))
    # clearsky = location.get_clearsky(times, model='ineichen', dni_extra=dni_extra)
    clearsky = location.get_clearsky(pd.DatetimeIndex(times), model='ineichen', dni_extra=dni_extra)
    # Vectorized solar time calculation (handles DST correctly)
    result_df['solar_time'] = calculate_solar_time_series(times, longitude, timezone)

    # Add solar position columns to result_df
    result_df['solar_zenith'] = solar_pos[zenith_col].values
    # solar altitude/elevation (pvlib uses "apparent_elevation" or "elevation")
    if elevation_col in solar_pos.columns:
        result_df['solar_altitude'] = solar_pos[elevation_col].values
    else:
        # fallback: convert zenith to altitude
        result_df['solar_altitude'] = 90.0 - result_df['solar_zenith']

    result_df['solar_azimuth'] = solar_pos['azimuth'].values

    # Add clearsky irradiance components
    result_df['clearsky_ghi'] = clearsky['ghi'].values
    result_df['clearsky_dni'] = clearsky['dni'].values
    result_df['clearsky_dhi'] = clearsky['dhi'].values

    # Safe clearness index (avoid division by zero / NaN)
    # If clearsky_ghi is zero or NaN, set clearness_index_kt to NaN (then clip later)
    clearsky_ghi = result_df['clearsky_ghi'].replace(0, np.nan)
    result_df['clearness_index_kt'] = result_df[ghi_col] / clearsky_ghi
    # Clip to a reasonable range (non-negative, allow small >1 values for momentary conditions)
    result_df['clearness_index_kt'] = result_df['clearness_index_kt'].clip(lower=0, upper=1.0)

    # Decompose GHI -> DNI, DHI using DISC (vectorized)
    # pvlib.irradiance.disc expects: ghi, zenith, times, pressure (pressure optional)
    # Calculate DNI using DISC model
    disc_out = irradiance.disc(
        ghi=result_df[ghi_col].values,
        solar_zenith=result_df['solar_zenith'].values,
        datetime_or_doy=times.dt.dayofyear.values
    )

    result_df['dni'] = disc_out['dni']

    # Calculate DHI from GHI and DNI
    result_df['dhi'] = result_df[ghi_col] - result_df['dni'] * np.cos(np.deg2rad(result_df['solar_zenith']))


    erbs_out = irradiance.erbs(result_df[ghi_col].values, result_df['solar_zenith'].values, times.dt.dayofyear.values)
    result_df['dni_erbs'] = erbs_out['dni']
    result_df['dhi_erbs'] = erbs_out['dhi']

    # Night-time mask: use solar altitude <= 0 (more robust) to zero out irradiance
    night_mask = result_df['solar_altitude'] <= 0
    result_df.loc[night_mask, ['dni', 'dhi', 'dni_erbs', 'dhi_erbs']] = 0.0
    result_df.loc[night_mask, 'clearness_index_kt'] = 0.0

    # Air mass (relative)
    result_df['air_mass'] = atmosphere.get_relative_airmass(result_df['solar_zenith'].values)

    return result_df

def calculate_solar_time_series(datetime_series, longitude, timezone):
    """
    Compute solar time for a Pandas datetime Series or DatetimeIndex,
    accounting for timezone, daylight savings, and longitude correction.
    """

    import pandas as pd
    import numpy as np
    import pytz

    # --- Normalize input ---
    if isinstance(datetime_series, pd.DatetimeIndex):
        datetimes = pd.Series(datetime_series)
    else:
        datetimes = datetime_series

    # Step 1: Ensure timezone awareness
    if datetimes.dt.tz is None:
        local_tz = pytz.timezone(timezone)
        local_times = datetimes.dt.tz_localize(local_tz)
    else:
        local_times = datetimes.dt.tz_convert(timezone)

    # Step 2: Convert to UTC
    utc_times = local_times.dt.tz_convert("UTC")

    # ✅ FIX HERE: Compute UTC offset in hours (handles DST correctly)
    # Instead of .dt.utcoffset(), use .apply(lambda x: x.utcoffset().total_seconds())
    utc_offset_hours = local_times.apply(lambda x: x.utcoffset().total_seconds() / 3600.0)

    # Step 4: Local Standard Time Meridian (LSTM)
    lstm = 15.0 * utc_offset_hours  # degrees

    # Step 5: Equation of Time (EoT)
    day_of_year = local_times.dt.dayofyear
    b = (360.0 / 365.0) * (day_of_year - 81)
    eot = 9.87 * np.sin(np.deg2rad(2 * b)) - 7.53 * np.cos(np.deg2rad(b)) - 1.5 * np.sin(np.deg2rad(b))

    # Step 6: Time correction (in minutes)
    time_correction = 4.0 * (longitude - lstm) + eot

    # Step 7: Apply correction to get solar time
    solar_time = local_times + pd.to_timedelta(time_correction, unit="m")

    return solar_time