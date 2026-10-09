import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import sin, cos, acos, asin, pi, exp, log, radians

# pvlib (and scipy) may fail on some Windows setups (e.g. Application Control blocking scipy DLL)
try:
    from pvlib.location import Location
    _PVLIB_AVAILABLE = True
except (ImportError, OSError) as _e:
    Location = None
    _PVLIB_AVAILABLE = False
    _PVLIB_IMPORT_ERROR = _e
else:
    _PVLIB_IMPORT_ERROR = None


import pytz
import Input_Parameters 
# import Input_Parameters 
# Utility Functions

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


def add_day_status(df, latitude):
    df['DayStatus'] = df.apply(lambda row: IsDay(row['SolarTime'], row['DateTime'], latitude), axis=1)
    return df

def calculate_dew_point(T, RH):
    es = 6.112 * exp((17.67 * T) / (T + 243.5))
    e = (RH / 100) * es
    Td = (243.5 * log(e / 6.112)) / (17.67 - log(e / 6.112))
    return Td

def Q_convection(WS, Ta, Ti):
    T_indoor = Ti + 273.15
    delta_T = abs(T_indoor - Ta)
    Infiltration = Input_Parameters.Al * np.sqrt(Input_Parameters.Cs * delta_T + Input_Parameters.Cw * WS**2)
    CFM = Infiltration * 0.001
    Q_inf_sen = 1005 * 1.25 * CFM * (Ta - Ti)
    return Q_inf_sen

def Q_conduction(Ta, Ti):
    Q_cond = Input_Parameters.UA * (Ta - Ti)
    return Q_cond


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


'''

NEW

'''
