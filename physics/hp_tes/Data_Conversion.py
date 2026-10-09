import pandas as pd
import numpy as np
import os
import glob
import random
from . import Input_Parameters

def prepare_live_forecast(df_raw):
    import Utility_functions

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
    import Utility_functions
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
    import Utility_functions
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