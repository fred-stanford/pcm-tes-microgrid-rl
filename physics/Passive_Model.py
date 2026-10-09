import pandas as pd
import numpy as np
from physics import Utility_functions
import Input_Parameters

# Define the columns and their bounds for normalization
cols_for_clustering_SHGC = ['WS', 'RH', 'Ta', 'Cloud', 'GHI', 'k_t']
cols_for_clustering_RCC = ['WS', 'RH', 'Ta', 'Cloud']
bounds = {
    'WS': (0.0, 10.0),  # Example bounds for Wind Speed [m/s]
    'RH': (0.0, 100.0),  # Example bounds for Relative Humidity [%]
    'Ta': (0.0, 40.0),  # Example bounds for Temperature in Celsius [°C]
    'GHI': (0.0, 1000.0),  # Example bounds for Global Horizontal Irradiance [W/m^2]
    'Cloud': (0.0, 1.0),  # Example bounds for Cloud cover fraction
    'k_t': (0.0, 1.0)  # Example bounds for clearness index
}

def passive_model(model_coefficients, df, Ti, Q_active, FCU_mode):
    '''
    This function returns the new indoor temperature based and current FCU mode based on current weather (passive heat transfer), indoor temperature,
    active heating/cooling, and previous FCU mode using a passive building thermal model.

    Inputs:
    model_coefficients: dictionary of model coefficients for the passive building thermal model
    df: pandas DataFrame containing current weather data
    Ti: current indoor temperature [°C]
    Q_active: current active heating/cooling power from HP-TES system and internal gain [kW]
    FCU_mode: previous FCU mode (1 for heating, -1 for cooling, 0 for off)

    Outputs:
    Ti_new: new indoor temperature at the next timestep [°C]
    FCU_mode_new: updated FCU mode at the next timestep (1 for heating, -1 for cooling, 0 for off)
    '''

    SHGCs = model_coefficients["Tuned Parameters"]["_SHGC_"]
    RCCs_day = model_coefficients["Tuned Parameters"]["_RCC_day_"]
    RCCs_night = model_coefficients["Tuned Parameters"]["_RCC_night_"]
    TC = Input_Parameters.TC # [kJ/K]
    
    day_centroids_SHGC_df = model_coefficients["SHGC Day Centroids"]
    day_centroids_RCC_df = model_coefficients["RCC Day Centroids"]
    night_centroids_RCC_df = model_coefficients["RCC Night Centroids"]
    
    day_centroids_SHGC = day_centroids_SHGC_df.to_numpy().T
    day_centroids_RCC = day_centroids_RCC_df.to_numpy().T
    night_centroids_RCC = night_centroids_RCC_df.to_numpy().T

    # Create space to store heat transfers
    NetHeatTransfers = []  # [kW]
    
    # Create space to store various sources of heat transfers.
    Q_cond = []  # [kW]
    Q_conv = []  # [kW]
    Q_radcool = []  # [kW]
    Q_radheat = []  # [kW]

    TemperatureAmbient = df['Ta'].iloc[0]  # [°C]
    WindSpeed = df['WS'].iloc[0]  # [m/s]
    RelativeHumidity = df['RH'].iloc[0]  # [%]
    GHI = df['GHI'].iloc[0]  # [W/m^2]

    local_time = df['DateTime'].iloc[0]
    k_t = df['k_t'].iloc[0]
    solar_time = df['SolarTime'].iloc[0]
    CloudCover = df['Cloud'].iloc[0]
    day_status = df['DayStatus'].iloc[0]

    # Conduction
    q_cd = Utility_functions.Q_conduction(TemperatureAmbient, Ti) / 1000  # [kW]
    Q_cond.append(q_cd)
    
    # Convection
    q_cv = Utility_functions.Q_convection(WindSpeed, TemperatureAmbient, Ti) / 1000  # [kW]
    Q_conv.append(q_cv)
    
    # Radiative cooling
    q_rc = Utility_functions.Q_radiativeCooling(Ti, TemperatureAmbient, RelativeHumidity, GHI, solar_time, day_status) / 1000  # [kW]

    # Radiative heat gain to the structure
    q_rh = Utility_functions.calculate_solarheatgain(solar_time, GHI, k_t, local_time) / 1000  # [kW]
    
    new_row = pd.DataFrame({
        'WS': [WindSpeed],
        'RH': [RelativeHumidity],
        'Ta': [TemperatureAmbient],
        'GHI': [GHI],
        'Cloud': [CloudCover],
        'k_t': [k_t]
    })
    
    if day_status == 0.0:
        normalized_new_row_rcc = Utility_functions.normalize_row(new_row, cols_for_clustering_RCC, bounds)
        # Find the nearest centroid for night time data
        nearest_rcc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_rcc, night_centroids_RCC, cols_for_clustering_RCC)
        Q_radcool.append(q_rc * RCCs_night[nearest_rcc_cluster])
        Q_radheat.append(0)
        Q_passive = q_cd + q_cv - q_rc * RCCs_night[nearest_rcc_cluster]  # [kW]
        NetHeatTransfers.append(Q_passive)
    else:
        normalized_new_row_shgc = Utility_functions.normalize_row(new_row, cols_for_clustering_SHGC, bounds)
        normalized_new_row_rcc = Utility_functions.normalize_row(new_row, cols_for_clustering_RCC, bounds)
        # Find the nearest centroid for day time data
        nearest_shgc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_shgc, day_centroids_SHGC, cols_for_clustering_SHGC)
        nearest_rcc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_rcc, day_centroids_RCC, cols_for_clustering_RCC)
        Q_radcool.append(q_rc * RCCs_day[nearest_rcc_cluster])
        Q_radheat.append(q_rh * SHGCs[nearest_shgc_cluster])

        Q_passive = q_cd + q_cv + q_rh * SHGCs[nearest_shgc_cluster] - q_rc * RCCs_day[nearest_rcc_cluster]  # [kW]
        NetHeatTransfers.append(Q_passive)
      
    Ti_new = Ti + Input_Parameters.δt * (Q_passive + Q_active)/TC 
    # C = s*kJ/s/(kJ/C)
    
    # Hysteresis Logic
    FCU_mode_new = 0
    # was in cooling mode before
    if FCU_mode == -1:
        if Ti_new >= Input_Parameters.T_high - Input_Parameters.T_delta:
            FCU_mode_new = -1
        else:
            FCU_mode_new = 0
    
    # was in heating mode before
    if FCU_mode == 1:
        if Ti_new <= Input_Parameters.T_low + Input_Parameters.T_delta:
            FCU_mode_new = 1
        else:
            FCU_mode_new = 0
    
    if FCU_mode == 0:
        if Ti_new >= Input_Parameters.T_high + Input_Parameters.T_delta:
            FCU_mode_new = -1
        elif Ti_new <= Input_Parameters.T_low - Input_Parameters.T_delta:
            FCU_mode_new = 1
        else:
            FCU_mode_new = 0

    return Ti_new, FCU_mode_new


def passive_load_forecast(model_coefficients, weather, Ti, E_load_forecast):

    '''
    This function forecasts the future passive heat transfer based on weather forecast data.
    
    Inputs:
    model_coefficients: dictionary of model coefficients for the passive building thermal model
    weather: pandas DataFrame containing weather forecast data
    Ti: current indoor temperature [°C]
    E_Load: pandas Series containing forecasted electrical load including internal heat gains [kW]

    Outputs:
    df: pandas DataFrame containing forecasted passive heat transfer and its components, which includes:
        'Conduction' [kW]
        'Convection' [kW]
        'Radiative Heat Gain' [kW]
        'Radiative Cooling' [kW]
        'Passive Heat Transfer' [kW]
        'Active Heat Gain' [kW]
        'Total Net Heat Gain' [kW]
    '''
    
    SHGCs = model_coefficients["Tuned Parameters"]["_SHGC_"]
    RCCs_day = model_coefficients["Tuned Parameters"]["_RCC_day_"]
    RCCs_night = model_coefficients["Tuned Parameters"]["_RCC_night_"]
    TC = Input_Parameters.TC # [kJ/K]
    
    day_centroids_SHGC_df = model_coefficients["SHGC Day Centroids"]
    day_centroids_RCC_df = model_coefficients["RCC Day Centroids"]
    night_centroids_RCC_df = model_coefficients["RCC Night Centroids"]
    
    day_centroids_SHGC = day_centroids_SHGC_df.to_numpy().T
    day_centroids_RCC = day_centroids_RCC_df.to_numpy().T
    night_centroids_RCC = night_centroids_RCC_df.to_numpy().T

    NumTime = weather.shape[0]

    # Create space to store heat transfers
    Passive_Heat_Transfers = []  # [kW]
    
    # Create space to store various sources of heat transfers.
    Q_cond = []  # [kW]
    Q_conv = []  # [kW]
    Q_radcool = []  # [kW]
    Q_radheat = []  # [kW]

    TemperatureAmbient = weather['Ta']  # [°C]
    WindSpeed = weather['WS']  # [m/s]
    RelativeHumidity = weather['RH']  # [%]
    GHI = weather['GHI']  # [W/m^2]

    local_time = weather['DateTime']
    k_t = weather['k_t']
    solar_time = weather['SolarTime']
    CloudCover = weather['Cloud']
    day_status = weather['DayStatus']

    Internal_Heat_Gain = E_load_forecast["Total Heat Gain"].reset_index(drop=True)  # [kW]

    # Timeseries simulation
    for t in range(NumTime):
        # Conduction
        q_cd = Utility_functions.Q_conduction(TemperatureAmbient.iloc[t], Ti) / 1000  # [kW]
        Q_cond.append(q_cd)
        
        # Convection
        q_cv = Utility_functions.Q_convection(WindSpeed.iloc[t], TemperatureAmbient.iloc[t], Ti) / 1000  # [kW]
        Q_conv.append(q_cv)
        
        # Radiative cooling
        q_rc = Utility_functions.Q_radiativeCooling(Ti, TemperatureAmbient.iloc[t], RelativeHumidity.iloc[t], GHI.iloc[t], solar_time.iloc[t], day_status.iloc[t]) / 1000  # [kW]

        # Radiative heat gain to the structure
        q_rh = Utility_functions.calculate_solarheatgain(solar_time.iloc[t], GHI.iloc[t], k_t.iloc[t], local_time.iloc[t]) / 1000  # [kW]
        
        new_row = pd.DataFrame({
            'WS': [WindSpeed.iloc[t]],
            'RH': [RelativeHumidity.iloc[t]],
            'Ta': [TemperatureAmbient.iloc[t]],
            'GHI': [GHI.iloc[t]],
            'Cloud': [CloudCover.iloc[t]],
            'k_t': [k_t.iloc[t]]
        })
        
        if day_status.iloc[t] == 0.0:
            normalized_new_row_rcc = Utility_functions.normalize_row(new_row, cols_for_clustering_RCC, bounds)
            # Find the nearest centroid for night time data
            nearest_rcc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_rcc, night_centroids_RCC, cols_for_clustering_RCC)
            Q_radcool.append(q_rc * RCCs_night[nearest_rcc_cluster])
            Q_radheat.append(0)
            Q_passive = q_cd + q_cv - q_rc * RCCs_night[nearest_rcc_cluster]  # [kW]
            Passive_Heat_Transfers.append(Q_passive)
        else:
            normalized_new_row_shgc = Utility_functions.normalize_row(new_row, cols_for_clustering_SHGC, bounds)
            normalized_new_row_rcc = Utility_functions.normalize_row(new_row, cols_for_clustering_RCC, bounds)
            # Find the nearest centroid for day time data
            nearest_shgc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_shgc, day_centroids_SHGC, cols_for_clustering_SHGC)
            nearest_rcc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_rcc, day_centroids_RCC, cols_for_clustering_RCC)
            Q_radcool.append(q_rc * RCCs_day[nearest_rcc_cluster])
            Q_radheat.append(q_rh * SHGCs[nearest_shgc_cluster])

            Q_passive = q_cd + q_cv + q_rh * SHGCs[nearest_shgc_cluster] - q_rc * RCCs_day[nearest_rcc_cluster]  # [kW]
            Passive_Heat_Transfers.append(Q_passive)

    columns = ['Conduction', 'Convection', 'Radiative Heat Gain', 'Radiative Cooling', 'Passive Heat Transfer', 'Internal Heat Gain', 'Total Net Heat Gain']
    
    output_df = pd.DataFrame(columns=columns)

    output_df['Conduction'] = Q_cond  # [kW]
    output_df['Convection'] = Q_conv  # [kW]
    output_df['Radiative Heat Gain'] = Q_radheat  # [kW]
    output_df['Radiative Cooling'] = Q_radcool  # [kW]
    output_df['Passive Heat Transfer'] = Passive_Heat_Transfers  # [kW]
    output_df['Internal Heat Gain'] = Internal_Heat_Gain  # [kW]
    output_df['Total Net Heat Gain'] = Passive_Heat_Transfers + Internal_Heat_Gain  # [kW]

    return output_df

def passive_model_reduced(Q_passive, Ti, Q_active, FCU_mode):
    '''
    This function returns the new indoor temperature based and current FCU mode based on already calculated passive heat transfer, 
    current indoor temperature, active heating/cooling, and previous FCU mode using a passive building thermal model.

    Inputs:
    Q_passive: current passive heat transfer [kW]
    Ti: current indoor temperature [°C]
    Q_active: current active heating/cooling power from HP-TES system and internal gain [kW]
    FCU_mode: previous FCU mode (1 for heating, -1 for cooling, 0 for off)

    Outputs:
    Ti_new: new indoor temperature at the next timestep [°C]
    FCU_mode_new: updated FCU mode at the next timestep (1 for heating, -1 for cooling, 0 for off)
    '''
    TC = Input_Parameters.TC # [kJ/K]

    Ti_new = Ti + Input_Parameters.δt * (Q_passive + Q_active)/TC 
    # C = s*kJ/s/(kJ/C)
    
    # Hysteresis Logic
    FCU_mode_new = 0
    # was in cooling mode before
    if FCU_mode == -1:
        if Ti_new >= Input_Parameters.T_high - Input_Parameters.T_delta:
            FCU_mode_new = -1
        else:
            FCU_mode_new = 0
    
    # was in heating mode before
    if FCU_mode == 1:
        if Ti_new <= Input_Parameters.T_low + Input_Parameters.T_delta:
            FCU_mode_new = 1
        else:
            FCU_mode_new = 0
    
    if FCU_mode == 0:
        if Ti_new >= Input_Parameters.T_high + Input_Parameters.T_delta:
            FCU_mode_new = -1
        elif Ti_new <= Input_Parameters.T_low - Input_Parameters.T_delta:
            FCU_mode_new = 1
        else:
            FCU_mode_new = 0

    return Ti_new, FCU_mode_new