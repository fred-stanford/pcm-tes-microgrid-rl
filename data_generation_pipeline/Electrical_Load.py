import numpy as np
import pandas as pd
import random

import Input_Parameters

def generate_schedules(schedule, weather, random_seed):
    datetime_col = weather['DateTime']
    time_resolution = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours
    total_intervals = len(datetime_col)
    intervals_per_hour = int(1 / time_resolution)
    intervals_per_day = 24 * intervals_per_hour
    
    # Calculate the number of days in the weather DataFrame
    num_days = total_intervals // intervals_per_day  # Full days only

    if schedule == "simple":
        return simple_schedule(weather, time_resolution, total_intervals, num_days)
    elif schedule == "complex":
        return complex_schedule(weather, time_resolution, total_intervals, num_days)
    elif schedule == "Bayesian":
        return bayes_schedule_FOB(weather, random_seed)
    elif schedule == "complex_bayesian":
        return complex_bayes_schedule_FOB(weather, random_seed)
    else:
        raise ValueError("Invalid schedule type")


def generate_load_forecast(
    schedule_type,
    start_of_day,
    horizon_days,
    time_resolution,
    random_seed,
):
    """
    Generate an electrical load forecast for a fixed window (e.g. 6 days), using the same
    mechanism and output format as generate_schedules.

    Use this when you have a forecast window (e.g. from archived_forecast_for_passive_load)
    and need a matching E_load forecast without building the full-year profile.

    Parameters
    ----------
    schedule_type : str
        Same as generate_schedules: "Bayesian", "simple", "complex", or "complex_bayesian".
    start_of_day : datetime or str
        Start of the forecast window (e.g. 2022-06-01 00:00:00).
    horizon_days : int or float
        Length of the window in days (e.g. 6).
    time_resolution : float
        Time step in minutes (e.g. 5).
    random_seed : int
        Seed for reproducibility (same role as in generate_schedules).

    Returns
    -------
    pd.DataFrame
        Same format as generate_schedules for the given schedule_type (e.g. for Bayesian:
        columns include DateTime, Total Occupancy, Total Load, Total Heat Gain, etc.).
    """
    start_dt = pd.Timestamp(start_of_day)
    num_steps = int(horizon_days * 24 * 60 / time_resolution)
    dt_index = pd.date_range(
        start=start_dt,
        periods=num_steps,
        freq=pd.Timedelta(minutes=float(time_resolution)),
    )
    weather_forecast = pd.DataFrame({"DateTime": dt_index})
    return generate_schedules(schedule_type, weather_forecast, random_seed)

def simple_schedule(weather, time_resolution, total_intervals, num_days):
    intervals_per_hour = int(1 / time_resolution)
    SimpleSchedule = np.zeros((total_intervals, 3))
    
    # Loop over the number of days in the weather DataFrame (not hardcoded 365)
    for i in range(num_days):
        for j in range(10 * intervals_per_hour, 17 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:  # Prevent index overflow
                SimpleSchedule[idx, :] = 1
    
    schedule = pd.DataFrame()
    schedule['Lighting'] = SimpleSchedule[:, 0] * Input_Parameters.PeakLighting
    schedule['Plugs'] = SimpleSchedule[:, 1] * Input_Parameters.PeakPlugLoad
    schedule['Total Occupancy'] = SimpleSchedule[:, 2] * (Input_Parameters.MaxOccupancy)
    schedule['Total Load'] = schedule['Lighting']+ schedule['Plugs']
    schedule['DateTime'] = weather['DateTime']
    return schedule


def complex_schedule(weather, time_resolution, total_intervals, num_days):
    intervals_per_hour = int(1 / time_resolution)
    ComplexSchedule = np.zeros((total_intervals, 3))
    
    # Loop over the number of days in the weather DataFrame (not hardcoded 365)
    for i in range(num_days):
        # Time block 1: 1 AM - 6 AM
        for j in range(1 * intervals_per_hour, 6 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0:2] = 0  # Lighting and Plugs off
                ComplexSchedule[idx, 2] = 1    # Occupancy on
        
        # Time block 2: 7 AM - 8 AM
        for j in range(7 * intervals_per_hour, 8 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0] = 1  # Lighting on
                ComplexSchedule[idx, 1] = 0  # Plugs off
                ComplexSchedule[idx, 2] = 1  # Occupancy on
        
        # Time block 3: 9 AM - 7 PM
        for j in range(9 * intervals_per_hour, 19 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0] = 1  # Lighting on
                ComplexSchedule[idx, 1] = 0  # Plugs off
                ComplexSchedule[idx, 2] = 0.5  # Occupancy at 50%
        
        # Time block 4: 8 PM - 9 PM
        for j in range(20 * intervals_per_hour, 21 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0:2] = 1  # Lighting and Plugs on
                ComplexSchedule[idx, 2] = 0.5  # Occupancy at 50%
        
        # Time block 5: 9 PM - 10 PM
        for j in range(21 * intervals_per_hour, 22 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, :] = 1  # Everything on
        
        # Time block 6: 11 PM - 12 AM
        for j in range(23 * intervals_per_hour, 24 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0] = 0  # Lighting off
                ComplexSchedule[idx, 1] = 1  # Plugs on
                ComplexSchedule[idx, 2] = 1  # Occupancy on
    
    # Create the schedule DataFrame
    schedule = pd.DataFrame()
    schedule['Lighting'] = ComplexSchedule[:, 0] * Input_Parameters.PeakLighting
    schedule['Plugs'] = ComplexSchedule[:, 1] * Input_Parameters.PeakPlugLoad
    schedule['Total Occupancy'] = ComplexSchedule[:, 2] * (Input_Parameters.MaxOccupancy)
    schedule['Total Load'] = schedule['Lighting']+ schedule['Plugs']
    schedule['DateTime'] = weather['DateTime']
    
    return schedule

# Helper function for weighted random sampling
def weighted_sample(choices, weights):
    return random.choices(choices, weights=weights, k=1)[0]

def draw_sched_FOB(weather):
    datetime_col = weather['DateTime']
    time_resolution = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours
    intervals_per_hour = int(1 / time_resolution)
    intervals_per_day = 24 * intervals_per_hour
    
    weekday = datetime_col.dt.weekday + 1  # Python weekday is 0-6 (Monday=0), so +1 to match Julia's 1-7
    occ_vec = np.zeros(len(datetime_col), dtype=int)

    # Define Probability Distributions
    weekday_probs = [0.42, 0.35, 0.35, 0.35, 0.42, 0, 0]  # Probability of showing up for each day of the week
    arrive_probs = [0.3, 0.4, 0.3]  # Probability of arriving at 8, 9, or 10 AM
    arrival_hours = [8, 9, 10]  # Corresponding to probabilities in arrive_probs
    depart_probs = [0.2, 0.4, 0.2, 0.2]  # Probability of leaving at 4, 5, 6, or 7 PM
    departure_hours = [16, 17, 18, 19]  # Corresponding to probabilities in depart_probs (4, 5, 6, 7 PM)

    for beginning_of_day in range(0, len(datetime_col), intervals_per_day):
        day_of_week = weekday[beginning_of_day]
        if random.random() <= weekday_probs[day_of_week - 1]:  # Sample whether they arrive that day
            arrival_hour = weighted_sample(arrival_hours, arrive_probs)  # Sample arrival hour based on probabilities
            departure_hour = weighted_sample(departure_hours, depart_probs)  # Sample departure hour based on probabilities
            
            # Convert hours to interval indices
            arrival_interval = beginning_of_day + int(arrival_hour * intervals_per_hour)
            departure_interval = beginning_of_day + int(departure_hour * intervals_per_hour)
            
            # Ensure we don't exceed array bounds
            arrival_interval = min(arrival_interval, len(occ_vec) - 1)
            departure_interval = min(departure_interval, len(occ_vec) - 1)
            
            # Set occupancy for all intervals between arrival and departure
            for interval in range(arrival_interval, departure_interval):
                occ_vec[interval] = 1
    return occ_vec


def draw_sched_FOB_complex(weather, prob_overnight=0.2):
    """
    Same as draw_sched_FOB but with a probability that an occupant stays overnight.
    At each potential departure (4-7 PM), the occupant may stay; if so, they remain
    until the next day's departure window (4-7 PM) and again roll for overnight.
    They can keep chaining overnight stays.
    """
    datetime_col = weather['DateTime']
    time_resolution = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600
    intervals_per_hour = int(1 / time_resolution)
    intervals_per_day = 24 * intervals_per_hour

    weekday = datetime_col.dt.weekday + 1
    occ_vec = np.zeros(len(datetime_col), dtype=int)

    weekday_probs = [0.42, 0.35, 0.35, 0.35, 0.42, 0, 0]
    arrive_probs = [0.3, 0.4, 0.3]
    arrival_hours = [8, 9, 10]
    depart_probs = [0.2, 0.4, 0.2, 0.2]
    departure_hours = [16, 17, 18, 19]

    in_building = False

    for beginning_of_day in range(0, len(datetime_col), intervals_per_day):
        end_of_day = min(beginning_of_day + intervals_per_day, len(occ_vec))
        day_of_week = weekday[beginning_of_day]

        if not in_building:
            # New day: do they show up?
            if random.random() <= weekday_probs[day_of_week - 1]:
                arrival_hour = weighted_sample(arrival_hours, arrive_probs)
                departure_hour = weighted_sample(departure_hours, depart_probs)
                arrival_interval = beginning_of_day + int(arrival_hour * intervals_per_hour)
                departure_interval = min(beginning_of_day + int(departure_hour * intervals_per_hour), len(occ_vec))

                for interval in range(arrival_interval, departure_interval):
                    if interval < len(occ_vec):
                        occ_vec[interval] = 1

                if random.random() <= prob_overnight:
                    for interval in range(departure_interval, end_of_day):
                        occ_vec[interval] = 1
                    in_building = True
        else:
            # In building from overnight: depart this day between 4-7 PM
            departure_hour = weighted_sample(departure_hours, depart_probs)
            departure_interval = min(beginning_of_day + int(departure_hour * intervals_per_hour), len(occ_vec))

            for interval in range(beginning_of_day, departure_interval):
                occ_vec[interval] = 1

            if random.random() <= prob_overnight:
                for interval in range(departure_interval, end_of_day):
                    if interval < len(occ_vec):
                        occ_vec[interval] = 1
                in_building = True
            else:
                in_building = False

    return occ_vec


def bayes_schedule_FOB(weather, input_random_seed):
    random_seed = int(input_random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    datetime_col = weather['DateTime']
    time_resolution = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours
    
    hour_step = np.arange(weather.shape[0])
    month = datetime_col.dt.month
    day = datetime_col.dt.day
    weekday = datetime_col.dt.weekday + 1  # Python weekday is 0-6 (Monday=0), so +1 to match Julia's 1-7
    hour = datetime_col.dt.hour
    minute = datetime_col.dt.minute

    # Initialize DataFrame for readability
    load_sched = pd.DataFrame({
        "Hour Step": hour_step,
        "DateTime": datetime_col,
        "Month": month,
        "Day": day,
        "Weekday": weekday,
        "Hour": hour,
        "Minute": minute
    })

    # ------ Draw Occupancy Schedule ---
    occupancy_df = load_sched.copy()

    for i in range(1, Input_Parameters.MaxOccupancy + 1):
        occupancy_vector = draw_sched_FOB(weather)
        occupancy_df[f"Person {i}"] = occupancy_vector

    tot_occ = occupancy_df.iloc[:, 7:].sum(axis=1)  # Sum occupancy rows at each timestep (adjusted index)
    occupancy_df["Total Occupancy"] = tot_occ
    load_sched["Total Occupancy"] = tot_occ

    # ------ Draw Loads Based on Occupancy ---

    # Load Probabilities and Specifications
    lighting_max_load = Input_Parameters.PeakLighting  # [kW] on if occupied
    comp_charge_low_weights = [0.50, 0.50, 0.0]
    comp_charge_high_weights = [0.25, 0.50, 0.25]
    comp_charge_loads = [0, 0.5, 1]  # [kW]
    kettle_low_weights = [0.70, 0.20, 0.10]
    kettle_high_weights = [0.50, 0.30, 0.20]
    kettle_loads = [0, 0.1333, 0.266]  # [kW]

    # Initializing new columns for loads
    load_sched["Lighting Load"] = 0.0
    load_sched["Computer Charging Load"] = 0.0
    load_sched["Kettle Load"] = 0.0

    # Assign loads based on occupancy
    for i, occ in enumerate(load_sched["Total Occupancy"]):
        if occ == 0:
            load_sched.at[i, "Computer Charging Load"] = 0
            load_sched.at[i, "Kettle Load"] = 0
            load_sched.at[i, "Lighting Load"] = 0
        elif occ <= 2:  # Light Occupancy
            load_sched.at[i, "Computer Charging Load"] = weighted_sample(comp_charge_loads, comp_charge_low_weights)
            load_sched.at[i, "Kettle Load"] = weighted_sample(kettle_loads, kettle_low_weights)
            load_sched.at[i, "Lighting Load"] = lighting_max_load
        else:  # High Occupancy
            load_sched.at[i, "Computer Charging Load"] = weighted_sample(comp_charge_loads, comp_charge_high_weights)
            load_sched.at[i, "Kettle Load"] = weighted_sample(kettle_loads, kettle_high_weights)
            load_sched.at[i, "Lighting Load"] = lighting_max_load

    load_sched["Total Load"] = np.round(load_sched["Lighting Load"] + load_sched["Computer Charging Load"] + load_sched["Kettle Load"], 3)

    load_sched["Total Heat Gain"] = load_sched["Total Occupancy"] * Input_Parameters.TotalPersonHeat + load_sched["Lighting Load"]  # [kW]
    return load_sched


def complex_bayes_schedule_FOB(weather, input_random_seed, prob_overnight=0.08, prob_baseline_load=0.5, baseline_load_kw=(0.02, 0.06), prob_training_load=0.05, training_load_kw=(0.5, 1.2)):
    """
    Complex Bayesian schedule with:
    - Occupancy that can stay overnight (prob_overnight per person per day)
    - Random computer loads independent of occupancy:
      - Baseline: high probability, low magnitude (idle/server)
      - Training job: low probability, high magnitude (compute-intensive)

    Parameters
    ----------
    weather : pd.DataFrame
        Must have 'DateTime' column.
    input_random_seed : int
        Seed for reproducibility.
    prob_overnight : float
        Probability per person, at each departure (4-7 PM), of staying overnight
        and departing the next day (4-7 PM); can chain across multiple nights.
    prob_baseline_load : float
        Probability per timestep that baseline computer load is on.
    baseline_load_kw : tuple
        (min_kw, max_kw) for uniform draw when baseline load is on.
    prob_training_load : float
        Probability per timestep that a compute-intensive training job runs.
    training_load_kw : tuple
        (min_kw, max_kw) for uniform draw when training load activates.
    """
    random_seed = int(input_random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    datetime_col = weather['DateTime']
    time_resolution = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600

    hour_step = np.arange(weather.shape[0])
    month = datetime_col.dt.month
    day = datetime_col.dt.day
    weekday = datetime_col.dt.weekday + 1
    hour = datetime_col.dt.hour
    minute = datetime_col.dt.minute

    load_sched = pd.DataFrame({
        "Hour Step": hour_step,
        "DateTime": datetime_col,
        "Month": month,
        "Day": day,
        "Weekday": weekday,
        "Hour": hour,
        "Minute": minute
    })

    # ------ Draw Occupancy Schedule (with overnight stay probability) ---
    occupancy_df = load_sched.copy()
    for i in range(1, Input_Parameters.MaxOccupancy + 1):
        occupancy_vector = draw_sched_FOB_complex(weather, prob_overnight=prob_overnight)
        occupancy_df[f"Person {i}"] = occupancy_vector

    tot_occ = occupancy_df.iloc[:, 7:].sum(axis=1)
    occupancy_df["Total Occupancy"] = tot_occ
    load_sched["Total Occupancy"] = tot_occ

    # ------ Draw Occupancy-Based Loads ---
    lighting_max_load = Input_Parameters.PeakLighting
    comp_charge_low_weights = [0.50, 0.50, 0.0]
    comp_charge_high_weights = [0.25, 0.50, 0.25]
    comp_charge_loads = [0, 0.5, 1]
    kettle_low_weights = [0.70, 0.20, 0.10]
    kettle_high_weights = [0.50, 0.30, 0.20]
    kettle_loads = [0, 0.1333, 0.266]

    load_sched["Lighting Load"] = 0.0
    load_sched["Computer Charging Load"] = 0.0
    load_sched["Kettle Load"] = 0.0

    for i, occ in enumerate(load_sched["Total Occupancy"]):
        if occ == 0:
            load_sched.at[i, "Computer Charging Load"] = 0
            load_sched.at[i, "Kettle Load"] = 0
            load_sched.at[i, "Lighting Load"] = 0
        elif occ <= 2:
            load_sched.at[i, "Computer Charging Load"] = weighted_sample(comp_charge_loads, comp_charge_low_weights)
            load_sched.at[i, "Kettle Load"] = weighted_sample(kettle_loads, kettle_low_weights)
            load_sched.at[i, "Lighting Load"] = lighting_max_load
        else:
            load_sched.at[i, "Computer Charging Load"] = weighted_sample(comp_charge_loads, comp_charge_high_weights)
            load_sched.at[i, "Kettle Load"] = weighted_sample(kettle_loads, kettle_high_weights)
            load_sched.at[i, "Lighting Load"] = lighting_max_load

    # ------ Random computer loads (independent of occupancy) ---
    # Baseline: high prob, low magnitude (idle computer/server)
    # Training job: low prob, high magnitude (compute-intensive)
    load_sched["Baseline Load"] = 0.0
    load_sched["Training Load"] = 0.0
    bl_min, bl_max = baseline_load_kw
    tl_min, tl_max = training_load_kw
    for i in range(len(load_sched)):
        if random.random() <= prob_baseline_load:
            load_sched.at[i, "Baseline Load"] = np.round(random.uniform(bl_min, bl_max), 3)
        if random.random() <= prob_training_load:
            load_sched.at[i, "Training Load"] = np.round(random.uniform(tl_min, tl_max), 3)

    load_sched["Total Load"] = np.round(
        load_sched["Lighting Load"] + load_sched["Computer Charging Load"] + load_sched["Kettle Load"]
        + load_sched["Baseline Load"] + load_sched["Training Load"],
        3
    )

    load_sched["Total Heat Gain"] = load_sched["Total Occupancy"] * Input_Parameters.TotalPersonHeat + load_sched["Lighting Load"]
    return load_sched


# check time alignment