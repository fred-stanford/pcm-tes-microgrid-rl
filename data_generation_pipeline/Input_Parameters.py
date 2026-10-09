# Input Parameters 

version = 1.0
code_name = "LBNL-RL"

calibration_file_path = "Calibration_Model_Input.xlsx"

################################################ Simulation Parameters ################################################
δt = 300 # [s] Time Step for Simulation
δt_in_hr = δt/3600 # [hr] Time Step for Simulation
T_interval = 10 # [1] Number of Computation intervals for Modelica model
discount_rate = 0.01 # for urgency score calculation

# Device initial SOC
Intial_B_SOC = 0.5 # [1] Initial State of Charge of the Battery
hot_PCM_T_start = 51 # [°C] starting temperature of PCM hot
cold_PCM_T_start = 11 # [°C] starting temperature of PCM cold

################################################ Location Parameters ################################################
Berg_tilt = 15 # [degrees] Tilt angle of the Berg east of south
standard_meridian_longitude = -120; # [DEGREES] Longitude of standard meridian
longitude = -122.4286 # [DEGREES] Longitude of site
latitude = 37.4636 # [DEGREES] Latitude of site
timezone_offset = -8 # non daylight saving

################################################ Berg Structure Parameters ################################################

# Berg Envelope Parameters
L_wall = 6.1 # [m] Length of the Wall
H_wall = 2.4 # [m] Height of the Wall
R_wall = 2.6 # [m^2·K/W] Thermal Resistance of the Wall  It's based on time units of seconds
R_floor = 3.0 # [m^2·K/W] Thermal Resistance of the Floor
H_Ceiling = 2.0 # [m] Interior Height of the Ceiling (78.5 inch) 
Area = L_wall * L_wall # [m^2] Floor/Ceiling Area 
UA = L_wall*H_wall*4*(1/R_wall) + Area*2*(1/R_floor) # [W/K] Overall Heat Transfer Coefficient of the Berg Envelope
Volume = L_wall * L_wall * H_Ceiling # [m^3]
e_Berg = 0.75 # [1] emissivity of the Berg structure's surface (fibreglass) https://www.thermoworks.com/emissivity-table/    
TC = 593.487492705318 # [kJ/K] Thermal Capacitance of the Building Envelope

# Infiltration Parameters
Cs = 0.000145 # [(L/s)^2/(cm^4*°K)] Stack coefficient for building(house height = 1 story)
Cw = 0.000319 # [(L/s)^2/(cm^4*(m/s)^2)] (between 0.000319 and 0.000246 (house height =1 story, shelter class=2(typical shelter for an insolated rural house))) 
# Wind coefficient for building(house height =1 story, shelter class=1(no obstructions or local shielding))

# Effective leakage area measured during Blower Door Test 
# ELA =  47.1 # [in^2] (between 38.3(Dan's interpolation) and 47.1(Jessie's interpolation))
Al =  303.9 # [cm^2] using Jessie's interpolation
T_d = 273.15 - 1.67 # [°K] 99% Design Temperature for Half Moon Bay

# Environmental Parameters
wind_height = 9.144 # [m] The height above ground at which wind speed is measured. The PVWatts default is 9.144 m.
albedo = 0.18 # [1] albedo of grass
n_air = 1 # [1] Index of reflaction of air
n_AR = 1.3 # [1] Index of reflaction of AR coating
n_glass = 1.526 # [1] Index of reflaction of glass as in PVWatt model for standard modules

# Interior Standard

# Thermal Comfort Setpoints for the Hysteresis Logic
T_low = 22.0 # [°C] Heating setpoint (kept consistent with top-level + hp_tes Input_Parameters.py)
T_high = 24.0 # [°C] Cooling setpoint (kept consistent with top-level + hp_tes Input_Parameters.py)
T_delta = 0.5 # [°C] Temperature Setpoint Deadband
T_indoor_constant = (T_low + T_high) / 2 # [°C] Constant Indoor Temperature (for the simplicity of a linear model) 

# Ventilation
Ventilation = 15 # [CFM/PPL] Building Standard
Ra = 0.06 # [CFM/ft^2] Area Outdoor Air Rate; Source:CEE226E Week 5 - Energy Modeling Questions - Slide 9
Rp = 5 # [CFM/PPL] People Outdoor Air Rate; Source:CEE226E Week 5 - Energy Modeling Questions - Slide 9   

# Forward Operating Base Electrical Load Parameters
PeakLighting = 0.1 # [KW] Dan's suggestion
PeakPlugLoad = 0.1 # [KW] Computer = 40W, Phone = 10W, 2*(40 + 10) = 100 [W] = 0.1 [kW]
MaxOccupancy = 4 # [PPL] 4-MAN Office Room Plan
PersonLatentHeat = 200; # [BTU/hr/PPL]  CEE226E Slide
PersonSensibleHeat = 300; # [BTU/hr/PPL]  CEE226E Slide
TotalPersonHeat = (PersonSensibleHeat + PersonLatentHeat)/3412.142 # [kW/PPL]

################################################ Equipments Parameters ################################################

# Fixed System Capacity (NEED TO BE UPDATED)
PV_Size_Rooftop = 0.32 # [kW]
PV_Size_GroundMount = 0.305 # [kW]
PVSize = PV_Size_Rooftop + PV_Size_GroundMount  # [kW] total per-module reference capacity
BatterySize = 3.6*3 # [kWh] 3 batteries in parallel
InverterSize = BatterySize/2 # [kW] 1/2 of the battery size by default


# Solar PV Parameters
# Temperature model parameters
noct_installed = 45  # Nominal Operating Cell Temperature [°C]
module_height = 1.0   # Module height above ground [m]
wind_height = 2.0    # Wind speed measurement height [m]
module_emissivity = 0.84
module_absorption = 0.83
module_surface_tilt = 30  # [degrees]
module_width = 1.0    # [m]
module_length = 1.6   # [m]

# PV performance parameters
optical_loss = 0.96   # Account for glass reflectance
P_dc0 = 1.0           # DC power at STC [kW/kWp]
Γ_t = -0.004          # Temperature coefficient [1/°C]
T_ref = 25            # Reference temperature [°C]
η_PV = 0.96           # System efficiency factor

# Storage Self-Discharge Rate Parameters
BatteryLoss = 0.00003 # [/hr]
# PCM_H_Loss = 0.01 # [/hr]
# PCM_C_Loss = 0.01 # [/hr]
BatteryDoD = 0.8 # [1] Battery Depth of Discharge

B_low = 0.3 # [1] Lower Battery SOC Setpoint
B_high = 0.9 # [1] Upper Battery SOC Setpoint
B_med_low = 0.45 # [1] Medium Lower Battery SOC Setpoint
B_med_high = 0.75 # [1] Medium Upper Battery SOC Setpoint
B_delta = 0.1 # [1] Battery SOC Setpoint Deadband

LMP_low = 60 # [$/MWh] Lower LMP Setpoint
LMP_high = 150 # [$/MWh] Upper LMP Setpoint
LMP_med_low = 90 # [$/MWh] Medium Lower LMP Setpoint
LMP_med_high = 120 # [$/MWh] Medium Upper LMP Setpoint

# Charging and Discharging Efficiency Parameters
# η_PCM_H = 0.95 # [1] PCM hot charging and discharging efficiency
# η_PCM_C = 0.95 # [1] PCM cold charging and discharging efficiency

# MaxDischarge = 0.8   # [1]  Max Depth of Discharge for Battery

η_PVIV = 0.94 # [1] PV(DC) to Home(AC) inverter efficiency
η_PV2B = 0.9327 # [1] RTE 0.87 — kept consistent with top-level Input_Parameters.py (was 0.98)
η_B2H = 0.9327  # [1] RTE 0.87 — kept consistent with top-level Input_Parameters.py (was 0.98)

# HP-TES System Parameters
cycling_wait_time = 600 # [s] specifying how long the heat pumps cycles
heatExchangerHotScalingFactor = 0.07 # [1] PCM hot heat exchanger conductivity
heatExchangerColdScalingFactor = 0.03 # [1] PCM hot heat exchanger conductivity
heat_pump_condenser_evaporator_volume = 0.005 # [m^3] determines how soon the heat pump supply water temperature can reach the setpoint when switching between cooling mode and heating mode
# UA_nominal = 40 # [W/°K] fan coil unit heat exchanger conductivity, the higher the value, the easier it is to transfer heat between air and water --- change this to increase FCU capacity
UA_nominal = 80 # [W/°K]

# Viessmann Heat Pump
MaxHeaPumCapHeaSet = 17000 # [W] The maximum heating capacity (thermal power output) of the heat pump
MaxHeaPumCapCooSet = 15000 # [W] The maximum cooling capacity (thermal power output) of the heat pump
MinHeaPumCapHeaSet = 6000 # [W] The minimum heating capacity (thermal power output) of the heat pump
MinHeaPumCapCooSet = 5000 # [W] The minimum cooling capacity (thermal power output) of the heat pump

# HP_heatingpower = 2 # [kW] Design heating power of the heat pump (electrical power input)
# HP_coolingpower = 2 # [kW] Design cooling power of the heat pump (electrical power input)

#max_heating_power = HP_heatingpower * 1.5 # [kW] The maximum heating capacity (electrical power input) of the heat pump
#max_cooling_power = HP_coolingpower * 1.5 # [kW] The maximum cooling capacity (electrical power input) of the heat pump

space_heating_thermal = 2.5 # [kW] default heating call thermal power requirement
space_cooling_thermal = 2.5 # [kW]  default cooling call thermal power requirement

HP_SP_C_charge = 7 # [°C] optimal water temperature for heat pumpcooling
HP_SP_H_charge = 45 # [°C] optimal water temperature for heat pump heating

# Aermec Heat Pump
# MaxHeaPumCapHea = 15230 # [W] The maximum heating capacity (electrical power input) of the heat pump
# MaxHeaPumCapCoo = 11606 # [W] The maximum cooling capacity (electrical power input) of the heat pump
# MinHeaPumCapHea = 15230*0.3 # [W] The minimum heating capacity (electrical power input) of the heat pump
# MinHeaPumCapCoo = 11606*0.3 # [W] The minimum cooling capacity (electrical power input) of the heat pump

PCM_mass = 112 # [kg] mass of PCM
PCM_conversion_factor = 3600000 # [J/kWh] conversion factor from kWh to J
PCM_H_latent_density = 42500 * PCM_mass / PCM_conversion_factor # [J/kg·K] * [kg] / [J/kWh] = [kWh/K]  latent density of PCM hot
PCM_C_latent_density = 31500 * PCM_mass / PCM_conversion_factor # [J/kg·K] * [kg] / [J/kWh] = [kWh/K]  latent density of PCM cold
PCM_H_sensible_density = 3150 * PCM_mass / PCM_conversion_factor # [J/kg·K] * [kg] / [J/kWh] = [kWh/K]  sensible heat capacity of PCM hot
PCM_C_sensible_density = 2050 * PCM_mass / PCM_conversion_factor # [J/kg·K] * [kg] / [J/kWh] = [kWh/K]  sensible heat capacity of PCM cold

PCM_H_lower_sensible = 44 # [°C] temperature of the lower bound of the sensible range of PCM hot
PCM_H_lower_latent = 49 # [°C] temperature of the lower bound of the latent range of PCM hot
PCM_H_upper_latent = 53 # [°C] temperature of the upper bound of the latent range of PCM hot
PCM_H_upper_sensible = 58 # [°C] temperature of the upper bound of the sensible range of PCM hot
PCM_C_lower_sensible = 4 # [°C] temperature of the lower bound of the sensible range of PCM cold
PCM_C_lower_latent = 9 # [°C] temperature of the lower bound of the latent range of PCM cold
PCM_C_upper_latent = 13 # [°C] temperature of the upper bound of the latent range of PCM cold
PCM_C_upper_sensible = 18 # [°C] temperature of the upper bound of the sensible range of PCM cold

PCM_H_Energy_Split = [(PCM_H_lower_latent - PCM_H_lower_sensible)*PCM_H_sensible_density, (PCM_H_upper_latent - PCM_H_lower_latent)*PCM_H_latent_density, (PCM_H_upper_sensible - PCM_H_upper_latent)*PCM_H_sensible_density]
PCM_C_Energy_Split = [(PCM_C_upper_sensible - PCM_C_upper_latent)*PCM_C_sensible_density, (PCM_C_upper_latent - PCM_C_lower_latent)*PCM_C_latent_density, (PCM_C_lower_latent - PCM_C_lower_sensible)*PCM_C_sensible_density]

PCM_H_Size = sum(PCM_H_Energy_Split) # [kWh]    
PCM_C_Size = sum(PCM_C_Energy_Split) # [kWh] 
PCM_H_Power_Capacity = PCM_H_Size/2 # [kW]
PCM_C_Power_Capacity = PCM_C_Size/2 # [kW]

################################# Reward function weights ##############################################
# Relative importance (i.e. coefficient) of each component of the reward function
# These will automatically be normalized so that the final reward value is within a range [0,1]

reward_weights = {
    'Battery SOC': 4,
    'PCM_Hot_SOC': 3,
    'PCM_Cold_SOC': 3
}

