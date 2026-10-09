# Input Parameters 

version = 1.0
code_name = "LBNL-RL"

calibration_file_path = "data/Calibration_Model_Input.xlsx"

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
T_low = 22.0 # [°C] Heating setpoint (heat toward this). MUST match physics/hp_tes/Input_Parameters.py.
T_high = 24.0 # [°C] Cooling setpoint (cool toward this). MUST match physics/hp_tes/Input_Parameters.py.
T_delta = 0.5 # [°C] Temperature Setpoint Deadband (thermostat hysteresis, ± around each setpoint)
T_indoor_constant = (T_low + T_high) / 2 # [°C] Constant Indoor Temperature (for the simplicity of a linear model)

# ── Comfort-scoring band for the reward (FR7) — aligned to the thermostat deadband ─────────────
# Discomfort is scored as excursions OUTSIDE this band, set to the thermostat's control range
# (setpoints ± hysteresis) so the controller is NOT penalized for the unavoidable hysteresis swings
# the deadband permits. The thermostat only CALLS heating below T_low-T_delta and cooling above
# T_high+T_delta, so within [T_low-T_delta, T_high+T_delta] the deadband is doing its job; discomfort
# is charged only BEYOND it — genuine control/capacity failure (e.g. during an outage). Applied
# identically to RBC and RL. Derived from T_low/T_high/T_delta so it can't drift out of sync — and
# T_low/T_high here MUST equal the dropin thermostat's (physics/hp_tes/Input_Parameters.py), both 22/24.
COMFORT_BAND_LOW = T_low - T_delta    # [°C] 21.5 = heating setpoint 22.0 − hysteresis 0.5
COMFORT_BAND_HIGH = T_high + T_delta  # [°C] 24.5 = cooling setpoint 24.0 + hysteresis 0.5

# Grid-tied reward constants (used by MF_RewardGridTied)
DEFAULT_ELECTRICITY_PRICE_USD_PER_KWH = 0.20 # [$/kWh] symmetric buy=sell when CSV has no Electricity_Price column
DISCOMFORT_USD_PER_DEGC_PER_H = 2.423 # [$/(°C·h)] linear discomfort cost outside [COMFORT_BAND_LOW, COMFORT_BAND_HIGH]

# Grid-tied V2 reward penalties (used by MF_RewardGridTiedV2) — all TUNABLE.
# reward = price_sell*export - price_buy*import - discomfort
#          - LOSS_OF_LOAD_PENALTY_USD_PER_KWH * loss_of_load_kWh
#          - infeasibility penalties (HP action not aligned with comfort call)
LOSS_OF_LOAD_PENALTY_USD_PER_KWH = 10.0 # [$/kWh] penalty for unserved load (e.g. during outages)
INFEASIBLE_ACTION_PENALTY = 0.25        # [$/substep] charging the wrong PCM store for the active comfort call
INFEASIBLE_NO_PCM_PENALTY = 0.25        # [$/substep] use-TES (action 3) while the aligned PCM is ~empty

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
# PVSize = 15 # [kW]
BatterySize = 3.6*3 # [kWh] 3 batteries in parallel
InverterSize = BatterySize/2 # [kW] 1/2 of the battery size by default

# ── Grid-tied V2 study (2x2 PCM-TES vs battery) ───────────────────────────────
# PV: "same as onsite" per study outline -> 0.32 kW * 12 panels = 3.84 kW.
# Used by Environment_GridTiedV2 to turn the CSV "PV Capacity Factor" [-] into power:
#   power_kW = PV Capacity Factor * PVSize ; energy_kWh = power_kW * δt_in_hr
PVSize = 0.32 * 12 # [kW] grid-tied PV array nameplate

# Battery sizing per scenario (TES vs no-TES). No-TES gets a cost-equivalent
# bigger pack = 2x the TES battery. Tunable: adjust BATTERY_TES_KWH only.
BATTERY_TES_KWH = 7.2                       # [kWh] TES scenarios (A, B) — 1 string / 2 Ecoflow
BATTERY_NO_TES_KWH = 2.0 * BATTERY_TES_KWH  # [kWh] no-TES scenarios (C, D) — cost-equivalent expansion


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
# NOTE (UNUSED): legacy pvlib-style derate block, not applied anywhere. PV energy is taken
# directly from the measured `PV Capacity Factor` column (Gen = CF * PVSize * dt_hr, see
# physics/blackbox.py) which already reflects real AC output, so no
# additional inverter/optical derate is applied by design. Kept for reference only.
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

# ── Battery cycling / degradation cost (B1) ───────────────────────────────────
# Amortized battery-replacement (wear) cost per kWh cycled — a real economic cost (the pack
# wears out and must be replaced), priced into the reward so the agent isn't handed a free
# battery. NOT modeled as physical capacity fade (negligible over a 1-yr horizon; only the
# $ cost matters here).
# Derivation (this project's pack = 2x EcoFlow, matching BATTERY_TES_KWH = 7.2 kWh):
#     CAPEX            = 2 units x $1600                 = $3200
#     cycle life       = 3500 cycles (to ~80% SOH)
#     capacity         = 7.2 kWh (nameplate)
#     cost / cycle     = $3200 / 3500                    = $0.914 / full cycle
#     cost / kWh disch = $0.914 / 7.2                    = $0.127 / kWh discharged
#   The reward charges TOTAL throughput = |dSOC| per step (BOTH charge and discharge), so a
#   full cycle registers 2x capacity of throughput -> halve the rate to avoid double-charging:
#     rate             = $0.127 / 2                      = $0.0635 / kWh of |dSOC| throughput
# Intensive: same $/kWh regardless of pack size. Utility-scale LFP would be ~$0.03/kWh; sweep
# this in the reward-weight study (B4).
BATTERY_DEGRADATION_USD_PER_KWH = 0.0635  # [$ / kWh of |dSOC| throughput]


def battery_soc_min_kwh(capacity_kwh, dod=None):
    """Minimum physical SOC [kWh] (reserve below usable band). DoD=0.8 -> 20% of pack."""
    if dod is None:
        dod = BatteryDoD
    return float(capacity_kwh) * (1.0 - float(dod))


def battery_usable_capacity_kwh(capacity_kwh, dod=None):
    """Usable energy band [kWh] between min SOC and full pack."""
    if dod is None:
        dod = BatteryDoD
    return float(capacity_kwh) * float(dod)


def battery_physical_soc_frac(soc_kwh, capacity_kwh):
    """Physical SOC as a fraction of nameplate pack capacity (e.g. 0.2–1.0 at DoD=0.8)."""
    return float(soc_kwh) / max(1e-9, float(capacity_kwh))


def battery_usable_soc_frac(soc_kwh, capacity_kwh, dod=None):
    """Map physical SOC [kWh] to agent-visible fraction in [0, 1] over the usable band."""
    usable = battery_usable_capacity_kwh(capacity_kwh, dod)
    if usable <= 0:
        return 0.0
    soc_min = battery_soc_min_kwh(capacity_kwh, dod)
    return max(0.0, min(1.0, (float(soc_kwh) - soc_min) / usable))

def battery_target_soc_kwh(capacity_kwh, target_usable_frac=None, dod=None):
    """Target physical SOC [kWh] at ``target_usable_frac`` through the usable DoD band."""
    if target_usable_frac is None:
        target_usable_frac = B_med_high
    soc_min = battery_soc_min_kwh(capacity_kwh, dod)
    usable = battery_usable_capacity_kwh(capacity_kwh, dod)
    return soc_min + float(target_usable_frac) * usable

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

η_PVIV = 0.94 # [1] UNUSED: PV(DC)->Home(AC) inverter efficiency (not applied; CF is measured AC — see note above)
# Round-trip efficiency = η_PV2B * η_B2H = 0.87, for the installed EcoFlow DELTA Pro (2x 3.6 kWh LFP).
# Basis: measured DC->AC DISCHARGE efficiency ~93% (thesolarlab / outdoorgearlab teardowns) + ~6-7%
# charge/conversion loss -> AC round-trip ~0.87. (The prior 0.93 was the discharge-only figure, too
# high for a round-trip; the original 0.98/0.98 -> 0.96 was optimistic.) Good B4 sensitivity point (0.85-0.90).
η_PV2B = 0.9327 # [1] PV(DC) to Battery(DC) inverter charging efficiency  (applied in loadmatch)
η_B2H = 0.9327  # [1] Battery discharging efficiency                       (applied in loadmatch)

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
# NOTE (M10 — read before editing): the cold latent bounds below (9-13 C) now ONLY set the TOTAL
# cold-store capacity PCM_C_Size (via PCM_C_Energy_Split). Stored energy is reported as
# calibrated-SOC x PCM_C_Size (see microgrid/environment.py and physics/rbc_rules.pcm_energies_kwh),
# so there is no longer a separate temperature->energy map (the old PCM_available_energy) that could
# disagree with the hp_tes model's 11-13 C phase-change window — both encode the same total latent
# (31500 J/(kg.K) x 4 K = 126 kJ/kg = the hp_tes COLD_PCM_LHEA=126000 J/kg; ~4.56 kWh total).
# Do NOT narrow this band alone: latent kWh = (upper-lower) x PCM_C_latent_density, so 9->11 without
# doubling the density would shrink PCM_C_Size (halve the latent) — i.e. accidentally resize the store.
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

