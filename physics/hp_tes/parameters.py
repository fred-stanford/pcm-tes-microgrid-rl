"""
hp_tes/parameters.py - All constants and flow rate helper for the HP-TES model.
"""

import math
from . import Input_Parameters

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

K_OFFSET = 273.15  # Celsius to Kelvin offset
CP_WATER = 4186.0  # J/(kg·K) specific heat of water
RHO_WATER = 1000.0  # kg/m³ density of water

# HP supply water setpoints (PLC xHeaSP; measured data shows 56-60°C during charging)
HP_SUPPLY_HEATING_K = K_OFFSET + 60.0  # 333.15 K
HP_SUPPLY_COOLING_K = K_OFFSET + 7.0   # 280.15 K

# PCM parameters (physical solidus/liquidus for real Sunamp material)
HOT_PCM_TSOL = K_OFFSET + 49.0   # 322.15 K — physical solidus
HOT_PCM_TLIQ = K_OFFSET + 53.0   # 326.15 K — physical liquidus
COLD_PCM_TSOL = K_OFFSET + 11.0  # 284.15 K
COLD_PCM_TLIQ = K_OFFSET + 13.0  # 286.15 K

TES_NOMINAL = 32_400_000  # 32.4 MJ (was 34.2 MJ, FMU Tes_nominal)
PCM_MASS = 112.0  # kg (Sunamp Q9 unit, was 271.4 from Input_Parameters)

# Per-material latent heat of fusion (actual PCM material property)
# PCM_MASS=112 kg is already the PCM mass itself, so no packing fraction needed.
# Old values (41584, 25200) had a packing fraction baked in that was calibrated
# for the FMU's wrong mass of 271 kg — now corrected.
HOT_PCM_LHEA = 150000.0  # J/kg  — calibrated to match 5.9 kWh deep discharge capacity (was 170000)
COLD_PCM_LHEA = 80000.0  # J/kg  — reduced to match cold discharge capacity (was 126000)
# L=80 is confirmed by the Apr 6-7 2026 deep complete cold discharge
# (17.7 h, T1 10.2->19.2, T3 6.3->20.2 -- both end well above the latent
# band, PCM fully melted, T_in mean 16.5 C). Energy balance on that
# dataset yields L = 78.2 kJ/kg, so the calibrated 80 is essentially the
# physical value for this PCM. The Sunamp datasheet 126 does NOT match.
#
# Known model limitation -- LOW driving-force cold discharge:
# when the load return water sits at or near the latent band (e.g. apr29
# and apr30 2026 cycles, T_in ~11-12 C), the single-bulk-node model
# under-predicts measured capacity by ~55%. It runs out of accessible
# cold once T_sh reaches T_in, while the real stratified tank still has
# a cold reservoir below it. Discharge-only fits in that regime push L
# to 230-300 kJ/kg as a proxy for stratification -- that is NOT the real
# L (Apr 6-7 disproves it). The fix is a layered/stratified PCM model,
# not a different L. See fit_cold_continuous.py and
# diagnose_cold_capacity.py for the supporting diagnostics.

# Per-material sensible heat capacity
HOT_PCM_C = 2800.0   # J/(kg·K)  — calibrated to match deep cycle capacity (was 3150)
COLD_PCM_C = 1500.0   # J/(kg·K)  — reduced to match cold discharge capacity (was 2050)

# Backward-compat alias (validate_components.py imports PCM_C_SENSIBLE)
PCM_C_SENSIBLE = 2050.0

# Heat loss coefficients (W/K) — component-level sweep on feb7_8 (48h, good mode mix)
# Hot and cold tanks are identical geometry/insulation, so no active/standby split
# is physically motivated; the STANDBY/ACTIVE symbols exist purely for schema symmetry.
#
# Source-of-truth note (added 2026-06-19, audit during multi-project alignment):
#   These values are LIVE — read by hp_tes/system.py:544 (HOT) and :599 (COLD)
#   for the ambient-loss term. The shared library hvac-component-models also
#   carries h_loss_W_per_K = 4.2 in PCM_Q9_HOT and 3.5 in PCM_Q3_COLD, so this
#   model and the Aris MF Combi model are numerically aligned.
#
#   They DIFFER from the Purdue paper's directly-measured *pure standby*
#   coefficient of 1.9 W/K (22 cool-down events). That is not a contradiction:
#   the 4.2 here is an *effective total-loss* coefficient — a single number
#   that has to absorb both standby and active-period losses because the
#   1-zone shell-loss formulation cannot represent them separately. The paper
#   pins 1.9 W/K for standby and treats the active part separately in a
#   2-node formulation. Both values are correct for their own definition.
#
#   Aligning code to the paper's 1.9 W/K would require a coordinated change
#   across THREE places at once: (a) split STANDBY=1.9, ACTIVE=<new fit> here;
#   (b) the same in hvac-component-models PCMParams + presets; (c) the call
#   site in system.py and the library's step_2node. That recalibration is
#   owned by the Aris session a96927b7. Do not flip values here in isolation.
H_FIXED_HOT = 4.2             # W/K — component sweep optimal=4.197, RMSE=0.834
H_FIXED_COLD = 3.5            # W/K — component sweep feb5_9 optimal=4.8 (RMSE=0.979), 3.5 for cross-segment stability

# Symmetric standby/active aliases (currently equal — split is schema only, not physics)
H_FIXED_HOT_STANDBY = H_FIXED_HOT
H_FIXED_HOT_ACTIVE = H_FIXED_HOT
H_FIXED_COLD_STANDBY = H_FIXED_COLD
H_FIXED_COLD_ACTIVE = H_FIXED_COLD

# Heat exchanger scaling factors (legacy power-fraction, kept for backward-compat)
HX_SCALE_HOT = 0.50
HX_SCALE_COLD = 0.30

# ── Sunamp Q9 Serpentine Coil Geometry ──
# Two parallel serpentine copper coils with metal fins.
# Each coil has ~27 pass segments (54 total segments / 2 coils).
# Water flows through all segments of one coil IN SERIES, not in parallel.
# "6 banks x 9 tubes" from the datasheet = 54 pass segments, not 54 parallel tubes.
TUBE_DI = 0.00405       # m  inner diameter
N_COILS = 2              # parallel serpentine coils
N_PASSES_PER_COIL = 27   # pass segments per coil (54 total / 2)
TUBE_LENGTH_PER_PASS = 0.485  # m  depth of one pass segment
TUBE_LENGTH_TOTAL = N_PASSES_PER_COIL * TUBE_LENGTH_PER_PASS  # ~13.1 m per coil
A_TUBE_TOTAL = math.pi * TUBE_DI * N_COILS * TUBE_LENGTH_TOTAL  # ~0.333 m² (same total area)
# Backward-compat aliases
N_TUBES_TOTAL = N_COILS * N_PASSES_PER_COIL  # 54 (for code that imports this)
TUBE_LENGTH = TUBE_LENGTH_PER_PASS

# ── Water Properties ──
# Temperature-dependent properties are now computed in tes_model.water_properties().
# These fixed values are kept for backward compatibility with scripts that import them.
WATER_PR = 4.31           # Prandtl number at 45 C (was 7.56 at 25C — fixed)
WATER_K_THERMAL = 0.638   # W/(m·K) at 45 C (was 0.598)
WATER_MU = 5.96e-4        # Pa·s at 45 C (was 8.9e-4 at 25C — fixed)

# ── PCM-Side Thermal Conductance (series resistance model) ──
# Overall HX UA computed as:  1/UA_total = 1/UA_water + 1/UA_pcm
#   UA_water = h_DB(Re,Pr,T) * A_tube  (~18,000 W/K at charge flow, negligible resistance)
#   UA_pcm = calibrated parameter: conduction through PCM + fin efficiency
# This makes UA_total nearly independent of flow rate (correct when
# PCM-side resistance is 97% of total). The calibrated UA_pcm has a
# direct physical meaning: thermal conductance from tube wall through
# fins and PCM to the bulk temperature node [W/K].
UA_PCM_HOT_CHARGE    = 870.0    # W/K — charge PCM-side conductance (fitted)
UA_PCM_HOT_DISCHARGE = 10000.0  # W/K — discharge: effectively infinite (ε saturated)
UA_PCM_COLD_CHARGE   = 1120.0   # W/K — fit_cold.py charge (Apr 8, Feb 5-6, Jan 22)
UA_PCM_COLD_DISCHARGE = 10000.0 # W/K — effectively infinite (ε saturated, small ΔT regime)

# Legacy DB_SCALING aliases (backward compat for older scripts)
DB_SCALING_HOT = 0.030
DB_SCALING_COLD = 0.030
DB_SCALING_HOT_DISCHARGE = 0.05
DB_SCALING_COLD_DISCHARGE = 0.05

# ── 2-Node Shell-and-Core PCM Parameters ──
F_SHELL_HOT = 0.40     # shell mass fraction — fit_discharge grid sweep (was 0.15)
F_SHELL_COLD = 0.40     # same geometry as hot (serpentine coil with fins)
UA_SC_HOT = 4930.0       # charging — series resistance fit
UA_SC_COLD = 5000.0      # charging — fit_cold.py
UA_SC_HOT_DISCHARGE = 1050.0  # discharge — serpentine DB fit
UA_SC_COLD_DISCHARGE = 5000.0 # discharge — fit_cold.py
UA_SC_HOT_STANDBY = 1000.0   # standby (no flow): hot PCM works fine with high coupling
UA_SC_COLD_STANDBY = 1000.0  # standby
R_SOLID_FACTOR = 0.3   # at solidus (lf=0), UA_SC scales to 30% — prevents core from overpowering water cooling

# C_eff smoothing buffer — ramp C_eff over [TSol-buf, TSol] and [TLiq, TLiq+buf]
# to prevent numerical oscillation at the latent-sensible boundary.
# Uses trapezoidal profile to conserve total latent heat.
CEFF_BUFFER = 2.0  # K — wider ramp prevents transient oscillation at TSol boundary

# ── Water Circuit Volumes ──
# Single loop volume (HP condenser to PCM and back). Was split into
# V_HP_OUTLET_L=5 + V_HOT_HTF_L=15 = 20L, but it's one continuous loop.
V_WATER_LOOP_L = 15.0     # L — piping between HP and PCM/FCU (excl. HP condenser)
V_FCU_WATER_L = 5.0        # L — FCU return water tracking (discharge loop)
# Legacy aliases (backward compat)
V_HP_OUTLET_L = 0.0
V_HOT_HTF_L = V_WATER_LOOP_L
V_COLD_HTF_L = V_WATER_LOOP_L

# FCU
# FCU UA — split by mode (component-level validation Feb 18)
# Cooling UA is higher: wet coil (condensation) + larger dT improves heat transfer
UA_HEATING = 110.0   # W/K — heating modes (M2, M3, M10), sweep optimal on feb7_8
UA_COOLING = 140.0   # W/K — cooling modes (M6, M7, M11), median of M6=130/M7=148
UA_NOMINAL = UA_HEATING  # backward-compat alias
FCU_FAN_POWER = 75.0  # W
FCU_FAN_MASS_FLOW = 0.15  # kg/s

# Pump parameters
PUMP_POWER = 75.0  # W per pump when running
PUMP_MASS_FLOW = 0.15  # kg/s (legacy fallback)

# Mode-dependent flow rates from WAGO PLC code (GPM -> kg/s)
GPM_TO_KGS = 0.06309
MODE_FLOW_RATES = {
    0:  0.0 * GPM_TO_KGS,
    1:  6.2 * GPM_TO_KGS,
    2:  3.0 * GPM_TO_KGS,
    3:  3.8 * GPM_TO_KGS,
    5:  6.2 * GPM_TO_KGS,
    6:  3.0 * GPM_TO_KGS,
    7:  3.8 * GPM_TO_KGS,
    10: 4.1 * GPM_TO_KGS,
    11: 4.1 * GPM_TO_KGS,
}


def get_mode_flow_rate(mode):
    """Return water mass flow rate [kg/s] for the given operating mode."""
    return MODE_FLOW_RATES.get(mode, 0.0)

# Thermostat setpoints
T_HEAT_SP = K_OFFSET + Input_Parameters.T_low   # 295.15 K (22°C heating setpoint)
T_COOL_SP = K_OFFSET + Input_Parameters.T_high  # 297.15 K (24°C cooling setpoint)
T_DELTA = Input_Parameters.T_delta  # 0.5 K hysteresis

# HP thermal inertia
HP_VOLUME = Input_Parameters.heat_pump_condenser_evaporator_volume  # 0.005 m³
HP_THERMAL_MASS = HP_VOLUME * RHO_WATER * CP_WATER  # J/K (~20930 J/K)

# HP compressor startup ramp (scroll compressor + refrigerant circuit stabilization)
# Real AWHP takes ~3 min to reach full capacity from cold start.
# Measured feb7_8: HP electric ramps from 0→~1kW over first 2-3 minutes.
# Linear ramp: ramp_factor = min(1.0, runtime / HP_STARTUP_TIME)
# Balance: short enough to not starve PCM charging in short cycles (~15 min),
# but long enough to soften the startup spike.
HP_STARTUP_TIME = 180.0  # seconds — linear ramp from 0→100% capacity

# HP protection limits
HP_MAX_SUPPLY_K = K_OFFSET + 65.0  # High-temp protection shutoff
HP_MIN_SUPPLY_K = K_OFFSET + 2.0   # Low-temp protection shutoff

# HP compressor lift limits (measured data: mean 7.1C heating, 6.5C cooling)
HP_MAX_LIFT_HEATING = 10.0  # K — max (T_supply - T_return) in heating mode
HP_MAX_LIFT_COOLING = 8.0   # K — max (T_return - T_supply) in cooling mode

# Effective water loop thermal mass for energy balance [J/K]
# HP condenser + loop piping (single loop, not split)
LOOP_THERMAL_MASS_HOT = (HP_VOLUME + V_WATER_LOOP_L / 1000.0) * RHO_WATER * CP_WATER
LOOP_THERMAL_MASS_COLD = (HP_VOLUME + V_WATER_LOOP_L / 1000.0) * RHO_WATER * CP_WATER

# SOC hysteresis thresholds
# Our SOC can exceed 1.0 (hot SOC~1.6 at 52°C, ~1.8 at 55°C).
# Let physics (HP supply vs PCM ΔT) limit charging, not SOC thresholds.
SOC_DISCHARGE_HIGH = 0.05
SOC_CHARGE_LOW = 0.95     # Resume charging below this (~48.7°C hot)
SOC_CHARGE_HIGH = 2.0     # Stop charging above this (effectively unlimited)

# COP and capacity polynomial coefficients
HEATING_COP_COEFF = [
    7.202003525081306,
    0.18062703077853665,
    -0.1477341556516753,
    -0.002381072467636457,
    0.0008319667199337608,
    0.0008384541446034042,
]
COOLING_COP_COEFF = [
    8.650914785834006,
    -0.20763083699950602,
    -0.06930220567653733,
    -0.0003181274443609665,
    0.001455586245110767,
    -0.000373410550444541,
]
HEATING_CAP_COEFF = [
    7662.690876075606,
    278.0926084677243,
    13.264680546280954,
    -2.744177232083285,
    -0.42115501152681524,
    -1.3852105061937135,
]
COOLING_CAP_COEFF = [
    2176.431007927367,
    14.1847062996531,
    -26.923757618843087,
    -0.10684511342535456,
    -0.628998968008253,
    -0.179778200817228,
]

# HP capacity scale factors — part-load derating
# The Viessmann AM2V 020028 lookup table is at 100% compressor speed (rated capacity).
# Real HP runs at part-load (VSD compressor) — measured ~50-60% of rated heating capacity.
# Scaling <1.0 helps HP Supply metrics but degrades Hot PCM on feb5_9 (short charging
# cycles get starved). Proper fix: compressor speed model or measured part-load curves.
HP_HEATING_CAP_SCALE = 1.0   # Keep at 1.0 — flat scaling degrades PCM charging accuracy
HP_COOLING_CAP_SCALE = 1.0   # Insufficient cooling data to calibrate; keep 1.0

# Legacy alias (backward compat)
HP_CAPACITY_PART_LOAD_FACTOR = HP_HEATING_CAP_SCALE
