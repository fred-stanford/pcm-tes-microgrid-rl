"""
hp_tes_rl_dropin/system.py - Top-level system assembly: HybridSystemState,
_simulate_interval, run_fmu.

DROP-IN REPLACEMENT for hp_tes/system.py.

The EXTERNAL CONTRACT is identical to hp_tes: same HybridSystemState /
_simulate_interval / run_fmu names and signatures, same snapshot keys, same
run_fmu (df, cleaned_output) column/key sets and units.

Only the PCM (TES) physics is swapped.  The old 2-node shell-and-core PCM model
(hot/cold_pcm_temp_shell/core, UA_SC coupling, effective_heat_capacity Euler
update, compute_water_to_pcm_ua heat transfer) is replaced by a single-node
``PCMNode`` per tank whose physics is the validated v2 ``PCMBattery``
calibration (NTU-epsilon with baked C_eff curves).  See ``_stepper.py``.

Everything else -- PLC valve gating, loop mixing volume, ambient standby loss,
HP model, FCU, controller, and mode logic -- is preserved.
"""

import math
import numpy as np
import pandas as pd
from . import Input_Parameters
from . import Data_Conversion

from .parameters import (
    K_OFFSET, CP_WATER, RHO_WATER,
    HP_SUPPLY_HEATING_K, HP_SUPPLY_COOLING_K,
    HOT_PCM_TSOL, HOT_PCM_TLIQ, COLD_PCM_TSOL, COLD_PCM_TLIQ,
    H_FIXED_HOT_STANDBY, H_FIXED_HOT_ACTIVE, H_FIXED_COLD,
    V_WATER_LOOP_L, V_FCU_WATER_L,
    FCU_FAN_POWER, FCU_FAN_MASS_FLOW,
    PUMP_POWER,
    UA_NOMINAL, UA_HEATING, UA_COOLING,
    HP_THERMAL_MASS, HP_STARTUP_TIME,
    HP_MAX_SUPPLY_K, HP_MIN_SUPPLY_K,
    HP_MAX_LIFT_HEATING, HP_MAX_LIFT_COOLING,
    LOOP_THERMAL_MASS_HOT, LOOP_THERMAL_MASS_COLD,
    SOC_DISCHARGE_HIGH,
    get_mode_flow_rate,
)
from .mixing_volume import MixingVolume
from .tes_model import compute_soc
from .hp_model import compute_cop, compute_cop_and_capacity
from .controller import (
    update_thermostat, update_soc_hysteresis, determine_mode,
)
from ._stepper import PCMNode, HOT_CHEM, COLD_CHEM

# Density [kg/L] used to convert the loop mass flow [kg/s] into the volumetric
# flow [L/min] the PCMNode NTU-epsilon correlation expects.  Matches the value
# baked into the stepper so mdot round-trips exactly to system_flow.
_RHO_LPM = 0.997


def _kgs_to_lpm(m_dot_kgs):
    """Loop mass flow [kg/s] -> volumetric flow [L/min] for the PCM stepper."""
    if m_dot_kgs <= 0:
        return 0.0
    return m_dot_kgs * 60.0 / _RHO_LPM


# ── External mode injection table ──
# Maps OC_Mod integer to the same 8-tuple that determine_mode() returns:
# (mode, hp_on, hp_heating, fcu_mode, charges_hot, charges_cold, discharges_hot, discharges_cold)
_EXTERNAL_MODE_FLAGS = {
    0:  (0.0,  False, True,  0,  False, False, False, False),
    1:  (1.0,  True,  True,  0,  True,  False, False, False),
    2:  (2.0,  False, True,  1,  False, False, True,  False),
    3:  (3.0,  True,  True,  1,  True,  False, True,  False),
    5:  (5.0,  True,  False, 0,  False, True,  False, False),
    6:  (6.0,  False, False, -1, False, False, False, True),
    7:  (7.0,  True,  False, -1, False, True,  False, True),
    10: (10.0, True,  True,  1,  False, False, False, False),
    11: (11.0, True,  False, -1, False, False, False, False),
}


# ═══════════════════════════════════════════════════════════════════════════════
# State Class
# ═══════════════════════════════════════════════════════════════════════════════

class HybridSystemState:
    """Persistent state between iterations (replaces FMU state save/restore)."""

    def __init__(self, state=None):
        # `state` is accepted for RL back-compat (the env constructs this as
        # HybridSystemState(self.state)); initial conditions come from Input_Parameters, so it
        # is intentionally unused here.
        # ── Single-node PCM tanks (v2 PCMBattery calibration) ──
        hot_init_C = float(Input_Parameters.hot_PCM_T_start)    # 51 °C
        cold_init_C = float(Input_Parameters.cold_PCM_T_start)  # 11 °C

        self.hot_node = PCMNode(HOT_CHEM, T_init_C=hot_init_C)
        self.cold_node = PCMNode(COLD_CHEM, T_init_C=cold_init_C)

        hot_init_K = K_OFFSET + hot_init_C
        cold_init_K = K_OFFSET + cold_init_C

        # Node temps [K] (for SOC, reporting, and backward-compat).  These are
        # the single-node bulk temperatures; the old shell/core split is gone.
        self.hot_pcm_temp = hot_init_K
        self.cold_pcm_temp = cold_init_K

        # HP state
        self.hp_on = False
        self.hp_mode_heating = True
        self.hp_supply_temp = hot_init_K
        self.hp_return_temp = hot_init_K

        # Control state
        self.exact_system_mode = 0.0
        self.fcu_mode = 0

        # SOC hysteresis flags
        hot_soc_init = compute_soc(self.hot_pcm_temp, HOT_PCM_TSOL, HOT_PCM_TLIQ)
        cold_soc_init = compute_soc(self.cold_pcm_temp, COLD_PCM_TSOL, COLD_PCM_TLIQ)
        self.hot_soc_not_discharged = (hot_soc_init >= SOC_DISCHARGE_HIGH)
        self.cold_soc_not_discharged = (cold_soc_init >= SOC_DISCHARGE_HIGH)
        self.hot_soc_not_charged = (hot_soc_init < 1.0)
        self.cold_soc_not_charged = (cold_soc_init < 1.0)

        # Thermostat hysteresis state
        self.call_for_heating = False
        self.call_for_cooling = False

        # HP compressor speed (0-1)
        self.hp_compressor_speed = 0.0

        # HP startup tracking (soft-start ramp)
        self.hp_on_prev = False
        self.hp_on_since = 0.0

        # Time tracking
        self.time = 0.0

        # ── Mixing volumes (single loop + FCU return tracker) ──
        self.loop_vol = MixingVolume(V_WATER_LOOP_L, self.hp_supply_temp)
        self.fcu_water_vol = MixingVolume(V_FCU_WATER_L, K_OFFSET + 23.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Time Integration
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_interval(state, schedule_input, room_temp_K, outside_air_temp_K,
                        t_start, t_end, substep_dt=5.0, external_mode=None,
                        external_flow=None):
    """
    Simulate one control interval [t_start, t_end] with fixed substeps.

    Returns a list of snapshot dicts (one per substep boundary) for
    constructing the output DataFrame.
    """
    n_steps = max(1, int(round((t_end - t_start) / substep_dt)))
    dt = (t_end - t_start) / n_steps

    room_C = room_temp_K - K_OFFSET

    snapshots = []

    for step_i in range(n_steps + 1):
        t = t_start + step_i * dt

        # ── Start-of-substep single-node PCM temps [K] ──
        state.hot_pcm_temp = K_OFFSET + state.hot_node.T
        state.cold_pcm_temp = K_OFFSET + state.cold_node.T
        hot_T0_K = state.hot_pcm_temp      # frozen for the whole substep
        cold_T0_K = state.cold_pcm_temp

        # ── Current SOCs (from node temp) ──
        hot_soc = compute_soc(state.hot_pcm_temp, HOT_PCM_TSOL, HOT_PCM_TLIQ)
        cold_soc = compute_soc(state.cold_pcm_temp, COLD_PCM_TSOL, COLD_PCM_TLIQ)

        # ── Mode determination ──
        if external_mode is not None:
            # External mode injection: bypass controller entirely
            flags = _EXTERNAL_MODE_FLAGS.get(int(external_mode), _EXTERNAL_MODE_FLAGS[0])
            (mode, hp_on, hp_heating, fcu_mode,
             charges_hot, charges_cold, discharges_hot, discharges_cold) = flags
            update_soc_hysteresis(state, hot_soc, cold_soc)  # still needed for SOC reporting
        else:
            # Normal controller path — unchanged
            update_thermostat(state, room_temp_K)
            update_soc_hysteresis(state, hot_soc, cold_soc)

            hot_can_discharge = state.hot_soc_not_discharged
            cold_can_discharge = state.cold_soc_not_discharged
            hot_can_charge = state.hot_soc_not_charged
            cold_can_charge = state.cold_soc_not_charged

            (mode, hp_on, hp_heating, fcu_mode,
             charges_hot, charges_cold,
             discharges_hot, discharges_cold) = determine_mode(
                schedule_input, state.call_for_heating, state.call_for_cooling,
                hot_can_discharge, cold_can_discharge,
                hot_can_charge, cold_can_charge
            )

        state.exact_system_mode = mode
        state.hp_on = hp_on
        state.hp_mode_heating = hp_heating
        state.fcu_mode = fcu_mode

        # Mode-dependent water flow rate (from WAGO PLC)
        system_flow = get_mode_flow_rate(mode)
        if external_flow is not None:
            system_flow = external_flow

        # Volumetric flow [L/min] for the PCM node NTU-epsilon correlation
        flow_lpm = _kgs_to_lpm(system_flow)

        # ── HP compressor startup tracking ──
        if hp_on and not state.hp_on_prev:
            state.hp_on_since = t

        # ── HP model: compute COP at current conditions (Viessmann lookup) ──
        cop_heating = compute_cop(outside_air_temp_K, state.hp_supply_temp, heating=True)
        cop_cooling = compute_cop(outside_air_temp_K, state.hp_supply_temp, heating=False)

        hp_thermal_power = 0.0
        hp_electric_power = 0.0
        hp_useful_thermal = 0.0
        hp_compressor_speed = 0.0
        target_supply = state.hp_supply_temp
        ramp_power = 0.0
        hp_thermal_power_max = 0.0

        if hp_on:
            hp_compressor_speed = 1.0
            if hp_heating:
                target_supply = HP_SUPPLY_HEATING_K
                cop_at_sp, cap_at_sp = compute_cop_and_capacity(
                    outside_air_temp_K, target_supply, True)
                _, cap_at_current = compute_cop_and_capacity(
                    outside_air_temp_K, state.hp_supply_temp, True)
                hp_thermal_power_max = cap_at_current
                ramp_power = cap_at_sp
            else:
                target_supply = HP_SUPPLY_COOLING_K
                cop_at_sp, cap_at_sp = compute_cop_and_capacity(
                    outside_air_temp_K, target_supply, False)
                _, cap_at_current = compute_cop_and_capacity(
                    outside_air_temp_K, state.hp_supply_temp, False)
                hp_thermal_power_max = cap_at_current
                ramp_power = cap_at_sp

        # ── HP soft-start ramp ──
        hp_ramp_factor = 1.0
        if hp_on and HP_STARTUP_TIME > 0:
            runtime = max(0.0, t - state.hp_on_since)
            hp_ramp_factor = min(1.0, runtime / HP_STARTUP_TIME)
            hp_thermal_power_max *= hp_ramp_factor
            ramp_power *= hp_ramp_factor

        state.hp_compressor_speed = hp_compressor_speed

        # ══════════════════════════════════════════════════════════════
        # Thermal chain: HP → FCU (if active) → TES → back to HP
        # ══════════════════════════════════════════════════════════════

        heat_transfer_hot = 0.0
        heat_transfer_cold = 0.0
        fcu_heat = 0.0
        fcu_entering_water_temp = state.hp_supply_temp
        fcu_leaving_water_temp = state.hp_supply_temp
        fan_mass_flow = 0.0
        fan_power = 0.0

        # Heat rate INTO each pcm node [W] (stepper sign convention: + heats the
        # node).  Drives the end-of-substep node update.  Distinct from the
        # heat_transfer_hot/cold OUTPUT signs (see below).
        Q_hot_into_pcm = 0.0
        Q_cold_into_pcm = 0.0

        # ── Pumps: simplified — pump1 when HP on, pump2 for pure discharge ──
        pump1_power = 0.0
        pump2_power = 0.0
        pump1_flow = 0.0
        pump2_flow = 0.0

        if hp_on and system_flow > 0:
            pump1_power = PUMP_POWER
            pump1_flow = system_flow
        elif (discharges_hot or discharges_cold) and system_flow > 0:
            pump2_power = PUMP_POWER
            pump2_flow = system_flow

        # ── Step 1: Discharge (pure discharge modes 2/6 only) ──
        # Water outlet temp is derived from the stepper's Q for energy
        # consistency: heat into water = -Q, so T_out = T_in - Q/(mdot*Cp).
        discharge_water_temp_hot = hot_T0_K
        discharge_water_temp_cold = cold_T0_K
        pcm_water_inlet_hot = hot_T0_K  # default: no flow → PCM temp

        if discharges_hot and not charges_hot and system_flow > 0:
            T_water_in_hot = state.fcu_water_vol.temp if fcu_mode != 0 else room_temp_K
            # v2 PCM physics: Q into pcm (negative during hot discharge; gate 'cold' enforces ≤0)
            Q = state.hot_node.step(T_water_in_hot - K_OFFSET, flow_lpm, dt,
                                    "discharge", apply=False)
            Q_hot_into_pcm = Q
            heat_transfer_hot = Q  # OUTPUT: negative = heat OUT of hot pcm
            discharge_water_temp_hot = T_water_in_hot - Q / (system_flow * CP_WATER)
            pcm_water_inlet_hot = T_water_in_hot

        if discharges_cold and not charges_cold and system_flow > 0:
            T_water_in_cold = state.fcu_water_vol.temp if fcu_mode != 0 else room_temp_K
            # v2 PCM physics: Q into pcm (positive during cold discharge: warm water heats cold tank)
            Q = state.cold_node.step(T_water_in_cold - K_OFFSET, flow_lpm, dt,
                                     "discharge", apply=False)
            Q_cold_into_pcm = Q
            heat_transfer_cold = Q  # OUTPUT: positive when cooling (cold delivered)
            discharge_water_temp_cold = T_water_in_cold - Q / (system_flow * CP_WATER)

        # ── Step 2: FCU model ── (unchanged from hp_tes; PCM-agnostic)
        if fcu_mode != 0 and system_flow > 0:
            fan_mass_flow = FCU_FAN_MASS_FLOW
            fan_power = FCU_FAN_POWER
            m_dot_fcu = pump1_flow + pump2_flow

            is_pure_discharge = ((discharges_hot and not charges_hot) or
                                 (discharges_cold and not charges_cold))

            if fcu_mode == 1:  # Heating
                if discharges_hot and not charges_hot:
                    raw_water_temp = discharge_water_temp_hot
                    fcu_entering_water_temp = state.loop_vol.update(raw_water_temp, system_flow, dt) if step_i > 0 else raw_water_temp
                else:
                    fcu_entering_water_temp = state.loop_vol.update(state.hp_supply_temp, system_flow, dt) if step_i > 0 else state.hp_supply_temp

                fcu_heat = UA_HEATING * (fcu_entering_water_temp - room_temp_K)
                fcu_heat = max(fcu_heat, 0.0)

            elif fcu_mode == -1:  # Cooling
                if discharges_cold and not charges_cold:
                    raw_water_temp = discharge_water_temp_cold
                    fcu_entering_water_temp = state.loop_vol.update(raw_water_temp, system_flow, dt) if step_i > 0 else raw_water_temp
                else:
                    fcu_entering_water_temp = state.loop_vol.update(state.hp_supply_temp, system_flow, dt) if step_i > 0 else state.hp_supply_temp

                fcu_heat = UA_COOLING * (fcu_entering_water_temp - room_temp_K)
                fcu_heat = min(fcu_heat, 0.0)

            # FCU leaving water temp (energy balance)
            if m_dot_fcu > 0:
                fcu_leaving_water_temp = fcu_entering_water_temp - fcu_heat / (m_dot_fcu * CP_WATER)
            else:
                fcu_leaving_water_temp = fcu_entering_water_temp

            # Close discharge loop: FCU return water → fcu_water_vol → PCM inlet
            if is_pure_discharge and step_i > 0:
                state.fcu_water_vol.update(fcu_leaving_water_temp, system_flow, dt)

        # ── Step 3: Charging PCM heat transfer (v2 NTU-epsilon) ──
        # Series circuit: in hybrid modes water has been through the FCU first.
        # PLC bypass valve: hot tank charges only when the water reaching it is
        # HOTTER than the tank node; cold tank only when COLDER.
        if charges_hot and hp_on:
            if fcu_mode != 0:
                pcm_water_inlet = fcu_leaving_water_temp
                pcm_water_inlet = state.loop_vol.update(pcm_water_inlet, system_flow, dt) if step_i > 0 else pcm_water_inlet
            else:
                pcm_water_inlet = state.loop_vol.update(state.hp_supply_temp, system_flow, dt) if step_i > 0 else state.hp_supply_temp
            # Bypass valve: no flow through PCM until supply water is hotter than PCM
            if pcm_water_inlet > hot_T0_K and system_flow > 0:
                Q = state.hot_node.step(pcm_water_inlet - K_OFFSET, flow_lpm, dt,
                                        "charge", apply=False)  # + into pcm
                Q_hot_into_pcm = Q
                heat_transfer_hot = Q  # OUTPUT: positive = heat INTO hot pcm
            else:
                Q_hot_into_pcm = 0.0
                heat_transfer_hot = 0.0  # bypass: water too cold, valve closed
            pcm_water_inlet_hot = pcm_water_inlet

        if charges_cold and hp_on:
            if fcu_mode != 0:
                pcm_water_inlet = fcu_leaving_water_temp
                pcm_water_inlet = state.loop_vol.update(pcm_water_inlet, system_flow, dt) if step_i > 0 else pcm_water_inlet
            else:
                pcm_water_inlet = state.loop_vol.update(state.hp_supply_temp, system_flow, dt) if step_i > 0 else state.hp_supply_temp
            # Bypass valve: no flow through cold PCM until supply water is colder than PCM
            if pcm_water_inlet < cold_T0_K and system_flow > 0:
                Q = state.cold_node.step(pcm_water_inlet - K_OFFSET, flow_lpm, dt,
                                         "charge", apply=False)  # - into pcm (cools it)
                Q_cold_into_pcm = Q
                heat_transfer_cold = -Q  # OUTPUT: positive when cooling (cold charged)
            else:
                Q_cold_into_pcm = 0.0
                heat_transfer_cold = 0.0  # bypass: water too warm, valve closed

        # ── HP power computation + supply temp update ── (unchanged)
        if hp_on:
            if charges_hot or charges_cold:
                BLEND_RANGE = 5.0  # K — blend over last 5°C of ramp
                if hp_heating:
                    blend = max(0.0, min(1.0,
                        (state.hp_supply_temp - (target_supply - BLEND_RANGE)) / BLEND_RANGE))
                else:
                    blend = max(0.0, min(1.0,
                        ((target_supply + BLEND_RANGE) - state.hp_supply_temp) / BLEND_RANGE))

                q_downstream = (abs(heat_transfer_hot) if charges_hot else abs(heat_transfer_cold)) + abs(fcu_heat)
                q_downstream = max(q_downstream, 50.0)
                hp_thermal_power = (1.0 - blend) * hp_thermal_power_max + blend * q_downstream

                cop = cop_heating if hp_heating else cop_cooling
                hp_electric_power = hp_thermal_power / cop if cop > 0 else 0.0
                hp_useful_thermal = hp_thermal_power

                if step_i > 0 and dt > 0:
                    if charges_hot:
                        q_net = hp_thermal_power - fcu_heat - heat_transfer_hot
                        C_loop = LOOP_THERMAL_MASS_HOT
                    else:  # charges_cold
                        q_net = -hp_thermal_power - fcu_heat + heat_transfer_cold
                        C_loop = LOOP_THERMAL_MASS_COLD

                    state.hp_supply_temp += q_net * dt / C_loop

                    if hp_heating:
                        state.hp_supply_temp = min(state.hp_supply_temp, target_supply)
                    else:
                        state.hp_supply_temp = max(state.hp_supply_temp, target_supply)

                    state.hp_supply_temp = max(state.hp_supply_temp, HP_MIN_SUPPLY_K)
                    state.hp_supply_temp = min(state.hp_supply_temp, HP_MAX_SUPPLY_K)

                    if hp_heating and system_flow > 0:
                        state.hp_supply_temp = min(state.hp_supply_temp,
                                                   state.hp_return_temp + HP_MAX_LIFT_HEATING)
                    elif not hp_heating and system_flow > 0:
                        state.hp_supply_temp = max(state.hp_supply_temp,
                                                   state.hp_return_temp - HP_MAX_LIFT_COOLING)

            else:
                hp_thermal_power = hp_thermal_power_max
                cop = cop_heating if hp_heating else cop_cooling
                hp_electric_power = hp_thermal_power / cop if cop > 0 else 0.0
                hp_useful_thermal = hp_thermal_power

                if step_i > 0 and dt > 0:
                    dT_max = max(ramp_power, 1000.0) * dt / HP_THERMAL_MASS
                    temp_diff = target_supply - state.hp_supply_temp
                    if abs(temp_diff) > 0:
                        sign = 1.0 if temp_diff > 0 else -1.0
                        state.hp_supply_temp += sign * min(abs(temp_diff), dT_max)

                    if hp_heating and system_flow > 0:
                        state.hp_supply_temp = min(state.hp_supply_temp,
                                                   state.hp_return_temp + HP_MAX_LIFT_HEATING)
                    elif not hp_heating and system_flow > 0:
                        state.hp_supply_temp = max(state.hp_supply_temp,
                                                   state.hp_return_temp - HP_MAX_LIFT_COOLING)
        elif not hp_on:
            if step_i > 0 and dt > 0:
                if system_flow > 0:
                    if fcu_mode != 0:
                        T_water_entering_hp = fcu_leaving_water_temp
                    elif discharges_hot:
                        T_water_entering_hp = discharge_water_temp_hot
                    elif discharges_cold:
                        T_water_entering_hp = discharge_water_temp_cold
                    else:
                        T_water_entering_hp = state.hp_supply_temp

                    tau = HP_THERMAL_MASS / (system_flow * CP_WATER)
                    alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
                    state.hp_supply_temp += alpha * (T_water_entering_hp - state.hp_supply_temp)
                else:
                    HP_STANDBY_UA = 5.0  # W/K — condenser to room air
                    Q_loss = HP_STANDBY_UA * (state.hp_supply_temp - room_temp_K)
                    state.hp_supply_temp -= Q_loss * dt / HP_THERMAL_MASS

        # ── HP return temp (last component in thermal chain) ──
        if charges_hot and system_flow > 0:
            pcm_inlet_for_return = fcu_leaving_water_temp if fcu_mode != 0 else state.hp_supply_temp
            state.hp_return_temp = pcm_inlet_for_return - heat_transfer_hot / (system_flow * CP_WATER)
        elif charges_cold and system_flow > 0:
            pcm_inlet_for_return = fcu_leaving_water_temp if fcu_mode != 0 else state.hp_supply_temp
            state.hp_return_temp = pcm_inlet_for_return + heat_transfer_cold / (system_flow * CP_WATER)
        elif fcu_mode != 0:
            state.hp_return_temp = fcu_leaving_water_temp
        else:
            state.hp_return_temp = state.hp_supply_temp

        # ── Update HP on/off history for next substep ──
        state.hp_on_prev = hp_on

        # ── Record snapshot BEFORE updating PCM node temps ──
        snapshots.append({
            'time': t,
            'schedule_input': float(schedule_input),
            'hp_electric_power': hp_electric_power,
            'hp_mode': 1.0 if hp_heating else -1.0 if hp_on else 0.0,
            'hp_on_off': 1.0 if hp_on else 0.0,
            'hp_useful_thermal': hp_useful_thermal,
            'pump1_power': pump1_power,
            'pump2_power': pump2_power,
            'cold_soc': cold_soc,
            'hot_soc': hot_soc,
            'cold_pcm_temp': state.cold_pcm_temp,
            'hot_pcm_temp': state.hot_pcm_temp,
            'heat_transfer_cold': heat_transfer_cold,
            'heat_transfer_hot': heat_transfer_hot,
            'fcu_heat': fcu_heat,
            'exact_system_mode': mode,
            'fcu_entering_water_temp': fcu_entering_water_temp,
            'fcu_leaving_water_temp': fcu_leaving_water_temp,
            'fan_mass_flow': fan_mass_flow,
            'hp_supply_temp': state.hp_supply_temp,
            'hp_return_temp': state.hp_return_temp,
            'hp_compressor_speed': hp_compressor_speed,
            'cop_heating': cop_heating,
            'cop_cooling': cop_cooling,
            'fcu_mode': float(fcu_mode),
            'pump1_flow': pump1_flow,
            'pump2_flow': pump2_flow,
            'fan_power': fan_power,
            'hot_soc_not_discharged': 1.0 if state.hot_soc_not_discharged else 0.0,
            'cold_soc_not_discharged': 1.0 if state.cold_soc_not_discharged else 0.0,
            'discharge_water_temp_hot': discharge_water_temp_hot,
            'pcm_water_inlet_hot': pcm_water_inlet_hot,
        })

        # ── Update PCM node temperatures: single-node Euler step ──
        # One Euler update per node folds together the v2 water-side coupling
        # (Q_*_into_pcm) and the ambient standby loss, using C_eff evaluated at
        # the start-of-substep temperature (frozen above).  Skipped on the last
        # snapshot so each substep is integrated exactly once across the
        # interval boundary (the next interval's step_i==0 continues it).
        if step_i < n_steps:
            # HOT node
            h_hot = H_FIXED_HOT_ACTIVE if (charges_hot or discharges_hot) else H_FIXED_HOT_STANDBY
            ce_hot = state.hot_node.ceff()  # at start-of-substep temp
            q_loss_hot = h_hot * (state.hot_node.T - room_C)
            state.hot_node.T += dt * (Q_hot_into_pcm - q_loss_hot) / max(ce_hot, 1e4)

            # COLD node
            ce_cold = state.cold_node.ceff()
            q_loss_cold = H_FIXED_COLD * (state.cold_node.T - room_C)
            state.cold_node.T += dt * (Q_cold_into_pcm - q_loss_cold) / max(ce_cold, 1e4)

    state.time = t_end
    return snapshots


# ═══════════════════════════════════════════════════════════════════════════════
# Main Interface
# ═══════════════════════════════════════════════════════════════════════════════

def run_fmu(iteration, input_variables_dict, time_points, model, external_mode=None,
            external_flow=None):
    """
    Run the Python replacement model for the HP-TES system.

    Signature and return format match hp_tes.run_fmu() / HP_TES.run_fmu() exactly.
    When external_mode is set (int), bypass the controller and force that mode.
    When external_flow is set (float), override system_flow regardless of mode.
    """
    schedule_input = int(input_variables_dict['schedule_input'])
    room_temp_K = float(input_variables_dict['room_temperature'])
    outside_air_temp_K = float(input_variables_dict['outside_air_temperature'])

    t_start = time_points[0]
    t_end = time_points[-1]

    # Run simulation with small substeps
    snapshots = _simulate_interval(
        model, schedule_input, room_temp_K, outside_air_temp_K,
        t_start, t_end, substep_dt=5.0, external_mode=external_mode,
        external_flow=external_flow
    )

    # Build raw DataFrame from snapshots
    snap_times = np.array([s['time'] for s in snapshots])

    # Interpolate onto requested time_points
    time_points = np.asarray(time_points, dtype=float)
    fields = [k for k in snapshots[0].keys() if k != 'time']

    interp_data = {'Time': time_points}
    for field in fields:
        raw = np.array([s[field] for s in snapshots])
        interp_data[field] = np.interp(time_points, snap_times, raw)

    # Map to output column names matching hp_tes / HP_TES.py exactly
    data = {
        'Time': interp_data['Time'],
        'FMU mode from MPC': interp_data['schedule_input'],
        'HP Electric Power': interp_data['hp_electric_power'],
        'HP Mode': interp_data['hp_mode'],
        'HP ON OFF': interp_data['hp_on_off'],
        'HP Useful Thermal Power': interp_data['hp_useful_thermal'],
        'Pump 1 Electric Power': interp_data['pump1_power'],
        'Pump 2 Electric Power': interp_data['pump2_power'],
        'PCM_Cold_SOC': interp_data['cold_soc'],
        'PCM_Hot_SOC': interp_data['hot_soc'],
        'PCM_Cold_Temp': interp_data['cold_pcm_temp'],
        'PCM_Hot_Temp': interp_data['hot_pcm_temp'],
        'Heat Transfer Cold PCM': interp_data['heat_transfer_cold'],
        'Heat Transfer Hot PCM': interp_data['heat_transfer_hot'],
        'FCU Heat Delivered': interp_data['fcu_heat'],
        'Exact System Mode': interp_data['exact_system_mode'],
        'FCU Entering Water Temperature': interp_data['fcu_entering_water_temp'],
        'FCU Leaving Water Temperature': interp_data['fcu_leaving_water_temp'],
        'Fan Coil Fan Mass Flow': interp_data['fan_mass_flow'],
        'Heat Pump Supply Water Temperature': interp_data['hp_supply_temp'],
        'Heat Pump Return Water Temperature': interp_data['hp_return_temp'],
        'Heat Pump Compressor Speed': interp_data['hp_compressor_speed'],
        'COP Heating': interp_data['cop_heating'],
        'COP Cooling': interp_data['cop_cooling'],
        'FCU Mode': interp_data['fcu_mode'],
        'Pump 1 Mass Flow': interp_data['pump1_flow'],
        'Pump 2 Mass Flow': interp_data['pump2_flow'],
        'Fan Coil Fan Electric Power': interp_data['fan_power'],
        'hot pcm not fully discharged': interp_data['hot_soc_not_discharged'],
        'cold pcm not fully discharged': interp_data['cold_soc_not_discharged'],
        'Discharge Water Temp Hot': interp_data['discharge_water_temp_hot'],
        'PCM Water Inlet Hot': interp_data['pcm_water_inlet_hot'],
    }

    df = pd.DataFrame(data)

    # Build cleaned_output matching hp_tes / HP_TES.py
    cleaned_output = {
        'Time': df['Time'].iloc[-1],
        'FMU mode from MPC': df['FMU mode from MPC'].iloc[-1],
        'Exact System Mode': df['Exact System Mode'].iloc[-1],
        'FCU Mode': df['FCU Mode'].iloc[-1],
        'HP Mode': df['HP Mode'].iloc[-1],
        'HP ON OFF': df['HP ON OFF'].iloc[-1],
        'hot pcm not fully discharged': df['hot pcm not fully discharged'].iloc[-1],
        'cold pcm not fully discharged': df['cold pcm not fully discharged'].iloc[-1],
        'PCM_Cold_SOC': df['PCM_Cold_SOC'].iloc[-1],
        'PCM_Hot_SOC': df['PCM_Hot_SOC'].iloc[-1],
        'PCM_Cold_Temp': df['PCM_Cold_Temp'].iloc[-1] - K_OFFSET,
        'PCM_Hot_Temp': df['PCM_Hot_Temp'].iloc[-1] - K_OFFSET,
        'FCU Entering Water Temperature': df['FCU Entering Water Temperature'].iloc[-1] - K_OFFSET,
        'FCU Leaving Water Temperature': df['FCU Leaving Water Temperature'].iloc[-1] - K_OFFSET,
        'Heat Pump Supply Water Temperature': df['Heat Pump Supply Water Temperature'].iloc[-1] - K_OFFSET,
        'Heat Pump Return Water Temperature': df['Heat Pump Return Water Temperature'].iloc[-1] - K_OFFSET,
        'Heat Pump Compressor Speed': df['Heat Pump Compressor Speed'].iloc[-1],
        'COP Heating': Data_Conversion.find_average(df, 'COP Heating'),
        'COP Cooling': Data_Conversion.find_average(df, 'COP Cooling'),
        'Fan Coil Fan Mass Flow': Data_Conversion.find_average(df, 'Fan Coil Fan Mass Flow'),
        'Pump 1 Mass Flow': Data_Conversion.find_average(df, 'Pump 1 Mass Flow'),
        'Pump 2 Mass Flow': Data_Conversion.find_average(df, 'Pump 2 Mass Flow'),
        'HP Electric Power': Data_Conversion.find_average(df, 'HP Electric Power') / 1000,
        'HP Useful Thermal Power': Data_Conversion.find_average(df, 'HP Useful Thermal Power') / 1000,
        'Pump 1 Electric Power': Data_Conversion.find_average(df, 'Pump 1 Electric Power') / 1000,
        'Pump 2 Electric Power': Data_Conversion.find_average(df, 'Pump 2 Electric Power') / 1000,
        'Heat Transfer Cold PCM': Data_Conversion.find_average(df, 'Heat Transfer Cold PCM') / 1000,
        'Heat Transfer Hot PCM': Data_Conversion.find_average(df, 'Heat Transfer Hot PCM') / 1000,
        'FCU Heat Delivered': Data_Conversion.find_average(df, 'FCU Heat Delivered') / 1000,
        'Fan Coil Fan Electric Power': Data_Conversion.find_average(df, 'Fan Coil Fan Electric Power') / 1000,
        'Discharge Water Temp Hot': df['Discharge Water Temp Hot'].iloc[-1] - K_OFFSET,
        'PCM Water Inlet Hot': df['PCM Water Inlet Hot'].iloc[-1] - K_OFFSET,
    }

    return df, cleaned_output


# RL back-compat: the RL blackbox calls `hp_tes.system.run_python_hp_tes(...)` (the submodule
# attribute), not just the package-level alias in __init__.py. Expose it on this module too.
run_python_hp_tes = run_fmu
