"""
hp_tes/controller.py - Mode selection and hysteresis control functions.
"""

from .parameters import (
    T_HEAT_SP, T_COOL_SP, T_DELTA,
    SOC_DISCHARGE_HIGH, SOC_CHARGE_LOW, SOC_CHARGE_HIGH,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level functions (backward compat, used by Controller class)
# ═══════════════════════════════════════════════════════════════════════════════

def update_thermostat(state, room_temp_K):
    """Update thermostat hysteresis for heating/cooling calls."""
    # Heating call
    if room_temp_K < (T_HEAT_SP - T_DELTA):
        state.call_for_heating = True
    elif room_temp_K > (T_HEAT_SP + T_DELTA):
        state.call_for_heating = False

    # Cooling call
    if room_temp_K > (T_COOL_SP + T_DELTA):
        state.call_for_cooling = True
    elif room_temp_K < (T_COOL_SP - T_DELTA):
        state.call_for_cooling = False


def update_soc_hysteresis(state, hot_soc, cold_soc):
    """Update SOC hysteresis flags for discharge/charge availability."""
    # Hot PCM discharge hysteresis
    if hot_soc >= SOC_DISCHARGE_HIGH:
        state.hot_soc_not_discharged = True
    elif hot_soc <= 0.0:
        state.hot_soc_not_discharged = False

    # Cold PCM discharge hysteresis
    if cold_soc >= SOC_DISCHARGE_HIGH:
        state.cold_soc_not_discharged = True
    elif cold_soc <= 0.0:
        state.cold_soc_not_discharged = False

    # Hot PCM charge hysteresis
    # SOC_CHARGE_HIGH >> 1.0 so physics (HP supply vs PCM ΔT) limits charging
    if hot_soc <= SOC_CHARGE_LOW:
        state.hot_soc_not_charged = True
    elif hot_soc >= SOC_CHARGE_HIGH:
        state.hot_soc_not_charged = False

    # Cold PCM charge hysteresis
    if cold_soc <= SOC_CHARGE_LOW:
        state.cold_soc_not_charged = True
    elif cold_soc >= SOC_CHARGE_HIGH:
        state.cold_soc_not_charged = False


def determine_mode(schedule_input, call_for_heating, call_for_cooling,
                   hot_can_discharge, cold_can_discharge,
                   hot_can_charge, cold_can_charge):
    """
    Determine operating mode from supervisor schedule, thermostat calls,
    and PCM charge/discharge availability.

    Returns
    -------
    mode : float
    hp_on : bool
    hp_heating : bool
    fcu_mode : int (-1, 0, 1)
    charges_hot : bool
    charges_cold : bool
    discharges_hot : bool
    discharges_cold : bool
    """
    if schedule_input == 0:
        # Standby — with safety override for thermal comfort
        if call_for_heating:
            return 10.0, True, True, 1, False, False, False, False
        elif call_for_cooling:
            return 11.0, True, False, -1, False, False, False, False
        else:
            return 0.0, False, True, 0, False, False, False, False

    elif schedule_input == 1:
        # Charge Hot TES
        if call_for_heating and hot_can_charge and hot_can_discharge:
            return 3.0, True, True, 1, True, False, True, False
        elif call_for_heating:
            return 10.0, True, True, 1, False, False, False, False
        else:
            if hot_can_charge:
                return 1.0, True, True, 0, True, False, False, False
            else:
                return 0.0, False, True, 0, False, False, False, False

    elif schedule_input == 2:
        # Charge Cold TES
        if call_for_cooling and cold_can_charge and cold_can_discharge:
            return 7.0, True, False, -1, False, True, False, True
        elif call_for_cooling:
            return 11.0, True, False, -1, False, False, False, False
        else:
            if cold_can_charge:
                return 5.0, True, False, 0, False, True, False, False
            else:
                return 0.0, False, True, 0, False, False, False, False

    elif schedule_input == 3:
        # Prioritize TES use
        if call_for_heating and hot_can_discharge:
            return 2.0, False, True, 1, False, False, True, False
        elif call_for_heating:
            return 10.0, True, True, 1, False, False, False, False
        elif call_for_cooling and cold_can_discharge:
            return 6.0, False, False, -1, False, False, False, True
        elif call_for_cooling:
            return 11.0, True, False, -1, False, False, False, False
        else:
            return 0.0, False, True, 0, False, False, False, False

    # Fallback
    return 0.0, False, True, 0, False, False, False, False
