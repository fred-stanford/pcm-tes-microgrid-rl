# =============================================================================
# LEGACY MODULE — NOT the controller used in the grid-tied A/B study.
# The shipped rule-based benchmark is physics/rbc_rules.py (wrapped by
# microgrid/agents/rbc.py). The ONLY symbol from this file on the live A/B path is
# `PCM_available_energy`; everything else (rule_based_simulation, rbc_battery_controller,
# urgency scores, the forecast/price-aware logic) is dead code kept for reference.
# Do NOT cite this file as the study's RBC benchmark.
# =============================================================================
import pandas as pd
import numpy as np
import Input_Parameters
from physics import HP_TES

# Need to update parameters (such as HP power ratings in the RBC and just refer to Input_Parameters) 

def rule_based_simulation(input_df, M_states):
    '''
    Added E_Load
    rbc does not that Ti as an input at all
    check time step
    '''
    print("DEBUG: Starting rule_based function")
    
    # Extract values from input DataFrame
    PVGeneration = M_states['PV']  # [kW]
    # print(f"DEBUG: PVGeneration = {PVGeneration} kW")

    # Extract PCM temperatures
    PCM_C_Temp = M_states['xPCMTemperatureCold']
    PCM_H_Temp = M_states['xPCMTemperatureHot']
    # print(f"DEBUG: PCM Temperatures - Cold: {PCM_C_Temp}°C, Hot: {PCM_H_Temp}°C")

    # Calculate PCM available energy
    PCM_C_Energy, PCM_H_Energy = PCM_available_energy(PCM_C_Temp, PCM_H_Temp) # [kWh]
    # print(f"DEBUG: PCM Available Energy - Cold: {PCM_C_Energy} kWh, Hot: {PCM_H_Energy} kWh")

    # Ambient temperature
    Ta = M_states['Ta']
    # print(f"DEBUG: Ambient Temperature = {Ta}°C")
    option = "NULL"

    # Heating and cooling flags
    heating_0 = 0
    cooling_0 = 0
    # print("DEBUG: M_states['Supervisor_CallHot'] = ", M_states['Supervisor_CallHot'])
    # print("DEBUG: M_states['Supervisor_CallCold'] = ", M_states['Supervisor_CallCold'])
    if M_states['Supervisor_CallHot'] == 1:
        heating_0 = 1
    if M_states['Supervisor_CallCold'] == 1:
        cooling_0 = 1
    # print(f"DEBUG: Heating flag = {heating_0}, Cooling flag = {cooling_0}")

    # Calculate electrical load from HP system at last timestep
    # E_Modelica_0 = M_states['E_Modelica_0']  # [kW] average power consumption in the last timestep
    # print(f"DEBUG: Electrical load from HP system = {E_Modelica_0} kW")

    # Extract Julia states
    # PV_Gen_0 = J_states[0]  # [kW] PV Generation during last timestep
    # B_SOC_0 = J_states[1]  # [kWh] Battery SOC at the start of last timestep
    # print(f"DEBUG: Previous PV Generation = {PV_Gen_0} kW, Previous Battery SOC = {B_SOC_0} kWh")

    # Heat pump return water temperature and compressor speed
    T_HP = M_states['xSupplyTemperatureHP']
    compressor_speed = M_states['Heat Pump Compressor Speed']
    # print(f"DEBUG: Heat Pump Supply Temperature = {T_HP}°C, Compressor Speed = {compressor_speed}")

    # Calculate battery SOC, curtailment, and loss of load
    # SOC, curtailment, loss_of_load = Get_battery_SOC(PV_Gen_0, E_Modelica_0, B_SOC_0) 
    # loss of load and curtailment from previous timestep should be better dealt with
    # print(f"DEBUG: Updated Battery SOC = {SOC} kWh, Curtailment = {curtailment} kW, Loss of Load = {loss_of_load} kW")

    # Initialize variables
    SOC_initial = M_states['Battery SOC']  # [kWh]

    Decision = 999
    Scenario = 999
    # print(f"DEBUG: Initial SOC = {SOC_initial} kWh")

    Available_PV = 0  # [kW]
    E_load = M_states['E_Load']  # [kW]
    # print(f"DEBUG: E_load = {E_load} kW")
    
    SOC = SOC_initial # [kWh]

    # Determine the previous TES mode (Do nothing, charge hot, charge cold, or discharge)
    previous_TES_state = 0

    if M_states['OC_Mod'] in {1, 3}:
        previous_TES_state = 1
    elif M_states['OC_Mod'] in {5, 7}:
        previous_TES_state = 2
    elif M_states['OC_Mod'] in {2, 4, 6, 8}:
        previous_TES_state = 3
    
    # print(f"DEBUG: Previous TES State = {previous_TES_state} (0=None, 1=Charge Hot, 2=Charge Cold, 3=Discharge)")
     
    urgency_H, urgency_C = Get_urgency_score(input_df, PCM_C_Energy, PCM_H_Energy)
    
    # Calculate the maximum heat pump heating and cooling capacity, as well as COP, and therefore calculate the heat pump electrical power input.

    # 2. Get the COP and Capacity functions
    cop_cooling = get_COP_cooling(Ta, T_HP) # [-] COP for cooling
    max_cooling_capacity = get_capacity_cooling(Ta, T_HP, compressor_speed) # [kW] max thermal power output for cooling

    cop_heating = get_COP_heating(Ta, T_HP) # [-] COP for heating
    max_heating_capacity = get_capacity_heating(Ta, T_HP, compressor_speed) # [kW] max thermal power output for heating

    cooling_power_input = min(max(max_cooling_capacity / cop_cooling, 0.5), 3) # [kW] max electrical power input for cooling
    heating_power_input = min(max(max_heating_capacity / cop_heating, 0.5), 3) # [kW] max electrical power input for heating

    loss_of_load_E = 0
    loss_of_load_Q = 0

    if PVGeneration == 0: # Night Time
        # print("DEBUG: Night Time Operation (PVGeneration = 0)")
        SOC, loss_of_load = discharge_battery(SOC, E_load) # Discharge battery to meet electrical load first
        if loss_of_load > 0: # Loss of load scenario
            loss_of_load_E = loss_of_load_E + loss_of_load
            if heating_0 == 1:
                option = "H"
                PCM_H_Energy, owed_power = discharge_PCM(option, PCM_H_Energy, Input_Parameters.space_heating_thermal)
                if owed_power > 0:
                    loss_of_load_Q = owed_power
                    Decision = 3
                    Scenario = 4
                else:
                    Decision = 3
                    Scenario = 2
            elif cooling_0 == 1:
                option = "C"
                PCM_C_Energy, owed_power = discharge_PCM(option, PCM_C_Energy, Input_Parameters.space_cooling_thermal)
                if owed_power > 0:
                    loss_of_load_Q = owed_power
                    Decision = 3
                    Scenario = 8
                else:
                    Decision = 3
                    Scenario = 6
            else:
                Decision = 0
                Scenario = 0
        else:
            if heating_0 == 1:
                option = "H"
                favorable = check_favorable_COP(option, input_df, PCM_H_Energy)
                if favorable == True: # heating call ON, HP heating favorable, electricity to thermal conversion favorable, prioritize using electricity
                    SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, heating_power_input, 0, Ta, T_HP, compressor_speed) # discharge battery to use HP
                    if owed_thermal_power > 0: # heating call not met by HP powered by battery discharge
                        PCM_H_Energy, owed_thermal_power = discharge_PCM(option, PCM_H_Energy, owed_thermal_power) # discharge PCM H
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        Decision = 3
                        Scenario = 4
                    else: # heating call met by HP powered by battery discharge, extra heat produced, charge PCM H (hybrid charge mode)
                        PCM_H_Energy, left_over_thermal = charge_PCM_simple(option, PCM_H_Energy, extra_thermal_power)
                        Decision = 1
                        Scenario = 3
                else: # heating call ON, HP heating unfavorable (or not urgent enough, therefore unneeded), prioritize using thermal
                    PCM_H_Energy, owed_thermal_power = discharge_PCM(option, PCM_H_Energy, Input_Parameters.space_heating_thermal) # discharge PCM H
                    if owed_thermal_power > 0: # heating call not met by PCM H discharge alone
                        SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, heating_power_input, 0, Ta, T_HP, compressor_speed) # this is not great, as HP should be operating in standard power
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        Decision = 3
                        Scenario = 4
                    else: # heating call met by discharging PCM H alone
                        Decision = 3
                        Scenario = 2
            elif cooling_0 == 1: # need optimization
                option = "C"
                favorable = check_favorable_COP(option, input_df, PCM_C_Energy)
                if favorable == True: # cooling call ON, HP cooling favorable, electricity to thermal conversion favorable, prioritize using electricity
                    SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, cooling_power_input, 0, Ta, T_HP, compressor_speed) # discharge battery to use HP
                    if owed_thermal_power > 0: # cooling call not met by HP powered by battery discharge
                        PCM_C_Energy, owed_thermal_power = discharge_PCM(option, PCM_C_Energy, owed_thermal_power) # discharge PCM C
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        Decision = 3
                        Scenario = 8
                    else: # cooling call met by HP powered by battery discharge, extra coolth produced, charge PCM C (hybrid charge mode)
                        PCM_C_Energy, left_over_thermal = charge_PCM_simple(option, PCM_C_Energy, extra_thermal_power)
                        Decision = 2
                        Scenario = 7
                else: # cooling call ON, HP cooling unfavorable (or not urgent enough, therefore unneeded), prioritize using thermal
                    PCM_C_Energy, owed_thermal_power = discharge_PCM(option, PCM_C_Energy, Input_Parameters.space_cooling_thermal) # discharge PCM C
                    if owed_thermal_power > 0: # cooling call not met by PCM C discharge alone
                        SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, cooling_power_input, 0, Ta, T_HP, compressor_speed) # this is not great, as HP should be operating in standard power
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        Decision = 3
                        Scenario = 8
                    else: # cooling call met by discharging PCM C alone
                        Decision = 3
                        Scenario = 6
            else: # no heating or cooling Call
                # print("DEBUG: No Heating or Cooling Call")
                # urgency_H, urgency_C = Get_urgency_score(input_df, PCM_C_Energy, PCM_H_Energy)
                if urgency_H >= 0.8: # heating is urgent
                    option = "H"
                    favorable = check_favorable_COP(option, input_df, PCM_H_Energy)
                    if favorable == True: # charge PCM H
                        SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP('N', option, SOC, heating_power_input, 0, Ta, T_HP, compressor_speed)
                        PCM_H_Energy, left_over_thermal = charge_PCM_simple(option, PCM_H_Energy, extra_thermal_power)
                        Decision = 1
                        Scenario = 1
                    else:
                        Decision = 0
                        Scenario = 0
                elif urgency_C >= 0.8: # cooling is urgent
                    option = "C"
                    favorable = check_favorable_COP(option, input_df, PCM_C_Energy)
                    if favorable == True: # charge PCM H
                        SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP('N', option, SOC, cooling_power_input, 0, Ta, T_HP, compressor_speed)
                        PCM_C_Energy, left_over_thermal = charge_PCM_simple(option, PCM_C_Energy, extra_thermal_power)
                        Decision = 2
                        Scenario = 5
                    else:
                        Decision = 0
                        Scenario = 0
                else: # urgent PCM does not exist
                    Decision = 0
                    Scenario = 0
    
    if PVGeneration > 0: # Day Time
        # print("DEBUG: Day Time Operation (PVGeneration > 0)")
        if PVGeneration < E_load:
            E_load_remain = E_load - PVGeneration # need to discharge battery to meet the remaining electrical load
            SOC, loss_of_load = discharge_battery(SOC, E_load_remain) # Discharge battery to meet electrical load first
            if loss_of_load > 0: # Loss of load scenario
                loss_of_load_E = loss_of_load_E + loss_of_load
                if heating_0 == 1:
                    option = "H"
                    PCM_H_Energy, owed_power = discharge_PCM(option, PCM_H_Energy, Input_Parameters.space_heating_thermal)
                    if owed_power > 0:
                        loss_of_load_Q = owed_power
                        Decision = 3
                        Scenario = 4
                    else:
                        Decision = 3
                        Scenario = 2
                elif cooling_0 == 1:
                    option = "C"
                    PCM_C_Energy, owed_power = discharge_PCM(option, PCM_C_Energy, Input_Parameters.space_cooling_thermal)
                    if owed_power > 0:
                        loss_of_load_Q = owed_power
                        Decision = 3
                        Scenario = 8
                    else:
                        Decision = 3
                        Scenario = 6
                else:
                    Decision = 0
                    Scenario = 0
            else: # PV + battery discharge already met electrical load, PV depleted
                if heating_0 == 1:
                    option = "H"
                    SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, heating_power_input, 0, Ta, T_HP, compressor_speed) # discharge battery to use HP
                    if owed_thermal_power > 0: # HP can't meet all the thermal power demand
                        PCM_H_Energy, owed_thermal_power = discharge_PCM(option, PCM_H_Energy, owed_thermal_power) # discharge PCM (controlled flow) to meet remaining thermal power
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                            Decision = 3
                            Scenario = 4
                        else:
                            Decision = 3
                            Scenario = 4
                    else:
                        Decision = 0
                        Scenario = 9
                elif cooling_0 == 1: # need optimization
                    option = "C"
                    SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, cooling_power_input, 0, Ta, T_HP, compressor_speed) # discharge battery to use HP
                    if owed_thermal_power > 0: # HP can't meet all the thermal power demand
                        PCM_C_Energy, owed_thermal_power = discharge_PCM(option, PCM_C_Energy, owed_thermal_power) # discharge PCM (controlled flow) to meet remaining thermal power
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                            Decision = 3
                            Scenario = 8
                        else:
                            Decision = 3
                            Scenario = 8
                    else:
                        Decision = 0
                        Scenario = 10
                else: # no heating or cooling Call
                    Decision = 0
                    Scenario = 0
        else: # PV already met electrical load
            Available_PV = PVGeneration - E_load
            if heating_0 == 1: 
                option = "H"
                if Available_PV >= heating_power_input: # PV can fully meet HP and have left over power, operating heat pump at its maximum capacity
                    if max_heating_capacity > Input_Parameters.space_heating_thermal: # the output thermal power of the heat pump is greater than the space heating thermal power requirement
                        left_over_power = Available_PV - heating_power_input # [kW] remaining PV power that can be used to charge the battery
                        left_over_thermal = max_heating_capacity - Input_Parameters.space_heating_thermal # [kW] remaining thermal power
                        PCM_H_Energy, left_over_thermal = charge_PCM_simple(option, PCM_H_Energy, left_over_thermal)
                        SOC, curtailed_power = charge_battery(SOC, left_over_power) # charge battery
                        # curtailment = curtailed_power
                        Decision = 1
                        Scenario = 3
                    else: # the output thermal power of the heat pump is less than the space heating thermal power requirement, use the heat pump and PCM hot discharge to meet the heating call
                        owed_thermal_power = Input_Parameters.space_heating_thermal - max_heating_capacity # [kW] thermal power needed to be discharged from the PCM
                        PCM_H_Energy, owed_thermal_power = discharge_PCM(option, PCM_H_Energy, owed_thermal_power)
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        else:
                            loss_of_load_Q = 0
                        Decision = 3
                        Scenario = 4
                else: # PV can't fully run heat pump at its maximum capacity, need battery discharge to operate heat pump at its maximum capacity
                    SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, heating_power_input, Available_PV, Ta, T_HP, compressor_speed) # discharge battery to use HP
                    # extra_thermal_power should be 0, or curtailed
                    if owed_thermal_power > 0: # heat pump unable to meet heating demand
                        PCM_H_Energy, owed_thermal_power = discharge_PCM(option, PCM_H_Energy, owed_thermal_power) # discharge PCM (controlled flow) to meet heating call
                        Decision = 3
                        Scenario = 4
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        else:
                            loss_of_load_Q = 0
                    else: # baseline heating
                        Decision = 0
                        Scenario = 9
            elif cooling_0 == 1: 
                option = "C"
                if Available_PV >= cooling_power_input: # PV can fully meet HP and have left over power, operating heat pump at its maximum capacity
                    if max_cooling_capacity > Input_Parameters.space_cooling_thermal: # the output thermal power of the heat pump is greater than the space cooling thermal power requirement
                        left_over_power = Available_PV - cooling_power_input # [kW] remaining PV power that can be used to charge the battery
                        left_over_thermal = max_cooling_capacity - Input_Parameters.space_cooling_thermal # [kW] remaining thermal power
                        PCM_C_Energy, left_over_thermal = charge_PCM_simple(option, PCM_C_Energy, left_over_thermal)
                        SOC, curtailed_power = charge_battery(SOC, left_over_power) # charge battery
                        # curtailment = curtailed_power
                        Decision = 2
                        Scenario = 7
                    else: # the output thermal power of the heat pump is less than the space cooling thermal power requirement, use the heat pump and PCM cold discharge to meet the cooling call
                        owed_thermal_power = Input_Parameters.space_cooling_thermal - max_cooling_capacity # [kW] thermal power needed to be discharged from the PCM
                        PCM_C_Energy, owed_thermal_power = discharge_PCM(option, PCM_C_Energy, owed_thermal_power)
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        else:
                            loss_of_load_Q = 0
                        Decision = 3
                        Scenario = 8
                else: # PV can't fully run heat pump at its maximum capacity, need battery discharge to operate heat pump at its maximum capacity
                    SOC, extra_thermal_power, owed_thermal_power = discharge_battery_to_HP(option, option, SOC, cooling_power_input, Available_PV, Ta, T_HP, compressor_speed) # discharge battery to use HP
                    # extra_thermal_power should be 0, or curtailed
                    if owed_thermal_power > 0: # heat pump unable to meet heating/cooling demand
                        PCM_C_Energy, owed_thermal_power = discharge_PCM(option, PCM_C_Energy, owed_thermal_power) # discharge PCM (controlled flow) to meet cooling call
                        Decision = 3
                        Scenario = 8
                        if owed_thermal_power > 0:
                            loss_of_load_Q = owed_thermal_power
                        else:
                            loss_of_load_Q = 0
                    else: # baseline cooling
                        Decision = 0
                        Scenario = 10
            else: # no heating or cooling call
                SOC, extra_power = charge_battery(SOC, Available_PV) # charge battery
                if extra_power > 0: # PV still available
                    # urgency_H, urgency_C = Get_urgency_score(input_df, PCM_C_Energy, PCM_H_Energy)
                    efficiency_H, efficiency_C = Get_efficiency_score(Ta, T_HP, compressor_speed)
                    if urgency_H >= 0.8: # Heating PCM urgent
                        option = "H"
                        PCM_H_Energy, left_over_power, left_over_thermal = charge_PCM(option, PCM_H_Energy, extra_power, heating_power_input, Ta, T_HP, compressor_speed)
                        # curtailment = left_over_power # [kW] curtailment
                        Decision = 1
                        Scenario = 1
                    elif urgency_C >= 0.8: # Cooling PCM urgent
                        option = "C"
                        PCM_C_Energy, left_over_power, left_over_thermal = charge_PCM(option, PCM_C_Energy, extra_power, cooling_power_input, Ta, T_HP, compressor_speed)
                        # curtailment = left_over_power # [kW] curtailment
                        Decision = 2
                        Scenario = 5
                    else:
                        if previous_TES_state == 1: # previously charging hot PCM
                            option = "H"
                            PCM_H_Energy, left_over_power, left_over_thermal = charge_PCM(option, PCM_H_Energy, extra_power, heating_power_input, Ta, T_HP, compressor_speed)
                            # curtailment = left_over_power # [kW] curtailment
                            Decision = 1
                            Scenario = 1
                        elif previous_TES_state == 2: # previously charging cold PCM
                            option = "C"
                            PCM_C_Energy, left_over_power, left_over_thermal = charge_PCM(option, PCM_C_Energy, extra_power, cooling_power_input, Ta, T_HP, compressor_speed)
                            # curtailment = left_over_power # [kW] curtailment
                            Decision = 2
                            Scenario = 5
                        else: # previously not charging PCM
                            if efficiency_H >= efficiency_C: # Heating PCM more efficient
                                option = "H"
                                PCM_H_Energy, left_over_power, left_over_thermal = charge_PCM(option, PCM_H_Energy, extra_power, heating_power_input, Ta, T_HP, compressor_speed)
                                # curtailment = left_over_power # [kW] curtailment
                                Decision = 1
                                Scenario = 1
                            else: # Cooling PCM more efficient
                                option = "C"
                                PCM_C_Energy, left_over_power, left_over_thermal = charge_PCM(option, PCM_C_Energy, extra_power, cooling_power_input, Ta, T_HP, compressor_speed)
                                # curtailment = left_over_power # [kW] curtailment
                                Decision = 2
                                Scenario = 5
                else: # only charge battery, no extra PV available
                    Decision = 0
                    Scenario = 0

    # The J_states_new consist of PV generation of the current iteration (data from current iteration of input_df) and the initial battery SOC at the start of the current iteration.
    J_states_new = [PVGeneration, SOC_initial]
    
    print(f"DEBUG: Final Decision = {Decision}, Scenario = {Scenario}")
    # print(f"DEBUG: Returning J_states_new = [{PVGeneration}, {SOC_initial}], Decision = {Decision}")
    urgency_scores = [urgency_H, urgency_C]
    return J_states_new, Decision, urgency_scores

def discharge_battery_to_HP(option1, option2, soc, required_power, available_power, Ta, T_HP, compressor_speed):
    '''
    Discharge battery to power the heat pump for heating/cooling.
    If there is a current heating/cooling call, then the required power is the heating/cooling power input.
    If there is no current heating/cooling call, then the thermal power demand is 0.

    The total available power for the heat pump is the sum of the current available PV power and the maximum dischargable power from the battery.

    Inputs:
    option1: "H" for heating, "C" for cooling, and "N" for no call ---> this represents the current heating/cooling call (FCU Mode)
    option2: "H" for heating or "C" for cooling ---> this represents the intended heating/cooling mode of the heat pump (HP Mode)
    soc: battery state of charge [kWh]
    required_power: required electrical power of the heat pump [kW] (maximum input power)
    available_power: current available PV power [kW]
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]

    Outputs:
    updated_soc: updated battery state of charge [kWh]
    extra_thermal_power: extra thermal power that can be used to charge the PCM [kW]
    owed_thermal_power: owed thermal power that needs to be discharged from the PCM [kW]
    '''
    # print(f"DEBUG: discharge_battery_to_HP called with option1={option1}, option2={option2}, soc={soc}, required_power={required_power}, available_power={available_power}")

    thermal_power = 0

    # Calculate power needed from the battery
    needed_discharge = required_power - available_power  # [kW] power needed from the battery
    dischargable_power = min(needed_discharge / Input_Parameters.η_B2H, soc / Input_Parameters.δt_in_hr, Input_Parameters.InverterSize)  # [kW]

    # Determine thermal power based on the option
    if option1 == "H":
        thermal_power = Input_Parameters.space_heating_thermal  # [kW] amount of heat needed from the heating call
    elif option1 == "C":
        thermal_power = Input_Parameters.space_cooling_thermal  # [kW] amount of coolth needed from the cooling call
    elif option1 == "N": 
        thermal_power = 0

    # Update battery SOC
    updated_soc = soc - dischargable_power * Input_Parameters.δt_in_hr  # [kWh] updated battery SOC
    # print(f"DEBUG: discharge_battery_to_HP - dischargable_power={dischargable_power}, updated_soc={updated_soc}")

    # Calculate discharged power (accounting for efficiency)
    discharged_power = dischargable_power * Input_Parameters.η_B2H

    # Total power available for the heat pump
    total_power_for_hp = available_power + discharged_power

    # Use the heat pump to generate thermal power
    output_thermal, left_over_power = use_HP(total_power_for_hp, required_power, option2, Ta, T_HP, compressor_speed)  # left_over_power should always be 0

    # Calculate extra or owed thermal power
    extra_thermal_power = 0
    owed_thermal_power = 0

    if output_thermal >= thermal_power:
        extra_thermal_power = output_thermal - thermal_power  # [kW] this can be used to charge the PCM
    else:
        owed_thermal_power = thermal_power - output_thermal  # [kW] this needs to be discharged from the PCM
    
    # print(f"DEBUG: discharge_battery_to_HP - output_thermal={output_thermal}, thermal_power={thermal_power}")
    # print(f"DEBUG: discharge_battery_to_HP - extra_thermal_power={extra_thermal_power}, owed_thermal_power={owed_thermal_power}")

    return updated_soc, extra_thermal_power, owed_thermal_power

def charge_battery(SOC, available_power):
    '''
    Charge battery with available power.

    Inputs:
    SOC: battery state of charge [kWh]
    available_power: available electrical power [kW]

    Outputs:
    updated_soc: updated battery state of charge [kWh]
    left_over_power: the left overpower that couldn't be used and still is available for future use [kW]
    '''
    # print(f"DEBUG: charge_battery called with SOC={SOC}, available_power={available_power}")
    chargable_power = min(available_power, (Input_Parameters.BatterySize - SOC) / Input_Parameters.δt_in_hr, Input_Parameters.InverterSize)  # [kW]
    updated_SOC = SOC + Input_Parameters.η_PV2B * chargable_power * Input_Parameters.δt_in_hr  # [kWh]
    left_over_power = available_power - chargable_power  # [kW]

    # print(f"DEBUG: charge_battery - updated_SOC={updated_SOC}, left_over_power={left_over_power}")
    return updated_SOC, left_over_power


def discharge_battery(SOC, required_power):
    '''
    Discharge battery to meet electrical load.

    Inputs:
    SOC: battery state of charge [kWh]
    required_power: required electrical power [kW]

    Outputs:
    updated_soc: updated battery state of charge [kWh]
    loss_of_load: the loss of load that couldn't be served due to insufficient battery capacity [kW]
    '''
    # print(f"DEBUG: discharge_battery called with SOC={SOC}, required_power={required_power}")
    dischargable_power = min(required_power / Input_Parameters.η_B2H, SOC / Input_Parameters.δt_in_hr, Input_Parameters.InverterSize)  # [kW]
    
    updated_SOC = SOC - dischargable_power * Input_Parameters.δt_in_hr  # [kWh]
    
    loss_of_load = required_power - dischargable_power * Input_Parameters.η_B2H  # [kW]

    discharged_power = dischargable_power * Input_Parameters.η_B2H
    # print(f"DEBUG: discharge_battery - updated_SOC={updated_SOC}, loss_of_load={loss_of_load}, discharged_power={discharged_power}")
    return updated_SOC, loss_of_load


def charge_PCM(option, SOC, available_power, max_input_power, Ta, T_HP, compressor_speed):
    '''
    Charge PCM with available power.

    Inputs:
    option: "H" for heating or "C" for cooling
    SOC: PCM state of charge [kWh]
    available_power: available electrical power [kW]
    max_input_power: maximum electrical power input to the heat pump [kW]
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]

    Outputs:
    updated_SOC: updated PCM state of charge [kWh]
    left_over_power: the left over electrical power that couldn't be used and still is available for future use [kW]
    left_over_thermal: the left over thermal power that couldn't be used and still is available for future use [kW]
    '''
    # use electricity to charge PCM
    # SOC: PCM state of charge
    # available_power: available electrical power for Heat Pump

    # when in Heating/Cooling Charge TES mode, max electrical power available is sent to the Heat Pump
    # the thermal output of the Heat Pump is used to charge the PCM storage.
    
    output_thermal, left_over_power = use_HP(available_power, max_input_power, option, Ta, T_HP, compressor_speed)  
    # send all available PV to HP, return thermal output and left over electrical power

    if option == "H":
        chargable_power = min(output_thermal, (Input_Parameters.PCM_H_Size - SOC) / Input_Parameters.δt_in_hr, Input_Parameters.PCM_H_Power_Capacity)  # [kW]
        updated_SOC = SOC + chargable_power * Input_Parameters.δt_in_hr  # [kWh]
        left_over_thermal = output_thermal - chargable_power  # [kW] 
        # the left_over_thermal technically stays in the system, but for now it is curtailed
    elif option == "C":
        chargable_power = min(output_thermal, (Input_Parameters.PCM_C_Size - SOC) / Input_Parameters.δt_in_hr, Input_Parameters.PCM_C_Power_Capacity)  # [kW]
        updated_SOC = SOC + chargable_power * Input_Parameters.δt_in_hr  # [kWh]
        left_over_thermal = output_thermal - chargable_power  # [kW] 
        # the left_over_thermal technically stays in the system, but for now it is curtailed
    else:
        return 0, 0, 0

    return updated_SOC, left_over_power, left_over_thermal

def charge_PCM_simple(option, SOC, available_thermal):
    '''
    Charge PCM with available thermal power.
    This function is different from the charge_PCM function because it does not use the available electrical power to charge the PCM, but rather the available thermal power.

    Inputs:
    option: "H" for heating or "C" for cooling
    SOC: PCM state of charge [kWh]
    available_thermal: available thermal power [kW]

    Outputs:
    updated_SOC: updated PCM state of charge [kWh]
    left_over_thermal: the left over thermal power that couldn't be used and still is available for future use [kW]
    '''
    # use heat/coolth to charge PCM
    # available_thermal: available thermal power to be used to charge PCM
    if option == "H":
        chargable_power = min(available_thermal, (Input_Parameters.PCM_H_Size - SOC) / Input_Parameters.δt_in_hr, Input_Parameters.PCM_H_Power_Capacity)  # [kW]
        updated_SOC = SOC + chargable_power * Input_Parameters.δt_in_hr  # [kWh]
        left_over_thermal = available_thermal - chargable_power  # [kW] 
        # the left_over_thermal technically stays in the system, but for now it is curtailed
    elif option == "C":
        chargable_power = min(available_thermal, (Input_Parameters.PCM_C_Size - SOC) / Input_Parameters.δt_in_hr, Input_Parameters.PCM_C_Power_Capacity)  # [kW]
        updated_SOC = SOC + chargable_power * Input_Parameters.δt_in_hr  # [kWh]
        left_over_thermal = available_thermal - chargable_power  # [kW] 
        # the left_over_thermal technically stays in the system, but for now it is curtailed
    else:
        return 0, 0

    return updated_SOC, left_over_thermal


def hybrid_charge_PCM(option, SOC, available_power, max_input_power, Ta, T_HP, compressor_speed):
    '''
    Hybrid operation: Uses heat pump to meet immediate demand and charges PCM with excess thermal power
    
    Inputs:
    option: "H" for heating or "C" for cooling
    SOC: current PCM state of charge [kWh]
    available_power: available electrical power [kW]
    max_input_power: maximum electrical power input to the heat pump [kW]
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]
    
    Outputs:
    updated_SOC: new PCM state of charge after charging [kWh]
    left_over_electrical: remaining electrical power not used by heat pump [kW]
    left_over_thermal: excess thermal power that couldn't be stored in PCM [kW]
    '''
    # Input validation
    if option not in ["H", "C"]:
        raise ValueError("Option must be 'H' for heating or 'C' for cooling")
    
    if SOC < 0:
        raise ValueError("SOC cannot be negative")
    
    if available_power < 0:
        raise ValueError("Available power cannot be negative")
    
    # Step 1: Convert electrical power to thermal power using heat pump
    output_thermal, left_over_electrical = use_HP(available_power, max_input_power, option, Ta, T_HP, compressor_speed)
    
    # Step 2: Calculate remaining thermal power after meeting immediate demand
    if option == "H":
        demand_power = Input_Parameters.space_heating_thermal  # [kW] amount of heat needed from the heating call
        remaining_thermal_power = output_thermal - demand_power  # [kW]
        
        # Only proceed if we have excess thermal power for charging
        if remaining_thermal_power > 0:
            # Calculate maximum chargeable power considering capacity and power limits
            available_capacity = Input_Parameters.PCM_H_Size - SOC  # [kWh]
            max_charge_from_capacity = available_capacity / Input_Parameters.δt_in_hr  # [kW]
            
            chargable_power = min(remaining_thermal_power, 
                                max_charge_from_capacity, 
                                Input_Parameters.PCM_H_Power_Capacity)  # [kW]
            
            updated_SOC = SOC + chargable_power * Input_Parameters.δt_in_hr  # [kWh]
            left_over_thermal = remaining_thermal_power - chargable_power  # [kW]
        else:
            # No excess thermal power for charging
            updated_SOC = SOC
            left_over_thermal = 0
            
    elif option == "C":
        demand_power = Input_Parameters.space_cooling_thermal  # [kW] amount of coolth needed from the cooling call
        remaining_thermal_power = output_thermal - demand_power  # [kW] cooling power (positive)
        
        # Check if we have excess cooling capacity for charging
        if remaining_thermal_power > 0:  # More cooling than needed
            # Calculate maximum chargeable cooling power
            available_capacity = Input_Parameters.PCM_C_Size - SOC  # [kWh]
            max_charge_from_capacity = available_capacity / Input_Parameters.δt_in_hr  # [kW]
            
            chargable_power = min(remaining_thermal_power, 
                                max_charge_from_capacity, 
                                Input_Parameters.PCM_C_Power_Capacity)  # [kW]
            
            updated_SOC = SOC + chargable_power * Input_Parameters.δt_in_hr  # [kWh]
            left_over_thermal = remaining_thermal_power - chargable_power  # [kW] (positive)
        else:
            # No excess cooling power for charging
            updated_SOC = SOC
            left_over_thermal = 0
    
    # Ensure SOC doesn't exceed maximum capacity due to numerical precision
    max_SOC = Input_Parameters.PCM_H_Size if option == "H" else Input_Parameters.PCM_C_Size
    updated_SOC = min(updated_SOC, max_SOC)
    
    return updated_SOC, left_over_electrical, left_over_thermal


def discharge_PCM(option, SOC, required_power):
    '''
    Discharges PCM thermal storage to meet heating or cooling demand
    
    Inputs:
    option: "H" for heating or "C" for cooling
    SOC: current PCM state of charge [kWh] thermal energy
    required_power: required thermal power output from the PCM storage [kW]
    
    Outputs:
    updated_SOC: new PCM state of charge after discharge [kWh]
    owed_power: remaining thermal power that is not met by the PCM storage discharge [kW]
    '''
    # Input validation
    if option not in ["H", "C"]:
        raise ValueError("Option must be 'H' for heating or 'C' for cooling")
    
    if SOC < 0:
        raise ValueError("SOC cannot be negative")
    
    # Calculate maximum discharge power based on SOC and time step
    max_discharge_from_SOC = SOC / Input_Parameters.δt_in_hr  # [kW]
    
    if option == "H":
        # For heating: limit by SOC, power capacity, and required power
        max_possible_discharge = min(required_power, 
                                   max_discharge_from_SOC, 
                                   Input_Parameters.PCM_H_Power_Capacity)
        
        actual_discharge_power = max(0, max_possible_discharge)  # Ensure non-negative
        updated_SOC = SOC - actual_discharge_power * Input_Parameters.δt_in_hr
        owed_power = required_power - actual_discharge_power
        
    elif option == "C":
        # For cooling: limit by SOC, power capacity, and required power
        max_possible_discharge = min(required_power, 
                                   max_discharge_from_SOC, 
                                   Input_Parameters.PCM_C_Power_Capacity)
        
        actual_discharge_power = max(0, max_possible_discharge)  # Ensure non-negative
        updated_SOC = SOC - actual_discharge_power * Input_Parameters.δt_in_hr
        owed_power = required_power - actual_discharge_power

    # Ensure SOC doesn't go negative due to numerical precision
    updated_SOC = max(0, updated_SOC)
    
    return updated_SOC, owed_power


def use_HP(input_power, max_input_power, option, Ta, T_HP, compressor_speed):
    '''
    Operates heat pump to convert electrical power to thermal power
    
    Inputs:
    input_power: incoming available electrical power to the heat pump [kW]
    max_input_power: maximum electrical power input to the heat pump [kW]
    option: "H" or "C"
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed from FMU [0-1]
    
    Outputs:
    output_thermal: outcoming thermal power [kW]
    left_over_power: remaining electrical power that wasn't used [kW]
    '''
    # Input validation
    if input_power < 0:
        raise ValueError("Input power cannot be negative")
    
    if compressor_speed < 0 or compressor_speed > 1:
        raise ValueError("Compressor speed must be between 0 and 1")
    
    if option not in ["H", "C"]:
        raise ValueError("Option must be 'H' or 'C'")
    
    # If no input power, return zeros
    if input_power <= 0:
        return 0.0, input_power

    left_over_power = 0
    if input_power > max_input_power:
        left_over_power = input_power - max_input_power # [kW] remaining electrical power that wasn't used
        if option == "H":
            output_thermal = get_capacity_heating(Ta, T_HP, compressor_speed) # [kW] heat gain (positive)
        elif option == "C":
            output_thermal = get_capacity_cooling(Ta, T_HP, compressor_speed) # [kW] heat loss
    else:
        if option == "H":
            output_thermal = input_power * get_COP_heating(Ta, T_HP) # [kW] heat gain (positive)
        elif option == "C":
            output_thermal = input_power * get_COP_cooling(Ta, T_HP) # [kW] heat loss

    return output_thermal, left_over_power
        


def Get_urgency_score(input_df, PCM_C_Energy, PCM_H_Energy):
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
    NumTime = len(input_df)

    PCM_C_Energy = max(0.1, PCM_C_Energy)
    PCM_H_Energy = max(0.1, PCM_H_Energy)
    
    Heating_Demand = sum(input_df["Heating_Load"].iloc[i] * (1 - (i / NumTime)**2) for i in range(NumTime))
    Cooling_Demand = sum(input_df["Cooling_Load"].iloc[i] * (1 - (i / NumTime)**2) for i in range(NumTime))

    urgency_H = (Heating_Demand / PCM_H_Energy) / ((Heating_Demand / PCM_H_Energy) + (Cooling_Demand / PCM_C_Energy))
    urgency_C = (Cooling_Demand / PCM_C_Energy) / ((Heating_Demand / PCM_H_Energy) + (Cooling_Demand / PCM_C_Energy))

    return urgency_H, urgency_C

def Get_efficiency_score(Ta, T_HP, compressor_speed):
    '''
    Calculate the efficiency score for heating and cooling based on the ambient temperature and the heat pump supply water temperature.
    The efficiency score is calculated as the inverse of the absolute difference between the heat pump supply water temperature and the optimal charging temperature.

    Inputs:
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]

    Outputs:
    efficiency_score_H: efficiency score for heating
    efficiency_score_C: efficiency score for cooling 
    '''
    # The efficiency score should only be the efficiency of the charging mode, not the hybrid charging mode.
    # Since hybrid charging mode is always decided by the heating/cooling call itself.

    # COP of HP not considered, but should be considered later.
    HP_SP_C_charge = Input_Parameters.HP_SP_C_charge
    HP_SP_H_charge = Input_Parameters.HP_SP_H_charge
    
    efficiency_score_H = 1 / (abs(T_HP - HP_SP_H_charge) + 0.1)
    efficiency_score_C = 1 / (abs(T_HP - HP_SP_C_charge) + 0.1)

    return efficiency_score_H, efficiency_score_C

def get_COP_heating(Ta, T_HP):
    '''
    Calculate the COP for heating based on the ambient temperature, the heat pump supply water temperature, and the compressor speed.

    Inputs:
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]

    Outputs:
    COP_H: COP for heating
    '''
    a0 = 7.202003525081306  
    a1 = 0.18062703077853665  
    a2 = -0.1477341556516753  
    a3 = -0.002381072467636457  
    a4 = 0.0008319667199337608  
    a5 = 0.0008384541446034042 
    
    COP_H=(a0 + a1*Ta + a2*T_HP + a3*Ta*T_HP + a4*Ta**2 + a5*T_HP**2)

    return COP_H


def get_COP_cooling(Ta, T_HP):
    '''
    Calculate the COP for cooling based on the ambient temperature, the heat pump supply water temperature, and the compressor speed.
    
    Inputs:
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]

    Outputs:
    COP_C: COP for cooling
    '''
    a0 = 8.650914785834006  
    a1 = -0.20763083699950602  
    a2 = -0.06930220567653733  
    a3 = -0.0003181274443609665  
    a4 = 0.001455586245110767  
    a5 = -0.000373410550444541  
    
    COP_C=(a0 + a1*Ta + a2*T_HP + a3*Ta*T_HP + a4*Ta**2 + a5*T_HP**2)

    return COP_C

def Get_battery_SOC(PVGen, Total_Load, SOC):
    '''
    Calculate the battery state of charge based on the PV generation and the total load.
    
    Inputs:
    PVGen: PV generation [kW]
    Total_Load: total load [kW]
    SOC: battery state of charge [kWh]

    Outputs:
    updated_SOC: updated battery state of charge [kWh]
    curtailment: curtailment (excess power that couldn't be used) [kW]
    lossofload_e: loss of load (power deficit that couldn't be served) [kW]
    '''
    # print(f"DEBUG: Get_battery_SOC called with PVGen={PVGen}, Total_Load={Total_Load}, SOC={SOC}")
    curtailment = 0
    lossofload_e = 0

    SOC = SOC * (1 - Input_Parameters.δt_in_hr * Input_Parameters.BatteryLoss)

    if PVGen > Total_Load:
        SOC, curtailment = charge_battery(SOC, PVGen - Total_Load)
    else:
        SOC, lossofload_e = discharge_battery(SOC, Total_Load - PVGen)

    # print(f"DEBUG: Get_battery_SOC - updated SOC={SOC}, curtailment={curtailment}, lossofload_e={lossofload_e}")
    return SOC, curtailment, lossofload_e



def PCM_available_energy(PCM_C_temp, PCM_H_temp):
    '''
    Calculate the available energy in the PCM storage based on the temperature.

    Inputs:
    PCM_C_temp: PCM cold temperature [°C]
    PCM_H_temp: PCM hot temperature [°C]

    Outputs:
    PCM_C_Energy: available energy in the PCM cold storage [kWh]
    PCM_H_Energy: available energy in the PCM hot storage [kWh]
    '''
    # print(f"DEBUG: PCM_available_energy called with PCM_C_temp={PCM_C_temp}, PCM_H_temp={PCM_H_temp}")
    # 10 degrees each way outside of the latent range
    
    PCM_C_Energy = 0
    PCM_H_Energy = 0
    '''
    if PCM_C_temp > 12:
        PCM_C_Energy = (22 - PCM_C_temp) * 2050 * 271.14 / (3.6 * 10**6)

    if 10 <= PCM_C_temp <= 12:
        PCM_C_Energy = ((12 - PCM_C_temp) * (126000 / 2) + 10 * 2050) * 271.14 / (3.6 * 10**6)

    if PCM_C_temp < 10:
        PCM_C_Energy = ((10 - max(0, PCM_C_temp)) * 2050 + 126000 + 10 * 2050) * 271.14 / (3.6 * 10**6)
    '''
    if PCM_C_temp > Input_Parameters.PCM_C_upper_sensible:
        PCM_C_Energy = 0
    if Input_Parameters.PCM_C_upper_latent <= PCM_C_temp <= Input_Parameters.PCM_C_upper_sensible:
        PCM_C_Energy = (Input_Parameters.PCM_C_upper_sensible - PCM_C_temp) * Input_Parameters.PCM_C_sensible_density
    if Input_Parameters.PCM_C_lower_latent <= PCM_C_temp <= Input_Parameters.PCM_C_upper_latent:
        PCM_C_Energy = (Input_Parameters.PCM_C_upper_latent - PCM_C_temp) * Input_Parameters.PCM_C_latent_density + Input_Parameters.PCM_C_Energy_Split[0]
    if Input_Parameters.PCM_C_lower_sensible <= PCM_C_temp <= Input_Parameters.PCM_C_lower_latent:
        PCM_C_Energy = (Input_Parameters.PCM_C_lower_latent - PCM_C_temp) * Input_Parameters.PCM_C_sensible_density + Input_Parameters.PCM_C_Energy_Split[0] + Input_Parameters.PCM_C_Energy_Split[1]
    if PCM_C_temp < Input_Parameters.PCM_C_lower_sensible:
        PCM_C_Energy = Input_Parameters.PCM_C_Energy_Split[0] + Input_Parameters.PCM_C_Energy_Split[1] + Input_Parameters.PCM_C_Energy_Split[2]
        
    if PCM_H_temp < Input_Parameters.PCM_H_lower_sensible:
        PCM_H_Energy = 0     
    if Input_Parameters.PCM_H_lower_sensible <= PCM_H_temp <= Input_Parameters.PCM_H_lower_latent:
        PCM_H_Energy = (PCM_H_temp - Input_Parameters.PCM_H_lower_sensible) * Input_Parameters.PCM_H_sensible_density 
    if Input_Parameters.PCM_H_lower_latent <= PCM_H_temp <= Input_Parameters.PCM_H_upper_latent:
        PCM_H_Energy = (PCM_H_temp - Input_Parameters.PCM_H_lower_latent) * Input_Parameters.PCM_H_latent_density + Input_Parameters.PCM_H_Energy_Split[0]
    if Input_Parameters.PCM_H_upper_latent <= PCM_H_temp <= Input_Parameters.PCM_H_upper_sensible:
        PCM_H_Energy = (PCM_H_temp - Input_Parameters.PCM_H_upper_latent) * Input_Parameters.PCM_H_sensible_density + Input_Parameters.PCM_H_Energy_Split[0] + Input_Parameters.PCM_H_Energy_Split[1]
    if PCM_H_temp > Input_Parameters.PCM_H_upper_sensible:
        PCM_H_Energy = Input_Parameters.PCM_H_Energy_Split[0] + Input_Parameters.PCM_H_Energy_Split[1] + Input_Parameters.PCM_H_Energy_Split[2]
    
    # print(f"DEBUG: PCM_available_energy - PCM_C_Energy={PCM_C_Energy}, PCM_H_Energy={PCM_H_Energy}")
    return PCM_C_Energy, PCM_H_Energy
    

def check_favorable_COP(option, input_df, PCM_SOC):
    '''
    Check if the heat pump operation is favorable based on the future loads and the PCM state of charge.

    Inputs:
    option: "H" for heating or "C" for cooling
    input_df: input DataFrame
    PCM_SOC: PCM state of charge [kWh]

    Outputs:
    favorable: True if the heat pump operation is favorable, False otherwise
    '''
    # first find the horizon until the solar is available again.
    # Find the first index where PV > 0
    first_positive_index = input_df["PV"].gt(0).idxmax() if any(input_df["PV"] > 0) else None

    # Split the DataFrame
    if first_positive_index is not None:
        df_night = input_df.iloc[:first_positive_index]  # Before first positive value
        df_after = input_df.iloc[first_positive_index:]  # From first positive value onwards
    else:
        df_night = input_df  # If no positive values, all goes in df_night
        df_after = pd.DataFrame()  # Empty DataFrame

    n = len(df_night)
    if n == 1:
        return False

    current_Ta = df_night.iloc[0]["Ta"]

    if option == "H":
        # Check if the current Ta is the highest in df_night until another higher Ta appears
        index = df_night["Ta"].gt(current_Ta).idxmax() if any(df_night["Ta"] > current_Ta) else None

        if index is None:
            index = n

        if index == 1:
            return False

        next_horizon = df_night.iloc[:index]  # Create the new horizon

        future_heating_load = next_horizon["Heating_Load"].sum() * Input_Parameters.δt_in_hr
        if PCM_SOC <= future_heating_load * 1.2:
            return True
        else:
            return False

    elif option == "C":
        # Check if the current Ta is the lowest in df_night until another lower Ta appears
        index = df_night["Ta"].lt(current_Ta).idxmax() if any(df_night["Ta"] < current_Ta) else None

        if index is None:
            index = n

        if index == 1:
            return False

        next_horizon = df_night.iloc[:index]  # Create the new horizon

        future_cooling_load = next_horizon["Cooling_Load"].sum() * Input_Parameters.δt_in_hr
        if PCM_SOC <= future_cooling_load * 1.2:
            return True
        else:
            return False


def get_capacity_cooling(Ta, T_HP, compressor_speed):
    '''
    Calculate the cooling capacity of the heat pump (thermal power output) based on the ambient temperature, the heat pump supply water temperature, and the compressor speed.
    This should set the MaxHeaPumCapCoo parameter, and the MinHeaPumCapCoo parameter should just be 0.3 * MaxHeaPumCapCoo.
    
    Inputs:
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]

    Outputs:
    max_cooling_capacity: maximum cooling capacity of the heat pump (thermal power output) [kW]
    '''
    a0 = 2.176431007927367  
    a1 = 0.0141847062996531  
    a2 = -0.026923757618843087  
    a3 = -0.00010684511342535456  
    a4 = -0.000628998968008253  
    a5 = -0.000179778200817228

    max_cooling_capacity = a0 + a1*Ta + a2*T_HP + a3*Ta*T_HP + a4*Ta**2 + a5*T_HP**2 # [kW] thermal power output for cooling
    # max_cooling_capacity = a0 + a1*Ta + a2*T_HP + a3*Ta*T_HP + a4*Ta**2 + a5*T_HP**2 # [kW] max thermal power output for cooling
    
    return max_cooling_capacity

def get_capacity_heating(Ta, T_HP, compressor_speed):
    '''
    Calculate the heating capacity of the heat pump (thermal power output) based on the ambient temperature, the heat pump supply water temperature, and the compressor speed.
    This should set the MaxHeaPumCapHea parameter, and the MinHeaPumCapHea parameter should just be 0.3 * MaxHeaPumCapHea.
    
    Inputs:
    Ta: ambient temperature [°C]
    T_HP: heat pump supply water temperature [°C]
    compressor_speed: heat pump compressor speed [0-1]

    Outputs:
    max_heating_capacity: maximum cooling capacity of the heat pump (thermal power output) [kW]
    '''

    a0 = 7.662690876075606  
    a1 = 0.2780926084677243  
    a2 = 0.013264680546280954 
    a3 = -0.002744177232083285  
    a4 = -0.00042115501152681524  
    a5 = -0.0013852105061937135

    # max_heating_capacity = compressor_speed*(a0 + a1*Ta + a2*T_HP + a3*Ta*T_HP + a4*Ta**2 + a5*T_HP**2) # [kW] max thermal power output for heating
    max_heating_capacity = a0 + a1*Ta + a2*T_HP + a3*Ta*T_HP + a4*Ta**2 + a5*T_HP**2 # [kW] thermal power output for heating

    return max_heating_capacity

def rbc_battery_controller(E_price):
    
    # Hysteresis Logic could also be done, would require past battery action as an input
    battery_action_new = 0

    if E_price <= Input_Parameters.LMP_low:
        battery_action_new = 2
    elif E_price <= Input_Parameters.LMP_med_low:
        battery_action_new = 1
    elif E_price <= Input_Parameters.LMP_med_high:
        battery_action_new = 0
    elif E_price <= Input_Parameters.LMP_high:
        battery_action_new = -1
    else:
        battery_action_new = -2

    return battery_action_new


'''
def rbc_battery_controller(Battery_SOC):

    B_SOC = Battery_SOC / Input_Parameters.BatterySize
    
    # Hysteresis Logic could also be done, would require past battery action as an input
    battery_action_new = 0

    if B_SOC <= Input_Parameters.B_low:
        battery_action_new = 2
    elif B_SOC <= Input_Parameters.B_med_low:
        battery_action_new = 1
    elif B_SOC <= Input_Parameters.B_med_high:
        battery_action_new = 0
    elif B_SOC <= Input_Parameters.B_high:
        battery_action_new = -1
    else:
        battery_action_new = -2

    return battery_action_new
'''