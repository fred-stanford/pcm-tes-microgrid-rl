"""
LoadMatchV2.py
--------------
Single-battery, DR/outage-aware electrical balancing for the grid-tied V2 study.

Promotes `gridtrading_events` (+ helpers) from the colleague's `changes/LoadMatch (1).py`
WITHOUT touching the legacy `LoadMatch.py` (kept intact for the old 2-battery pipeline).

Added vs the original: optional `battery_capacity` / `inverter_size` arguments so each
scenario (TES vs no-TES) can pass its own pack size instead of mutating Input_Parameters
globals. Defaults fall back to IP.BatterySize / IP.InverterSize.

UNITS (verified against the codebase audit):
  Gen, E_Load, M_Load, B_SOC            : [kWh]   (energy over the δt interval)
  battery_capacity / battery_max_*_power: [kWh]
  outputs (grid_purchase/sale, curtailment, loss_of_load, B_SOC_new): [kWh]
The caller (Environment_GridTiedV2) MUST pass Gen/E_Load/M_Load already in kWh.
"""

import Input_Parameters


def _max_chargeable_energy(
    desired_charge_energy,
    B_SOC,
    battery_capacity,
    battery_max_charge_power,
    charging_efficiency,
    allow_import,
    net_energy,
):
    '''Max battery charge energy input [kWh] achievable this step (0 if none).'''
    available_capacity_charge = battery_capacity - B_SOC
    if available_capacity_charge <= 0:
        return 0

    max_chargeable = min(
        desired_charge_energy,
        battery_max_charge_power,
        available_capacity_charge / charging_efficiency,
    )
    if not allow_import:
        max_chargeable = min(max_chargeable, max(0, net_energy))
    return max_chargeable


def _max_dischargeable_energy(
    desired_discharge_energy, B_SOC, battery_max_discharge_power, battery_soc_min=0.0,
):
    '''Max battery discharge energy drawn from pack [kWh] achievable this step (0 if none).'''
    available_discharge = max(0.0, B_SOC - battery_soc_min)
    return min(desired_discharge_energy, battery_max_discharge_power, available_discharge)


def _natural_dispatch(
    B_SOC,
    net_energy,
    battery_capacity,
    battery_max_charge_power,
    battery_max_discharge_power,
    charging_efficiency,
    discharging_efficiency,
    allow_import,
    allow_export,
    battery_soc_min=0.0,
):
    '''
    Serve load from PV, then battery, then grid; absorb surplus into battery then export/curtail.
    Used for battery_action == 0 or when a suggested forced action is infeasible.
    '''
    grid_purchase = 0
    grid_sale = 0
    curtailment = 0
    loss_of_load = 0
    B_SOC_new = B_SOC

    if net_energy > 0:
        available_capacity_charge = battery_capacity - B_SOC
        max_chargeable_energy = min(
            net_energy,
            battery_max_charge_power,
            available_capacity_charge / charging_efficiency,
        )

        actual_charge_stored = max_chargeable_energy * charging_efficiency
        B_SOC_new = B_SOC + actual_charge_stored

        remaining_excess = net_energy - max_chargeable_energy
        if remaining_excess > 0:
            if allow_export:
                grid_sale = remaining_excess
            else:
                curtailment += remaining_excess

    elif net_energy < 0:
        energy_deficit = -net_energy

        max_dischargeable_energy = min(
            energy_deficit / discharging_efficiency,
            battery_max_discharge_power,
            max(0.0, B_SOC - battery_soc_min),
        )

        actual_discharge_output = max_dischargeable_energy * discharging_efficiency
        B_SOC_new = B_SOC - max_dischargeable_energy

        remaining_deficit = energy_deficit - actual_discharge_output
        if remaining_deficit > 0:
            if allow_import:
                grid_purchase = remaining_deficit
            else:
                loss_of_load += remaining_deficit

    return B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load


def _forced_charge_dispatch(
    desired_charge_energy,
    B_SOC,
    net_energy,
    battery_capacity,
    battery_max_charge_power,
    charging_efficiency,
    allow_import,
    allow_export,
):
    '''Execute a feasible forced-charge action; caller must verify feasibility first.'''
    max_chargeable_energy = _max_chargeable_energy(
        desired_charge_energy,
        B_SOC,
        battery_capacity,
        battery_max_charge_power,
        charging_efficiency,
        allow_import,
        net_energy,
    )

    grid_purchase = 0
    grid_sale = 0
    curtailment = 0
    loss_of_load = 0

    actual_charge_stored = max_chargeable_energy * charging_efficiency
    B_SOC_new = B_SOC + actual_charge_stored

    energy_balance = net_energy - max_chargeable_energy
    if energy_balance < 0:
        if allow_import:
            grid_purchase = -energy_balance
        else:
            loss_of_load = -energy_balance
    elif energy_balance > 0:
        if allow_export:
            grid_sale = energy_balance
        else:
            curtailment += energy_balance

    return B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load


def _forced_discharge_dispatch(
    desired_discharge_energy,
    B_SOC,
    net_energy,
    battery_max_discharge_power,
    discharging_efficiency,
    allow_import,
    allow_export,
    battery_soc_min=0.0,
):
    '''Execute a feasible forced-discharge action; caller must verify feasibility first.'''
    max_dischargeable_energy = _max_dischargeable_energy(
        desired_discharge_energy, B_SOC, battery_max_discharge_power, battery_soc_min,
    )

    grid_purchase = 0
    grid_sale = 0
    curtailment = 0
    loss_of_load = 0

    actual_discharge_output = max_dischargeable_energy * discharging_efficiency
    B_SOC_new = B_SOC - max_dischargeable_energy

    energy_balance = net_energy + actual_discharge_output
    if energy_balance < 0:
        if allow_import:
            grid_purchase = -energy_balance
        else:
            loss_of_load += -energy_balance
    elif energy_balance > 0:
        if allow_export:
            grid_sale = energy_balance
        else:
            curtailment += energy_balance

    return B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load


def gridtrading_events(
    Gen,
    E_Load,
    M_Load,
    B_SOC,
    battery_action,
    DR_event,
    Grid_outage,
    battery_capacity=None,
    inverter_size=None,
):
    '''
    Given solar generation, total electrical demand, and battery soc, and the battery action, as well as DR event (1 meaning yes
    , so we lose all grid import capacity, export capacity unlimited) and grid outage event (1 meaning yes, lose both grid import
    and export capacity)
    calculate the new battery soc through charging and discharging, considering efficiency and capacity constraints,
    and return updated battery soc and any grid energy purchase and sale, as well as loss of load or curtailment (if grid trading is
    not an option)

    battery_action is a suggested action: if it cannot be executed under current constraints (battery limits,
    DR, grid outage), natural dispatch is used instead (PV -> battery -> grid for deficits/surplus).

    Inputs:
    Gen: solar generation energy over the time interval [kWh]
    E_Load: electrical load energy over the time interval [kWh]
    M_Load: modelica (HP-TES system) load energy over the time interval [kWh]
    B_SOC: current battery state of charge [kWh]
    battery_action:
        -2: full discharge
        -1: medium discharge
        0: no charge or discharge
        1: medium charge
        2: full charge
    DR_event:
        0: no demand response event happening, normal capacity
        1: demand response event happening, lose all grid import capacity, export capacity unlimited
    Grid_outage:
        0: no grid outage event happening, normal capacity
        1: grid outage event happening, lose all grid import and export capacity
    battery_capacity: pack size [kWh]; defaults to Input_Parameters.BatterySize (per-scenario override).
    inverter_size: power limit [kW]; defaults to Input_Parameters.InverterSize (per-scenario override).

    Outputs:
    B_SOC_new: updated battery state of charge [kWh]
    grid_purchase: energy purchased from the grid over the time interval [kWh]
    grid_sale: energy sold to the grid over the time interval [kWh]
    curtailment: excess energy that could not be exported [kWh]
    loss_of_load: energy deficit that could not be served [kWh]
    '''
    if DR_event not in (0, 1) or Grid_outage not in (0, 1):
        raise ValueError("DR_event and Grid_outage must be 0 or 1.")

    # M6/M12: DR is an announced critical-peak PRICE event, NOT an import block — the grid stays
    # connected during DR and the env raises the buy price (so DR-window import/cost are non-vacuous,
    # controller-discriminating metrics; a good controller pre-positions storage to import less).
    # Only a physical OUTAGE (fault / PSPS) blocks import and export (islanded operation).
    # (DR_event is retained in the signature for logging/back-compat but no longer gates import.)
    allow_import = Grid_outage == 0
    allow_export = Grid_outage == 0

    delta_t = Input_Parameters.δt/3600 # [hr]

    if battery_capacity is None:
        battery_capacity = Input_Parameters.BatterySize  # [kWh]
    if inverter_size is None:
        inverter_size = Input_Parameters.InverterSize    # [kW]

    battery_soc_min = Input_Parameters.battery_soc_min_kwh(battery_capacity)

    battery_max_charge_power = inverter_size * delta_t # [kWh]
    battery_max_discharge_power = inverter_size * delta_t # [kWh]
    charging_efficiency = Input_Parameters.η_PV2B # [1]
    discharging_efficiency = Input_Parameters.η_B2H # [1]

    total_servable_load = E_Load + M_Load # [kWh]
    net_energy = Gen - total_servable_load # [kWh]

    if battery_action == 2:
        desired_charge_energy = battery_max_charge_power
    elif battery_action == 1:
        desired_charge_energy = battery_max_charge_power * 0.5
    elif battery_action == 0:
        desired_charge_energy = 0
    elif battery_action == -1:
        desired_charge_energy = -battery_max_discharge_power * 0.5
    elif battery_action == -2:
        desired_charge_energy = -battery_max_discharge_power
    else:
        raise ValueError("Invalid battery action")

    dispatch_kwargs = dict(
        B_SOC=B_SOC,
        net_energy=net_energy,
        battery_capacity=battery_capacity,
        battery_max_charge_power=battery_max_charge_power,
        battery_max_discharge_power=battery_max_discharge_power,
        charging_efficiency=charging_efficiency,
        discharging_efficiency=discharging_efficiency,
        allow_import=allow_import,
        allow_export=allow_export,
        battery_soc_min=battery_soc_min,
    )

    if desired_charge_energy > 0:
        charge_feasible = _max_chargeable_energy(
            desired_charge_energy,
            B_SOC,
            battery_capacity,
            battery_max_charge_power,
            charging_efficiency,
            allow_import,
            net_energy,
        ) > 0
        if charge_feasible:
            B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load = _forced_charge_dispatch(
                desired_charge_energy,
                B_SOC,
                net_energy,
                battery_capacity,
                battery_max_charge_power,
                charging_efficiency,
                allow_import,
                allow_export,
            )
        else:
            B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load = _natural_dispatch(
                **dispatch_kwargs
            )

    elif desired_charge_energy < 0:
        desired_discharge_energy = -desired_charge_energy
        discharge_feasible = _max_dischargeable_energy(
            desired_discharge_energy, B_SOC, battery_max_discharge_power, battery_soc_min,
        ) > 0
        if discharge_feasible:
            B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load = _forced_discharge_dispatch(
                desired_discharge_energy,
                B_SOC,
                net_energy,
                battery_max_discharge_power,
                discharging_efficiency,
                allow_import,
                allow_export,
                battery_soc_min,
            )
        else:
            B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load = _natural_dispatch(
                **dispatch_kwargs
            )

    else:
        B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load = _natural_dispatch(**dispatch_kwargs)

    B_SOC_new = max(battery_soc_min, min(B_SOC_new, battery_capacity))
    B_SOC_new = round(B_SOC_new * (1 - delta_t * Input_Parameters.BatteryLoss), 3)

    return B_SOC_new, grid_purchase, grid_sale, curtailment, loss_of_load
