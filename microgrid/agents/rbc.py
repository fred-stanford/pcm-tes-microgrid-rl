"""
RBCAgent — rule-based controller wrapped as an MF_AgentBase agent.

Lets the grid-tied V2 RBC (NEW_RBC_SIMPLE.decide_actions) run through the SAME
environment + evaluation loop as the RL agent. `decide_action` returns the discrete
action encoded the way GridTiedEnv decodes it:
    action = fmu_action * 5 + (battery_action + 2)        (fmu∈{0..3}, batt∈{-2..2})

The new CSV has no `PV_Power` column (only `PV Capacity Factor`), so we inject
`PV_Power = PV Capacity Factor * env.pv_size` [kW] into the row before calling the
RBC (the RBC reasons about instantaneous power, not energy — no δt_hr here).
"""

from microgrid.agents.base import Agent
from physics import rbc_rules as rbc
import Input_Parameters as IP

IDLE_ACTION = 2


class RBCAgent(Agent):
    def __init__(self, num_actions: int = 20, system_type: str = None):
        super().__init__(num_actions=num_actions)
        self.system_type = system_type  # if None, taken from the env each step

    def _encode(self, fmu_action: int, battery_action: int) -> int:
        return int(fmu_action) * 5 + (int(battery_action) + 2)

    def decide_action(self, environment) -> int:
        # gymnasium 1.x no longer forwards attributes through wrappers -> use unwrapped.
        base = environment.unwrapped
        row = base.get_row_now()
        row_d = row.to_dict() if hasattr(row, "to_dict") else dict(row)

        # Inject instantaneous PV power [kW] expected by the RBC.
        if "PV_Power" not in row_d or row_d.get("PV_Power") is None:
            pv_size = float(getattr(base, "pv_size"))
            row_d["PV_Power"] = float(row_d["PV Capacity Factor"]) * pv_size

        # M6/M12: the env applies RUNTIME-sampled events (not the CSV columns), so mirror them here
        # — otherwise the RBC would see stale baked events while the physics uses runtime ones. Also
        # expose the DR/PSPS advance-notice signals so the RBC pre-charges on notice (parity).
        if getattr(base, "enable_events", False):
            ev = base.event_now()
            row_d["Grid Outage Event"] = int(round(ev["outage"]))
            row_d["DR Event"] = int(round(ev["dr"]))
            row_d["Grid Electricity Buy Price"] = float(
                row_d.get("Grid Electricity Buy Price", 0.0) or 0.0) + ev["dr_price_adder"]
            row_d["DR Notice"] = float(ev["dr_notice"])
            row_d["PSPS Warning"] = float(ev["psps_warning"])

        system_type = self.system_type or getattr(base, "system_type", "hp_tes")
        fmu_states_prev = getattr(base, "last_fmu_states", None) or None

        decision = rbc.decide_actions(
            row_d,
            base.state,
            fmu_states=fmu_states_prev,
            system_type=system_type,
            battery_capacity=float(getattr(base, "battery_capacity", IP.BatterySize)),
        )
        return self._encode(decision.fmu_action, decision.battery_action)
