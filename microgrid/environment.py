from typing import Tuple, Dict, Any

import numpy as np
import Input_Parameters as IP
from physics import hp_tes
from physics import grid_events

from microgrid.environment_base import GridTiedEnvBase
from physics.blackbox import Fast_Backbox_GridTied


# Action encoding (same as legacy grid-tied): action ∈ [0,19]
#   hp_action      = action // 5         ∈ {0,1,2,3}   HP/TES schedule
#   battery_action = (action % 5) - 2    ∈ {-2,-1,0,+1,+2}
# Idle = (hp=0, batt=0) = 0*5 + 2 = 2
IDLE_ACTION = 2
NUM_ACTIONS = 20


def decode_action(action: int) -> Tuple[int, int]:
    a = int(action)
    return a // 5, (a % 5) - 2


class GridTiedEnv(GridTiedEnvBase):
    """
    Grid-tied V2 environment (single battery, DR/outage aware).

    Scenario knobs:
      enable_tes      : if False (no-TES scenarios C/D), hp_action is forced to 0
                        (HP standby still serves comfort) — only the battery acts.
      system_type     : "hp_tes" or "hp_only" (informational / for the RBC agent).
      battery_capacity: pack size [kWh] (TES vs no-TES). Defaults to IP.BatterySize.
      pv_size         : PV nameplate [kW]. Defaults to IP.PVSize.
      inverter_size   : inverter power [kW]. Defaults to battery_capacity / 2.

    Publishes into info (per substep) everything the V2 reward / logger need:
      grid_purchase_kWh, grid_sale_kWh, curtailment_kWh, loss_of_load_kWh,
      price_buy, price_sell, hp_elec_kWh, fmu_action, battery_action,
      DR/outage flags, comfort calls, and the PRE-step PCM SOC (for the
      infeasibility penalty).
    """

    def __init__(self, *args,
                 battery_capacity: float = None,
                 pv_size: float = None,
                 inverter_size: float = None,
                 enable_tes: bool = True,
                 system_type: str = "hp_tes",
                 num_actions: int = NUM_ACTIONS,
                 enable_events: bool = True,
                 event_episode_frac: float = 0.5,
                 dr_case: str = "moderate",
                 event_force: str = "auto",
                 event_reset_seed: int = None,
                 **kwargs):
        # Set scenario attributes BEFORE super().__init__ (it calls initializeState()).
        self.battery_capacity = float(battery_capacity) if battery_capacity is not None else float(IP.BatterySize)
        self.pv_size = float(pv_size) if pv_size is not None else float(IP.PVSize)
        self.inverter_size = float(inverter_size) if inverter_size is not None else (self.battery_capacity / 2.0)
        self.enable_tes = bool(enable_tes)
        self.system_type = str(system_type)
        # M6/M12: grid-service events are sampled at RUNTIME (per episode), not read from the CSV.
        self.enable_events = bool(enable_events)
        self.event_episode_frac = float(event_episode_frac)
        self.dr_case = str(dr_case)
        # How events are drawn at reset (FR5). Training default: force="auto" (event-stratified
        # enrichment) + fresh env-RNG draw each episode. Validation/selection overrides these to
        # force=None (true marginal rate) + a fixed event_reset_seed so the best-checkpoint metric
        # is representative of real conditions and uses a fixed event realization each evaluation (the
        # dominant eval-to-eval variance; the small initial-SOC draw is not seeded but is negligible/yr).
        self.event_force = event_force
        self.event_reset_seed = event_reset_seed
        self._events = None            # dict of per-step arrays for the current episode window
        super().__init__(*args, num_actions=num_actions, **kwargs)

    # ── single-battery state ────────────────────────────────────────────────
    def initializeState(self):
        r = self._rng.random()
        soc_min = IP.battery_soc_min_kwh(self.battery_capacity)
        usable = IP.battery_usable_capacity_kwh(self.battery_capacity)
        state = {
            "Indoor Temperature": 23.5,
            "FCU Mode": 0,
            "Battery SOC": soc_min + r * usable,   # [kWh] within usable DoD band
            "T_HP": 20.0,
            "compressor_speed": 0.0,
            "PCM_Hot_SOC": 0.5,
            "PCM_Cold_SOC": 0.5,
            "PCM_Hot_Temp": float(IP.hot_PCM_T_start),
            "PCM_Cold_Temp": float(IP.cold_PCM_T_start),
            "PCM_Hot_E_kWh": 0.0,
            "PCM_Cold_E_kWh": 0.0,
            "call_for_heating": False,
            "call_for_cooling": False,
            "Price Buy": 0.0,
            "Price Sell": 0.0,
            "DR Event": 0,
            "Grid Outage Event": 0,
            "DR Notice": 0.0,
            "PSPS Warning": 0.0,
            "last_action": IDLE_ACTION,
        }
        return state

    def reset_environment(self):
        self.state = self.initializeState()
        self.model = hp_tes.system.HybridSystemState(self.state)
        self.last_fmu_states: dict = {}
        # Sample a fresh runtime event realization for this episode window (M6/M12).
        # Event-stratified: a fraction of episodes are forced to contain an event (learnability).
        if self.enable_events:
            self.sample_events(seed=self.event_reset_seed, force=self.event_force)

    # ── runtime grid-service events (M6/M12) ───────────────────────────────────
    def sample_events(self, seed=None, force="auto"):
        """Sample an event realization over the current [t0, t0+episode] window.
        `seed` -> reproducible realization (used by the Monte-Carlo eval); else uses the env RNG.
        `force`="auto" applies event-stratified training enrichment; None -> true marginal rate.
        """
        n = self.episode_steps + 1  # +1 for the boundary observation
        dts = self.data_all["DateTime"].iloc[self.t0:self.t0 + n]
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        f = grid_events.choose_force(rng, self.event_episode_frac) if force == "auto" else force
        self._events = grid_events.sample_episode(dts, rng, force=f, dr_case=self.dr_case)
        return self._events

    def event_now(self) -> dict:
        """Event values at the current episode position (self.t), aligned with obs / step."""
        z = {"outage": 0.0, "dr": 0.0, "dr_price_adder": 0.0, "dr_notice": 0.0, "psps_warning": 0.0}
        if not self.enable_events or self._events is None:
            return z
        k = min(int(self.t), len(self._events["outage"]) - 1)
        return {key: float(arr[k]) for key, arr in self._events.items()}

    # ── single-battery step (overrides base which uses "Battery SOC 1/2") ────
    def step(self, action: int):
        action = int(action)
        soc_prev = float(self.state.get("Battery SOC", 0.0))

        new_state, info = self.step_environment(action)
        if info is None:
            info = {}

        soc_new = float(new_state.get("Battery SOC", soc_prev))
        info["battery_throughput_kWh"] = abs(soc_new - soc_prev)
        # Level metrics for the tracker (macro-end values; PCM SOC + pv_gen_kWh
        # are already published by step_environment).
        info["battery_soc_kWh"] = soc_new
        info["battery_soc_frac"] = IP.battery_physical_soc_frac(soc_new, self.battery_capacity)

        self.state = new_state

        reward_obj = self.reward_class(self.state, info=info, dt_s=self.dt_s)
        reward = float(reward_obj.compute())

        self.last_info = info
        self.t += 1
        self.iteration += 1

        terminated = False
        truncated = bool(self.t >= self.episode_steps)
        # On time-limit truncation, return the TRUE next observation (state is already
        # advanced and self.t incremented) so SB3's handle_timeout_termination bootstraps
        # the TD target from a valid state, not an out-of-distribution zeros vector
        # (which the old code returned at every episode boundary). Guard the rare window
        # edge (t0 == max_start) where the next index runs one past the dataset.
        if self._idx() < len(self.data_all):
            obs = self.obs_builder.build(self)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def step_environment(self, action: int) -> Tuple[dict, Dict[str, Any]]:
        # First step: force idle so we never act on an uninitialized hp_tes model.
        if self.t == 0:
            action = IDLE_ACTION

        hp_action, battery_action = decode_action(action)

        # No-TES scenarios: HP never charges/discharges PCM (standby only).
        if not self.enable_tes:
            hp_action = 0

        row = self.get_row_now()
        # M6/M12: overwrite the CSV's (now-ignored) event columns with the runtime-sampled
        # realization, and apply the DR critical-peak PRICE adder (grid stays UP during DR, so
        # import is allowed but expensive; outage still hard-blocks import in loadmatch).
        ev = self.event_now()
        if self.enable_events:
            row = row.copy()
            row["Grid Outage Event"] = int(round(ev["outage"]))
            row["DR Event"] = int(round(ev["dr"]))
            row["Grid Electricity Buy Price"] = float(row["Grid Electricity Buy Price"]) + ev["dr_price_adder"]

        # PRE-step PCM SOC snapshot (used by the reward's infeasibility penalty,
        # which must be judged on the storage state at decision time).
        pcm_h_soc_pre = float(self.state.get("PCM_Hot_SOC", 0.5))
        pcm_c_soc_pre = float(self.state.get("PCM_Cold_SOC", 0.5))

        (new_state, fmu_result_timeseries, fmu_states,
         grid_purchase, grid_sale, curtailment, loss_of_load, passive_dict) = Fast_Backbox_GridTied(
            iteration=self.iteration,
            fmu_action=hp_action,
            battery_action=battery_action,
            row=row,
            model=self.model,
            past_states=self.state,
            model_coefficients=self.model_coefficients,
            battery_capacity=self.battery_capacity,
            pv_size=self.pv_size,
            inverter_size=self.inverter_size,
        )

        # HP electrical consumption [kWh] (fmu power outputs are kW).
        hp_elec_power = (
            fmu_states.get("HP Electric Power", 0.0)
            + fmu_states.get("Pump 1 Electric Power", 0.0)
            + fmu_states.get("Pump 2 Electric Power", 0.0)
            + fmu_states.get("Fan Coil Fan Electric Power", 0.0)
        )
        hp_elec_kWh = float(hp_elec_power) * IP.δt / 3600.0

        pcm_h_temp = float(fmu_states["PCM_Hot_Temp"])
        pcm_c_temp = float(fmu_states["PCM_Cold_Temp"])
        # Reported stored energy [kWh] = calibrated SOC x total store capacity (M10): consistent with
        # the hp_tes model's own SOC and preserving the total capacity (cold ~4.6 / hot ~6.3 kWh_th),
        # instead of a separate temperature->energy map that used a different phase-change window.
        pcm_c_e = float(fmu_states["PCM_Cold_SOC"]) * IP.PCM_C_Size  # cold [kWh]
        pcm_h_e = float(fmu_states["PCM_Hot_SOC"]) * IP.PCM_H_Size   # hot  [kWh]

        # Comfort calls from the hp_tes thermostat (evaluated on the entering Ti,
        # i.e. the decision-time temperature). The blackbox mutates self.model.
        call_h = bool(getattr(self.model, "call_for_heating", False))
        call_c = bool(getattr(self.model, "call_for_cooling", False))

        price_buy = float(row["Grid Electricity Buy Price"])
        price_sell = float(row["Grid Electricity Sell Price"])
        dr = int(row["DR Event"])
        outage = int(row["Grid Outage Event"])

        next_state = {
            "Indoor Temperature": float(new_state["Indoor Temperature"]),
            "FCU Mode": int(new_state["FCU Mode"]),
            "Battery SOC": float(new_state["Battery SOC"]),
            "T_HP": float(new_state["T_HP"]),
            "compressor_speed": float(new_state["compressor_speed"]),
            "PCM_Hot_SOC": float(fmu_states["PCM_Hot_SOC"]),
            "PCM_Cold_SOC": float(fmu_states["PCM_Cold_SOC"]),
            "PCM_Hot_Temp": pcm_h_temp,
            "PCM_Cold_Temp": pcm_c_temp,
            "PCM_Hot_E_kWh": float(pcm_h_e),
            "PCM_Cold_E_kWh": float(pcm_c_e),
            "call_for_heating": call_h,
            "call_for_cooling": call_c,
            "Price Buy": price_buy,
            "Price Sell": price_sell,
            "DR Event": dr,
            "Grid Outage Event": outage,
            "DR Notice": float(ev["dr_notice"]),
            "PSPS Warning": float(ev["psps_warning"]),
            "last_action": int(self.state.get("last_action", IDLE_ACTION)),
        }

        info = {
            # Electrical balance outcomes [kWh]
            "grid_purchase_kWh": float(grid_purchase),
            "grid_sale_kWh": float(grid_sale),
            "curtailment_kWh": float(curtailment),
            "loss_of_load_kWh": float(loss_of_load),
            # Prices ($/kWh) for the reward
            "price_buy": price_buy,
            "price_sell": price_sell,
            # HP consumption
            "hp_elec_kWh": float(hp_elec_kWh),
            # Decoded action (after no-TES forcing)
            "fmu_action": int(hp_action),
            "battery_action": int(battery_action),
            # Grid context
            "DR Event": dr,
            "Grid Outage Event": outage,
            "DR Notice": float(ev["dr_notice"]),
            "PSPS Warning": float(ev["psps_warning"]),
            "dr_price_adder": float(ev["dr_price_adder"]),
            # Comfort calls + pre-step PCM (for infeasibility penalty)
            "call_for_heating": call_h,
            "call_for_cooling": call_c,
            "pcm_hot_soc_pre": pcm_h_soc_pre,
            "pcm_cold_soc_pre": pcm_c_soc_pre,
            "PCM_Hot_SOC": float(fmu_states["PCM_Hot_SOC"]),
            "PCM_Cold_SOC": float(fmu_states["PCM_Cold_SOC"]),
            "PCM_Hot_E_kWh": float(pcm_h_e),
            "PCM_Cold_E_kWh": float(pcm_c_e),
            # Logging / debug
            "DateTime": row.get("DateTime", None),
            "pv_capacity_factor": float(row["PV Capacity Factor"]),
            "pv_gen_kWh": float(row["PV Capacity Factor"]) * self.pv_size * IP.δt / 3600.0,
            "e_load_kw": float(row.get("E_Load", 0.0)),
            "ambient_temp_C": float(row["Ta"]),
            "FCU Mode": int(new_state["FCU Mode"]),
            "Indoor Temperature": float(new_state["Indoor Temperature"]),
            # Passive model (5-min substep)
            "Conduction": float(passive_dict["Conduction"]),
            "Convection": float(passive_dict["Convection"]),
            "Radiative Heat Gain": float(passive_dict["Radiative Heat Gain"]),
            "Radiative Cooling": float(passive_dict["Radiative Cooling"]),
            "Passive Heat Transfer": float(passive_dict["Passive Heat Transfer"]),
            "Internal Heat Gain": float(passive_dict["Internal Heat Gain"]),
            "Net Total Heat Gain": float(passive_dict["Passive Heat Transfer"])
                                   + float(passive_dict["Internal Heat Gain"]),
            "Occupancy": float(row.get("Occupancy", 0.0)),
            "Iteration": int(row.get("Iteration", self.iteration)),
            "data_idx": int(self._idx()),
            "internal_heat_gain_elec_kw": float(
                row.get("Internal Heat Gain (Electrical Load)", 0.0)
            ),
        }

        self.last_fmu_states = fmu_states
        return next_state, info
