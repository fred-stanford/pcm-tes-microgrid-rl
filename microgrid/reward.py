"""
GridTiedReward — grid-tied economic + comfort reward.

reward = price_sell * grid_sale            (export revenue, EEC)
       - price_buy  * grid_purchase        (import cost, TOU; price_buy includes the DR
                                            critical-peak adder during DR windows — M6a)
       - discomfort                        (thermal comfort)
       - loss_of_load penalty              (M6/M12: emergency backup energy at VoLL — during an
                                            outage the deficit is served by a virtual backup priced
                                            at VoLL; comfort is maintained and this term is the cost
                                            of leaning on it. DR is a price event, not a block, so
                                            deficits arise only during outages.)
       - battery degradation cost          (B1: amortized replacement/wear cost, $/kWh of |dSOC|
                                            throughput; see Input_Parameters derivation)
       - infeasibility penalty             (HP action not aligned with comfort call)

Buy and sell prices are read SEPARATELY from info (the V2 CSV gives asymmetric
`Grid Electricity Buy Price` and `Grid Electricity Sell Price`). The environment
publishes them as `price_buy` / `price_sell`.

Infeasibility (HP action ∈ {0 standby, 1 charge hot, 2 charge cold, 3 use TES}),
judged on the PRE-step PCM SOC and the active comfort call:
  * action 3 (use TES) while the aligned PCM store is ~empty   -> INFEASIBLE_NO_PCM_PENALTY
  * heating call + action 2 (charge cold, wrong store)         -> INFEASIBLE_ACTION_PENALTY
  * cooling call + action 1 (charge hot, wrong store)          -> INFEASIBLE_ACTION_PENALTY

All penalties are tunable constants in Input_Parameters.
"""

import Input_Parameters as IP

# A PCM store with SOC below this is considered "empty" for use-TES feasibility.
PCM_LOW_SOC_THRESHOLD = 0.1

FMU_STANDBY = 0
FMU_CHARGE_HOT = 1
FMU_CHARGE_COLD = 2
FMU_USE_TES = 3


class GridTiedReward:
    def __init__(self, state, info=None, dt_s=None):
        self.state = state
        self.info = info or {}
        self.dt_s = float(dt_s)

    def compute(self) -> float:
        dt_h = float(self.dt_s) / 3600.0

        price_buy = float(self.info.get("price_buy", IP.DEFAULT_ELECTRICITY_PRICE_USD_PER_KWH))
        price_sell = float(self.info.get("price_sell", IP.DEFAULT_ELECTRICITY_PRICE_USD_PER_KWH))
        grid_purchase = float(self.info.get("grid_purchase_kWh", 0.0))
        grid_sale = float(self.info.get("grid_sale_kWh", 0.0))
        loss_of_load = float(self.info.get("loss_of_load_kWh", 0.0))

        # ── Economic: export revenue minus import cost ($) ──
        rew_grid_sale = +price_sell * grid_sale
        rew_grid_purchase = -price_buy * grid_purchase

        # ── Thermal discomfort ($) ──
        Ti = float(self.state.get("Indoor Temperature", 0.0))
        T_low = float(IP.COMFORT_BAND_LOW)    # FR7: aligned to the HP-TES thermostat control range,
        T_high = float(IP.COMFORT_BAND_HIGH)  # not the free-float setpoints (see Input_Parameters)
        deviation = max(0.0, Ti - T_high, T_low - Ti)
        rew_comfort = -IP.DISCOMFORT_USD_PER_DEGC_PER_H * deviation * dt_h

        # ── Emergency backup energy at VoLL ($) — M6/M12 ──
        # `loss_of_load` is the deficit on-site resources can't serve during an outage; it is modelled
        # as met by a virtual backup priced at VoLL (comfort maintained). Penalty = VoLL x backup kWh.
        rew_loss_of_load = -IP.LOSS_OF_LOAD_PENALTY_USD_PER_KWH * loss_of_load

        # ── Battery degradation / cycling cost ($) — B1 ──
        # Amortized wear cost per kWh of |dSOC| throughput (charge AND discharge). The rate is
        # derived so a full cycle costs the pack's amortized replacement cost (see Input_Parameters).
        throughput = float(self.info.get("battery_throughput_kWh", 0.0))
        rew_degradation = -IP.BATTERY_DEGRADATION_USD_PER_KWH * throughput

        # ── Infeasible HP action penalty ($) ──
        hp_action = int(self.info.get("fmu_action", FMU_STANDBY))
        heating = bool(self.info.get("call_for_heating", False))
        cooling = bool(self.info.get("call_for_cooling", False))
        pcm_h = float(self.info.get("pcm_hot_soc_pre", self.state.get("PCM_Hot_SOC", 0.0)))
        pcm_c = float(self.info.get("pcm_cold_soc_pre", self.state.get("PCM_Cold_SOC", 0.0)))

        penalty = 0.0
        if hp_action == FMU_USE_TES:
            if heating and pcm_h < PCM_LOW_SOC_THRESHOLD:
                penalty += IP.INFEASIBLE_NO_PCM_PENALTY
            elif cooling and pcm_c < PCM_LOW_SOC_THRESHOLD:
                penalty += IP.INFEASIBLE_NO_PCM_PENALTY
        if heating and hp_action == FMU_CHARGE_COLD:
            penalty += IP.INFEASIBLE_ACTION_PENALTY
        if cooling and hp_action == FMU_CHARGE_HOT:
            penalty += IP.INFEASIBLE_ACTION_PENALTY
        rew_infeasible = -penalty

        reward_total = (
            rew_grid_sale + rew_grid_purchase
            + rew_comfort + rew_loss_of_load + rew_degradation + rew_infeasible
        )

        self.info["rew_grid_sale"] = float(rew_grid_sale)
        self.info["rew_grid_purchase"] = float(rew_grid_purchase)
        self.info["rew_comfort"] = float(rew_comfort)
        self.info["rew_loss_of_load"] = float(rew_loss_of_load)
        self.info["rew_degradation"] = float(rew_degradation)
        self.info["rew_infeasible"] = float(rew_infeasible)
        self.info["reward_total"] = float(reward_total)

        return float(reward_total)
