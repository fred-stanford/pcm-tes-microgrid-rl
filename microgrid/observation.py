import numpy as np
import pandas as pd
from gymnasium import spaces

import Input_Parameters as IP


class ObsBuilder:
    """
    Grid-tied V2 observation — 19 dims, NO forecasts, NO urgency scores.

    [ Battery_SOC, Ti, FCU_Mode,
      PV_Capacity_Factor, E_Load, Ta,
      time_sin, time_cos, doy_sin, doy_cos,
      PCM_Hot_SOC, PCM_Cold_SOC,
      Price_Buy, Price_Sell,
      DR_Event, Grid_Outage_Event,
      DR_Notice, PSPS_Warning,
      last_action ]

    DR_Notice / PSPS_Warning are advance-notice DISPATCH signals (0..1 countdown to onset), not
    forecasts (M6/M12): DR is announced ~hours ahead, PSPS ~a day ahead, both given identically to
    the RBC and RL controllers. DR also raises Price_Buy (critical-peak). Price_Buy/DR/outage/notice
    come from the runtime event realization (env.event_now()), not the CSV.

    `last_action` is the previous RL macro-action (0..19), NOT the FCU mode.
    SOC is normalized over the usable DoD band (0 = min SOC, 1 = full pack).
    """

    N = 19

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        low = np.full(self.N, -100.0, dtype=np.float32)
        high = np.full(self.N, 100.0, dtype=np.float32)
        self._observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def build(self, env) -> np.ndarray:
        idx = env._idx()
        row = env.data_all.iloc[idx]

        dt = pd.to_datetime(row["DateTime"])
        hour = dt.hour + dt.minute / 60.0
        theta_time = 2.0 * np.pi * (hour / 24.0)
        time_sin, time_cos = np.sin(theta_time), np.cos(theta_time)

        day_of_year = dt.timetuple().tm_yday
        theta_doy = 2.0 * np.pi * ((day_of_year - 1) / 365.0)
        doy_sin, doy_cos = np.sin(theta_doy), np.cos(theta_doy)

        SOC = float(env.state["Battery SOC"])
        Ti = float(env.state["Indoor Temperature"])
        FCU_prev = float(env.state["FCU Mode"])

        PV_cf = float(row["PV Capacity Factor"])
        E_load_now = float(row["E_Load"])
        Ta_now = float(row["Ta"])

        PCM_H = float(env.state.get("PCM_Hot_SOC", 0.5))
        PCM_C = float(env.state.get("PCM_Cold_SOC", 0.5))

        price_sell = float(row["Grid Electricity Sell Price"])
        # M6/M12: DR/outage/notice come from the RUNTIME event realization (not the CSV), and DR
        # elevates the buy price (critical-peak). Fall back to CSV columns if events are disabled.
        if getattr(env, "enable_events", False):
            ev = env.event_now()
            dr = ev["dr"]; outage = ev["outage"]
            dr_notice = ev["dr_notice"]; psps_warning = ev["psps_warning"]
            price_buy = float(row["Grid Electricity Buy Price"]) + ev["dr_price_adder"]
        else:
            dr = float(row["DR Event"]); outage = float(row["Grid Outage Event"])
            dr_notice = 0.0; psps_warning = 0.0
            price_buy = float(row["Grid Electricity Buy Price"])

        last_action = float(env.state.get("last_action", 0))

        obs = np.array([
            SOC, Ti, FCU_prev,
            PV_cf, E_load_now, Ta_now,
            time_sin, time_cos, doy_sin, doy_cos,
            PCM_H, PCM_C,
            price_buy, price_sell,
            dr, outage, dr_notice, psps_warning,
            last_action,
        ], dtype=np.float32)

        if self.normalize:
            batt_cap = float(getattr(env, "battery_capacity", IP.BatterySize))
            obs[0] = IP.battery_usable_soc_frac(obs[0], batt_cap)  # SOC -> [0,1] usable band
            obs[1] = (obs[1] - 23.5) / 5.0                # Ti
            # obs[2] FCU mode already in {-1,0,1}
            # obs[3] PV capacity factor already ~[0,1]
            obs[4] = obs[4] / max(1.0, float(IP.InverterSize))  # E_load
            obs[5] = (obs[5] - 15.0) / 15.0               # Ta
            # obs[6..9] sin/cos already in [-1,1]
            # obs[10..11] PCM SOC already ~[0,1]
            obs[12] = obs[12] / 1.15                      # buy price (base peak ~0.53 + DR adder ~0.60)
            obs[13] = obs[13] / 1.05                      # sell price (max ~1.03)
            # obs[14..17] DR / outage / DR-notice / PSPS-warning already in [0,1]
            obs[18] = obs[18] / 19.0                      # last_action (0..19)

        return obs
