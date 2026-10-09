"""
EpisodeCsvLogger — per-episode CSV logger for the grid-tied DQN.

SB3 BaseCallback replacement for the keras-rl EpisodeTimeLoggerGridTied. Accumulates
the V2 reward components and physical KPIs from `info` over each episode and writes
one CSV row when the episode ends. Assumes a single (DummyVecEnv) environment.
"""

import os
import csv
from stable_baselines3.common.callbacks import BaseCallback

# Flow metrics — summed over the episode ($/kWh totals).
_ACC_KEYS = [
    "rew_grid_purchase", "rew_grid_sale", "rew_comfort", "rew_loss_of_load", "rew_infeasible",
    "grid_purchase_kWh", "grid_sale_kWh", "curtailment_kWh", "loss_of_load_kWh",
    "hp_elec_kWh", "battery_throughput_kWh", "pv_gen_kWh",
]

# Level metrics — averaged over the episode (state of charge, temperature).
# battery_soc_kWh [kWh] is physical energy; battery_soc_frac is physical SOC / pack size (~0.2–1).
_MEAN_KEYS = [
    "battery_soc_kWh", "battery_soc_frac",
    "PCM_Hot_SOC", "PCM_Cold_SOC",
    "Indoor Temperature",
]

# Action breakdown — count of macro-steps per component value over the episode.
# hp_action ∈ {0 standby, 1 charge hot, 2 charge cold, 3 use TES};
# battery_action ∈ {-2,-1,0,+1,+2}. Counts sum to ep_len.
_HP_ACTIONS = [0, 1, 2, 3]
_BATT_ACTIONS = [-2, -1, 0, 1, 2]
_BATT_LABEL = {-2: "m2", -1: "m1", 0: "0", 1: "p1", 2: "p2"}


class EpisodeCsvLogger(BaseCallback):
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._episode = 0
        self._reset_acc()

    def _reset_acc(self):
        self._ep_reward = 0.0
        self._ep_len = 0
        self._acc = {k: 0.0 for k in _ACC_KEYS}
        self._mean_acc = {k: 0.0 for k in _MEAN_KEYS}
        self._hp_counts = {a: 0 for a in _HP_ACTIONS}
        self._batt_counts = {a: 0 for a in _BATT_ACTIONS}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            self._ep_reward += float(rewards[i]) if i < len(rewards) else 0.0
            self._ep_len += 1
            for k in _ACC_KEYS:
                self._acc[k] += float(info.get(k, 0.0))
            for k in _MEAN_KEYS:
                self._mean_acc[k] += float(info.get(k, 0.0))
            hp = int(info.get("fmu_action", 0))
            batt = int(info.get("battery_action", 0))
            if hp in self._hp_counts:
                self._hp_counts[hp] += 1
            if batt in self._batt_counts:
                self._batt_counts[batt] += 1
            if i < len(dones) and dones[i]:
                self._flush()
                self._reset_acc()
        return True

    def _flush(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        n = max(1, self._ep_len)
        means = {f"{k}_mean": self._mean_acc[k] / n for k in _MEAN_KEYS}
        hp_counts = {f"act_hp{a}": self._hp_counts[a] for a in _HP_ACTIONS}
        batt_counts = {f"act_batt{_BATT_LABEL[a]}": self._batt_counts[a] for a in _BATT_ACTIONS}
        row = {
            "episode": self._episode,
            "timesteps": self.num_timesteps,
            "ep_len": self._ep_len,
            "episode_reward": self._ep_reward,
            **self._acc,
            **means,
            **hp_counts,
            **batt_counts,
        }
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

        self._episode += 1
