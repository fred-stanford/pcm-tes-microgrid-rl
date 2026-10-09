"""
GridTiedEnvBase — generic gymnasium scaffolding for the grid-tied microgrid env.

Handles episode windowing over the dataset, RNG seeding, the obs/reward hooks and
the gym reset plumbing. Subclasses (``GridTiedEnv``) implement ``initializeState``,
``reset_environment``, ``step_environment`` and ``step``.

``obs_builder`` and ``reward_class`` are REQUIRED — the env is always built with the
grid-tied observation/reward (no implicit legacy defaults).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from physics import hp_tes

import Input_Parameters as IP

REQUIRED_COLUMNS = [
    "DateTime", "Ta", "E_Load",
    "Total Electrical Load Forecasted",
    "Total Heating Load Forecasted",
    "Total Cooling Load Forecasted",
]


class GridTiedEnvBase(gym.Env, ABC):
    metadata = {"render_modes": []}

    def __init__(self, data_df: pd.DataFrame, model_coefficients, *,
                 episode_days: int = 7, obs_builder=None, reward_class=None,
                 seed: int = 0, num_actions: int = 20):
        super().__init__()
        if obs_builder is None or reward_class is None:
            raise ValueError("GridTiedEnv requires obs_builder and reward_class.")

        self.dt_s = IP.δt
        self.steps_per_day = int(24 * 3600 / self.dt_s)
        self.episode_steps = int(episode_days * self.steps_per_day)
        self._rng = np.random.default_rng(seed)

        self.data_all = data_df.reset_index(drop=True)
        self.model_coefficients = model_coefficients
        missing = [c for c in REQUIRED_COLUMNS if c not in self.data_all.columns]
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")

        self.obs_builder = obs_builder
        self.reward_class = reward_class
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = self.obs_builder.observation_space

        self.t0 = 0
        self.t = 0
        self.iteration = 1
        self.state = self.initializeState()
        self.model = hp_tes.system.HybridSystemState(self.state)
        self.last_info: Dict[str, Any] = {}

    # ── subclass hooks ────────────────────────────────────────────────────────
    @abstractmethod
    def initializeState(self) -> dict: ...
    @abstractmethod
    def reset_environment(self): ...
    @abstractmethod
    def step_environment(self, action: int): ...
    @abstractmethod
    def step(self, action: int): ...

    # ── dataset windowing ─────────────────────────────────────────────────────
    def _idx(self) -> int:
        return int(self.t0 + self.t)

    def get_row_now(self) -> pd.Series:
        return self.data_all.iloc[self._idx()]

    def get_data_now(self) -> pd.DataFrame:
        i = self._idx()
        return self.data_all.iloc[i:i + 1]

    def get_data_slice(self, horizon_steps: int) -> pd.DataFrame:
        i = self._idx()
        j = min(i + horizon_steps, len(self.data_all))
        return self.data_all.iloc[i:j]

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=None)
        max_start = len(self.data_all) - self.episode_steps
        self.t0 = 0 if max_start <= 0 else int(self._rng.integers(0, max_start + 1))
        self.t = 0
        self.iteration = 1
        self.state = self.initializeState()
        self.last_info = {}
        self.reset_environment()
        return self.obs_builder.build(self), {}

    def close(self):
        pass
