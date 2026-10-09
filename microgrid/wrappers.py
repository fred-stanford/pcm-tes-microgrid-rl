"""
ActionRepeat — 15-min macro-step wrapper (3 substeps of 5-min physics).

One agent decision is held for ``repeat`` substeps; the per-substep rewards are summed
and the physical KPIs + reward components are accumulated into the macro-step ``info``.
Optionally penalises switching the macro action between consecutive decisions.
"""
import gymnasium as gym

# Idle macro-action = (hp=0, batt=0) = 0*5 + (0+2) = 2. Reset seeds last_action with
# this (NOT 0, which decodes to a full battery discharge).
IDLE_ACTION = 2

# Per-substep physical metrics summed over the macro-step.
_PHYSICAL_KEYS = (
    "grid_purchase_kWh", "grid_sale_kWh", "curtailment_kWh", "loss_of_load_kWh",
    "hp_elec_kWh", "battery_throughput_kWh", "pv_gen_kWh",
)
# Per-substep reward components summed over the macro-step.
_REWARD_COMPONENT_KEYS = (
    "rew_grid_purchase", "rew_grid_sale", "rew_comfort", "rew_loss_of_load",
    "rew_degradation", "rew_infeasible",
)


class ActionRepeat(gym.Wrapper):
    """Repeat each action for ``repeat`` substeps; sum reward, accumulate KPIs."""

    def __init__(self, env, repeat: int = 3, switching_cost: float = 0.0, on_substep=None):
        super().__init__(env)
        self.action_repeat = repeat
        self.switching_cost = switching_cost
        self.on_substep = on_substep
        self.last_action = IDLE_ACTION
        self.episode_steps = env.episode_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = IDLE_ACTION
        self.env.state["last_action"] = IDLE_ACTION
        self.env.state["prev_macro_action"] = IDLE_ACTION
        self.env.state["macro_action"] = IDLE_ACTION
        self.env.state["macro_substep_idx"] = 0
        return obs, info

    def step(self, action):
        action = int(action)

        prev_macro = self.last_action
        switching = (action != self.last_action)
        self.last_action = action
        self.env.state["last_action"] = action

        total_reward = 0.0
        acc_phys = {k: 0.0 for k in _PHYSICAL_KEYS}
        acc_rew = {k: 0.0 for k in _REWARD_COMPONENT_KEYS}

        obs = None
        terminated = truncated = False
        info = {}
        for i in range(self.action_repeat):
            self.env.state["prev_macro_action"] = prev_macro
            self.env.state["macro_action"] = action
            self.env.state["macro_substep_idx"] = i

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            for k in _PHYSICAL_KEYS:
                acc_phys[k] += float(info.get(k, 0.0))
            for k in _REWARD_COMPONENT_KEYS:
                acc_rew[k] += float(info.get(k, 0.0))
            if self.on_substep is not None:
                self.on_substep(int(action), info, self.env)
            if terminated or truncated:
                break

        for k in _PHYSICAL_KEYS:
            info[k] = acc_phys[k]
        for k in _REWARD_COMPONENT_KEYS:
            info[k] = acc_rew[k]

        if switching and self.switching_cost > 0:
            total_reward -= self.switching_cost
            info["rew_switching"] = -self.switching_cost
        else:
            info["rew_switching"] = 0.0

        return obs, total_reward, terminated, truncated, info
