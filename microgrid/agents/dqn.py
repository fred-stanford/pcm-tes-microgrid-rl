"""
DQNAgent — DQN controller backed by stable-baselines3.

Wraps SB3's DQN as an ``Agent`` so it runs through the same env + evaluation loop as
``RBCAgent``. The env is expected to already be wrapped with ``ActionRepeat`` (15-min
macro-step). ``DQNConfig`` holds all hyperparameters and the run-folder paths.
"""

import os
from dataclasses import dataclass

from microgrid.agents.base import Agent


@dataclass
class DQNConfig:
    # output location: <base_dir>/<run_name>/{model.zip, episode_log.csv}
    run_name: str = "DQN"
    base_dir: str = "runs"
    # core DQN hyperparameters (SB3 native)
    gamma: float = 0.99
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_interval: int = 1_000
    # network
    hidden_units: int = 128
    hidden_layers: int = 3
    # exploration (ε annealed over `exploration_fraction` of training)
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.1
    # run
    total_timesteps: int = 500_000
    seed: int = 0
    # validation-based model selection (M2): evaluate on a held-out validation env every
    # `eval_freq` steps and keep best_model.zip. Optional generous early-stop is OFF by default
    # (best-model selection already makes overtraining harmless; it just saves compute).
    eval_freq: int = 100_000          # 0 disables inline validation eval
    early_stop: bool = False
    early_stop_patience: int = 5      # consecutive non-improving evals before stopping
    early_stop_min_evals: int = 5

    @property
    def net_arch(self):
        return [self.hidden_units] * self.hidden_layers

    @property
    def run_dir(self) -> str:
        return os.path.join(self.base_dir, self.run_name)

    @property
    def model_path(self) -> str:
        return os.path.join(self.run_dir, "model.zip")

    @property
    def best_model_path(self) -> str:
        # EvalCallback saves the best-on-validation model here as best_model.zip
        return os.path.join(self.run_dir, "best_model.zip")

    @property
    def episode_log_path(self) -> str:
        return os.path.join(self.run_dir, "episode_log.csv")


class DQNAgent(Agent):
    def __init__(self, config: DQNConfig = None, num_actions: int = 20):
        super().__init__(num_actions=num_actions)
        self.config = config or DQNConfig()
        self.model = None

    def build(self, env):
        from stable_baselines3 import DQN
        c = self.config
        self.model = DQN(
            "MlpPolicy", env,
            gamma=c.gamma,
            learning_rate=c.learning_rate,
            buffer_size=c.buffer_size,
            batch_size=c.batch_size,
            learning_starts=c.learning_starts,
            train_freq=c.train_freq,
            target_update_interval=c.target_update_interval,
            exploration_fraction=c.exploration_fraction,
            exploration_initial_eps=c.exploration_initial_eps,
            exploration_final_eps=c.exploration_final_eps,
            policy_kwargs=dict(net_arch=c.net_arch),
            seed=c.seed,
            verbose=1,
        )
        return self.model

    def train(self, env, callback=None, total_timesteps=None, eval_env=None):
        """Train the DQN. If `eval_env` is given, an SB3 EvalCallback evaluates on it every
        `config.eval_freq` steps, saves the best-on-validation model to best_model.zip, and logs
        the validation curve to evaluations.npz (M2: validation-based model selection). The final
        weights are always also saved to model.zip so the best-vs-final gap can be reported.
        """
        from stable_baselines3.common.callbacks import CallbackList, EvalCallback
        if self.model is None:
            self.build(env)
        os.makedirs(self.config.run_dir, exist_ok=True)

        cbs = []
        if callback is not None:
            cbs.append(callback)
        if eval_env is not None and self.config.eval_freq and self.config.eval_freq > 0:
            after = None
            if self.config.early_stop:
                from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
                after = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=self.config.early_stop_patience,
                    min_evals=self.config.early_stop_min_evals, verbose=1,
                )
            cbs.append(EvalCallback(
                eval_env,
                best_model_save_path=self.config.run_dir,   # -> best_model.zip
                log_path=self.config.run_dir,               # -> evaluations.npz (validation curve)
                eval_freq=self.config.eval_freq,
                n_eval_episodes=1,                           # deterministic full-year rollout
                deterministic=True,
                callback_after_eval=after,
                verbose=1,
            ))
        cb = CallbackList(cbs) if cbs else None

        self.model.learn(
            total_timesteps=total_timesteps or self.config.total_timesteps,
            callback=cb,
        )
        self.save()  # final-step weights (model.zip) — kept for the best-vs-final gap
        return {"model": self.config.model_path, "best_model": self.config.best_model_path}

    def save(self, path: str = None):
        os.makedirs(self.config.run_dir, exist_ok=True)
        self.model.save(path or self.config.model_path)

    def load(self, path: str = None, env=None):
        from stable_baselines3 import DQN
        self.model = DQN.load(path or self.config.model_path, env=env)
        return self.model

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_action_greedy(self, obs) -> int:
        a, _ = self.model.predict(obs, deterministic=True)
        return int(a)

    def decide_action(self, environment) -> int:
        # gymnasium 1.x no longer forwards attributes through wrappers -> use unwrapped.
        base = environment.unwrapped
        obs = base.obs_builder.build(base)
        a, _ = self.model.predict(obs, deterministic=True)
        return int(a)
