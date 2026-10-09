"""Agent — minimal interface shared by the RBC and DQN controllers."""
from abc import ABC, abstractmethod


class Agent(ABC):
    """Base controller. ``decide_action`` maps the env state to a discrete action [0, 19]."""

    def __init__(self, num_actions: int = 20):
        self.num_actions = num_actions

    @abstractmethod
    def decide_action(self, environment) -> int:
        ...
