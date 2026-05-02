"""Test fixtures for the validate package."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def toy_mdp_2state_2action():
    """Tiny MDP for testing FQE / DR-OPE correctness.

    States: {0, 1}; actions: {0=stay, 1=advance}; horizon: 5 steps.
    Reward: +1 first time entering state 1, 0 otherwise.
    Transitions:
      (s=0, a=0=stay)    -> stays in state 0 deterministically
      (s=0, a=1=advance) -> 50% advance to 1, 50% stay
      (s=1, *)           -> absorbing in state 1

    True value of always-advance from s=0 = P(ever enter state 1 in 5 steps)
    = 1 - 0.5^5 = 0.96875.
    """
    transitions = {
        (0, 0): {0: 1.0},
        (0, 1): {0: 0.5, 1: 0.5},
        (1, 0): {1: 1.0},
        (1, 1): {1: 1.0},
    }
    rewards = lambda s, a, sn: 1.0 if (sn == 1 and s != 1) else 0.0
    horizon = 5
    true_value = {
        "always_advance": 1 - 0.5**5,
        "always_stay": 0.0,
    }
    return {
        "transitions": transitions,
        "rewards": rewards,
        "horizon": horizon,
        "true_value": true_value,
    }
