"""Player v0 — small MLP, 45-dim obs (8 nearest bullets). Used by CPU REINFORCE (`train_player.py`)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env.env import ACTIONS
from qrokkun_env.obs import OBS_DIM, MAX_BULLETS, vectorize

OBS_DIM_V0 = OBS_DIM
MAX_BULLETS_V0 = MAX_BULLETS


class PlayerV0(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM_V0, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))

    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.policy(self.body(x)))


# Back-compat alias used in early scripts
PlayerMLP = PlayerV0
