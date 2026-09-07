"""Player v1 — ActorCritic, 45-dim obs (8 bullets). BC+PPO vs scripted (`train_player_gpu.py`, both_v1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env.env import ACTIONS
from qrokkun_env.obs import OBS_DIM, MAX_BULLETS, vectorize

OBS_DIM_V1 = OBS_DIM
MAX_BULLETS_V1 = MAX_BULLETS


class PlayerV1(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM_V1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


# Aliases seen in older train scripts
ActorCritic = PlayerV1
PlayerAC = PlayerV1
