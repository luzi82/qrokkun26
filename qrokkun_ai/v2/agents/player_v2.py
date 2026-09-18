"""Player v2 — local ActorCritic implementation with 45-dimensional observations."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env.env import ACTIONS
from qrokkun_ai.v2.obs import OBS_DIM, MAX_BULLETS, vectorize

OBS_DIM_V2 = OBS_DIM
MAX_BULLETS_V2 = MAX_BULLETS


class PlayerV2(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM_V2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


ActorCritic = PlayerV2
PlayerAC = PlayerV2

__all__ = ["PlayerV2", "PlayerAC", "ActorCritic", "OBS_DIM_V2", "MAX_BULLETS_V2", "vectorize"]
