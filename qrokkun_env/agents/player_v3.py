"""Player v3 — ActorCritic on rich 390-dim obs (64 nearest bullets, zero-padded).

From `train_both_v3_gpu.py` / `obs_rich.py`. Checkpoint: both_v3_player.pt
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env.env import ACTIONS
from qrokkun_env.obs_rich import MAX_BULLETS_RICH, OBS_DIM_RICH, vectorize_rich

OBS_DIM_V3 = OBS_DIM_RICH
MAX_BULLETS_V3 = MAX_BULLETS_RICH
vectorize = vectorize_rich


class PlayerV3(nn.Module):
    def __init__(self, hidden: int = 512) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM_V3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


PlayerAC = PlayerV3
