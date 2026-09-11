"""Player v4 — attention over masked 64-bullet set; 9-way discrete actions."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env.agents.encoder_v4 import BulletSetEncoder
from qrokkun_env.agents.obs_v4 import encode_obs
from qrokkun_env.env import ACTIONS


class PlayerV4(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = BulletSetEncoder(d_model=d_model)
        self.body = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(
        self, player: torch.Tensor, bullets: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        h = self.body(self.encoder(player, bullets, pad_mask))
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)

    def forward_with_attn(
        self, player: torch.Tensor, bullets: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """Like forward() but also returns per-head cross-attention weights [B, nhead, K]."""
        feat, attn = self.encoder.forward_with_attn(player, bullets, pad_mask)
        h = self.body(feat)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1), attn


def obs_tensors(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p, b, m = encode_obs(env)
    return (
        torch.tensor(p, dtype=torch.float32, device=device),
        torch.tensor(b, dtype=torch.float32, device=device),
        torch.tensor(m, dtype=torch.bool, device=device),
    )


PlayerAC = PlayerV4
