"""Production ranked top-k Player architecture (non-attention).

This is the formalised version of the ``flat8_hybrid`` positive control that
won the Phase 2 matched architecture/objective experiment: the nearest-ranked
top-k bullet slots are flattened together with the player features and a live
mask, then fed to a two-layer tanh MLP with policy/value heads.

It deliberately does NOT subclass or share weights with :class:`PlayerV4`, and
its checkpoints carry explicit architecture metadata so that a loader can never
confuse the two (see :mod:`qrokkun_env.agents.player_checkpoints`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4
from qrokkun_env.env import ACTIONS

ARCHITECTURE = "player_ranked_topk"
ARCHITECTURE_VERSION = 1
DEFAULT_TOP_K = 8
DEFAULT_HIDDEN = 472


class PlayerRankedTopK(nn.Module):
    """Flattened nearest-ranked top-k bullet MLP policy/value network."""

    architecture = ARCHITECTURE
    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, top_k: int = DEFAULT_TOP_K, hidden: int = DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.hidden = int(hidden)
        self.player_feat = PLAYER_FEAT_V4
        self.bullet_feat = BULLET_FEAT_V4
        self.max_bullets = MAX_BULLETS_V4
        self.in_dim = PLAYER_FEAT_V4 + self.top_k * BULLET_FEAT_V4 + self.top_k
        self.body = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(self.hidden, len(ACTIONS))
        self.value = nn.Linear(self.hidden, 1)

    def architecture_metadata(self) -> dict[str, object]:
        """Self-describing metadata embedded in every checkpoint of this net."""
        return {
            "architecture": self.architecture,
            "architecture_version": self.architecture_version,
            "top_k": self.top_k,
            "hidden": self.hidden,
            "actions": list(ACTIONS),
            "n_actions": len(ACTIONS),
            "player_feat": self.player_feat,
            "bullet_feat": self.bullet_feat,
            "max_bullets": self.max_bullets,
            "in_dim": self.in_dim,
            "param_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def features(self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        b = bullets[:, : self.top_k]
        p = pad[:, : self.top_k]
        live = (~p).to(player.dtype)
        b = b * live.unsqueeze(-1)
        return torch.cat([player, b.reshape(b.shape[0], -1), live], dim=-1)

    def forward(
        self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        h = self.body(self.features(player, bullets, pad))
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


def obs_tensors(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the live env into a batch-of-1 (player, bullets, pad) tuple."""
    from qrokkun_env.agents.obs_v4 import encode_obs

    p, b, m = encode_obs(env)
    return (
        torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(b, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(m, dtype=torch.bool, device=device).unsqueeze(0),
    )


@torch.no_grad()
def argmax_action(net: PlayerRankedTopK, env, device: torch.device) -> int:
    """Deterministic (argmax) action for the current env state."""
    was_training = net.training
    net.eval()
    dist, _value = net(*obs_tensors(env, device))
    if was_training:
        net.train()
    return int(dist.logits.argmax(dim=-1).item())
