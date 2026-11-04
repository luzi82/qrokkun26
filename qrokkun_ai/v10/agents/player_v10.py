"""PlayerV10 — Decoupled Actor-Critic with Zero-Inertia Kinematic Clearance Features (120s Cap)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_ai.v10.obs_v10 import (
    BULLET_FEAT_V10,
    DEFAULT_HORIZON_V10,
    DEFAULT_TOP_K_V10,
    MAX_BULLETS_V10,
    PLAYER_FEAT_V10,
    SAFETY_THRESHOLD_PX,
    batch_kinematic_clearances_v10,
    encode_obs_v10,
)
from qrokkun_env.env import ACTIONS

ARCHITECTURE = "player_v10_kinematic_decoupled"
ARCHITECTURE_VERSION = 1
DEFAULT_TOP_K = DEFAULT_TOP_K_V10
DEFAULT_HIDDEN = 472
DEFAULT_LOOKAHEAD_HORIZON = DEFAULT_HORIZON_V10
CLEARANCE_FEAT_V10 = len(ACTIONS) * 2  # 9 cl_norm + 9 safe_flag = 18


class CheckpointV10Error(ValueError):
    """Raised when a PlayerV10 checkpoint fails schema, version, or hash verification."""


class PlayerV10(nn.Module):
    """Decoupled Actor-Critic MLP with zero-inertia kinematic clearance features."""

    architecture = ARCHITECTURE
    architecture_version = ARCHITECTURE_VERSION

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        hidden: int = DEFAULT_HIDDEN,
        lookahead_horizon: int = DEFAULT_LOOKAHEAD_HORIZON,
    ) -> None:
        super().__init__()
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
            or top_k > MAX_BULLETS_V10
        ):
            raise ValueError(f"invalid top_k: {top_k!r}")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
            raise ValueError(f"invalid hidden: {hidden!r}")
        if (
            isinstance(lookahead_horizon, bool)
            or not isinstance(lookahead_horizon, int)
            or lookahead_horizon <= 0
        ):
            raise ValueError(f"invalid lookahead_horizon: {lookahead_horizon!r}")

        self.top_k = int(top_k)
        self.hidden = int(hidden)
        self.lookahead_horizon = int(lookahead_horizon)
        self.player_feat = PLAYER_FEAT_V10
        self.bullet_feat = BULLET_FEAT_V10
        self.max_bullets = MAX_BULLETS_V10
        self.in_dim = (
            PLAYER_FEAT_V10
            + self.top_k * BULLET_FEAT_V10
            + self.top_k
            + CLEARANCE_FEAT_V10
        )

        # Decoupled Policy Backbone & Head
        self.policy_body = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(self.hidden, len(ACTIONS))

        # Decoupled Value Backbone & Head (zero shared parameters with policy)
        self.value_body = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(self.hidden, 1)

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.policy_body.parameters()) + list(self.policy_head.parameters())

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self.value_body.parameters()) + list(self.value_head.parameters())

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "architecture_version": self.architecture_version,
            "top_k": self.top_k,
            "hidden": self.hidden,
            "lookahead_horizon": self.lookahead_horizon,
            "actions": list(ACTIONS),
            "n_actions": len(ACTIONS),
            "player_feat": self.player_feat,
            "bullet_feat": self.bullet_feat,
            "max_bullets": self.max_bullets,
            "in_dim": self.in_dim,
            "param_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def features(
        self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor
    ) -> torch.Tensor:
        b = bullets[:, : self.top_k]
        p = pad[:, : self.top_k]
        live = (~p).to(player.dtype)
        b_masked = b * live.unsqueeze(-1)
        with torch.no_grad():
            clearances = batch_kinematic_clearances_v10(
                player,
                bullets,
                pad,
                horizon=self.lookahead_horizon,
                top_k=self.top_k,
            )
            cl_norm = torch.clamp(clearances, -10.0, 30.0) / 15.0
            safe_flag = (clearances >= SAFETY_THRESHOLD_PX).to(player.dtype)
        return torch.cat(
            [player, b_masked.reshape(b_masked.shape[0], -1), live, cl_norm, safe_flag],
            dim=-1,
        )

    def forward_policy(
        self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor
    ) -> Categorical:
        feat = self.features(player, bullets, pad)
        h_pi = self.policy_body(feat)
        return Categorical(logits=self.policy_head(h_pi))

    def forward_value(
        self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor
    ) -> torch.Tensor:
        feat = self.features(player, bullets, pad)
        h_v = self.value_body(feat)
        return self.value_head(h_v).squeeze(-1)

    def forward(
        self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        feat = self.features(player, bullets, pad)
        h_pi = self.policy_body(feat)
        h_v = self.value_body(feat)
        return Categorical(logits=self.policy_head(h_pi)), self.value_head(h_v).squeeze(-1)


@torch.no_grad()
def argmax_action_v10(net: PlayerV10, env, device: torch.device) -> int:
    was_training = net.training
    net.eval()
    p, b, m = encode_obs_v10(env)
    p_t = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
    b_t = torch.tensor(b, dtype=torch.float32, device=device).unsqueeze(0)
    m_t = torch.tensor(m, dtype=torch.bool, device=device).unsqueeze(0)
    dist = net.forward_policy(p_t, b_t, m_t)
    if was_training:
        net.train()
    return int(dist.logits.argmax(dim=-1).item())


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Deterministic SHA-256 over sorted parameter names, dtypes, shapes, and little-endian bytes."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(b"\x00")
        h.update(repr(tuple(tensor.shape)).encode("utf-8"))
        h.update(b"\x00")
        h.update(tensor.numpy().tobytes())
        h.update(b"\x00")
    return h.hexdigest()


def save_v10_checkpoint(
    net: PlayerV10,
    path: Path | str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    sd_hash = state_dict_sha256(sd)
    payload: dict[str, Any] = {
        **net.architecture_metadata(),
        "state_dict": sd,
        "state_dict_sha256": sd_hash,
        "extra": {} if extra is None else dict(extra),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return payload


def _require_exact_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointV10Error(f"{name} must be an exact int, got {value!r}")
    return value


def load_v10_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
    *,
    eval_mode: bool = True,
) -> tuple[PlayerV10, dict[str, Any]]:
    path = Path(path)
    dev = torch.device(device)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointV10Error(f"failed to load checkpoint {path}: {exc}") from exc

    if not isinstance(ckpt, dict):
        raise CheckpointV10Error("checkpoint payload must be a dict")
    if ckpt.get("architecture") != ARCHITECTURE:
        raise CheckpointV10Error(
            f"unsupported architecture: {ckpt.get('architecture')!r} (expected {ARCHITECTURE!r})"
        )
    arch_ver = _require_exact_int(ckpt.get("architecture_version"), "architecture_version")
    if arch_ver != ARCHITECTURE_VERSION:
        raise CheckpointV10Error(
            f"unsupported architecture_version: {arch_ver!r} (expected {ARCHITECTURE_VERSION})"
        )
    top_k = _require_exact_int(ckpt.get("top_k"), "top_k")
    hidden = _require_exact_int(ckpt.get("hidden"), "hidden")
    horizon = _require_exact_int(ckpt.get("lookahead_horizon"), "lookahead_horizon")
    in_dim = _require_exact_int(ckpt.get("in_dim"), "in_dim")

    net = PlayerV10(top_k=top_k, hidden=hidden, lookahead_horizon=horizon)
    if net.in_dim != in_dim:
        raise CheckpointV10Error(
            f"in_dim mismatch: checkpoint {in_dim} vs reconstructed {net.in_dim}"
        )
    if ckpt.get("actions") != list(ACTIONS):
        raise CheckpointV10Error("actions ordering mismatch in checkpoint")

    sd = ckpt.get("state_dict")
    if not isinstance(sd, dict):
        raise CheckpointV10Error("checkpoint missing state_dict")
    expected_hash = ckpt.get("state_dict_sha256")
    if not isinstance(expected_hash, str) or state_dict_sha256(sd) != expected_hash:
        raise CheckpointV10Error("state_dict_sha256 verification failed")

    try:
        net.load_state_dict(sd, strict=True)
    except Exception as exc:
        raise CheckpointV10Error(f"state_dict load failed: {exc}") from exc

    net.to(dev)
    if eval_mode:
        net.eval()
    return net, ckpt
