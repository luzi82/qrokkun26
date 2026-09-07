"""Shared bullet-set encoder: per-bullet MLP + masked cross-attention from player query."""

from __future__ import annotations

import torch
import torch.nn as nn

from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, PLAYER_FEAT_V4


class BulletSetEncoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, ff: int = 256) -> None:
        super().__init__()
        self.player_proj = nn.Sequential(
            nn.Linear(PLAYER_FEAT_V4, d_model),
            nn.Tanh(),
        )
        self.bullet_proj = nn.Sequential(
            nn.Linear(BULLET_FEAT_V4, d_model),
            nn.Tanh(),
        )
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff),
            nn.GELU(),
            nn.Linear(ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model * 2

    def forward(self, player: torch.Tensor, bullets: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """player [B,P], bullets [B,K,F], pad_mask [B,K] True=PAD."""
        pq = self.player_proj(player).unsqueeze(1)  # [B,1,D]
        bk = self.bullet_proj(bullets)  # [B,K,D]
        # If all padded (no bullets), MultiheadAttention can nan — use zeros.
        all_pad = pad_mask.all(dim=1)  # [B]
        attn_out, _ = self.attn(pq, bk, bk, key_padding_mask=pad_mask, need_weights=False)
        attn_out = attn_out.squeeze(1)
        attn_out = self.norm(attn_out + self.ff(attn_out))
        if all_pad.any():
            attn_out = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(attn_out), attn_out)
        p = self.player_proj(player)
        return torch.cat([p, attn_out], dim=-1)
