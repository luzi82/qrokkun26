"""Player v2 — same network/obs as v1 (45-dim). Trained in `train_both_v2_gpu.py`.

No separate architecture; kept as its own module so checkpoints/docs map cleanly
to a version label. Prefer importing PlayerV1 if you only need the class.
"""

from __future__ import annotations

from qrokkun_env.agents.player_v1 import (
    MAX_BULLETS_V1 as MAX_BULLETS_V2,
    OBS_DIM_V1 as OBS_DIM_V2,
    PlayerV1 as PlayerV2,
    vectorize,
)

PlayerAC = PlayerV2
__all__ = ["PlayerV2", "PlayerAC", "OBS_DIM_V2", "MAX_BULLETS_V2", "vectorize"]
