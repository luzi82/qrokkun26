"""Numbers mirrored from Godot scripts (player.gd / bullet.gd / main.gd / game_state.gd)."""

from __future__ import annotations

DT = 1.0 / 60.0

VIEW_W, VIEW_H = 320.0, 240.0
# GameState.FIELD = Rect2(8, 20, 304, 212)
FIELD_X, FIELD_Y = 8.0, 20.0
FIELD_W, FIELD_H = 304.0, 212.0

PLAYER_MAX_SPEED = 102.0
PLAYER_MARGIN = 6.0
PLAYER_RADIUS = 4.0

# bullet kind -> radius (bullet.gd)
BULLET_RADIUS = {0: 2.0, 1: 2.6, 2: 3.4, 3: 2.0}
BULLET_CULL = (-24.0, 344.0, -24.0, 264.0)  # x0,x1,y0,y1

SPAWN_ACC_START = -0.7
