# Gv20 / Pv21 exploratory implementation

Gv20 lives in `qrokkun_env/gv20.py`; the old v10 environment, scripted spawner, kinds, speeds, double spawns, nine-feature observations, and v10 player stack are unchanged. Godot integration is deferred; no Godot scripts or sprites were changed.

Gv20 uses one bullet radius (2.6 px) and constant velocities. Scripted S speeds are independent Uniform[30, 200] px/s. Its clock replays the old accumulator starting at -0.7, incrementing elapsed before checking the old interval, without double events, then subtracts the first event time (frame 70). The first five shifted frames are 0, 27, 53, 79, 104. Reset creates the frame-zero bullet while the player is centered.

S samples a uniform angle and intersects the center ray with the expanded outer rectangle x[0,320], y[12,240]. Aim targets player plus independent x/y Normal(0, 13.2²) noise, without clipping. A numerically zero direction falls back to (1,0). S, field, and player random streams are independently derived from the episode seed.

Field bullets fire only on positive multiples of 60 frames. They use the same ray map, directly aim at the current player, and take the mean S speed from frames [k-60,k), excluding same-frame S events. An empty window skips the field bullet.

Observations store at most 64 bullets, nearest first with stable ties: dx/304, dy/212, vx/200, vy/200, with an explicit padding mask and nonzero padding velocity sentinels. There is no bullet kind or radius feature. Player features and zero-inertia movement follow v10. Pure learned helpers provide speed 115+85a and radial aim hypot(304,212)*(tanh(s)+1)/2 with a normalized 2D direction. There is no S trainer.

Pv21 clones v10's trainer and decoupled player network, with Gv20 observations and clearance scaling, a distinct architecture id, and checkpoint rejection for v10. Gamma, lambda, loss weights, episodes per update, 7200-step cap, top-k, and horizon retain v10 defaults. `python -m qrokkun_ai.v21.train.train_player_v21` is the entry point; `--quick` selects a tiny CPU smoke configuration.

An exploratory 500-update Pv21 train was launched on GB10 before this commit, from the overlay on parent `f5692ca` (seed 17, 7200-step cap). Run directory: `/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261109-1803-v21-exploratory`. It is not a promotion. Godot remains deferred. Tests were added before implementation; validation used an existing Python environment.
