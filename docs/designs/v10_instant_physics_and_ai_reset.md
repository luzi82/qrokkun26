# Instantaneous Player Physics & PlayerV10 (120s Cap) Reset

* **Author:** luzi82-bot
* **Date:** 2026-11-04
* **Status:** Approved by owner (luzi82)

---

## 1. The Problem

During Qrokkun26 AI experiments (`v0` through `v5`), Player AI policies hit a survival ceiling around 36s–43s against the scripted Spawner. Root-cause physics inspection revealed an unintended velocity inertia defect in both the Godot game (`scripts/player.gd`) and the Python simulation environment (`qrokkun_env/env.py`):

1. **Asymmetric inertia in old physics**:
   - Releasing keys (`idle`, `dx = 0, dy = 0`) set player velocity (`pvx, pvy`) to `(0, 0)` instantaneously.
   - Pressing a direction key (`dx != 0` or `dy != 0`) updated velocity via `move_toward(pvx, pvy, tx, ty, PLAYER_ACCEL * dt)` with `PLAYER_ACCEL = 980.0` and `PLAYER_MAX_SPEED = 102.0`.
   - Reversing direction at full speed required ~12 frames (0.20 seconds) to reach full velocity in the new direction, causing the ship to continue drifting in the old direction for ~6 frames (~5.1 pixels) after the new direction was pressed.
2. **Invalidated historical AI premise**:
   - All `v0`–`v5` Player/Spawner models, checkpoints, distillation datasets, and journal reports were trained and evaluated under this unintended inertia physics.
   - Furthermore, the 70-second episode cap (`4200` frames) right-censored long episodes before the game's true maximum spawn rate (`80s`) and highest rank title (`120s ETERNAL`).

The owner confirmed that the intended game design has **zero inertia**: pressing any direction moves the ship at constant `MAX_SPEED` in that direction immediately, changing direction switches velocity to the new direction immediately, and releasing keys stops the ship immediately.

---

## 2. Goals & Non-Goals

### Goals
1. **Archive legacy inertia era**: Preserve the entire committed `v0`–`v5` repository tree at git tag `legacy-inertia-physics-v5` (pointing to commit `af3902958836ce296ac6095d03afaa5fb96428bb`, pushed to `origin`). External NAS artifacts under `/mnt/minion/88_minion_ro/projects/qrokkun26/runs/` remain untouched on read-only storage.
2. **Fix Player Physics (Zero Inertia)**:
   - Update `scripts/player.gd` and `qrokkun_env/env.py` so player velocity is instantaneously `(f32(dx * PLAYER_MAX_SPEED), f32(dy * C.PLAYER_MAX_SPEED))` when moving and `(0.0, 0.0)` when `idle`.
   - Regenerate `qrokkun_env/tests/fixtures/monkey_seed42_120.jsonl` using Godot 4.7.2 (`/storage/hermes/tools/godot/godot`) and verify Godot ↔ Python replay parity (`ATOL = 0.05` in `qrokkun_env/monkey_verify.py` and `qrokkun_env/tests/test_monkey_parity.py`, plus exact float32 velocity tests in `qrokkun_env/tests/test_env.py`).
3. **Clean Up Legacy Inertia AI & Docs**:
   - Remove `qrokkun_ai/v0`, `qrokkun_ai/v1`, `qrokkun_ai/v2`, `qrokkun_ai/v3`, `qrokkun_ai/v4`, `qrokkun_ai/v5`, `artifacts/`, and all `docs/journal/*` files except `docs/journal/README.md` from active `main` (preserved at git tag `legacy-inertia-physics-v5`).
   - Update `qrokkun_ai/__init__.py`, `docs/AGENT_VERSIONS.md`, and `README.md`, and add `docs/journal/1793769600_v10_zero_inertia_epoch_reset.txt` documenting the physics fix, the `legacy-inertia-physics-v5` archive tag, and the `v10` (120s cap) baseline.
4. **Bootstrap `qrokkun_ai/v10` under 120s Cap (`7200` frames)**:
   - Define `qrokkun_ai/v10` as the clean starting generation for zero-inertia physics.
   - Standardize the episode cap to **120 seconds (`7200` steps)**, using disjoint seed windows for training (`100000+`), validation/selection (`2000..2029`), and paired evaluation (`3000..3029`), targeting **>= 60.0s mean survival** on `PAIRED_EVAL_SEEDS` (`3000..3029`) against the scripted Spawner.

### Non-Goals
- Modifying Spawner bullet physics, bullet speeds, or scripted spawn intervals (`_spawn_interval`, `_bullet_speed` in `qrokkun_env/env.py` remain unchanged).
- Training a learned Spawner before PlayerV10 achieves the >= 60.0s mean survival target against the scripted Spawner.
- Keeping backward compatibility shims for `v0`–`v5` checkpoints in active `main`.

---

## 3. Technical Plan

### 3.1 Zero-Inertia Physics Specification & Float32 Parity

In `scripts/player.gd`:
```gdscript
	if dir != Vector2.ZERO:
		dir = dir.normalized()
		velocity = dir * MAX_SPEED
		_set_look(dir)
	else:
		# Snap-stop on key release (no coasting slide).
		velocity = Vector2.ZERO
		_set_look(Vector2.ZERO)
```

In `qrokkun_env/env.py`:
```python
    def _integrate_player(self, dx: float, dy: float) -> None:
        if dx == 0.0 and dy == 0.0:
            self.pvx = 0.0
            self.pvy = 0.0
        else:
            self.pvx = f32(dx * C.PLAYER_MAX_SPEED)
            self.pvy = f32(dy * C.PLAYER_MAX_SPEED)
        self.px = f32(self.px + f32(self.pvx * self.dt))
        self.py = f32(self.py + f32(self.pvy * self.dt))
        self.px = f32(
            min(
                max(self.px, C.FIELD_X + C.PLAYER_MARGIN),
                C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN,
            )
        )
        self.py = f32(
            min(
                max(self.py, C.FIELD_Y + C.PLAYER_MARGIN),
                C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN,
            )
        )
```

**Float32 Parity Contract**:
- For cardinal directions (`n`, `s`, `e`, `w`), `abs(pvx)` / `abs(pvy)` equals `102.0` (`0x42cc0000`) or `0.0`.
- For diagonal directions (`ne`, `se`, `sw`, `nw`), both Godot 4 `Vector2(±1, ±1).normalized() * 102.0` and Python `f32((±1.0 / math.hypot(1.0, 1.0)) * 102.0)` produce the exact same IEEE-754 binary32 value `±72.12489318847656` (`0x42903ff2` / `0xc2903ff2`).
- Replay verification (`qrokkun_env/monkey_verify.py` and `qrokkun_env/tests/test_monkey_parity.py`) checks 120 frames recorded from Godot 4.7.2 (`seed=42`, `QROKKUN_MONKEY=1 QROKKUN_SEED=42 QROKKUN_FRAMES=120`) within `ATOL = 0.05`.

### 3.2 `qrokkun_ai/v10` Observation Contract (`qrokkun_ai/v10/obs_v10.py`)

`encode_obs_v10(env: Qrokkun26Env)` consumes a live `Qrokkun26Env` instance directly (partial observability of current frame state; does not access future RNG or bullet `moved` flags) and returns `(player, bullets, pad)`:

1. **`player`: `np.ndarray` of shape `(6,)`, `dtype=float32` (`PLAYER_FEAT_V10 = 6`)**:
   - Index 0: `(env.px - (C.FIELD_X + C.FIELD_W * 0.5)) / (C.FIELD_W * 0.5)` (normalized X in `[-1, 1]`)
   - Index 1: `(env.py - (C.FIELD_Y + C.FIELD_H * 0.5)) / (C.FIELD_H * 0.5)` (normalized Y in `[-1, 1]`)
   - Index 2: `(env.px - (C.FIELD_X + C.PLAYER_MARGIN)) / C.FIELD_W` (normalized distance to left wall in `[0, 1]`)
   - Index 3: `((C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN) - env.px) / C.FIELD_W` (normalized distance to right wall in `[0, 1]`)
   - Index 4: `min(env.elapsed / 120.0, 1.0)` (normalized elapsed time for 120s cap in `[0, 1]`)
   - Index 5: `float(env.spawn_acc)` (spawner accumulator; starts at `C.SPAWN_ACC_START = -0.7`)
2. **`bullets`: `np.ndarray` of shape `(64, 9)`, `dtype=float32` (`MAX_BULLETS_V10 = 64`, `BULLET_FEAT_V10 = 9`)**:
   - Live bullets sorted nearest-first by Euclidean distance `(b.x - env.px)**2 + (b.y - env.py)**2`, truncated to `64`.
   - Each live slot `i`: `[(b.x - env.px) / C.FIELD_W, (b.y - env.py) / C.FIELD_H, b.vx / 125.0, b.vy / 125.0, b.radius / 4.0, onehot_kind_0, onehot_kind_1, onehot_kind_2, onehot_kind_3]`.
   - Each padded slot `i >= len(ordered)`: `[0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]`.
3. **`pad`: `np.ndarray` of shape `(64,)`, `dtype=bool`**:
   - `False` for live slots (`i < len(ordered)`), `True` for empty padded slots.

### 3.3 Discrete-Frame Zero-Inertia Kinematic Clearance (`batch_kinematic_clearances_v10`)

`batch_kinematic_clearances_v10(player: Tensor, bullets: Tensor, pad: Tensor, horizon: int = 12, top_k: int = 12) -> Tensor`:
- **Inputs**: `player` `[N, 6]`, `bullets` `[N, 64, 9]`, `pad` `[N, 64]` (`bool`, `True` = padded).
- **Output**: `min_clear` `[N, 9]` (`float32`), representing the minimum signed surface-to-surface clearance (in pixels) over discrete steps `k = 1..horizon` if candidate action `a in 0..8` is held constant for `horizon` frames, evaluated against the top-`top_k` observed bullets:
  - Reconstructs initial player pixel coordinates `px0, py0` from `player[:, 0:2]` and initial bullet pixel coordinates `bx0, by0`, velocities `bvx, bvy`, and radii `br` from `bullets[:, :top_k, :]`.
  - For each candidate action `a in 0..8` with unit direction `(ux_a, uy_a)`:
    - Constant velocity `vx_a = ux_a * C.PLAYER_MAX_SPEED`, `vy_a = uy_a * C.PLAYER_MAX_SPEED` (`0.0` for `idle`).
    - At each discrete step `k = 1..horizon`:
      - Player position steps and clamps to wall margins `[C.FIELD_X + C.PLAYER_MARGIN, C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN]` and `[C.FIELD_Y + C.PLAYER_MARGIN, C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN]`.
      - Observed bullets step linearly: `bx_k = bx_{k-1} + bvx * C.DT`, `by_k = by_{k-1} + bvy * C.DT` (approximate model assuming visible bullets move each tick).
      - Signed clearance to bullet `j`: `c_{k,a,j} = hypot(bx_k - px_{k,a}, by_k - py_{k,a}) - (C.PLAYER_RADIUS + br_j)`.
      - Padded slots (`pad[:, j] == True`) are masked to `+999.0` before taking `.min(dim=-1)`.
      - If all slots are padded (`0` bullets on field), `min_clear` is `+999.0` for all 9 actions (finite, never `inf`/`nan`).

### 3.4 `PlayerV10` Decoupled Architecture & Checkpoint Schema (`qrokkun_ai/v10/agents/player_v10.py`)

- **Constants**:
  - `ARCHITECTURE = "player_v10_kinematic_decoupled"`
  - `ARCHITECTURE_VERSION = 1`
  - `DEFAULT_TOP_K = 12`
  - `DEFAULT_HIDDEN = 472`
  - `DEFAULT_LOOKAHEAD_HORIZON = 12`
  - `SAFETY_THRESHOLD_PX = 2.0`
- **Feature Assembly (`PlayerV10.features(player, bullets, pad)`)**:
  - Slices `b = bullets[:, :top_k]` (`[N, 12, 9]`) and `p = pad[:, :top_k]` (`[N, 12]`), zeroes padded slots via `live = (~p).to(player.dtype)`.
  - Computes `clearances = batch_kinematic_clearances_v10(player, bullets, pad, horizon=self.lookahead_horizon, top_k=self.top_k)` (`[N, 9]`).
  - Normalizes clearance: `cl_norm = torch.clamp(clearances, -10.0, 30.0) / 15.0` (`[N, 9]`).
  - Computes relative safety flag (relative to observed top-`K` bullets): `safe_flag = (clearances >= SAFETY_THRESHOLD_PX).to(player.dtype)` (`[N, 9]`).
  - Concatenates `[player, b.reshape(N, -1), live, cl_norm, safe_flag]` -> total input dimension `in_dim = 6 + 12 * 9 + 12 + 9 + 9 = 144`.
- **Decoupled Networks inside `PlayerV10(nn.Module)`**:
  - **Actor**: `self.policy_body` (`Linear(144, 472) -> Tanh -> Linear(472, 472) -> Tanh`) + `self.policy_head` (`Linear(472, 9)`).
  - **Critic**: `self.value_body` (`Linear(144, 472) -> Tanh -> Linear(472, 472) -> Tanh`) + `self.value_head` (`Linear(472, 1)`).
  - Because `policy_body` and `value_body` share zero parameters, `value_loss.backward()` has strictly zero gradient on actor parameters.
- **Checkpoint Envelope**:
  - `save_v10_checkpoint(net, path, extra=None)` and `load_v10_checkpoint(path, device)` verify `architecture == "player_v10_kinematic_decoupled"`, `architecture_version == 1` (exact `int`), `top_k`, `hidden`, `lookahead_horizon`, `in_dim`, and `state_dict_sha256`.

### 3.5 Episode Cap, Termination vs. Truncation, and Evaluation Contract

- **Episode Cap**: `MAX_STEPS_V10 = 7200` (`120.0` seconds at `60` Hz), enforced by the rollout collector and evaluator loop.
- **Death Precedence at Cap Boundary**:
  - On step `7200`, if `env.step(a)` returns `done=True` (`env.dead == True`), the episode is classified as `terminated = True`, `truncated = False`, `censored = False` (receives `-1.0` death penalty in `shaped_reward_v10` and `0.0` bootstrap value).
  - Only if the loop completes `7200` steps with `done == False` (`env.dead == False`) is the episode classified as `terminated = False`, `truncated = True`, `censored = True`, with post-step value bootstrap `V(s_7200)` computed on the final un-reset observation.
- **Seed Separation**:
  - Training rollout seeds: start at `100000`, advancing by `episodes_per_update` each update (never overlapping validation or evaluation seeds).
  - Validation / checkpoint selection seeds: `VAL_SEEDS_V10 = list(range(2000, 2030))` (`30` seeds, `7200` steps). Best checkpoint (`best_player_v10.pt`) is selected solely on `VAL_SEEDS_V10` mean survival.
  - Final paired benchmark seeds: `PAIRED_EVAL_SEEDS_V10 = list(range(3000, 3030))` (`30` seeds, `7200` steps).

---

## 4. Alternatives Considered & Risks

1. **Retain `v0`–`v5` packages in `main` vs. Tag-and-Remove**:
   - Keeping `v0`–`v5` in `main` was rejected because their tests and checkpoints depend on the removed inertia physics. Tagging `legacy-inertia-physics-v5` (`af39029`) preserves full git history while giving `v10` a clean slate.
2. **Shared Actor-Critic Backbone vs. Decoupled Actor-Critic**:
   - Shared backbones (recorded under tag `legacy-inertia-physics-v5` in `docs/journal/1792550642_player_v5_gamma_lambda_horizon_results.txt`) suffered severe policy washout when value MSE spiked. Decoupling `policy_body` and `value_body` eliminates value-gradient interference at the cost of ~2x parameter count (~500k params, negligible on GB10).
3. **Risks & Mitigations**:
   - *Partial observability of `batch_kinematic_clearances_v10`*: The clearance feature only sees the nearest `top_k=12` bullets and does not know whether a newly spawned bullet on its first frame has `moved=False`. Consequently, `safe_flag` is an observation-level feature input to the neural network, not an infallible hard shield; the policy MLP + RL learns to combine `cl_norm`, wall distance, and raw bullet features to handle multi-bullet crowding and center positioning.

---

## 5. Detailed Implementation Plan & Phase Checkpoints

### Phase 1: Zero-Inertia Physics & Godot Parity
1. **Modify `scripts/player.gd`**:
   - Replace `velocity = velocity.move_toward(dir * MAX_SPEED, ACCEL * delta)` with `velocity = dir * MAX_SPEED`.
2. **Modify `qrokkun_env/env.py`**:
   - In `Qrokkun26Env._integrate_player`, replace `_move_toward(...)` call with `self.pvx = f32(dx * C.PLAYER_MAX_SPEED)` and `self.pvy = f32(dy * C.PLAYER_MAX_SPEED)`.
3. **Modify `qrokkun_env/tests/test_env.py`**:
   - Add `test_instantaneous_velocity_no_inertia()` asserting 1-frame full speed for all 8 directions, immediate reversal (`"e"` -> `"w"` sets `pvx == -102.0` in 1 frame), and exact binary32 diagonal speed `f32(102.0 / math.sqrt(2.0))`.
4. **Modify `qrokkun_env/tests/fixtures/monkey_seed42_120.jsonl`**:
   - Regenerate using `/storage/hermes/tools/godot/godot` with `xvfb-run -a env QROKKUN_MONKEY=1 QROKKUN_SEED=42 QROKKUN_FRAMES=120 QROKKUN_OUT=/storage/hermes/project/qrokkun26/qrokkun_env/tests/fixtures/monkey_seed42_120.jsonl /storage/hermes/tools/godot/godot --path /storage/hermes/project/qrokkun26 --fixed-fps 60`.
   - **Checkpoint 1 Verification**: Run `pytest qrokkun_env/tests -q` and `python3 qrokkun_env/monkey_verify.py qrokkun_env/tests/fixtures/monkey_seed42_120.jsonl`.

### Phase 2: Legacy Cleanup & V10 Package Implementation
5. **Delete Legacy Directories & Files**:
   - Delete directories: `qrokkun_ai/v0/`, `qrokkun_ai/v1/`, `qrokkun_ai/v2/`, `qrokkun_ai/v3/`, `qrokkun_ai/v4/`, `qrokkun_ai/v5/`, `artifacts/`.
   - Delete legacy journal files under `docs/journal/` matching `docs/journal/17*` (retaining `docs/journal/README.md`).
6. **Update Documentation & Root Package**:
   - Modify `qrokkun_ai/__init__.py` to reference `qrokkun_ai.v10`.
   - Modify `docs/AGENT_VERSIONS.md` and `README.md` to document `v10` and the `legacy-inertia-physics-v5` archive tag.
   - Create `docs/journal/1793769600_v10_zero_inertia_epoch_reset.txt`.
7. **Create `qrokkun_ai/v10` Modules & Tests**:
   - Create `qrokkun_ai/v10/__init__.py`
   - Create `qrokkun_ai/v10/obs_v10.py`
   - Create `qrokkun_ai/v10/agents/__init__.py`
   - Create `qrokkun_ai/v10/agents/player_v10.py`
   - Create `qrokkun_ai/v10/train/__init__.py`
   - Create `qrokkun_ai/v10/train/train_player_v10.py`
   - Create `qrokkun_ai/v10/tests/__init__.py`
   - Create `qrokkun_ai/v10/tests/test_v10.py`
   - **Checkpoint 2 Verification**: Run `pytest qrokkun_env/tests qrokkun_ai/v10/tests -q`.
