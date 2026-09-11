# Player / Spawner / Train version map

## Agents (networks + obs/action)

Canonical under `qrokkun_env/agents/`.

| Ver | Player | Spawner |
|-----|--------|---------|
| v0 | `agents/player_v0.py` | *(scripted)* |
| v1 | `agents/player_v1.py` | `agents/spawner_v1.py` |
| v2 | `agents/player_v2.py` | `agents/spawner_v2.py` |
| v3 | `agents/player_v3.py` | `agents/spawner_v3.py` |
| v4 | `agents/player_v4.py` | `agents/spawner_v4.py` |

## Training loops

Canonical under `qrokkun_env/train/`. Top-level `train_*.py` files are **back-compat shims**.

| Ver | Train module | Old shim |
|-----|--------------|----------|
| v0 player | `train/player_v0.py` | `train_player.py` |
| v1 player | `train/player_v1.py` | `train_player_gpu.py` |
| v1 player long | `train/player_v1_long.py` | `train_player_long.py` |
| v1 spawner | `train/spawner_v1.py` | `train_spawner_gpu.py` |
| v1 both | `train/both_v1.py` | `train_both_gpu.py` |
| v2 both | `train/both_v2.py` | `train_both_v2_gpu.py` |
| v3 both | `train/both_v3.py` | `train_both_v3_gpu.py` |
| v4 both | `train/both_v4.py` | `train_both_v4_gpu.py` |

Train loops must alias agent classes (e.g. `PlayerAC = PlayerV4` / `SpawnerAC = SpawnerV4`), not redefine nets.

## v4 train temperature (v4.2)

`--temp-p` / `--temp-s` must be **1.0**. Any other value is rejected immediately
(CLI names kept). Non-1.0 temperature is not supported: it would desync rollout
log-probs from the PPO update. Exploration uses on-policy sampling + entropy, not
temperature scaling.

## v4.3 evaluation modes + corner probe

Eval modes (`sample_policy`, `rng_jitter`) — see `qrokkun_env/eval_modes.py`:

| Mode tag | sample_policy | rng_jitter | Use |
|----------|---------------|------------|-----|
| `det_det` | False | False | Debug / deterministic collapse |
| `det_stoch` | False | True | Robustness; prefer for flee×newS |
| `stoch_stoch` | True | True | Deploy sanity; `torch.random.fork_rng()` + `manual_seed(episode_seed)` |

Status/compare keep legacy keys (`new_vs_new`, …) as **det_det** means, and add
suffixed fields (`new_vs_new_det_stoch`, …) plus optional `by_mode` summaries.
Paired seeds: `PAIRED_EVAL_SEEDS` (3000..3029).

**Corner probe** (independent diagnostic — do **not** mix into new×new mean):

```bash
python -m qrokkun_env.corner_probe --spawner runs/both_v4_spawner.pt \
  --out runs/both_v4_corner_probe.json
# or: python -m qrokkun_env.train.both_v4 --corner-probe
```

Player is **locked** (idle + force `px,py` each frame) at 9 sites: center, 4 corners
(including bottom-right), 4 edge midpoints. Records first-hit, hit rate, closest
approach, birth/aim stats.


## v4.4 terminated vs truncated + final-value bootstrap

On episode end for each trained agent trajectory (`train/both_v4.py`):

| End | `terminated` | `truncated` | `last_value` | GAE mask on last step |
|-----|--------------|------------|--------------|------------------------|
| True death (`env.dead`) | True | False | `0` | no bootstrap |
| `max_steps` time-limit | False | True | `V(final_obs)` | bootstrap |

GAE accepts `last_value` (does **not** always append `0.0` after values). Spawner
time-limit must **not** receive the old “failed to kill” (−2) terminal shaping;
kill bonus (+5) applies only on true death.

## v4.5 Player / Spawner time scales

- Player transitions: `delta_frames=1` (one physics frame).
- Spawner: frames between spawn decisions; last transition also records frames to
  terminal/truncation (included in bootstrap discount).
- Per step: `gamma_t = gamma_frame ** delta_frames_t` (and `lam` similarly).
- CLI `--gamma` / `--lam` are **per-frame**; prefer this over an undocumented shared
  event-level gamma for both agents.

## v4.6 PPO diagnostics + latest/best/snapshot checkpoints

Logged each update on jsonl (`ppo_p` / `ppo_s`) and on status when eval runs:

| Field | Meaning |
|-------|---------|
| `ratio_mean` / `ratio_std` | Full-batch ratio at **update start** (sanity ≈1) |
| `approx_kl` | Mean Schulman approx KL over minibatches |
| `clipfrac` | Fraction of ratios outside `[1-clip, 1+clip]` |
| `entropy` | Policy entropy |
| `explained_variance` | Value vs GAE returns (pre-update) |
| `term_rate_*` / `trunc_rate_*` | True death vs time-limit fractions |

### Checkpoint selection (separate P / S objectives)

| Artifact | Path (defaults) | When | Selection metric |
|----------|-----------------|------|------------------|
| latest player | `runs/both_v4_player.pt` | each eval cadence | — (always overwrite) |
| latest spawner | `runs/both_v4_spawner.pt` | each eval cadence | — |
| **best_player** | `runs/both_v4_player_best.pt` | metric **improves** (maximize) | primarily **newP×scripted** (`newP_vs_scripted` / prefer `*_det_stoch` if present) |
| **best_spawner** | `runs/both_v4_spawner_best.pt` | metric **improves** (minimize flee survival) | primarily **flee×newS_det_stoch** |
| snapshot | `runs/snapshots/{player,spawner}_update_N.pt` | `--snapshot-every N` (>0) | copy of latest |

**Do not** use `new_vs_new_det_det` / `new×new` as the sole selector for either agent (known corner exploit; observation only). See `qrokkun_env/train/checkpoints_v4.py`.

CLI: `--out-player-best`, `--out-spawner-best`, `--snapshot-every`, `--snapshot-dir`.

## v4.7 reset-mode plumbing (mechanism only; fraction default 0.0)

`--random-fraction` (float, default **0.0**) — CLI plumbing only. Opening
`fraction > 0` as a new training baseline is **v4.8**, out of scope here.

- `prepare_initial_state(env, ...)` (`qrokkun_env/reset_modes.py`): thin reset
  reusing `env.reset()` (empty field, `pvx=pvy=0`, elapsed/spawn_acc/rng
  reset), then overrides only the Player position to a uniformly random point
  within a small radius (`RANDOM_PLAYER_RADIUS`, 40px) of the field center.
  Does **not** spawn bullets and does **not** run any dynamics burn-in
  (deferred to v5).
- Explicit training modes (`qrokkun_env/reset_modes.py`): `self_normal`,
  `self_random_player` (trains **Player only**; Spawner traj never packed),
  `p_vs_scripted_normal` / `p_vs_scripted_random`, `s_vs_flee_normal`
  (**always** normal reset regardless of `--random-fraction`).
- Credit/batch split: Spawner PPO batch (`ppo_update_spawner`) is filtered to
  **only** normal-reset trajectories (`reset_mode == "normal"`) each update;
  Player may mix normal + random-reset trajectories when fraction>0.
- Metadata: `Traj.reset_mode`, per-update jsonl fields `surv_by_mode`,
  `reset_mode_counts`, `random_fraction`, `n_s_total` / `n_s_normal`
  (effective Spawner sample count before/after the normal-reset filter).
- Official eval (`eval_pair`, `eval_pair_stats`, `compare`, best-ckpt
  selection) is **unchanged**: always calls `run_episode` with its defaults
  (`initial_reset=True`, `reset_mode="normal"`) — i.e. normal `env.reset()`
  only. Never evaluates from random starts.
- **Default path** (`--random-fraction 0.0`, the default): `prepare_initial_state`
  is never called; the loop mix reduces to the v4.6 proportions
  (`self, self, p_vs_scripted, s_vs_flee, self, s_vs_flee`, all normal reset) —
  no behavior change vs pre-v4.7.
- `build_parser()` / `reject_non_unit_temp()` were extracted to
  `qrokkun_env/train/both_v4_args.py` (argparse only, no torch import) so CLI
  default/plumbing tests don't require torch to be installed; `both_v4.py`
  re-exports both names for backward compatibility.

## v4.8 fraction-only A/B experiment layer (diagnostic only)

`qrokkun_env/train/ab_v48.py` — a standalone config/manifest generator built
**on top of** `--random-fraction` (v4.7); it does **not** touch
`both_v4.py`/`both_v4_args.py`, so the trainer's normal one-command behavior
and default (`--random-fraction 0.0`) are completely unchanged.

- `build_paired_configs(seed=, hours=, treatment_fraction=, out_dir=,
  extra_argv=())` builds a validated `(baseline_args, treatment_args)` pair
  (both parsed via the *unmodified* `both_v4_args.build_parser()`): baseline
  is forced to `--random-fraction 0.0`, treatment gets the caller's
  fraction, `extra_argv` is applied identically to both arms (guaranteeing
  matching seed/hours/training knobs by construction), and each arm gets
  distinct output paths under `out_dir/baseline` / `out_dir/treatment`.
- `validate_paired_args(baseline_args, treatment_args)` rejects: baseline
  `random_fraction != 0.0`; treatment fraction outside `0 < f <= 1.0`; any
  `PAIRED_KNOB_FIELDS` value differing between arms ("incompatible paired
  config"); any `OUTPUT_PATH_FIELDS` value duplicated between arms.
- `build_manifest(args, arm, git_head_value=None)` / `write_manifest(...)` /
  `write_paired_manifests(...)` persist a machine-readable JSON manifest
  **next to each arm's own `--status` file** (`<status-stem>_ab_manifest_v48.json`),
  recording: `arm`, `random_fraction`, `seed`, all paired training/eval
  knobs, `git_head` (best-effort `git rev-parse HEAD`), `eval_reset_mode:
  "normal"`, `promotion: "forbidden_by_this_tool"`, and the `safety_gates`
  list (compare normal-reset unseen-seed Player mean+median vs baseline;
  corner probe must not worsen at any site; monitor P `approx_kl`/`clipfrac`/
  `explained_variance`; monitor effective Spawner sample count
  `n_s_normal`/`n_s_total`; new×new is observation only; promotion of a
  treatment run to the default baseline is forbidden by this script/tool).
  `write_paired_manifests` validates the pair **before** writing anything.
- `python -m qrokkun_env.train.ab_v48 --seed S --hours H --treatment-fraction
  F --out-dir DIR [--extra-argv ...]` is the reproducible, explicit
  invocation layer: it validates + writes both manifests, then prints the
  exact `python -m qrokkun_env.train.both_v4 ...` command line for each arm
  (identical flags except `--random-fraction` and output paths). It never
  launches training itself and never promotes/renames a checkpoint.
- Tests: `qrokkun_env/tests/test_v48_ab_experiment.py` (argparse/json only,
  no torch required — mirrors the v4.7 parser test style).
- v4.8 does not modify PPO, rewards, gamma/lambda, aim/birth geometry, the
  checkpoint selector, dynamics burn-in, or any default training mode; it is
  diagnostic only and does **not** by itself authorize any baseline change.
