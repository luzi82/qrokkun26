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

