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

Train loops must alias agent classes (`PlayerAC = PlayerV3`, etc.), not redefine nets.
