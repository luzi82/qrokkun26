# Player / Spawner version map

Canonical **networks + obs/action helpers** live under `qrokkun_env/agents/`.
Training loops in `qrokkun_env/train_*.py` must **alias** those classes (not redefine them).

| Ver | Player module | Spawner module | Train entrypoints | Checkpoints |
|-----|---------------|----------------|-------------------|-------------|
| v0 | `agents/player_v0.py` | *(scripted)* | `train_player.py` | early CPU MLP |
| v1 | `agents/player_v1.py` | `agents/spawner_v1.py` | `train_player_gpu.py`, `train_spawner_gpu.py`, `train_both_gpu.py` | `player_gpu.pt`, `baseline_v1/spawner_gpu.pt` |
| v2 | `agents/player_v2.py` | `agents/spawner_v2.py` | `train_both_v2_gpu.py` | short both_v2 run |
| v3 | `agents/player_v3.py` | `agents/spawner_v3.py` | `train_both_v3_gpu.py` | `both_v3_*.pt` |
| v4 | *(planned)* | *(planned)* | — | continuous S + mask/one-hot/attention |

`train_both_v3` exposes `PlayerAC = PlayerV3` and `SpawnerAC = SpawnerV3`.
