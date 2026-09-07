# Player / Spawner version map

Each version’s **network + obs/action helpers** live under `qrokkun_env/agents/`.
Training loops remain in `qrokkun_env/train_*.py` (historical entrypoints).

| Ver | Player module | Spawner module | Notes / checkpoints |
|-----|---------------|----------------|---------------------|
| v0 | `agents/player_v0.py` | *(scripted only)* | CPU REINFORCE; `train_player.py` |
| v1 | `agents/player_v1.py` | `agents/spawner_v1.py` | 45-dim / 288-act; `player_gpu.pt`, `baseline_v1/spawner_gpu.pt` |
| v2 | `agents/player_v2.py` (=v1 arch) | `agents/spawner_v2.py` | kind→1152, 8-dim S obs; short both_v2 run |
| v3 | `agents/player_v3.py` | `agents/spawner_v3.py` | rich 390-dim (64 bullets, zero-pad); `both_v3_*.pt` |
| v4 | *(planned)* | *(planned)* | continuous S + mask/one-hot/attention |

Lost / never standalone: nothing recovered beyond what’s in-repo.
