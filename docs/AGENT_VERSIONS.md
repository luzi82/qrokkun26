# Agent versions

## v10

Current implementation. Zero-inertia player physics with a decoupled actor-critic.

| | |
| --- | --- |
| Player | `qrokkun_ai.v10.agents.player_v10.PlayerV10` |
| Train entry | `qrokkun_ai.v10.train.train_player_v10` |
| Episode cap | 120s / 7200-step |
| Physics | zero-inertia player physics |
| Policy | decoupled actor-critic |

`v0`..`v5` are archived at git tag `legacy-inertia-physics-v5`.
