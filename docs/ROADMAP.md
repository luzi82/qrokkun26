# Qrokkun26 roadmap

0–1 **Playable core** — done  
Godot 4 dodge game, Windows/Linux export, Grok Bot mascot + 8-way look dirs, zero-inertia instantaneous movement + snap-stop, hit animation, denser bullets.

2 **RL env (`qrokkun_env`)** — done  
Pure-Python zero-inertia simulator (`Qrokkun26Env`) + Godot float32 monkey parity (`monkey_seed42_120.jsonl`, 0 mismatches) + rule sanity tests.

3 **AI (`qrokkun_ai/v10`)** — in progress  
- Legacy inertia-physics generations (`v0`–`v5`) archived at git tag `legacy-inertia-physics-v5`.
- `PlayerV10` (zero-inertia 12-frame discrete kinematic clearance features + Decoupled Actor-Critic MLP, 120s / 7200-step cap): reaches **106.16s mean / 119.63s median** (15/30 seeds surviving full 120s cap) by `u=100` on GB10 against scripted Spawner.
- Next: further `PlayerV10` refinement and `v10` Spawner (`S`) adversarial co-training.

4 **Later**  
Playability polish (SFX/BGM, replay/ghost), leaderboards / extra modes.
