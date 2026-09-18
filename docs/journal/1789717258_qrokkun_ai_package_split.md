# qrokkun_ai package split

Moved: agents, train, obs, reset_modes, eval_modes, corner_probe, render_*, frozen_spawner_*, checkpoints, tools.
Stayed in qrokkun_env: env core, policies, sanity.
No shims at old qrokkun_env.agents / qrokkun_env.train paths.
Train entry: python -m qrokkun_ai.train.both_v4
