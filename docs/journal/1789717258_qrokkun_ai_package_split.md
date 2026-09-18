# qrokkun_ai package split

Moved: agents, train, obs, reset_modes, eval_modes, corner_probe, render_*, frozen_spawner_*, checkpoints, tools.
Stayed in qrokkun_env: env core, policies, sanity.
No shims at old qrokkun_env.agents / qrokkun_env.train paths.
Current train entry: python -m qrokkun_ai.v4.train.both_v4
Legacy flat `qrokkun_ai` module and CLI paths were intentionally removed when
the package was split into independently importable v0-v5 histories.
