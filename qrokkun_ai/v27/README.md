# Pv27c / Pv27w owner contract

Fresh original Pv21 u400 actor+critic, fresh Adam; frozen stochastic CPU Sv24
best u980. No Pv26 resume, heldout, selection or promotion. Formal CUDA:
u5, 12 episodes/update, seed17, cap7200, hidden472/obs84. Validation uses
Pv26 deterministic P and stochastic S modes with seeds600000..600015 at u0
and terminal u5 (explicit adaptation of Pv26 terminal u20).

Parent deploys an immutable git archive and supplies its actual 40-hex commit:

```bash
python -m qrokkun_ai.v27.supervise --arm Pv27c --out-dir /shared/new-run --source-commit ACTUAL_40_HEX --device cuda --player-checkpoint /shared/pv21-u400.pt --spawner-checkpoint /shared/sv24-u980.pt
```

`run` accepts the same options for one arm. `compare DIRECTORY` compares saved
weights offline. `--help` describes reduced `--nonformal --synthetic --device cpu
--updates 1 --episodes 1 --cap 2 --validation 1 --hidden 8` fixtures.
Output directories must not exist; no resume or overwrite. Input validation
reuses v25 locks and adds the actual S state hash. Source digest covers production
Python in qrokkun_ai and qrokkun_env, excluding tests and cache; includes inherited
dependencies. No .git access is required. Source commit is archive provenance
supplied by parent, not inferred or independently authenticated.

Serial creates no pool. Workers use spawn, dedicated pipes and fixed round-robin
episode assignment, persistent for a batch with no refill. Worker count is the
minimum of batch size and logical CPUs constrained by affinity and available
cgroup quotas. Dead episodes leave workers idle. Each worker owns CPU frozen S,
CPU generator and environments; CPU threads=1/no_grad. CUDA/P/teacher/action
sampling and the separate PPO permutation generator stay in parent. All frame
commands are sent before the barrier; results merge in original episode order.
Training episode seeds and P/S/PPO offsets match Pv26. No forward capture hooks,
raw RNG/frame logs, replay, CAS, compression or SQLite.

Checkpoints u0..u5 retain actual weights and bounded Adam/permutation states.
Atomic manifests, progress, evaluation and per-update summaries retain source/input
hashes, runtime flags, death/censor/bootstrap counts, PID/task/CPU statistics,
collector/PPO/evaluation/total timings and normalized throughput. Collector costs
include spawn, startup IPC, barrier and cleanup. Total includes checkpoint and
validation costs. Workers are closed/reaped on success, errors and owner stop.
Free space uses a conservative bounded checkpoint estimate, not Pv26 trace size.

Comparison reports named raw-bit differences at every update. Only final u5
actor+critic bits determine WEIGHTPASS=0 or DIFFERENT=2. Runtime/incomplete=3,
owner stop=130. Intermediate differences do not stop either arm or add a pass
gate. Timing ratios are separate; multiprocessing has no guaranteed speedup.
No automatic Pv28, rerun or new trace. Parent owns readiness review, deployment
and formal launch; this implementation performed none of those actions.
