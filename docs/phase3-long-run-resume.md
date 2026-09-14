# Phase 3 long-run and resume

The Phase 3 control and auxiliary-retention tools support bounded, resumable
experimental runs. They remain experimental throughout: every PPO artifact is
non-production-compatible and no command promotes a checkpoint.

```bash
PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention \
  --init-checkpoint artifacts/player.pt --teacher artifacts/teacher.pt \
  --run-dir runs/phase3-control-01 --device cuda --max-updates 50 --seed 123

PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention_aux \
  --init-checkpoint artifacts/player.pt --teacher artifacts/teacher.pt \
  --run-dir runs/phase3-aux-01 --device cuda --end-time 20260916-0700
```

`--out-dir` remains supported. If `--run-dir` is omitted, it is the run
directory. `--max-updates N` caps completed updates (across resumes).
`--end-time YYYYMMDD-HHMM` is an Asia/Hong_Kong wall-clock deadline. It may be
combined with `--max-updates`; the first condition observed at an update
boundary stops the run. A deadline is checked before starting an update.

Each run directory contains immutable `run.json`, append-only `status.jsonl`,
append-only `ppo_updates.jsonl` (or `ppo_aux_updates.jsonl`), atomic
`recovery.pt`, reports, and experimental snapshots/final checkpoint.
`run.json` records tool/arm, input identities, Git/device/Torch provenance,
locked knobs, stop arguments, effective seed, and the no-promotion guarantee.

Resume only with the same arguments and identities:

```bash
PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention \
  --init-checkpoint artifacts/player.pt --teacher artifacts/teacher.pt \
  --run-dir runs/phase3-control-01 --device cuda --max-updates 50 --seed 123 --resume
```

Resume fails closed if `run.json` or `recovery.pt` is absent/corrupt, or the
immutable contract differs. It begins at `completed_update + 1` and never
truncates the journals. After every fully completed update the model, Adam
state, totals, Python/NumPy/Torch CPU/CUDA RNG state are atomically saved. The
auxiliary arm additionally saves frozen alpha/calibration information and all
retention sampler-generator state.

SIGINT and SIGTERM request an orderly stop: the active update is allowed to
finish, then its recovery boundary is written and status is recorded as
`interrupted`. There is no claim of a mid-optimizer checkpoint.

Supplying `--seed` seeds Python, NumPy, Torch CPU and available CUDA streams.
Without it, historical default behavior is retained (the existing Torch seed
schedule is used). Resume restores saved RNG state. This is best-effort
continuation on the same software and hardware; CUDA execution is not promised
bitwise-identical across systems or versions.
