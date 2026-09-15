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
directory. `--max-updates N` is the total completed-update target across
resumes, rather than a per-invocation cap. `--end-time YYYYMMDD-HHMM` is an
Asia/Hong_Kong wall-clock deadline. The two controls form one effective stop
budget; at an update boundary, deadline takes precedence when both are
reached.

| Supplied controls | Effective completed-update target | Effective HKT deadline |
| --- | ---: | --- |
| `--end-time` only | 2147483647 | supplied deadline |
| `--max-updates` only | supplied target | 2099-12-31 23:59 HKT |
| neither | 200 | 2099-12-31 23:59 HKT |
| both | supplied target | supplied deadline |

A deadline is checked before starting each update and again at its completed
boundary. Scheduled snapshots remain exactly at updates 0, 10, 25, 50, 100,
and 200 when reached; an additional terminal snapshot/checkpoint and report
are written at the actual stop update when it is not scheduled.

Each run directory contains immutable `run.json`, append-only `status.jsonl`,
append-only `ppo_updates.jsonl` (or `ppo_aux_updates.jsonl`), atomic
`recovery.pt`, reports, and experimental snapshots/final checkpoint.
`run.json` records tool/arm, input identities, Git/device/Torch provenance,
locked knobs, resolved effective stop budget, effective seed, and the
no-promotion guarantee.

Resume retains every experiment-defining identity: tool/arm, checkpoint and
teacher identities/hashes, effective seed, locked PPO/retention knobs and
objective, no-promotion guarantee, and all recovery/model/optimizer state.
The sole resume-time exception is an effective stop-budget extension. Both
values must be at least the currently authorized values and at least one must
increase. Lower targets or earlier deadlines fail closed. Resuming with the
already authorized pair is allowed and creates no duplicate audit record.

For a recognized legacy run whose `run.json` has
`{"end_time_hkt": ..., "max_updates": null}` and which completed update 200
by 06:45 HKT on 2026-09-15, extend the deadline to 12:00 HKT with
**only** `--end-time`:

```bash
PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention \
  --init-checkpoint artifacts/player.pt --teacher artifacts/teacher.pt \
  --run-dir runs/phase3-control-01 --end-time 20260915-1200 --seed 123 --resume
```

For this end-time-only invocation, the effective target resolves to
2147483647, so update 201 is authorized. Passing `--max-updates 200` would
retain target 200 and cannot begin update 201. The focused regression covering
the parser/default, legacy authorization, and update-boundary path passed
without a production change; no production fix was required.

`run.json` is never rewritten. Each approved extension appends and fsyncs one
`stop_budget_amendments.jsonl` record with the explicit
`stop_budget_extended` event, prior and new effective values, HKT timestamp,
and current runtime provenance. Resume resolves authority from the original
run contract plus this strictly contiguous, monotonic chain; malformed,
reordered, or tampered records fail closed.

There is one narrow compatibility adoption for actual pre-extension Phase 3
folders: their `stop_args` shape is exactly
`{"end_time_hkt": ..., "max_updates": null}`. It is interpreted as its
historical effective target of 200 and may be extended only monotonically.
Its first amendment records the current runtime commit/provenance. This is not
a general provenance bypass: only runtime fields (Git commit, dirty status,
Torch version, and device) may evolve for that recognized legacy shape;
inputs and their identities, tool/arm, seed, knobs, objective, and
no-promotion remain exact matches.

An ordinary same-budget resume remains:

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
