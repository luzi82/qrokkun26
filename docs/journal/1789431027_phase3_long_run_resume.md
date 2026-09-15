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
`recovery.pt`, `recovery_archives/update_<N>.pt` full recoveries at every
200 completed updates, reports, and experimental snapshots/final checkpoint.
Diagnostic Player snapshots remain on the pre-registered 0/10/25/50/100/200
schedule and are distinct from full recovery archives.
`run.json` records `schema_version: 2`, tool/arm, input identities,
Git/device/Torch provenance, locked knobs, resolved effective stop budget,
effective seed, and the no-promotion guarantee. Version 2 is the current
resumable-run schema. Schema v2 is required for all Phase 3 training resume
paths. Version 1 and versionless run directories are archive-only and cannot
be resumed by these scripts; there is no migration.

Resume retains every experiment-defining identity: tool/arm, checkpoint and
teacher identities/hashes, effective seed, locked PPO/retention knobs and
objective, no-promotion guarantee, and all recovery/model/optimizer state.
The sole resume-time exception is an effective stop-budget revision. A
requested completed-update target is accepted when it is an integer at least
the actual completed update, including when it is below the previously
authorized target. A requested HKT deadline is accepted when it is strictly
after the current HKT time, including when it is earlier than the previously
authorized deadline. Targets below completed, deadlines at or before now, and
malformed values fail closed. Resuming with the already authorized pair is
allowed and creates no duplicate audit record. Control and auxiliary resume
loops consume the revised authorized budget; a target equal to completed
performs no further update.

`run.json` is never rewritten. Each approved revision appends and fsyncs one
`stop_budget_amendments.jsonl` record with the explicit
`stop_budget_extended` event, prior and new effective values, completed
update, HKT timestamp, and current runtime provenance. Resume resolves
authority from the original run contract plus this strictly contiguous chain;
malformed, reordered, or non-contiguous records fail closed.

Resume accepts only an explicit exact integer `schema_version: 2`. Missing,
old, unsupported, boolean, and floating-point versions fail closed before
recovery or environment evaluation. Existing versionless/older `run.json`
files, including schema v1, are archival only and cannot be resumed; they
are never rewritten. There is no migration, legacy stop-budget interpretation,
or runtime-provenance bypass.

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
retention sampler-generator state. At every positive multiple of 200 completed
updates the same full recovery payload is also copied to
`recovery_archives/update_<N>.pt` and kept indefinitely.

`--resume-from-update N` rewinds the same run directory to an archived
boundary. `N` must be a positive multiple of 200 that has an archive. The
flag itself selects resume mode; explicit `--resume` is equivalent and not
required. The invocation must supply `--max-updates`, `--end-time`, or both:
the supplied target must be strictly greater than `N`, and a supplied deadline
must be strictly later than current HKT. End-time only resolves the target to
2147483647; max-updates only resolves the deadline to 2099-12-31 23:59 HKT.
The selected archive's historical stop budget is ignored; this invocation's
resolved pair is recorded and used.

This rewind is destructive: persisted progress, later recovery archives, and
stop-budget amendments after `N` are deleted in the same run directory, the
selected archive is restored as `recovery.pt`, and training continues at
`N + 1`. Ordinary `--resume` keeps latest-`recovery.pt` behavior and does not
truncate history.

```bash
# ordinary latest boundary
PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention_aux \
  --init-checkpoint INIT.pt --teacher TEACHER.pt --run-dir RUN --resume \
  --end-time 20260915-1930 --device cuda

# exact rewind to archived update 400; target must be > 400
PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention_aux \
  --init-checkpoint INIT.pt --teacher TEACHER.pt --run-dir RUN \
  --resume-from-update 400 --max-updates 800 --device cuda
```

SIGINT and SIGTERM request an orderly stop: the active update is allowed to
finish, then its recovery boundary is written and status is recorded as
`interrupted`. There is no claim of a mid-optimizer checkpoint.

Supplying `--seed` seeds Python, NumPy, Torch CPU and available CUDA streams.
Without it, historical default behavior is retained (the existing Torch seed
schedule is used). Resume restores saved RNG state. This is best-effort
continuation on the same software and hardware; CUDA execution is not promised
bitwise-identical across systems or versions.
