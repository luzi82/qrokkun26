# Pv28c / Pv28m — Episode-Parallel Rollout & Reproducibility Harness (`qrokkun_ai/v28`)

Author: `luzi82`

## Overview

`qrokkun_ai/v28` pairs a single-process serial control arm (`Pv28c`) with a persistent multi-worker parallel arm (`Pv28m`) to verify `0-ULP` bitwise weight reproducibility while maximizing rollout collection throughput on GB10:

- **CPU Episode-Level Rollout (`batch=1`, `threads=1`):** Each episode (`0..7200` frames) executes end-to-end on CPU with fresh per-episode `PlayerV21` (`seed + 26_000_000`) and `Spawner` (`seed + 25_000_000` train / `seed + 24_000_000` val) generators, action-only `Spawner` sampling (skipping unused `log_prob`), and zero per-frame IPC or NAS `statvfs` calls.
- **Persistent `SharedMemory` Worker Pool (`Pv28m`):** Workers spawn once at run startup, receive updated `PlayerV21` weights via shared memory (`sync_weights`), write completed episode arrays directly into pre-allocated `SharedMemory` slots, and support big-core CPU affinity (`--workers`, `--pin-cores`).
- **Central CUDA Teacher & Deterministic PPO:** Episodes merge in canonical index order `0..E-1`, compute episode-local Python-float GAE matching `Pv21`, move to CUDA once per update to compute batched `compute_kinematic_expert_targets` (omitted during validation), and run deterministic CUDA PPO (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`).
- **6-Digit Recovery Checkpoints & Binary-Search Comparison:** Formal `--updates {5, 100, 1000}` save `checkpoint-{u:06d}.pt` and validate at intervals `{1, 10, 100}` (always including `000000` and terminal). Supports exact `--resume` after interruption. Offline `compare` checks terminal Actor+Critic raw bytes first (`WEIGHTPASS`, exit `0`) and binary-searches intermediate checkpoints (`WEIGHTFAIL`, exit `2`) only if terminal weights diverge.

## Superseded Decisions (Relative to `Pv27`)

For `v28` only (preserving `v26` and `v27` historical records unchanged):
1. Replaces `Pv27w`'s per-frame `Pipe` barrier, central CUDA rollout inference, and per-collection process respawning with autonomous CPU episode workers over `SharedMemory`.
2. Expands `Pv27`'s `u5`-only formal cap to `{5, 100, 1000}` with scheduled 6-digit checkpoints (`checkpoint-{u:06d}.pt`), periodic validation, and exact `--resume`.
3. Replaces `Pv27`'s all-checkpoints eager scan with terminal-first equality plus `O(log K)` binary-search divergence localization.

## Commands

```bash
# Sequential paired run (Pv28c -> Pv28m -> offline compare)
python -m qrokkun_ai.v28.supervise \
  --out-dir /mnt/minion/88_minion_rw/projects/qrokkun26/runs/YYYYMMDD-HHMM-pv28-cm-u5 \
  --source-commit <40-hex-sha> \
  --updates 5

# Single arm (supports --resume on existing out-dir)
python -m qrokkun_ai.v28.run \
  --arm Pv28m \
  --out-dir /path/to/Pv28m \
  --source-commit <40-hex-sha> \
  --updates 100 \
  --workers 16

# Offline comparison (rewrites <pair-dir>/comparison.json; exit 0=WEIGHTPASS, 2=WEIGHTFAIL)
python -m qrokkun_ai.v28.compare /path/to/pair-dir
```
