# Pv28c / Pv28m Episode-Parallel Rollout & Reproducibility Harness (`qrokkun_ai/v28`)

* **Author:** luzi82
* **Date:** 2026-11-14

## Overview

`qrokkun_ai/v28` introduces a paired execution comparison and resumable training harness consisting of a single-process serial control arm (`Pv28c`) and a persistent multi-process parallel arm (`Pv28m`). Both arms train `PlayerV21` (`hidden=472`, 9 actions) warm-started from `Pv21 u400` with fresh Adam optimizers against the frozen stochastic `Sv24 best u980` spawner, preserving all `Pv21` hyperparameters, rewards, clearance teacher targets, and random-seed schedules.

`Pv27w` failed to speed up `Pv27c` because `78%` of collection time stayed in the central process (per-frame tiny-batch CUDA kernel launches, per-frame NAS free-space queries, and per-frame GPU-CPU synchronizations), while per-frame cross-process pipe barriers and per-update process respawning consumed the remaining parallel gains. `v28` removes the per-frame barrier by running each complete episode (`0..7200` frames) on a single CPU thread (`batch=1`, `threads=1`) inside persistent worker processes, then transferring the merged update batch to CUDA once per update for teacher-target computation and PPO optimization. Formal budgets support `updates in {5, 100, 1000}` with scheduled recovery/diagnostic checkpoints at intervals `{1, 10, 100}`. Offline comparison checks terminal Actor and Critic parameter raw-byte equality (`WEIGHTPASS`) first, falling back to binary search over saved intermediate checkpoints only when terminal weights diverge, and reports stage-level speedup ratios separately.

## Analysis

Six isolated zero-context Goldfish reviews (Design Critic, Codebase Details, Zoom-Out/YAGNI, Comprehension/UX, Implementation Readiness/SRE, and Structural/Migration) audited the concept against the repository, and the author resolved all design trade-offs as follows:

1. **GAE Arithmetic and Deferred Teacher Adapter (`Blocker` — Resolved):**
   * `rollouts_to_batch_v21` requires pre-populated teacher fields and computes per-episode reverse Generalized Advantage Estimation (GAE) using Python `float` (`float64`) scalars before converting advantages to `float32`.
   * `v28` provides its own batch assembly routine that preserves episode-local Python `float` GAE recurrence (matching `Pv21` rounding, `reward_scale=0.05`, `gamma=0.9995`, `lam=0.999`, terminal masks where death on frame `7200` is terminal rather than censored, and survivor `V(s_T)` bootstrap), transfers the consolidated observation batch to CUDA once, and fills `expert_actions` and `expert_logits` via one batched `compute_kinematic_expert_targets` call on CUDA (omitted during validation).

2. **Large-Episode IPC Pipe Deadlock and Shared-Memory Slots (`Blocker` — Resolved):**
   * One `7200`-frame episode carries roughly `7.64 MiB` of raw observations alone (`~91.6 MiB` across 12 episodes), exceeding the `64 KiB` Linux pipe buffer and risking deadlock if sent directly over `multiprocessing.Pipe`.
   * `Pv28m` allocates fixed `multiprocessing.shared_memory` episode slots (`16` slots x `7200` frames, `~130 MiB` total RAM). Workers write trajectory arrays directly into their assigned shared-memory slot and send only a compact completion header (`< 256` bytes) over the pipe.

3. **Persistent Worker Policy Synchronization and Per-Episode RNG Isolation (`Major` — Resolved):**
   * Reusing `v27`'s `Env` advances the `Spawner` RNG stream across episodes, and model constructors (`PlayerV21()`, `Spawner()`) advance global CPU Torch RNG.
   * Persistent workers receive an immutable CPU `PlayerV21` weight snapshot tagged with an explicit `policy_version` (acknowledging receipt before executing tasks), instantiate a fresh `Env` and fresh per-episode CPU `torch.Generator` instances (`P: seed + 26_000_000`, `S_train: seed + 25_000_000`, `S_val: seed + 24_000_000`) for every episode task, and never draw from global RNG during rollouts. Programmatic enumeration confirmed zero seed collisions across `u=1..1000` (`episodes 100_000..111_999`, `val 600_000..600_015`, `perm 27_000_017`).

4. **CPU Behavior-Policy Fidelity and Action-Only Spawner Sampling (`Major` — Resolved):**
   * Behavior-policy `old_log_probs`, pre-action `values`, and capped-survivor `bootstrap_value` are recorded directly from the CPU `batch=1` policy forwards that generated each episode, never recomputed on CUDA.
   * A `v28`-local action-only `Spawner` sampler omits the unused `Spawner.log_prob()` pass while preserving `float64` bearing math and the exact draw order (`birth` normal, `speed` integer, `fire` normal).

5. **Deterministic CUDA Execution (`Major` — Confirmed by Author):**
   * Both arms enable deterministic CUDA/cuBLAS execution before CUDA initialization (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `cuda.matmul.allow_tf32=False`, `matmul_precision='highest'`) so `1000` updates (`~13,500+` CUDA PPO steps) remain bitwise reproducible across paired runs on GB10.

6. **Checkpoint Purpose: Exact `--resume` Recovery & Binary-Search Divergence Localization (`Major` — Confirmed by Author):**
   * Scheduled checkpoints (`checkpoint-{u:06d}.pt` with 6-digit zero-padding and explicit integer `int(u)` sorting at intervals `{1, 10, 100}` for `updates in {5, 100, 1000}`, always including `000000` and terminal) serve two operational purposes:
     1. **Exact crash/interruption recovery (`--resume`):** Because episode RNGs depend only on `(update, episode_index)`, saving `{model, actor, critic, training_rng, completed_updates, timings}` at each scheduled checkpoint makes resuming from `checkpoint-{u:06d}.pt` bitwise identical to an uninterrupted run (restoring Adam step/moment tensors onto their exact expected devices).
     2. **Binary-search divergence localization:** Offline comparison checks **terminal** Actor+Critic named parameter raw bytes first. If terminal weights match, it declares `WEIGHTPASS` (`exit 0`) immediately. Only if terminal weights differ (`WEIGHTFAIL`, `exit 2`) does it run binary search over the sorted integer checkpoint schedule to identify `(last_equal_update, first_diverged_update)` and the differing tensor names.

7. **Heterogeneous GB10 Worker Tuning, Cancellation, and Fingerprint Isolation (`Major` / `Nit` — Resolved):**
   * GB10 combines `10x Cortex-X925` (`3.9 GHz`) and `10x Cortex-A725` (`2.8 GHz`) cores. `Pv28m` exposes `--workers` (with dynamic episode-queue dispatch and optional big-core CPU affinity) so a fast `u5` benchmark on GB10 empirically selects the fastest worker configuration before `u100` and `u1000`.
   * Cooperative cancellation uses an in-memory `mp.Event` checked inside CPU episode loops, while `shutil.disk_usage(out)` runs only at coarse update/checkpoint boundaries. Pool lifecycle lives in a run-level `try/finally` block.
   * `v28` computes its own production-only `source_tree_sha` (excluding `**/tests/**`, `docs/`, and caches per `AGENTS.md`) without importing `v26` validation contracts or editing `v21`..`v27`.

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * Implement `qrokkun_ai/v28` with `Pv28c` (serial in-process CPU episodes, no pool) and `Pv28m` (persistent spawned CPU worker pool + `SharedMemory` episode slots).
  * Support formal `--updates {5, 100, 1000}` with scheduled checkpoint save and 16-seed validation intervals `{1, 10, 100}` (`checkpoint-000000.pt` .. `checkpoint-{updates:06d}.pt`), exact `--resume` from the latest scheduled checkpoint on interruption, and `--nonformal` / `--synthetic` test modes.
  * Enable deterministic CUDA execution on both arms and preserve `Pv21` model architecture, locked input checkpoints, fresh Adam initialization, episode seed schedule, Python-float episode-local GAE, CUDA batched teacher targets, and CUDA PPO updates.
  * Gate `WEIGHTPASS` on terminal Actor+Critic named parameter raw-byte equality, using binary search across intermediate checkpoints to localize the earliest diverged milestone if terminal weights mismatch; report stage-level and end-to-end speedup ratios (`Pv28c / Pv28m`) independently.
  * Benchmark worker count/affinity on GB10 at `u5` if needed to lock the fastest `Pv28m` setting before running `u100` and `u1000`.
  * Comply with `AGENTS.md` test budgets (`15-45` tests, `< 30s` target / `90s` hard cap on 1 CPU thread, `1-2` subprocess signal tests).
* **Out-of-Scope Non-Goals:**
  * Bitwise trajectory or checkpoint equivalence against historical `Pv25`, `Pv26`, or `Pv27` runs.
  * 64-seed heldout strength evaluation, checkpoint selection (`best.pt`), or model promotion.
  * Editing any existing file under `qrokkun_ai/v21`..`v27` or `qrokkun_env`.

## Alternatives Weighed

* **Require bitwise equality against archived `Pv27c` checkpoints:** Rejected by author (`luzi82`). Matching `Pv27c` bit-for-bit locks `PlayerV21` rollout inference to frame-by-frame eager CUDA at dynamic batch sizes `<= 12`, capping speedup below `~3x`. Pairing `Pv28c` against `Pv28m` proves multi-worker determinism while removing per-frame GPU/IPC barriers.
* **Require every intermediate checkpoint to be loaded and compared on passing runs:** Rejected by author (`luzi82`). Intermediate checkpoints exist for crash recovery (`--resume`) and binary-search fault localization; checking terminal equality first avoids redundant I/O when the final state matches, while binary search locates the earliest diverged milestone if terminal state differs.
* **Per-frame cross-process `Pipe` barriers (`Pv27w` pattern):** Rejected based on GB10 profiling showing per-frame wakeups and tiny-batch GPU launches negated parallel speedup.
* **Returning whole `7.8 MiB` episodes over `multiprocessing.Pipe`:** Rejected after SRE Goldfish review due to `64 KiB` pipe buffer blocking risks; replaced by pre-allocated `SharedMemory` episode slots.
* **Rewriting GAE recurrence in `float32` on CUDA:** Rejected after Design/Codebase Goldfish reviews in favor of preserving `Pv21`'s exact episode-local Python-float GAE recurrence on CPU.

## Key Decisions & Staging for Next Steps

* **Arms:** `Pv28c` (serial CPU rollout + CUDA teacher/PPO) vs `Pv28m` (persistent multi-worker CPU rollout + `SharedMemory` + CUDA teacher/PPO).
* **Determinism:** Deterministic CUDA/cuBLAS enabled by default on both arms; CPU rollouts run at `threads=1`, `batch=1` with fresh per-episode `P` and `S` generators.
* **Schedules & Recovery:** Formal `--updates {5, 100, 1000}` with checkpoint/validation intervals `{1, 10, 100}` using 6-digit filenames (`checkpoint-{u:06d}.pt` plus explicit integer `int(u)` sorting) and exact `--resume` support from the last completed milestone checkpoint.
* **Comparison Rule:** Check terminal checkpoint Actor+Critic raw bytes first -> `WEIGHTPASS` (`exit 0`); on mismatch, binary-search saved checkpoints to report `(last_equal_update, first_diverged_update)` -> `WEIGHTFAIL` (`exit 2`).
* **GB10 Execution Staging:** Run `u5` first (and compare worker count/big-core affinity if needed to pick the fastest `Pv28m` setting), then execute `u100` and `u1000`.
