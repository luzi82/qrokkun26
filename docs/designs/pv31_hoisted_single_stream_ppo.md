# Pv31m (with Conditional Pv31c Fallback): Single-Stream Hoisted-Feature PPO Optimization (`qrokkun_ai/v31`)

* **Author:** luzi82
* **Date:** 2026-11-15
* **Status:** Approved after EGM Goldfish Review (`deleg_3534638e` — all Majors resolved)

## Superseded Decisions & Lineage

* **`Pv28c / Pv28m` (`062cc9c`, `74b87b4`):** Established autonomous episode-parallel CPU rollouts (`threads=1, batch=1`, persistent `spawn` worker pool + `SharedMemory` slots pinned to GB10's 10 `3.9 GHz` Cortex-X925 big cores) with central deterministic CUDA teacher targets and PPO (`WEIGHTPASS` across `u5`, `u100`, `u1000`).
* **`Pv29c / Pv29m` (`52d18dc`, `442fa21`):** Updated formal benchmark budget to `updates=10`, `episodes=20` (`E20`), `seed=17`, `cap=7200`, checkpoint cadence `1` (`u0..u10`), validation cadence `10` (`u0, u10`), achieving `WEIGHTPASS` (`u10` `model_state_sha = 875a2d40bf89086438c784d66b6629f4593a76287d79b74e708990eb4fe5db0a`), `rollout_s = 91.48s` (`8.02x`), and serial single-stream `ppo_s = 43.40s` (`total_s = 151.93s`).
* **`Pv30c / Pv30m` (`5c19fcb`, `16f4364` — Superseded by `Pv31` for PPO execution):** Attempted to accelerate PPO via two non-default CUDA streams and a 2-worker `ThreadPoolExecutor` (`cuda_actor_critic_streams`). While `Pv30` achieved `0-ULP` reproducibility (`WEIGHTPASS`, reproducing the exact `u10` `model_state_sha = 875a2d40bf89086438c784d66b6629f4593a76287d79b74e708990eb4fe5db0a`), `ppo_s` regressed by `+52%..+59%` (`69.02s` / `66.13s` vs `Pv29m`'s `43.40s`). Nsight Systems nanosecond traces (`assets/pv30-cm/gpu-probe-final-nsys/ppo.json`) proved that GPU active duty cycle was only `10.5%` (`1.49 ms/minibatch`), `58.8%` of GPU kernels (`413/703` per minibatch) came from recomputing parameter-free `net.features()` inside every minibatch across 4 epochs, `~31.6` `cudaStreamSynchronize` / `.item()` stalls occurred per minibatch, CPython GIL + 3-stage barriers doubled host launch latency, and the unpinned learner process frequently migrated onto `2.8 GHz` Cortex-A725 efficiency cores (`3.40x` slower Python/autograd dispatch than `3.9 GHz` Cortex-X925).
* **`Pv31` Decision (Owner-confirmed 2026-11-15):** Supersede `Pv30`'s dual-stream / multi-thread PPO with Tier-1 single-stream eager optimizations and single-big-core learner affinity. Execute `Pv31m` (`u10`, `E20`, `10` pinned workers) and compare `u0..u10` checkpoints raw-bit-for-bit against the archived `Pv30c` reference (`/mnt/minion/88_minion_ro/projects/qrokkun26/runs/20261115-1759-pv30-cm-u10/Pv30c`); launch `Pv31c` **only if** pre-formal CUDA smoke or formal comparison shows `Pv31` does not reproduce `Pv30c` `0-ULP`.

## Overview

`qrokkun_ai/v31` is a versioned adaptation of `qrokkun_ai/v29`–`v30` (leaving all `v21`–`v30` and `qrokkun_env` source files and deployed immutable execution archives untouched; note that because `v28`–`v30` compute `source_digest()` recursively across `qrokkun_ai/**/*.py`, historical `v28`–`v30` reproductions execute from their respective archived commits/snapshots rather than `HEAD`) that accelerates CUDA PPO by `~4.5x–4.8x` vs `Pv30m` (`~3.0x` vs `Pv29m`) while preserving exact `0-ULP` bitwise equivalence to `Pv29c` / `Pv30c`:

1. **Single-Stream Serial Dispatch (`ppo_backend = "cuda_hoisted_single_stream"` on CUDA, `"cpu_serial"` on CPU):** Removes `ThreadPoolExecutor` and non-default CUDA streams, eliminating GIL contention and per-minibatch multi-stream barriers.
2. **Hoisted Parameter-Free Features & Teacher Probabilities:** `PlayerV21.features(player, bullets, pad)` and `F.softmax(exp_logits, dim=-1)` have zero trainable parameters and operate strictly row-wise (`dim=0` independent). Computing `all_feat = net.features(player, bullets, pad)` and `all_t_probs = F.softmax(exp_logits, dim=-1)` once under `torch.no_grad()` before the `4`-epoch loop and indexing `all_feat[mb]` / `all_t_probs[mb]` eliminates `~55%` of per-minibatch kernel launches (`~2.4M` micro-kernels over `u10`). Peak memory at worst-case `20 * 7200 = 144,000` frames is `~62 MiB` (`< 0.05%` of GB10's `128 GiB` memory).
3. **Zero Host-GPU Synchronizations Inside the Minibatch Loop:**
   * Replaces 12 per-parameter `torch.isfinite(grad).all()` DtoH syncs, per-minibatch `bool(torch.isfinite(loss))`, and `clip_grad_norm_(..., error_if_nonfinite=True)` internal DtoH syncs with `clip_grad_norm_(..., error_if_nonfinite=False)` returning 0-dim `a_norm` and `c_norm` tensors, accumulating a sticky boolean tensor on device:
     `nonfinite_flag |= (~torch.isfinite(a_norm)) | (~torch.isfinite(c_norm)) | (~torch.isfinite(actor_loss)) | (~torch.isfinite(value_loss))`
     and checking `bool(nonfinite_flag.item())` plus `finite(net.state_dict())` **once** after the 4-epoch loop (raising `FloatingPointError` before `train()` returns or any checkpoint is written).
   * Accumulates diagnostic metrics (`pl_sum`, `vl_sum`, `dl_sum`, `kl_sum`, `agr_sum`) as 0-dim `torch.float64` tensors on device, calling `.item()` **once** at the end of `train()`.
   * Preserves `Categorical(logits=..., validate_args=False)` and passes `dist.logits` to `F.cross_entropy` and `F.log_softmax` identically to `Pv29`/`Pv30` so every floating-point tensor operation on weights, gradients, and Adam moments matches `Pv29`/`Pv30` to `0 ULP`.
4. **Single-X925 Big-Core Learner Affinity:** After `EpisodePool` spawns and pins its 10 CPU rollout workers (`CPUs [5..9, 15..19]`) — and before the first `.backward()` spawns PyTorch's C++ autograd worker thread — `run.py` pins the parent learner process to the top-frequency CPU core (`ranked_cpus[0][0]`, `CPU 5` on GB10) when `--pin-cores` is `"on"` (or `"auto"` with identifiable frequency info). During `EpisodePool.collect`, the parent sleeps inside `poll()` (`< 0.01%` CPU); during `assemble_batch` and `train`, all 10 rollout workers block idle in `conn.recv()`, giving the Python main thread and PyTorch C++ autograd thread exclusive `3.9 GHz` L1/L2 locality without Cortex-A725 migration or cross-cluster DSU ping-pong (`1.44s/update` vs `4.92s` on A725 and `3.19s` across split X925 clusters).

## Pre-Implementation Empirical Proof on GB10

A read-only in-memory GB10 probe (`80,526` frames = 1 full `E20` update, `hidden=472`, `minibatch=512`, `4` epochs, `seed=27000017`, deterministic CUDA) confirmed:
* `hoisted all_feat[mb]` vs `net.features(player[mb], bullets[mb], pad[mb])`: **`0` ULP differences across all `4,096` sampled rows**.
* Post-update `weight_diff(Pv30, Pv31)`: **`[]` (all 12 Actor+Critic parameter tensors `0-ULP` raw-byte identical)**; Actor Adam state `0-ULP` identical; Critic Adam state `0-ULP` identical; `training_rng` state identical.
* Single-update PPO wall time (`80,526` frames): **`1.440s–1.444s`** on `CPU 5` (`Cortex-X925`), projecting `u10 ppo_s ≈ 14.4s` (`~4.8x` faster than `Pv30m`'s `69.02s`, `~3.0x` faster than `Pv29m`'s `43.40s`).

## Architecture, Comparison & Fallback Contract (`qrokkun_ai/v31`)

* **Modules (`qrokkun_ai/v31/`):**
  * `__init__.py`, `core.py`, `run.py`, `compare.py`, `supervise.py`, `README.md`, and `tests/{__init__.py,test_contract.py,test_inherited.py,test_ppo_fast.py}`.
  * Schemas & CLI identities: `pv31_manifest_v1`, `pv31_checkpoint_v1`, arms `Pv31c` and `Pv31m`.
  * `source_digest()` hashes `qrokkun_ai/**/*.py`, `qrokkun_env/**/*.py`, and `qrokkun_ai/v25/reference.json` (excluding `tests` and `__pycache__`, per `AGENTS.md` §2.5).
* **Cross-Version Reference Comparison (`Pv30c` vs `Pv31m`) & Paired Fallback (`Pv31c` vs `Pv31m`):**
  * `compare.py` supports two modes:
    1. **Paired same-version mode (`python -m qrokkun_ai.v31.compare <pair_root>`):** Compares `<pair_root>/Pv31c` vs `<pair_root>/Pv31m` with strict `manifest["semantic"]` equality (used for self-contained nonformal smokes and if `Pv31c` fallback is ever needed).
    2. **Cross-version reference mode (`python -m qrokkun_ai.v31.compare <out_root> --reference-dir <ref_arm_dir> --candidate-dir <cand_arm_dir>`):** Compares an archived completed `Pv30c` reference arm (`pv30_manifest_v1` / `pv30_checkpoint_v1`, copied to local scratch before `torch.load` per NAS rules) against `Pv31m` (`pv31_manifest_v1` / `pv31_checkpoint_v1`). Strictly verifies identical scientific and deterministic-runtime invariants (`updates`, `checkpoint_cadence`, `checkpoint_boundaries`, `validation_cadence`, `validation_boundaries`, `episodes`, `cap`, `seed`, `validation`, `validation_seeds`, `hidden`, `nonformal`, `synthetic`, `hyperparameters`, `initial_player_state_sha`, `spawner_state_sha`, `inputs`, `fresh_adam`, RNG offsets, `permutation_seed`, `rollout` runtime, and `learner` device/deterministic/precision/library settings), allowing only the intentional versioned fields (`schema`, `source_commit`, `source_tree_sha`, `ppo_backend`, `cuda_streams_enabled`, `host_dispatch_workers`, `kernel_overlap_measured`) to differ and recording both manifests' provenance in `comparison.json`. Any invariant mismatch, missing/incomplete checkpoint, or non-finite tensor raises an error (`exit 3`) and halts rather than being misclassified as `WEIGHTFAIL` (`exit 2`).
  * In both modes, `compare.py` linearly inspects all scheduled checkpoints `u0..u10` (`12` named Actor+Critic tensors raw bytes, shapes, dtypes, and finiteness), derives primary verdict (`WEIGHTPASS` exit `0` vs `WEIGHTFAIL` exit `2`) from terminal `u10`, and computes stage/total wall-clock ratios (`reference / candidate`).
* **Execution Plan & Conditional `Pv31c` Trigger:**
  1. Implement `qrokkun_ai/v31`, verify unit tests (`<= 45` tests, `< 30s` on 1 CPU thread per `AGENTS.md` §2.2) + full pytest + `gpt-6.1-sol` code review, commit clean source + preregistration journal.
  2. Deploy immutable `git archive` snapshot to GB10 `/home/ovm-hms-bot/projects/qrokkun26/pv31_execution/<commit_sha>` and verify all file SHA-256 digests (including `qrokkun_ai/v25/reference.json`).
  3. Run nonformal full-cap CUDA smoke matching the archived `Pv30c` smoke invariants (`--nonformal --updates 1 --episodes 3 --cap 7200 --validation 2 --workers 2 --pin-cores on`) AND run `qrokkun_ai.v31.compare` in `--reference-dir` mode against a hash-verified local copy of `20261115-1756-pv30-final-nonformal-cuda-smoke/Pv30c` (`u0, u1`). Any execution/invariant error (`exit 3`) halts immediately for diagnosis.
  4. **Branch A (Expected — `u1` smoke achieves `WEIGHTPASS` `exit 0` against `Pv30c`):** Launch formal `Pv31m` (`u10`, `E20`, `workers=10`, `pin-cores=on`) into `/mnt/minion/88_minion_rw/projects/qrokkun26/runs/YYYYMMDD-HHMM-pv31-m-u10/Pv31m`, then run `qrokkun_ai.v31.compare` against a hash-verified local copy of archived `Pv30c` (`20261115-1759-pv30-cm-u10/Pv30c`). If formal `u10` achieves `WEIGHTPASS` (`exit 0`), `Pv31c` is unnecessary and skipped per owner instruction; only if formal `u10` completes cleanly with `WEIGHTFAIL` (`exit 2`, invariants matched) do we execute `Pv31c` to test same-version `Pv31c == Pv31m`.
  5. **Branch B (Fallback — if `u1` smoke completes cleanly with `WEIGHTFAIL` `exit 2` against `Pv30c`):** Launch full sequential `qrokkun_ai.v31.supervise` (`Pv31c -> Pv31m -> compare`) per owner instruction (`如果結果預計無法還原 Pv30c ,就開 Pv31c`).

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * Implement `qrokkun_ai/v31` with Tier-1 single-stream hoisted-feature PPO and single-big-core learner affinity without modifying `v21`–`v30` or `qrokkun_env`.
  * Verify `0-ULP` bitwise weight reproducibility (`u0..u10`, primary `u10`) and measure `ppo_s` and `total_s` speedup on GB10 under the formal `u10 / E20 / seed17 / cap7200` contract.
  * Obey `AGENTS.md` §2 anti-nitpicking and test-budget guardrails (`15–45` tests, `< 30s` single-thread CPU).
* **Out-of-Scope Non-Goals:**
  * Tier-2 `CUDA Graph` capture or `torch.compile`.
  * Full environment event tracing, 64-seed heldout strength evaluation, checkpoint selection, or promotion.
