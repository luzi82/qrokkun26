# Sv22: fixed Pv21 u400 opponent

Local implementation only. This round trains S against deterministic argmax P;
P is hash checked with the unchanged v21 loader, frozen and never optimized.
See preregistration [journal](../../docs/journal/2026-11-10-1794269002-sv22-contract.txt).

Parent launch after review and CUDA smoke (from repository root):

```bash
python -m qrokkun_ai.v22.supervise \
  --device cuda --threads 1 --seed 17 \
  --player-checkpoint /mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261109-1803-v21-exploratory/best_player_v21.pt \
  --expected-player-sha256 b8466bb13972de62003e16b0b570567a016d0dab4dd5667e66a3666974cfd5f8 \
  --out-dir artifacts/sv22/formal-seed17 \
  --episodes-per-update 16 --eval-every 10 --eval-seeds 16 --final-seeds 64 \
  --train-deadline-unix 1794305700 --deadline-unix 1794308400
```

Default cap is exactly 7200 frames, 120 seconds. Death takes precedence at the
cap. `--quick` uses 120 frames and marks artifacts nonformal. Never compare its
results to full-cap runs. `--updates` is an optional safety maximum. Deadlines
are absolute Unix timestamps; `--end-time` aliases the hard deadline. The hard
limit cannot exceed owner cutoff even if a later value is supplied. No launch
after hard deadline. Training stops at 18:15 HKT, leaving time for automatic final evaluation before 19:00. Actual final runtime is unmeasured;
parent must assess CUDA throughput and enforce cutoff independently.

Validation diagnostic, using the same manifest seed counts and quick setting:

```bash
python -m qrokkun_ai.v22.run validate --device cuda --out-dir artifacts/sv22/formal-seed17
python -m qrokkun_ai.v22.run final --device cuda --out-dir artifacts/sv22/formal-seed17
```

Final requires training-closed.json and verifies locked best.pt hash BEFORE
accessing heldout seeds. It claims the one-time test exclusively before access;
an interruption leaves partial per-mode results plus final-incomplete.json and
cannot be retried in that directory. Do not delete the claim to peek again.
No final seeds were used during implementation. Final compares stochastic learned
S to scripted Gv20 and frozen stochastic initial.pt using paired bootstrap 95%
intervals in seconds. Both upper bounds must be below zero to support learning
against both controls. Otherwise report inconclusive. Deterministic S is secondary
only; deterministic P/S can repeat trajectories when actual randomness agrees.
Validation controls are evaluated once; final controls once in their own window.

Architecture: separate 325->128->5 actor and 325->128->1 critic, instantaneous
public position/velocity/time + nearest64 relative bullet rows and explicit live
mask. No recurrent state, P features/logits/action, or action history. The five
raw Gaussian actions map to ray angle pi*(tanh(a0)+1), speed115+85*tanh(a1),
Gv20 learned_aim_delta(a2,(a3,a4)). Direction zero uses Gv20 +x fallback. Initial
aim scalar -3 keeps targets near P; actor head initially constant, with broad
angle and moderate speed exploration. Exact raw sampled coordinates and log
probabilities are retained; PPO never clips actions. Entropy is Gaussian raw
entropy with constant configurable coefficient, not transformed-action entropy.
Actor and critic train on CPU; S callbacks run only at events on CPU to avoid
individual GPU transfers. P inference is batched across all live simulations on
the selected device every tick. This is deliberately simple; S events are not
queued for GPU inference. Production schedule, field add, independent RNGs and
initial bullet movement delay are inherited without changes.

Rewards are negative decision intervals/7200. MC returns use gamma=lambda=1;
cap is a finite task boundary with zero continuation, not an infinite-horizon
bootstrap. The first frame0 decision receives all time until the next event.
Advantages normalize over each update. PPO clip .2, grad clip .5, Adam3e-4,
value coefficient .5, entropy .001, four epochs, minibatch256. Configurable CLI
values are recorded in manifest.json. Logs distinguish rollout and validation,
including deaths, censors, per-seed times, losses, gradient norm, parameter delta.

Atomic initial/latest/best/periodic snapshots retain optimizer and RNG plus
manifest and frozen P hashes. latest may contain a partial PPO update after a
signal; interrupted rollout data is discarded. **Automatic resume is deliberately
unsupported.** Checkpoints are preserved for inspection/recovery engineering;
there is no silent optimizer reset. Each new run/variant requires an empty output
directory. Final artifacts never overwrite a prior claimed heldout run. Training
and validation use independent action generators; evaluation cannot consume train
RNG. Signal/deadline checks occur each simulation tick, event, and minibatch;
one in-flight device operation cannot be preempted by Python.

Evidence lives in gitignored tmp/sv22/: red.log, green-core.log, green.log,
smoke.log, interrupted.log, regression.log. The first test was run RED before
implementation; subsequent integration tests were added after implementation.
The actual u400 CPU quick smoke completed one update (two episodes), finite
loss/gradient and nonzero S parameter delta; no formal learning claim. The signal
smoke interrupted a full-cap evaluation with no death and saved stopped:signal-15
latest including optimizer and RNG. No CUDA experiment was run by implementer.

Review fixes (2026-11-10): both deadline flags reject NaN and infinities.
Every guard check independently checks the owner hard cutoff, even if its
configured deadline has become corrupt. Formal runs require exactly 16 validation
seeds, 64 final seeds, and 7200 frames. Reduced counts require `--nonformal`
(full cap) or `--quick` (120 frames). Final conclusions are gated by the saved
manifest; nonformal runs and a one-pair interval cannot claim evidence of learning.

The three primary modes persist their individual results, comparisons and
conclusion to `final.json` before the deterministic diagnostic starts.
`final-diagnostic-incomplete.json` records an interrupted diagnostic separately;
it does not mark completed primary results incomplete. The exclusive final claim
still forbids reruns, including after interruptions.

`validate` can run during training once `best.pt` exists: it opens one immutable
snapshot of the atomically replaced file and uses only validation seeds.
Only `final` requires `training-closed.json` and its best hash.
Supply the manifest's quick/nonformal flags and seed counts to either CLI.
Diagnostics write `validate-status.json` / `final-status.json` so live validation
does not overwrite the trainer's `status.json`.

Atomic `status.json` reports baseline, rollout, PPO, validation and terminal states,
wallclock, elapsed seconds, completed update and attempted update. Checkpoints
also distinguish attempted/partial from completed updates. One fsynced
`progress.jsonl` record per completed PPO update carries wallclock, elapsed,
update, loss, rollout and validation (null when absent or interrupted).
Interrupted rollout episodes are discarded; latest parameters/optimizer/RNG
are preserved, with no heldout access. A partial PPO update may have changed
parameters but never increments the completed update counter.

Review evidence: `tmp/sv22/review-fixes/`, including strict regression RED,
GREEN suite and actual u400 CPU smoke logs. Smoke flags:
`--device cpu --threads 1 --seed 17 --quick --updates 1 --episodes-per-update 2 --eval-every 1 --eval-seeds 2 --final-seeds 1`.
The final count is configuration only: smoke never invokes final evaluation.

Launch preparation: the parent deploys an isolated immutable snapshot, not the
existing dirty remote clone. Run the supervisor command above remotely on GB10
under a managed parent SSH session. No nohup, disown, setsid, or resume. The
supervisor runs train then final only after successful training exit and a hash
verified closed best checkpoint; manifest arguments and counts are reused.
It exclusively claims `<out-dir>.launcher/`; an existing claim is inspected
idempotently without another child. Occupied training directories are refused.
`supervisor.json`, `train.stdout`, and `final.stdout` live in that sibling launcher
directory (the trainer requires an empty output directory). Logs are fsynced at
child exit; status reports PID, live child PID, stage, exit, time and log paths.

The independent wall and monotonic watchdog checks at most every 0.2 seconds,
starts TERM 10 seconds before the supplied hard deadline (at latest 18:59:50 HKT),
waits at most 5 seconds, then kills and reaps its direct child. SIGINT/SIGTERM
also clean up that child. This is an audited direct Python workload with no
subprocesses; it is not containment against malicious descendants. Nonfinite,
late, or inverted deadlines fail closed. No final launch follows failed training
or missing closure/best. No automatic resume or final retry is supported.

Evaluation appends and fsyncs each completed seed to `*-episodes.jsonl`.
`*-completion.json` explicitly lists completed and unfinished seeds; unfinished
seeds have no invented durations. Interrupted primary final writes terminal
`primary_incomplete` with mode and reason to `final-status.json`; earlier completed
modes remain intact, and no partial mode is compared. The one-time claim remains.
Launch-preparation tests use synthetic mocked final records and disposable CPU
children only; evidence is in `tmp/sv22/launch-prep/`.
