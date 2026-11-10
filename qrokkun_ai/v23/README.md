# Sv23 owner contract and review handoff

Sv23 is a new policy, initialized from scratch. Production P is deterministic,
strictly hash-locked Pv21 u400. Sv22 u880 is loaded only as a frozen comparator;
its original environment mapping and seed+7,000,000 raw Gaussian sampling remain
unchanged. No Sv22 Guard, runner, supervisor, optimizer, or training state is used.

The public observation remains 325 components. Separate actor and critic each
have a 128-unit hidden layer. The actor outputs **eight components**: five means
(birth x/y, speed raw, unbounded aim dx/dy pixels), then three scene-dependent
scale logits. The executable action has **three random degrees of freedom**:
wrapped birth bearing, actual physical speed, wrapped firing bearing.

Birth uses the normalized vector; exact zero means east. Max-absolute rescaling
preserves even subnormal nonzero directions. The atan2 derivative alone uses a
radius floor of 1e-12 to avoid float32 overflow at the singularity; there is no
forward fallback sector. Initial mean bias is (1,0), speed raw zero, aim (0,0).
Only scale rows have small random output weights so their initial values already
vary by scene. Initial scales are approximately (3 rad, 50 px/s, .10 rad); 3 rad
is near uniform on the circle, not exactly uniform. Scale sigmoid bounds are
(.01..pi rad, .5..170 px/s, .01..pi rad), explicit numerical engineering bounds.

Speed is Normal with mean 115+85*tanh(raw), conditioned on [30,200] px/s. Sampling
uses float64 inverse CDF with an open 52-bit uniform to avoid inverse-CDF endpoints;
velocity is never clamped. The differentiable CDF normalizer is included in PPO.
Each angular density sums normal images -5..5 using float64 logsumexp after
wrapping the difference. At sigma <= pi, the first omitted image is at least
11 sigma away; omitted integrated mass is below 4e-28. This is seam-continuous.

The aim target is the pre-movement player position plus direct pixel offsets.
For the **stored sampled birth bearing**, production ray_position defines spawn;
spawn-to-target bearing defines conditional firing mean. Noise is added to that
bearing at spawn. Zero offset therefore still permits firing exploration.
Production's aim-at-spawn <1e-12 fallback is +x. PPO recomputes this conditional
mean from the stored birth bearing and scores all three physical quantities.
Rollout old likelihood and stored actions use float64. Entropy coefficient is
zero: no Gaussian surrogate is called physical entropy.

Gv20 inheritance supplies the exact scripted schedule, frame-zero decision,
pre-movement scene, radius 2.6, constant velocity, reset bullet first-tick delay,
and field direct-aim rule using previous 60-frame window of S speed history excluding the same
frame. Only S spawning is overridden. All active episodes batch frozen-P device
inference each 60 Hz tick; S sampling and PPO run on CPU with torch threads=1.
MC returns have gamma=lambda=1 and reward -elapsed frames/7200, through true death
or 7,200-frame cap. No extra S event is created on a P movement.

The owner's preregistration, supplied before implementation or trials, is fixed:
training seeds start at 2,000,000; validation 600000..600015; final 700000..700063.
Training uses a dedicated seed-17 generator; Sv23 evaluation uses one generator
per seed initialized with seed+23,000,000. Validation controls (scripted Gv20,
saved initial Sv23, Sv22 u880) run once and never consume training RNG. Stochastic
S is primary. Minimum validation survival selects best, with ties retaining the
earlier checkpoint. Deterministic S is a secondary diagnostic. Paired bootstrap
reports learned-minus-each-control mean survival and 95% CIs; reduced runs always
remain inconclusive. Rollout survival is separately reported on training seeds.
Bounded 24-bin birth histograms and three-scale min/mean/max accompany updates.

Formal default is exactly 1,000 completed PPO updates, 16 episodes/update, full
7,200-frame cap, validation at update 1 and every 10 including 1000. No default
wall-clock deadline exists. An optional explicit --deadline-unix is supported.
Formal runs require seed 17; alternative seeds and reduced counts require --nonformal or --quick; --quick caps at 120 frames.
There is no implicit or explicit resume. Output must be empty. initial/latest,
best, validation-periodic and update-1000 snapshots contain model, Adam, RNG,
architecture, config, hashes, and attempted/completed counters. On interruption,
interrupted.pt captures partial work and latest.pt retains a completed boundary.
No training-closed lock is written on interruption; errors propagate nonzero.

Use the supervisor for automatic train -> one-time final. It owns an exclusive
launcher directory, runs direct children, forwards signals, kills/reaps after a
10-second grace, and preserves child failure. Owner-stop prevents final launch.
Only clean u1000 permits final. Closure hash-locks best and the separate u1000
snapshot. Final claims exclusively before seed use, never reruns an interrupted
claim, writes completed seed records and mode progress as it proceeds, persists
primary results before deterministic diagnostics, and never selects using final.
This version evaluates the locked best only; optional secondary u1000 evaluation
is omitted. Use plain managed SSH for deployment; the implementation task did
not deploy or use SSH, GPU, training host, other agents, commits, pushes, or PRs.

## Commands for the parent

CPU environment used here:
`/storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python`.
Run from repository root. The GPU interpreter and immutable output path below
must be chosen by the parent after independent review. CUDA commands were **not
executed** by this implementation task.

```bash
# Executed CPU smoke, NONFORMAL, no final seeds:
OMP_NUM_THREADS=1 /storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python -m qrokkun_ai.v23.run train --out-dir tmp/sv23/cpu-quick-reviewed --device cpu --player-checkpoint artifacts/sv22-media-inputs/pv21-u400.pt --comparator-checkpoint artifacts/sv22-media-inputs/sv22-u880.pt --quick --updates 1 --episodes-per-update 2 --eval-seeds 2

# Parent archive preparation: fill the actual reviewed commit containing Sv23.
# Run in the source clone; use a fresh absolute deployment directory.
SOURCE_COMMIT=REPLACE_WITH_ACTUAL_REVIEWED_40_HEX_COMMIT
ARCHIVE_DIR=/absolute/path/to/fresh/sv23-archive
mkdir -p "$ARCHIVE_DIR"
git archive "$SOURCE_COMMIT" | tar -x -C "$ARCHIVE_DIR"
cd "$ARCHIVE_DIR"
# Supply hash-locked artifacts at these paths before either CUDA command:
# artifacts/sv22-media-inputs/pv21-u400.pt
# artifacts/sv22-media-inputs/sv22-u880.pt

# Parent: full-cap CUDA smoke (omit --quick; no final at one update):
python -m qrokkun_ai.v23.run train --source-commit "$SOURCE_COMMIT" --out-dir runs/sv23-cuda-fullcap-smoke --device cuda --nonformal --updates 1 --episodes-per-update 2 --eval-seeds 2

# Parent: formal, fixed 1000 updates then automatic final; new empty paths:
python -m qrokkun_ai.v23.supervise --source-commit "$SOURCE_COMMIT" --out-dir runs/sv23-u1000-immutable --device cuda --updates 1000 --episodes-per-update 16 --eval-seeds 16 --final-seeds 64 --seed 17

# Executed test commands (logs in tmp/sv23):
OMP_NUM_THREADS=1 /storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python -m pytest qrokkun_ai/v23/tests -q
OMP_NUM_THREADS=1 GODOT_BIN=/storage/hermes/tools/godot/godot /storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python -m pytest -q
OMP_NUM_THREADS=1 GODOT_BIN=/storage/hermes/tools/godot/godot /storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python -m pytest -q --ignore=qrokkun_ai/v23
```

For deployment, supply both local artifacts at the listed relative paths or pass
--player-checkpoint and --comparator-checkpoint. Every command verifies their
file hashes, and P additionally uses the production loader's state hash/schema.
--source-commit accepts only a literal 40-hex SHA and bypasses Git entirely.
The supervisor forwards it unchanged to train and final; final requires the same
configuration flags as train. Without the flag, local development can discover
Git HEAD; unavailable Git/HEAD fails with an actionable --source-commit error.
The manifest records the supplied source commit, its original config flag, and
exact source/production file hashes. A dirty local smoke may name current HEAD:
the file hashes record the actual working content; the SHA does not assert that
uncommitted content belongs to that commit. Archive deployment must name the
actual reviewed commit containing the archived files; never invent a SHA.
The owner task records under /storage/hermes/home/dev_gpt_1/sv23-u1000 were read
only. No accessible approved continuous-spawner-policy SKILL.md was located.

## Evidence

Every requested vertical started with a failing test and an actual RED run.
Logs 01..09 contain birth, wrapped density, truncated speed, state scales,
conditional likelihood, inherited environment, real P/PPO, lifecycle, supervisor
RED/GREEN pairs. Tests cover quadrature, moments, seam, scale dependence,
old=new ratio 1, exact sample/recompute, sigma edges, target geometry, finite
near-zero gradients, frozen P identity, real Adam change, checkpoint rejection,
exclusive launch, and signal reaping. Further RED/GREEN pairs 11,14,15 cover
inverse-CDF endpoints, subnormal float32 gradients, and primary-result retention
when a mocked final diagnostic is interrupted. Exact1000 and final-lifecycle
fixtures are explicitly synthetic unit tests, not policy trials. Actual final
seeds were never run. No learning or performance claim follows from the smoke.

Final test totals and smoke metrics are recorded in the new contract journal.
The initial full run hit a pytest test_core module-name collision; adding the
Sv23 tests package resolved it. Historical Sv22 tests have time-dependent failures
once its TRAIN_HARD cutoff expires; their source and constants remain untouched.

Latest evidence: **17 Sv23 tests passed**. Full project: **141 passed, 12 failed**;
the same 12 historical Sv22 deadline failures appear in the isolated 136-test
baseline (124 passed, 12 failed). Godot regressions passed. Logs:
`tmp/sv23/new-tests-final-v2.log`, `full-project-final-v2.log`, and
`baseline136-tests.log`. The actual reviewed quick smoke completed update 1
with finite loss -0.007297011, gradient norm 0.887229681, parameter delta
0.100203134, and frozen P with no gradients. Both 120-frame episodes were
censored; all results are nonformal. No actual final evaluation ran.

Launch-fix verification: **27 Sv23 tests passed**; project regression **151 passed,
12 deselected** (only the previously proved expired Sv22 tests, not a full-suite
all-green claim). RED/GREEN, exact commands, exclusion list, and actual CPU
one-update smoke with explicit current-HEAD provenance are saved under
`tmp/sv23/launch-fixes/`. Formal seed 18 is rejected; explicit-source resolution
is tested in a temporary cwd without `.git`, with Git invocation forbidden.

Supervisor cancellation is cooperative: SIGTERM/SIGINT requests and the optional
explicit deadline are checked at stage entry, after preflight and argument
construction immediately before Popen, and while monitoring the child. A request
observed during closure reading or best-checkpoint hashing persists `stopped`
and returns without spawning final. These boundaries do not promise hard real-time
cancellation or eliminate every signal/spawn race. Formal runs remain seed 17,
1000 updates, with no dated deadline.
