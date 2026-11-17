Pv37m is the Gv36 ordinary PPO control. Pv38m adds the previously validated fixed death-search teaching pipeline. The model remains Pv35 input64 / teacher12 / clearance12 / hidden472. Historical Pv37-control/Pv37-search trial results keep their original names.

`formal-contract.json` supersedes the historical v37 trial contract for future execution. Both arms start the same untrained seed17 embedding and fresh Adam against the same strictly loaded Sv24 u980. Training and initialization seed are recorded separately. Formal defaults are locked; no resume, retry, tuning or early stopping is supported. Extra actor steps and search CPU time mean equal episode/update budgets are not compute matched.

Run from an immutable source tree containing both versions and both required JSON files. In a checkout, the supplied full commit must contain the executed bytes; an uncommitted implementation intentionally fails archive verification. In a `.git`-free deployment the explicit commit is provenance supplied by the operator and the dependency digest identifies the actual bytes. Parent must verify the committed archive, runtime pairing and CUDA smoke before launch.

Use the actual full implementation commit for `SOURCE_COMMIT`, the strict Sv24 u980 input for `SPAWNER`, and unused output directories. The following use the project Python on the execution machine:

```bash
python -m qrokkun_ai.v37.run train --out-dir "$CONTROL" --source-commit "$SOURCE_COMMIT" --spawner-checkpoint "$SPAWNER"
python -m qrokkun_ai.v37.run final --out-dir "$CONTROL" --source-commit "$SOURCE_COMMIT" --spawner-checkpoint "$SPAWNER"
python -m qrokkun_ai.v38.run train --out-dir "$SEARCH" --control-dir "$CONTROL" --source-commit "$SOURCE_COMMIT" --spawner-checkpoint "$SPAWNER"
python -m qrokkun_ai.v38.run final --out-dir "$SEARCH" --control-dir "$CONTROL" --source-commit "$SOURCE_COMMIT" --spawner-checkpoint "$SPAWNER"
```

Alternatively launch one exclusive sequential pair:

```bash
python -m qrokkun_ai.v38.supervise --out-dir "$PAIR" --source-commit "$SOURCE_COMMIT" --spawner-checkpoint "$SPAWNER"
```

Its order is control train → control final → search train → search final/comparison. It owns the newly created pair directory, `Pv37/`, `Pv38/`, stage stdout logs and `supervisor.json` with PID and child exits. Signal its supervisor PID to request cooperative stop. Any observed child failure or owner stop prevents successors; TERM and KILL cleanup waits are bounded. No automatic retry is performed. An existing pair directory is rejected before status writing.

A separate, unused nonformal pair path exercises only two train stages and never accesses final seeds:

```bash
python -m qrokkun_ai.v38.supervise --out-dir "$SMOKE_PAIR" --source-commit "$SOURCE_COMMIT" --spawner-checkpoint "$SPAWNER" --nonformal --updates 2 --episodes 3 --workers 2 --validation 2 --cap 7200 --seed 17 --device cpu --pin-cores off
```

Every module supports `--help`. Formal defaults are 1000 updates, E20, cap7200, seed17, hidden472, W10/pinning on, validation16 every10 including u0. Checkpoint and validation schedules contain 101 records. Earliest maximum integer total frames selects BEST; terminal1000 is secondary. Search pins the completed new control manifest, closure, status, checkpoint identities and final results before training. Per-update selected demos remain on disk; the free-space estimate includes their conservative uncompressed size plus checkpoints.

Final exclusively claims each run, evaluates BEST and TERMINAL once (explicit reuse if identical) with P argmax / S stochastic, cap7200, common previously seen seeds800000..800063 and `core.S_VAL_OFFSET` (24000000). It never enables search. Pv38 compares against saved Pv37 results without rerunning control. Positive candidate-minus-control seconds favor Pv38. Bootstrap is descriptive, 10000 paired resamples, seed17; a CI containing zero establishes neither improvement nor equivalence. There is one initialization/training seed and no minimum gain or promotion rule.

Formal records: [implementation](../../docs/journal/2026-11-17-1794873702-pv37-pv38-implementation.txt), [preregistration](../../docs/journal/2026-11-17-1794873702-pv37-pv38-prereg.txt).

The completed control is pinned before search begins: all 101 evaluation JSONs and 101 checkpoints, four pointers, updates log, manifest/closure/status and all final records (214 files). Candidate final preflight requires exact pin readback. Death on frame7200 remains a completed death; only cap survival is censored.
