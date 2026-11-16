# Pv35m input64 (implementation; formal run not started)

The actor and critic each receive 344 features: player[6], raw filtered bullets[64,4], live mask[64], clearance[9], safe flags[9]. The unchanged Pv34 harmless predicate runs before stable distance ordering. Teacher and engineered clearance retain the first 12 bullets. `top_k=12` is the inherited legacy teacher argument, explicitly described in architecture metadata; network capacity is `policy_input_top_k=64`. Both heads use hidden472 and the unchanged decoupled Tanh architecture. Spawn workers install scoped v35 bindings; bindings restore on exit and exception. Pv31 numerical training backend is unchanged; pool cleanup inherits v32.

Initialization uses the RNG isolated v33 seed17 helper to reconstruct Pv34 random u0, never trained best/terminal P. The paired embedding copies old player/bullet columns 0:54 to 0:54, mask54:66 to262:274, clearance66:84 to326:344, for both first layers. Extra bullet54:262 and mask274:326 columns are zero; all later weights and biases are copied. This is explicitly an initialization adaptation from old fan_in84, not native fan_in344 random initialization. New columns can learn immediately. Expanded GEMM may round differently; mapped weights are exact but logits and trajectories are not claimed bit identical. Adam is fresh. The inherited player checkpoint CLI argument is unused legacy provenance and is never loaded. Sv24u980 file/state/lineage verification remains unchanged.

`control-reference.json` pins the actual closed Pv34 run `20261116-1219-pv34m-threat-filter-u1000`, its 101 validation evaluations, u0/best210/terminal1000 and saved common64 final results. Runtime checks validate protocol and completed records, including the random u0 identity. No Pv34 episodes are rerun. No optimizer resume, seed search, new final window, accuracy or promotion claim is introduced.

Formal protocol remains u1000/E20/cap7200/train17/init17, 10 pinned CPU workers, frozen S, validation600000..600015 every10 (also u0), earliest best primary and terminal secondary. Final uses the previously seen 800000..800063 window once, reporting descriptive paired differences. Tests use synthetic data only for runtime wiring, not research outcomes.

Parent commands after independent review/commit, using a committed archive with required runtime JSON included and no `.git` needed:

```bash
# SOURCE_COMMIT must be the actual reviewed 40-character commit, even in git archives.
# RUN is a new approved run directory; use separate smoke/formal directories.
python -m qrokkun_ai.v35.supervise --out-dir "$RUN" --source-commit "$SOURCE_COMMIT" --device cuda --nonformal --updates 1 --episodes 3 --workers 2 --pin-cores off --cap 7200 --validation 2
python -m qrokkun_ai.v35.supervise --out-dir "$FORMAL_RUN" --source-commit "$SOURCE_COMMIT" --device cuda
```

Nonformal has no final stage. Formal supervisor launches dedicated v35 train then final, with no retries. All observed failures and owner stops abort successors and are recorded, including cleanup failures. Public final preflight enters the v35 fingerprint context. Version-local fingerprint traverses production Python dependencies and includes v25/reference.json and v35/control-reference.json; excludes tests/docs. No global dependencies added.

Acceptance command (approved environment TMPDIR, caches outside source):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GIT_CEILING_DIRECTORIES="$TMPDIR" PYTHONPATH=. .venv/bin/python -m pytest qrokkun_ai/v35/tests --ignore=tmp --import-mode=importlib -p no:cacheprovider --basetemp="$TMPDIR/pv35-continuation-tests" -q
```
