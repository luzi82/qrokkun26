# Pv32m: owner-authorized warm-start baseline

Pv32m starts BOTH actor and critic from the original Pv21 u400 weights, with fresh Adam and a new lineage at u0. It directly imports `v31.core.train`, `EpisodePool`, `assemble_batch` and optimizers. There is no random initialization arm, resume, retry, restart, tuning or score-based stopping. The synthetic option is exclusively a clearly marked nonformal CPU fixture.

The unchanged Pv31m backend is `cuda_hoisted_single_stream`: feature hoisting, deferred host synchronization, no dual streams or host dispatch threads. CPU workers use batch 1 with the original P/S RNG offsets and canonical episode merge. Ten persistent workers are placed first using v31 frequency-ranked eligible cores; v31 then pins the learner to its selected big core (CPU 5 on the approved GB10 topology). CUDA is deterministic, without AMP, TF32, compile or CUDA Graphs. Architecture (hidden 472, obs 84), teacher, actions, rewards, clearance and all PPO hyperparameters remain inherited. S is the locked frozen stochastic Sv24 best u980.

Formal configuration is u1000, E20, cap7200, seed17, workers10. Update u uses seeds `100000+(u-1)*20` through `100000+u*20-1`. Validation uses original seeds 600000..600015 at u0 and every ten updates through u1000. P trains stochastically and evaluates by deterministic argmax; S is always stochastic. Independent episode generators preserve training RNG. On the real formal GB10 run only, u0 must match the locked v25 per-seed frames/death/censor reference. A mismatch aborts; weights are never changed to force agreement.

Selection uses integer total frames, strict greater only; u0 is eligible and ties retain the earlier update. `best.json`, `initial.json`, and `latest.json` point to six-digit checkpoints. There are 101 checkpoints and evaluations, and 1000 update JSONL records containing episode summaries. Every checkpoint retains both full optimizer states and permutation/global RNG metadata for provenance, without offering resume. The terminal u1000 checkpoint is always saved. Status reports best and latest validated updates separately. Disk space is checked at startup and checkpoint boundaries, never per frame.

`closed.json` locks the manifest and initial/best/terminal checkpoint hashes after all training, terminal validation and pool shutdown succeed. Final is exclusively claimed before evaluation and can run only once, after formal closure. It loads and synchronizes original, selected best, and terminal CPU weights into the current v31 pool, ten workers, with the validation scene/S mapping. Each arm completes exactly seeds 800000..800063, in bounded batches of 16. If best equals terminal, terminal explicitly references the best result. Each result records checkpoint file/state hashes, update, modes, seed, frames, death/censor and cap. Primary is paired best minus original seconds; secondary is terminal minus original. Bootstrap uses 10000 samples, seed17, 95% CI, descriptive only. This window is the **previously used Pv25 common benchmark, NOT fresh heldout**. No other model arm or window is implemented.

Run from an extracted source archive without Git; supply its final 40-hex source commit explicitly. Source fingerprint covers production Python and exact `v25/reference.json`, excludes tests/cache/documents, and is checked at closure. Immutable P/S files and their metadata use the existing v25 loader. Output directories and launcher/final claims are exclusive and never reused.

```bash
# Formal, only after independent authorization and deployment:
python -m qrokkun_ai.v32.supervise --out-dir /approved/new-run --source-commit <final-40-hex>
# Separate stages, if supervision is performed externally:
python -m qrokkun_ai.v32.run train --out-dir /approved/new-run --source-commit <final-40-hex>
python -m qrokkun_ai.v32.run final --out-dir /approved/new-run --source-commit <same-final-40-hex>
# Locked-input nonformal smoke, never final:
python -m qrokkun_ai.v32.supervise --out-dir /approved/new-smoke --source-commit <final-40-hex> --nonformal --device cpu --updates 1 --episodes 3 --workers 2 --validation 2 --cap 7200 --pin-cores off
```

The supervisor positively owns its direct child, forwards cooperative stop with bounded escalation/reaping, requires train exit zero and full formal closure before final, and rechecks stop after preflight before spawn. Observed errors, owner signals and child exits are recorded; failures never retry or launch a successor. Existing v31 pool lifecycle owns worker/shared-memory cleanup; v32 also restores learner affinity and thread count and reports exceptions it observes during cleanup. Stage claims prevent duplicates from overwriting owner status.

This is one training seed. Future initialization comparisons must match update/episode exposure, not claim equal frame exposure. Pv25 E12 is historical reference only; Pv32 E20 and the newer implementation are not strict Pv25 training replication. Performance speed is not agent strength. No minimum improvement or legacy survival threshold gate applies. Future random-weight work requires separate authorization.
