# Pv26c／Pv26t：u20 執行等價診斷

本套件只回答相同輸入下 serial／thread 執行是否逐位元相同，以及首次可觀測差異在哪裡。名稱固定為 **Pv26c**（control，1 worker）與 **Pv26t**（thread，預設 4 workers）；不是強度測試、選模或 promotion。Pv27 隨機初始化與 B filter 均未實作。

Pv21 u400 actor／critic warm start；兩個 Adam 全新。固定 stochastic Sv24 best u980，CPU、eval、requires_grad=False，callback 內 no_grad。來源檔案／state SHA 與 update／architecture metadata 由 v25 載入器及 v26 額外 S state 檢查鎖定。CLI 可以指定 GB10 本地副本，不能更換 hash。

P 使用每個 episode 專屬的 `torch.Generator(device=P.device)`，seed=`episode_seed+26000000`；每次對中央 forward 所得的單列 probs 呼叫 explicit-generator multinomial。這是新 mapping，並非歷史 Pv25 replay。S train seed=`episode_seed+25000000`，validation=`episode_seed+24000000`。PPO permutation 另用 device-local generator，seed=`27000000+17`。Python／NumPy legacy／torch CPU／全部 CUDA global states 都記錄；沒有建立 NumPy Generator，因此 registry 為空。未抽樣的環境 RNG 在 episode_end 明列 unused。

兩臂用同一 collector、相同原始 active batch shape／順序，先中央 P forward／teacher／action，再只將個別 env.step 派給 executor。每 frame 等全部完成後按 episode index 合併；沒有 episode refill、policy async 或完成順序編號。c 不建立 executor。每批 12 episodes 完成才 PPO，原 v21 觀察、clearance features、teacher、reward、GAE 與 PPO 數值計算保留。每個環境只有一個 worker 使用；torch threads=1 在啟動時設定。

正式鎖定 u20、12 episodes/update、7200 cap、seed17；validation 為 u0/u10/u20，固定 600000..600015，deterministic P／stochastic S，保留 fork_rng。無 heldout、best 選擇或 final promotion。訓練 seeds 從 100000 依 update／episode 順序遞增。超參數為 gamma .9995、lambda .999、reward scale .05、actor lr 3e-4、critic lr 1e-3、Adam eps 1e-8、PPO 4 epochs／512 minibatch／clip .2／maxgrad1、DAgger1／PPO .3／entropy .005、hard/soft .65/.35。

## 追蹤與完整性

`trace.py` 使用 8 MiB 未壓縮事件門檻的 gzip shards，以及 SQLite 壓縮 CAS；沒有每 frame 一個檔案，也不把完整 u20 raw trace 放進 RAM。tensor 使用 contiguous uint8 view，Python float 使用原始 IEEE754 binary64，NumPy 包含 dtype／shape（structured array 保留欄位描述）。SHA 只作索引；實際 bytes 全部保存並逐引用驗證。±0、NaN payload 保留。SQLite index 在磁碟，未變 tensors 可共用 blob。每事件 commit，正常完成才產生 closed.json；中斷保留 shards／CAS，不偽造完整閉合。

reset 記錄 frame0 既有子彈、全部 RNG 初態；P frame 記錄 active IDs、encoded observation、實際 features、raw head logits／distribution logits／probs／value、teacher、action／logp、全域 RNG 與逐 episode 抽樣的 pre／post／output。G 記錄完整前後 player／clock／bullets／histories／RNG、S 公開 observation／means／std／三次原始抽樣／physical action／logp、出生／offscreen 移除／collision、reward／done。子彈 ID 為 episode-local birth ordinal，origin 0=S、1=field；structured bullet row 的欄位名稱與 dtype 在 blob descriptor 中。死亡時子彈不會被移除，故不將死亡猜成 removal。

訓練記錄完整 rollout／flattened 順序、batch（含 GAE／returns／advantages）、每 epoch permutation、每 minibatch indices，actor／critic 各自 clip 前後 gradient、step 前後 model／Adam／global RNG／training RNG，以及 losses。u0 與每個 completed update 都有 model／optimizers／全部 global RNG checkpoint；episode RNG 已於 episode_end 全量閉合且下批重新建立。checkpoint JSON 與 trace 的 checkpoint event 必須完全相符。

時間、thread ID、worker labels 僅在 audit／manifest，科學比較不含它們。trace I/O 沒有抽 RNG。**所有時間都含追蹤成本，不能宣稱未加儀器的加速。** GPU utilization 由 parent 外部採集，本套件不安裝 sampler。

## Scientific completeness operating contract

Each arm is validated independently before alignment. Scientific event payloads use
closed field sets and mandatory raw tensor references with exact dtype and shape.
P and bootstrap dimensions follow active episode count; observations are player6,
bullets64×4 and pad64, features84 (top12), actions9. Rollouts retain every collected
sample in episode/frame order, including rewards, teacher outputs and bootstrap;
the batch must match that rollout and its inherited GAE/return calculation. Actor
snapshots require all five loss components, teacher probabilities, logits and
values. Python float positions retain binary64 bits; tensor losses retain scalar
float32 references. Empty structured environment bullet arrays, initial None
gradients, deterministic validation with no P draws, and the unused empty NumPy
Generator registry are valid. One-sample explained variance may be NaN under the
inherited PPO calculation; its warning must remain visible.

A collection with cap survivors requires exactly one bootstrap before episode_end,
in active episode order, with final encoded observations and float-converted values
bound to episode_end and training rollouts. All-death collections have no bootstrap;
death on the cap frame takes precedence over censoring.

The execution grammar requires initial global/opponent/checkpoint/validation,
training resets, each P then its canonical episode-index G barrier, optional
bootstrap and episode ends, rollouts, batch, four ordered epochs of permutation and
contiguous minibatches. Each minibatch requires actor preclip/postclip/prestep/
poststep then critic preclip/postclip/prestep/poststep. Checkpoint follows the full
update, then scheduled validation, update_end and finally global_final. Permutation
and minibatch indices must agree. Thread completion times are audit data. Stable
key alignment remains necessary across legitimate differing death frames; missing
mandatory data or impossible execution order is exit 3 even if both arms have the
same resealed defect. This is an integrity check, not remote execution attestation.
No new trace fields or manifest version are introduced by these validation rules.

## Parent 執行命令

本輪只執行 bounded synthetic CPU，未讀真實 checkpoint 進行 EP、未啟動 GPU、未提交。parent 負責獨立 readiness review、修正後 fresh review、local commit、GB10 部署／正式 launch／比較。先把 `SOURCE` 設為 parent 實際審核後的 40-hex commit；archive 無 git 時同樣必須明填，另記實際 source-tree digest。不要以目前 base SHA 代表尚未 commit 的新增程式。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GIT_CEILING_DIRECTORIES="$TMPDIR"
PY=/storage/hermes/project/qrokkun26/.venv/bin/python
SOURCE=<actual-parent-reviewed-40-hex-commit>
$PY -m pytest qrokkun_ai/v26/tests --ignore=tmp -q
$PY -m pytest --ignore=tmp -q
$PY -m qrokkun_ai.v26.run --arm Pv26c --out-dir unused --trace-dir unused-trace --source-commit "$SOURCE" --estimate
```

估算只用架構／上限整數計算，不建立模型、不讀 checkpoint、不碰 GPU。包含全部 birth 上界與直線最低速度對應的 cull 存活上界、完整邊界 snapshots、optimizer CAS 中確定重用的 tensors、metadata margin；不抵扣壓縮收益。兩臂預算是單臂兩倍。啟動時重新檢查實際 free space；執行中低於 1 GiB watermark 即 fail closed，不能 drop trace 或降級。`--estimate` 不建立輸出目錄。

有界 smoke（路徑必須不存在；證據放 ignored tmp，TMPDIR 保持實際批准環境值，勿改成 /tmp）：

```bash
$PY -m qrokkun_ai.v26.supervise \
  --out-dir tmp/pv26-implementation/parent-smoke \
  --trace-dir tmp/pv26-implementation/parent-smoke-traces \
  --source-commit "$SOURCE" --nonformal --synthetic --device cpu \
  --updates 1 --episodes 2 --cap 80 --validation 2 --hidden 8
```

正式命令由 parent 在 GB10 的 CUDA Python 執行；先確認 NVMe／NAS 的**當下**容量。既知 2.8 TB NVMe／2 TB NAS 並不取代 preflight。範例：

```bash
PY=/path/to/GB10/cuda-venv/bin/python
SOURCE=<parent-reviewed-40-hex-commit>
$PY -m qrokkun_ai.v26.supervise \
  --out-dir /mnt/minion/88_minion_rw/projects/qrokkun26/runs/YYYYMMDD-HHMM-pv26-u20 \
  --trace-dir /path/to/local-NVMe/pv26_trace_storage/run \
  --source-commit "$SOURCE" --device cuda \
  --player-checkpoint /path/to/local-copy/pv21-u400.pt \
  --spawner-checkpoint /path/to/local-copy/sv24-u980.pt
```

預設不更改 dtype／precision 或 CUDA kernel flags；可選 `--deterministic` 同時在兩臂設 deterministic algorithms、CUBLAS_WORKSPACE_CONFIG=:4096:8 與 cuDNN deterministic／benchmark flags，全部寫入 manifest。沒有 CUDA 時直接失敗，不偽裝成 CUDA CPU run。

supervisor exclusive claim 後按 c→t→compare 執行，保存每個 child 的真實 exitcode／completed counters。差異不造成重跑或提前終止 u20；compare 只在兩臂都正常完成後執行。owner SIGINT／SIGTERM 會停止、reap 當前 direct child，不再啟動下一臂或 comparator。已佔用輸出不覆寫、不 resume。checkpoint 載入錯誤、runtime error、nonfinite、容量不足會失敗且保存已有追蹤。

```bash
$PY -m qrokkun_ai.v26.compare /path/to/Pv26c /path/to/Pv26t \
  --out-report /path/to/comparison.json
```

exit 0=完整且 full trace 相同，2=完整但不同，3=不完整／損毀／semantic identity 不符。報告另外列出 final 每個 named actor／critic tensor 的 raw memcmp、weights_bitwise_equal、full_trace_equal、每 update 首差異，以及 raw byte／tensor index／數值差（數值僅供解釋）。只以 actor＋critic 全部 raw bits 相同判定權重成功；trace 若不同會獨立揭示。不以 allclose 判成功。首差異是觀察到的差異，不保證已證明最終因果。事件用穩定 kind／update／episode／frame／optstep 辨識，不按 walltime 對齊；結構缺失直接報結構差異／不完整。

live manifest 指向真正 NVMe trace 路徑，`trace_durable_bundle=false`。parent 必須完成兩臂重 trace 的 NAS copy，逐檔 source hash 與 NAS readback hash 相符並重新驗證全部 blobs／closures 後，才可另立 durable bundle 記錄。不可只複製 NAS metadata 就聲稱完整 durable；若離線比較需改 trace 路徑，應保存原 manifest，建立明確的 relocation 記錄及重做 closure 身份鏈，不手改 manifest 後沿用舊 hash。

## 本地證據（2026-11-12）

證據位於 ignored `tmp/pv26-implementation/`，不納 Git。01 trace、02 RNG kernel、03 真實 CPU training、04 run／compare、05 supervisor、09 reset 各有 RED/GREEN logs；擴充測試與 smoke／容量估算另有保存。所有結果只對實際完成的 bounded fixture 成立，未產生正式 u20 結果。最終精確測試數與未解限制見事前合約的實作補記。

可選 NAS identity／metadata 整合測試：`PV25_TEST_REAL_INPUTS=1 $PY -m pytest qrokkun_ai/v26/tests/test_run_compare.py --ignore=tmp -q`。只讀 hash／checkpoint metadata，不執行真實 episode；未 opt-in 或 NAS 未掛載則 skip。

最終本輪結果：完整 repo **226 passed、28 skipped**（`tmp/pv26-implementation/20-final-full-suite.log`）；最新版 sequential pair smoke **complete_equal**，c/t/compare exitcodes **0/0/0**，full trace 與 actor/critic bits 均相同（`21-review-pair-smoke.log`，完整 report 在 `review-pair/comparison.json`）。兩臂各 1 update／2 train episodes／80 cap／2 validation seeds／hidden8，trace bytes 為 **11,241,605／11,239,157**。正式預估 **829,905,218,304 bytes/arm**，pair **1,659,810,436,608 bytes**。尚未驗證 CUDA／正式 u20／真實 checkpoint EP；NAS durable copy 與 parent review／commit／launch 仍待 parent 完成。

## 2026-11-12 accepted readiness corrections

This amendment supersedes the earlier comparator and signal claims above. Formal
u20 remains **NOT launched**; parent must perform a fresh review after these fixes.
Historical evidence under `tmp/pv26-implementation/` remains unchanged. New evidence
is exclusively under `tmp/pv26-fixes/`.

Both manifests require typed source/provenance, architecture, runtime, RNG mapping,
limits, flags, no resume and empty heldout. `source_commit` is the explicit parent
commit provenance, not discovered or invented Git metadata in an archive.
`source_tree_sha` hashes the sorted relative paths and bytes of **all Python files
present in qrokkun_ai and qrokkun_env**, including v21/v23/v24/v25/v26 helpers and
package initializers. Producer and comparator check their actual current code
content against that digest; a run also checks it again before closure. Use the
comparator from the **same frozen archive** as both producers. Parent independently
verifies the deployment archive inventory and hashes (including an updated inventory rather than assuming the earlier 171-file
archive count still applies), and supplies its actual reviewed commit SHA.
The test-only repeated `a` commit is synthetic provenance, never deployment proof.
Initial P and frozen S identities are recomputed from resolved named tensors using
`state_dict_sha256`; model descriptors and manifest labels alone cannot establish
identity. Initial Adam state must be empty.

Every model, gradient and Adam tensor position requires a complete typed raw
reference, with stage-specific gradient None and optimizer-state coverage. Final
weight PASS explicitly resolves and compares every named actor/critic tensor byte
string. Joint descriptor-only omissions fail even when both bundles are resealed.
P draws are reconciled against exact active episode order and contiguous counters,
with reset/pre/post/G/end state continuity and output/action binding. CPU P draws
are additionally regenerated using recorded probabilities. CUDA P initial states
use the serialized 16-byte seed/zero-offset layout; CUDA draw execution itself is
not replayed by the CPU comparator. CUDA execution remains unverified here.

The comparator replays CPU environment transitions using the actual traced frozen
S tensors and recorded P actions. This independently checks reset events (including
rng_init and frame-0 births), all three S draws per sample, S outputs, Python field
uniform draws, bullet origins/IDs/birth/removal/collision records, histories, all
raw state boundaries and end counters/unused sources. A truly unused RNG needs no
invented draws. No allclose is used. This adds CPU validation work; comparator wall
time is not an execution-performance measurement. Run with OMP/MKL threads=1 and
the producer's CPU PyTorch/NumPy runtime for exact replay.

Stable `(kind, key)` alignment uses a SQLite scratch index under the existing
approved `$TMPDIR`, outside model/trace directories and outside the immutable
bundle seal. It streams raw CAS, retains only live episodes and per-update coverage
in RAM, and realigns after differing death frames. Missing required producer
coverage is invalid; legitimate differing frame counts produce structural
differences and `complete_different`. The report includes missing-event counts and
the first structural missing key per update even after an earlier numeric difference.
First differences follow chronological source
positions, never lexical kind ordering or walltimes. Scratch SQLite stores both
streams' JSON events plus an aligned table; provision **at least twice the combined
uncompressed JSON event size plus SQLite index/sort headroom** in approved TMPDIR.
This scratch estimate is additional to the unchanged producer raw-storage estimate;
no formal scratch capacity has been certified. Scratch is deleted on ordinary exit
and exceptions; interrupted processes may leave scratch directories for inspection.

Cancellation is cooperative. A stop delivered during status writing or log opening
is checked before any launch. On Linux, the parent blocks INT/TERM, checks stopped
and pending signals, then commits a launch and publishes the child handle before
restoring the original mask. A signal queued after that commit belongs to the
committed launch; restoring delivery terminates/reaps its published child. This is
not an impossible wall-clock transaction between signal generation and process
creation. A stdlib-only child trampoline restores the other inherited original mask entries and unblocks INT/TERM
before exec and before torch imports; no Python preexec_fn is used. Later arms and
compare never start after an observed stop. Parent masks and handlers are restored
on exceptions. Exit codes remain 0 equal, 2 different, 3 runtime/invalid and 130
stopped for the pair CLI.

Exact local validation commands (preserve the approved TMPDIR; output directories
must be fresh):

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GIT_CEILING_DIRECTORIES="$TMPDIR" PV25_TEST_REAL_INPUTS=1
PY=/storage/hermes/project/qrokkun26/.venv/bin/python
$PY -m pytest qrokkun_ai/v26/tests --ignore=tmp -p no:cacheprovider --basetemp=tmp/pv26-fixes/frozen-focused-tmp -q
$PY -m pytest --ignore=tmp -p no:cacheprovider --basetemp="$TMPDIR/pv26-fixes-frozen-full" -q
SOURCE=<actual-parent-reviewed-40-hex-commit>
$PY -m qrokkun_ai.v26.supervise \
  --out-dir tmp/pv26-fixes/pair --trace-dir tmp/pv26-fixes/pair-traces \
  --source-commit "$SOURCE" --nonformal --synthetic --device cpu \
  --updates 1 --episodes 2 --cap 80 --validation 2 --hidden 8
$PY -m qrokkun_ai.v26.compare tmp/pv26-fixes/pair/Pv26c tmp/pv26-fixes/pair/Pv26t \
  --expected-commit "$SOURCE" --out-report tmp/pv26-fixes/pair/recomparison.json
```

Ordinary v26 unit tests require no NAS files. `PV25_TEST_REAL_INPUTS=1` opts into
read-only metadata/hash checks, skipping when NAS is absent. No real-checkpoint
episodes, GPU runs, deployment, commit or push are part of this amendment.

Final frozen fix validation (supersedes intermediate runs): **66 passed**, 2
warnings, focused suite (`tmp/pv26-fixes/16-frozen-focused.log`, 294.59 s);
**269 passed, 26 skipped**, 2 warnings, full repository suite
(`17-frozen-full.log`, 354.72 s), with PV25_TEST_REAL_INPUTS=1. Warnings are the
inherited PPO variance calculation on the one-sample cap-1 censored fixture; no
variance/math adjustment was made to hide them. Serial/workers2/workers4 trace and
original-PPO parity controls pass. Frozen pair `18-frozen-pair.log` is
complete_equal, 721 events/arm, c/t/compare 0/0/0, full trace and final resolved
actor/critic bytes equal; trace sizes 11,241,044 / 11,241,016 bytes. CLI controls
(`20-frozen-cli-checks.log`) return 0 equal, 3 for jointly stripped weight bytes,
and 3 for wrong expected commit. Real death-frame comparison
(`22-frozen-death-alignment.log`, `frozen-death-report.json`) returns 2 with
1,067 / 1,191 events, 179 keys missing left / 55 missing right, and matching
update-2 P differences attributed to update 2. Source inventory
`21-frozen-source-inventory.json` records 90 Python files and digest
`44d1691f3dccd7ef31eb56b985684087e7b3f30e0bdf49b6f0391bc787547044`.
No formal run or real-checkpoint episode was performed. Fresh parent review,
actual commit provenance, CUDA behavior, formal execution, formal scratch/free
capacity and durable NAS readback remain outstanding.

### Operating contract (2026-11-12; manifest schema 2)

New bundles require manifest `schema: 2`. Schema 1 bundles are not complete under
this contract. The lossless raw blob representation is unchanged. Each existing
PPO forward supplies actual full `features`, actor-head `raw_logits`, and the
categorical distribution's entropy-computed cached `probs` at all four actor
stages, alongside existing normalized `logits` and `values`. The complete batch
and minibatch `indices` bind these outputs to inputs. Bootstrap records actual
critic-body input `features` and values from its existing value-only forward;
it has no actor logits or probabilities. Hooks observe detached clones and are
removed in `finally`; no extra numerical model forward is performed. Readback
requires typed raw references with batch, feature84, action9 and value dimensions.
Initial player and spawner input locks remain as specified above.

After independently validating completeness of both bundles, `status` and CLI
exit 0 (`complete_equal`) or 2 (`complete_different`) refer solely to final-update
named actor+critic tensor bytes (`weights_bitwise_equal`; U20 for formal runs).
`per_update_weights` includes every update 0 through the configured final update,
all named tensor results, and `earliest_weight_difference` reports the earliest
weight mismatch. `all_updates_weights_bitwise_equal` is diagnostic, not an extra
success gate. Earlier mismatches that converge to identical final weights give
exit 0 and remain visible. `full_trace_equal`, `trace_status`, `first_difference`,
`first_difference_per_update`, and structural differences are independent trace
diagnostics; diagnostic-only differences also give weight PASS/exit 0. Incomplete
or invalid bundles cannot PASS (exit 3). Both arms must use a CUDA device for formal
execution (`torch.device(device).type == 'cuda'`); CPU requires `--nonformal`.

Comparison follows chronological source events with stable key alignment. Within
an event it checks sampler pre-states and model inputs/features/raw outputs before
draw outputs/post-states and physical effects, including nested S distributions.
The first difference is the first observable difference under that traversal,
not proof of an ultimate cause. All comparisons use raw bytes, without allclose
or another P forward. Exposed run/supervise/compare CLI boundaries map runtime
exceptions to 3 and owner interruption (including KeyboardInterrupt) to 130 while
printing tracebacks and preserving cleanup. Argparse usage errors return 3 for all
three public module CLIs; help remains 0. Exit 2 is reserved for a completed
comparison with different final weights. Python
`main()` APIs retain their existing exception behavior. Supervisor comparison exit
0/2 carries the same final-weight semantics; stop prevents the next stage.

Arm completion remains provisional through trace closure, arm closure/status
publication and restoration of the handlers present at arm entry. A stop observed
before the final flag check after restoration publishes owned arm status `stopped`
and raises `InterruptedError` (CLI 130), so the supervisor cannot advance. Full
immutable trace/closure data may already exist; it does not override stopped
status or the child exit code. Only the arm that claimed the output may publish
its status. Completion commits when those handlers are restored and the final
flag check succeeds. A stop arriving after that cooperative handoff is outside
the arm's cancellation window; no wall-clock atomicity with signal generation or
future supervisor launches is promised.

### 2026-11-13 test-suite consolidation (supersedes 286-test expansion)

Per owner review, the Cartesian-product reseal and subprocess expansions in
`test_completeness.py`, `test_fixes3.py`, `test_readiness.py`, and
`test_cancel_finalize.py` (286 tests, ~513 s CPU) were consolidated into
`test_validation_contracts.py` and `test_cli_and_signals.py` alongside the 6 core
test modules, without modifying any production `qrokkun_ai/v26/*.py` module:
- `qrokkun_ai/v26/tests`: **41 passed, 0 warnings** (~101.7 s on single-thread CPU).
- Updated `source_tree_sha`: `0876fcc49688b83e821fc932f5bf7d846ac09f5a5e9684c41dd1253d3e2771c2`.
