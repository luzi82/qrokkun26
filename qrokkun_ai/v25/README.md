# Pv25 C：Pv21 warm start 對固定 Sv24

狀態：實作完成，等待 parent 獨立 review、核實 smoke 與 GB10 launch。尚無 Pv25 正式訓練或 heldout 結果。完整 owner 合約見 [繁中文件](../../docs/journal/2026-11-11-1794400966-pv25-warm-start-contract.txt)。

假說：保留原 Pv21 u400 的 actor、critic、architecture、observation 與 clearance teacher，對固定 Sv24 best u980 繼續 on-policy 學習，可能提高 P 存活時間。首次嚴格超過 66.63020833333333 秒，尤其是否在 u500（含）以前，是觀察指標，沒有保證，不作 early stop 或調參依據。

## 固定輸入與繼承

| 項目 | 固定 identity |
| --- | --- |
| P 原檔 | `/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261109-1803-v21-exploratory/best_player_v21.pt` |
| P file SHA256 | `b8466bb13972de62003e16b0b570567a016d0dab4dd5667e66a3666974cfd5f8` |
| P state SHA256 | `250f3c3f1053b18129f992e97cb9ae15139fc2e5fa962b5491db5dddb2eb8f91` |
| P metadata | update 400；`player_v21_kinematic_decoupled` v1；hidden 472、top_k 12、horizon 12、in_dim 84 |
| S 原檔 | `/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261111-1040-sv24-seed17-u1000/best.pt` |
| S file SHA256 | `554290b6c17a280dbfb8bb4076583a1230b278bf73a24a701f66fca76e5bfcc0` |
| S metadata | completed/attempted 980、partial false；`sv24_global_scale_physical_v1` v1；source P file/state 必須等於上列 |

CLI 可換 GB10 local copy 的檔案位置，不能換 hashes。S 固定在 CPU、eval、requires_grad false，callback 在 no_grad；每次驗證與每次完整 P update 後核對 S state digest 與 grad/device。原始 P weights 精確載入，兩個 Adam 都重新建立、初始 state 為空，Pv25 completed update 從 0 起算。`initial.pt` 永久保留 u0；新 checkpoint type 是 `pv25_warm_start_v1`，architecture 仍如實寫 PlayerV21，可由 v21 loader strict load。

直接重用 v21 collector（僅以私有 globals 字典注入 env factory）、obs、shaped reward、GAE、teacher 與 train_step，不修改任何舊版 source。固定 seed 17、12 episodes/update、gamma .9995、lambda .999、reward scale .05、actor lr .0003、critic lr .001、Adam eps 1e-8、PPO 4 epochs、clip .2、minibatch 512、grad norm 1、entropy .005、dagger 1、PPO weight .3，teacher hard/soft imitation .65/.35 不變。沒有 B filter。

## 時間、RNG 與選模

正式模式完成 1000 個 PPO updates，無預設日曆期限。每集最多 7200 frames（120 秒），cap frame 真死亡優先，只有存活截尾用 P value bootstrap。訓練 P stochastic，驗證與最終 P deterministic argmax；S 一律 stochastic。

訓練第 u 次 seeds 為 `100000+(u-1)*12 .. 100000+u*12-1`，S 每集獨立 CPU generator `seed+25000000`；P sampling 與 optimizer 使用 training torch CPU/CUDA RNG。驗證/final S 為每集 `seed+24000000`；eval 以 fork_rng 保存／還原 training CPU/CUDA state。場景 RNG 使用原環境 seed/reset 語義。

u0 與每 10 updates（包含 u1000）使用固定 16 seeds `600000..600015`。直接呼叫 `v24.run.episodes`，沿用 player_actions、S physical mapping、CPU generator、16 個環境動態 active ordering。正式 u0 必須逐 seed 比對歷史 u980 的 frames/death/censored/cap；不接受只比平均或容差。CUDA batch near-argmax 差異若造成 mismatch，立即失敗，不能修 weights/actions 或放寬 gate。

`reference.json` 是唯讀擷取指定 progress 的 `completed_updates=980 / metrics.validation.per_seed`：source SHA `735c4aeecb893d8a6ef0a39fb7953a1a8e05bcbc3120cf694607973d325214df`，reference bytes SHA `af7638d6606cb4e9041f9746d73ec075c54d21ada0e8357f6ee6cb2fd083ae59`，import 時驗證。16 frames 依序為 4327,3498,4391,2614,4227,4360,3201,3067,5418,5226,4353,3110,5158,3171,3307,4537，全部 death、無 censor。

選模最大化 P validation mean，平手保留較早者，u0 也有資格。selection mean 使用整數 frames 總和除以 `60*n`，避免 numpy/逐秒累加末位差異把 u0 誤認 crossing；歷史 API 原始 summary 仍保留。strict crossing 為 mean > 66.63020833333333。記錄首次 crossing、首次 by500；u500 保存 summary（best/latest hash）、checkpoint，u1000 保存 terminal 與 best，持續觀察到 clean 1000，不因 crossing 提早結束。

## 一次性 final 與操作

新 heldout window `800000..800063` 是任何存取前鎖定的工程預設。clean formal u1000 與終點 validation 後才鎖 best，與原 Pv21 u400 配對 64 unique seeds，各自整批 64 動態 active，兩者每 seed 相同 S API 與 CPU generator +24000000。不得使用舊 Sv24 `700000` window 或 v21 `3000` diagnostics。保存所有 death/censor flags、逐集完成清單與 unfinished seeds。比較 Pv25−original 的配對平均秒差，seed 17 的 10000 次 paired bootstrap 95% CI，CI 包含 0 為 indistinguishable，不重選、不重跑。

直接 final CLI 也檢查 manifest/config/source、閉合 completed/validation 1000、partial false、所有 checkpoint hashes、reference/input identities；在 episode 前 exclusive `final-claimed.json`，失敗或中止不移除，禁止重跑／覆蓋。final 每集及寫入前查 guard。沒有 resume 或自動 restart。

輸出必須全新空目錄，exclusive train claim；supervisor 另有 sibling `.launcher` claim，正向管理單一 direct child。train 成功且正式 clean 1000 才進 final；取消後重新檢查再 Popen；SIGTERM 後最多 10 秒，kill 並 wait/reap。status、manifest、JSONL、training-closed 與 checkpoints 使用 fsync，原子 JSON/checkpoint replace 後 fsync parent directory。中止保存 interrupted checkpoint，完整計數與 attempted、partial、validation_pending 分開，失敗 snapshot 不升為 best。

下列是交付 parent 的命令範本，請先把 `PV25_SOURCE_COMMIT` 設成 parent 最終 review/commit 後實際打包的 40-hex SHA，`PV25_P`、`PV25_S` 設成上列原檔或 hash 相同的 GB10 local copies，`PV25_OUT` 設成不存在或空的新正式目錄，`PV25_PYTHON` 設成 GB10 Python。無 `.git` 的 archive 必須提供 source commit；封存時包含 v25 reference.json、未變動的 v21/v24/env 依賴。工作目錄須為 repo root；TMPDIR 設成允許的 scratch。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. "$PV25_PYTHON" -m qrokkun_ai.v25.supervise \
  --out-dir "$PV25_OUT" --source-commit "$PV25_SOURCE_COMMIT" \
  --player-checkpoint "$PV25_P" --spawner-checkpoint "$PV25_S" \
  --device cuda --seed 17 --updates 1000 --episodes-per-update 12 \
  --max-steps 7200 --eval-seeds 16 --final-seeds 64
```

Parent smoke（真實 locked inputs、全 cap、1 update/1 training episode/2 eval seeds；不同新目錄）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. "$PV25_PYTHON" -m qrokkun_ai.v25.supervise \
  --out-dir "$PV25_SMOKE_OUT" --source-commit "$PV25_SOURCE_COMMIT" \
  --player-checkpoint "$PV25_P" --spawner-checkpoint "$PV25_S" \
  --device cuda --seed 17 --nonformal --updates 1 --episodes-per-update 1 \
  --max-steps 7200 --eval-seeds 2 --final-seeds 64
```

Nonformal 明確繞過歷史 gate，metadata/status 為 nonformal/nonformal_complete；可用較小 max-steps 做 parent smoke，但不能進 final、u500 promotion 或當作正式結果。無 CLI hash bypass；測試只以明確 monkeypatch 注入 synthetic models。可加 `--deadline-unix` 有限值作操作停止，停止不進 final，沒有歷史日期 cutoff。

本機 CPU synthetic 驗證（無需 NAS；沒有真實 checkpoint episodes）。先指定 CPU Python 並建立 scratch；不設定 `PV25_TEST_REAL_INPUTS=1` 時，唯一的 real-artifact integration test 會明確 skip，synthetic passes 不代表真實 artifact 已驗證：

```bash
export PV25_PYTHON=/storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python
export TMPDIR="$PWD/tmp/pv25-fixes"
mkdir -p "$TMPDIR"
unset PV25_TEST_REAL_INPUTS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. GIT_CEILING_DIRECTORIES="$TMPDIR" \
  "$PV25_PYTHON" -m pytest qrokkun_ai/v25/tests -q -rs
```

`test_real_locked_input_metadata_only` 是 optional integration，只有 `PV25_TEST_REAL_INPUTS=1` 才啟用；`PV25_TEST_PLAYER_CHECKPOINT` 與 `PV25_TEST_SPAWNER_CHECKPOINT` 可指定 hash 相同的 local copies，未設定時使用上列固定 NAS 路徑。未 opt in 或任一 artifact 不存在時 skip；opt in 且兩檔存在時必須通過原本 strict file hashes、loader metadata、P state digest、P trainability 與 S frozen/eval 檢查，錯誤不可 skip，沒有 CLI hash bypass。此測試只讀 metadata/weights，不執行真實 checkpoint episodes。

Parent 在 NAS 可用時先明確執行完整 test node，確認 **1 passed、0 skipped**，再以相同 opt-in 環境跑 full suite。real integration skip 不可列為 verified integration pass：

```bash
export PV25_TEST_REAL_INPUTS=1
export PV25_TEST_PLAYER_CHECKPOINT="$PV25_P"
export PV25_TEST_SPAWNER_CHECKPOINT="$PV25_S"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. GIT_CEILING_DIRECTORIES="$TMPDIR" \
  "$PV25_PYTHON" -m pytest \
  qrokkun_ai/v25/tests/test_run.py::test_real_locked_input_metadata_only -q -rs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. GIT_CEILING_DIRECTORIES="$TMPDIR" \
  "$PV25_PYTHON" -m pytest --ignore=tmp -q -rs
```

無 NAS 的完整 synthetic suite 使用下列命令，integration 仍 conditional skip：

```bash
unset PV25_TEST_REAL_INPUTS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. GIT_CEILING_DIRECTORIES="$TMPDIR" \
  "$PV25_PYTHON" -m pytest --ignore=tmp -q -rs
```

Synthetic lifecycle fixtures 明確提供固定 40-hex 測試 SHA，不代表真實 source provenance。`test_portability.py` 複製 runtime source、v25 tests/reference 到乾淨 TMPDIR snapshot（包含尚未 tracked 的 v25），設定 Git ceiling 並確認 Git discovery 失敗；在其中執行現有 focused tests，以及 subprocess CLI parser/resolver 的 explicit SHA 成功與 missing SHA 拒絕。子程序不 opt in NAS，也不修改 production resolver。正式 archive CLI 仍必須提供實際打包 commit SHA。

本次 RED/GREEN、full pytest、diff check 與 summary evidence 在 repo ignored `tmp/pv25-fixes/`；先前 `tmp/pv25-implementation/` raw reports 保持原樣。完整 pytest 的 `--ignore=tmp` 排除 scratch source snapshots 的重複收集；`GIT_CEILING_DIRECTORIES="$TMPDIR"` 避免既有 v23 archive 測試向上發現 repo `.git`。最終 review、commit、push、verified smoke 與 GB10 正式啟動由 parent 負責。
