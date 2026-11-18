# 遊戲與模型版本速查

本文件統一記錄 G／P／S 各版本的用途與主要差異；技術細節與實驗證據見各版原始碼、合約及附日期的實驗紀錄。

- **G = game environment / rules**：遊戲環境與規則，不是神經網路權重。
- **P = player**：學習讓玩家存活更久；**S = spawner**：學習縮短 P 的存活時間。
- `v20`、`v21` 等是版本識別，數字不是訓練更新次數。
- `u400`、`u980` 是某次訓練的 checkpoint/update。Pv21 u400 仍是 Pv21；Sv24 best u980 與終點 u1000 是不同 checkpoint。
- Warm start 可沿用權重而建立新的訓練譜系：Pv25 從 Pv21 u400 載入，但 fresh Adam、自己的 update 從 0 起算；其 1000-update 預算不是接續計為 u1400。

## v10（歷史基線）

用途：零慣性玩家物理下的 decoupled actor-critic 歷史基線；原文件的「Current implementation」僅描述當時版本。

| 項目 | 歷史內容 |
| --- | --- |
| Player | `qrokkun_ai.v10.agents.player_v10.PlayerV10` |
| Train entry | `qrokkun_ai.v10.train.train_player_v10` |
| Episode cap | 120s / 7200-step |
| Physics / Policy | zero-inertia / decoupled actor-critic |

來源：[玩家](../qrokkun_ai/v10/agents/player_v10.py)、[訓練入口](../qrokkun_ai/v10/train/train_player_v10.py)。
`v0`..`v5` are archived at git tag `legacy-inertia-physics-v5`.

## Gv20 — 零慣性遊戲規則

用途：提供正常 Godot 遊玩及 Python 訓練使用的規則基礎；scripted 隨機生成器與排程，不是 learned model，沒有權重初始化或 optimizer。

- 玩家即時移動／停止；子彈等速、半徑 2.6。S 由中心射線取邊界出生點，瞄準偏移為每軸 Gaussian σ=13.2，速度均勻取 30..200。
- 生成排程將首次事件對齊 frame 0；每 60 frames 額外 field bullet 使用前 60 frames 的 S 平均速度、直接瞄準玩家，排除同 frame S。
- 正常 F5 使用 scripted Gv20，並非 Sv22–Sv24。Godot/Python parity 證據限於報告所列排程、碰撞與顯式隨機變量 replay；兩引擎同 seed 不保證同 RNG stream，該次無圖形／手動驗證。
- 來源：[Python 規則](../qrokkun_env/gv20.py)、[README 的引擎說明](../README.md#playable-gv20-and-engine-parity)；[2026-11-10 整合與驗證紀錄](journal/2026-11-10-1794266611-gv20-godot-integration-and-verification.txt)。

## Pv21 — 以運動學 clearance 輔助躲彈

用途：以 clearance teacher 模仿學習加 PPO，提高原始 scripted Gv20（含 field bullets）下的玩家存活時間。

- Actor/critic 為完全分離的 MLP，各兩層 hidden 472；觀察編碼最多最近 64 顆子彈，網路使用其中最近 12 顆及 mask，加入 12-frame 運動學 clearance／安全旗標，合計 84 維；輸出 9 個離散動作（停留＋八方向）。
- 新建 PlayerV21、分開 Adam 訓練 actor/critic；對手是 Gv20 腳本，不是 Sv24。歷史探索訓練共 500 updates，固定 30-seed validation 選出 u400；u400 是後續實驗指定的起始權重或凍結對手，不是另一版本。
- 來源：[觀察／teacher](../qrokkun_ai/v21/obs_v21.py)、[模型](../qrokkun_ai/v21/agents/player_v21.py)、[trainer](../qrokkun_ai/v21/train/train_player_v21.py)；[2026-11-09 探索結果](journal/2026-11-09-1794238038-gv20-pv21-500-results.txt)（非取代 v10 的 promotion）。

## Sv22 — raw Gaussian 生成策略

用途：在 Gv20 排程下訓練 S，縮短凍結 Pv21 u400（deterministic argmax）的存活時間。

- 公開即時位置／速度／時間與最近 64 顆子彈及 mask 共 325 維；分離 actor/critic hidden 128。Actor 輸出五個 raw Gaussian 均值，另學習五個全域 `log_std`；噪聲加在非線性轉換前的 raw 座標。
- 五個 raw 座標經非線性映射為出生角、速度及瞄準半徑／方向；PPO 計算 raw 座標 likelihood。從頭初始化 S／Adam，非沿用 P 權重；沿用 Gv20 的 field 規則。
- 採期限預算，固定 16-seed validation 選模，預定閉合後一次性 64-seed final。2026-11-10 擁有人提前停止，完成 1011 updates、best u880；未使用 final heldout，不能當作正式 heldout 學習證明。
- 來源：[core](../qrokkun_ai/v22/core.py)、[runner 說明](../qrokkun_ai/v22/README.md)；[2026-11-10 合約](journal/2026-11-10-1794269002-sv22-contract.txt)、[同日停止報告](journal/2026-11-10-1794311518-sv22-training-stop-report.txt)。

## Sv23 — 實體條件分布與狀態相依尺度

用途：改以實體出生角、速度、發射角的條件分布訓練 S，對手仍為凍結 Pv21 u400 argmax。

- 沿用 325 維公開觀察與分離 hidden-128 actor/critic；actor **8 outputs = 5 mean components + 3 state-dependent scale logits**。出生二維向量正規化取角（精確零指東），速度採 30..200 px/s 實體 truncated Normal。
- Aim 是玩家位置加無界像素偏移；發射角以實際 sampled birth 到目標的方向為均值，再加 spawn-relative wrapped angular noise，likelihood 保留此條件關係。
- 從頭初始化 S／Adam；預算 1000 updates、固定 16-seed validation 選 best，閉合後預定一次性 64-seed final。2026-11-11 擁有人於 750 completed updates 取消；u750 validation 未完成，沒有正式 final。
- 來源：[core](../qrokkun_ai/v23/core.py)、[runner 說明](../qrokkun_ai/v23/README.md)；[2026-11-10 合約](journal/2026-11-10-1794320707-sv23-contract.txt)、[2026-11-11 停止報告](journal/2026-11-11-1794351856-sv23-training-stop-report.txt)。

## Sv24 — 全域有界尺度與像素 lead offset

用途：調整 Sv23 的尺度、瞄準偏移及近零向量梯度，繼續最小化凍結 Pv21 u400 argmax 的存活時間。

- 325 維觀察、分離 hidden-128 actor/critic；actor **只輸出 5 mean components**，另有 **3 個全域、與狀態無關的可訓練 scale logits**，不是 8 個 actor outputs。
- 保留出生方向與實體 truncated speed（30..200）；aim 每軸為 `48*tanh(raw)` 像素，fire std 上限 0.25 rad；出生角反向梯度近零時平滑抑制。
- 從頭初始化 S／Adam；1000-update 預算、固定 16-seed validation 選 best、閉合後一次性 64-seed heldout。2026-11-11 已完成；選出 **best u980**，不是 terminal u1000。報告的學習證據限於固定 P、單一訓練 seed 與預定對照協定。
- 來源：[core](../qrokkun_ai/v24/core.py)、[設計說明](../qrokkun_ai/v24/README.md)；[2026-11-11 完成報告（含策略契約及結果）](journal/2026-11-11-1794395723-sv24-training-completion-report.txt)。

## Pv25 — Pv21 warm start 對固定 Sv24

用途：以原 Pv21 網路繼續提高玩家對固定 stochastic Sv24 best u980 的存活時間；不是新網路架構。

- 精確繼承 Pv21 u400 actor/critic 權重、84 維輸入、hidden 472、9 動作、原觀察／reward／clearance teacher／PPO；重新建立兩個 Adam，自己的 update 從 0 起算。不包含新子彈篩選設計。
- Sv24 u980 全程凍結且 stochastic；P 訓練 stochastic，validation/final 為 deterministic argmax。正式預算 1000 completed updates，不因 crossing 提前停止。
- u0 及每 10 updates 用固定 16 seeds；門檻為嚴格超過 66.63020833333333 秒（約 66.63 秒），只屬固定 validation 觀察，不能當 heldout 泛化。完整閉合及 terminal validation 後才鎖 best，以全新 64 seeds 與原 Pv21 作一次性配對 final。
- 有日期的狀態：2026-11-11 啟動報告已證實正式啟動；其快照尚無 final。此處不推定目前進度或已完成，結案須另附可查證報告。
- 來源：[README](../qrokkun_ai/v25/README.md)、[training](../qrokkun_ai/v25/training.py)、[run](../qrokkun_ai/v25/run.py)、[core](../qrokkun_ai/v25/core.py)；[2026-11-11 合約](journal/2026-11-11-1794400966-pv25-warm-start-contract.txt)、[同日啟動報告](journal/2026-11-11-1794403738-pv25-training-launch-report.txt)。

- 結案（2026-11-12）：已完成 1000 updates，validation best **u670**；一次性全新 heldout 約 **69.44 秒 vs 原 Pv21 u400 70.20 秒**，配對差 95% CI [−5.31592, 3.88388] 包含 0，未證實改善或變差。見[完成報告](journal/2026-11-12-1794433829-pv25-training-completion-report.txt)；Nov 11 啟動敘述保留為歷史快照。

## 後續追加約定

每次建立／訓練新的 G／P／S 版本，都在本目錄依時間追加新條目，保留既有條目與歷史結果；不另建重複目錄。新 checkpoint／update 不自動構成新版本，也不逐 checkpoint 新增條目。

至少記錄用途、架構／觀察／動作差異、初始化及 optimizer 是否繼承、凍結對手與模式、訓練預算／評估規則，並連結原始碼、事前合約與結果報告。狀態與結果須有日期、可由保存證據核實；不要作即時進度 ticker，也不要把舊結果改寫成新結果。後續結案以附日期補充及報告連結保存。

可複製的最小模板：

```text
## G/P/SvNN — 用途（建立日期）
架構／觀察／動作：主要差異。
初始化／optimizer／凍結對手與模式：來源及繼承方式。
預算／選模／final：updates、seeds、閉合規則。
狀態／結果（日期）：可核實摘要；來源／合約／報告連結。
```

## Pv26c／Pv26t — 執行逐位元等價診斷（2026-11-12）

- Pv26c 為 serial workers1；Pv26t 為相同中央 batch／每 frame barrier 的 env.step thread pool，預設 workers4。沿用 Pv21 架構／觀察／reward／teacher／PPO；Pv21 u400 actor/critic warm start、fresh Adam，固定 stochastic CPU Sv24 best u980。
- 正式 u20、12 episodes/update、7200 cap、seed17；固定 validation u0/u10/u20，無 heldout／選模／promotion。新增每 episode P generator seed+26000000，兩臂相同；不宣稱重播歷史 Pv25。
- full lossless streaming trace／RNG／gradients／Adam／每 update checkpoints；離線報首差異及 actor＋critic raw bits。只回答執行等價，沒有強度結論。random-init Pv27+ 與 B filter 延後且未實作。
- 狀態（2026-11-13）：本地 bounded synthetic CPU 驗證與測試套件精簡完成（`v26` 41 passed、全套 244 passed / 26 skipped，生產模組未變）；當時快照 GB10 正式 `u20` 尚未啟動。
- 結案（2026-11-14）：GB10 CUDA 正式 `u20` 兩臂與離線完整重播比對正常完成（結束碼 `0/0/0`，`complete_equal`）；`u=0..20` 全部 21 個檢查點之 Actor＋Critic 具名張量 raw bits 與兩臂各 `1,346,718` 筆完整追蹤事件 100% 一致（`weights_bitwise_equal=true`、`full_trace_equal=true`；原始 trace 留存 GB10 NVMe，`trace_durable_bundle=false`）。見[結案報告](journal/2026-11-14-1794629266-pv26-u20-completion-report.txt)。
- 來源：[README](../qrokkun_ai/v26/README.md)、[runner](../qrokkun_ai/v26/run.py)、[comparator](../qrokkun_ai/v26/compare.py)、[事前合約](journal/2026-11-12-1794460557-pv26-thread-bitwise-contract.txt)、[測試精簡紀錄](journal/2026-11-13-1794502979-pv26-test-consolidation-report.txt)、[結案報告](journal/2026-11-14-1794629266-pv26-u20-completion-report.txt)。

Pv27c / Pv27w — owner-authorized u5 serial versus pinned spawn CPU workers.
See qrokkun_ai/v27/README.md. Fresh original Pv21 u400 and frozen CPU Sv24 u980;
terminal validation adapted to u5. Bounded weights and summaries replace Pv26
full tracing for this experiment only; historical Pv26 conclusions remain intact.
Readiness review, deployment and formal launch are pending parent ownership.

Pv27c／Pv27w 結案補充（2026-11-14）：兩臂正式完成 5/5 updates（exit 0、WEIGHTPASS），u0..u5 每次 12 個具名張量 raw bits 一致、各 240,606 training frames；實際 spawn worker 工作已確認。收集加 batch 組裝慢 4.8263%、全程慢 1.6837%，沒有加速；單一 seed17 配對，無 full trace／heldout／強度或 promotion 結論。見[完成報告](journal/2026-11-14-1794634360-pv27-cw-u5-completion-report.txt)。先前 pending 紀錄保留為歷史快照；不自動啟動 Pv28。

## Pv28c／Pv28m — 整集平行 CPU Rollout 與可重現加速基準（2026-11-14）

- 架構／觀察／動作：沿用 Pv21（`hidden=472`、`in_dim=84`、9 actions）；`Pv28c` 單核依序執行完整單場，`Pv28m` 以長駐 `spawn` 子行程池＋預配置 `SharedMemory` 平行執行完整單場（`batch=1`、`threads=1`、省略未使用之 `Spawner.log_prob` 與每幀 NAS 檢查），合併後於確定性 CUDA 整批計算 `expert_targets` 與 PPO。
- 初始化／optimizer／凍結對手與模式：原始 `Pv21 u400` actor/critic＋fresh Adam，對戰凍結 stochastic CPU `Sv24 best u980`；訓練 P stochastic、驗證 P deterministic argmax；建立 `Pv28` 新數值基準，不宣稱重播 `Pv25..Pv27`。
- 預算／檢查點／驗收：正式支援 `updates ∈ {5, 100, 1000}`（對應檢查點與 16-seed 驗證間距 `{1, 10, 100}`，六位數檔名 `checkpoint-{u:06d}.pt`、支援精確 `--resume`）；離線先驗終點 Actor＋Critic raw bits（`WEIGHTPASS`），不一致時二分搜尋定位首個分歧點，速度比獨立報告；無 heldout 或 promotion。
- 狀態（2026-11-14）：實作、本地測試（`v28` 18 passed、全套 279 passed / 28 skipped）與 `gpt-6.1-sol` 審查完成，當時快照待部署 GB10 執行 `u5` ➔ `u100` ➔ `u1000`。
- 結案（2026-11-15）：GB10 CUDA 正式 `u5`、`u100`、`u1000`（`Pv28m` 綁定 10 顆 `3.9 GHz` X925 大核）全數正常完成並達成 **`WEIGHTPASS`（0 ULP 差異）**；`u1000`（`49,888,272` 幀）對戰收集加速 **`6.69x`（`7,259.4 FPS`）**、驗證加速 **`7.33x`**、全程耗時由 `49,766.75s` 縮至 **`10,223.14s`（總加速 `4.87x`）**，且獨立 `u100` 與 `u1000` 於第 100 輪之模型雜湊完全一致。
- 來源：[README](../qrokkun_ai/v28/README.md)、[概念規格書](../concepts_brief.md)、[事前合約](journal/2026-11-14-1794642996-pv28-cm-preregistration-contract.txt)、[結案報告](journal/2026-11-15-1794714252-pv28-cm-u5-u100-u1000-completion-report.txt)。

- **Pv29c / Pv29m**（2026-11-15，預註冊／尚未正式執行）：[版本契約](../qrokkun_ai/v29/README.md)。在 Pv28 演算法上固定 u10/E20、10 pinned workers 對 serial，checkpoint cadence1／validation cadence10；全更新線性 raw-bit 比較，terminal-only verdict。取代 Pv28 此配對的預算／比較排程決策，不改寫歷史結論。

Pv29c／Pv29m 結案補充（2026-11-15）：正式 u10/E20 兩臂完成 10/10 updates（complete_equal、exit 0），PRIMARY u10 **WEIGHTPASS**；獨立 u0..u10 每次 12 個具名 actor/critic 張量 raw bits 一致，各 805,265 training frames。完整收集 wall 加速 **8.0187x**、全程 **5.8143x**；單一 seed17，無 full trace／heldout／promotion 或泛化結論。見[完成報告](journal/2026-11-15-1794724028-pv29-cm-u10-completion-report.txt)。上述預註冊 pending 文字保留為當時快照，由本次日期結案補充取代執行狀態。


## 2026-11-15 Pv30 actor／critic CUDA streams 實作

新增 [v30 說明](../qrokkun_ai/v30/README.md)：c/m 均使用同一新版 PPO，正式預算 u10/E20/seed17，僅驗證同版本重現性，不要求與 Pv29 位元一致。兩條非預設 CUDA streams、分區梯度 guards、每 minibatch join；CPU 明確 serial。父層預定 c→m 完整配對及額外獨立 fresh-init m u10，新 output 驗證 m/m。CPU 最終 19 passed、1 CUDA skipped；未執行 GPU、正式配對或效能量測，不能宣稱加速或 GPU 重現性。細節與命令見日期 journal。


Pv30c／Pv30m u10 結案補充（2026-11-15）：三臂完成 10/10、exit 0；c/m 與事前規劃的獨立 fresh m1/m2 均 **WEIGHTPASS**，u0..u10 每次 12 個具名張量 raw bits 一致，各 805,265 training frames。同環境重現性達成；多核心 rollout 加速 **8.1498x**、全程 **5.1943x**，但三臂共享同一平行 PPO，**未證實 PPO 加速**；相對歷史 Pv29m，m1/m2 PPO 分別慢 59.04%／52.39%（描述性比較，無因果隔離）。有限 Nsight 探針觀察到 76 對跨分支 kernel 重疊，不能據此宣稱效能提升。見[完成報告](journal/2026-11-15-1794738370-pv30-cm-u10-completion-report.txt)。上述待執行文字保留為日期快照，本次結案補充取代其 pending 執行狀態；無 full trace／heldout／promotion 或跨環境保證。

## Pv31m（含條件備援 Pv31c）— 單流外提特徵與延遲同步 PPO 加速（2026-11-15）

- 架構／觀察／動作：沿用 Pv21（`hidden=472`、`in_dim=84`、9 actions）；取代 `Pv30` 之雙 CUDA streams＋雙執行緒 `ThreadPoolExecutor`，改為單流序列派發（`cuda_hoisted_single_stream`），於 4 個 PPO epochs 前整批外提無參數 `net.features()` 與老師 `F.softmax(exp_logits)`、移除迴圈內 `.item()` 與逐張量同步（改於期末單次同步與有限值檢查），並於 `EpisodePool` 綁定 10 顆 X925 大核後將主行程固定綁定於單顆 `3.9 GHz` X925 大核（`CPU 5`）。保留歷史 `Pv29`／`Pv30` 產物與結論不變。
- 初始化／optimizer／凍結對手與模式：原始 `Pv21 u400` actor/critic＋fresh Adam，對戰凍結 stochastic CPU `Sv24 best u980`；正式預算 `u10 / E20 / seed=17 / cap=7200`。
- 結案（2026-11-15）：GB10 CUDA 非正式 smoke（`u1, E3`）與正式 `Pv31m`（`u10, E20, 805,265` frames）直接對照本地雜湊驗證之封存 `Pv30c` 皆達成 **`WEIGHTPASS`（exit 0，`u0..u10` 全部 11 個檢查點之 12 個具名 Actor＋Critic 張量 `0-ULP` 完全一致，終點 `u10` 共享 `state_dict_sha256 = 875a2d40bf89086438c784d66b6629f4593a76287d79b74e708990eb4fe5db0a`）**，無須啟動條件備援 `Pv31c`。`Pv31m` 的 `ppo_s` 降至 **`15.25s`**（穩態 `u2..u10` 平均 `1.46s/update`；較 `Pv30c` `73.92s` 快 **`4.85x`**、較 `Pv30m` `69.02s` 快 **`4.53x`**、較單流序列基準 `Pv29m` `43.40s` 快 **`2.85x`**），`rollout_s = 90.98s`（**`8.22x`** vs `Pv30c`），全程 `total_s` 縮短至 **`122.31s`**（較 `Pv30c` 快 **`7.55x`**、較 `Pv30m` 快 **`1.45x`**、較 `Pv29m` 快 **`1.24x`**）；單一 `seed=17`，無 heldout 或 promotion。
- 來源：[README](../qrokkun_ai/v31/README.md)、[EGM 設計規格書](designs/pv31_hoisted_single_stream_ppo.md)、[事前合約](journal/2026-11-15-1794744069-pv31-m-preregistration-contract.txt)、[啟動前勘誤補正](journal/2026-11-15-1794744180-pv31-m-preregistration-prelaunch-erratum.txt)、[結案報告](journal/2026-11-15-1794744574-pv31-m-u10-completion-report.txt)。

### Pv32m — original Pv21 u400 warm-start, u1000 / E20

Implementation: `qrokkun_ai/v32/`. Preserves Pv31m PPO and episode-parallel CPU sampling directly, fresh Adam/new u0 lineage, frozen Sv24 u980. Validation u0/u10..u1000 selects strict integer-total-frame best (ties earlier); terminal saved independently. One-shot paired final uses the previously used Pv25 benchmark, primary best and secondary terminal versus original. No legacy survival gate, no random-init arm, no launch. Pv25 E12 remains historical reference; this E20 baseline supports a future separately authorized episode-matched initialization comparison. See `qrokkun_ai/v32/README.md` for protocol and commands.


Pv32m warm-start u1000 結案補充（2026-11-16）：正式 u1000/E20/seed17 完成 20000 episodes、83378713 frames，train/final exit0。主要 validation-selected best u120 對 original 的共同基準差 +0.488542 秒，95% 配對 CI 涵蓋 0，尚未證實改善；terminal u1000 為次要結果。此為未來 random-init 比較的 warm-start 對照，並非最低改善門檻；800000..800063 為曾使用窗口，非新 heldout，尚未啟動 random 臂。見[完成報告](journal/2026-11-16-1794763715-pv32-warm-start-u1000-completion-report.txt)。本日期附錄取代先前 pending／未啟動的執行狀態，舊條目與預註冊保留為歷史快照。


## Pv33m：隨機初始化 matched control（實驗用途，不晉升）

本版本以已完成 Pv32m 暖啟動為參考，比較相同「額外」1000 updates、每次 20 episodes 的訓練預算；不宣稱終生預訓練算力相同。來源基底為 `3c33391...`，並非最終 source SHA；封存啟動必須明確提供 40 位 `--source-commit`，不依賴 git。

所有 actor／critic 權重與 bias 均使用 PlayerV21 預設 nn.Linear 初始化，在 CPU `fork_rng(devices=[])` 內固定 seed 17，還原呼叫者 RNG；所有模型建構後重設與 Pv32 相同的訓練 seed 17、permutation seed 27000017。兩個 Adam 全新，u0 實際驗證且可入選。保留 clearance teacher imitation，並非 teacher-free pure PPO。沿用 v31 `cuda_hoisted_single_stream`、feature hoist、deferred sync 與 v32 pool lifecycle；不更動觀測、獎勵、PPO 或架構。Sv24 u980 凍結、stochastic CPU，驗證原始 P lineage 但不載入 P 權重初始化 learner。

每 10 updates 用 600000..600015 驗證，整數 frames 總和嚴格較大才更新 best，同分保留較早者；保存 101 checkpoint／evaluation。無 score floor、early stop、重試或 resume。只有完整正式訓練與 terminal 驗證後才能獨占 claim final。PRIMARY 為 Pv33 selected best 減保存的 Pv32 best u120；SECONDARY 為 terminal u1000 相減。僅新增 Pv33 best 與 terminal 各 64 episodes（兩者相同可重用），不重跑 Pv32 或原始 Pv21。800000..800063 是先前 Pv25／Pv32 已見 common window，非新 heldout。10000 paired bootstrap、seed 17、95% CI 僅描述用途；CI 涵蓋零不構成失敗。保存 u0..1000 曲線、實際 frames 與時間比較，不用 final 重新選模。

固定單一初始化與單一訓練 seed 不能證明初始化普遍優劣，不將速度或強度差異直接歸因於初始化。原始 Pv21 保存分數 70.19765625 秒僅供輔助參考。暖參考為 `20261115-2205-pv32m-warm-start-u1000`，training commit `32745bac06ca422d52598d4039ea47e0b3afff63`；完整固定舊檔雜湊見 `qrokkun_ai/v33/warm-reference.json`，不修改舊產物。可透過 `--warm-reference-dir` 指定 GB10 上相同內容的可攜副本；v32 與 v33 的版本／source fingerprint 本來就不同。

本次僅完成實作與 CPU fixture 驗證；未執行 GPU、CUDA smoke 或正式訓練。獨立測試、一次 fresh source readiness review、部署、CUDA smoke 與正式執行由 parent 處理。


Pv33m random-init u1000 結案補充（2026-11-16）：train／final／supervisor 均 exit 0、complete；完成 20000 episodes、81377765 frames。與保存 Pv32 暖臂比較相同額外 u1000／E20 預算（非相同 frames 或終生預訓練算力）。PRIMARY best u910 減 warm best u120 為 -2.2963541666666667 秒，95% CI [-7.069075520833333, 2.3875130208333326]；SECONDARY terminal 相減為 -4.347395833333334 秒，95% CI [-9.05005859375, 0.6251692708333266]。點估計偏向暖啟動，兩個 CI 均涵蓋 0，未證實暖臂較好，也不證明等價；單一 seed、已見共同 benchmark 限制推論。僅新增隨機 best／terminal 各 64 episodes，暖／Pv21 結果重用保存資料。見[完成報告](journal/2026-11-16-1794797146-pv33-random-init-u1000-completion-report.txt)。本日期結案附錄取代先前 pending／未執行 CUDA smoke、正式訓練或 random 臂之執行狀態；原條目、預註冊與歷史結論保留為當時快照。

## Pv34m：保守不可能碰撞觀測過濾（待正式執行）

沿用 Pv33m 從零初始化、Sv24u980、v31 數值核心與 v32 資源生命週期；只在 P 穩定距離排序／top64 前，依完整生命期可達集合外包移除可證無法碰撞的子彈。環境與 teacher 演算法不變，teacher 數值允許隨觀測改變。協定與證明見 [v34](../qrokkun_ai/v34/README.md)。未取代歷史結論；正式 CPU 獨立驗證、部署、CUDA smoke 與 u1000 由父流程負責。


Pv34m 威脅觀測過濾 u1000 結案補充（2026-11-16）：train／final／supervisor 均 exit0、complete；20000 episodes、81274532 frames，與 Pv33 相同額外 u1000/E20 預算（非相同 frames／時間）。PRIMARY best u210 對 Pv33 best u910 差 -1.1083333333333332 秒，95% CI [-5.627656249999999, 3.5167057291666652]；SECONDARY terminal 差 +1.8497395833333337 秒，95% CI [-3.568639322916667, 7.137239583333329]。兩個 CI 均涵蓋零，未證實改善，亦非等價或無效證明。過濾移除約23.4%子彈觀測出現次數，非環境實體子彈；單一 seed、已見共同 benchmark 限制推論。見[完成報告](journal/2026-11-16-1794816495-pv34-threat-filter-u1000-completion-report.txt)。本日期附錄僅取代舊 pending／尚未執行狀態（supersedes execution status only）；原預註冊、歷史條目與結論逐字保留。

### Pv35m — input64, teacher12 / clearance12 (2026-11-16; implementation only)

New `qrokkun_ai/v35` expands both actor and critic inputs84→344, hidden472, following unchanged Pv34 filtering/order. Exact saved Pv34 random u0 realization is embedded with only additional raw/mask columns zeroed; fresh Adam and frozen Sv24. Versioned model metadata, spawn worker hooks, dedicated supervisor/final and production dependency fingerprint are local to v35. Closed Pv34 best210/terminal1000 common64 results are reused, never rerun. Formal protocol unchanged; parent independent review, CUDA smoke and launch pending. See `qrokkun_ai/v35/README.md`. No scientific result or promotion claim.


Pv35m input64 u1000 結案補充（2026-11-17）：完成 1000 updates／20000 episodes／80777784 frames；PRIMARY best u330 common64 71.6138021s vs Pv34 best u210 67.2815104s，候選減對照 +4.3322917s（95% 配對 CI [-1.1203581, +9.7862826]）；SECONDARY terminal +1.8638021s（CI [-2.6112305, +6.3424674]）。兩個 CI 均跨 0，未證實改善或等價；單一 init/train seed17、已見 common64，非新 heldout，無模型晉升。見[完成報告](journal/2026-11-17-1794856288-pv35-input64-u1000-completion-report.txt)。本附錄僅取代舊 pending 執行狀態；歷史條目保留。


### Gv36 integer player — Python only (2026-11-17)

Opt-in `qrokkun_env.gv36.Gv36Env`: signed integer movement counts, canonical
P-position key, cached float32 coordinates, per-axis wall reset, fixed 60 Hz
and immutable P-only snapshot/restore; inherited Gv20 S/bullets and 64-slot
observations. 17 new tests; environment suite 39 passed. See [2026-11-17-1794858266-gv36-integer-player-implementation.txt](journal/2026-11-17-1794858266-gv36-integer-player-implementation.txt).
Owner naming decision supersedes Pv36 death-search-teaching name with **Pv37**;
design unfinished, no AI v36/v37 implementation or training. Gv36 changes rules
and needs a new control before Pv37; no old Pv35 direct causal comparison,
new baseline/control model assignment, or Godot mirror in this task.

Gv36 cardinal normalization (2026-11-17): current Gv36 commits `x1,y1` in
`0..9`: `q,r=divmod(cardinal,10)`, `origin+=17*q`, `cardinal=r`; diagonal
counts stay unchanged. This supersedes only the unbounded cardinal representation.
Algebraic origins may leave the legal rectangle; canonical keys and cached physics
remain unchanged from original Gv36. Valid old snapshots are accepted and normalized
after validation. Pv37 is not implemented.
See [verification](journal/2026-11-17-1794859706-gv36-cardinal-normalization.txt).


Pv37: two roles `control` / `search`, both Gv36 normalized integer cached P and Pv35 model64 untrained seed17 embedding with fresh Adam; frozen stochastic Sv24u980. H60/beam256/top8 death teaching; teacher12/clearance12 unchanged; CE0.1 separate actor Adam step per update. Nonformal paired trial only; no final trial evaluation; formal pending owner after trial. Historical Gv36 naming/reports unchanged. See `docs/journal/2026-11-17-1794861120-pv37-implementation.txt`.


Pv37 bounded trial 結案（2026-11-17）：Gv36 matched untrained u0、兩臂各 2 updates/E3；search 6 次搜尋、5 條成功路徑、300 幀示範及 2 次額外 actor 更新，僅證明接線，無強度／promotion 結論。見[結果報告](journal/2026-11-17-1794872251-pv37-bounded-trial-results.txt)。本附錄僅取代先前未執行 trial 的狀態，歷史條目保留；正式 1000/E20 control→learning 與完整 final 協定仍待 owner 確認及實作，尚未啟動。

### Pv37m / Pv38m — owner-authorized Gv36 formal pair (2026-11-17)

Pv37m: Gv36 control, ordinary unchanged Pv31 PPO + teacher12, Pv35 input64/clearance12/hidden472, same untrained seed17 embedding and fresh Adam. Pv38m: same engine and initialization plus fixed D−12/H60/beam256/top8 death-search teaching and one actor-only 0.1 mean-CE extra step per update when demonstrations exist. Gv36 retains its name. Both use frozen stochastic Sv24 u980. Formal budget1000/E20/cap7200/W10, validation16 each10; BEST primary and terminal secondary use the common previously seen64-seed final window. Current contract: `qrokkun_ai/v38/formal-contract.json`; executable public entrypoints and pair supervisor are under v37/v38. This supersedes only future role labels, never archived trial conclusions. Author implementation/tests only; no formal run started.

Formal records: [implementation](journal/2026-11-17-1794873702-pv37-pv38-implementation.txt), [preregistration](journal/2026-11-17-1794873702-pv37-pv38-prereg.txt).

Pv37／Pv38 正式來源鎖定（建立時 planned／not started）：[PREREG 與固定啟動命令](journal/2026-11-17-1794875161-pv37-pv38-formal-source-lock.txt)。


Pv37／Pv38 正式死亡搜尋教學結案（2026-11-17）：四階段皆 exit0，各1000 updates／20000 episodes。PRIMARY BEST（Pv37 u160 vs Pv38 u190）候選減對照 −0.12421875 s，95% CI [−4.999225260416666, 4.743522135416663]；SECONDARY terminal −0.8052083333333335 s，CI [−5.0886458333333335, 3.6456640624999985]。局部搜尋成功率85.1548%，但單一 seed17、已見 common64 未證實純網路強度改善或等價；不含 final 的 train elapsed 增加約36.74%。見[完成報告](journal/2026-11-17-1794911352-pv37-pv38-death-search-completion-report.txt)。本附錄僅取代舊 formal pending／尚未開始狀態，保留全部歷史報告與實驗來源；無自動後續訓練授權。

Pv38 [示範轉移診斷](journal/2026-11-17-1794922844-pv38-demonstration-transfer-diagnostic.txt)：抽樣單步 CE 小降、固定路徑逃生 6/12→8/12；無新模型版本或正式強度修訂。

Pv38 [離線 aux 預算診斷](journal/2026-11-17-1794924868-pv38-offline-aux-budget-diagnostic.txt)：額外 32 步降低封包 CE，但局部逃生 8/12→7/12，尚未顯示預算增加有益；非正式新版本。

Pv38 夜間研究附錄（2026-11-18）：[教學診斷 phases 1～5 結案](journal/2026-11-18-1794945415-pv38-overnight-teaching-diagnostics.txt)。集合擬合未得淨局部收益；student-visited set32 局部 9/12，但已見整集均差 −11.9146s；修正解析 teacher 控制器 +15.4740s（95% CI [6.8264, 24.8482]），不代表 NN 改善。phase6 另行進行、結果未就緒；非新官方模型版本，歷史結論保留。


Pv38 研究完成附錄（2026-11-18）：[phase6 暖啟動配對 NN teacher-target 適應](journal/2026-11-18-1794948671-pv38-matched-teacher-nn-adaptation.txt)。兩臂各100 updates／2000 episodes；terminal100 已見 common64 候選減對照 −0.313020833s，95% CI [−4.662194010, 3.998476563]，未證實改善，亦不證明等價，不採用或晉升。計算產物與閉合已驗證，主程序退出碼未知（null）；不宣稱 trainer exit0。此日期附錄僅取代舊 phase6 進行中狀態，舊 phases1～5 報告與條目保留為歷史快照；phase7 診斷待完成，無其結果。

Pv38 研究附錄（2026-11-18）：[phase7 緊急目標曝露診斷](journal/2026-11-18-1794950249-pv38-urgent-target-exposure.txt)。首次 rollout 僅20/85836（0.02330024698%）改動；關鍵 argmax 擬合仍弱，非因果或強度證明。phase6 −0.3130s、CI跨零結論保留；此條僅取代 phase7 待完成狀態。phase8 另行執行中、結果未就緒，無新版本或晉升。

Pv38 phase8 結案（2026-11-18）：[緊急 teacher 權重100倍完成報告](journal/2026-11-18-1794960787-pv38-urgent-weight-completion.txt)。100 updates／2000 episodes；terminal100 已見 common64 候選69.6955729167s vs 重用未加權對照71.7296875s，差−2.0341145833s、95% CI [−7.2286458333, 3.2496419271]，未證實改善或等價，不晉升。05:39:28 HKT 計算完成、phase8 exit0；08時段完成獨立驗證及本次文件結案，僅取代舊 phase8 pending 狀態。舊 STOP、phase6 UNKNOWN exit 與全部歷史報告保留；H1 特徵僅為未實作假說。

Pv38 H1 非正式表徵研究結案（2026-11-18）：[加性18維完成報告](journal/2026-11-18-1794964970-pv38-h1-additive-representation.txt)。100 updates／2000 episodes／8502308幀；terminal100 已見common64差+0.2989583333s，95% CI [-5.1351822917, 5.7834830729]，未證實整局收益或等價，不晉升、未指派Pv39；局部一例改善非泛化證明。僅取代H1未實作狀態，保留全部歷史結論。

Pv38 初始 teacher-only 梯度診斷（2026-11-18）：[首批機制紀錄](journal/2026-11-18-1794966928-pv38-initial-teacher-gradient-audit.txt)。85836狀態中20緊急；legacy／零H1全actor緊急加權範數比1.1252%／1.09763%，cos +0.073586／+0.062523；共同舊梯度精確相同，新增H1區塊僅弱負向。非訓練、非PPO抵消或收益證明；歷史H1整局CI跨零結論保留，無Pv39或晉升。

Pv39 planned／formal NOT STARTED（2026-11-18）：[hard-start40＋E40 預註冊](journal/2026-11-18-1794984309-pv39-hard-start40-preregistration.txt)。沿用P35／Pv38架構與同一未訓練u0、fresh both Adam；固定1000 updates×40，正常難度BEST為PRIMARY、terminal1000為SECONDARY，對應舊正式Pv38已見64配對CI。工程gates已完成；整套介入不隔離難度與batch size，無強度或晉升結論。須先提交本預註冊才正式計算，歷史結論保留。

Pv39 正式完成（2026-11-18）：[hard-start40＋E40結案](journal/2026-11-18-1795011071-pv39-hard-start40-results.txt)。1000 updates／40000 episodes／75526395幀，22:03:36 HKT完成，端到端7.1935612078小時；PRIMARY BEST790 vs原Pv38 BEST190差+0.66875s，95% CI [-4.67501953125, 6.232298177083333]；SECONDARY terminal1000差−1.73203125s，CI [-6.7480078125, 3.2167447916666663]。兩CI跨零，未證實正常難度改善或等價，不晉升、不依final換BEST；整套介入與算力未匹配、單seed17及已見64限制保留。僅取代planned／formal NOT STARTED現況，原預註冊條目與歷史陰性結論保留。
