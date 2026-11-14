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
- 狀態（2026-11-14）：實作、本地測試（`v28` 18 passed、全套 279 passed / 28 skipped）與 `gpt-6.1-sol` 審查完成，待部署 GB10 執行 `u5` ➔ `u100` ➔ `u1000`。來源：[README](../qrokkun_ai/v28/README.md)、[概念規格書](../concepts_brief.md)、[事前合約](journal/2026-11-14-1794642996-pv28-cm-preregistration-contract.txt)。
