# Pv29c / Pv29m：u10、E20 配對預註冊

狀態：實作及微型 CPU 驗證；尚未正式執行 GB10 配對。此版本只取代 Pv28 的實驗預算、排程與比較方式，歷史結果保持原意。

正式預設固定 seed17、u10、每次 20 episodes、cap7200、hidden472／obs84。Pv29c 單行程不建立 Pool；Pv29m 使用 persistent SharedMemory spawn、10 workers、pin-on，沿用 Pv28 最大頻率優先 selector 與動態佇列。若 CPU budget 或可辨識最高頻率核心不足 10 個，直接失敗，不輪用或溢出至小核心。

沿用 Pv28 生產核心數值：CPU threads1／batch1、P/S 獨立每 episode generator、canonical index merge、Python float episode GAE、中央 deterministic CUDA clearance teacher／PPO、獎勵與 seed offsets 完全不變。從原始 Pv21 u400 actor+critic 與全新 Adam 開始；凍結 stochastic CPU Sv24 best u980。輸入路徑及檔案 SHA 沿用 v25、狀態身分沿用 v24/v26，並明確驗證初始 P 狀態；不是從 Pv28 u1000 接續。

Checkpoint cadence=1：保存 u0..u10，檔名六位數。Validation cadence=10：僅 u0/u10，seeds600000..600015。兩者在 manifest 分別記錄。checkpoint-only 更新不執行 validation，也不產生 evaluation 檔；validation 使用 episode-local RNG，不改變訓練 permutation generator。正式禁止 resume；繼承的非正式技術恢復介面未擴充。

兩臂依 c→m 完成後才比較；每次線性讀取 u0..u10 所有具名 actor+critic resolved tensor raw bits，shape/dtype 亦須一致。PRIMARY verdict 只看 terminal u10：0=WEIGHTPASS、2=WEIGHTFAIL；中間差異只報告輔助資訊，允許重新一致，不使用二分搜尋、allclose 或檔案雜湊代替位元比較。執行異常／nonfinite／資源錯誤=3，owner stop=130，均不是科學差異或通過。輸出目錄以 exclusive mkdir 保護。保留取消清理、SharedMemory/PID 回收及 frozen S 檢查。

正式啟動命令（僅供父層完成獨立測試、readiness review、部署及 CUDA smoke 後使用；本次未執行）：

```bash
python -m qrokkun_ai.v29.supervise --out-dir /mnt/minion/88_minion_rw/projects/qrokkun26/runs/YYYYMMDD-HHMM-pv29-cm-u10 --source-commit <40-hex-sha>
python -m qrokkun_ai.v29.compare /path/to/pair
```

source_tree_sha 沿用 Pv28 的全 qrokkun_ai／qrokkun_env 生產 Python 範圍，涵蓋 v29 與繼承依賴，排除 tests／docs／cache；顯式 source-commit 不依賴 .git，支援原始碼 archive。

計時：pool_init 含 spawn/ready；weight_sync 含 CPU 複製與 worker ACK；pure_rollout 為各 worker 每次 collect 的 episode 計算時間累加後取最大值（serial 為加總），排除 IPC/merge，並非 collection wall latency。rollout_s 是完整訓練 collect wall（含派發、等待、canonical copy）。batch_teacher_gae 為組批／teacher／GAE；collector=pool_init+weight_sync+rollout+batch_teacher_gae+pool_shutdown；PPO、validation、checkpoint 分列，total 含啟動、I/O、清理。FPS 及 c/m wall ratio 隨各時間分母報告；pure_rollout 是 worker busy critical lane 指標，不能與其他 wall 欄相加。validation 自身完整計時，不混入訓練 frames。checkpoint 時間在完成後累計，checkpoint 內 snapshot 不含該次寫入耗時，以 closed.json 為最終總計。

單一配對 seed17 不構成統計信賴區間。沒有自動重跑、正式續跑、trace、下一版本、heldout 或 promotion。正式 GPU 重現性與吞吐量仍待實測，不預先宣稱通過。
