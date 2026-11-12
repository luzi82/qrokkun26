# AGENTS.md

本檔同時約束實作 Agent 與 Code Review Agent：抓住真實工程錯誤，避免吹毛求疵與測試膨脹。

## 1. 專案定位與環境速查（Environment & Paths）

受控內部研究與遊戲 AI 專案（Godot 4 + Python `qrokkun_env` / `qrokkun_ai`）。

### 本機（`ovm-hms-26a`）

- 工作區：`/storage/hermes/project/qrokkun26`
- CPU 測試 Python：`/storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python`
- 執行 pytest 時設定（加上 `--ignore=tmp` 與 `GIT_CEILING_DIRECTORIES` 避免收集歷史暫存快照）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  GIT_CEILING_DIRECTORIES="$TMPDIR" PYTHONPATH=. \
  /storage/hermes/home/dev_grok_0/venv-qrokkun-cpu/bin/python -m pytest --ignore=tmp -q
```

### GPU 訓練機（`GB10`）

- 主機：`ovm-hms-bot@192.168.0.41`
- 路徑：`/home/ovm-hms-bot/projects/qrokkun26/qrokkun26`
- 未獲明確指示，不得干擾正在執行的訓練工作。

### NAS

- 唯讀：`/mnt/minion/88_minion_ro/projects/qrokkun26`
- 讀寫：`/mnt/minion/88_minion_rw/projects/qrokkun26`（含 `runs/`、`assets/`、`demo_videos/`）
- 硬體與目錄規範：`/mnt/minion/88_luzi82_to_minion_ro/projects/qrokkun26-ai/guideline.md`

### 版本與實驗紀錄

- 維護於 `docs/AGENT_VERSIONS.md`、`docs/journal/` 與根目錄 `README.md`。
- 不擅自覆寫或刪除歷史實驗產物與歷史結論。
- 決策變更須明確標註取代關係（superseded decisions）。

## 2. 反吹毛求疵工程守則（Anti-Nitpicking & Engineering Guardrails）

以下五條同時約束實作與 Code Review。

### 守則 1：威脅模型定位

防真實工程錯誤，不防自我蓄意偽造。

本專案運行於受控單人／內部研究環境，並非對外公開的不可信輸入伺服器。驗證只針對真實會發生的錯誤：

- 物理／獎勵／演算法數學正確性
- RNG 污染與跨執行緒競態
- 張量維度／dtype／NaN／Inf 異常
- 孤兒程序、檔案或資料庫連線洩漏
- 磁碟空間不足、意外覆寫既有實驗輸出

禁止假設自己的訓練器或腳本會蓄意竄改內部產物、重簽 SHA256 封條來欺騙自己的比較器。禁止為此堆疊多重防偽邏輯與重封裝測試。

### 守則 2：測試金字塔與反笛卡兒積硬預算

禁止用重型 E2E／檔案 I/O 測試窮舉純邏輯分支。

- Schema、欄位缺漏、型別檢查、參數邊界：在記憶體內直接呼叫目標函式（毫秒級）。
- 檔案複製、壓縮或完整環境重播：每條關鍵路徑只留 **1～3** 個代表性案例，確認接線正確。

禁止無交叉分支意義的 `@pytest.mark.parametrize` 連乘。

- 共用同一驗證迴圈或同一個 helper 的多個欄位／階段：挑 **1～2** 個代表值。
- 程式碼裡沒有交叉條件分支時，禁止多層 `parametrize` 相乘（例如幾十個欄位 × 單邊／雙邊 × 多種壞值）。

數量與時間預算（適用於新增版本模組 `v27+`，不強制追溯重寫既有已封存模組）：

- 單一版本模組（`qrokkun_ai/vXX/tests`）測試數量控制在 **15～45** 項以內。
- 單執行緒 CPU 目標 **< 30 秒**，硬上限 **90 秒**。
- fixture 的 `--cap`、`--episodes`、`--hidden` 壓到能觸發目標行為的最小規模。

### 守則 3：子行程與訊號中斷的合作式邊界

CLI 參數解析（`argparse`、`--help`、錯誤回傳碼）與例外轉換，一律在行程內呼叫入口函式。禁止為了測參數錯誤而啟動會載入 PyTorch 的真實子行程。

訊號處理採合作式取消（Cooperative Cancellation）：

- 驗證執行中收到 `SIGINT`／`SIGTERM` 能終止並回收直接子行程、標記中斷狀態、且不啟動後續階段。
- 真實子行程測試最多 **1～2** 個代表性案例。
- 禁止攔截 `signal.signal`，針對「寫完檔案到恢復 handler 退出前幾毫秒」排列微秒級訊號縫隙。

### 守則 4：Code Review 收斂門檻

只有下列三類可列為 Blocker／Major：

1. 物理／演算法／RNG 數值與可重現性錯誤。
2. 梯度／權重／評估指標計算，或實驗身分鎖定錯誤。
3. 資源洩漏：孤兒程序、未關閉 DB／檔案、磁碟爆滿、覆寫正式產物。

下列意見一律標為 **Nitpick / Out-of-Scope**，直接結案。禁止為此再開修補與測試膨脹迴圈：

- 假設手動破壞內部檔案結構並重封裝。
- 退出瞬間的微秒級訊號視窗。
- 重複測試 Python／PyTorch 標準庫既有行為。

### 守則 5：程式碼指紋隔離與暫存衛生

新模組（`v27+`）若計算 `source_tree_sha` 等程式碼指紋，只納入影響執行與數值的生產程式碼，排除 `**/tests/**` 與文件檔案，避免維護測試時破壞實驗封條（既有 `v26` 已定義的指紋範圍維持原契約，不追溯改動）。

Pytest `--basetemp` 與大型中繼暫存寫入核准的 `$TMPDIR`，驗證後清理；或遵守 `guideline.md` 的暫存規範。嚴禁污染 `${HOME}`。禁止在 `<repo>/tmp/` 殘留數 GB 或數萬個測試暫存檔。
