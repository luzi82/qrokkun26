# Pv31m（含條件備援 Pv31c）：單流外提特徵與延遲同步 PPO 加速（u10、E20）

狀態：第一階段 Eager 結構清理實作；取代 `Pv30` 的雙 CUDA streams＋雙 `ThreadPoolExecutor` PPO 派發（保留歷史 `Pv29`／`Pv30` 結論不變）。

## 設計重點
1. **單流序列派發（`cuda_hoisted_single_stream`）**：移除 `ThreadPoolExecutor` 與非預設 CUDA streams，消除 CPython GIL 爭用與每個 minibatch 三段式同步屏障。
2. **無參數特徵與老師機率外提**：在進入 4 個 PPO epochs 迴圈前，於 `torch.no_grad()` 下整批計算一次 `all_feat = net.features(player, bullets, pad)` 與 `all_t_probs = F.softmax(exp_logits, dim=-1)`，迴圈內直接索引 `all_feat[mb]`／`all_t_probs[mb]`，消除約 55% 的碎 kernel 發射與重複幾何計算，且維持與逐 minibatch 計算 `0-ULP` 完全相等。
3. **零迴圈內 Host-GPU 阻塞**：minibatch 內以 0-dim `float64` 張量累加損失與指標、以布林張量紀錄 `a_norm / c_norm / actor_loss / value_loss` 非有限值，於 4 個 epochs 完成後才執行單次 `.item()` 同步與 `finite(net.state_dict())` 檢查。
4. **主行程單顆大核綁定**：於 `EpisodePool` 建立並綁定 10 個 CPU rollout workers 後，將主行程綁定於最高頻大核（GB10 `CPU 5` Cortex-X925 `3.9 GHz`），避免主行程與 PyTorch C++ autograd 執行緒漂移至 `2.8 GHz` A725 小核或跨叢集 DSU 乒乓。

## 比較與條件備援規則
- 正式預設：`seed=17`、`updates=10`、`episodes=20`、`cap=7200`、`hidden=472`、`workers=10`、`pin-cores=on`，checkpoint cadence=1（`u0..u10`），validation cadence=10（`u0, u10`）。
- 由於 `Pv31` 在確定性 CUDA 下預期可 `0-ULP` 逐位元還原 `Pv30c`，優先單獨執行 `Pv31m` 並以 `--reference-dir` 比對本地雜湊驗證之封存 `Pv30c`（`u0..u10` 全部 12 個具名 Actor＋Critic 張量 raw bits，以 `u10` 為主結論：`0=WEIGHTPASS`、`2=WEIGHTFAIL`、`3=執行／契約錯誤`）。
- 僅當科學 invariants 完全一致但權重出現 `WEIGHTFAIL`（`exit 2`）時，才啟動同版本 `Pv31c` 對照臂以驗證 `Pv31c == Pv31m`。

```bash
python -m qrokkun_ai.v31.run --arm Pv31m --out-dir /path/to/run/Pv31m --source-commit <40-hex-sha>
python -m qrokkun_ai.v31.compare /path/to/run --reference-dir /local/copy/Pv30c --candidate-dir /path/to/run/Pv31m
```
