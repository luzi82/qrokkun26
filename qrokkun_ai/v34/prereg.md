# Pv34m 預先登記

唯一處理變項：在 P 觀測距離排序與截斷前，移除可保守證明在實際移除前不可能碰撞的子彈。對照為 pinned、已完成的 Pv33m。數學假設、誤差界、保守限制及全部訓練契約見 README。

假設為減少無害近彈佔用64格可能提升存活表現；正式執行前不主張改善。兩組從相同 seed17 原始 h472 全權重出發，各 u1000／E20。frames 與時間為實測結果，不視為相等預算。

主要比較：各組依固定 validation 整數 sum frames 嚴格最佳 checkpoint，在既有800000..800063共同 benchmark 的配對存活秒差；平手選早，u0 合格。次要比較 terminal u1000。固定 bootstrap seed17、10000 samples、95% interval，僅作描述，該 window 並非新 heldout。不做 seed search、得分導向 retry、early stop 或改動 teacher。

正式命令介面：`python -m qrokkun_ai.v34.run train --out-dir <new-run-directory> --source-commit <40hex>`。正式預設 CUDA、u1000、E20、cap7200、10 workers、pin on、每10更新16集 validation。final 為獨立一次性 claim 命令，未完成正式 closure 不能執行。非正式 smoke 可使用 `--nonformal --updates 1 --episodes 20 --validation 16 --cap 7200 --device cuda`，沒有 FINAL。CPU 測試可搭配 synthetic、小 hidden／cap；非正式訓練不讀 Pv33 run。

本次僅作者 CPU 驗證。獨立 CPU 驗證、readiness、immutable deployment、真實 CUDA u1 smoke、正式 u1000 與 final 均由父流程執行。本文件不記錄正式結果。
