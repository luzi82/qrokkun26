# Pv34m：保守不可能碰撞的 P 觀測過濾

本實驗尚未正式執行，沒有改善結論。對照為已完成 Pv33m；只改 P 看見哪些子彈，環境子彈、物理、碰撞、S 的觀測／動作及亂數不變。這是充分條件，不追求最大移除量。模型使用時必須呼叫 `qrokkun_ai.v34.adapter.encode_obs(env)`，不能搭配舊 encoder。

## 有限生命期與安全證明

假設目前 Gv20／LearnedEnv 的固定 dt=1/60、線性固定速度與 float32 積分。P 中心合法矩形為 [14,306]×[26,226]；最大速度 102，碰撞半徑 4，子彈採各自 `b.radius`（現行 constructor 半徑 2.6）。真正移除矩形是 `C.BULLET_CULL=(-24,344,-24,264)`，不是畫面 FIELD。每步順序為 P 移動、子彈移動／新生停留、cull、碰撞；等號與切線都保留。

設 u=2^-24、η=2^-149。先精確取得實際量化位移 q_i=f32(v_i dt)。每次加法的座標誤差上界 E_x=u(344+|q_x|)+η、E_y=u(264+|q_y|)+η，因加法前所有尚存座標在 cull 內；包含最後被移除的步驟。m 次移動後實際子彈在 b+m q 加上各軸 ±m E_i 內。計算時另以 128 倍 binary64 epsilon 乘所有項絕對量和作判定保護，不以實測調整門檻。

任一軸 |q_i|>E_i，便有向外單調最小進度 |q_i|-E_i。距該方向真正 cull 邊界 D_i，則 N_i=ceil(D_i/(|q_i|-E_i))+2 次移動後必定已出界。為避免巨大商的 binary64 條件數問題，只接受商 ≤2^32 的軸，否則該軸不提供證明（不是到此幀數後假設安全）。取 N=min N_i 是實際生命期的保守上界，包含最後被移除且不檢碰撞的步驟。新生 `moved=False` 另加 d=1 個不移動步驟；並未忽略這一幀的 P 移動。無有效上界、非有限值、非標準 dt、已在 cull 外、極端位移或不合法 P 起點，一律保留。

任意動作序列（包含任意轉彎）下，P 在 k 幀後位於初始位置半徑 k V 的圓內，V=102 dt(1+u)^2+sqrt(2)u·512。前項包住速度及乘法量化，後項包住座標加法；矩形截斷不增加相對合法起點的距離。此為可達集合外包，不使用目前 heading 或 policy，也不枚舉九個固定方向。

取初始相對位置向量單位方向 n，碰撞總半徑 R=4+b.radius，E=hypot(E_x,E_y)。所有未移除幀的分離投影下界為

`|b-P| - R - d V + m (n·q - V - E)`，其中 `0 ≤ m ≤ N`。

這是仿射式，兩端最小值仍嚴格大於 binary64 保護量才移除。另一充分條件以各軸端點及 ±NE_i 外包整條軌跡，若全程在 P 合法矩形膨脹 R 之外，同樣可移除。兩條證明均覆蓋所有中間實數時間，故必然覆蓋離散碰撞幀。保守外包可能保留實際無害彈；無害與「當前遠離」、「速度較快」、「12 幀內不撞」不同。

每顆子彈固定數量算術，過濾 O(目前活彈數)，無逐彈 7200 幀模擬。所有活彈先過濾，再交原 v31 穩定歐氏距離排序及截斷 64；原 padding、6 P features、4 bullet features、top12、84 維輸入及 native PlayerV21 h472 不變。

## 訓練與評估契約

從 u0 重新初始化所有 actor／critic，沿用 v33 `random_player`：CPU fork_rng(devices=[])、seed17、不改呼叫端 RNG。原始 state SHA 必須為 `49a7c9426a1054fc000caf359fc1a0a3c90db49adfe6a53c3275350a9b5a7942`，全新 Adam，不載 P 預訓練。S 完全沿用 v33 Sv24u980 檔案、state、metadata 鎖定。模型建構後重設 Python／NumPy／Torch CPU／CUDA seed17，permutation27000017。

v34 adapter 以 finally 恢復 v31 hooks；importable worker wrapper 在 spawn 子程序重新安裝，數值核心沿用 v31，資源生命週期沿用 v32。P actor、critic、軌跡、批次 clearance／teacher 全部使用同一過濾觀測。teacher 演算法、權重、loss 不變，但數值輸出可隨觀測改變，不主張位元相同。

正式預算 u1000、每更新20集、cap7200、10 persistent CPU workers、SharedMemory 原始 episode index 合併、父程序 spawn 後 pin5；保留 v31 CUDA single stream／feature hoist／deferred sync／deterministic PPO，不啟用 AMP、TF32、compile、graphs。train scene seed100000+(u-1)20..+19，P offset26000000，S train25000000／validation24000000。相同 updates／episodes 不代表相同 frames 或時間；各更新保存實際 frames、各集死亡／censor 與計時，以及 considered／discarded／retained 統計（train 與 validation 分開在各自記錄）。

u0 及每10更新固定16個600000..600015 seeds，獨立 RNG，P argmax／S stochastic；BEST 只比較整數 sum frames 嚴格增加，平手取早，u0 可獲選。不以歷史 u0 分數設門檻。保存101份六位 checkpoint、101份 validation、1000行更新記錄，checkpoint 含完整 Adam 及 RNG 來源。不提供 resume、自動重跑、seed search、early stop。來源指紋以靜態 import 相依閉包包含使用的歷史 production Python 與 v34 production、控制 pin；排除 tests 與文件。`--source-commit` 接受明確40位hex，不依賴 .git。

控制 pin 為本目錄 `control-reference.json`，載入會驗113個檔案 hash、完成狀態、協定及選定 checkpoint。final 僅正式完成後獨佔 claim 一次，使用原 benchmark800000..800063，非新 heldout；best 對 Pv33 best910 為 primary、terminal1000 對 terminal1000 為 secondary，paired bootstrap 僅描述。

## 執行入口與限制

`python -m qrokkun_ai.v34.supervise --out-dir <新目錄> --source-commit <40hex>` 啟動受監督流程，預設正式 CUDA。train 成功且正式 closure／來源 preflight 通過後，才啟動一次 v34 final；final 自行於 preflight 後獨佔 claim。非正式模式只執行 train，失敗或收到停止訊號不啟動後續階段。直接子行程先 TERM 等待最多10秒，再 KILL 等待最多5秒；退出、已觀察失敗與停止事件記錄於相鄰 `<新目錄>.launcher/supervisor.json`，stdout 亦保存在該目錄。不重試、不 resume。

單階段入口仍為 `python -m qrokkun_ai.v34.run train`／`final`。小型 CPU 測試可加 `--nonformal --synthetic --device cpu --updates 1 --episodes 1 --validation 1 --workers 1 --pin-cores off --hidden 8 --cap 3`，不讀對照。CUDA u1 smoke 可使用 `--nonformal --device cuda --updates 1 --episodes 20 --validation 16 --workers 10 --cap 7200` 與真實 S，無 FINAL。CUDA smoke、資源釋放與更新後權重的實際執行驗證尚待完成；本次未執行 CUDA、真實訓練或 final。

物理／速度／矩形規則若改變，須重新推導，不能直接套用此證明。過濾是觀測消融，不是安全控制器；正式結果前不宣稱學習效果。
