# FGSM C2 正式對照

2026/11/16。狀態：三輪訓練、獨立推論與 CSV 完成；Kaggle 已收到使用者回報，未獨立核驗平台。

假設：較強 embedding 擾動是否改善 C1 訓練上升、驗證停滯的現象。沿用已測試的 FGSM，唯一主要變因為對控制組加入 epsilon=0.00416186、clean／adversarial 各半；相對 C1 只改 epsilon。保留 seed 42、sqrt-inverse、2e-5、batch 2／累積 8、半輪評估、AMP、原 8,100／900 切分與指定 BERT 基底。

上限一次三輪、程序 3,600 秒；非有限梯度／loss、無成功更新或重載不一致即停止。完成條件：六點曲線、最佳 checkpoint 重載、兩次 validation 精度比較、一次 FP32 benchmark CSV、來源 hash 與 registry 核對。若 C2 也無優勢即停止 FGSM 幅度搜尋，不自動追加實驗。Kaggle 由使用者提交。

既有 C1 分數已登錄，沒有重複 C1；完整訓練預算已完成 4／9，本輪預留一次。產物：`outputs/weighted_fgsm_high/`；日誌：`outputs/weighted_fgsm_high.training.log`。恢復時先檢查日誌與 summary，不重啟已執行的訓練。啟動前已讀 AGENTS、檢查 Git 狀態與路徑不存在，保留前輪未提交修改。

## 結果與證據

| Epoch | Train F1 | Validation F1 |
|---|---:|---:|
| 0.5 | 87.7480% | 87.7022% |
| 1.0 | 91.8233% | 88.8441% |
| 1.5 | 92.8730% | 88.5830% |
| 2.0 | 95.6264% | 89.4571% |
| 2.5 | 96.1374% | 90.6454% |
| 3.0 | 96.9839% | 90.9159% |

最佳 epoch 3.0，驗證 Macro-F1 **90.9159%**，相對控制組差 **-1.7330 個百分點**。三類 F1 差：{'negative': -0.4031, 'neutral': -3.3201, 'positive': -1.4757}（百分點）。

訓練與重載 2337.33 秒（38.96 分鐘），退出碼 0；欄位定義：`history.json` 各列 `seconds` 是自該 epoch 開始的經過時間（每個 epoch 開頭重設，含評估時間），x.5 與 x.0 兩列重疊，不可全部相加（相加得 3,488 秒）；只加整數 epoch 列（764.4＋763.4＋802.2＝2,330.0 秒）才接近 `total_seconds`，差額約 7 秒推測為最後的模型重載（僅推測）；1521 次嘗試／1520 次成功，AMP 跳過 1 次，scheduler=1520。顯存峰值 1905.74 MiB。六點曲線已目視核對。

獨立 AMP／FP32 驗證標籤差異 0 筆；AMP 重現 checkpoint，兩者 F1 為 90.9159%／90.9159%。不以此宣稱訓練梯度等價。

FP32 CSV 已核對 1,000 筆、id,pred、唯一 ID／原順序／合法標籤；與 weighted_quarter 差 40 筆、C1 差 23 筆，與所有既有 CSV 不同。SHA-256：`dd1fcd7f2396596716d3facbb9d31cb44826bb9833ae608aab7bd2502b92da42`。

證據：[summary](../../outputs/weighted_fgsm_high/summary.json)、[設定](../../outputs/weighted_fgsm_high/run_config.json)、[曲線](../../outputs/weighted_fgsm_high/curves.png)、[精度及來源 hash](../../outputs/weighted_fgsm_high/control_comparison.json)、[CSV](../../outputs/weighted_fgsm_high/predictions.csv)、[推論腳本](../../outputs/weighted_fgsm_high/evaluate_candidate.py)。沿用 C1 腳本，只改 C2 名稱／epsilon；兩次 validation 與一次 benchmark 推論，未額外重訓。

## 決策與交接

C1、C2 均未改善驗證，停止本輪 FGSM 幅度搜尋，不追加第三個 epsilon 或確認切分。保留 weighted_quarter 為 Kaggle 首選、weighted_sqrt 為驗證基準。核心完整訓練已用 **5／9**；剩餘 4 次為有條件確認預算，不要求跑滿。C2 Kaggle 回報未超越既有首選，未代提交。下一個具體工作為 weighted_quarter 的獨立封裝及乾淨環境推論演練；最終版本／學號與平台收件仍未確認。

## 訓練完成時的收尾檢核（Kaggle 回報前）

適用項 8／8 通過；不適用：無。規範與範圍：已讀 AGENTS，僅 HW3 與指定資料／模型；目的完成：三輪、曲線、重載與 CSV；查重與成本：一次預定 C2，沿用既有測試與推論工具，訓練程式未變；既有工作保全：保留未提交修改，13 個原實驗及 11 份 checkpoint／CSV／角色前後一致；驗證證據：退出碼、六點評估、scheduler、精度／CSV 與曲線核對；實驗可追溯：registry、refresh／--check，14 runs／12 checkpoints；文件與狀態：入口、計畫與連結更新；交付誠實：如實記錄下降、Kaggle 未知及未交件。

## 2026/11/16 Kaggle 回報登錄

使用者回報 FGSM C2 為 **0.9038**，比 C1 的 0.8909 高 **1.29 個百分點**，但比目前首選 weighted_quarter 的 0.9122 低 **0.84 個百分點**，亦比控制組對應的 weighted_sqrt 0.9049 低 **0.11 個百分點**。這是使用者回報，未登入平台獨立核驗。

維持 weighted_quarter 為 Kaggle 首選、停止本輪 FGSM 幅度搜尋。下一項仍為首選的獨立封裝與乾淨環境推論演練。本次僅登錄分數與文件，未新增訓練／推論／提交，完整訓練預算維持 **5／9**。

本次登錄檢核：適用項 **8／8 通過**；不適用：無。規範與範圍：已讀 AGENTS，僅 HW3；目的完成：分數、來源與比較已登錄；查重與成本：沿用既有產物，未新增實驗；既有工作保全：先看 diff，前後核對模型、CSV、角色與非目標紀錄不變；驗證證據：refresh、--check、連結與 diff 檢查通過；實驗可追溯：14 runs／12 checkpoints 核對；文件與狀態：報告、README、計畫與總表同步；交付誠實：使用者回報與平台核驗分開，未宣稱達標或交件。
