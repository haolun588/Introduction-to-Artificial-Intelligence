# FGSM C1：小 epsilon 正式對照

執行日期：2026/11/15–11/16（Asia/Taipei）。狀態：三輪訓練、獨立推論與 CSV 核對完成；驗證與 Kaggle 回報均未改善。使用者已於 11/16 回報分數，未登入平台獨立核驗。

## 目的與預先固定的限制

沿用 [FGSM 校準](FGSM_Calibration_Report.md) 的實作與 `epsilon=0.00208093`，測試局部 embedding sign-gradient 擾動能否改善原 holdout 的 Macro-F1 及三類 F1。對照為 `weighted_half_epoch_control`；唯一主要變因是 FGSM（clean／adversarial loss 各半）。保留 seed 42、sqrt-inverse、2e-5、batch 2／累積 8、原 8,100／900 切分、空白前處理、AMP、梯度檢查點、半輪評估及 3 epochs。

本輪只跑 C1 一次，訓練程序硬上限 3,600 秒；非有限梯度／loss、無成功更新、重載不一致即停止。不改 epsilon、不合併 quarter-inverse 或 layer LR，不自動追加 C2。訓練完成後做兩次 validation 推論（AMP／FP32）及一次 FP32 benchmark 推論，上限 15 分鐘，核對與原最佳 CSV 的差異。

決策資訊：相對控制組的 Macro-F1、各類 F1、六點曲線、重載與精度一致性。依既有計畫，增益 ≥0.3 個百分點優先做額外切分確認；小增益不稱為突破。訓練前尚未得到 Kaggle 回報，不代為提交；後續回報見下方登錄。

## 執行

```powershell
.venv/Scripts/python.exe main.py train --device cuda --epochs 3 --eval-interval 0.5 --class-weighting sqrt-inverse --learning-rate 2e-5 --fgsm-epsilon 0.00208093 --output-dir outputs/weighted_fgsm_low --checkpoint outputs/weighted_fgsm_low/checkpoint
```

程序以 `subprocess.run(..., timeout=3600)` 執行，stdout／stderr 保存於 `outputs/weighted_fgsm_low.training.log`。程式自行驗證官方基底 hash，輸出 `run_config.json`、`history.json`、`curves.png`、checkpoint 與 `summary.json`；既有候選不覆寫。

## 正式訓練結果

| Epoch | Train Macro-F1 | Validation Macro-F1 |
|---|---:|---:|
| 0.5 | 88.1077% | 88.3716% |
| 1.0 | 92.8691% | 89.6103% |
| 1.5 | 94.2055% | 90.0556% |
| **2.0** | **96.7604%** | **91.7264%** |
| 2.5 | 97.3663% | 90.9414% |
| 3.0 | 98.0117% | 91.2514% |

依六個評估點的 validation Macro-F1 選擇 epoch **2.0**，未採用最後一輪權重。訓練及重載共 **2,424.16 秒（40.40 分鐘）**，不含 Python 啟動／基底載入；低於程序 60 分鐘上限。欄位定義：`history.json` 各列 `seconds` 是自該 epoch 開始的經過時間（每個 epoch 開頭重設，含 epoch 內先前評估點與本評估點的評估時間），x.5 與 x.0 兩列重疊，不可全部相加（相加得 3,576 秒）；只加整數 epoch 列（772.8＋760.5＋875.0＝2,408.3 秒）才接近 `total_seconds`，差額約 16 秒推測為最後的模型重載（僅推測）。1,521 次更新嘗試、1,520 次成功、1 次 AMP 跳過，scheduler 僅前進 1,520 次。Peak allocated VRAM **1,905.74 MiB**，模型重載 loss／混淆矩陣一致。

| 指標 | 共同控制組 | C1 最佳 | 差（百分點） |
|---|---:|---:|---:|
| Macro-F1 | 92.6488% | 91.7264% | **−0.9224** |
| negative F1 | 97.0359% | 96.8847% | −0.1511 |
| neutral F1 | 86.4662% | 85.2713% | −1.1948 |
| positive F1 | 94.4444% | 93.0233% | −1.4212 |

三類 F1 均下降，C1 未達額外切分確認門檻。混淆矩陣（順序 negative／neutral／positive，列為真值）為 `[[622,10,4],[20,110,4],[6,4,120]]`；錯判 48 筆，控制組 44 筆。訓練 F1 持續上升而最佳驗證停在 epoch 2，不能以訓練分數宣稱泛化改善。

證據：[summary](../../outputs/weighted_fgsm_low/summary.json)、[run config](../../outputs/weighted_fgsm_low/run_config.json)、[history](../../outputs/weighted_fgsm_low/history.json)、[曲線](../../outputs/weighted_fgsm_low/curves.png)、[原始日誌](../../outputs/weighted_fgsm_low.training.log)。曲線已目視確認六個點、圖例與軸標示完整。監看期間曾暫未收到收尾結果，最終以退出碼 0、六點 history 與成功重載確認正常完成；沒有重跑訓練。

## 獨立推論與提交檔

使用 [本輪推論腳本](../../outputs/weighted_fgsm_low/evaluate_candidate.py) 呼叫既有 `audit_precision.infer`，完成兩次 validation（AMP／FP32）與一次 FP32 benchmark 推論，均正常退出。原控制組 20 個設定／來源欄位逐一相同；FGSM epsilon、固定 ID、labels、模型來源與資料 hash 已核對。

900 筆 validation 的 AMP／FP32 標籤差異 **0**，Macro-F1 均為 **91.7264%**，錯判均為 48 筆。AMP loss／混淆矩陣重現 checkpoint；機率最大差 **0.00125492**、平均差 **0.00001914**。這支持保留 FP32 正式推論，並不證明訓練中的 AMP／FP32 梯度相同。兩次 validation 及 benchmark 的純推論時間為 5.02／3.89／4.44 秒，不含載入與 warmup；未額外執行 benchmark AMP。

[predictions.csv](../../outputs/weighted_fgsm_low/predictions.csv) 已核對 1,000 筆、`id,pred`、唯一 ID、原始 benchmark 順序及合法標籤。與共同控制組相差 **33 筆**、目前 Kaggle 首選 weighted_quarter 相差 **40 筆**、原候選 weighted_lr3e5 相差 **39 筆**，並與全部 12 份既有 CSV 不同。沒有手改答案，沒有代為提交。

- CSV SHA-256：`fe1d1640ccfba7ef65b861258b615083c5bf823a03d1e697db2672d86cfe3c2b`
- 模型 SHA-256：`308b801c5ac85dd9cfcfeba1b10c350fc48970bacce27a52af0f59ab19348f2f`
- [完整比較與來源 hash](../../outputs/weighted_fgsm_low/control_comparison.json)、[逐筆機率](../../outputs/weighted_fgsm_low/precision/probabilities.csv)、[錯判](../../outputs/weighted_fgsm_low/precision/errors.csv)。

## 決策與交接

C1 的整體及三類 F1 均下降，Kaggle 回報亦未改善，**不取代控制組／Kaggle 首選、不安排 C1 額外切分確認**。CSV 保留為已完成實驗的對照；不能將本機 91.73% 當成 Kaggle 分數。

核心完整訓練已完成 **4／9**，剩餘上限 5 次。下一項為原定兩個 epsilon 中的最後一個 **C2：0.00416186**，其假設是較強局部擾動能否減少目前訓練上升、驗證停滯的現象；這是待驗證假設，C1 本身沒有提供改善證據。其餘仍同共同控制組，上限一次 3 epochs／60 分鐘；使用與 C1 相同的非有限梯度、無成功更新、重載失敗停止條件。若 C2 也無優勢，停止本輪 FGSM 幅度搜尋，不自動添加 epsilon 或混合失敗設定。C2 **尚未啟動**。

```powershell
# 下一項指令，尚未執行
.venv/Scripts/python.exe main.py train --device cuda --epochs 3 --eval-interval 0.5 --class-weighting sqrt-inverse --learning-rate 2e-5 --fgsm-epsilon 0.00416186 --output-dir outputs/weighted_fgsm_high --checkpoint outputs/weighted_fgsm_high/checkpoint
```

本輪已完成訓練、六點曲線、模型重載、獨立推論、CSV、實驗索引與使用者分數回報登錄。剩餘工作為後續研究及最終交件，不把本輪完成稱為達成 Kaggle 92%。

## 訓練完成時的收尾檢核（Kaggle 回報前）

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：已讀 AGENTS，修改限 HW3，指定基底與固定課程切分；未提交平台。 |
| 目的完成 | 通過：完成唯一一次 C1 三輪、六點曲線、選模重載、精度核對及 CSV。 |
| 查重與成本 | 通過：沿用上一輪 35 項測試／短跑與既有工具，訓練程式未變故不重跑測試；只新增 C1 與三次必要推論。 |
| 既有工作保全 | 通過：先看相關 diff，保留上一輪未提交工作；12 個既有候選的 CSV、10 份 checkpoint 與角色紀錄核對不變。 |
| 驗證證據 | 通過：退出碼 0、六點 history、1,520 次 scheduler／成功更新一致、重載與獨立 AMP 指標重現、900 筆精度標籤一致及 1,000 筆 CSV；曲線目視核對。 |
| 實驗可追溯 | 通過：設定、模型／CSV／程式 hash 與比較證據一致；registry、refresh／--check，13 runs／11 checkpoints。 |
| 文件與狀態 | 通過：README、計畫、課程進度、索引與報告同步，連結／diff 檢查通過；C2 尚未執行。 |
| 交付誠實 | 通過：如實記錄 −0.9224 個百分點與三類下降；Kaggle 未知，未宣稱達標或交件。 |

檢核：適用項 **8／8 通過**；不適用：無。

## 2026/11/16 Kaggle 回報登錄

使用者回報 C1 `weighted_fgsm_low` 為 **0.8909**，比共同控制組對應的 weighted_sqrt 0.9049 低 **1.40 個百分點**，比目前首選 weighted_quarter 0.9122 低 **2.13 個百分點**。本機驗證與 Kaggle 回報均未改善，保留現有候選；這不是平台獨立核驗，也不推論 benchmark 逐筆標籤。

registry 已標示 `user_reported`、回報日期及來源，總表同步更新。沒有新增訓練、推論或平台操作；完整訓練預算維持 **4／9**，C2 尚未啟動，預先固定的 epsilon 與研究上限不變。

本次登錄檢核：適用項 **8／8 通過**；不適用：無。規範與範圍：已讀 AGENTS，僅更新 HW3 分數與文件；目的完成：分數、來源、比較及狀態已登錄；查重與成本：沿用既有產物，未重訓或重推論；既有工作保全：檢視 diff，前後核對模型／CSV／來源與角色不變；驗證證據：refresh、--check、文件連結與 diff 檢查；實驗可追溯：13 runs／11 checkpoints 對應核對；文件與狀態：README、計畫、報告與總表一致；交付誠實：明確區分使用者回報與平台核驗，未宣稱達標或交件。
