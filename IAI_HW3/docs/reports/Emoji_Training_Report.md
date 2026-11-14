> 歷史實驗報告；內文的「下一步」代表當時判斷。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# Emoji 簡短描述對照

2026/11/14 完成。最佳第 3 輪驗證 Macro-F1 **91.68%**，低於原加權模型 **92.65%**，下降 **0.97 個百分點**。依驗證結果不取代原模型；本輪仍產生 Kaggle 對照檔，尚無平台分數。

## 規則與設定

新增 `emoji-short-v1`：以 `preprocessing.py` 中固定 34 種描述取代支援的符號，例如 heart、crying face、thumbs up。描述不使用情緒類別標籤；同一符號連續出現只保留一次描述，保留分隔後再次出現的符號。支援單一符號附帶 variation selector；未支援的符號、膚色修飾及 ZWJ 組合保留原樣。ASCII 表情、原 URL、否定詞、hashtag 與標點保持原樣。這是有限集合的啟發式前處理，不是完整 emoji parser。

訓練、驗證與推論共用規則；checkpoint 保存前處理版本，推論依此讀取。預設仍為 `whitespace-v1`，既有 checkpoint 保持相容。已核對 run config 唯一差異為 `preprocessing`，其餘維持 2e-5、seed 42、反平方根加權 CE、固定 8,100／900 切分、3 epochs、長度 64、batch 2／累積 8、原排程與 AMP。

訓練集 203 筆改變（negative 88、neutral 28、positive 87），未知 token 由 272 降至 83；超過長度 64 的仍為 2 筆，最長仍為 67。未新增前處理後跨訓練／驗證集合的相同文字。完整規則副本與盤點存於 `outputs/emoji_short_audit/summary.json`。

## 結果

| Epoch | Train loss | Val loss | Train Macro-F1 | Val Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 0.149196 | 0.230266 | 92.65% | 89.82% |
| 2 | 0.076497 | 0.252470 | 96.50% | 89.62% |
| 3（入選） | 0.051425 | 0.278234 | 98.14% | **91.68%** |

評估 loss 為未加權 CE。訓練及重載驗證共 1,040.94 秒（17.35 分鐘），不含初始載入及 benchmark 推論；成功更新 1,517 次，AMP 略過 4 次，峰值配置顯存 1,904.24 MiB。

| 指標 | 原加權模型 | Emoji 描述 |
|---|---:|---:|
| Macro-F1 | 92.65% | 91.68% |
| negative F1 | 97.04% | 96.57% |
| neutral F1 | 86.47% | 84.73% |
| positive F1 | 94.44% | 93.75% |

最佳混淆矩陣（列真實、欄預測，negative／neutral／positive）：`[[619,14,3],[20,111,3],[7,3,120]]`。三類 F1 均未提升；減少未知 token 沒有在本輪轉為整體驗證改善。僅測此規則、seed 與設定，不能推論所有 emoji 轉文字方式均無效。驗證 loss 逐輪上升，不優先直接延長訓練。

## 驗證與重現

13 項測試通過，涵蓋舊模式相容、描述與重複符號、無 emoji 文字不變、未支援組合保留、重複處理穩定、訓練／推論 token 一致，以及既有切分與加權 loss 測試。最佳 checkpoint 重載的 loss／混淆矩陣核對通過，曲線已檢視。

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --learning-rate 2e-5 --seed 42 --preprocessing emoji-short-v1 --output-dir outputs/weighted_emoji --checkpoint outputs/weighted_emoji/checkpoint
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_emoji/checkpoint --predictions outputs/weighted_emoji/predictions.csv
```

以上目錄已有結果，重跑需指定新路徑。模型、設定、指標、曲線與 Kaggle CSV 位於 `outputs/weighted_emoji/`。執行此 checkpoint 時須保留支援 `emoji-short-v1` 的 `preprocessing.py`。

Kaggle CSV 的 1,000 筆、欄名、唯一 ID、原始順序及合法標籤核對通過。與原 2e-5 加權模型有 22 筆預測不同，與目前 Kaggle 最高分的 3e-5 單模型有 25 筆不同；核對結果見 `prediction_audit.json`。
