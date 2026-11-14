> 歷史實驗報告；內文的「下一步」代表當時判斷。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# 兩模型錯判分析與等權 soft voting

## 後續 Kaggle 對照

使用者於 2026/11/13 指示即使驗證未提升，也產生集成 Kaggle 對照檔。新增 `--generate-benchmark` 選項，以獨立目錄保存這次結果；驗證模型選擇結論不變。以下首次分析的「未產生」描述指原始 `outputs/ensemble_lr2_lr3/`，後續提交實驗使用 `outputs/ensemble_lr2_lr3_submission/`。

```powershell
.venv/Scripts/python.exe -m tools.analyze_ensemble --device cuda --generate-benchmark --output-dir outputs/ensemble_lr2_lr3_submission
```

新目錄中的 `predictions.csv` 使用兩個 checkpoint 的 softmax 機率等權平均；使用者於 2026/11/14 回報本版本 Kaggle **0.9052（90.52%）**，與 `3e-5` 單模型並列目前已回報最高分，未由工具登入平台核驗。集成與 `3e-5` 提交檔有 8 筆預測不同，但平台回報分數相同，不能據此推論逐筆正確性相同。集成未提高已回報 Kaggle 分數，且需兩個 checkpoint；同分時優先保留較簡單的 `3e-5` 單模型作為 Kaggle 候選，驗證最佳仍為 `2e-5`。

2026/11/13 完成。固定 900 筆驗證集上，`2e-5 + 3e-5` 等權 soft voting 的 Macro-F1 為 **92.58%**，低於最佳單模型 **92.65%**（下降 0.07 個百分點），本次不採用集成，也未產生集成 benchmark 預測。驗證最佳仍為 `weighted_sqrt`；已回報 Kaggle 最佳仍為 `weighted_lr3e5`（0.9052），兩者保留。

## 方法與核對

新增 `analyze_ensemble.py`，依既有固定切分逐筆輸出兩模型的類別機率與錯判；不重新訓練、不改標籤，也不搜尋集成權重。每次只載入一個 checkpoint，使用其 tokenizer、前處理與長度設定；驗證依原設定採 CUDA AMP、batch 2，float32 softmax 後以 0.5／0.5 平均。平票依既定類別順序取第一個最大值。

核對資料／切分 SHA-256、checkpoint 的驗證 ID 與類別順序，兩個單模型的混淆矩陣皆重現原訓練結果。另從輸出 CSV 獨立重算 F1，檢查 900 個唯一 ID 及順序、機率平均與 argmax，均通過。驗證結果達不到最佳單模型時不執行 benchmark 推論。

## 結果

| 指標 | 2e-5 | 3e-5 | 等權 soft voting |
|---|---:|---:|---:|
| Macro-F1 | **92.65%** | 92.10% | 92.58% |
| negative F1 | **97.04%** | 96.48% | 96.88% |
| neutral F1 | 86.47% | 85.71% | **86.79%** |
| positive F1 | **94.44%** | 94.12% | 94.07% |
| 錯判筆數 | 44 | 49 | 45 |

集成混淆矩陣（列真實、欄預測；negative／neutral／positive）：`[[621,12,3],[18,115,1],[7,4,119]]`。

兩模型皆正確 846 筆、僅 2e-5 正確 10 筆、僅 3e-5 正確 5 筆、共同錯判 39 筆。兩模型預測不同僅 15／900 筆（1.67%），共同錯判占 2e-5 錯判的 39／44（88.64%），互補有限。共同錯判按真實標籤分為 negative 13、neutral 17、positive 9。

相較 2e-5，集成修正 ID `1925`、`5563`，卻新增 ID `1193`、`3933`、`7202` 的錯誤。neutral F1 的小幅改善未抵銷其他類別退步。

## 錯判文字檢查

已逐筆檢視兩模型錯判聯集共 54 筆。以下是質性觀察，不是互斥分類統計，也不能據此認定原標籤錯誤。

- **詢問／求助與負面情緒邊界**：如 ID `2236`、`6763` 為帶有實際困難的請求，標籤 negative 卻共同預測 neutral；ID `3957`、`4917` 提及錯過航班或延誤，標籤 neutral 卻共同預測 negative。下一步應優先關注 neutral 與 negative 的混淆。
- **反諷與混合情緒**：ID `1495`、`7841` 的感謝措辭被共同預測 positive，但標籤 negative；ID `8500` 感謝解決延誤，標籤 positive 卻共同預測 negative。不能用刪除否定詞、抱怨詞或標點的方式修正。
- **缺少對話或圖片背景**：ID `5580` 僅列人名與登機門，ID `5847` 主要是 mention、地點與 URL；僅憑現有文字難以確定情緒。保留原資料及標籤，不引入 benchmark 標籤或外部資料。
- **Emoji 可能提供訊號**：ID `1123` 含愛心與笑臉，只有 3e-5 正確；ID `4562`、`5831`、`6532` 含拇指、眨眼或哭臉且兩模型皆錯。這只支持進一步檢查 tokenizer 與訓練集頻率，尚不足以證明 emoji 轉文字一定改善。

## 下一步

優先以 **2e-5 加權設定、訓練 seed 43** 做獨立對照，保持目前資料切分不變。此訓練已於 2026/11/14 完成，最佳驗證 Macro-F1 91.33%，詳見 [Seed43_Report.md](Seed43_Report.md)；seed 42／43 互補與集成尚未測試。另可先在訓練集盤點 emoji 的實際 token 表示，再決定是否值得一輪轉文字對照。暫不擴大集成權重搜尋，以免反覆迎合這 900 筆驗證資料。本輪分數差距很小，未做統計顯著性判定。

## 重現與產物

```powershell
.venv/Scripts/python.exe -m tools.analyze_ensemble --device cuda
```

預設結果目錄 `outputs/ensemble_lr2_lr3/` 已存在，重跑需指定新的 `--output-dir`。其中 `validation_predictions.csv` 保存全體 900 筆的文字、標籤、預測與機率，`validation_errors.csv` 保存錯判聯集 54 筆，`summary.json` 保存指標與來源設定，`audit.json` 保存核對結果。執行需保留兩份來源 checkpoint；目前交件單模型推論入口不變。
