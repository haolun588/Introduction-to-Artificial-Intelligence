> 歷史實驗報告；內文的「下一步」代表當時判斷。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# seed 42／43 等權集成對照

2026/11/14 完成。相同 2e-5 加權設定的兩個 seed 做等權 soft voting，固定驗證 Macro-F1 **92.46%**，低於 seed 42 單模型 **92.65%**，下降 **0.19 個百分點**。驗證選模仍保留 seed 42；依使用者要求，另產生可提交 Kaggle 的 `outputs/ensemble_seed42_seed43/predictions.csv`。使用者於 2026/11/14 回報 Kaggle **0.8983（89.83%）**，比 seed 42 單模型 0.9049 低 **0.66 個百分點**，比目前最高分 0.9052 低 **0.69 個百分點**；不取代既有最佳候選。此分數由使用者回報，未由工具登入平台核驗。

## 方法與結果

沿用固定 900 筆驗證集，依 checkpoint 保存的 tokenizer、前處理及長度設定推論。每次載入一個模型，CUDA AMP、batch 2，float32 softmax 後以 0.5／0.5 平均。類別順序為 negative／neutral／positive，argmax 平票取第一個最大值。沒有搜尋權重或重新訓練。

| 指標 | seed 42 | seed 43 | 等權集成 |
|---|---:|---:|---:|
| Macro-F1 | **92.65%** | 91.33% | 92.46% |
| negative F1 | **97.04%** | 96.31% | 96.79% |
| neutral F1 | 86.47% | 85.50% | **86.89%** |
| positive F1 | **94.44%** | 92.19% | 93.70% |
| 錯判筆數 | 44 | 53 | 46 |
| Kaggle（使用者回報） | 0.9049 | 0.8878 | 0.8983 |

集成混淆矩陣（列真實、欄預測）：`[[619,13,4],[17,116,1],[7,4,119]]`。

## 互補分析

- 兩模型皆正確 842 筆，僅 seed 42 正確 14 筆，僅 seed 43 正確 5 筆，共同錯判 39 筆。
- 共有 20 筆預測不同（2.22%），其中包含 1 筆兩模型皆錯但預測類別不同。錯判聯集共 58 筆。
- 集成修正 seed 42 的 3 筆錯誤（ID `1925`、`2524`、`5563`），新增 5 筆錯誤（`2086`、`3689`、`3933`、`5081`、`6462`）。
- neutral F1 小幅提升，但其他兩類下降。共同錯判仍占 seed 42 錯判的 39／44（88.64%），本輪互補未轉為整體驗證提升。差距很小，未做統計顯著性判定。

## 核對與產物

資料／切分雜湊、驗證 ID、類別順序均核對通過；兩個單模型重現各自保存的混淆矩陣。從輸出 CSV 獨立重算三組 F1，核對 900 筆 ID 與機率平均、argmax，均通過。

Kaggle CSV 共 1,000 筆，`id,pred` 欄名、唯一 ID、benchmark 原始順序及合法標籤核對通過。與 seed 42 單模型有 12 筆不同、seed 43 有 15 筆不同、目前最高分的 3e-5 單模型有 **17 筆不同**，與先前學習率集成亦有 17 筆不同。尚未由工具提交 Kaggle。

`outputs/ensemble_seed42_seed43/` 保存提交檔 `predictions.csv`、全部驗證預測與機率 `validation_predictions.csv`、58 筆錯判聯集 `validation_errors.csv`、`summary.json` 與 `audit.json`。原 checkpoint 與既有提交檔不變。

```powershell
.venv/Scripts/python.exe -m tools.analyze_ensemble --pair seed --generate-benchmark --device cuda
```

新增 `--pair seed` 支援此次比較；預設 `--pair learning-rate` 保留原學習率配對。輸出目錄已使用，重跑需指定新的 `--output-dir`。

後續訓練集 emoji 盤點已完成，詳見 [Emoji_Audit_Report.md](Emoji_Audit_Report.md)；結果支持安排轉文字對照，規則實作與訓練尚未啟動。暫不擴大 seed 或集成權重搜尋。
