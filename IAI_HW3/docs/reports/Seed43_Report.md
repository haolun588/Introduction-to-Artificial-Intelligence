> 歷史實驗報告；內文的「下一步」代表當時判斷。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# 加權 BERT 訓練 seed 43 對照

2026/11/14 完成。最佳為第 3 輪，驗證 Macro-F1 **91.33%**，比相同設定 seed 42 的 **92.65%** 低 **1.31 個百分點**（以未四捨五入數值計算）。驗證選模仍保留 seed 42；依使用者希望取得 Kaggle 對照檔的偏好，本輪亦產生獨立提交 CSV。使用者於 2026/11/14 回報 Kaggle **0.8878（88.78%）**，比相同設定 seed 42 的 0.9049 低 **1.71 個百分點**，比目前最高分 0.9052 低 **1.74 個百分點**。此為使用者回報，未由工具登入平台核驗；本輪不取代既有最佳候選。

## 設定

已逐欄核對 `weighted_sqrt/run_config.json` 與本輪設定，唯一差異為訓練 seed 42 → 43。維持原固定切分 8,100／900，資料與 split SHA-256、實際樣本 ID 均相同；未重新切分資料。

其餘為同一 BERT 基底、空白正規化與原 URL、反平方根加權 CE、學習率 2e-5、3 epochs、max length 64、micro-batch 2、累積 8、AdamW weight decay 0.01、10% linear warmup／decay、AMP 與梯度檢查點。變更訓練 seed 會影響分類 head 初始化、訓練順序與 dropout 等隨機性。

## 每輪結果

| Epoch | Train loss | Val loss | Train Macro-F1 | Val Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 0.177406 | 0.247776 | 90.89% | 86.58% |
| 2 | 0.076723 | 0.222031 | 96.75% | 91.26% |
| 3（入選） | 0.056038 | 0.254320 | 97.94% | **91.33%** |

Loss 為未加權 CE。訓練與重載驗證共 1,011.50 秒（16.86 分鐘），不含初始載入與 benchmark 推論。成功更新 1,518 次，AMP 略過 3 次；峰值配置顯存 1,904.24 MiB。第 3 輪相較第 2 輪驗證 loss 上升，而 F1 僅小幅提升；依既定 Macro-F1 規則選第 3 輪。

| 指標 | seed 42 | seed 43 |
|---|---:|---:|
| Macro-F1 | **92.65%** | 91.33% |
| negative F1 | 97.04% | 96.31% |
| neutral F1 | 86.47% | 85.50% |
| positive F1 | 94.44% | 92.19% |
| Kaggle（使用者回報） | 0.9049 | 0.8878 |

最佳混淆矩陣（列真實、欄預測，negative／neutral／positive）：`[[614,16,6],[17,115,2],[8,4,118]]`。錯判 53 筆，seed 42 為 44 筆。兩個 seed 的結果顯示此次設定對訓練隨機性有可觀差異；僅兩次訓練不足以估計穩定的平均表現或斷言特定 seed 普遍較好。

## 驗證及重現

訓練程式未修改。最佳 checkpoint 重載後的 loss 與混淆矩陣核對通過，曲線已檢視。設定、history、summary、曲線、checkpoint 及預測位於 `outputs/weighted_seed43/`。

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --learning-rate 2e-5 --seed 43 --output-dir outputs/weighted_seed43 --checkpoint outputs/weighted_seed43/checkpoint
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_seed43/checkpoint --predictions outputs/weighted_seed43/predictions.csv
```

以上路徑已使用，重跑須指定新路徑。後續 seed 42／43 互補與等權集成已完成，驗證 Macro-F1 92.46%，仍低於 seed 42 單模型；另產生 Kaggle 對照 CSV，詳見 [seed 集成報告](Seed_Ensemble_Report.md)。

提交 CSV 已核對 1,000 筆、`id,pred` 欄名、唯一 ID、benchmark 原始順序與合法標籤；詳見 `prediction_audit.json`。與 seed 42 的 2e-5 有 27 筆預測不同，與 3e-5 有 30 筆不同。
