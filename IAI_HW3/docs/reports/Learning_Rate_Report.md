> 歷史實驗報告；內文的「下一步」代表當時判斷。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# 學習率 1e-5 對照實驗

2026/11/13 完成。最佳為第 3 輪，驗證 Macro-F1 **91.46%**，低於原加權模型的 **92.65%**，下降 **1.19 個百分點**。因此仍保留 `outputs/weighted_sqrt/checkpoint/`，不以本輪模型取代。

## 設定與結果

已核對兩份 run config，唯一差異為 learning rate：`2e-5` → `1e-5`。沿用固定 8,100／900 切分、seed 42、原 URL、反平方根類別權重、max length 64、micro-batch 2、累積 8、3 epochs、10% linear warmup／decay、AMP 與梯度檢查點，從相同 BERT 基底重新訓練。

| Epoch | Train loss | Val loss | Train Macro-F1 | Val Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 0.183537 | 0.214834 | 90.92% | 90.30% |
| 2 | 0.109911 | 0.220807 | 94.82% | 90.99% |
| 3（入選） | 0.092794 | 0.236094 | 96.06% | **91.46%** |

Loss 為未加權 CE。訓練與重載驗證共 1,397.88 秒（23.30 分鐘），不含初始載入與 benchmark 推論。成功更新 1,518 次，AMP 略過 3 次；峰值配置顯存 1,904.24 MiB。第一輪耗時 675.09 秒，後兩輪為 351.04／362.14 秒；耗時受當時機器狀態影響，不宜直接歸因於學習率。

| 指標 | 原加權 2e-5 | 本輪 1e-5 |
|---|---:|---:|
| Macro-F1 | **92.65%** | 91.46% |
| negative F1 | 97.04% | 96.40% |
| neutral F1 | 86.47% | 85.07% |
| positive F1 | 94.44% | 92.91% |

最佳混淆矩陣（列為真實、欄為預測，negative／neutral／positive）：`[[616,16,4],[18,114,2],[8,4,118]]`。

三類 F1 均低於原加權版本。雖然本輪驗證 F1 逐輪上升，驗證 loss 亦逐輪上升，不能據此推論延長訓練必定改善。本結論只涵蓋目前 seed、切分及 3 epochs 設定。

## 驗證與產物

- 既有 11 項測試通過。
- 最佳 checkpoint 重載後的驗證 loss 與混淆矩陣核對通過。
- 曲線已檢視；`predictions.csv` 的 1,000 筆、`id,pred` 欄名、唯一 ID、原始順序與合法標籤檢查通過，核對結果保存於 `prediction_audit.json`。
- 設定、history、summary、曲線、checkpoint 及推論結果存於 `outputs/weighted_lr1e5/`。
- 使用者於 2026/11/13 回報本輪 Kaggle **0.9012（90.12%）**，低於原加權模型 **0.9049（90.49%）**，下降 **0.37 個百分點**。此分數由使用者回報，未由工具登入平台核驗。驗證與 Kaggle 比較方向一致，繼續保留原加權模型。

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --learning-rate 1e-5 --output-dir outputs/weighted_lr1e5 --checkpoint outputs/weighted_lr1e5/checkpoint
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_lr1e5/checkpoint --predictions outputs/weighted_lr1e5/predictions.csv
```

上述路徑已使用，重跑需指定新路徑。相同加權設定下的 `3e-5` 對照已完成，最佳驗證 Macro-F1 92.10%，仍保留 2e-5。詳見 [3e-5 對照報告](Learning_Rate_3e5_Report.md)。
