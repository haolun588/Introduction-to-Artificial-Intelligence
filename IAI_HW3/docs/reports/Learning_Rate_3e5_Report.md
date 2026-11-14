> 歷史實驗報告；內文的「下一步」代表當時判斷。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# 學習率 3e-5 對照實驗

2026/11/13 完成。最佳第 3 輪驗證 Macro-F1 **92.10%**，低於原加權模型（2e-5）的 **92.65%**，下降 **0.54 個百分點**（以未四捨五入數值計算）。因此仍保留 `outputs/weighted_sqrt/checkpoint/`。

## 實驗設定

已核對兩份 run config，唯一差異為 learning rate：`2e-5` → `3e-5`。從相同 BERT 基底重新訓練，沿用固定 8,100／900 切分、seed 42、空白正規化與原 URL、反平方根類別權重、max length 64、micro-batch 2、梯度累積 8、AdamW、weight decay 0.01、3 epochs、10% linear warmup／decay、AMP 與梯度檢查點。

## 每輪結果

| Epoch | Train loss | Val loss | Train Macro-F1 | Val Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 0.130017 | 0.209359 | 93.54% | 90.99% |
| 2 | 0.054960 | 0.225369 | 97.77% | 91.00% |
| 3（入選） | 0.034246 | 0.275157 | 98.81% | **92.10%** |

Loss 為未加權 CE。訓練與重載驗證共 1,064.72 秒（17.75 分鐘），不含初始載入及 benchmark 推論。成功更新 1,518 次，AMP 略過 3 次；峰值配置顯存 1,904.24 MiB。

## 學習率比較

| 指標 | 1e-5 | 2e-5（保留） | 3e-5 |
|---|---:|---:|---:|
| 最佳驗證 Macro-F1 | 91.46% | **92.65%** | 92.10% |
| negative F1 | 96.40% | **97.04%** | 96.48% |
| neutral F1 | 85.07% | **86.47%** | 85.71% |
| positive F1 | 92.91% | **94.44%** | 94.12% |
| Kaggle（使用者回報） | 0.9012 | 0.9049 | **0.9052** |

本輪最佳混淆矩陣（列為真實、欄為預測，negative／neutral／positive）：`[[617,15,4],[19,114,1],[7,3,120]]`。

本輪三類 F1 皆低於 2e-5。訓練 loss 下降但驗證 loss 上升，顯示過擬合跡象；模型仍依預定的驗證 Macro-F1 選取。結論限於此 seed、切分及 3 epochs，沒有多 seed 重跑證據。

## 驗證與產物

- 訓練程式未修改，沿用前輪已通過 11 項測試的版本。
- 最佳 checkpoint 重載後的 loss 與混淆矩陣核對通過，曲線已檢視。
- 設定、history、summary、曲線與 checkpoint 位於 `outputs/weighted_lr3e5/`。
- `predictions.csv` 共 1,000 筆；`id,pred` 欄名、唯一 ID、原始順序與合法標籤核對通過，詳見 `prediction_audit.json`。
- 使用者於 2026/11/13 回報本輪 Kaggle **0.9052（90.52%）**，比 2e-5 的 0.9049 高 **0.03 個百分點**，為目前已回報的最高 Kaggle 分數；未由工具登入平台核驗。驗證集最佳仍為 2e-5，兩個 checkpoint 均保留。依工作計畫所記評分表，90.52% 已達 90 分門檻（90.5%），最終以課程及平台判定為準。

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --learning-rate 3e-5 --output-dir outputs/weighted_lr3e5 --checkpoint outputs/weighted_lr3e5/checkpoint
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_lr3e5/checkpoint --predictions outputs/weighted_lr3e5/predictions.csv
```

上述路徑已使用，重跑須指定新路徑。三個候選學習率比較已完成；後續錯判分析與首次集成比較亦已完成，詳見 [錯判與集成報告](Ensemble_Error_Analysis_Report.md)。
