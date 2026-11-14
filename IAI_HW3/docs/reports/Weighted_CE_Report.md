> 歷史實驗報告；內文的「下一步」代表當時判斷。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# 反平方根加權 cross-entropy 對照實驗

2026/11/13 完成。最佳為第 3 epoch，固定驗證集 Macro-F1 **92.65%**，相較未加權基準 **91.57%** 提升 **1.08 個百分點**。三類 F1 均上升；使用者回報此模型 Kaggle 分數為 **0.9049（90.49%）**，比基準 0.8973 提升 **0.76 個百分點**。

兩輪 Kaggle 分數皆由使用者回報，未由工具登入平台核驗。加權模型距工作計畫列出的 0.905 門檻仍差 0.0001（0.01 個百分點），不可四捨五入視為達標。

## 實驗控制與 loss

- 重新從指定 BERT 基底初始化，沿用 seed 42、8,100／900 固定切分、空白正規化、max length 64、micro-batch 2、累積 8、AdamW 2e-5、weight decay 0.01、10% warmup／linear decay、3 epochs、AMP 與梯度檢查點。
- 已逐一核對基準 run config 的全部共同欄位，包括模型 revision、訓練／驗證 ID、資料與切分 SHA-256，均相同。唯一指定實驗變因為 `--class-weighting sqrt-inverse`。
- 權重由本次訓練集頻數 5,728／1,201／1,171 計算，先取 `1/sqrt(n_class)`，再使訓練樣本的平均權重為 1。
- negative／neutral／positive 權重為 **0.740353／1.616846／1.637426**。
- 訓練 loss 為加權 CE 總和除以累積窗口實際樣本數，尾端不足窗口亦採實際樣本數。此為固定尺度的樣本平均，不是每個 micro-batch 各自除以權重總和。
- 驗證維持原始分布；每輪 train／validation loss 都以未加權 CE 評估，以便和基準比較。模型選擇指標仍為 validation Macro-F1。
- API 定義參考 [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)。

## 每輪結果

| Epoch | Train loss | Val loss | Train Macro-F1 | Val Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 0.145064 | 0.223606 | 92.91% | 90.67% |
| 2 | 0.068268 | 0.230837 | 96.92% | 89.74% |
| 3（入選） | 0.049274 | 0.261161 | 98.11% | **92.65%** |

訓練迴圈至 checkpoint 重載核對共 1,217.59 秒（20.29 分鐘），不含初始載入及 benchmark 推論。成功 optimizer 更新 1,518 次，AMP 略過 3 次；scheduler 只隨成功更新前進。PyTorch 峰值配置顯存 1,904.24 MiB，約 1.86 GiB。使用 RTX 3050 Laptop 4 GB、Python 3.12.10、PyTorch 2.9.1+cu126。

## 與基準最佳模型比較

| 指標 | 未加權基準 | 加權 CE | 變化（百分點） |
|---|---:|---:|---:|
| Macro-F1 | 91.57% | **92.65%** | +1.08 |
| negative F1 | 96.65% | 97.04% | +0.39 |
| neutral F1 | 84.73% | 86.47% | +1.73 |
| positive F1 | 93.33% | 94.44% | +1.11 |
| neutral precision | 86.72% | 87.12% | +0.40 |
| neutral recall | 82.84% | 85.82% | +2.99 |

變化以未四捨五入值計算。Neutral 的 precision 與 recall 均改善，不是僅靠犧牲 precision 提高 recall；但目前只有一個 seed、一次固定驗證切分，還不能視為穩定的泛化改善。

混淆矩陣：列為真實標籤，欄為預測標籤。

| 真實／預測 | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 622 | 12 | 2 |
| neutral | 18 | 115 | 1 |
| positive | 6 | 5 | 119 |

驗證 loss 仍隨 epoch 上升，train／validation 亦有差距；因此不以增加 epochs 作為下一步的預設改善方式。

## 驗證與產物

- 8 項單元測試通過，含類別權重比值、訓練平均權重為 1、缺失類別拒絕、加權累積梯度與整個窗口等價、尾端不足 batch，以及既有切分測試。
- 最佳 checkpoint 重載後的驗證 loss 與混淆矩陣一致。
- `outputs/weighted_sqrt/checkpoint/` 保存第 3 輪模型、tokenizer 與完整設定。
- `outputs/weighted_sqrt/predictions.csv` 為可提交的 1,000 筆 `id,pred` 預測；唯一 ID、原始順序、標籤合法性均核對通過。預測分布為 negative 734、neutral 124、positive 142。
- `outputs/weighted_sqrt/` 另存 `history.json`、`summary.json`、`comparison.json`、`prediction_check.json`、曲線、run config、環境與切分副本。曲線已檢視。
- 透過 SHA-256 確認 `saved_models/checkpoint/` 的基準權重與 `outputs/baseline/predictions.csv` 均未改動。預設推論仍載入基準，使用加權模型時須明確傳入 `--checkpoint outputs/weighted_sqrt/checkpoint`。
- 使用者已提交新模型至 Kaggle 並回報 0.9049；E3 提交狀態未確認。輸出目錄被 Git 忽略，正式交件時需納入選定模型。

下一項依工作計畫比較 URL 正規化，保持本輪加權設定與其他參數不變，另存對照實驗。
