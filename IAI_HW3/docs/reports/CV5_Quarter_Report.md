# 五折交叉驗證對照（weighted_quarter vs weighted_sqrt）

日期：2026/11/16。狀態：已完成（5 次完整訓練＋1 次 OOF／集成 FP32 推論＋1 次配對比較）；集成 CSV 未提交 Kaggle，Kaggle 分數為 `null`。對應 [92% 計畫](../Target_92_Plan.md) 新研究路線步驟 3a（預先登記見該檔「步驟 3a」，訓練前寫下）。

## 目的與假設

- 目的：在較低雜訊的 OOF 9,000 筆 Macro-F1 下，比較 quarter-inverse 與 sqrt-inverse 類別權重，決定步驟 3 後續單因素實驗的基準設定。
- 假設（事前）：quarter 在 OOF 上不劣於、或優於 sqrt。對照：[`cv5_sqrt`](CV5_Sqrt_Report.md)（OOF 89.03%）。
- 唯一主要變因：class weighting（sqrt-inverse → quarter-inverse）。5 折切分（`outputs/cv5_seed42/`）、seed 42、3 epochs、每 epoch 評估（eval-interval 1.0）等其餘設定全同。原單模型 weighted_quarter 的 eval-interval 為 0.5、訓練資料 8,100 筆，本 CV 刻意不跟，以免混入第二個因素。
- 預算：5 次完整訓練；決策規則已預先固定（見計畫）。

## 方法

- 訓練：`bash tools/run_cv5.sh cv5_quarter quarter-inverse`，5 折各訓練一次，各折類別權重依其訓練集重算。
- OOF：與 `cv5_sqrt` 相同（FP32、batch 2），產物於 `outputs/cv5_quarter_oof/`；集成取 fold 0／1／2 的 FP32 softmax 等權平均（預先固定，符合最多 3 個模型），僅作提交檔。
- 配對比較：`python -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_quarter_oof --output outputs/cv5_compare_sqrt_vs_quarter.json`。檢查兩邊的切分檔 SHA-256、資料集與各 fold 分數可由 OOF 預測重算；對 9,000 筆做配對 bootstrap（seed 20261116、2,000 次）。Δ 一律為 quarter − sqrt。

## 結果

數字取自 `outputs/cv5_compare_sqrt_vs_quarter.json`、兩邊的 `oof_metrics.json` 與各折 `history.json`／`summary.json`。

| 指標 | sqrt | quarter | Δ（quarter − sqrt，百分點） |
|---|---:|---:|---:|
| OOF Macro-F1（9,000 筆） | 89.03% | 88.80% | **−0.23** |
| negative F1 | 96.01% | 96.00% | −0.01 |
| neutral F1 | 80.78% | 80.15% | −0.63 |
| positive F1 | 90.29% | 90.23% | −0.06 |
| fold 平均 Macro-F1 | 89.00% | 88.78% | −0.22 |
| fold 樣本標準差（百分點） | 1.98 | 1.70 | — |
| 原 900 驗證 id 在 OOF 中的 Macro-F1 | 91.28% | 90.74% | −0.54 |

quarter 的 OOF 混淆矩陣（列為真實，欄為預測，順序 negative／neutral／positive）：`[[6134,166,64],[221,1046,68],[60,63,1178]]`。

逐 fold 驗證 Macro-F1（FP32 OOF 重算；quarter 勝 2/5 折）：

| fold | sqrt | quarter | Δ（百分點） |
|---:|---:|---:|---:|
| 0 | 89.50 | 88.79 | −0.71 |
| 1 | 87.70 | 88.15 | +0.45 |
| 2 | 92.26 | 91.64 | −0.62 |
| 3 | 88.01 | 88.18 | +0.17 |
| 4 | 87.55 | 87.16 | −0.39 |

配對 bootstrap（9,000 筆、seed 20261116、2,000 次）：Δ 平均 −0.24 個百分點，95% 區間 **[−0.71, +0.21]**，Δ≤0 的比例 84.4%。兩邊預測不同的有 179 筆：sqrt 答對 91、quarter 答對 81。

quarter 各折訓練情形（每 epoch 驗證 Macro-F1 為訓練時 AMP 評估）：

| fold | epoch 1 | epoch 2 | epoch 3 | 最佳 epoch | 訓練秒數 |
|---:|---:|---:|---:|---:|---:|
| 0 | 87.05 | 87.86 | 88.79 | 3 | 1,977 |
| 1 | 86.98 | 88.15 | 87.33 | 2 | 1,720 |
| 2 | 89.96 | 90.82 | 91.50 | 3 | 952 |
| 3 | 86.11 | 87.46 | 88.18 | 3 | 956 |
| 4 | 84.40 | 86.84 | 87.16 | 3 | 1,044 |

- 5 折全部通過 reload 驗證；跳過 AMP 更新 2–3 次（約 1,350 次中的 0.2%）。
- 前兩折明顯較慢（約 29–33 分鐘，後三折約 16–17 分鐘），原因不明，推測為 GPU 降頻（推測，未驗證）。
- 集成 CSV：`outputs/cv5_quarter_oof/submissions/cv5_folds012_mean.csv`，1,000 筆，SHA-256 `a122c68fcaa8f56e87a22a42ecd6832f720c5281366ad173b7dd1f82ee79f496`；未提交 Kaggle。

## 結論

標示「已驗證」為本次數據直接支持；「推論」「推測」為尚未檢驗的解釋。

1. 已驗證（依預先固定的決策規則）：95% 區間 [−0.71, +0.21] 包含 0，視為無顯著差異；後續基準取 OOF 較高者，即 **sqrt（OOF 89.03%），標註「差異不顯著」**。本步只決定基準，不代表 sqrt 優於 quarter。
2. 推論：900 驗證（sqrt 92.65% vs quarter 91.91%）、Kaggle（sqrt 0.9049 vs quarter 0.9122，使用者回報）與 OOF（sqrt 高 0.23 個百分點）三者方向不一，差距都很小，一致支持「兩者實質相當」。Kaggle 的 0.73 個百分點約對應 7 筆；且原 Kaggle quarter 模型 eval-interval 為 0.5、訓練資料為 8,100 筆，與本 CV 設定不同，不能直接對照。
3. 已驗證：差異主要在 neutral（−0.63 個百分點），其餘兩類幾乎相同；逐 fold 方向不一（2 勝 3 敗），沒有一致增益。
4. 推測：兩種設定共 10 個 fold 中有 8 個的最佳 epoch 是最後一個（第 3 個），且多數曲線仍在上升，支持「訓練不足」的推測。這是後續單因素實驗（例如 epoch 數）的候選，本步不改。

## 限制

- bootstrap 只反映評估樣本的抽樣雜訊，不含訓練隨機性；每折只用 seed 42 訓練一次，區間不涵蓋訓練 seed 變異。
- 每折以其驗證折挑最佳 epoch，OOF 有輕微樂觀偏誤（候選僅 3 個 epoch）；兩邊作法相同，對 Δ 的影響應相近（推論）。
- fold 0／1／2 集成為預先固定，無法以 OOF 評估，也未有 Kaggle 結果。
- 同折內的 AMP 評估與 FP32 OOF 有微小差異。

## 產物

- 切分：`outputs/cv5_seed42/`（與 `cv5_sqrt` 共用）
- 各折訓練：`outputs/cv5_quarter/fold{0..4}/`
- checkpoint：`saved_models/cv5_quarter/fold{0..4}/`
- 日誌：`outputs/cv5_quarter_train.log`
- OOF：`outputs/cv5_quarter_oof/`（`oof_metrics.json`、`oof_predictions.csv`、各折 logits npz、`submissions/cv5_folds012_mean.csv`）
- 配對比較：`outputs/cv5_compare_sqrt_vs_quarter.json`
- 登錄：[registry](../../experiments/registry.json) 的 `cross_validations`（`cv5_quarter`，含 `comparison` 欄位），摘要見 [實驗總表](../Experiment_Results.md)

## 重現

```powershell
bash tools/run_cv5.sh cv5_quarter quarter-inverse   # 5 折訓練後自動呼叫 tools.cv_oof 產生 OOF 與集成 CSV；輸出目錄須為新的
.venv/Scripts/python.exe -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_quarter_oof --output outputs/cv5_compare_sqrt_vs_quarter.json
.venv/Scripts/python.exe tools/refresh_experiments.py --check
```

## 耗時與環境

- 訓練 2026/11/16 14:34–16:27（日誌時間），各折訓練秒數合計約 6,649 秒（約 111 分鐘）。
- Python 3.12.10、torch 2.9.1+cu126、CUDA 12.6、NVIDIA GeForce RTX 3050 Laptop GPU（4 GB）。
