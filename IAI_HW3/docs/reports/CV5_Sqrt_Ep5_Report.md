# 五折交叉驗證對照（sqrt 3 epochs vs 5 epochs）

日期：2026/11/16。狀態：已完成（5 次完整訓練＋1 次 OOF／集成 FP32 推論＋1 次配對比較）；集成 CSV 未提交 Kaggle，Kaggle 分數為 `null`。對應 [92% 計畫](../Target_92_Plan.md) 新研究路線步驟 3b（預先登記見該檔「步驟 3b」，訓練前寫下）。

## 目的與假設

- 目的：在較低雜訊的 OOF 9,000 筆 Macro-F1 下，檢驗訓練長度（3 → 5 epochs）是否有可偵測的效果，決定後續單因素實驗的基準。
- 假設（事前）：現行 3 epochs 訓練不足；5 epochs 的 OOF Macro-F1 高於 3 epochs。依據為步驟 2／3a 共 10 折中 8 折最佳 epoch 為最後一個。對照：[`cv5_sqrt`](CV5_Sqrt_Report.md)（3 epochs，OOF 89.03%）。
- 唯一變因：`--epochs 3 → 5`。連帶變化：學習率排程隨總步數變長（warmup 步數與線性衰減跨度變長），屬「訓練長度」因素的一部分，不另拆；因此同一個 epoch 編號在兩組的學習率不同，每 epoch 分數不可直接對照。其餘設定（5 折切分 `outputs/cv5_seed42/`、seed 42、sqrt-inverse、eval-interval 1.0）全同。
- 預算：5 次完整訓練；決策規則已預先固定（見計畫）。

## 方法

- 訓練：`bash tools/run_cv5.sh cv5_sqrt_ep5 sqrt-inverse 5`，5 折各訓練一次。
- OOF：與 `cv5_sqrt` 相同（FP32、batch 2），產物於 `outputs/cv5_sqrt_ep5_oof/`；集成取 fold 0／1／2 的 FP32 softmax 等權平均（預先固定，符合最多 3 個模型），僅作提交檔。
- 配對比較：`python -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_sqrt_ep5_oof --output outputs/cv5_compare_sqrt_vs_sqrt_ep5.json`。檢查切分檔 SHA-256、資料集與各 fold 分數可由 OOF 預測重算；配對 bootstrap（seed 20261116、2,000 次）。Δ 一律為 ep5 − ep3。

## 結果

數字取自 `outputs/cv5_compare_sqrt_vs_sqrt_ep5.json`、兩邊的 `oof_metrics.json` 與各折 `history.json`／`summary.json`。

| 指標 | ep3（cv5_sqrt） | ep5（cv5_sqrt_ep5） | Δ（ep5 − ep3，百分點） |
|---|---:|---:|---:|
| OOF Macro-F1（9,000 筆） | 89.03% | 89.22% | **+0.19** |
| negative F1 | 96.01% | 96.02% | +0.01 |
| neutral F1 | 80.78% | 80.75% | −0.04 |
| positive F1 | 90.29% | 90.90% | +0.61 |
| fold 平均 Macro-F1 | 89.00% | 89.22% | +0.22 |
| fold 樣本標準差（百分點） | 1.98 | 1.51 | — |
| 原 900 驗證 id 在 OOF 中的 Macro-F1 | 91.28% | 90.42% | −0.86 |

ep5 的 OOF 混淆矩陣（列為真實，欄為預測，順序 negative／neutral／positive）：`[[6137,170,57],[219,1059,57],[63,59,1179]]`。

逐 fold 驗證 Macro-F1（FP32 OOF 重算；ep5 勝 3/5 折）：

| fold | ep3 | ep5 | Δ（百分點） |
|---:|---:|---:|---:|
| 0 | 89.50 | 88.11 | −1.38 |
| 1 | 87.70 | 88.56 | +0.86 |
| 2 | 92.26 | 91.85 | −0.41 |
| 3 | 88.01 | 89.11 | +1.10 |
| 4 | 87.55 | 88.46 | +0.91 |

配對 bootstrap（9,000 筆、seed 20261116、2,000 次）：Δ 平均 +0.19 個百分點，95% 區間 **[−0.38, +0.76]**，Δ≤0 的比例 26.9%。兩邊預測不同的有 292 筆：ep3 答對 134、ep5 答對 141。

ep5 各折每 epoch 驗證 Macro-F1（訓練時 AMP 評估，%）：

| fold | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | 最佳 epoch | 訓練秒數 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 85.54 | 88.11 | 87.61 | 87.47 | 87.22 | 2 | 1,643 |
| 1 | 86.26 | 88.30 | 88.56 | 88.44 | 88.16 | 3 | 1,658 |
| 2 | 89.24 | 90.22 | 91.18 | 91.03 | 91.85 | 5 | 1,821 |
| 3 | 85.49 | 86.29 | 89.06 | 88.53 | 89.11 | 5 | 1,638 |
| 4 | 85.67 | 86.94 | 88.18 | 88.14 | 88.46 | 5 | 1,702 |

- 5 折全部通過 reload 驗證；跳過 AMP 更新 3–4 次（每折約 2,250 次嘗試中）。
- 集成 CSV：`outputs/cv5_sqrt_ep5_oof/submissions/cv5_folds012_mean.csv`，1,000 筆，SHA-256 `351963b9048f12b5c3fdc8c37d52e4db832361d83d257ebd32449afcb3d1d988`；未提交 Kaggle。

## 結論

標示「已驗證」為本次數據直接支持；「推論」「推測」為尚未檢驗的解釋。

1. 已驗證（依預先固定的決策規則）：95% 區間 [−0.38, +0.76] 包含 0，視為無顯著差異；後續基準取 OOF 較高者，即 **cv5_sqrt_ep5（OOF 89.22%），標註「差異不顯著」**。
2. 推論：+0.19 個百分點的差距小於、且與選模偏誤同方向（5 個候選 epoch 比 3 個候選更樂觀），因此不能解讀為 5 epochs 實質較好；最保守的說法是「訓練長度 3 → 5 沒有可偵測的效果」。
3. 附加觀察（依預先登記，不作決策依據）：最佳 epoch 分布為 2、3、5、5、5，多數折為 5，依登記記為「可能仍訓練不足」。但 3 個最佳在 5 的折中，第 3 → 5 epoch 的增幅只有 +0.67、+0.05、+0.28 個百分點，曲線已大致平坦（推測：訓練長度不是主要瓶頸）。兩組學習率排程不同，同 epoch 分數不可直接對照。
4. 推論：步驟 3a（−0.23）與 3b（+0.19）的差距都落在約 ±0.5 個百分點的區間內，現行單 seed 的 CV 比較只能偵測約 0.5 個百分點以上的效果；更小的改進需要多 seed 或更強的方法。
5. 差異主要在 positive（+0.61 個百分點），其餘兩類幾乎不變；逐 fold 方向不一（3 勝 2 敗），fold 0 反而下降 1.38 個百分點。

## 限制

- bootstrap 只反映評估樣本的抽樣雜訊，不含訓練隨機性；每折只用 seed 42 訓練一次，區間不涵蓋訓練 seed 變異。
- 每折以其驗證折挑最佳 epoch，OOF 有選模樂觀偏誤；ep5 有 5 個候選、ep3 只有 3 個，偏誤大小不同且偏向 ep5，+0.19 的 Δ 可能部分來自此（推論）。
- 學習率排程連帶變化，無法把「更多 epoch」與「更長排程」拆開。
- fold 0／1／2 集成為預先固定，無法以 OOF 評估，也未有 Kaggle 結果。
- 同折內的 AMP 評估與 FP32 OOF 有微小差異。

## 產物

- 切分：`outputs/cv5_seed42/`（與 `cv5_sqrt` 共用）
- 各折訓練：`outputs/cv5_sqrt_ep5/fold{0..4}/`
- checkpoint：`saved_models/cv5_sqrt_ep5/fold{0..4}/`
- 日誌：`outputs/cv5_sqrt_ep5_train.log`
- OOF：`outputs/cv5_sqrt_ep5_oof/`（`oof_metrics.json`、`oof_predictions.csv`、各折 logits npz、`submissions/cv5_folds012_mean.csv`）
- 配對比較：`outputs/cv5_compare_sqrt_vs_sqrt_ep5.json`
- 登錄：[registry](../../experiments/registry.json) 的 `cross_validations`（`cv5_sqrt_ep5`，含 `comparison` 欄位）

## 重現

```powershell
bash tools/run_cv5.sh cv5_sqrt_ep5 sqrt-inverse 5   # 5 折訓練後自動呼叫 tools.cv_oof 產生 OOF 與集成 CSV；輸出目錄須為新的
.venv/Scripts/python.exe -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_sqrt_ep5_oof --output outputs/cv5_compare_sqrt_vs_sqrt_ep5.json
.venv/Scripts/python.exe tools/refresh_experiments.py --check
```

## 耗時與環境

- 訓練 2026/11/16 19:48–22:12（日誌時間，含 OOF 推論），各折訓練秒數合計約 8,463 秒（約 141 分鐘），每折約 27.5–30.5 分鐘。
- Python 3.12.10、torch 2.9.1+cu126、CUDA 12.6、NVIDIA GeForce RTX 3050 Laptop GPU（4 GB）。
