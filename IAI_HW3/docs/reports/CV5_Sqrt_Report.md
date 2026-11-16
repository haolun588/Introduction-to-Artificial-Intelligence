# 五折交叉驗證基準（weighted_sqrt 設定）

日期：2026/11/16。狀態：已完成（5 次完整訓練＋1 次 OOF／集成 FP32 推論）；集成 CSV 未提交 Kaggle，Kaggle 分數為 `null`。對應 [92% 計畫](../Target_92_Plan.md) 新研究路線步驟 2。

## 目的與假設

- 目的：以較低雜訊的 OOF 9,000 筆 Macro-F1 建立現行設定（weighted_sqrt）的可靠水準與 fold 間波動，作為步驟 3 的比較基準。
- 假設（事前）：900 筆驗證集雜訊大，單一切分的分數不能可靠排序設定。對照：同一設定在 5 個不同驗證折上的分數分布。
- 唯一主要變因：切分（原固定 8,100／900 → 五折 7,200／1,800 左右）；訓練設定與 weighted_sqrt 相同。
- 預算：5 次完整訓練（每折 1 次）。合計與先前 5 次共 10 次，計畫上限已依 [調整紀錄](../Target_92_Plan.md) 改寫。

## 方法

- 切分：`main.py kfold --k 5 --seed 42 --output-dir outputs/cv5_seed42`；演算法 `per-label seeded greedy grouped kfold v1`（`src/eda.py:grouped_kfold`），群組鍵為空白正規化文字，避免重複文字跨 train／validation；各類別逐標籤貪婪分配。產物 `fold0..4.json` 與 `folds_manifest.json`（記錄各檔 SHA-256）。
- 訓練：`tools/run_cv5_sqrt.sh`，5 折各訓練一次，設定同 weighted_sqrt：sqrt-inverse 類別權重（依各折訓練集重算）、seed 42、3 epochs、每 epoch 評估、batch 2×累積 8、lr 2e-5、warmup 0.1、max_length 64、whitespace-v1 前處理、AMP、梯度檢查點。每折只在其訓練折訓練，並以其驗證折挑選最佳 epoch。
- OOF：`python -m tools.cv_oof`（FP32、batch 2）對每個 fold checkpoint 推論其驗證折，拼成 9,000 筆 OOF，計算整體與各類指標；另對 benchmark 推論。
- 集成規則：預先固定取 fold 0／1／2 的 FP32 softmax 等權平均（符合最多 3 個模型），不依分數選折。此集成無法以 OOF 評估（各模型的驗證折不同），僅作提交檔。

## 結果

數字取自 `outputs/cv5_sqrt_oof/oof_metrics.json` 與各折 `history.json`。

| 指標 | 數值 |
|---|---:|
| OOF Macro-F1（9,000 筆） | **89.03%** |
| fold 平均 Macro-F1 | 89.00% |
| fold 樣本標準差 | 1.98 個百分點 |
| 原 900 驗證 id 在 OOF 中的 Macro-F1 | 91.28% |
| 單模型 weighted_sqrt 同 900 筆（既有紀錄） | 92.65%（Kaggle 0.9049） |

| 類別 | precision | recall | F1 |
|---|---:|---:|---:|
| negative | 95.61% | 96.42% | 96.01% |
| neutral | 82.95% | 78.73% | 80.78% |
| positive | 89.81% | 90.78% | 90.29% |

OOF 混淆矩陣（列為真實，欄為預測，順序 negative／neutral／positive）：`[[6136,158,70],[220,1051,64],[62,58,1181]]`。

各 fold 驗證 Macro-F1（每 epoch 為訓練時 AMP 評估；「OOF」欄為 FP32 重新推論）：

| fold | 訓練／驗證筆數 | epoch 1 | epoch 2 | epoch 3 | 最佳 epoch | OOF（FP32） | 訓練秒數 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 7,199／1,801 | 85.97 | 88.75 | 89.50 | 3 | 89.50 | 1,262 |
| 1 | 7,200／1,800 | 87.55 | 87.33 | 87.70 | 3 | 87.70 | 939 |
| 2 | 7,200／1,800 | 88.68 | 91.35 | 92.17 | 3 | 92.26 | 948 |
| 3 | 7,200／1,800 | 86.08 | 87.38 | 88.01 | 3 | 88.01 | 1,594 |
| 4 | 7,201／1,799 | 86.81 | 87.55 | 87.31 | 2 | 87.55 | 1,811 |

- 5 折全部通過 reload 驗證；跳過 AMP 更新 3–4 次（占約 1,350 次的 0.3%）。fold 2 的 AMP／FP32 差異（92.17 vs 92.26）來自評估精度。
- 集成 CSV：`outputs/cv5_sqrt_oof/submissions/cv5_folds012_mean.csv`，1,000 筆，SHA-256 `7037e623cf00017c02c2ec1482a1ae903b61bf2c78a07e1b130e55d4e7d61453`；未提交 Kaggle。

## 結論

標示「已驗證」為本次數據直接支持；「推測」為尚未檢驗的解釋。

1. 已驗證：現行設定較可靠的水準約 89.0%（OOF），低於 900 驗證的 92.65%。原 900 筆在 OOF 中的 91.28% 也高於整體約 2.25 個百分點，表示那 900 筆相對容易。單模型 92.65% 與 OOF 中同 900 筆 91.28% 的其餘差距，可能來自單模型訓練資料 90% 對 80%（推測）。
2. 推論：fold 間標準差約 2 個百分點（每折約 1,800 筆），900 筆的抽樣雜訊應更大；sqrt 與 quarter 在 900 驗證上 0.74 個百分點的差距不足以排序，支持改用 OOF 比較設定。
3. 已驗證：neutral 是最弱類別（F1 80.78%），最大錯誤為 neutral 判成 negative（220 筆）。
4. 推測：5 折中 4 折最佳 epoch 是最後一個，且曲線多仍上升，可能訓練不足。這是步驟 3 之後可考慮的單一因素，本步不改。
5. 決策：OOF 89.03% 作為步驟 3 的基準；步驟 3 先以相同 5 折、相同設定，只將 class weighting 改為 quarter-inverse。

## 限制

- 每折以其驗證折挑最佳 epoch，OOF 有輕微樂觀偏誤（候選僅 3 個 epoch）。
- fold 0／1／2 集成為預先固定，無法以 OOF 評估，也未有 Kaggle 結果。
- 單一切分 seed（42）與單一訓練 seed；fold 標準差只描述切分間波動，不含訓練 seed 變異。
- 同折內的 AMP 評估與 FP32 OOF 有微小差異（fold 2 為 0.09 個百分點）。

## 產物

- 切分：`outputs/cv5_seed42/`（`folds_manifest.json`、`fold0..4.json`）
- 各折訓練：`outputs/cv5_sqrt/fold{0..4}/`（`run_config.json`、`history.json`、`curves.png`、`summary.json` 等）
- checkpoint：`saved_models/cv5_sqrt/fold{0..4}/`
- 日誌：`outputs/cv5_sqrt_train.log`
- OOF：`outputs/cv5_sqrt_oof/`（`oof_metrics.json`、`oof_predictions.csv`、各折 logits npz、`submissions/cv5_folds012_mean.csv`）
- 登錄：[registry](../../experiments/registry.json) 的 `cross_validations`（`cv5_sqrt`），摘要見 [實驗總表](../Experiment_Results.md)

## 重現

```powershell
.venv/Scripts/python.exe main.py kfold --k 5 --seed 42 --output-dir outputs/cv5_seed42
bash tools/run_cv5_sqrt.sh   # 已有 summary.json 的折會跳過；輸出目錄須為新的
.venv/Scripts/python.exe tools/refresh_experiments.py --check
```

## 耗時與環境

- 訓練 12:27–14:18（日誌時間，2026/11/16），各折訓練秒數合計約 6,555 秒（約 109 分鐘）；OOF 推論於 14:20 完成。
- Python 3.12.10、torch 2.9.1+cu126、CUDA 12.6、NVIDIA GeForce RTX 3050 Laptop GPU（4 GB），峰值配置約 1,904 MB。
