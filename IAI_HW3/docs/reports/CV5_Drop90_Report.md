# 五折交叉驗證：排除高信心不一致樣本（cv5_sqrt_ep5 vs cv5_sqrt_ep5_drop90）

日期：2026/11/16–17。狀態：已完成（5 次完整訓練＋1 次 OOF／集成 FP32 推論＋1 次配對比較）；**未達預先登記門檻，不列候選，停止錯標排除方向**。集成 CSV 由工具自動產生但狀態 `not_submitted`，Kaggle 分數為 `null`。對應 [92% 計畫](../Target_92_Plan.md) 第二次路線調整順序 2（預先登記見該檔「順序 2 預先登記」，訓練前寫下）；抽查依據見 [錯標抽查報告](Label_Noise_Audit_Report.md)。

## 目的與假設

- 目的：檢驗「自訓練集排除 OOF 高信心不一致樣本」能否提升 OOF Macro-F1。
- 假設（事前）：高信心不一致樣本多含錯標或模糊標籤，排除後 OOF／Kaggle 會改善。對照：`cv5_sqrt_ep5`（[報告](CV5_Sqrt_Ep5_Report.md)，OOF 89.22%）。
- 唯一變因：以 `--exclude-train-ids` 自每折訓練子集排除 473 筆；validation 不變。連帶變化：類別權重依過濾後的訓練子集重算（sqrt-inverse 規則不變）。其餘（`cv5_seed42` 切分、seed 42、sqrt-inverse、5 epochs、eval-interval 1.0）全同。
- 預算：5 次完整訓練，不追加。
- 決策規則（預先固定）：OOF Δ（drop90 − ep5）≥ +0.5 個百分點 → 列為候選並產生集成 CSV；否則停止此方向。次要觀察 neutral recall（原 79.3%），不改決策。

## 排除清單

- 路徑：`outputs/label_noise_audit_20261116/exclude_ids_p090.txt`，SHA-256 `ca7aeec4ea3e81a2e1a4d88dc9b6fae52e0d26b05447d3bb8c573664f77288da`（與各折 `run_config.json` 記錄一致），473 筆。
- 各折實際自訓練子集移除筆數（讀自各折 `run_config.json` 的 `exclude_train_ids`）：落在該折 validation 的樣本不移除（validation 不變）。

| fold | 清單筆數 | 自訓練移除 | 位於 validation（保留） | 排除後訓練筆數 | validation 筆數 |
|---:|---:|---:|---:|---:|---:|
| 0 | 473 | 385 | 88 | 6,814 | 1,801 |
| 1 | 473 | 368 | 105 | 6,832 | 1,800 |
| 2 | 473 | 399 | 74 | 6,801 | 1,800 |
| 3 | 473 | 375 | 98 | 6,825 | 1,800 |
| 4 | 473 | 365 | 108 | 6,836 | 1,799 |

每折移除＋保留 = 473，核對一致；各折移除合計 1,892 = 473 × 4（每筆清單樣本在 5 折中 4 折屬訓練、1 折屬 validation）。

## 方法

- 訓練：`bash tools/run_cv5.sh cv5_sqrt_ep5_drop90 sqrt-inverse 5 --exclude-train-ids outputs/label_noise_audit_20261116/exclude_ids_p090.txt`，5 折各訓練一次；日誌 `outputs/cv5_sqrt_ep5_drop90_train.log`。
- OOF：與對照相同（FP32、batch 2）；集成取 fold 0／1／2 的 FP32 softmax 等權平均（預先固定，符合最多 3 個模型），僅作提交檔。
- 配對比較：`python -m tools.compare_oof --a outputs/cv5_sqrt_ep5_oof --b outputs/cv5_sqrt_ep5_drop90_oof --output outputs/cv5_compare_ep5_vs_drop90.json`；檢查切分檔 SHA-256、資料集，各 fold 分數可由 OOF 預測重算；配對 bootstrap（seed 20261116、2,000 次）。Δ 一律為 drop90 − ep5。

## 結果

數字取自 `outputs/cv5_compare_ep5_vs_drop90.json`、兩邊的 `oof_metrics.json` 與各折 `history.json`／`summary.json`。

| 指標 | ep5（cv5_sqrt_ep5） | drop90 | Δ（drop90 − ep5，百分點） |
|---|---:|---:|---:|
| OOF Macro-F1（9,000 筆） | 89.22% | 89.34% | **+0.12** |
| negative F1 | 96.02% | 96.08% | +0.06 |
| neutral F1 | 80.75% | 81.10% | +0.35 |
| positive F1 | 90.90% | 90.85% | −0.05 |
| fold 平均 Macro-F1 | 89.22% | 89.34% | +0.12 |
| fold 樣本標準差（百分點） | 1.51 | 1.66 | — |
| 原 900 驗證 id 在 OOF 中的 Macro-F1 | 90.42% | 90.56% | +0.14 |

逐 fold 驗證 Macro-F1（FP32 OOF 重算；drop90 勝 3/5 折）：

| fold | ep5 | drop90 | Δ（百分點） |
|---:|---:|---:|---:|
| 0 | 88.11 | 88.70 | +0.59 |
| 1 | 88.56 | 89.10 | +0.54 |
| 2 | 91.85 | 92.12 | +0.27 |
| 3 | 89.11 | 89.09 | −0.02 |
| 4 | 88.46 | 87.69 | −0.77 |

各類 precision／recall／F1（%，OOF）：

| 類別 | ep5 P | ep5 R | ep5 F1 | drop90 P | drop90 R | drop90 F1 |
|---|---:|---:|---:|---:|---:|---:|
| negative | 95.61 | 96.43 | 96.02 | 95.31 | 96.86 | 96.08 |
| neutral | 82.22 | 79.33 | 80.75 | 85.29 | 77.30 | 81.10 |
| positive | 91.18 | 90.62 | 90.90 | 90.10 | 91.62 | 90.85 |

OOF 混淆矩陣（列為真實，欄為預測，順序 negative／neutral／positive）：

| 設定 | 矩陣 |
|---|---|
| ep5 | `[[6137,170,57],[219,1059,57],[63,59,1179]]` |
| drop90 | `[[6164,130,70],[242,1032,61],[61,48,1192]]` |

配對 bootstrap（9,000 筆、seed 20261116、2,000 次）：Δ 平均 +0.13 個百分點，95% 區間 **[−0.49, +0.76]**，Δ≤0 的比例 35.1%。兩邊預測不同的有 361 筆：ep5 答對 163、drop90 答對 176。

drop90 各折每 epoch 驗證 Macro-F1（訓練時 AMP 評估，%）：

| fold | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | 最佳 epoch | 訓練秒數 | 跳過／嘗試 AMP 更新 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 87.12 | 88.57 | 87.79 | 87.97 | 88.70 | 5 | 1,751 | 4／2,130 |
| 1 | 85.93 | 88.50 | 88.36 | 88.03 | 89.10 | 5 | 1,590 | 4／2,135 |
| 2 | 88.39 | 91.26 | 91.69 | 92.12 | 91.57 | 4 | 1,689 | 3／2,130 |
| 3 | 85.09 | 88.11 | 88.68 | 88.98 | 89.09 | 5 | 1,574 | 3／2,135 |
| 4 | 86.63 | 87.61 | 87.62 | 87.69 | 87.63 | 4 | 1,578 | 3／2,140 |

- 5 折全部通過 reload 驗證。
- 集成 CSV：`outputs/cv5_sqrt_ep5_drop90_oof/submissions/cv5_folds012_mean.csv`，SHA-256 `a112b5c89e62e39e072b9914d3641ae22d08021a1dc47676aa55338ab0c488d4`；未提交 Kaggle，狀態 `not_submitted`。

## 判定

- **未達預先登記門檻**：OOF Δ +0.12 個百分點 < +0.5。不列候選、停止錯標排除方向、不提交 Kaggle；下一步為第二次路線調整順序 3（MLM 續訓，待設計）。
- 已驗證：95% 區間 [−0.49, +0.76] 包含 0；fold 勝 3/5，但差距小且方向不一（fold 4 −0.77）。
- 已知偏誤（偏向 drop90）：排除清單由對照 OOF 產生，新 OOF 對自身略偏樂觀；即使如此仍未達門檻。

## 次要觀察（不作決策依據）

- neutral recall 79.3% → 77.3%（−2.0 個百分點），neutral → negative 219 → 242；negative recall（96.43 → 96.86）與 positive recall（90.62 → 91.62）上升；neutral precision 82.2 → 85.3。與預先登記的風險方向一致（被排除者多為 neutral，排除後 neutral 召回下降）。
- 推測（未檢驗）：排除也移除了部分標籤正確的 neutral 樣本（抽查中 A 組約 35% 為原標籤可能正確）。
- 最佳 epoch 為 5、5、4、5、4。

## 限制

- bootstrap 只反映評估樣本的抽樣雜訊，不含訓練隨機性；每折只用 seed 42 訓練一次。
- 每折以其驗證折挑最佳 epoch，OOF 有選模樂觀偏誤；兩組候選 epoch 數相同（5），此偏誤對比較影響較小（推論）。
- 排除清單來自對照 OOF（選樣偏誤，偏向 drop90，未量化）。
- 類別權重隨過濾後訓練子集重算，與「排除」連帶變化，未拆開。
- fold 0／1／2 集成為預先固定，無法以 OOF 評估，也沒有 Kaggle 結果。
- 同折內的 AMP 評估與 FP32 OOF 有微小差異。

## 產物

- 排除清單：`outputs/label_noise_audit_20261116/exclude_ids_p090.txt`
- 切分：`outputs/cv5_seed42/`（與對照共用）
- 各折訓練：`outputs/cv5_sqrt_ep5_drop90/fold{0..4}/`
- checkpoint：`saved_models/cv5_sqrt_ep5_drop90/fold{0..4}/`
- 日誌：`outputs/cv5_sqrt_ep5_drop90_train.log`
- OOF：`outputs/cv5_sqrt_ep5_drop90_oof/`（`oof_metrics.json`、`oof_predictions.csv`、各折 logits npz、`submissions/cv5_folds012_mean.csv`）
- 配對比較：`outputs/cv5_compare_ep5_vs_drop90.json`
- 登錄：[registry](../../experiments/registry.json) 的 `cross_validations`（`cv5_sqrt_ep5_drop90`，含 `comparison` 欄位）

## 重現

```powershell
bash tools/run_cv5.sh cv5_sqrt_ep5_drop90 sqrt-inverse 5 --exclude-train-ids outputs/label_noise_audit_20261116/exclude_ids_p090.txt   # 輸出目錄須為新的
.venv/Scripts/python.exe -m tools.compare_oof --a outputs/cv5_sqrt_ep5_oof --b outputs/cv5_sqrt_ep5_drop90_oof --output outputs/cv5_compare_ep5_vs_drop90.json
.venv/Scripts/python.exe tools/refresh_experiments.py --check
```

## 耗時與環境

- 訓練 2026/11/16 22:54–11/17 01:14（日誌時間，含 OOF 推論），各折訓練秒數合計 8,182 秒（約 136 分鐘），每折約 26–29 分鐘。
- Python 3.12.10、torch 2.9.1+cu126、CUDA 12.6、NVIDIA GeForce RTX 3050 Laptop GPU（4 GB）（沿用同機同環境，與對照報告相同）。
