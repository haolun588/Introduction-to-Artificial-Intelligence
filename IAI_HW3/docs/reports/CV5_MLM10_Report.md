# 五折交叉驗證：MLM 續訓 10 epochs 後微調（cv5_sqrt_ep5 vs cv5_sqrt_ep5_mlm10）

日期：2026/11/17。狀態：已完成（5 次 MLM 續訓＋5 次微調＋1 次 OOF／集成 FP32 推論＋1 次配對比較）；**未達預先登記門檻，不列候選、不提交 Kaggle（使用者後續仍提交，見 Kaggle 結果）**（OOF 為目前 CV 組中最高）。集成 CSV 由工具自動產生但狀態 `not_submitted`，Kaggle 分數為 `null`。對應 [92% 計畫](../Target_92_Plan.md) 第二次路線調整順序 3（預先登記見該檔「順序 3 預先登記」，訓練前寫下）。

## 目的與假設

- 目的：檢驗「只用各折訓練子集文字做 MLM 續訓（任務適應預訓練）」能否提升 OOF Macro-F1。
- 假設（事前）：續訓讓 BERT 適應航空推文語言，預期 +0.3～1 個百分點（推測）。對照：`cv5_sqrt_ep5`（[報告](CV5_Sqrt_Ep5_Report.md)，OOF 89.22%）。
- 唯一變因：微調初始權重改為該折 MLM 續訓後的權重（分類頭仍新初始化）。其餘（`cv5_seed42` 切分、seed 42、sqrt-inverse、5 epochs、eval-interval 1.0）全同。
- MLM 設定（預先固定、不調參）：起點 `google-bert/bert-base-uncased`；語料僅該折訓練子集文字（約 7,200 筆）；遮罩 15%（80/10/10）；lr 5e-5、batch 2、10 epochs，儲存最後一個 epoch，不做選擇；validation 文字只用來記錄 MLM eval loss。
- 預算：5 次 MLM 續訓＋5 次微調，不追加（fold 4 的重跑見「事故紀錄」，為崩潰後的修正重跑，非追加實驗）。
- 決策規則（預先固定）：OOF Δ（mlm10 − ep5）≥ +0.5 個百分點 → 列為候選並交使用者上傳 Kaggle；否則停止此方向，與使用者討論備選或退路。

## 方法

- 訓練：`bash tools/run_cv5_mlm.sh cv5_sqrt_ep5_mlm10 sqrt-inverse 5 10`（先以 fold 0 試跑）；每折先續訓 MLM、再從該折 MLM 權重微調。
- OOF：與對照相同（FP32、batch 2）；集成取 fold 0／1／2 的 FP32 softmax 等權平均（預先固定，符合最多 3 個模型），僅作提交檔。
- 配對比較：`python -m tools.compare_oof --a outputs/cv5_sqrt_ep5_oof --b outputs/cv5_sqrt_ep5_mlm10_oof --output outputs/cv5_compare_ep5_vs_mlm10.json`；檢查切分檔 SHA-256、資料集，各 fold 分數可由 OOF 預測重算；配對 bootstrap（seed 20261116、2,000 次）。Δ 一律為 mlm10 − ep5。

## MLM 續訓結果

數字取自各折 `outputs/cv5_sqrt_ep5_mlm10_mlm/fold{N}/summary.json`、`history.json`。每折每 epoch 的 MLM 驗證 loss 皆單調下降。

| fold | 訓練文字筆數 | epoch 1 驗證 loss | epoch 10 驗證 loss | 最終訓練 loss | 耗時（秒） |
|---:|---:|---:|---:|---:|---:|
| 0 | 7,199 | 2.587 | 2.198 | 1.823 | 3,217 |
| 1 | 7,200 | 2.576 | 2.170 | 1.879 | 3,458 |
| 2 | 7,200 | 2.611 | 2.239 | 1.852 | 3,813 |
| 3 | 7,200 | 2.565 | 2.169 | 1.851 | 3,227 |
| 4 | 7,201 | 2.577 | 2.185 | 1.857 | 3,181 |

各折 MLM 續訓約 53–64 分鐘、微調約 28–37 分鐘，含載入與推論每折約 80–100 分鐘（筆電 GPU 降頻）。

## 結果

數字取自 `outputs/cv5_compare_ep5_vs_mlm10.json`、兩邊的 `oof_metrics.json` 與各折 `history.json`／`summary.json`。

| 指標 | ep5（cv5_sqrt_ep5） | mlm10 | Δ（mlm10 − ep5，百分點） |
|---|---:|---:|---:|
| OOF Macro-F1（9,000 筆） | 89.22% | 89.63% | **+0.41** |
| negative F1 | 96.02% | 96.15% | +0.13 |
| neutral F1 | 80.75% | 81.66% | +0.91 |
| positive F1 | 90.90% | 91.09% | +0.18 |
| fold 平均 Macro-F1 | 89.22% | 89.64% | +0.42 |
| fold 樣本標準差（百分點） | 1.51 | 1.58 | — |
| 原 900 驗證 id 在 OOF 中的 Macro-F1 | 90.42% | 90.89% | +0.47 |

逐 fold 驗證 Macro-F1（FP32 OOF 重算；mlm10 勝 3/5 折）：

| fold | ep5 | mlm10 | Δ（百分點） |
|---:|---:|---:|---:|
| 0 | 88.11 | 89.31 | +1.20 |
| 1 | 88.56 | 89.60 | +1.04 |
| 2 | 91.85 | 92.28 | +0.43 |
| 3 | 89.11 | 88.92 | −0.19 |
| 4 | 88.46 | 88.08 | −0.38 |

各類 precision／recall／F1（%，OOF）：

| 類別 | ep5 P | ep5 R | ep5 F1 | mlm10 P | mlm10 R | mlm10 F1 |
|---|---:|---:|---:|---:|---:|---:|
| negative | 95.61 | 96.43 | 96.02 | 95.79 | 96.51 | 96.15 |
| neutral | 82.22 | 79.33 | 80.75 | 82.12 | 81.20 | 81.66 |
| positive | 91.18 | 90.62 | 90.90 | 92.27 | 89.93 | 91.09 |

OOF 混淆矩陣（列為真實，欄為預測，順序 negative／neutral／positive）：

| 設定 | 矩陣 |
|---|---|
| ep5 | `[[6137,170,57],[219,1059,57],[63,59,1179]]` |
| mlm10 | `[[6142,169,53],[206,1084,45],[64,67,1170]]` |

配對 bootstrap（9,000 筆、seed 20261116、2,000 次）：Δ 平均 +0.41 個百分點，95% 區間 **[−0.21, +1.06]**，Δ≤0 的比例 10.3%。兩邊預測不同的有 366 筆：ep5 答對 163、mlm10 答對 184。

mlm10 微調各折每 epoch 驗證 Macro-F1（訓練時 AMP 評估，%）：

| fold | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | 最佳 epoch | 微調秒數 | 跳過／嘗試 AMP 更新 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 86.25 | 88.18 | 87.53 | 88.67 | 89.31 | 5 | 1,889 | 3／2,250 |
| 1 | 89.60 | 89.05 | 89.26 | 89.35 | 88.91 | 1 | 1,989 | 3／2,250 |
| 2 | 90.36 | 90.90 | 89.99 | 92.25 | 92.28 | 5 | 2,224 | 3／2,250 |
| 3 | 87.78 | 87.82 | 87.78 | 88.92 | 88.43 | 4 | 1,789 | 3／2,250 |
| 4 | 85.05 | 86.26 | 87.87 | 88.08 | 87.19 | 4 | 1,674 | 3／2,255 |

集成 CSV：`outputs/cv5_sqrt_ep5_mlm10_oof/submissions/cv5_folds012_mean.csv`，SHA-256 `ad1124f0a7cca3f9722c0627c652ec8f3084a9304c807c7cbc1ffadb5a890755`；未提交 Kaggle，狀態 `not_submitted`。

## 判定

- **未達預先登記門檻**：OOF Δ +0.41 個百分點 < +0.5。不列候選、不提交 Kaggle；與使用者討論後續（備選或退路）。
- 已驗證：Δ 方向為正、為目前 OOF 最高的 CV 組（89.63%）；但 95% 區間 [−0.21, +1.06] 包含 0，fold 勝 3/5 且 fold 3、4 為負。增益集中在 neutral（F1 +0.91，recall 79.3% → 81.2%）。
- 前三折（fold 0、1、2）Δ 均為正、後兩折為負，原因未檢驗。

## Kaggle 結果

- `cv5_folds012_mean`（`outputs/cv5_sqrt_ep5_mlm10_oof/submissions/cv5_folds012_mean.csv`，SHA-256 `ad1124f0...a890755`）Kaggle 0.9152，來源為使用者 2026/11/17 回報（user_reported，非平台核驗）；因另一份同名檔（`cv5_sqrt_ep5` 版）先前已回報 0.8971，此分數依時序推定為 mlm10 版（推定）。
- 對照：`cv5_sqrt_ep5` 集成 0.8971（Δ +1.81 個百分點）；`weighted_quarter` 0.9122（非 MLM，目前第二高）。
- 偏離預先登記：OOF Δ 僅 +0.41（未達預先登記門檻 +0.5），依預先登記本不提交；此次提交為使用者決定，記錄於此。
- OOF 與 Kaggle 對照：cv5_sqrt_ep5 OOF 89.22 → Kaggle 89.71；mlm10 OOF 89.63 → Kaggle 91.52。
- 解讀（推論）：Kaggle Δ（+1.81）遠大於 OOF Δ（+0.41）。先前觀察到只換 seed 即可使 Kaggle 變動約 1.7 個百分點（0.9049 vs 0.8878），故 +1.81 不能視為 MLM 效果大小的可靠估計。

## 事故紀錄與修正

- **事故**：fold 4 首次 MLM 續訓於第 4 epoch 崩潰，`AssertionError: No inf checks were recorded for this optimizer`。
- **原因**：fold 4 訓練集 7,201 筆（奇數），batch 2 → 3,601 batch；最後一個梯度累積窗口只有 1 筆，該筆抽到零遮罩 → 無 backward 卻呼叫 `scaler.step`。
- **修正**：`src/mlm.py` 在整個窗口無 backward 時跳過 optimizer／scheduler 更新並記錄 `empty_windows`；新增 2 個回歸測試（全部測試 118 OK）。
- **為何只重跑 fold 4**：fold 0–3 每窗口都是完整 8 個 batch，從未進入該路徑，修正不影響其行為，故不重跑。fold 4 以修正後程式碼重跑，其 `training_code_sha256`（`src/mlm.py`）與 fold 0–3 不同，fold 4 summary 的 `empty_windows` = 1。
- 崩潰產物保留於 `outputs/cv5_sqrt_ep5_mlm10_mlm_fold4_crashed_20261117/`；日誌 `outputs/cv5_sqrt_ep5_mlm10_fold4_train.log`、`outputs/cv5_sqrt_ep5_mlm10_folds1-4_train.log`（含崩潰）、`outputs/cv5_sqrt_ep5_mlm10_fold4_rerun_train.log`。

## 合規說明

- 講義 p.14 要求只用提供資料切出的 train／validation 訓練與驗證：續訓只用各折訓練子集文字，不用標籤、benchmark 文字或 validation 文字（validation 文字僅用於記錄 MLM eval loss）。
- 講義 p.9 規定「ONLY use google-bert/bert-base-uncased as your base model」，全講義未提及預訓練／續訓，也未界定 base model 範圍。「起點為指定模型、續訓屬自建訓練管線」為主會話推測，非講義明文。使用者於 2026/11/17 知悉此不確定性，判斷風險可接受並決定執行（使用者決定，非講義確認）。若之後助教回覆不允許，此方向產物不得用於交件。見 [計畫](../Target_92_Plan.md)「順序 3 預先登記」。

## 限制

- 單 seed（42）、每折訓練一次；bootstrap 只反映評估樣本的抽樣雜訊，不含訓練隨機性。單 seed 五折 CV 偵測力約 0.5 個百分點，Δ +0.41 落在偵測力以下，無法區分真增益與雜訊。
- 每折以其驗證折挑最佳 epoch，OOF 有選模樂觀偏誤；兩組皆 5 epochs 候選，影響相近（推論）。
- fold 4 的 MLM 續訓程式碼與 fold 0–3 不同（僅修正空窗口處理，未影響其他折，推論）。
- fold 0／1／2 集成為預先固定，無法以 OOF 評估，也沒有 Kaggle 結果。
- 同折內的 AMP 評估與 FP32 OOF 有微小差異。

## 產物

- 切分：`outputs/cv5_seed42/`（與對照共用）
- MLM 續訓：`outputs/cv5_sqrt_ep5_mlm10_mlm/fold{0..4}/`；權重 `saved_models/cv5_sqrt_ep5_mlm10_mlm/fold{0..4}/`
- 微調：`outputs/cv5_sqrt_ep5_mlm10/fold{0..4}/`；checkpoint `saved_models/cv5_sqrt_ep5_mlm10/fold{0..4}/`（`training_config.json` 的 `init_from` 記錄各折 MLM 權重路徑與 SHA-256）
- 日誌：`outputs/cv5_sqrt_ep5_mlm10_fold0_train.log`、`..._folds1-4_train.log`、`..._fold4_train.log`、`..._fold4_rerun_train.log`
- OOF：`outputs/cv5_sqrt_ep5_mlm10_oof/`（`oof_metrics.json`、`oof_predictions.csv`、各折 logits npz、`submissions/cv5_folds012_mean.csv`）
- 配對比較：`outputs/cv5_compare_ep5_vs_mlm10.json`
- 登錄：[registry](../../experiments/registry.json) 的 `cross_validations`（`cv5_sqrt_ep5_mlm10`，含 `comparison` 欄位）

## 重現

```powershell
bash tools/run_cv5_mlm.sh cv5_sqrt_ep5_mlm10 sqrt-inverse 5 10   # 輸出目錄須為新的
.venv/Scripts/python.exe -m tools.compare_oof --a outputs/cv5_sqrt_ep5_oof --b outputs/cv5_sqrt_ep5_mlm10_oof --output outputs/cv5_compare_ep5_vs_mlm10.json
.venv/Scripts/python.exe tools/refresh_experiments.py --check
```

## 耗時與環境

- 2026/11/17 12:51 起至 20:39 OOF 完成（日誌時間，含 fold 4 崩潰與重跑）；各折 MLM 續訓秒數合計 16,896、微調秒數合計 9,565。
- Python 3.12.10、torch 2.9.1+cu126、CUDA 12.6、NVIDIA GeForce RTX 3050 Laptop GPU（4 GB）（沿用同機同環境，與對照報告相同）。
