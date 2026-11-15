# 較溫和類別權重對照 A

2026/11/15。三輪訓練正常結束，最佳 validation Macro-F1 **91.9119%**，較半輪控制組下降 **0.7369 個百分點**；三類 F1 均下降。使用者於同日回報 Kaggle **0.9122**，已提升為 Kaggle 首選候選；此回報未登入平台獨立核驗。索引與收尾檢查見文末。

假設：將類別權重由 `n^-0.5` 改為 `n^-0.25`，可改善 precision／recall 平衡與泛化。對照為 `weighted_half_epoch_control`；唯一主要變因為權重指數，其餘維持指定 BERT 基底、固定 8,100／900 切分、seed 42、原空白前處理、batch 2／累積 8、max_length 64、AdamW 2e-5、weight decay 0.01、warmup 10%、AMP、梯度檢查點與每半輪評估。兩種權重均只依 train 計算，正規化成訓練樣本平均權重 1。

決策依據：最佳 validation Macro-F1、三類 precision／recall／F1、錯判數及與控制組的差異；≥0.3 個百分點為後續確認優先門檻，不是統計顯著性。無論結果是否改善都產生真實模型 CSV，內容相同不重複提交；平台分數保持未知，不代為提交。

上限：1 次完整訓練、3 epochs、45 分鐘；測試失敗、來源／切分不符、非有限 loss、重載失敗或超時即停。完成條件為六點曲線、最佳 checkpoint 重載、AMP／FP32 驗證、1,000 筆 CSV、對照報告與 registry／snapshot 核對。輸出 `outputs/weighted_quarter/`，不覆寫既有成果。本輪啟動後核心訓練預算累計 2／9；不自動開始設定 B。

## 實作與執行

`class_weights` 新增 `quarter-inverse`，保留原 `sqrt-inverse` 的運算方式。三類順序為 negative／neutral／positive，本輪權重為 `[0.8762142062, 1.2948673964, 1.3030822277]`。run config 與半輪控制組逐欄比較，差異只有 `class_weighting`、`class_weights` 與 `training_code_sha256`；來源 hash 另與實際程式核對。

執行 `.venv/Scripts/python.exe -m unittest tests.test_training tests.test_model_source`，17 項測試通過。新增測試以計數 `[16, 4, 1]` 檢查權重比 `[1, sqrt(2), 2]`、訓練樣本平均權重 1、相較平方根權重更溫和，以及缺類別時拒絕計算。

```powershell
.venv/Scripts/python.exe -u main.py train --device cuda --seed 42 --epochs 3 --eval-interval 0.5 --batch-size 2 --accumulation-steps 8 --learning-rate 2e-5 --class-weighting quarter-inverse --max-length 64 --preprocessing whitespace-v1 --output-dir outputs/weighted_quarter --checkpoint outputs/weighted_quarter/checkpoint
```

同一指令不能直接覆寫既有產物。啟動器將單次訓練限制在 2,700 秒，保存 [執行狀態](../../outputs/weighted_quarter.execution.json) 與 [日誌](../../outputs/weighted_quarter.log)。

## 結果與對照

| Epoch | Train Macro-F1 | Validation Macro-F1 | 成功／嘗試更新 | 累積 skip |
|---:|---:|---:|---:|---:|
| 0.5 | 87.4747% | 87.6708% | 252／254 | 2 |
| 1.0 | 93.0236% | 88.8994% | 505／507 | 2 |
| 1.5 | 94.9254% | 89.9218% | 759／761 | 2 |
| 2.0 | 96.4081% | 88.0994% | 1,011／1,014 | 3 |
| 2.5 | 97.4010% | 90.1401% | 1,265／1,268 | 3 |
| 3.0 | 98.0614% | **91.9119%** | 1,518／1,521 | 3 |

最佳 checkpoint 為 epoch 3。六個評估點均保存於 [history](../../outputs/weighted_quarter/history.json)，[曲線](../../outputs/weighted_quarter/curves.png) 已目視核對點數、train／validation 圖例及軸標。訓練 loss 持續下降而驗證 loss 有波動，不能只依 train 表現判斷泛化提升。

| 指標 | 半輪控制組 | 較溫和權重 A | 差值（百分點） |
|---|---:|---:|---:|
| Macro-F1 | 92.6488% | 91.9119% | -0.7369 |
| negative F1 | 97.0359% | 96.4761% | -0.5598 |
| neutral F1 | 86.4662% | 85.6089% | -0.8573 |
| positive F1 | 94.4444% | 93.6508% | -0.7937 |
| 錯判筆數 | 44／900 | 50／900 | 多 6 筆 |

| A 類別 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 96.0998% | 96.8553% | 96.4761% | 636 |
| neutral | 84.6715% | 86.5672% | 85.6089% | 134 |
| positive | 96.7213% | 90.7692% | 93.6508% | 130 |

neutral recall 較控制組略升（85.8209% → 86.5672%），precision 卻下降（87.1212% → 84.6715%），不足以補償整體損失。混淆矩陣按 negative／neutral／positive 排列，列為真實、欄為預測：`[[616,17,3],[17,116,1],[8,4,118]]`。本次未支持「較溫和權重改善泛化」的假設，也不表示其他資料切分必然相同。

[summary](../../outputs/weighted_quarter/summary.json) 記錄正常重載成功、總耗時 **1,177.51 秒（約 19.63 分鐘）**、CUDA peak allocated **1,904.24 MiB**。共 1,521 次更新嘗試、1,518 次成功、3 次 AMP skip；scheduler 前進 1,518 步。skip 出現時間與控制組不同，屬改變 loss 後的實際訓練結果，不是新增手動變因。總耗時與控制組後補的 epoch 區間時間定義不同，不作速度提升比較。

## 重載、精度與預測

訓練內重載與獨立 AMP 驗證均重現保存的 loss／混淆矩陣。AMP／FP32 各驗證 900 筆，標籤差異 **0**、Macro-F1 均為 91.9119%、錯判均為 50；機率最大差 0.00247562、平均差 0.00001832。保留 AMP 訓練／驗證及 FP32 正式推論，證據見 [設定與精度比較](../../outputs/weighted_quarter/control_comparison.json)、[逐筆機率](../../outputs/weighted_quarter/precision/probabilities.csv) 與 [錯判紀錄](../../outputs/weighted_quarter/precision/errors.csv)。這次未額外跑 benchmark AMP 比較。

```powershell
.venv/Scripts/python.exe main.py predict --device cuda --batch-size 2 --checkpoint outputs/weighted_quarter/checkpoint --predictions outputs/weighted_quarter/predictions.csv
```

已產生 [1,000 筆 CSV](../../outputs/weighted_quarter/predictions.csv)，欄位、唯一 ID、原始順序與標籤均核對。與半輪控制組／weighted_sqrt 相差 **11 筆**，與當時 Kaggle 單模型候選 weighted_lr3e5 相差 **14 筆**，且與所有既有 CSV 均不同。沿用實驗均保存可提交 CSV 的偏好，保留此探索檔；未代為提交，不推測平台分數。

- CSV SHA-256：`800684403958197f4de9e0641b829c3240e24e7f49f9de8253196bf5c7911294`
- 模型權重 SHA-256：`03abfa2f4014b2a5c51932280e02679475b3f2483a18642d74e564f924fe5dd6`
- 固定基底、原始資料、切分與程式來源 hash：見 [run config](../../outputs/weighted_quarter/run_config.json) 及 [snapshot](../../experiments/snapshot.json)。

## 決策與後續

本輪本機驗證未改善，未達原定 +0.3 個百分點門檻。後續使用者回報 Kaggle 0.9122，較原最高 0.9052 提升 0.0070（0.70 個百分點），距 0.9200 仍差 0.0078（0.78 個百分點）。registry 已將 Kaggle 候選改為 weighted_quarter，分數來源標示使用者回報、狀態 user_reported；原驗證基準與共同控制組不變，release_candidate 仍為 null。

本機與 Kaggle 排序不同，因此保留本輪供第三階段穩定性確認時評估，不能僅憑單一平台分數宣稱跨切分改善。下一項仍按原計畫進行 B，不據此擴大權重搜尋。本次僅登錄回報，未新增訓練、推論或平台操作；新候選尚未完成獨立封裝演練。

核心完整訓練預算已完成 **2／9**，剩餘上限 7 次；本輪只執行 1 次完整訓練、1 次 FP32 benchmark 推論及 2 次獨立驗證精度推論，未重跑既有模型訓練。下一步為 B：AdamW 分層學習率，上層 2e-5、往下逐層乘 0.9、head 2e-5，回到控制組的反平方根權重；本輪未開始 B。

## 訓練完成時的收尾檢核（Kaggle 回報前）

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：開始與恢復上下文時均讀 AGENTS；變更限 HW3，沿用指定基底與課程切分。 |
| 目的完成 | 通過：較溫和權重實作、一次三輪訓練、六點曲線、重載、精度驗證與 CSV 均完成。 |
| 查重與成本 | 通過：沿用半輪控制組，只改預定權重指數；未重訓既有模型，核心預算累計 2／9。 |
| 既有工作保全 | 通過：先檢視 diff；與本輪開始前 snapshot 比較，8 份 checkpoint、10 份 CSV／來源紀錄、演練包與確認切分均不變。 |
| 驗證證據 | 通過：17 項相關測試、重載、AMP／FP32 900 筆一致、1,000 筆 CSV 及曲線目視核對。 |
| 實驗可追溯 | 通過：registry 已登錄；`python tools/refresh_experiments.py` 及 `--check` 均通過，核對 11 runs／9 checkpoints 及所有 CSV，保存比較證據 hash。 |
| 文件與狀態 | 通過：README、計畫、交件清單、索引與總表已同步；6 份 Markdown 共 92 個本機連結存在，`git diff --check` 無空白錯誤。 |
| 交付誠實 | 通過：結果下降如實記錄；Kaggle 分數未知、未提交，未宣稱達 92% 或完成交件；B 尚未執行。 |

檢核：適用項 **8／8 通過**；不適用：無。本輪 A 完成，下一個具體工作為分層學習率 B。

## 2026/11/15 Kaggle 回報登錄檢核

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：已讀 AGENTS，僅更新 HW3 分數、候選與相關文件。 |
| 目的完成 | 通過：0.9122 已登錄為使用者回報，Kaggle 候選已更新。 |
| 查重與成本 | 通過：沿用現有模型與 CSV，沒有新增訓練或推論，預算仍為 2／9。 |
| 既有工作保全 | 通過：檢視既有 diff；9 份 checkpoint、11 份 CSV／來源紀錄與演練包均不變。 |
| 驗證證據 | 通過：refresh、--check、git diff --check 通過；5 份 Markdown 的 62 個本機連結有效。 |
| 實驗可追溯 | 通過：11 runs／9 checkpoints 核對通過，分數對應既有 weighted_quarter 產物。 |
| 文件與狀態 | 通過：總表、README、計畫與交件清單同步，新候選封裝仍列待辦。 |
| 交付誠實 | 通過：使用者回報與平台核驗分開記錄，未宣稱達 92%、泛化已確認或正式交件完成。 |

本次分數登錄檢核：適用項 **8／8 通過**；不適用：無。
