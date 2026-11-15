# AMP／FP32 一致性檢查

日期：2026/11/15。狀態：已完成 8 次推論，兩個候選的 900 筆驗證與 1,000 筆 benchmark 標籤皆無 AMP／FP32 差異；歷史驗證與原提交 CSV 重現通過。

## 假設與預算

- 假設：現有驗證的 FP16 autocast 與正式推論 FP32 可能改變邊界樣本的標籤；尚未將此視為 Kaggle 失分原因。
- 對照：沿用 `weighted_sqrt`、`weighted_lr3e5` 的已保存最佳 checkpoint、固定 900 筆 validation、原 1,000 筆 benchmark；不重新訓練，不使用 benchmark 標籤。
- 唯一主要變因：模型 forward 是否啟用 CUDA FP16 autocast；相同 FP32 權重、batch 2、文字順序、動態 padding、前處理及 max_length。機率統一以 FP32 softmax 計算；預測以 logits argmax 決定，與原程式一致。
- 決策資訊：逐筆標籤、各類 precision／recall／F1、混淆矩陣、錯判數、機率絕對差、信心與 top-2 margin、耗時；AMP 是否重現已保存驗證、FP32 是否重現既有提交 CSV。
- 上限：0 次訓練，2 模型 × 2 資料集合 × 2 精度＝8 次完整推論，另每種精度暖機 3 個 batch；最多 30 分鐘推論時間，沒有必要不重跑。
- 停止條件：來源／ID／類別對應不符、非有限數值或超時立即停止；若歷史驗證或 CSV 無法重現，保留結果並先調查，不進入新訓練。若發現精度造成標籤差異，另記採用方式及必要修正；若只有小幅機率差、標籤全相同，保留 AMP 驗證與 FP32 正式推論，後續新候選重做相應核對。
- 耗時採暖機後、CUDA synchronize 前後包住完整推論迴圈（含 CPU 搬移及驗證 loss），不含模型載入、tokenization、寫檔；每項只量一次，僅作本機操作參考，不作效能基準宣稱。

## 重現與產物

從 `IAI_HW3` 執行，輸出目錄必須不存在或為空：

```powershell
.venv/Scripts/python.exe -m tools.audit_precision --output-dir outputs/precision_audit_20261115
```

工具只讀既有模型與 CSV，將 summary、兩種精度的逐筆機率／預測、錯判／差異清單及重現 CSV 寫入獨立目錄。原始 checkpoint、歷史指標與提交檔均保留。

本次結果：[summary.json](../../outputs/precision_audit_20261115/summary.json)。每個模型的 `validation/` 有 `probabilities.csv`、`errors.csv`、`disagreements.csv`；`benchmark/` 有 `probabilities.csv`、`disagreements.csv`、`predictions_amp.csv`、`predictions_fp32.csv`。benchmark 不計算 F1 或錯判率，沒有使用其標籤。

## 結果與精度決策

| 候選 | AMP Macro-F1 | FP32 Macro-F1 | 驗證錯判（兩者相同） | 驗證／benchmark 標籤差異 | FP32 與原 CSV 差異 |
|---|---:|---:|---:|---:|---:|
| `weighted_sqrt` | 92.6488% | 92.6488% | 44／900 | 0／0 | 0／1,000 |
| `weighted_lr3e5` | 92.1045% | 92.1045% | 49／900 | 0／0 | 0／1,000 |

AMP 驗證的混淆矩陣與 loss 均重現已保存結果。FP32 loss 分別為 0.26112823、0.27520391，AMP 為 0.26116083、0.27515700；此小幅數值差沒有改變預測標籤。兩個候選的 AMP benchmark CSV 也與各自原檔標籤一致。

**決策：保留 AMP 訓練／驗證及 FP32 正式單模型推論，不因本次結果重訓或替換候選。** 本次受測模型的 benchmark 預測完全相同，因此沒有證據將現有 Kaggle 分數差距歸因於這兩條精度路徑。結果只適用於本次模型、batch 2、資料與環境；未測 CPU、其他 batch、其他硬體或未來 checkpoint，不將此次結論當作普遍數值一致保證。

各類指標如下，AMP 與 FP32 完全相同；support 為 negative 636、neutral 134、positive 130：

| 候選 | 類別 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| `weighted_sqrt` | negative | 96.2848% | 97.7987% | 97.0359% |
| `weighted_sqrt` | neutral | 87.1212% | 85.8209% | 86.4662% |
| `weighted_sqrt` | positive | 97.5410% | 91.5385% | 94.4444% |
| `weighted_lr3e5` | negative | 95.9565% | 97.0126% | 96.4816% |
| `weighted_lr3e5` | neutral | 86.3636% | 85.0746% | 85.7143% |
| `weighted_lr3e5` | positive | 96.0000% | 92.3077% | 94.1176% |

機率絕對差以所有樣本 × 三類計算（0–1 機率尺度）：

| 候選 | 資料 | 平均差 | P99 差 | 最大差 | AMP／FP32 耗時 |
|---|---|---:|---:|---:|---:|
| `weighted_sqrt` | validation | 0.00002410 | 0.00053398 | 0.00424320 | 4.55／3.69 秒 |
| `weighted_sqrt` | benchmark | 0.00001814 | 0.00052877 | 0.00171806 | 4.99／4.19 秒 |
| `weighted_lr3e5` | validation | 0.00002068 | 0.00035081 | 0.00420320 | 4.56／4.45 秒 |
| `weighted_lr3e5` | benchmark | 0.00002220 | 0.00067801 | 0.00299981 | 6.22／5.39 秒 |

逐筆信心與 top-2 margin 已保存在機率 CSV；summary 記錄平均信心、最小 margin、最大信心差。最大機率差約 0.00424（0.424 個百分點），不能將「標籤一致」寫成「機率完全一致」。

## 成本、來源與後續工作

- 執行環境：Python 3.12.10、PyTorch 2.9.1+cu126、Transformers 4.57.3、RTX 3050 Laptop GPU；FP32 matmul precision 為 `highest`、CUDA matmul TF32 關閉。完整環境設定見 summary。
- 本次使用 0 次訓練、8／8 次預定完整推論；8 個計時迴圈合計約 38.03 秒，summary 起訖約 41 秒（不含 Python 啟動與 import）。新控制組／新設定的完整訓練預算尚未使用。
- snapshot 保存本次 summary、逐筆 CSV、程式與資料來源 hash；推論前核對 checkpoint 的全部登錄檔案 hash。兩份 FP32 重現 CSV 與原檔的 SHA-256 也一致。
- 完成的是本機精度與既有結果重現檢查，尚未做獨立封裝、乾淨解壓推論或平台收件核驗，也未改變 Kaggle 回報與最終候選狀態。
- 下一個具體工作：對目前 Kaggle 單模型候選做獨立封裝與乾淨解壓推論演練。新訓練之前的 optimizer／scheduler／累積／選模檢查與每半輪評估仍待辦，不能把本次精度檢查視為整個第一階段完成。

## 收尾檢核

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：已讀 AGENTS，僅變更 HW3；未重訓、未加入外部資料、未提交平台。 |
| 目的完成 | 通過：兩個候選共 8 次推論，標籤、F1、機率、耗時與重現結果均保留。 |
| 查重與成本 | 通過：沿用現有模型／切分；0 次訓練，推論未超過上限。 |
| 既有工作保全 | 通過：修改前檢視 diff，保留既有 Emoji 分數更新；輸出使用新目錄，模型與原 CSV hash 不變。 |
| 驗證證據 | 通過：`python -m unittest tests.test_precision_audit tests.test_training` 8／8 通過；CSV 回讀核對 7,600 筆各精度預測、機率、錯判與指標；4 份 benchmark 重現 CSV hash 與原檔一致；diff 無空白錯誤。 |
| 實驗可追溯 | 通過：獨立 audit 登錄未新增訓練候選；`python tools/refresh_experiments.py` 與 `--check` 通過，9 候選／7 checkpoint 保持一致；稽核程式／資料 hash 與實際來源相同。 |
| 文件與狀態 | 通過：入口、計畫、交件清單與本報告已更新；6 份相關 Markdown 共 67 個本機連結核對存在。 |
| 交付誠實 | 通過：明確區分精度檢查完成、Kaggle 目標未達成與交件尚未完成。 |

檢核：適用項 8／8 通過；不適用：無。上述通過僅指本次精度稽核工作。
