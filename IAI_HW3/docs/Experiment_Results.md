# HW3 實驗總表

由 `python tools/refresh_experiments.py` 產生；不要手動修改此檔。
更新來源為 [registry.json](../experiments/registry.json)，完整雜湊與指標見 [snapshot.json](../experiments/snapshot.json)。

驗證分數讀取本機 summary／checkpoint；Kaggle 分數來自歷史使用者回報，未登入平台核驗。`待回報` 不等於未提交。

| 實驗 | 驗證 Macro-F1 | Kaggle | 與 Kaggle 候選不同筆數 | 決策 |
|---|---:|---:|---:|---|
| [未加權基準](../docs/reports/Baseline_Report.md) | 91.57% | 0.8973 | 14 | 保留基準；預設 checkpoint 仍指向此版本 |
| [加權 2e-5 / seed 42](../docs/reports/Weighted_CE_Report.md) | 92.65% | 0.9049 | 11 | 驗證基準 |
| [URL 正規化](../docs/reports/URL_Normalization_Report.md) | 91.28% | 0.8977 | 32 | 未採用；保留對照 |
| [加權 1e-5](../docs/reports/Learning_Rate_Report.md) | 91.46% | 0.9012 | 20 | 未採用；保留對照 |
| [加權 3e-5](../docs/reports/Learning_Rate_3e5_Report.md) | 92.10% | 0.9052 | 14 | 原 Kaggle 單模型候選；保留已驗證封裝 |
| [2e-5 / 3e-5 集成](../docs/reports/Ensemble_Error_Analysis_Report.md) | 92.58% | 0.9052 | 14 | 與原單模型候選同分；保留對照 |
| [加權 2e-5 / seed 43](../docs/reports/Seed43_Report.md) | 91.33% | 0.8878 | 28 | 未採用；保留對照 |
| [seed 42 / 43 集成](../docs/reports/Seed_Ensemble_Report.md) | 92.46% | 0.8983 | 17 | 未採用；保留對照 |
| [Emoji 簡短描述](../docs/reports/Emoji_Training_Report.md) | 91.68% | 0.9009 | 17 | 驗證與 Kaggle 均未改善；保留對照 |
| [半輪評估 / 加權 2e-5 控制組](../docs/reports/Half_Epoch_Control_Report.md) | 92.65% | 未新增提交 | 11 | 新方法共同控制組；CSV 與 weighted_sqrt 相同，不重複提交 |
| [較溫和 n^-0.25 權重](../docs/reports/Quarter_Weight_Report.md) | 91.91% | 0.9122 | 0 | Kaggle 最高分單模型候選；本機驗證下降，跨切分穩定性待確認；訓練報告僅記錄驗證集 AMP／FP32 檢查（標籤差異 0），不屬 precision_audit_20261115，未跑 benchmark AMP |
| [分層學習率 / decay 0.9](../docs/reports/Layer_LR_Report.md) | 92.03% | 0.8989 | 21 | 驗證與 Kaggle 均未改善；保留對照，不擴大 decay 搜尋 |
| [FGSM C1 / epsilon 0.00208093](../docs/reports/FGSM_Low_Report.md) | 91.73% | 0.8909 | 40 | 驗證與 Kaggle 回報均未改善；保留對照，未替換候選 |
| [FGSM C2 / epsilon 0.00416186](../docs/reports/FGSM_High_Report.md) | 90.92% | 0.9038 | 40 | 驗證與 Kaggle 均未超越既有首選；停止本輪 FGSM 搜尋，保留對照 |

## 模型與提交檔

所有路徑相對於 HW3 根目錄；每份 CSV 均核對 1,000 筆、`id,pred`、唯一 ID、原始順序與合法標籤。
模型檔、tokenizer 與設定逐檔保存 SHA-256；總表工具僅核對檔案，另行推論的證據列於下方稽核紀錄。

| 實驗 ID | checkpoint | CSV | CSV SHA-256（前 12 碼） |
|---|---|---|---|
| `baseline` | `saved_models/checkpoint` | [predictions.csv](../outputs/baseline/predictions.csv) | `be202424f6c7` |
| `weighted_sqrt` | `outputs/weighted_sqrt/checkpoint` | [predictions.csv](../outputs/weighted_sqrt/predictions.csv) | `7330b4e257aa` |
| `weighted_url` | `outputs/weighted_url/checkpoint` | [predictions.csv](../outputs/weighted_url/predictions.csv) | `38153fc07626` |
| `weighted_lr1e5` | `outputs/weighted_lr1e5/checkpoint` | [predictions.csv](../outputs/weighted_lr1e5/predictions.csv) | `b36f7e9e20a6` |
| `weighted_lr3e5` | `outputs/weighted_lr3e5/checkpoint` | [predictions.csv](../outputs/weighted_lr3e5/predictions.csv) | `b6bae1443d4c` |
| `ensemble_lr2_lr3_submission` | `outputs/weighted_sqrt/checkpoint`<br>`outputs/weighted_lr3e5/checkpoint` | [predictions.csv](../outputs/ensemble_lr2_lr3_submission/predictions.csv) | `dfda98c4126e` |
| `weighted_seed43` | `outputs/weighted_seed43/checkpoint` | [predictions.csv](../outputs/weighted_seed43/predictions.csv) | `d5673039e2bf` |
| `ensemble_seed42_seed43` | `outputs/weighted_sqrt/checkpoint`<br>`outputs/weighted_seed43/checkpoint` | [predictions.csv](../outputs/ensemble_seed42_seed43/predictions.csv) | `7695d07df372` |
| `weighted_emoji` | `outputs/weighted_emoji/checkpoint` | [predictions.csv](../outputs/weighted_emoji/predictions.csv) | `92fcdd3bc380` |
| `weighted_half_epoch_control` | `outputs/weighted_half_epoch_control/checkpoint` | [predictions.csv](../outputs/weighted_half_epoch_control/predictions.csv) | `7330b4e257aa` |
| `weighted_quarter` | `outputs/weighted_quarter/checkpoint` | [predictions.csv](../outputs/weighted_quarter/predictions.csv) | `800684403958` |
| `weighted_layer_lr` | `outputs/weighted_layer_lr/checkpoint` | [predictions.csv](../outputs/weighted_layer_lr/predictions.csv) | `47c39add40d1` |
| `weighted_fgsm_low` | `outputs/weighted_fgsm_low/checkpoint` | [predictions.csv](../outputs/weighted_fgsm_low/predictions.csv) | `fe1d1640ccfb` |
| `weighted_fgsm_high` | `outputs/weighted_fgsm_high/checkpoint` | [predictions.csv](../outputs/weighted_fgsm_high/predictions.csv) | `dd1fcd7f2396` |

## 推論稽核（不新增訓練候選）

- [precision_audit_20261115](../docs/reports/Precision_Audit_Report.md)：已完成；歷史驗證與 FP32 提交 CSV 重現通過。
  - `weighted_sqrt`：AMP／FP32 驗證 F1 92.6488%／92.6488%；驗證／benchmark 標籤差異 0／0 筆。
  - `weighted_lr3e5`：AMP／FP32 驗證 F1 92.1045%／92.1045%；驗證／benchmark 標籤差異 0／0 筆。

## 候選封裝演練

- [weighted_lr3e5 獨立封裝](../docs/reports/Package_Rehearsal_Report.md)：新 venv 安裝、乾淨解壓推論通過，1,000 筆與原 CSV 完全相同；此為演練，尚非最終交件。
- [演練 ZIP](../outputs/package_rehearsal_20261115/build/HW3_rehearsal_weighted_lr3e5.zip)；SHA-256 前 12 碼 `6e7476cda021`。

## 方法校準（不列入正式候選）

- [fgsm_20261115](../docs/reports/FGSM_Calibration_Report.md)：訓練資料尺度校準及兩組 smoke 完成；精度差異與限制見報告，尚無完整訓練成效。

## 比較與交件狀態

- 共用固定切分：8,100 train／900 validation。
- 驗證參考：`weighted_sqrt`；Kaggle 候選：`weighted_quarter`。
- 後續新方法共同控制組：`weighted_half_epoch_control`。
- 最終交件版本：`尚未指定`。
- `saved_models/checkpoint/` 仍是 `baseline`；不要直接把它當成最佳交件模型。
- `ensemble_lr2_lr3/` 是同組模型的初次分析；總表採用帶 CSV 的 `ensemble_lr2_lr3_submission/`，避免重複計數。
- `smoke/` 僅驗證管線，`emoji_audit/`、`emoji_short_audit/` 僅盤點，不列入正式模型比較。
- 相同 Kaggle 分數不代表 CSV 相同；不同筆數以目前 Kaggle 單模型候選為基準。

- 後續確認切分已預先固定：seed 142（8100／900）、seed 242（8100／900）；尚未在額外切分訓練。

後續優先順序見 [92% 計畫](Target_92_Plan.md)，操作入口見 [README](../README.md)。
