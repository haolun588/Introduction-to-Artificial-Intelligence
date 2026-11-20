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
| [固定切分 MLM 續訓 10 epochs / seed 42](../docs/reports/Full8100_MLM10_Report.md) | 91.54% | 0.9032 | 32 | 非候選：Kaggle 低於 cv5_sqrt_ep5_mlm10（0.9152）；CSV 為 FP32 推論，此線停止，僅保留記錄 |
| [固定切分 MLM 續訓 10 epochs / seed 43](../docs/reports/Full8100_MLM10_Report.md) | 91.96% | 0.9080 | 43 | 非候選：Kaggle 低於 cv5_sqrt_ep5_mlm10（0.9152）；CSV 為 FP32 推論，此線停止，僅保留記錄 |
| [固定切分 MLM 續訓 10 epochs / seed 44](../docs/reports/Full8100_MLM10_Report.md) | 91.41% | 0.9021 | 38 | 非候選：Kaggle 低於 cv5_sqrt_ep5_mlm10（0.9152）；CSV 為 FP32 推論，此線停止，僅保留記錄 |

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
| `full8100_mlm10_seed42` | `saved_models/full8100_mlm10/seed42` | [predictions.csv](../outputs/full8100_mlm10_ensemble_seed42-43-44/submissions/model0_benchmark_fp32.csv) | `4f6b42a0f38f` |
| `full8100_mlm10_seed43` | `saved_models/full8100_mlm10/seed43` | [predictions.csv](../outputs/full8100_mlm10_ensemble_seed42-43-44/submissions/model1_benchmark_fp32.csv) | `8cf85729e974` |
| `full8100_mlm10_seed44` | `saved_models/full8100_mlm10/seed44` | [predictions.csv](../outputs/full8100_mlm10_ensemble_seed42-43-44/submissions/model2_benchmark_fp32.csv) | `689c684a6d40` |

## 推論稽核（不新增訓練候選）

- [precision_audit_20261115](../docs/reports/Precision_Audit_Report.md)：已完成；歷史驗證與 FP32 提交 CSV 重現通過。
  - `weighted_sqrt`：AMP／FP32 驗證 F1 92.6488%／92.6488%；驗證／benchmark 標籤差異 0／0 筆。
  - `weighted_lr3e5`：AMP／FP32 驗證 F1 92.1045%／92.1045%；驗證／benchmark 標籤差異 0／0 筆。

## 候選封裝演練

- [weighted_lr3e5 獨立封裝](../docs/reports/Package_Rehearsal_Report.md)：新 venv 安裝、乾淨解壓推論通過，1,000 筆與原 CSV 完全相同；此為演練，尚非最終交件。
- [演練 ZIP](../outputs/package_rehearsal_20261115/build/HW3_rehearsal_weighted_lr3e5.zip)；SHA-256 前 12 碼 `6e7476cda021`。

## 檢查點等權集成（固定切分；不列入上方單模型比較）

| 集成 | 成員 | 驗證 Macro-F1 | Kaggle | 與 Kaggle 候選不同筆數 | CSV | CSV SHA-256（前 12 碼） |
|---|---|---:|---:|---:|---|---|
| [固定切分 MLM 續訓 10 epochs / 3 seed 等權集成](../docs/reports/Full8100_MLM10_Report.md) | `full8100_mlm10_seed42`<br>`full8100_mlm10_seed43`<br>`full8100_mlm10_seed44` | 91.93% | 0.9079 | 34 | [ensemble_mean_benchmark_fp32.csv](../outputs/full8100_mlm10_ensemble_seed42-43-44/submissions/ensemble_mean_benchmark_fp32.csv) | `88d7616bddae` |

## 方法校準（不列入正式候選）

- [fgsm_20261115](../docs/reports/FGSM_Calibration_Report.md)：訓練資料尺度校準及兩組 smoke 完成；精度差異與限制見報告，尚無完整訓練成效。

## 交叉驗證（OOF；不列入上方單模型比較）

- [五折 CV weighted_sqrt](../docs/reports/CV5_Sqrt_Report.md)：5 折、9,000 筆 OOF；OOF Macro-F1 **89.03%**，fold 平均 89.00% ± 1.98 個百分點（樣本標準差）。
- 三類 F1：negative 96.01%／neutral 80.78%／positive 90.29%。
- 集成 CSV：`outputs/cv5_sqrt_oof/submissions/cv5_folds012_mean.csv`（fold 0、1、2 等權；SHA-256 前 12 碼 `7037e623cf00`）；提交狀態 `not_submitted`，Kaggle 待回報。

## 交叉驗證（OOF；不列入上方單模型比較）

- [五折 CV weighted_quarter](../docs/reports/CV5_Quarter_Report.md)：5 折、9,000 筆 OOF；OOF Macro-F1 **88.80%**，fold 平均 88.78% ± 1.70 個百分點（樣本標準差）。
- 三類 F1：negative 96.00%／neutral 80.15%／positive 90.23%。
- 集成 CSV：`outputs/cv5_quarter_oof/submissions/cv5_folds012_mean.csv`（fold 0、1、2 等權；SHA-256 前 12 碼 `a122c68fcaa8`）；提交狀態 `not_submitted`，Kaggle 待回報。
- 配對比較（`cv5_quarter` − `cv5_sqrt`，[結果](../outputs/cv5_compare_sqrt_vs_quarter.json)）：Δ OOF Macro-F1 -0.23 個百分點，配對 bootstrap 95% 區間 [-0.71, +0.21]（2,000 次，seed 20261116），Δ≤0 比例 84.4%；cv5_quarter 勝 2/5 折。

## 交叉驗證（OOF；不列入上方單模型比較）

- [五折 CV weighted_sqrt 5 epochs](../docs/reports/CV5_Sqrt_Ep5_Report.md)：5 折、9,000 筆 OOF；OOF Macro-F1 **89.22%**，fold 平均 89.22% ± 1.51 個百分點（樣本標準差）。
- 三類 F1：negative 96.02%／neutral 80.75%／positive 90.90%。
- 集成 CSV：`outputs/cv5_sqrt_ep5_oof/submissions/cv5_folds012_mean.csv`（fold 0、1、2 等權；SHA-256 前 12 碼 `351963b9048f`）；提交狀態 `user_reported`，Kaggle 0.8971。
- 配對比較（`cv5_sqrt_ep5` − `cv5_sqrt`，[結果](../outputs/cv5_compare_sqrt_vs_sqrt_ep5.json)）：Δ OOF Macro-F1 +0.19 個百分點，配對 bootstrap 95% 區間 [-0.38, +0.76]（2,000 次，seed 20261116），Δ≤0 比例 26.9%；cv5_sqrt_ep5 勝 3/5 折。

## 交叉驗證（OOF；不列入上方單模型比較）

- [五折 CV weighted_sqrt 5 epochs 排除 473 筆高信心不一致樣本](../docs/reports/CV5_Drop90_Report.md)：5 折、9,000 筆 OOF；OOF Macro-F1 **89.34%**，fold 平均 89.34% ± 1.66 個百分點（樣本標準差）。
- 三類 F1：negative 96.08%／neutral 81.10%／positive 90.85%。
- 集成 CSV：`outputs/cv5_sqrt_ep5_drop90_oof/submissions/cv5_folds012_mean.csv`（fold 0、1、2 等權；SHA-256 前 12 碼 `a112b5c89e62`）；提交狀態 `not_submitted`，Kaggle 待回報。
- 配對比較（`cv5_sqrt_ep5_drop90` − `cv5_sqrt_ep5`，[結果](../outputs/cv5_compare_ep5_vs_drop90.json)）：Δ OOF Macro-F1 +0.12 個百分點，配對 bootstrap 95% 區間 [-0.49, +0.76]（2,000 次，seed 20261116），Δ≤0 比例 35.1%；cv5_sqrt_ep5_drop90 勝 3/5 折。

## 交叉驗證（OOF；不列入上方單模型比較）

- [五折 CV weighted_sqrt 5 epochs 微調自各折 MLM 續訓 10 epochs 權重（僅用該折訓練子集文字）](../docs/reports/CV5_MLM10_Report.md)：5 折、9,000 筆 OOF；OOF Macro-F1 **89.63%**，fold 平均 89.64% ± 1.58 個百分點（樣本標準差）。
- 三類 F1：negative 96.15%／neutral 81.66%／positive 91.09%。
- 集成 CSV：`outputs/cv5_sqrt_ep5_mlm10_oof/submissions/cv5_folds012_mean.csv`（fold 0、1、2 等權；SHA-256 前 12 碼 `ad1124f0a7cc`）；提交狀態 `user_reported`，Kaggle 0.9152。
- 配對比較（`cv5_sqrt_ep5_mlm10` − `cv5_sqrt_ep5`，[結果](../outputs/cv5_compare_ep5_vs_mlm10.json)）：Δ OOF Macro-F1 +0.41 個百分點，配對 bootstrap 95% 區間 [-0.21, +1.06]（2,000 次，seed 20261116），Δ≤0 比例 10.3%；cv5_sqrt_ep5_mlm10 勝 3/5 折。

## 交叉驗證（OOF；不列入上方單模型比較）

- [五折 CV weighted_sqrt 5 epochs 微調自共用 MLM 續訓 30 epochs 權重（續訓語料為全部 9,000 筆文字，非嚴格 OOF）](../docs/reports/CV5_MLMAll30_Report.md)：5 折、9,000 筆 OOF；OOF Macro-F1 **90.21%**，fold 平均 90.21% ± 1.38 個百分點（樣本標準差）。
- 三類 F1：negative 96.43%／neutral 82.78%／positive 91.44%。
- 集成 CSV：`outputs/cv5_sqrt_ep5_mlmall30_oof/submissions/cv5_folds012_mean.csv`（fold 0、1、2 等權；SHA-256 前 12 碼 `80f97a038780`）；提交狀態 `user_reported`，Kaggle 0.9142。
- 配對比較（`cv5_sqrt_ep5_mlmall30` − `cv5_sqrt_ep5_mlm10`，[結果](../outputs/cv5_compare_mlm10_vs_mlmall30.json)）：Δ OOF Macro-F1 +0.58 個百分點，配對 bootstrap 95% 區間 [-0.05, +1.16]（2,000 次，seed 20261116），Δ≤0 比例 3.9%；cv5_sqrt_ep5_mlmall30 勝 5/5 折。

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
