# HW3 實驗總表

由 `python tools/refresh_experiments.py` 產生；不要手動修改此檔。
更新來源為 [registry.json](../experiments/registry.json)，完整雜湊與指標見 [snapshot.json](../experiments/snapshot.json)。

驗證分數讀取本機 summary／checkpoint；Kaggle 分數來自歷史使用者回報，未登入平台核驗。`待回報` 不等於未提交。

| 實驗 | 驗證 Macro-F1 | Kaggle | 與 Kaggle 候選不同筆數 | 決策 |
|---|---:|---:|---:|---|
| [未加權基準](../docs/reports/Baseline_Report.md) | 91.57% | 0.8973 | 16 | 保留基準；預設 checkpoint 仍指向此版本 |
| [加權 2e-5 / seed 42](../docs/reports/Weighted_CE_Report.md) | 92.65% | 0.9049 | 15 | 驗證基準 |
| [URL 正規化](../docs/reports/URL_Normalization_Report.md) | 91.28% | 0.8977 | 32 | 未採用；保留對照 |
| [加權 1e-5](../docs/reports/Learning_Rate_Report.md) | 91.46% | 0.9012 | 26 | 未採用；保留對照 |
| [加權 3e-5](../docs/reports/Learning_Rate_3e5_Report.md) | 92.10% | 0.9052 | 0 | Kaggle 單模型候選 |
| [2e-5 / 3e-5 集成](../docs/reports/Ensemble_Error_Analysis_Report.md) | 92.58% | 0.9052 | 8 | 同分，優先單模型 |
| [加權 2e-5 / seed 43](../docs/reports/Seed43_Report.md) | 91.33% | 0.8878 | 30 | 未採用；保留對照 |
| [seed 42 / 43 集成](../docs/reports/Seed_Ensemble_Report.md) | 92.46% | 0.8983 | 17 | 未採用；保留對照 |
| [Emoji 簡短描述](../docs/reports/Emoji_Training_Report.md) | 91.68% | 待回報 | 25 | 驗證未改善；Kaggle 待回報 |

## 模型與提交檔

所有路徑相對於 HW3 根目錄；每份 CSV 均核對 1,000 筆、`id,pred`、唯一 ID、原始順序與合法標籤。
模型檔、tokenizer 與設定逐檔保存 SHA-256；本次為檔案核對，沒有重新執行推論。

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

## 比較與交件狀態

- 共用固定切分：8,100 train／900 validation。
- 驗證參考：`weighted_sqrt`；Kaggle 候選：`weighted_lr3e5`。
- 最終交件版本：`尚未指定`。
- `saved_models/checkpoint/` 仍是 `baseline`；不要直接把它當成最佳交件模型。
- `ensemble_lr2_lr3/` 是同組模型的初次分析；總表採用帶 CSV 的 `ensemble_lr2_lr3_submission/`，避免重複計數。
- `smoke/` 僅驗證管線，`emoji_audit/`、`emoji_short_audit/` 僅盤點，不列入正式模型比較。
- 相同 Kaggle 分數不代表 CSV 相同；不同筆數以目前 Kaggle 單模型候選為基準。

後續優先順序見 [92% 計畫](Target_92_Plan.md)，操作入口見 [README](../README.md)。
