# 固定切分 8,100／900：MLM 續訓 10 epochs 後微調，3 seed（full8100_mlm10）

日期：2026/11/19。狀態：訓練、集成與 Kaggle 4/4 已完成；**判定：主版本維持 `cv5_sqrt_ep5_mlm10`（0.9152），此線停止；已登錄 [registry](../../experiments/registry.json)（3 個單模型 run＋`checkpoint_ensembles` 1 筆）並由 `refresh_experiments.py` 稽核，結果見[總表](../Experiment_Results.md)。** 對應 [92% 計畫](../Target_92_Plan.md)「方向 3 預先登記：full8100_mlm10」。

## 目的與設定差異

- 目的：檢驗最終交件型態（固定 8,100 筆訓練文字做 MLM 續訓，再微調）在 Kaggle 上能否超過 `cv5_sqrt_ep5_mlm10` 的 0.9152。以 Kaggle 為主判斷，900 驗證只用來挑 epoch 與記錄。
- 對照：`cv5_sqrt_ep5_mlm10`（[報告](CV5_MLM10_Report.md)，五折約 7,200 筆／折）、`weighted_quarter`、`weighted_sqrt`（單模型，非 MLM，8,100 筆）。
- 差異：訓練資料由約 7,200 增為 8,100，並用 3 個 seed（42、43、44）訓練後做等權集成；MLM 設定同 mlm10（10 epochs、lr 5e-5、遮罩 15%、batch 2、累積 8）。
- 切分：`outputs/eda/split.json`（8,100／900）。每個 seed：MLM 續訓（只用 8,100 筆訓練文字）→ sqrt-inverse 微調 5 epochs，以 900 驗證挑最佳 epoch。
- 嚴格性：三個 seed 的 `run_config.init_from.validation_overlap` 均為 `allowed: false`、`n_overlap_ids: 0`、`strict_oof: true`（`mlm_run_config.n_texts` 8100、`mlm_corpus: split-train`）。
- 重現（見下方「執行紀錄」的預設差異說明）：`bash tools/run_full_mlm.sh full8100_mlm10 10 5 42`（試跑），之後 `... 10 5 42 43 44`；log `outputs/full8100_mlm10_train.log`（2026/11/19 12:46:46 起，17:32:17 集成完成，exit 0）。試跑的 1 模型集成 `outputs/full8100_mlm10_ensemble_seed42/` 之 model0 CSV 與 3 模型目錄的 model0 SHA-256 完全相同（可重現）。

## 執行紀錄

- 本次訓練使用的程式碼為 17:32 前的版本；17:34 起 `main.py`／`src/` 有另一工作的改動（新增 `--eval-batch-size` 與計時），不影響本實驗產物。
- **預設差異（重現須知）：** 本實驗在 commit `8094b18` 之前的預設下執行（`run_config.json` 記錄 `batch_size` 2、`accumulation_steps` 8、`gradient_checkpointing: true`；評估 batch 沿用 batch size 2，`run_config` 沒有 eval batch 欄位，由當時程式行為推得）。`8094b18` 起 `train` 評估 batch 預設 64、`train` 與 `pretrain-mlm` 的 gradient checkpointing 預設關閉，因此現行 `tools/run_full_mlm.sh` 直接執行不再與本實驗同設定。依 [Speed_Check_Report](Speed_Check_Report.md)（1 epoch、seed 42 實測）：微調的評估 batch 64 與關閉 checkpointing 皆為位元相同（權重 SHA-256 相同、最大權重差 0、驗證混淆矩陣相同），只有 val loss 有約 1e-5 的浮點累加差；MLM 關閉 checkpointing 的 train MLM loss 與舊設定完全相同，但 MLM 評估 batch 若改為 64，驗證遮罩會改變，val MLM loss 不可逐位比較（且 peak reserved 4,166 MiB 超標；現行 `pretrain-mlm` 評估 batch 仍沿用 `--batch-size`，未採用 64）。上述等價性只針對 1 epoch 短測；10 epoch MLM 加 5 epoch 微調全程是否逐位相同**未驗證**（推測相同）。若要最保險地重現，建議明確加 `--gradient-checkpointing`（並以 `--eval-batch-size 2` 指定評估 batch），且 `main.py`／`src/` 已與 `training_code_sha256` 記錄的版本不同，不保證逐位元一致。
- 16:15 前後約 15 分鐘訓練速度降到約一半（GPU 使用率約 18%，電源、溫度、VRAM 正常，原因未確認），之後恢復；不影響結果，只影響 seed 44 的 MLM 耗時。

## 訓練紀錄

數字取自各 seed 的 `outputs/full8100_mlm10_mlm/seed*/summary.json`、`outputs/full8100_mlm10/seed*/summary.json`、`history.json`。

| seed | MLM 最終驗證 loss（epoch 1） | MLM 秒數 | 微調秒數 | 最佳 epoch | 900 驗證 Macro-F1（%） | 各 epoch（1–5） |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 2.157（2.597） | 3,532 | 2,109 | 3 | 91.54 | 89.97／88.62／91.54／91.35／91.50 |
| 43 | 2.155（2.578） | 3,550 | 2,002 | 3 | 91.96 | 87.13／91.44／91.96／91.78／91.71 |
| 44 | 2.060（2.473） | 3,565 | 1,771 | 4 | 91.41 | 89.62／91.03／90.97／91.41／91.34 |

MLM optimizer 更新：seed 42、43 為 5,066／5,070，seed 44 為 5,065／5,070；empty_windows 均為 0。曲線：`outputs/full8100_mlm10_mlm/seed*/curves.png`、`outputs/full8100_mlm10/seed*/curves.png`。

## 900 驗證

- 單模型：91.54／91.96／91.41%（上表）。
- 3 模型等權 FP32 softmax 集成：Macro-F1 **91.93%**（0.919267）；三類 F1 negative 96.47、neutral 85.61、positive 93.70；混淆矩陣（列真實，欄預測，negative／neutral／positive）`[[615,17,4],[17,116,1],[7,4,119]]`。
- 資料來源：`outputs/full8100_mlm10_ensemble_seed42-43-44/summary.json`。

## Kaggle 與判定

上傳檔位於 `outputs/full8100_mlm10_ensemble_seed42-43-44/submissions/`。Kaggle 分數為使用者回報（seed 42 與集成 2026/11/19、seed 43／44 2026/11/20，後兩者為 Kaggle 截圖），未經平台核驗。

| CSV | 內容 | SHA-256（前 12 碼） | Kaggle |
|---|---|---|---:|
| `model0_benchmark_fp32.csv` | seed 42 單模型 | `4f6b42a0f38f` | **0.9032** |
| `model1_benchmark_fp32.csv` | seed 43 單模型 | `8cf85729e974` | **0.9080**（使用者 2026/11/20 回報） |
| `model2_benchmark_fp32.csv` | seed 44 單模型 | `689c684a6d40` | **0.9021**（使用者 2026/11/20 回報） |
| `ensemble_mean_benchmark_fp32.csv` | 3 模型等權集成 | `88d7616bddae` | **0.9079** |

- 預先登記規則：任一候選 ≥ 0.92 → 取 Kaggle 最高者為主要候選；最高者介於 0.9152 與 0.92 → 成為新主版本但不再延伸；全部 ≤ 0.9152 → 主版本維持 mlm10。
- **判定定案：** 4/4 已回報（0.9032／0.9080／0.9021、集成 0.9079），全部 ≤ 0.9152 → 依預先登記，主版本維持 `cv5_sqrt_ep5_mlm10`（0.9152），此線停止，不追加上傳。
- 觀察（推測，不下因果結論）：
  - 最高為 seed 43 單模型 0.9080；3 模型集成 0.9079 與最好的單模型幾乎相同，本次集成沒有帶來可見增益。
  - 三個單模型範圍 0.9021–0.9080（差 0.59 個百分點），只換 seed 即有此差距。
  - 多 900 筆訓練加 MLM 的單模型（0.9032／0.9080／0.9021）未明顯高於非 MLM 的 `weighted_sqrt` 單模型（0.9049）；集成 0.9079 也低於 mlm10 五折集成 0.9152；差距大致在單次 Kaggle 約 ±1 個百分點雜訊內。
  - 900 驗證 seed 43 最高（91.96%）且 Kaggle 也最高（0.9080），但僅 3 個點，不能據此說 900 驗證可預測 Kaggle（推測）。
- 參考對照（使用者回報）：mlm10 五折 fold0/1/2 集成 0.9152（主版本）、mlmall30 0.9142、`weighted_quarter` 0.9122（保險，900 驗證 91.91%）、`weighted_sqrt` 0.9049（900 驗證 92.65%）。

## 限制

- 單次 Kaggle 雜訊約 ±1 個百分點；900 驗證與 Kaggle 排序曾不一致（此實驗 900 驗證 91.4–92.0%，Kaggle 單模型 0.9032）。
- 各 seed 以 900 驗證挑最佳 epoch，驗證分數有選模樂觀偏誤；本實驗無 OOF，與 mlm10 的 OOF 不可直接比較。
- 從 4 個上傳挑最高是以 Kaggle 選擇（使用者已同意 Kaggle 即成績），分數本身不代表泛化提升。
- 合規：只用課程提供資料的訓練文字，不用 benchmark、不用驗證文字續訓；講義 p.9「base model」續訓風險使用者已於 2026/11/17 接受；集成 3 個模型符合上限。

## 待辦

1. 已完成：`tools/refresh_experiments.py` 支援 single run 的 `predictions` 路徑與 `init_from` 稽核，並新增頂層 `checkpoint_ensembles`；已登錄 registry，`refresh_experiments.py` 與 `--check` 通過。（無剩餘待辦）

## 產物

- MLM 續訓：`outputs/full8100_mlm10_mlm/seed{42,43,44}/`；權重 `saved_models/full8100_mlm10_mlm/seed{42,43,44}/`
- 微調：`outputs/full8100_mlm10/seed{42,43,44}/`；checkpoint `saved_models/full8100_mlm10/seed{42,43,44}/`
- 集成與提交 CSV：`outputs/full8100_mlm10_ensemble_seed42-43-44/`（`summary.json`、`submissions/`）；試跑 `outputs/full8100_mlm10_ensemble_seed42/`
- 登錄：尚未登錄（見待辦）
