# 五折交叉驗證：全資料文字 MLM 續訓 30 epochs 後微調（cv5_sqrt_ep5_mlmall30）

日期：2026/11/19。狀態：已完成（1 次共用 MLM 續訓＋5 次微調＋1 次 OOF／集成 FP32 推論＋1 次配對比較）。**判定：Kaggle 0.9142 < 0.9152，主版本維持 `cv5_sqrt_ep5_mlm10`，不延伸此線。** 對應 [92% 計畫](../Target_92_Plan.md)「方向 2 預先登記」（訓練前寫下，2026/11/18 使用者確認）。

## 目的與設定差異

- 目的：檢驗「MLM 續訓看全部 9,000 筆文字、訓練更久」能否使 Kaggle 高於 mlm10 的 0.9152。以 Kaggle 為主判斷，OOF 只作參考。
- 對照：`cv5_sqrt_ep5_mlm10`（[報告](CV5_MLM10_Report.md)）。與其差異（兩個因素同時改，無法分辨哪個有效）：
  1. 續訓語料為 `dataset/dataset.csv` 全部 9,000 筆文字（只用文字，不用標籤、不碰 benchmark）；mlm10 為各折訓練子集約 7,200 筆。
  2. 續訓 30 epochs（mlm10 為 10）；只續訓 1 次，5 折共用同一 checkpoint（mlm10 每折各自續訓）。
- 其餘相同：lr 5e-5、遮罩 15%（80/10/10）、batch 2、累積 8、seed 42、儲存最後 epoch；微調 sqrt-inverse、5 epochs、切分 `outputs/cv5_seed42`。
- 重現：`bash tools/run_cv5_mlm_shared.sh cv5_sqrt_ep5_mlmall30 sqrt-inverse 5 30`；log `outputs/cv5_sqrt_ep5_mlmall30_train.log`。程式新增 `pretrain-mlm --mlm-corpus all-dataset`、`train --allow-mlm-val-overlap`。

## 非嚴格 OOF（重要）

續訓語料含各折驗證文字（不含標籤），各折 `run_config` 的 `init_from.validation_overlap.strict_oof` 為 `false`。因此下列 OOF 與對 mlm10 的比較**偏向 mlmall30（偏樂觀）**，只作參考，不作判斷依據。

## 訓練紀錄

- MLM 續訓（`outputs/cv5_sqrt_ep5_mlmall30_mlm/shared/`）：訓練 MLM loss epoch 1 為 3.264、epoch 30 為 1.254；無驗證 loss（無留出文字）；曲線 [curves.png](../../outputs/cv5_sqrt_ep5_mlmall30_mlm/shared/curves.png)。2026/11/18 23:55 起至 11/19 03:28，共 12,743 秒（每 epoch 約 450 秒），optimizer 更新 16,882／嘗試 16,890，empty_windows 0。
- 微調各折（`outputs/cv5_sqrt_ep5_mlmall30/fold{0..4}/`，各有 `curves.png`）：

| fold | 最佳 epoch | 最佳驗證 Macro-F1（訓練時 AMP，%） | 微調秒數 |
|---:|---:|---:|---:|
| 0 | 4 | 90.08 | 1,948 |
| 1 | 4 | 89.97 | 1,889 |
| 2 | 5 | 92.43 | 1,952 |
| 3 | 2 | 89.88 | 1,897 |
| 4 | 4 | 88.61 | 1,912 |

## OOF 結果（非嚴格）

數字取自 `outputs/cv5_sqrt_ep5_mlmall30_oof/oof_metrics.json`。

| 指標 | mlm10 | mlmall30 |
|---|---:|---:|
| OOF Macro-F1（9,000 筆） | 89.63% | **90.21%** |
| negative／neutral／positive F1 | 96.15／81.66／91.09 | 96.43／82.78／91.44 |
| fold 平均 ± 樣本標準差 | 89.64 ± 1.58 | 90.21 ± 1.38 |
| 原 900 驗證 id 在 OOF 中 | 90.89% | 90.93% |

mlmall30 OOF 混淆矩陣（列真實，欄預測，negative／neutral／positive）：`[[6155,145,64],[195,1079,61],[52,48,1201]]`。

## 對 mlm10 的配對比較（偏向 mlmall30）

`outputs/cv5_compare_mlm10_vs_mlmall30.json`（a = mlm10，b = mlmall30；seed 20261116、2,000 次 bootstrap）：Δ OOF Macro-F1 +0.58 個百分點，95% 區間 **[−0.05, +1.16]**，Δ≤0 比例 3.85%；兩邊預測不同 326 筆：mlm10 對 135、mlmall30 對 174。逐 fold（FP32 OOF，mlmall30 勝 5/5）：

| fold | mlm10 | mlmall30 | Δ（百分點） |
|---:|---:|---:|---:|
| 0 | 89.31 | 90.08 | +0.77 |
| 1 | 89.60 | 89.97 | +0.37 |
| 2 | 92.28 | 92.43 | +0.15 |
| 3 | 88.92 | 89.96 | +1.04 |
| 4 | 88.08 | 88.61 | +0.53 |

此比較因續訓看過驗證文字而偏向 mlmall30，**不作判斷**；無法區分正向差異來自洩漏或真實效果。

## Kaggle 與判定

- 集成 CSV：`outputs/cv5_sqrt_ep5_mlmall30_oof/submissions/cv5_folds012_mean.csv`（fold 0、1、2 等權 FP32 softmax 平均），SHA-256 `80f97a038780e90d0df7123325bee8284fd27eb2f4e048cbfef2e1977622797c`。
- Kaggle **0.9142**（使用者 2026/11/19 回報，未經平台核驗）。
- 預先登記規則：≥0.92 主要候選；0.9152～0.92 沿此線延伸；**< 0.9152 主版本維持 mlm10**。0.9142 < 0.9152 → **主版本維持 `cv5_sqrt_ep5_mlm10`（0.9152）**；mlmall30 不成為主版本，不延伸此線（更長續訓、更高遮罩率不做）。
- 下一步：待與使用者討論後續方向。

## 限制

- 差距 0.10 個百分點，遠小於單次 Kaggle 比較雜訊約 ±1 個百分點（只換 seed 曾差 1.7 個百分點）；只能說「全文＋30 epochs 未見改善」，不能說變差。
- 一次改了語料範圍與 epoch 數兩個因素，無法分辨各自效果。
- 單 seed、單次；OOF 非嚴格，且每折以驗證折挑最佳 epoch 有選模樂觀偏誤。
- 合規：只用課程提供資料文字；講義 p.9「base model」續訓風險使用者已於 2026/11/17 接受（見 [MLM10 報告](CV5_MLM10_Report.md)合規說明）。

## 產物

- MLM 續訓：`outputs/cv5_sqrt_ep5_mlmall30_mlm/shared/`；權重 `saved_models/cv5_sqrt_ep5_mlmall30_mlm/shared/`
- 微調：`outputs/cv5_sqrt_ep5_mlmall30/fold{0..4}/`；checkpoint `saved_models/cv5_sqrt_ep5_mlmall30/fold{0..4}/`
- OOF：`outputs/cv5_sqrt_ep5_mlmall30_oof/`；比較：`outputs/cv5_compare_mlm10_vs_mlmall30.json`
- 登錄：[registry](../../experiments/registry.json) `cross_validations`（`cv5_sqrt_ep5_mlmall30`）
