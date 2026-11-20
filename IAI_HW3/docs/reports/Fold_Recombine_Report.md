# mlm10 fold 重組：5 折等權集成（離線，不訓練）

日期：2026/11/21。狀態：已完成；**判定：Kaggle 0.9137 低於 folds012 的 0.9152，主版本維持 `cv5_sqrt_ep5_mlm10` folds012，本方向停止。** 對應 [92% 計畫](../Target_92_Plan.md)「方向 4 預先登記」。依 AGENTS.md 第 5 節，後處理分析不寫入 registry，只在計畫調整紀錄登錄。

## 目的與方法

- 目的：檢驗把 mlm10 五折 CV 的 5 個 fold 模型全部等權集成（5 折），是否比已上傳的 3 折（folds012，Kaggle 0.9152）更穩定、分數更高（假設，未驗證）。
- 方法：只用既有 benchmark logits `outputs/cv5_sqrt_ep5_mlm10_oof/logits_benchmark_fold{0..4}.npz`，離線重組，不訓練、不重新推論；與 `tools/cv_oof.py` 相同（FP32 softmax 等權平均、argmax）。工具 `tools/recombine_folds.py`，測試 `tests/test_recombine_folds.py`。
- 主要候選唯一一份：`outputs/cv5_sqrt_ep5_mlm10_recombine/cv5_folds01234_mean.csv`；Kaggle 上傳預先固定 1 次。另產生 10 個 3 折、5 個 4 折組合 CSV（`diagnostic/`）僅作診斷，不上傳。
- 為何不從組合中挑：OOF 每筆只有一個 fold 的預測，子集集成沒有無洩漏的標籤評估；以 Kaggle 在組合中挑等於對測試集調參。
- 集成上限：本方向經使用者同意，將「集成上限 3」放寬為 5 折（僅限此 CV 的 5 個 fold 模型）。

## 重現驗證

- 工具以 folds 0 1 2 重現的 `check_folds012.csv`，SHA-256 與既有 `cv5_folds012_mean.csv` 一致：`ad1124f0a7cca3f9722c0627c652ec8f3084a9304c807c7cbc1ffadb5a890755`（已驗證）。
- `tests/test_recombine_folds.py` 6 個測試 OK。

## Kaggle 結果

Kaggle 分數為使用者回報（2026/11/21），未經平台核驗。

| CSV | 折 | SHA-256 | Kaggle |
|---|---|---|---:|
| `cv5_folds012_mean.csv`（主版本） | 0 1 2 | `ad1124f0…0755` | 0.9152 |
| `cv5_folds01234_mean.csv` | 0 1 2 3 4 | `6db6ece2f35086221c019baa4e3ca50a3dec97a429c228f2fc5294d9bbf7f7a7` | **0.9137**（Δ −0.0015） |

## 診斷（資料：`outputs/cv5_sqrt_ep5_mlm10_recombine/diagnostic_summary.json`）

| 項目 | 與 folds012 不一致筆數（共 1,000） | 預測分布（neg／neu／pos） |
|---|---:|---|
| folds012（基準） | 0 | 722／131／147 |
| 5 折 | 14 | 730／126／144 |
| 4 折（5 個組合） | 12–20 | neutral 124–128 |
| 3 折（其餘 9 個組合） | 12–25 | neutral 124–131 |
| 單折（5 個，對 folds012） | 21–36 | neutral 122–142 |
| 單折兩兩（10 對） | 30–53（兩兩之間） | — |

5 折與 folds012 的 14 筆差異：neutral→negative 7、positive→negative 3、negative→neutral 2、neutral→positive 1、positive→neutral 1。

## 判定

預先登記：Kaggle > 0.9152 才成為新主版本；≤ 0.9152 → 維持 folds012，本方向停止，不再上傳其他組合。結果 0.9137 ≤ 0.9152，**主版本維持 folds012（0.9152），停止**。

## 限制與解讀（推測）

- 只有 14 筆預測改變、Kaggle −0.0015，與同設定不同 seed 間的 Kaggle 浮動（約 0.6 個百分點）同量級，無法區分是 5 折較差或雜訊。
- 變動多往 negative 偏、neutral 預測減少（730／126／144 對 722／131／147），可能與 neutral 被少數類吃掉有關，未驗證。
- 該 CSV 不是 OOF 評估，沒有標籤可算驗證指標；結論只依單次 Kaggle。
- 合規：沿用 mlm10 的條件（只用訓練文字續訓；MLM base model 疑慮使用者已於 2026/11/17 接受）；集成 5 個 fold 模型超過原 3 個上限，僅為診斷／實驗用途並經使用者同意，**不作為交件集成**（交件仍受最多 3 模型限制）。

## 重現

```
.venv/Scripts/python.exe -m tools.recombine_folds --oof-dir outputs/cv5_sqrt_ep5_mlm10_oof --folds 0 1 2 3 4 --output outputs/cv5_sqrt_ep5_mlm10_recombine/cv5_folds01234_mean.csv
```
（診斷表由 3、4 折與單折組合以同一函式算出，彙整於 `diagnostic_summary.json`；臨時彙整腳本已移除。）
