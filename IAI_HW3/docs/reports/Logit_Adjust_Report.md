# 事後 logit 偏移校正

日期：2026/11/16。狀態：已完成（0 次訓練、2 模型 × 2 資料集合共 4 次 FP32 推論）；Kaggle 分數待使用者回報。

## 目的

檢驗推測「benchmark 類別比例與訓練不同，調整決策邊界可提升 Macro-F1」。此推測的前提（benchmark 比例不同）本身未經驗證；本次只驗證校正在固定 900 筆驗證集上是否有效，並產生可提交檔。

## 方法

- 模型：`outputs/weighted_sqrt/checkpoint`、`outputs/weighted_quarter/checkpoint`；沿用 `load_checkpoint`、`loader`、原前處理與 max_length 64，FP32（無 autocast）、batch 2（與精度稽核相同）。
- 公式：`adjusted = logits − τ · log(π)`；τ>0 偏向少數類，τ<0 偏向多數類。
- π：8,100 筆訓練切分（標籤順序 negative／neutral／positive）：5,728／1,201／1,171，即 0.70716／0.14827／0.14457。
- τ 網格（預先固定、寫死於程式）：−1.0 至 1.0，間隔 0.1，共 21 點。
- 選 τ 規則（只用驗證集）：驗證 Macro-F1 最高者；並列取 |τ| 最小，再並列取較小 τ。另記錄驗證 Macro-F1 在最佳值 0.2 個百分點內的 τ。
- 重現：`.venv/Scripts/python.exe -m tools.logit_adjust`（輸出目錄須為空）；`--check` 由已存 logits 重算並核對 summary、CSV、提交檔。
- 產物：`outputs/logit_adjust_20261116/`（`<run>/logits_{val,benchmark}.npz`、`summary.json`、`tau_table.csv`、`meta.json`、`submissions/`）。

## τ=0 重現驗收（已驗證）

| run | 驗證 Macro-F1（τ=0） | 既有紀錄 | benchmark 與原 predictions.csv 差異 |
|---|---:|---:|---:|
| `weighted_sqrt` | 0.9264883043107858 | 0.9264883043107858 | 0／1,000 |
| `weighted_quarter` | 0.9191192187866243 | 0.9191192187866243 | 0／1,000 |

兩者皆通過，之後才產生 CSV。

## τ–驗證 Macro-F1（%）

| τ | weighted_sqrt | weighted_quarter |
|---:|---:|---:|
| −1.0 | 91.60 | 91.34 |
| −0.5 | 92.48 | 91.88 |
| −0.3 | 92.48 | 91.75 |
| 0.0 | **92.65** | 91.91 |
| +0.1 | 92.65 | 91.91 |
| +0.2 | 92.50 | 92.08 |
| +0.3 | 92.50 | 92.24 |
| +0.4 | 92.50 | **92.41** |
| +0.6 | 92.50 | 92.41 |
| +0.7 | 92.36 | 92.14 |
| +1.0 | 92.36 | 92.01 |

- `weighted_sqrt`：最佳 τ=0（τ=0.0 與 0.1 並列，取 |τ| 最小），相對 τ=0 差 0。0.2 個百分點內的 τ：−0.5 至 +0.6（平坦）。校正沒有改善驗證。
- `weighted_quarter`：最佳 τ=+0.4（+0.5、+0.6 並列），92.4071%，相對 τ=0 為 +0.4952 個百分點（約 4 筆驗證樣本：neutral 召回多 2 筆、positive 召回多 1 筆、neutral→negative 少錯 2 筆，混淆矩陣 [[616,17,3],[15,118,1],[7,4,119]]）。0.2 個百分點內的 τ：+0.3 至 +0.6（窄）。
- 完整 21 點、各類 precision／recall／F1 與混淆矩陣見 `tau_table.csv`／`summary.json`。

## benchmark 預測分布（negative／neutral／positive）

| τ | weighted_sqrt | 與原 CSV 差異 | weighted_quarter | 與原 CSV 差異 |
|---:|---|---:|---|---:|
| −1.0 | 742／116／142 | 8 | 737／126／137 | 6 |
| −0.5 | 736／122／142 | 2 | 733／129／138 | 2 |
| 0.0 | 734／124／142 | 0 | 731／131／138 | 0 |
| +0.4 | 733／125／142 | 1 | 728／134／138 | 3 |
| +0.6 | 732／126／142 | 2 | 728／134／138 | 3 |
| +1.0 | 725／133／142 | 9 | 721／140／139 | 10 |

預測分布的變動很小：最佳 τ 下 weighted_quarter 只改 3 筆，且皆為 negative 轉為 neutral（已逐筆核對）。訓練比例為 70.7%／14.8%／14.5%，benchmark 預測比例約 73%／13%／14%，並未顯示出明顯的比例偏移（這是預測分布，不是真實標籤比例，benchmark 無標籤，無法驗證）。

## 產生的 CSV

| 檔案 | SHA-256 前 12 碼 | 與原檔差異 | 與 weighted_quarter 原 CSV 差異 |
|---|---|---:|---:|
| `outputs/logit_adjust_20261116/submissions/weighted_quarter_tau+0.4.csv` | dd6bfb49799a | 3 | 3 |

（`weighted_sqrt` 最佳 τ=0 與其原 predictions.csv 完全相同，不產生重複檔。）1,000 筆，`id,pred`，順序與 benchmark.csv 相同，標籤皆合法。

## 限制與狀態

- 已驗證：τ=0 重現、網格、公式、CSV 格式、`--check` 重算一致、單元測試與 `refresh_experiments.py --check` 通過。
- 驗證集只有 900 筆（neutral 134、positive 130）；weighted_quarter 的 +0.5 個百分點只相當於約 4 筆樣本，在雜訊範圍內，不能視為確定改善。weighted_sqrt 在 τ=0 附近已最佳，且曲線平坦。
- 最佳 τ 在驗證集上選取，驗證集同時也是選模型依據（checkpoint 選擇），存在輕微樂觀偏誤。
- benchmark 類別比例與訓練不同僅為推測，未驗證；校正對 Kaggle 分數的影響未知，且預期很小（最多 3 筆預測改變）。Kaggle 分數待使用者回報。
- 校正只改決策邊界，未改變模型、前處理或訓練；未提交平台。
