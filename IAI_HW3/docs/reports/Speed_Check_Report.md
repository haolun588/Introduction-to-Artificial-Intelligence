# 訓練加速等價性實測（speed_check）

2026/11/19。量測加速設定的速度效益與數值等價性，**不是為了提高分數，不產生交件候選、不產生 CSV**。預先登記見 [Target_92_Plan 調整紀錄](../Target_92_Plan.md)。結論：微調可安全採用 `--eval-batch-size 64 --no-gradient-checkpointing`（位元相同，1 epoch 約 467 秒降到 203 秒，約 2.3 倍）；MLM 可採用 `--no-gradient-checkpointing`，但評估 batch 不能用 64。

假設與對照：每組只比上一組多一個變因（A 現行 → B 評估 batch 64 → C 關閉 gradient checkpointing → D batch 8／累積 2 → E batch 16／累積 1）。共同設定：seed 42、`--split outputs/eda/split.json`、`--class-weighting sqrt-inverse`、`--epochs 1`、不用 `--init-from`；輸出 `outputs/speed_check/<組別>`、checkpoint `saved_models/speed_check/<組別>`。預算上限 60 分鐘 GPU 時間，任一組 OOM 或 peak reserved ≥ 3,900 MiB 即放棄該組及更大 batch 組。

輔助腳本 `outputs/speed_check/_run.sh`（含時間戳記的執行包裝）為上一輪留下，保留不動。

## 微調結果（1 epoch，seed 42）

| 組別 | 設定差異 | 總秒數 | 訓練段 | 評估段 | 驗證 Macro-F1 | peak reserved（MB） |
|---|---|---:|---:|---:|---:|---:|
| A | 現行（batch 2、累積 8、checkpointing 開） | 466.7 | 305.3 | 142.9 | 0.89005 | 2036 |
| B | A + `--eval-batch-size 64` | 381.6 | 369.2 | 10.0 | 0.89005 | 2036 |
| C | B + `--no-gradient-checkpointing` | 203.0 | 189.8 | 10.8 | 0.89005 | 2074 |
| D | C + batch 8／累積 2 | 97.4 | 84.1 | 10.9 | 0.88087 | 2214 |
| E | C + batch 16／累積 1 | 84.9 | 71.7 | 10.6 | 0.89349 | 2336 |

全部在 3,900 MiB 停止條件以下，E 的條件式（D peak reserved < 3,000 MiB）成立而執行。

**等價性（已驗證）**：B vs A、C vs B 的 `model.safetensors` SHA-256 相同、最大權重差 0、驗證混淆矩陣相同（`[[599,27,10],[16,113,5],[2,7,121]]`），代表訓練位元相同；評估 batch 與 gradient checkpointing 不改變訓練結果。B 的 val loss 與 A 差 7.9e-6，來自評估 batch 不同的浮點累加順序。

**B 訓練段反而較慢（369 vs 305 秒）**：B 與 A 的訓練位元相同，訓練工作量沒有差別，因此屬環境雜訊；推測為 GPU 降頻（log 內 B 最長無輸出 42 秒），但沒有溫度／時脈取樣，未證實。結論：單次量測的訓練段速度有雜訊，各組速度差距以量級看待。

**D／E 非等價**：dropout 亂數序列與 padding 長度都不同。1 epoch 的 F1 差異（D −0.0092、E +0.0034）在單次雜訊範圍內，**不據此判定分數優劣**；要採用須另行登記完整訓練對照。

## MLM 結果（`pretrain-mlm` 1 epoch）

| 組別 | 設定 | 總秒數 | 訓練段 | 評估段 | train MLM loss | peak reserved（MB） |
|---|---|---:|---:|---:|---:|---:|
| M-A | 現行 | 334.6 | 327.4 | 6.06 | 2.8940 | 2028 |
| M-C | `--no-gradient-checkpointing --eval-batch-size 64` | 224.5 | 221.4 | 2.11 | 2.8940 | **4166** |
| M-D | — | 未執行 | | | | |

M-C 的 train MLM loss 與 M-A 完全相同（2.894016），但 peak reserved 4,166 MiB **超過停止條件 3,900 MiB**，已接近／超過 4 GB 實體顯存（Windows 可能溢出到共用記憶體），因此依條件跳過 M-D。推測超標來自 MLM 評估 batch 64 × 30,522 維詞表的 logits（**未驗證**，未另做只改一項的對照）。M-C 的 val MLM loss（2.5098 對 2.5469）不可與 M-A 逐位比較：MLM 驗證遮罩在固定 RNG 下依評估 batch 物化，改評估 batch 會改變遮罩。

## 樣本／分鐘與 log 靜默時間（供監控門檻）

| 組別 | 中位數（樣本/分） | 最低視窗 | 最長無輸出 |
|---|---:|---:|---|
| A | 1702 | 1368 | 149.4 秒（評估） |
| B | 1686 | 728 | 42.5 秒（啟動；啟動到第一行 42 秒） |
| C | 2587 | 2259 | 13.4 秒 |
| D | 5815 | 5603 | 12.6 秒 |
| E | 6777 | 6605 | 12.5 秒 |
| M-A | 1552 | 1221 | 19.7 秒 |
| M-C | 2276 | 1914 | 12.5 秒 |

速率以相鄰兩行 `batch b/N` 的樣本增量／時間差計算（8,100 筆訓練樣本；略過第一段）。重算腳本在工作 scratchpad，輸出與上表一致；原始資料為 `outputs/speed_check/*.log`（每行前綴 unix 時間戳記）與 `*.gpu`（結束後溫度與時脈）。

## 結論與採用建議

1. **微調**：可採用 `--eval-batch-size 64 --no-gradient-checkpointing`，位元相同，1 epoch 約 467 → 203 秒（約 2.3 倍）；評估段 143 → 11 秒。
2. **MLM**：可採用 `--no-gradient-checkpointing`（train loss 相同、約 1.5 倍），但評估 batch 不能用 64，需改小（例如 16）並另測記憶體；改評估 batch 會讓 val MLM loss 與舊 run 不可逐位比較。
3. batch 8／16 約 3.6～4.3 倍但非等價，列為待登記的後續實驗，本輪不採用。
4. **監控腳本** [`tools/gpu_watch.sh`](../../tools/gpu_watch.sh) 依上述數據校正門檻，速度改以樣本/分鐘計（與 batch size 無關）：

| 門檻 | 預設 | 理由 |
|---|---|---|
| `STALL_SEC` | 180 | 加速設定下最長無輸出約 12～20 秒、啟動到第一行最長 42 秒；現行設定評估靜默 149 秒，180 秒不誤報又比舊版 300 秒早發現。 |
| `LOW_UTIL`／`LOW_UTIL_SEC` | 10%／120 秒 | 評估與存檔都短於此，持續低使用率代表卡住。 |
| `VRAM_MIB`／`VRAM_SEC` | 3990／60 | 4 GB 卡接近滿載會溢出共用記憶體而大幅變慢；M-C 即達 4,166 MiB。 |
| `TEMP_C`／`CLK_MHZ` | 87／1200（各 180 秒） | 與舊版相同。 |
| 速度 | 以「有效時間」計：視窗 `WIN_SEC=120`、完整視窗＝有效時間跨度 ≥ 0.8×`WIN_SEC`；每 stage 取前 3 個完整視窗（取樣間隔有效時間 `WIN_SEC`/2）速率中位數為基準；< 0.5 倍異常、≥ 0.6 倍恢復；基準前低於 `MIN_SPM=500` 也算異常 | 不同設定的速度差 4 倍以上，固定門檻不可行；C 設定每 epoch 訓練段約 190 秒、E 設定 5 epoch 訓練共約 6 分鐘，舊的 300 秒視窗約 13 分鐘才有基準，故縮為 120 秒。評估期間（出現 `Epoch E: train F1=`／`train MLM=` 摘要行）的行間隔，有效時間改記為上一個正常間隔，不把評估時間算進速率；不再用固定間隔清空視窗（舊 `GAP_RESET=90` 在速率低於約 267 樣本/分鐘時每次都清空而漏報嚴重變慢），完全沒有新行交給 `STALL_SEC`。監控只判斷啟動之後新增的 log 內容（記錄啟動時檔案大小），不受追加式 log 的舊 run 影響。 |

**已於 2026/11/19 採用為預設**：`train` 評估 batch 預設 64（`--eval-batch-size` 未指定時）；`train` 與 `pretrain-mlm` 的 gradient checkpointing 預設關閉（以 `--gradient-checkpointing` 開啟，舊的 `--no-gradient-checkpointing` 仍可用）；`pretrain-mlm` 評估 batch 仍沿用 `--batch-size`，未採用 64。batch 8／16 非等價，未採用。

## 沿用成果、未完成與限制

- 沿用：既有 8,100／900 切分、`sqrt-inverse` 權重、`main.py train`／`pretrain-mlm`；新增程式僅 `--eval-batch-size` 與計時／記憶體欄位（預設不指定時行為不變）。`tools/cv_oof.py` 的 OOF 推論 batch 未更動。
- 每組只量一次（單次量測、seed 42），訓練段速度有雜訊（見 B）；溫度與時脈只在每組結束後取樣一次，過程中降頻只是推測。
- 無逐筆預測一致率：等價性以權重 hash、權重差與驗證混淆矩陣證明；D／E 與 M-C 未做逐筆比較。
- M-C 記憶體超標的原因未驗證；MLM 評估 batch 16 的記憶體與速度未測。
- 本輪產物不是訓練 run：未寫入 registry 的 runs，也不產生候選；是否計入核心完整訓練預算未另行判定。
