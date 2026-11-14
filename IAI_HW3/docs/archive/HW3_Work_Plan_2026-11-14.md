> 原指令路徑可能已調整；現行程式位置與操作指令請見 [README](../../README.md)。

> 封存快照：保留整理前的內容；其中進度與「下一步」可能已過時。 現況以 [實驗總表](../Experiment_Results.md) 與 [最新計畫](../Target_92_Plan.md) 為準。指令的工作目錄為 HW3 根目錄。

# IAI HW3｜情緒分類工作計畫

以助教《HW3: Sentiment Analysis》第 17–32 頁的 Hands-on Tutorials on NLP 六個主題為骨架。第 7–15 頁提供作業與交件規格。工作區原始講義位於 `docs/HW3.pdf`。

## 2026/11/14 更新：以 Kaggle 92% 為目標

後續執行順序以 [Target_92_Plan.md](../Target_92_Plan.md) 為準：評估與推論一致性檢查 → 新控制組 → 有限的分層學習率／FGSM／較溫和類別權重對照 → 額外固定切分確認 → 有條件集成與最終重訓 → 交件演練。核心上限 9 次訓練，另有條件增加 1 次全資料重訓；目前僅制定計畫，尚未啟動。下方保留歷史實驗與課程主題，舊的「下一項」敘述不代表最新排程。

## 目標、工作區與分工

用 `google-bert/bert-base-uncased` 微調三分類模型，輸出 positive、neutral、negative，並交付可執行的程式與 checkpoint。以驗證集 Macro-F1 選模型，最終成績由 Kaggle benchmark 決定。

| 路徑 | 用途 |
|---|---|
| `dataset/dataset.csv` | 9,000 筆有標籤資料，從這裡切出 train／validation。 |
| `dataset/benchmark.csv` | 1,000 筆無標籤資料，正式推論時使用。 |
| `docs/HW3.pdf`、`docs/HW3_Work_Plan.md` | 助教講義與工作計畫。 |
| `main.py` | 已提供 `inspect`、`eda`、`prepare`、`train`、`predict`；訓練迴圈位於 `training.py`。 |
| `README.md`、`requirements.txt` | 執行入口、路徑、已驗證版本及指令；完整環境版本另存 `requirements-lock.txt`。 |
| `saved_models/checkpoint/` | 助教指定的 checkpoint 路徑。保存入選模型、tokenizer、類別順序、最大長度及前處理設定。 |
| `outputs/` | 實驗指標、曲線與 Kaggle 預測檔，於實際執行時建立。 |

| 里程碑 | Codex | 你 | 完成條件 |
|---|---|---|---|
| 11/13–11/15｜EDA 與基準 | 固定切分、實作自訂前處理、BERT 訓練、驗證、畫圖、推論，輸出 `id,pred` CSV。 | 確認本機 GPU，安排短跑。 | 少量 batch 跑通，checkpoint 可重載，完成第一輪完整基準。 |
| 11/16–11/23｜優化 | 按下列六個主題分析，逐項比較變因，整理各類 F1 與耗時。 | 安排正式訓練資源，依驗證結果選定優先實驗。 | 有固定切分下可重現的最佳單模型與實驗紀錄。 |
| 11/24–11/29｜確認及交件演練 | 視預算驗證 FGSM 或 ensemble，重跑入選 checkpoint，核對 ZIP 與預測。 | 登入 Kaggle／E3，確認隊名、提交版本與平台結果。 | 解壓後可重現推論，兩平台均確認收到正確版本。 |

日期為建議進度。E3 與 Kaggle 正式截止均為 **2026/11/30（一）23:59**。

## 1. EDA

**2026/11/12 已完成第一項實作與實測**：入口 `python main.py eda`，seed 42，依空白正規化文字分組後分層切分為 train 8,100／validation 900。訓練集 token 長度 P50／P90／P95 = 31／40／43，`max_length=64` 截斷率 0.02%。完整報告、固定 ID 與 tokenizer 範例位於 `outputs/eda/`；操作方式見 README。下一步進行基準訓練短跑。

| 標籤 | 筆數 | 比例 |
|---|---:|---:|
| negative | 6,364 | 70.71% |
| neutral | 1,335 | 14.83% |
| positive | 1,301 | 14.46% |

本機 `dataset.csv` 的 9,000 筆皆有非空白文字及唯一 ID。原文字串比唯一文字多 12 筆，相同文字的標籤一致。初步用正則檢查，633 筆含 URL 標記、1,558 筆含 hashtag，所有樣本皆有符合簡單規則的 @mention。這些是全檔盤點，前處理決策仍以訓練切分的細查為依據。

- 建議約 90／10 分層切分，以相同原文為群組，將同組留在同一側，核對兩側無完全相同的文字。固定 seed，保存分割 ID 與類別數，後續實驗共用。
- 在訓練集用指定 BERT tokenizer 統計中位數、P90、P95 與各候選長度的截斷率。長度計入特殊 token，批內動態 padding。選擇能保留大多數資料的最短候選，依實測顯存修正。
- 按類別檢查 URL、emoji、hashtag、mention 的頻率，抽樣查看 tokenizer 如何處理。若 @mention 幾乎固定在開頭，可先看是否只是同一帳號或是否會擠占序列長度，再決定是否測替換。

## 2. Data Preprocessing

**2026/11/13 已完成 URL 正規化對照**：沿用加權設定，URL 統一為 `http`；最佳第 2 輪驗證 Macro-F1 91.28%，低於保留原 URL 的 92.65%，本次不採用。結果見 [URL_Normalization_Report.md](../reports/URL_Normalization_Report.md)。使用者回報 URL 正規化 Kaggle 0.8977，低於原加權模型 0.9049。保留原加權模型，下一項比較學習率 1e-5；其他設定不變。

基準先做自訂的空白正規化，保留否定詞、標點、hashtag 字詞、原始 URL 與 emoji。訓練、驗證、推論共用同一前處理及設定。

| 改動 | 對照及保留依據 |
|---|---|
| URL 正規化或移除 | 固定其他設定，比較原文、統一佔位字與移除。URL 出現率有類別差異，先比較正規化是否保留有用訊號。檢查 `[URL]` 等字串的實際切詞結果。 |
| Emoji 轉文字或移除 | 檢查 emoji 是否常成為未知 token，並抽查錯判。優先比較保留原樣與英文描述，再視結果比較移除。 |
| Mention 或 hashtag 處理 | EDA 顯示有明確重複模式時再試，且保留 hashtag 的字詞內容作第一個對照。 |

## 3. Class Imbalance Handling

**2026/11/13 已完成第一個加權 CE 對照**：反平方根權重的第 3 輪入選，驗證 Macro-F1 92.65%，相較基準 91.57% 提升 1.08 個百分點；neutral F1 由 84.73% 升至 86.47%。模型、曲線與 1,000 筆預測位於 `outputs/weighted_sqrt/`，詳見 [Weighted_CE_Report.md](../reports/Weighted_CE_Report.md)。使用者回報基準 Kaggle 0.8973、加權模型 0.9049，提升 0.76 個百分點。下一項比較 URL 正規化。

先用未加權 cross-entropy 建立基準。若混淆矩陣顯示 neutral 或 positive 常被預測成 negative，第一個正式改進試驗採 weighted cross-entropy。權重由訓練集頻數計算，可先試較溫和的反頻率平方根。若有必要，再用 balanced sampling 做獨立對照，驗證集維持原始分布。

比較 Macro-F1、三類各自的 F1、precision、recall 和混淆矩陣。留意少數類 recall 上升時 precision 是否明顯下降。

## 4. Model Training Tips

**2026/11/13 已完成學習率 `1e-5` 對照**：三輪驗證 Macro-F1 為 90.30%、90.99%、91.46%，最佳第 3 輪低於原加權模型 92.65%，因此仍保留 `weighted_sqrt`。沿用反平方根加權 CE、原 URL、seed 42 與固定 8,100／900 切分；run config 唯一差異為 learning rate。11 項測試、最佳 checkpoint 重載核對及 1,000 筆預測格式檢查通過，曲線已檢視。結果存於 `outputs/weighted_lr1e5/`，詳見 [Learning_Rate_Report.md](../reports/Learning_Rate_Report.md)。使用者回報本輪 Kaggle **0.9012**，低於原加權模型 0.9049，下降 0.37 個百分點；後續 `3e-5` 對照亦已完成，最佳驗證 Macro-F1 92.10%，仍低於 2e-5 的 92.65%；三個候選學習率比較完成，保留 2e-5。詳見 [3e-5 對照報告](../reports/Learning_Rate_3e5_Report.md)。本輪 checkpoint 重載及 1,000 筆預測格式核對通過。使用者回報 3e-5 的 Kaggle **0.9052（90.52%）**，比 2e-5 高 0.03 個百分點，為目前已回報的最高 Kaggle 分數，且達下列評分表的 90 分門檻。驗證集最佳仍為 2e-5，兩者均保留。錯判分析與首次集成比較已完成，見第 6 節。

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --learning-rate 1e-5 --output-dir outputs/weighted_lr1e5 --checkpoint outputs/weighted_lr1e5/checkpoint
```

本輪已完成，重跑需指定新路徑。

**2026/11/12 已完成基準流程、GPU 短跑及完整 3 epochs 基準**：完整驗證 Macro-F1 依序為 90.09%、88.78%、91.57%，第 3 輪入選。最佳 checkpoint 重載與 1,000 筆 `id,pred` 推論核對通過；完整訓練約 19.7 分鐘，峰值配置顯存約 1.86 GiB。詳見 [Baseline_Report.md](../reports/Baseline_Report.md)及 [Baseline_Smoke_Report.md](../reports/Baseline_Smoke_Report.md)。下一項比較加權 cross-entropy，保留本輪未加權基準。

- **基準**：BERT 接線性三分類 head、AdamW、linear warmup／decay。建議起點為學習率 2e-5、最多 3 epochs，實際長度與 batch 由 EDA 及顯存短跑決定。
- **本機短跑**：RTX 3050 Laptop 4 GB 可先用 micro-batch 2 試幾個 forward／backward steps，記錄峰值顯存與耗時。視 CUDA 環境測混合精度，以梯度累積調整有效 batch。累積時 scheduler 按 optimizer 更新次數前進。顯存或速度不足，再用其他 GPU 執行同一份程式。
- **曲線與 checkpoint**：每 epoch 記錄 train／validation loss 與兩側 Macro-F1。用固定驗證集的 Macro-F1 保存最佳 checkpoint。延長訓練時可加入 early stopping。Macro-F1 是三類 F1 的等權平均。
- **小範圍搜尋**：先比 1e-5、2e-5、3e-5。再依曲線選擇是否比較 linear／cosine 排程、weight decay 或有效 batch。基準穩定後才比較線性 head 與小型 MLP head。
- **程式責任**：自行組合資料切分、DataLoader、訓練／驗證迴圈、checkpoint 與推論；使用模型和 tokenizer 套件載入指定基底。Checkpoint 與推論輸出都使用同一份類別對照。

## 5. Adversarial Training

在基準穩定、低成本實驗做完且仍有 GPU 時間時，才試 FGSM。使用乾淨輸入的 loss 梯度，在 embedding 上施加小幅擾動，再以原始與擾動輸入的 loss 訓練。以無擾動為基準，比較少量 epsilon，記錄驗證 Macro-F1、最差類別 F1、顯存與耗時。驗證及正式推論使用乾淨輸入。

## 6. Ensemble Learning

**2026/11/13 已完成首次兩模型錯判分析與等權 soft voting**：`2e-5`、`3e-5` 集成驗證 Macro-F1 92.58%，低於單模型 92.65%，因此未依驗證結果選用。其後使用者要求仍產生 Kaggle 對照，已輸出 `outputs/ensemble_lr2_lr3_submission/predictions.csv`，1,000 筆格式與順序核對通過。使用者於 2026/11/14 回報集成 Kaggle **0.9052**，與 `3e-5` 單模型同分，未提高目前最高分；同分時優先保留較簡單的 `3e-5` 單模型作為 Kaggle 候選，驗證最佳仍為 `2e-5`。兩模型共同錯判 39 筆、分歧 15 筆，互補有限；主要需關注 neutral／negative 邊界與混合情緒。詳見 [錯判與集成報告](../reports/Ensemble_Error_Analysis_Report.md)。2026/11/14 已完成 2e-5 加權設定的 seed 43 對照：固定切分不變、run config 唯一差異為 seed。最佳第 3 輪驗證 Macro-F1 91.33%，低於 seed 42 的 92.65%；重載核對通過，結果與 Kaggle 對照 CSV 位於 `outputs/weighted_seed43/`。使用者於 2026/11/14 回報本輪 Kaggle **0.8878**，比相同設定 seed 42 低 1.71 個百分點，比目前最高分 0.9052 低 1.74 個百分點；不取代既有最佳候選。詳見 [Seed43_Report.md](../reports/Seed43_Report.md)。2026/11/14 seed 42／43 互補與等權集成已完成，驗證 Macro-F1 92.46%，未超過最佳單模型。共同錯判 39 筆、分歧 20 筆；集成修正 3 筆、新增 5 筆錯誤。已產生並核對 `outputs/ensemble_seed42_seed43/predictions.csv`。使用者於 2026/11/14 回報 Kaggle **0.8983**，比 seed 42 單模型低 0.66 個百分點，比目前最高分 0.9052 低 0.69 個百分點；不取代既有最佳候選。詳見 [Seed_Ensemble_Report.md](../reports/Seed_Ensemble_Report.md)。2026/11/14 訓練集 emoji／tokenizer 盤點已完成，249 筆受影響、88 種符號單獨切詞皆為 `[UNK]`。詳見 [Emoji_Audit_Report.md](../reports/Emoji_Audit_Report.md)。2026/11/14 已實作 `emoji-short-v1`，13 項測試通過，訓練集 203 筆改變，無新增跨集合相同文字；2e-5／seed 42 三輪對照已完成，最佳驗證 Macro-F1 91.68%，低於原加權 92.65%。Checkpoint 重載與 CSV 核對通過，結果及提交檔存於 `outputs/weighted_emoji/`，尚無 Kaggle 分數。詳見 [Emoji_Training_Report.md](../reports/Emoji_Training_Report.md)。

從使用相同 BERT 基底、不同 seed 或訓練設定的模型挑 2–3 個，優先考慮誤判有互補的組合。以固定驗證集比較最佳單模型、soft voting 與 hard voting。Soft voting 先依樣本 ID 與類別順序對齊，再平均 softmax 機率。Hard voting 若平票，預先定義以平均機率判定。只有驗證分數能支持額外成本時才交 ensemble，最多三個模型。

## 實驗順序與紀錄

先完成 EDA、固定切分與可重跑基準，再優先比較加權 loss 和 URL 正規化。之後依錯判與曲線決定 emoji 處理、balanced sampler 或訓練參數。FGSM 和 ensemble 保留為最後選項。每輪記錄切分、seed、前處理、超參數、最佳 epoch、三類 F1、Macro-F1、耗時與 checkpoint。小幅進步可用另一訓練 seed 重跑比較。

## 評分與交件

| Kaggle Benchmark Macro-F1 | 得分 |
|---|---:|
| ≥ 92% | 100 |
| ≥ 90.5% 且 < 92% | 90 |
| ≥ 89% 且 < 90.5% | 80 |
| < 89% | 0 |

- [ ] 訓練與驗證只使用助教提供的有標籤資料切分，benchmark 僅供最終推論。
- [ ] 依講義第 11 頁建立 `HW3_{student_id}/`，含 `dataset/` 下兩份 CSV、`saved_models/checkpoint/`、`main.py`、`README.md`、`requirements.txt`，以及執行所需檔案。備妥每 epoch 的訓練與驗證曲線。
- [ ] 從乾淨解壓目錄照 README 跑推論，確認 checkpoint 可載入。提交 CSV 首列須為 `id,pred`，依 `benchmark.csv` 的列順序輸出全部 1,000 個原始 ID，每個 ID 恰好一筆，`pred` 須為 `positive`、`neutral` 或 `negative`。
- [ ] E3 上傳 `HW3_{student_id}.zip`。檔名錯誤扣 10 分，助教無法執行得 0 分。
- [ ] Kaggle 隊名用學號，帳號綁課堂名單上的 email，每 24 小時最多提交 3 次。[Kaggle 競賽邀請](https://www.kaggle.com/t/89f6224e259745938f459f25926f095c)
