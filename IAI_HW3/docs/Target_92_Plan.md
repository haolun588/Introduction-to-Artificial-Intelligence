# Kaggle 92% 改進計畫

制定：2026/11/14；更新：2026/11/17（第二次路線調整）。Kaggle 目標 ≥0.9200，目前最高使用者回報 0.9152（cv5_sqrt_ep5_mlm10 集成；非 MLM 最高為 weighted_quarter 0.9122）。A、B、C1、C2 均已完成；FGSM 兩設定驗證均未改善，停止幅度搜尋。步驟 1（logit 偏移校正）、步驟 2（五折 CV）、步驟 3a（sqrt vs quarter CV 重比）、步驟 3b（訓練長度 3 → 5 epochs）已完成；步驟 3c（label smoothing）已中止並由第二次調整取代，順序 0（上傳集成 CSV）已由使用者回報 Kaggle 0.8971（見「路線調整（2026-11-16，第二次）」），下一步待定。順序 2 錯標抽查已通過（[報告](reports/Label_Noise_Audit_Report.md)），`cv5_sqrt_ep5_drop90` 已完成（[報告](reports/CV5_Drop90_Report.md)）：OOF 89.34%，Δ +0.12 個百分點，未達預先登記門檻 +0.5，順序 2 已停止；順序 3 MLM 續訓完成（OOF 89.63%，Δ +0.41，未達門檻；Kaggle 0.9152）。下一步：方向 2（cv5_sqrt_ep5_mlmall30）已完成（[報告](reports/CV5_MLMAll30_Report.md)），Kaggle 0.9142 < 0.9152，依預先登記主版本維持 mlm10（目前最高仍為 mlm10 0.9152）；方向 3（full8100_mlm10）已預先登記、未開工；下一步待與使用者討論。

## 路線調整（2026-11-16，第二次）

本節取代下方「新研究路線」步驟 3 的後續（label smoothing／R-Drop，原步驟 3c 之後）與步驟 4；舊內容保留供對照。步驟 1、2、3a、3b 的結果與決策不變。

**調整原因。**

- (a) 近期單因素實驗（類別權重、訓練長度）預期效果約 0.1–0.3 個百分點，低於單 seed 五折 CV 約 0.5 個百分點的偵測力，結果幾乎注定不顯著（3a 的 Δ −0.23、3b 的 Δ +0.19 即為例，區間均含 0）。
- (b) CV 線（OOF 89.22%）從未回 Kaggle 驗證，OOF 與 Kaggle（目前最佳使用者回報 0.9122）的對應未知；已知的 900 驗證與 Kaggle 排序不一致，OOF 是否較能預測 Kaggle 屬未驗證。
- (c) 原計畫沒有停損規則，也沒有全部失敗時的退路。

**規則。**

- 只排預期效果 ≥0.5 個百分點的方向。
- 每個方向至多 1 組 CV（5 次訓練）。
- OOF Δ < +0.5 個百分點即停，換下一個方向。
- 方向產生候選時，做 3 模型集成 CSV 交 Kaggle，同時累積 OOF 與 Kaggle 的對應資料。集成上限 3 個模型（作業紅線）。
- 仍遵守作業紅線：只用提供資料、只用 bert-base-uncased、不用 benchmark 標籤或文字訓練。

**順序。**

| 順序 | 內容 | 說明 |
|---|---|---|
| 0 不訓練 | 上傳 `cv5_sqrt_ep5` 的 fold 0／1／2 等權集成 CSV：`outputs/cv5_sqrt_ep5_oof/submissions/cv5_folds012_mean.csv`（SHA-256 前 12 碼 `351963b9048f`，見 [實驗總表](Experiment_Results.md)） | 由使用者上傳 Kaggle 並回報分數；這是第一筆 OOF（89.22%）與 Kaggle 的對應資料。結果：Kaggle 0.8971（使用者回報，未經平台核驗）；OOF 89.22% |
| 1 多 seed | 同設定（sqrt-inverse、5 epochs、同一份 `cv5_seed42` 切分）只改訓練 seed 為 43，跑一組 CV | 兩個 seed 的 OOF 機率平均後可直接在 OOF 上量測集成效果，並估計 seed 變異 |
| 2 錯標降權 | 先用現有 OOF 找「高信心卻與標籤不同」的樣本（不需訓練），優先人工抽看 neutral；確認確有錯標後才跑一組 CV 降權 | 抽看結果若顯示並非錯標，則不訓練，直接換下一方向。結果：抽查通過（[報告](reports/Label_Noise_Audit_Report.md)；A+B annotation ≠ label 36/60 = 60%，C 組一致率 100%，|H| = 473），使用者選擇排除版本（w=0），預先登記見下「順序 2 預先登記」。**結果（`cv5_sqrt_ep5_drop90`，[報告](reports/CV5_Drop90_Report.md)）：OOF 89.34%（對照 89.22%），Δ +0.12 個百分點，paired bootstrap 95% [−0.49, +0.76]，fold 勝 3/5；未達門檻 +0.5，停止此方向，轉順序 3** |
| 3 MLM 續訓 | 只用 train 文字做 MLM 續訓（benchmark 文字不可用） | 作業紅線：不得使用 benchmark 文字。已預先登記，見下 **結果（`cv5_sqrt_ep5_mlm10`，[報告](reports/CV5_MLM10_Report.md)）：OOF 89.63%（對照 89.22%），Δ +0.41 個百分點，paired bootstrap 95% [−0.21, +1.06]，fold 勝 3/5；未達門檻 +0.5，不列候選、不提交 Kaggle，待與使用者討論後續（備選或退路）；Kaggle 0.9152（使用者回報，目前最高）** |
| 4 備選 | label smoothing、R-Drop、pooler penalized tanh | 原步驟 3c 預先登記的 label smoothing 見下，其 `cv5_sqrt_ep5_ls01` 已中止。penalized tanh（x>0 為 tanh(x)，x≤0 為 0.25·tanh(x)；Eger 等人 2018 EMNLP〈Is it Time to Swish?〉比較中跨 NLP 任務最穩定，但該比較為從零訓練的小型網路）：只替換 BERT pooler 的 tanh，encoder 的 GELU 不動（換掉會破壞預訓練權重，且有改動 base model 的合規疑慮）；預期效果 ±0.3 個百分點內（推測），低於單 seed CV 約 0.5 個百分點的偵測力。若執行：唯一變因為 pooler 激活，1 組 CV（約 2.5 小時，不需續訓），Δ < +0.5 即停；執行前另寫預先登記 |

**退路。** 上述全部未達門檻（OOF Δ ≥ +0.5 個百分點，或 Kaggle 超越 0.9122）→ 交件採 Kaggle 最佳的 `weighted_quarter`（0.9122，使用者回報），並補做其獨立封裝與解壓推論演練（目前尚未演練）。

**已驗證與推測的區分。** 已驗證：OOF 數字（sqrt ep3 89.03%、quarter 88.80%、sqrt ep5 89.22%）與 Kaggle 回報（weighted_quarter 0.9122、weighted_sqrt 0.9049）。推測：「單 seed 五折 CV 約只能偵測 ≥0.5 個百分點」、各方向預期增益、OOF 與 Kaggle 的對應關係，均待步驟 0 及後續資料檢驗。

**順序 2 預先登記：cv5_sqrt_ep5_drop90（2026-11-16，訓練前寫下）**

- 假設：OOF 高信心不一致樣本多含錯標或模糊標籤，自訓練集排除可改善 OOF／Kaggle。
- 對照：`cv5_sqrt_ep5`（OOF 89.22%，Kaggle 集成 0.8971，使用者回報）；同切分 `cv5_seed42`、sqrt-inverse、5 epochs、seed 42。
- 唯一變因：以 `--exclude-train-ids outputs/label_noise_audit_20261116/exclude_ids_p090.txt`（473 筆）自每折訓練子集排除；validation 不變。實作細節：類別權重依過濾後的訓練子集計算。
- 指令：`bash tools/run_cv5.sh cv5_sqrt_ep5_drop90 sqrt-inverse 5 --exclude-train-ids outputs/label_noise_audit_20261116/exclude_ids_p090.txt`；比較 `python -m tools.compare_oof --a outputs/cv5_sqrt_ep5_oof --b outputs/cv5_sqrt_ep5_drop90_oof --output outputs/cv5_compare_ep5_vs_drop90.json`。
- 預算：5 次訓練，不追加。
- 決策規則（預先固定）：OOF Δ（drop90 − ep5）≥ +0.5 個百分點 → 列為候選，產生 fold 0／1／2 集成 CSV 交使用者上傳 Kaggle；Δ < +0.5 → 停止此方向，換順序 3。另外觀察 neutral recall（原 79.3%）是否下降（次要，不改決策）。
- 已知偏誤：排除清單由 `cv5_sqrt_ep5` 的 OOF 產生，而產生每筆 OOF 的模型訓練時看過其他折的驗證資料，因此新 CV 的 OOF 對自身略偏樂觀（未量化）；Kaggle 不受此影響，最終以 Kaggle 為準。
- 風險：A 組約 35% 的被排除樣本（14/40）原標籤可能正確（多為 neutral 問句），排除可能加重模型對 neutral 問句的負面偏誤。追加檢查（見[抽查報告](reports/Label_Noise_Audit_Report.md)）顯示此風險較原估計低，預先登記不變。

**順序 2 結果與判定（2026-11-17，[報告](reports/CV5_Drop90_Report.md)）：** OOF Macro-F1 89.34%（對照 `cv5_sqrt_ep5` 89.22%），Δ +0.12 個百分點，paired bootstrap 95% [−0.49, +0.76]，fold 勝 3/5，最佳 epoch 5、5、4、5、4。未達預先固定門檻 +0.5 → 不列候選、停止錯標排除方向、不產生 Kaggle 提交（集成 CSV 由工具自動產生，狀態 `not_submitted`）；下一步為順序 3（MLM 續訓），待設計。次要觀察：neutral recall 79.3% → 77.3%（−2.0），neutral → negative 219 → 242；negative、positive recall 上升，與預先登記風險方向一致（推測排除也移除了部分標籤正確的 neutral 樣本，未檢驗）。已知偏誤（排除清單來自對照 OOF，新 OOF 略偏樂觀）偏向 drop90，仍未達門檻。

**順序 3 預先登記：cv5_sqrt_ep5_mlm10（2026-11-17，訓練前寫下）**

- 假設：只用各折訓練子集文字做 MLM 續訓（任務適應預訓練，TAPT），讓 BERT 適應航空推文語言，可提升分類；預期 +0.3～1 個百分點（推測，未驗證）。
- 對照：`cv5_sqrt_ep5`（OOF 89.22%）；同切分 `cv5_seed42`、sqrt-inverse、5 epochs、seed 42。
- 唯一變因：微調初始權重改為該折 MLM 續訓後的權重（分類頭仍新初始化）。
- MLM 設定（預先固定、不調參）：起點 `google-bert/bert-base-uncased`；語料僅該折訓練子集文字（約 7,200 筆），前處理與微調相同；不用標籤、不用 benchmark 文字、不用 validation 文字；遮罩 15%（80/10/10）；lr 5e-5；warmup 6%；linear decay；weight decay 0.01；10 epochs；batch／累積／max_length 與微調相同；儲存最後一個 epoch，不做選擇；validation 文字只用來記錄 MLM eval loss。
- 為何每折各自續訓：若用全部 9,000 筆文字續訓一次，每折 validation 文字已被看過，OOF 會被直接高估；每折各自續訓可避免此洩漏。最終交件模型（若成功）也只用原訓練子集 8,100 筆文字續訓。
- 指令：先 `bash tools/run_cv5_mlm.sh cv5_sqrt_ep5_mlm10 sqrt-inverse 5 10 0`（fold 0 試跑），通過後 `bash tools/run_cv5_mlm.sh cv5_sqrt_ep5_mlm10 sqrt-inverse 5 10`；比較 `python -m tools.compare_oof --a outputs/cv5_sqrt_ep5_oof --b outputs/cv5_sqrt_ep5_mlm10_oof --output outputs/cv5_compare_ep5_vs_mlm10.json`。
- fold 0 試跑：只檢查流程正常、MLM val loss 是否下降、實際耗時；**不以 fold 0 的分類分數做決策**。若 MLM val loss 未下降或流程異常，停下檢查，不繼續。
- 預算：5 次 MLM 續訓 + 5 次微調，不追加；估計每折約 75 分鐘、整組 6–7 小時（推測，以試跑實測為準）。
- 決策規則（預先固定）：OOF Δ ≥ +0.5 個百分點 → 候選，產生 fold 0／1／2 集成 CSV 交使用者上傳 Kaggle；Δ < +0.5 → 停止此方向，備選（label smoothing、R-Drop）預期效果小，屆時與使用者討論是否執行退路。
- 合規：講義 p.14 要求只用提供資料切出的 train／validation 訓練與驗證，本設計只用訓練子集文字，符合此條。講義 p.9 規定「ONLY use google-bert/bert-base-uncased as your base model」，全講義未提及預訓練／續訓／MLM，也未定義 base model 的範圍；主會話判讀「起點為指定模型、續訓屬自建訓練管線（p.9 #2）」應屬合規，但這是推測、非講義明文。使用者於 2026-11-17 知悉此不確定性，判斷風險可接受並決定執行（使用者決定，非講義確認）。若之後助教回覆不允許，此方向產物不得用於交件。
- 交件影響：若成功，交件需附 MLM 續訓程式、紀錄與曲線，確保從官方 checkpoint 起可完整重現。
- 交件規則（使用者 2026-11-17 指示）：若最終分數最高的是 MLM 續訓版本，交件時：
  1. 附一份文件，主動說明講義 p.9「ONLY use google-bert/bert-base-uncased as your base model」未定義是否涵蓋自行續訓；寫明本做法的流程（從官方 checkpoint 出發、只用訓練子集文字、未用 benchmark）與判讀理由。
  2. 同時附上第二高分的非 MLM 版本作為保險，含其 checkpoint、推論程式與 CSV，並在 README 明確標示主要版本與保險版本。
  - 待確認事項（未知，不可假設）：E3 壓縮檔放兩組 checkpoint 的格式與大小限制；Kaggle 最終計分提交可選幾筆。

### 方向 2 預先登記：cv5_sqrt_ep5_mlmall30（2026-11-18 使用者確認，訓練前寫下）

狀態：**已確認**。使用者 2026/11/18 確認本預先登記，設定不變；開工前程式改動須經測試與程式碼審查。

- 背景：使用者說明 Kaggle 無 public／private 之分（leaderboard 分數即成績），目標是最終有一個 Kaggle > 0.92 的模型；方法用盡後可用 Kaggle 在已登錄候選間挑選（屆時另寫上傳次數上限）。因此本實驗以 Kaggle 為主判斷，OOF 只參考。
- 假設：MLM 續訓看更多文字、更多 epoch，可使 Kaggle 高於 mlm10 的 0.9152。
- 設定（與 `cv5_sqrt_ep5_mlm10` 的差異）：續訓語料為 `dataset/dataset.csv` 全部 9,000 筆文字（只用文字、不用標籤、不碰 benchmark；mlm10 為每折約 7,200 筆）；續訓 30 epochs（mlm10 為 10）；只續訓 1 次，5 折共用同一 checkpoint（mlm10 每折各自續訓）；續訓無留出文字，只記錄訓練 loss。其餘相同：lr 5e-5、遮罩 15%（80/10/10）、batch 2、累積 8、seed 42、儲存最後 epoch；微調 sqrt-inverse、5 epochs、切分 `outputs/cv5_seed42`。
- 時間估計：續訓約 3.3 小時（每 epoch 約 6.6 分鐘），微調 5 折約 2.75 小時，合計約 6 小時。
- 判斷（以 fold 0/1/2 集成 CSV 的 Kaggle 分數）：≥ 0.92 → 主要候選；介於 0.9152 與 0.92 → 沿此線延伸（更長續訓或更高遮罩率）；< 0.9152 → 主版本維持 mlm10。OOF 因續訓看過各折驗證文字而偏高，只作參考、不作判斷。
- 限制：單次 Kaggle 比較雜訊約 ±1 個百分點（只換 seed 曾差 1.7 個百分點）；一次改兩個因素（語料範圍、epoch 數），無法分辨哪一個有效。
- 合規：只用課程提供資料的文字；講義 p.9「base model」續訓風險使用者已於 11/17 接受；若此版最高，依交件規則附合規說明文件，並附非 MLM 版本（weighted_quarter 0.9122）作保險。
- 開工前必要程式改動（需測試與程式碼審查後才可訓練）：(1) `src/mlm.py` 新增使用全部 dataset 文字的語料模式；(2) `check_init_from` 目前會拒絕續訓語料與驗證集重疊，需新增「明確指定才放行」的旗標，並寫入 run_config，避免被誤當嚴格 OOF；(3) `tools/run_cv5_mlm.sh` 或新腳本改為只續訓一次、5 折共用；(4) 確認 `tools/refresh_experiments.py` 相容。

**結果與判定（2026-11-19，[報告](reports/CV5_MLMAll30_Report.md)）**
- 非嚴格 OOF（續訓看過各折驗證文字，偏樂觀，只作參考）：Macro-F1 90.21%（mlm10 為 89.63%）；對 mlm10 配對比較 Δ +0.58 個百分點，95% 區間 [−0.05, +1.16]，mlmall30 勝 5/5 折，但此比較因洩漏偏向 mlmall30，不作判斷。
- Kaggle：fold 0/1/2 集成 0.9142（使用者回報，未經平台核驗）。依預先登記規則，< 0.9152 → **主版本維持 `cv5_sqrt_ep5_mlm10`（0.9152）**；mlmall30 不成為主版本，不延伸此線（更長續訓、更高遮罩率不做）。
- 限制：差距 0.10 個百分點，遠小於單次 Kaggle 比較雜訊約 ±1 個百分點，只能說「全文＋30 epochs 未見改善」，不能說變差；且一次改了語料範圍與 epoch 數兩個因素。

**交接（2026/11/19）**
- 已完成：方向 2 程式改動（131 測試 OK）、共用 MLM 續訓 30 epochs、5 折微調、OOF、配對比較、集成 CSV 上傳（Kaggle 0.9142）、registry／報告／總表登錄。
- 未完成：無預先登記的剩餘工作；後續方向未定。
- 下一個具體動作：與使用者討論後續方向（不自行決定）。
- 未 commit 的工作：git status 中所有 HW3 改動仍未 commit。

### 方向 3 預先登記：full8100_mlm10（2026-11-19 草稿，未開工）

狀態：**已準備、未開工，等使用者指示開工。** 腳本 `tools/run_full_mlm.sh`、集成工具 `tools/ensemble_checkpoints.py` 與測試 `tests/test_ensemble_checkpoints.py` 已備妥，尚無任何訓練或推論產物。

- 背景：`cv5_sqrt_ep5_mlm10`（Kaggle 0.9152，目前最高）為五折模型，每個只用約 7,200 筆訓練文字與標籤；順序 3 預先登記（`cv5_sqrt_ep5_mlm10`）已寫「最終交件模型只用原訓練子集 8,100 筆文字續訓」，此設定尚未做過。Kaggle 即成績；只換 seed Kaggle 曾差 1.7 個百分點；使用者同意方法用盡後可用 Kaggle 在已登錄候選間挑選。
- 假設：同 mlm10 設定，訓練資料由約 7,200 增為 8,100，加上多 seed，可使至少一個候選 Kaggle ≥ 0.92（推測，未驗證）。
- 設定：切分 `outputs/eda/split.json`（train 8,100／validation 900，與 `weighted_quarter` 相同）；每個 seed 各自 MLM 續訓 10 epochs（只用 8,100 筆訓練文字；lr 5e-5、遮罩 15% 80/10/10、warmup 6%、weight decay 0.01、batch 2、累積 8、儲存最後 epoch），再 sqrt-inverse 微調 5 epochs，以 900 驗證挑最佳 epoch；seeds 42、43、44（MLM 與微調用同一 seed）。
- 對照：mlm10 集成 0.9152（主版本）、`weighted_quarter` 0.9122（保險）。
- 預算：3 次 MLM 續訓＋3 次微調，不追加。估計每個 seed MLM 約 68 分鐘（依 mlmall30 每 epoch 450 秒 × 8,100/9,000 推算）、微調約 36 分鐘，合計約 5.2 小時（推測，以實測為準）。
- 指令：`bash tools/run_full_mlm.sh full8100_mlm10 10 5 42 43 44`，log `outputs/full8100_mlm10_train.log`。建議先只跑 seed 42（`... 10 5 42`）確認流程與耗時；3 個 seed 全部完成後腳本自動呼叫 `python -m tools.ensemble_checkpoints`，輸出 `outputs/full8100_mlm10_ensemble_seed42-43-44/`（單模型與 3 模型等權 softmax 集成 CSV、SHA-256、`summary.json`）。只跑 seed 42 時會產生 `outputs/full8100_mlm10_ensemble_seed42/`（1 模型，僅供流程檢查，不上傳）。
- Kaggle 上傳（預先固定上限 4 次）：3 個單模型 CSV ＋ 1 個 3 模型等權集成 CSV，全部由使用者上傳回報，之後不再為此實驗追加上傳。
- 判定（預先固定）：任一候選 ≥ 0.92 → 取 Kaggle 最高者為主要候選；最高者介於 0.9152 與 0.92 → 該者成為新主版本，但目標未達、不再延伸此線（另與使用者討論）；全部 ≤ 0.9152 → 主版本維持 mlm10。900 驗證分數只用來挑 epoch 與記錄，不作判定。
- 限制：900 驗證與 Kaggle 排序曾不一致；單次 Kaggle 雜訊約 ±1 個百分點；從 4 個上傳挑最高是以 Kaggle 選擇（Kaggle 即成績，使用者已同意），但分數本身不代表泛化提升。
- 合規：只用課程提供資料的訓練文字，不用 benchmark、不用 validation 文字續訓（`check_init_from` 嚴格檢查，不加 `--allow-mlm-val-overlap`）；講義 p.9「base model」續訓風險使用者已於 11/17 接受；若最終主版本為 MLM 版，依交件規則附說明文件與非 MLM 保險版 `weighted_quarter`。集成 3 個模型符合上限。
- 尚未登錄：registry、實驗總表、報告均待有產物後再寫，屆時執行 `refresh_experiments.py`。

## 新研究路線（2026/11/16 起，取代原第三、四階段）

> 註：步驟 3 後續（label smoothing／R-Drop）與步驟 4 已由上方「路線調整（2026-11-16，第二次）」取代；步驟 1、2、3a、3b 維持有效。

**動機。** 900 筆驗證集雜訊大：neutral 約 133 筆，幾筆錯判即可改變排名。驗證與 Kaggle 的排序也不一致（已觀察）：weighted_sqrt 驗證 92.65%／Kaggle 0.9049，weighted_quarter 驗證 91.91%／Kaggle 0.9122。因此先讓評估可靠，再找方法。「benchmark 類別比例與訓練不同」目前只是推測，尚無證據。

| 步驟 | 內容 | 要回答的問題 | 進入下一步的判斷 |
|---|---|---|---|
| 1 logit 偏移校正（不訓練） | 在 weighted_sqrt、weighted_quarter checkpoint 的 FP32 logits 上計算 logits − τ·log π_train（π 為訓練集類別比例）；τ 取預先固定的網格，只用 900 驗證選 τ；τ=0 必須逐位元重現原 CSV；不依 Kaggle 回饋反覆微調。至多產出 3 份有意義的 CSV，由使用者自行提交 | 類別先驗偏移是否為驗證與 Kaggle 不一致的原因之一 | Kaggle 隨 τ 呈一致方向變化，支持類別比例推測，步驟 2 起把 τ 校正納入固定後處理；否則記錄為無效並繼續 |
| 2 五折交叉驗證（**已完成 2026/11/16**，[報告](reports/CV5_Sqrt_Report.md)：OOF Macro-F1 89.03%，作為步驟 3 基準） | 在 9,000 筆上做分層五折（沿用既有文字群組分層邏輯，避免重複文字跨 fold）。以 OOF 9,000 筆 Macro-F1 及三類 F1 為之後所有比較的主要指標，原 900 驗證降為參考。可取 3 個 fold 模型做等權集成（課程上限 3 個模型）；各模型只在其訓練 fold 訓練、在其驗證 fold 選模，符合訓練／驗證切分規定 | 在較低雜訊的指標下，現行設定的真實水準與波動 | 先以 weighted_sqrt 設定建立 CV 基準，基準建立後才進入步驟 3 |
| 3 用 CV 重比設定（3a、3b 有效；label smoothing／R-Drop 後續**已由第二次調整取代**） | 用 CV 重比 sqrt 與 quarter 權重，再逐一試 label smoothing、R-Drop（對標籤雜訊較穩健，為假設）。每次只改一個因素 | 哪些改動在 OOF 上有穩定增益 | 看 OOF Macro-F1 差距與各 fold 是否一致，不只看平均 |
| 4 進階（有條件；**已由第二次調整取代**，改列為上方順序 2、3） | 用 OOF 預測找疑似錯標樣本並降權；僅用訓練集文字做 MLM 續訓（不得使用 benchmark 文字） | 標籤雜訊與領域適應是否還有空間 | 只有步驟 3 找到穩定增益後才考慮 |
| 不做 | 再跑 FGSM（另有 AMP overflow 會中止訓練的已知缺陷，見 `src/training.py` 的 `fgsm_delta`）；在單一 900 驗證集上繼續調參 | — | — |

**步驟 3a：CV 重比 sqrt vs quarter（預先登記，2026/11/16，訓練前寫下；已完成 2026/11/16，[報告](reports/CV5_Quarter_Report.md)）**

- 假設：quarter（較溫和的類別權重）在 OOF 上不劣於、或優於 sqrt。900 驗證上 sqrt 92.65% 對 quarter 91.91% 的差距，已被步驟 2 證明在雜訊範圍內，而 Kaggle 為 quarter 0.9122 > sqrt 0.9049。
- 對照：`cv5_sqrt`（OOF 89.03%）。唯一變因：class weighting sqrt-inverse → quarter-inverse；其餘設定、5 折切分、seed 42 全同（評估頻率維持每 epoch 一次，與 CV sqrt 相同）。
- 執行：`bash tools/run_cv5.sh cv5_quarter quarter-inverse`，再以 `python -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_quarter_oof --output <新 json>` 做配對 bootstrap（seed 20261116、2000 次）。
- 預算：5 次完整訓練（約 110 分鐘）；任一折 NaN 或 reload 驗證失敗即停止分析；最多因當機重跑 1 次。
- 決策規則（預先固定）：Δ = OOF(quarter) − OOF(sqrt)。配對 bootstrap 95% 區間下界 > 0 → quarter 為後續基準；上界 < 0 → sqrt 為後續基準；否則視為無顯著差異，後續基準取 OOF 較高者並標註「差異不顯著」。Kaggle 分數不用於此決策。
- 預期決策資訊：決定步驟 3 後續單因素實驗（label smoothing、R-Drop、epoch 數）的基準設定。
- **決策結果（2026/11/16）**：Δ（quarter − sqrt）= −0.23 個百分點（OOF 88.80% vs 89.03%），配對 bootstrap 95% 區間 [−0.71, +0.21] 含 0，依規則視為**無顯著差異**；後續基準維持 **sqrt**（OOF 較高者），標註「差異不顯著」。逐 fold quarter 勝 2/5。

**步驟 3b：訓練長度 3 → 5 epochs（預先登記，2026/11/16，訓練前寫下）**

- 假設：現行 3 epochs 訓練不足；5 epochs 的 OOF Macro-F1 高於 3 epochs。依據：步驟 2／3a 共 10 折中 8 折最佳 epoch 為最後一個。
- 對照：`cv5_sqrt`（3 epochs，OOF 89.03%）；唯一變因：`--epochs 3 → 5`。附註：此因素本身連帶改變學習率排程長度（warmup 步數與線性衰減跨度隨總步數變長），屬「訓練長度」因素的一部分，不另拆；其餘設定、5 折切分、seed 42、class weighting sqrt-inverse、eval-interval 1.0 全同。
- 指令：`bash tools/run_cv5.sh cv5_sqrt_ep5 sqrt-inverse 5`；比較 `python -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_sqrt_ep5_oof --output outputs/cv5_compare_sqrt_vs_sqrt_ep5.json`。
- 預算：5 次完整訓練，每次約 1.7 倍時間（合計約 3 小時，依先前每折 16–33 分鐘推估）；任一折 NaN／reload 驗證失敗即停止分析；最多因當機重跑 1 次。
- 決策規則（同 3a，預先固定）：Δ = OOF(ep5) − OOF(ep3)；配對 bootstrap 95% 區間下界 > 0 → ep5 為後續基準；上界 < 0 → 維持 ep3；否則視為不顯著，基準取 OOF 較高者並標註。Kaggle 分數不用於決策。
- 附加觀察（不作決策依據）：各折最佳 epoch 分布；若 ep5 多數折最佳 epoch 仍為 5，記為「可能仍訓練不足」；若多落在 3–4，記為訓練長度已足。
- 已知限制：最佳 epoch 從 5 個候選中挑，OOF 的選模樂觀偏誤大於 3 個候選，比較略偏向 ep5，報告需註明。
- **決策結果（2026/11/16，[報告](reports/CV5_Sqrt_Ep5_Report.md)）**：Δ（ep5 − ep3）= +0.19 個百分點（OOF 89.22% vs 89.03%），配對 bootstrap 95% 區間 [−0.38, +0.76] 含 0，依規則視為**無顯著差異**；後續基準改為 **cv5_sqrt_ep5**（OOF 較高者），標註「差異不顯著」。逐 fold ep5 勝 3/5；最佳 epoch 分布 2、3、5、5、5（依登記記為「可能仍訓練不足」，但第 3 → 5 epoch 增幅小、曲線大致平坦，訓練長度可能不是主要瓶頸，屬推測）。+0.19 小於且同向於選模偏誤，保守解讀為訓練長度 3 → 5 無可偵測效果。

**步驟 3c：label smoothing ε = 0.1（預先登記，2026/11/16，訓練前寫下；已由第二次調整取代，`cv5_sqrt_ep5_ls01` 於 fold 0 中止、未產生結果，不執行下列決策規則）**

- 假設：訓練資料可能有標籤雜訊（neutral ↔ negative 混淆最多），label smoothing 降低模型對錯標樣本的過度自信，提升 OOF Macro-F1，尤其 neutral F1。
- 對照：`cv5_sqrt_ep5`（OOF 89.22%）；唯一變因：`--label-smoothing 0 → 0.1`（ε 預先固定為常用值 0.1，不做網格搜尋）；其餘設定（sqrt-inverse、5 epochs、eval-interval 1.0）、相同 5 折、seed 42 全同。
- 與類別權重的組合方式：每筆樣本 loss_i = w[y_i] × CE_ε(logits_i, y_i)，CE_ε 為未加權的 `F.cross_entropy(..., label_smoothing=ε)`（即 (1−ε)·NLL(y_i) + ε·mean_j NLL(j)）；權重只依真實類別套用，之後沿用「加權和／實際視窗樣本數」正規化與梯度累積。刻意不使用 PyTorch 的 `weight=` 與 `label_smoothing=` 組合（該組合會把平滑項依各目標類別加權）。只影響訓練 loss；評估 loss 仍為未加權、未平滑 CE，Macro-F1 與選模邏輯不變。ε = 0 時與既有實作逐位元相同。
- 指令：`bash tools/run_cv5.sh cv5_sqrt_ep5_ls01 sqrt-inverse 5 --label-smoothing 0.1`；比較 `python -m tools.compare_oof --a outputs/cv5_sqrt_ep5_oof --b outputs/cv5_sqrt_ep5_ls01_oof --output outputs/cv5_compare_ep5_vs_ep5_ls01.json`。
- 預算：5 次完整訓練（約 2.5 小時）；任一折 NaN／reload 驗證失敗即停止分析；最多因當機重跑 1 次。
- 決策規則（同 3a／3b，預先固定）：Δ = OOF(ls0.1) − OOF(ep5)；配對 bootstrap 95% 區間下界 > 0 → ls0.1 成為後續基準；上界 < 0 → 維持 cv5_sqrt_ep5；否則視為不顯著，基準取 OOF 較高者並標註。Kaggle 分數不用於決策。
- 附加觀察（不作決策依據）：neutral F1 的 Δ；各折最佳 epoch 分布。label smoothing 會改變訓練／驗證 loss 的數值與機率校準，但選模依 Macro-F1，不受影響。
- 已知限制：單 seed、只測單一 ε 值；此比較約只能偵測 ≥0.5 個百分點的效果。

**彈性調整規則。** 每步結束先檢視結果再決定下一步；可依證據調換順序、跳過、合併或新增步驟。不設時程；訓練次數上限為每個 CV 比較組 5 次、每次只改一個因素（見調整紀錄）。仍遵守課程規則：只用提供資料、只用 bert-base-uncased、集成最多 3 個模型、不用 benchmark 標籤或文字訓練。每次調整都要寫進下方調整紀錄。

**調整紀錄**

| 日期 | 調整 | 原因 |
|---|---|---|
| 2026/11/16 | 建立新研究路線，取代原第三、四階段 | 驗證雜訊大且與 Kaggle 排序不一致 |
| 2026/11/16 | 步驟 1 完成（[報告](reports/Logit_Adjust_Report.md)）：類別先驗推測證據弱，下一步進入步驟 2；τ 校正暫不納入固定後處理 | benchmark 預測比例約 73／13／14%，接近訓練比例；weighted_sqrt 最佳 τ=0；weighted_quarter τ=+0.4 驗證 +0.50 個百分點（約 4 筆，在雜訊內），benchmark 只改 3 筆。sqrt 與 quarter 的 11 筆差異多數不是先驗偏移造成。若使用者回報 τ=+0.4 的 Kaggle 分數有明顯變化再重新評估 |
| 2026/11/16 | 步驟 2 完成（[報告](reports/CV5_Sqrt_Report.md)）。(a) 訓練次數上限調整：原「核心上限 9 次完整訓練」已用 5 次，本步再用 5 次（合計 10 次）；該上限屬已被新研究路線取代的舊計畫，且使用者指示開始步驟 2，故調整。之後步驟 3 的新上限為「每個 CV 比較組 5 次訓練，每次只改一個因素」。(b) OOF 成為主要指標，原 900 驗證降為參考 | 單一 900 驗證雜訊大；OOF 9,000 筆 Macro-F1 89.03%、fold 標準差約 1.98 個百分點，原 900 筆在 OOF 中為 91.28%（較整體容易）。步驟 3 以相同 5 折、相同設定只改 class weighting 為 quarter-inverse |
| 2026/11/16 | 步驟 3 開始：先做步驟 3a（sqrt vs quarter 的 CV 重比，預先登記見上）。評估頻率刻意維持 eval-interval 1.0 | 與 `cv5_sqrt` 保持單一變因；原單模型 weighted_quarter 用 0.5，這裡不跟，否則 class weighting 與評估頻率兩個因素混在一起 |
| 2026/11/16 | 步驟 3a 完成（[報告](reports/CV5_Quarter_Report.md)）：sqrt 與 quarter 在 OOF 上無顯著差異，後續基準維持 sqrt（差異不顯著）。步驟 3 的下一個單因素實驗尚未決定 | Δ −0.23 個百分點、95% 區間 [−0.71, +0.21]，逐 fold 2/5 勝；兩邊 10 個 fold 中 8 個最佳 epoch 為最後一個（訓練不足為推測） |
| 2026/11/16 | 步驟 3b 開始：使用者同意把訓練長度實驗（sqrt 基準上 epochs 3 → 5）提前於 label smoothing／R-Drop（預先登記見上）。label smoothing／R-Drop 延後 | 步驟 2／3a 共 10 折中 8 折最佳 epoch 為最後一個（第 3 個），曲線多仍上升，訓練不足是最有證據的下一個因素 |
| 2026/11/16 | 步驟 3b 完成（[報告](reports/CV5_Sqrt_Ep5_Report.md)）：ep5 與 ep3 在 OOF 上無顯著差異，後續基準改為 cv5_sqrt_ep5（OOF 較高者，差異不顯著）。下一個單因素實驗尚未決定 | Δ +0.19 個百分點、95% 區間 [−0.38, +0.76]，逐 fold 3/5 勝；步驟 3a 與 3b 的差距都在約 ±0.5 個百分點內，單 seed CV 只能偵測約 0.5 個百分點以上的效果 |
| 2026/11/16 | 步驟 3c 開始：新增 label smoothing（ε = 0.1）單因素實驗，基準為 cv5_sqrt_ep5（預先登記見上）；新增 `--label-smoothing` 參數與 `run_cv5.sh` 額外參數轉傳 | 使用者選定下一個單因素實驗；步驟 3a／3b 差距都在約 ±0.5 個百分點內，訓練長度與類別權重已無明顯可調空間，改測標籤雜訊假設 |
| 2026/11/16（第二次） | 第二次路線調整（見文件開頭「路線調整（2026-11-16，第二次）」）：取代步驟 3 後續（label smoothing／R-Drop）與步驟 4；改採停損規則（只排預期 ≥0.5 個百分點的方向、每方向至多 1 組 CV、OOF Δ < +0.5 個百分點即停），順序為 0 上傳 cv5_sqrt_ep5 集成 CSV、1 多 seed 43、2 錯標降權、3 MLM 續訓、4 備選；全部未達門檻則交件採 weighted_quarter（0.9122）並補封裝演練。使用者於當日同意 | 近期單因素實驗預期效果約 0.1–0.3 個百分點，低於單 seed 五折 CV 約 0.5 個百分點的偵測力；CV 線（OOF 89.22%）從未回 Kaggle 驗證，OOF 與 Kaggle 對應未知；原計畫無停損規則與退路 |
| 2026/11/16（第二次） | `cv5_sqrt_ep5_ls01`（label smoothing 0.1、sqrt-inverse、5 epochs）於 fold 0 訓練中途經使用者同意中止，未產生 checkpoint 或指標；`outputs/cv5_sqrt_ep5_ls01/` 與 `outputs/cv5_sqrt_ep5_ls01_train.log` 保留未刪；`saved_models/cv5_sqrt_ep5_ls01/fold0/` 為空資料夾，同樣保留未刪，不列入任何比較，也不計入 registry 的 runs | 配合第二次路線調整，label smoothing 降為備選 |
| 2026/11/16（第二次） | 順序 2 錯標抽查完成（[報告](reports/Label_Noise_Audit_Report.md)）：依預先登記規則通過（C 組一致率 100% ≥ 80%；A+B annotation ≠ label 36/60 = 60% ≥ 50%；\|H\| = 473 ≥ 90）。不訓練的後處理分析，不寫進 registry runs | 抽樣 60 筆小樣本；只計盲標 high confidence 約 32%；A 組 14/40 為盲標亦認為 neutral 的單純詢問，推測模型對 neutral 問句有系統性負面偏誤 |
| 2026/11/16（第二次） | `cv5_sqrt_ep5_drop90` 預先登記（見「順序 2 預先登記」，訓練前寫下）：使用者選擇排除版本（w=0），排除 473 筆高信心不一致樣本，唯一變因為訓練子集排除 | 抽查通過；排除比降權更單純；決策門檻 OOF Δ ≥ +0.5 個百分點 |
| 2026/11/16（第二次） | 追加檢查完成（neutral 問句的高信心誤判，[報告](reports/Label_Noise_Audit_Report.md)）：模型對問句的 neutral recall 反而較高，負面偏誤推測不成立；drop90 風險較原估計低，預先登記不變。同時修正抽查計分腳本 ambiguous 欄定義（其餘數字不變） | 不訓練的後處理分析；產物 `outputs/question_bias_check_20261116/` |
| 2026/11/17（第二次） | `cv5_sqrt_ep5_drop90` 完成（[報告](reports/CV5_Drop90_Report.md)）：OOF 89.34% vs 對照 89.22%，Δ +0.12 個百分點（95% [−0.49, +0.76]，fold 勝 3/5），未達預先登記門檻 +0.5 → 停止錯標排除方向、不提交 Kaggle；下一步順序 3（MLM 續訓）待設計。已登錄 registry `cross_validations` | 排除後 neutral recall 79.3% → 77.3%，推測排除也移除部分標籤正確的 neutral 樣本；單 seed CV 只能偵測約 0.5 個百分點以上的效果 |
| 2026/11/17（第二次） | `cv5_sqrt_ep5_mlm10` 預先登記（見「順序 3 預先登記」，訓練前寫下）：每折僅用該折訓練子集文字做 MLM 續訓 10 epochs 後再微調，唯一變因為微調初始權重。使用者於當日知悉講義 p.9 對「base model」未明確界定預訓練／續訓、合規為主會話推測而非講義明文，判斷風險可接受並決定執行（使用者決定，非講義確認）；若之後助教回覆不允許，此方向產物不得用於交件 | 任務適應預訓練（TAPT）預期 +0.3～1 個百分點（推測）；決策門檻 OOF Δ ≥ +0.5 個百分點 |
| 2026/11/17（第二次） | 新增 MLM 版本交件規則（使用者指示，寫入「順序 3 預先登記」交件影響）：若最高分為 MLM 續訓版本，交件附合規說明文件，並附第二高分非 MLM 版本（checkpoint、推論程式、CSV）作保險，README 標示主要／保險版本；E3 壓縮檔格式與大小限制、Kaggle 最終提交可選筆數待確認 | 僅文件規則，不涉訓練；降低助教不接受 MLM 續訓的風險 |
| 2026/11/17（第二次） | 備選清單新增 pooler penalized tanh（使用者提議，源自課堂介紹），排在 MLM 續訓之後、與 LS／R-Drop 同層；未預先登記、未實作 | 只動分類前最後一段，分類頭激活通常不是瓶頸，預期效果小（推測）；實作成本低 |
| 2026/11/17（第二次） | `cv5_sqrt_ep5_mlm10` fold 4 MLM 續訓於第 4 epoch 崩潰（`AssertionError: No inf checks were recorded for this optimizer`）：fold 4 訓練集 7,201 筆（奇數），batch 2 → 最後一個梯度累積窗口只有 1 筆，該筆抽到零遮罩，無 backward 卻呼叫 `scaler.step`。修正 `src/mlm.py`（整個窗口無 backward 時跳過 optimizer／scheduler 更新並記錄 `empty_windows`），新增 2 個回歸測試（共 118 OK）；只重跑 fold 4，崩潰產物保留於 `outputs/cv5_sqrt_ep5_mlm10_mlm_fold4_crashed_20261117/` | fold 0–3 每窗口皆完整 8 個 batch、從未進入該路徑，修正不影響其行為，故不重跑；fold 4 的 `training_code_sha256` 與 fold 0–3 不同，summary 的 `empty_windows` = 1（見[報告](reports/CV5_MLM10_Report.md)） |
| 2026/11/17（第二次） | `cv5_sqrt_ep5_mlm10` 完成（[報告](reports/CV5_MLM10_Report.md)）：OOF 89.63% vs 對照 89.22%，Δ +0.41 個百分點（95% [−0.21, +1.06]，fold 勝 3/5），未達預先登記門檻 +0.5 → 不列候選、不提交 Kaggle，為目前 OOF 最高的 CV 組；後續（備選或退路）待與使用者討論。已登錄 registry `cross_validations` | Δ 方向為正但區間跨 0，單 seed CV 偵測力約 0.5 個百分點，無法區分真增益與雜訊 |
| 2026/11/17（第二次） | 使用者上傳 mlm10 集成 CSV，Kaggle 0.9152，高於 weighted_quarter 0.9122，成為目前最高；依使用者交件規則（見「順序 3 預先登記」交件影響），MLM 版本為主要候選，須附合規說明文件，並以 weighted_quarter（0.9122）為保險版本 | 偏離預先登記（OOF 未達門檻仍提交）由使用者決定；Kaggle 與 OOF 差距受 benchmark 雜訊影響（seed 變異約 1.7 個百分點） |
| 2026/11/17（第二次） | 使用者說明 Kaggle 無 public／private 之分、目標為最終一個 > 0.92 的模型，方法用盡後可用 Kaggle 在已登錄候選間挑選；使用者選擇方向 2（cv5_sqrt_ep5_mlmall30，見「方向 2 預先登記草稿」），指示先不開工 | mlm10 是唯一在 OOF 與 Kaggle 皆為正的方向；以 Kaggle 為最終判斷後，嚴格 OOF 比較的價值下降，方向 2 只需續訓一次、總時間約 6 小時（方向 1 每折各自續訓 30 epochs 約 15 小時） |
| 2026/11/18（第二次） | 方向 2 `cv5_sqrt_ep5_mlmall30` 預先登記經使用者確認（設定、判斷規則不變，見「方向 2 預先登記」）；開工前新增：`pretrain-mlm --mlm-corpus all-dataset`、`train --allow-mlm-val-overlap`（寫入 run_config 標示非嚴格 OOF）、`tools/run_cv5_mlm_shared.sh`（續訓一次、5 折共用） | 使用者確認；以 Kaggle fold 0/1/2 集成分數判斷，OOF 只參考 |
| 2026/11/19（第二次） | 方向 2 `cv5_sqrt_ep5_mlmall30` 完成（[報告](reports/CV5_MLMAll30_Report.md)）：非嚴格 OOF 90.21%（偏樂觀，只參考），Kaggle 0.9142（使用者回報）< 0.9152 → 依預先登記判斷主版本維持 mlm10，不延伸此線；一次改兩個因素且差距遠小於 Kaggle 雜訊，只能說未見改善；下一步待與使用者討論 | 預先登記規則：< 0.9152 主版本維持 mlm10 |
| 2026/11/19（第二次） | 訓練加速等價性實測（speed_check）預先登記（訓練前寫下）。**目的**：量測加速設定的速度效益與數值等價性；不是為了提高分數，不產生交件候選。**新增程式**：`--eval-batch-size`（預設 None＝沿用 `--batch-size`，不指定時行為不變）與計時拆解（history 每列 `train_seconds`／`eval_seconds`；summary `total_train_seconds`、`total_eval_seconds`、`peak_allocated_mb`、`peak_reserved_mb`）。**共同設定**：seed 42、`--split outputs/eda/split.json`、`--class-weighting sqrt-inverse`、`--epochs 1`、不用 `--init-from`；輸出 `outputs/speed_check/<組別>`、checkpoint `saved_models/speed_check/<組別>`。**組別（每組只比上一組多一個變因）**：A 現行設定（對照，batch 2、累積 8、gradient checkpointing 開）；B = A + `--eval-batch-size 64`；C = B + `--no-gradient-checkpointing`；D = C + `--batch-size 8 --accumulation-steps 2`（等效 batch 16 不變）；E = C + `--batch-size 16 --accumulation-steps 1`（僅在 D 峰值 reserved VRAM < 3,000 MiB 時才跑）。MLM：M-A 現行 `pretrain-mlm` 1 epoch；M-C = M-A + `--no-gradient-checkpointing --eval-batch-size 64`；M-D = M-C + `--batch-size 8 --accumulation-steps 2`（同樣條件式：M-C 峰值 reserved < 3,000 MiB 才跑）。**假設**：B 的訓練過程與 A 位元相同，只有評估更快，驗證 macro-F1 差 ≤ 0.002；C 數值近乎相同（驗證 macro-F1 差 ≤ 0.005），訓練段加快 20～35%；D／E 因 dropout 亂數與 padding 不同而非等價，只記錄速度與分數差，不據單次 1 epoch 結果判定分數優劣。**決策資訊**：哪些設定可作為後續長時間訓練的預設；監控腳本的新速度門檻。**上限與停止條件**：全部合計 ≤ 60 分鐘 GPU 時間；任一組 OOM 或 reserved VRAM ≥ 3,900 MiB 即放棄該組及更大 batch 組。**注意**：MLM 的驗證遮罩在固定 RNG 下以評估 batch 物化，改 `--eval-batch-size` 會改變驗證遮罩，M-C 的 val MLM loss 與 M-A 不能逐位比較（只作量級參考）；`tools/cv_oof.py` 的 BATCH_SIZE=2 為 OOF 推論設定，未更動。 | 目前每個 CV 折訓練與 MLM 續訓耗時長，預先量測可安全套用的加速設定，以決定後續長時間訓練的預設 |
| 2026/11/19（第二次） | 訓練加速等價性實測（speed_check）結果摘要（[報告](reports/Speed_Check_Report.md)）。**微調**：B（`--eval-batch-size 64`）、C（再加 `--no-gradient-checkpointing`）與 A 的權重 SHA-256 相同、驗證混淆矩陣相同（位元等價），1 epoch 由 467 秒降到 203 秒（約 2.3 倍），可作為後續微調預設；D／E（batch 8／16）約 3.6～4.3 倍但非等價，F1 差異在單次雜訊內，列為待登記的後續實驗。**MLM**：M-C 的 train MLM loss 與 M-A 相同、224.5 對 334.6 秒，但 peak reserved 4,166 MiB 超過停止條件 3,900 MiB（推測來自評估 batch 64 的詞表 logits，未驗證），M-D 跳過；MLM 可採用 `--no-gradient-checkpointing`，評估 batch 須改小並另測。監控腳本 `tools/gpu_watch.sh` 已依實測校正門檻。本輪不是訓練 run，未寫入 registry。 | 實測後確定長時間訓練的加速預設與監控門檻 |

## 現況與保留版本

完整分數只維護於 [實驗總表](Experiment_Results.md)，來源為 [registry](../experiments/registry.json) 與各輪原始 summary。已盤點 12 個完整訓練、2 組集成，另有短跑及資料盤點。

保留 `weighted_sqrt` 為驗證基準，`weighted_quarter` 為目前 Kaggle 最高分候選；原候選 `weighted_lr3e5` 與同分集成均保留，新候選尚未封裝演練。`weighted_half_epoch_control` 是後續新方法共同控制組，最佳權重與 CSV 和 `weighted_sqrt` 相同。已建立 [檔案雜湊與候選清單](../experiments/snapshot.json)，對應 14 份 CSV 與 12 份 checkpoint metadata；推論證據見各輪報告。`saved_models/checkpoint/` 仍是早期基準，交件時應從最終候選複製至獨立封裝目錄，不能誤當作最佳模型。

既有小範圍學習率搜尋已完成；URL、emoji、另一 seed 與兩組集成都未帶來已知最高分突破。停止擴大這些方向的盲目搜尋。neutral 為最弱類，但 benchmark 無標籤，不能假定 Kaggle 失分也全來自 neutral。0.78 個百分點 Macro-F1 不能直接換算成固定需修正幾筆。

## 方向評估與執行優先順序

目前瓶頸是提升能否在未見資料上重現，以及交件版本是否可靠。從目前兩個 seed 的差異與低互補集成結果，還沒有足夠證據支持擴大 seed 或集成搜尋。以下排序是依本機結果與實作成本的判斷，不是已證實的性能提升：

| 優先度 | 工作 | 理由與完成條件 |
|---|---|---|
| P0 已完成 | 對兩個保留候選執行 AMP／FP32 驗證比較 | 11/15 完成 8 次推論，驗證與 benchmark 標籤皆一致；保留 AMP 驗證／FP32 推論，未發現精度導致分數差距的證據 |
| P0 已完成 | 先做候選的獨立封裝與解壓推論演練 | 11/15 已完成 `weighted_lr3e5` ZIP、新 venv 安裝與離線推論，原 CSV 完全重現；保留演練包；新候選 weighted_quarter 尚未演練，正式版本／學號仍待定 |
| P1 已完成 | 分層學習率 B | 驗證整體與三類 F1 均下降；Kaggle 回報亦未改善，保留對照，不擴大 decay 搜尋 |
| P2 | 有希望的候選做額外切分確認 | 確認泛化方向，避免只迎合目前 900 筆驗證集 |
| P3 已完成 | FGSM C1／C2 | 兩者驗證未改善，停止本輪搜尋；C2 Kaggle 回報未超越首選 |
| 暫停 | 更多 URL／emoji 規則、盲目 seed、集成權重搜尋 | 已做設定未提供突破證據；只有新證據出現才重新打開 |

PyTorch 的 [AMP 文件](https://docs.pytorch.org/docs/2.9/amp.html) 說明 autocast 會按運算選擇精度；這支持檢查兩條路徑，但不證明本模型的預測實際不同。當前判斷不涉及替換指定基底或新增外部訓練資料。

## 第一階段：確認評估可信，凍結比較方式（11/14–11/15）

**（2026/11/16 起，下一項改以上方「新研究路線」為準；weighted_quarter 封裝演練不列為研究優先。）** C1／C2 不進入額外切分確認；預定確認切分仍保留。完整訓練預算已用 **5／9**，詳細結果見 [C2 報告](reports/FGSM_High_Report.md)。

1. **已完成**：在兩個保留 checkpoint 上比較 AMP／FP32 標籤、Macro-F1、機率與 runtime，重現歷史驗證與原提交 CSV。標籤全相同，保留現行精度路徑；若未來候選出現標籤差異，再統一其比較與推論方式。
2. **已完成檢查**：optimizer／scheduler、梯度累積、加權 loss、checkpoint 選取與 ID／類別對齊未發現需先重跑的既有行為 bug。保留原 optimizer／loss／schedule 策略，補記 AMP skip 與 scheduler 步數；27 項測試通過。
3. **已完成**：每 0.5 epoch 對齊累積窗口評估與更新最佳 checkpoint，已跑一次 3 epochs 的 2e-5 控制組。最佳仍為 epoch 3，權重與 CSV 和舊控制組相同，不重複提交。後續方法使用相同評估頻率；額外選模機會不能直接宣稱對舊流程公平提升。
4. 實驗總表、config／來源 hash／驗證／Kaggle／CSV hash 已建立。跨精度檢查已輸出各類 precision／recall／F1、錯判數、信心與機率，並以獨立 audit 登錄；Emoji 分數已於 2026/11/15 回報並登錄於實驗總表，未改善最高分，維持目前優先順序。使用 `python tools/refresh_experiments.py --check` 確認總表未過期。

## 第二階段：至多四個新設定，改善泛化（11/15–11/19）

均使用原空白前處理、原 URL、seed 42、固定 8,100／900 切分、相同有效 batch 與新控制組評估頻率。每次只改一個主要因素，不組合失敗方法。

| 順序 | 具體實驗 | 要回答的問題 | 上限 |
|---|---|---|---|
| A 已完成 | 將類別權重由 `n^-0.5` 改為較溫和的 `n^-0.25`，仍正規化為訓練樣本平均權重 1；其餘同控制組 | 本機整體與三類 F1 均下降，Kaggle 回報提升；保留候選，不追加權重搜尋 | 已用 1 個設定 |
| B 已完成 | AdamW 分層學習率：最上 encoder 層 2e-5，每往下一層乘 0.9，分類 head 2e-5；其餘同控制組 | 本機整體與三類 F1 均下降，未支持改善；Kaggle 回報亦下降，不追加 decay 搜尋 | 已用 1 個設定 |
| C1／C2 已完成 | 兩個預先固定 epsilon 的 FGSM | 兩者整體與三類驗證 F1 均下降，停止幅度搜尋 | 已用 2／2 個設定 |

上述是待驗證假設，不預先承諾哪個會提高 Kaggle。FGSM 的兩個 epsilon 已依固定 train 子集（48 筆，每類 16 筆）非 padding／特殊 token 的 word embedding RMS 之 5%／10% 固定並寫入 smoke config，詳細值與精度診斷見校準報告；未使用 benchmark 回饋選擇擾動幅度。擾動界限、mask、有限梯度、AMP／累積縮放、乾淨推論與權重不變性已檢查。每輪仍從同一 BERT 基底重新訓練。註：真實 BERT 的 AMP／FP32 梯度相對 L2 < 2% 數值等價檢查**未通過**（dropout 開啟時在首個 batch 停止；關閉 dropout 的診斷 batch 0 為 5.3756%，`precision_relative_l2_under_2pct=false`，見[校準報告](reports/FGSM_Calibration_Report.md)）；`tests/test_fgsm.py` 的 CUDA 測試（AMP 對 FP32 相對 L2 < 2%）只用 tiny BERT、dropout=0，屬實作正確性檢查，不等於真實 BERT 梯度數值等價。

FGSM 概念依據：[Goodfellow 等，Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)。論文支持方法來源，不是本作業能改善分數的證據；此處採 embedding 上的適配，須實測。不要將 L2 正規化梯度的 FGM 實作誤標為 FGSM。

以新控制組為基準：Macro-F1 增益 **≥ 0.3 個百分點**為優先確認候選；0–0.3 視為待確認小增益，不稱為突破。這是資源分配門檻，不是統計顯著性。若 neutral 提升但整體下降，記錄權衡，不只報最好看的類別。四個設定均無穩定優勢就停止本輪搜尋，先檢討證據，不自動再加大量參數。

## 第三階段：確認候選，而非繼續迎合原驗證集（11/19–11/23）

（2026/11/16 起由新研究路線取代，保留原文供對照）

第一階段已預先固定兩份額外的 90／10 分層文字群組切分（seed **142、242**），保存 ID 與 hash 於 registry／snapshot；兩份各為 8,100／900，尚未訓練。只讓第二階段最有希望的一個設定與新控制組各跑兩份，**共 4 次訓練**。兩者使用相同訓練 seed 42、切分、評估頻率及計算預算。

檢查三份切分的配對分數差、平均差、三類 F1 與最差切分。候選至少在兩份切分改善且平均改善，才優先進入後續；若方向相反或微小，明確標為不穩定。切分可能互相重疊，且這是同一資料集，不是假裝取得全新獨立測試集，也不將結果當成顯著性證明。不得反覆更換切分直到候選勝出。

若第二階段無候選，這四次確認訓練暫不執行；先回顧錯判、規則與預算。不可只因原驗證已超過 92%，就認為 Kaggle 目標完成。

## 第四階段：有條件集成與最後資料利用（11/23–11/26）

（2026/11/16 起由新研究路線取代，保留原文供對照）

- 只挑經確認的新候選與舊強模型，先在**相同未參與其訓練的樣本**比較互補；試一組預先指定的等權組合，最多三個模型。不平均各模型在自身訓練樣本上的分數，不把不同切分的 checkpoint 當成共用 holdout。
- 若集成沒有整體提升或誤判仍高度重疊，優先保留單模型；仍可按使用者偏好產生標示清楚的實驗提交檔。
- 2026/11/14 已核對講義第 14 頁：要求只在提供資料切出的訓練／驗證集合訓練及驗證，未明示免驗證的全資料重訓。**預設不執行全 9,000 筆重訓**；僅在課程規則另有明確允許時作額外候選。若允許，epoch／步數規則必須預先選定，不能拿已加入訓練的 900 筆再報驗證分數，並依全部訓練資料重算類別權重。既有最佳模型保留。
- 若不允許全資料重訓，直接使用符合規定的最佳驗證 checkpoint。始終不引入外部訓練資料、benchmark 標籤或換掉指定 BERT 基底。

## Kaggle 提交與決策

沿用使用者希望充分利用名額的偏好，正式實驗完成後均產生 1,000 筆合法 CSV，無論是否超過原驗證分數。每次記錄與目前最佳 CSV 的不同筆數；內容完全相同就不重複消耗提交名額。

按工作計畫記載的每 24 小時最多 3 次，實際剩餘額度以平台為準，不假設午夜重置。可提交三個**不同且有實驗意義**的候選：一個驗證優先候選、一個不同方法的探索候選、一個已確認設定的最終重訓／集成候選；沒有候選時不為填滿額度捏造變體。由使用者提交並回報，未授權工具代為提交。

Kaggle 作為外部結果紀錄，不逐筆猜 benchmark 標籤、不依 leaderboard 不斷微調權重或門檻。分數 ≥0.9200 時確認是課程採計欄位，保留對應 CSV、模型與設定，進入交件；不能將顯示分數當成尚未公布的最終榜保證。

## 預算、停止與交件（11/27–11/29）

（2026/11/16 起，下列次數上限已由新研究路線的調整紀錄取代，保留原文供對照。）核心上限：新控制組 1 次＋新設定 4 次＋額外切分確認 4 次＝**9 次完整訓練**；如規則允許，再加 1 次全資料重訓。是否執行後段由前段結果決定。依現有每輪約 17–23 分鐘，普通訓練估 20–30 分鐘、FGSM 粗估 30–50 分鐘，總計約 **4–6 GPU 小時**；這是估算，不含實作、除錯及機器降速，FGSM 需短跑後修正。

目前已完成 **5／9 次**：共同控制組、A、B、C1、C2。剩餘 4 次確認預算為有條件執行，暫不啟動；先完成目前 Kaggle 候選封裝演練。C2 結果與來源見 [報告](reports/FGSM_High_Report.md)。

達 92% 即停止非必要探索。11/26 前仍未達標則保留最佳已知候選，停止無上限搜尋；11/27–11/29 優先完成乾淨解壓推論、依賴與 checkpoint 打包、CSV hash 對應、學號檔名及 E3／Kaggle 選定版本核對。依現有工作計畫截止為 11/30 23:59，最後交件前再核對平台公告。

完成標準分開記錄：研究目標達成（Kaggle 採計 ≥92%）、實驗流程正確、交件可重現。若分數未達標，誠實記錄，不能把「已跑完計畫」寫成「已達92%」。
