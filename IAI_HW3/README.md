# IAI HW3｜Sentiment Analysis

在此目錄執行資料盤點。需 Python 3.10 以上，盤點使用標準函式庫。

```powershell
python main.py inspect
```

`dataset/` 內已有課程提供的 `dataset.csv` 與 `benchmark.csv`，講義位於 `docs/HW3.pdf`。如果從補充包建立工作區，將補充包內容解壓至既有 `IAI_HW3` 根目錄。實驗順序、分工及交件規格見 [HW3_Work_Plan.md](docs/HW3_Work_Plan.md)。

第一項 EDA 已完成。從本目錄執行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-eda.txt
.venv/Scripts/python.exe main.py eda
.venv/Scripts/python.exe -m unittest test_eda.py
```

第一次需要網路下載 tokenizer，之後使用 `outputs/eda/tokenizer/` 的本機副本。EDA 不需下載 BERT 模型權重或安裝 PyTorch；已以 Python 3.12 驗證。

預設 seed 42，依類別及空白正規化後的文字群組切分為 8,100／900 筆，避免重複文字跨集合。`outputs/eda/split.json` 保存 ID、類別數、seed 與資料 SHA-256；`train.csv`、`validation.csv` 保留原始文字。後續實驗應載入這份切分，不重新切分。若資料或切分參數改變，程式拒絕覆蓋既有切分，請用新的 `--output-dir`。

報告見 `outputs/eda/EDA_Report.md`，細部統計及 token 範例見 `outputs/eda/eda.json`。統計只使用訓練集；P50／P90／P95 為 31／40／43，建議 `max_length=64`（截斷率 0.02%），訓練時採批內動態 padding。基準前處理共用 `eda.preprocess`，只正規化空白。Emoji 偵測使用近似 Unicode 範圍，非完整 emoji 辨識。

## BERT 基準訓練與推論

已實作自訂 PyTorch 訓練／驗證迴圈及離線 checkpoint 推論。安裝環境與下載指定基底模型（在本目錄執行）：

RTX 3050 Laptop 的 GPU 短跑與完整 3 epochs 基準皆已完成。最佳為第 3 輪，固定驗證集 Macro-F1 **91.57%**；詳見 [完整基準報告](docs/Baseline_Report.md)與 [短跑報告](docs/Baseline_Smoke_Report.md)。最佳模型位於 `saved_models/checkpoint/`，1,000 筆預測位於 `outputs/baseline/predictions.csv`；使用者已提交並回報 Kaggle **0.8973**。完整環境已保存於 `requirements-lock.txt`；重建同版本環境時，先安裝下列 CUDA wheel，再用 `-r requirements-lock.txt` 取代 `-r requirements.txt`。

```powershell
.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe main.py prepare
```

CUDA 套件安裝方式依 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)。沒有 NVIDIA GPU 可改用 CPU wheel 的 `https://download.pytorch.org/whl/cpu`。`prepare` 只下載指定 revision 的 `google-bert/bert-base-uncased`；後續訓練與推論只讀本機檔案。

先執行短跑與程式測試：

```powershell
.venv/Scripts/python.exe -m unittest test_eda.py test_training.py
.venv/Scripts/python.exe main.py train --smoke --device cuda
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/smoke/checkpoint --predictions outputs/smoke/predictions.csv
```

短跑從固定切分的每一類各抽 16 筆，train／validation 各 48 筆，只跑 1 epoch，用於驗證 forward／backward、儲存、重載與推論。其 F1 不代表完整驗證集表現，預測檔不應提交 Kaggle。短跑預設保存在 `outputs/smoke/`，不會寫入交件 checkpoint。

完整基準指令（本機已完成此輪，重跑須指定新 output／checkpoint／predictions 路徑）：

```powershell
.venv/Scripts/python.exe main.py train --device cuda
.venv/Scripts/python.exe main.py predict --device cuda --predictions outputs/baseline/predictions.csv
```

預設 max length 64、micro-batch 2、梯度累積 8（有效 batch 16）、AdamW 2e-5、weight decay 0.01、10% linear warmup／decay、3 epochs、未加權 cross-entropy。CUDA 預設啟用 FP16 AMP 與梯度檢查點；可用 `--no-amp`、`--no-gradient-checkpointing` 關閉。最後不足一個累積窗口時依實際樣本數縮放 loss；scheduler 只在成功更新 optimizer 後前進。

每 epoch 結束後以當時模型的 eval 模式評估 train／validation，記錄 loss、Macro-F1、各類 precision／recall／F1 與混淆矩陣。`outputs/baseline/` 保存 `history.json`、`curves.png`、`run_config.json`、`environment.json`、固定切分副本及 `summary.json`。依 validation Macro-F1 選最佳模型，存入 `saved_models/checkpoint/`，並重新載入核對驗證 loss 與混淆矩陣。設定檔保存類別順序、前處理、長度、seed、實際樣本 ID 與資料／切分雜湊；相同 seed 不保證跨 GPU／套件版本位元一致。

為保留實驗結果，程式拒絕覆寫非空的實驗／checkpoint 目錄與既有預測檔。重新實驗時同時指定新路徑，例如 `--output-dir outputs/baseline_seed43 --checkpoint saved_models/baseline_seed43 --seed 43`。`--seed` 只影響訓練，仍共用既有 `--split`。

推論依 checkpoint 設定做相同前處理與 tokenization，依工作計畫的提交規格輸出 `id,pred` 並保留 `benchmark.csv` 原始 ID 順序。執行程式需一併保留 `eda.py`、`training.py`、`preprocessing.py` 及 requirements 檔案。

訓練完成後將入選 checkpoint 放在 `saved_models/checkpoint/`。交件目錄名稱及 ZIP 檔名使用 `HW3_{student_id}`，並依講義納入資料、checkpoint、可執行程式與所需檔案。

## 加權 loss 對照實驗

2026/11/13 已完成，最佳驗證 Macro-F1 **92.65%**（基準 91.57%，提升 1.08 個百分點）。報告見 [加權 CE 對照](docs/Weighted_CE_Report.md)，可提交檔案為 `outputs/weighted_sqrt/predictions.csv`；使用者已回報此版本 Kaggle **0.9049**（基準 0.8973，提升 0.76 個百分點）。基準模型仍留在 `saved_models/checkpoint/`，加權模型位於下列獨立路徑。以下為本輪使用的指令，重跑需改新路徑。

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --output-dir outputs/weighted_sqrt --checkpoint outputs/weighted_sqrt/checkpoint
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_sqrt/checkpoint --predictions outputs/weighted_sqrt/predictions.csv
```

權重只使用本次訓練樣本的類別頻數：先取 `1/sqrt(n_class)`，再正規化，使訓練樣本的平均權重為 1。固定完整切分的 negative／neutral／positive 權重約為 0.740353／1.616846／1.637426。

訓練目標為加權 CE 總和除以梯度累積窗口的實際樣本數；不是各 micro-batch 分別使用 weighted mean。此做法保留類別權重效果，且最後不足一個窗口仍正確縮放。權重與計算規則保存於 run config 及 checkpoint 設定。每輪圖表中的 train／validation loss 均維持未加權 CE，方便與基準比較；模型仍以未改分布的 validation Macro-F1 選取。預設 `--class-weighting none` 保留未加權行為。API 語義參考 [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)。

## URL 正規化對照

2026/11/13 已完成：最佳驗證 Macro-F1 **91.28%**，低於原加權模型 **92.65%**，因此保留原加權版本，未採用 URL 正規化。詳見 [URL 對照報告](docs/URL_Normalization_Report.md)。使用者已回報本輪 Kaggle **0.8977**，低於原加權模型 0.9049；本輪模型與預測保留作實驗紀錄。

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --preprocessing url-http-v1 --output-dir outputs/weighted_url --checkpoint outputs/weighted_url/checkpoint
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_url/checkpoint --predictions outputs/weighted_url/predictions.csv
```

以上路徑已有結果，重跑須改用新路徑。預設前處理仍是 `whitespace-v1`；`url-http-v1` 將 URL 換成 `http` 並正規化空白。推論自動讀取 checkpoint 的前處理版本。完整測試指令為 `.venv/Scripts/python.exe -m unittest test_eda.py test_training.py test_preprocessing.py`；tokenizer 整合測試需要先執行 `prepare`。

## Kaggle 提交格式

提交 CSV 的首列固定為 `id,pred`。依 `dataset/benchmark.csv` 的列順序保留原始 `id`，每個 ID 恰好輸出一筆預測，共 1,000 筆。`pred` 使用小寫類別名稱 `positive`、`neutral` 或 `negative`。

```csv
id,pred
0,positive
1,positive
2,neutral
```
