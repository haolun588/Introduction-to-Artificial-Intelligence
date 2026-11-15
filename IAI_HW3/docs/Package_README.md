# HW3 Sentiment Analysis｜weighted_lr3e5 候選演練包

此包使用 `google-bert/bert-base-uncased` 微調的三分類 checkpoint。這是可重現性演練版本，尚未指定最終交件版本，也未填正式學號檔名。

## 安裝（Windows、Python 3.12）

在解壓後、包含 `main.py` 的目錄執行。環境須使用 64 位元 Python 3.12；本包的 CUDA lock 是 Windows／NVIDIA 環境版本。

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements/lock.txt
.venv/Scripts/python.exe -m pip check
```

若已有本包 lock 對應的 wheelhouse，可將上面兩個 `pip install` 指令替換為以下離線安裝；wheelhouse 是安裝媒體，不包含於模型 ZIP：

```powershell
.venv/Scripts/python.exe -m pip install --no-index --find-links "D:/path/to/wheelhouse" -r requirements.txt -r requirements/lock.txt
```

本次演練使用原下載快取的 39 個 wheels，在全新 venv 執行上述離線安裝。沒有 NVIDIA GPU 時可另裝 CPU torch wheel、只使用 `requirements.txt`；CPU 路徑與其他作業系統尚未於本次演練實測，不能套用 CUDA lock 或宣稱結果已核對。

## 檢查與推論

```powershell
.venv/Scripts/python.exe verify_package.py
.venv/Scripts/python.exe main.py inspect
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.venv/Scripts/python.exe main.py predict --device cuda --batch-size 2 --predictions outputs/predictions.csv
.venv/Scripts/python.exe verify_package.py --predictions outputs/predictions.csv
```

預設 checkpoint 是本包內的 `saved_models/checkpoint/`，已放入 `weighted_lr3e5`，不需下載基底或存取原開發目錄。推論為 FP32。`outputs/predictions.csv` 必須尚不存在；重跑請另指定新路徑，避免覆寫。`verify_package.py` 先驗證 manifest 內全部檔案，指定 `--predictions` 時再核對 `id,pred` 欄位、1,000 筆原始順序、合法標籤、逐筆結果與參考 CSV 的 SHA-256。

`reference/predictions.csv` 是歷史模型推論的核對基準，不會被 `main.py predict` 讀取。使用其他 benchmark 時，保留原檔並用新的資料目錄；原 manifest／參考 CSV 只對隨包附帶的資料有效。

## 包內資料與訓練紀錄

- `dataset/`：課程提供的 9,000 筆標註資料及 1,000 筆 benchmark。
- `saved_models/checkpoint/`：模型、tokenizer、類別對應及訓練設定。
- `outputs/eda/split.json`：固定 8,100／900 文字群組分層切分。
- `training_artifacts/`：每 epoch 曲線、history、run config、summary 與當時環境紀錄。
- `package_manifest.json`：候選、來源路徑、檔案大小與 SHA-256；來源路徑僅供追溯，不用來載入模型。

歷史 summary 中的絕對路徑屬原訓練紀錄；實際推論只使用本包相對路徑。封裝中的目前程式可執行既有模型，但不能用此次封裝 hash 追補為歷史訓練程式來源證明。

## 重訓同一設定（不屬本次演練）

基底權重沒有重複放入 ZIP。若需訓練，連網時移除上述兩個 offline 環境變數，再執行 `prepare` 下載並驗證固定官方 revision；或將相同且通過 `verify-base` 的本機基底放在 `outputs/base_model/`。只使用課程 train／validation，不使用 benchmark 或額外資料訓練。

```powershell
.venv/Scripts/python.exe main.py prepare
.venv/Scripts/python.exe main.py verify-base
.venv/Scripts/python.exe main.py train --device cuda --seed 42 --epochs 3 --batch-size 2 --accumulation-steps 8 --learning-rate 3e-5 --class-weighting sqrt-inverse --max-length 64 --preprocessing whitespace-v1 --split outputs/eda/split.json --output-dir outputs/retrain_lr3e5 --checkpoint outputs/retrain_lr3e5/checkpoint
```

重新訓練時明確指定新的 checkpoint，避免寫入包內模型；同設定重訓仍可能受硬體／數值隨機性影響，不保證產生逐位相同權重。
