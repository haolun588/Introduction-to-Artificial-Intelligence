# IAI HW3｜Sentiment Analysis

**所有 HW3 工作開始前，先閱讀並遵守 [最高優先工作規範 AGENTS.md](AGENTS.md)；完成後須通過其中的收尾檢核表。**

使用指定的 `google-bert/bert-base-uncased` 微調三分類模型。**先看 [實驗總表](docs/Experiment_Results.md)，下一步依 [92% 改進計畫](docs/Target_92_Plan.md)。** 文件整理於 2026/11/14；歷史報告集中在 `docs/reports/`。

## 目前狀態

| 用途 | 版本 | 已知結果 |
|---|---|---|
| 驗證比較基準 | `outputs/weighted_sqrt/checkpoint/` | 固定驗證 Macro-F1 92.65% |
| Kaggle 單模型候選 | `outputs/weighted_lr3e5/checkpoint/` | 使用者回報 0.9052 |
| 目前預設 checkpoint | `saved_models/checkpoint/` | **早期 baseline，尚未替換成交件版本** |

已完成 7 個完整訓練、2 組集成。Emoji 對照有 CSV，Kaggle 分數待回報。92% 是 Kaggle 目標；驗證分數超過 92% 不代表達標。最終交件版本尚未指定，尚未完成乾淨解壓推論演練。

## 目錄與閱讀入口

```text
IAI_HW3/
  AGENTS.md                       工作前必讀規範與收尾檢核表
  main.py                         inspect / eda / prepare / verify-base / train / predict
  requirements.txt                作業指定的依賴入口，轉入 requirements/train.txt
  src/                            核心程式套件
    training.py, eda.py, preprocessing.py
    model_source.py               指定基底的固定雜湊與離線驗證
  tools/                          分析與維護工具
    analyze_ensemble.py, audit_emoji.py, refresh_experiments.py
  tests/                          切分、訓練、前處理與來源驗證測試
    test_*.py
  requirements/                   依賴清單
    train.txt, eda.txt, lock.txt
  dataset/                        課程原始資料
  docs/
    README.md                     文件索引
    Experiment_Results.md         自動產生的實驗總表
    Target_92_Plan.md              最新方向、預算與停止條件
    HW3_Work_Plan.md               課程要求與交件清單
    reports/                      各輪實驗證據
    archive/                      整理前的長文快照
    HW3.pdf                       原始講義
  experiments/
    registry.json                 實驗登錄、Kaggle 回報與候選角色
    snapshot.json                 可追溯的指標、設定與檔案雜湊
  outputs/                        本機切分、基底模型、實驗產物（不進 Git）
  saved_models/checkpoint/         目前仍為 baseline（權重不進 Git）
  .venv/                          本機 Python 環境（不進 Git）
```

原模型及輸出路徑保留，以維持已保存設定與重現指令的相容性。更多文件見 [文件索引](docs/README.md)。

2026/11/14 已將 13 個檔案分類至上述資料夾。`main.py` 與 `requirements.txt` 保留在根目錄以符合交件結構；核心程式以 `src` 套件匯入，不需額外安裝專案。`src/` 內的模組透過 `main.py` 使用；分析工具改用下列 `-m tools...` 指令。

## 日常使用

以下指令都從 `IAI_HW3` 目錄執行；若目前在 repository 根目錄，先執行 `Set-Location IAI_HW3`。

```powershell
# 資料盤點、現有測試、實驗索引完整性（不啟動訓練）
.venv/Scripts/python.exe main.py inspect
.venv/Scripts/python.exe -m unittest discover -p "test_*.py"
.venv/Scripts/python.exe tools/refresh_experiments.py --check
```

分析工具的說明入口（不執行分析／推論）：

```powershell
.venv/Scripts/python.exe -m tools.analyze_ensemble --help
.venv/Scripts/python.exe -m tools.audit_emoji --help
```

需要用目前 Kaggle 候選重新推論時，**明確指定 checkpoint**；預測檔必須使用尚不存在的路徑：

```powershell
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_lr3e5/checkpoint --predictions outputs/review_lr3e5/predictions.csv
```

已有可用 CSV 是 `outputs/weighted_lr3e5/predictions.csv`。提交格式固定為 `id,pred`，依 `dataset/benchmark.csv` 原始 ID 順序輸出 1,000 筆；標籤限 `negative`、`neutral`、`positive`。程式拒絕覆寫既有 CSV 及非空訓練目錄。

新對照以新的 run ID 同時指定輸出及 checkpoint，以下只是操作範例，尚未執行：

```powershell
.venv/Scripts/python.exe main.py train --device cuda --class-weighting sqrt-inverse --learning-rate 2e-5 --output-dir outputs/weighted_control_v2 --checkpoint outputs/weighted_control_v2/checkpoint
```

此範例仍為現有的每 epoch 評估；計畫中的每半輪評估、分層學習率、較溫和權重及 FGSM 尚未實作。

## 環境與固定切分

本機已完成安裝與模型下載。只在新環境執行下列步驟；訓練環境已以 Python 3.12 驗證：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install -r requirements/lock.txt
.venv/Scripts/python.exe main.py prepare
.venv/Scripts/python.exe main.py eda
```

一般安裝仍用 `-r requirements.txt`，會載入 `requirements/train.txt` 及 `requirements/eda.txt`；`requirements/lock.txt` 保留完整已驗證版本。沒有 NVIDIA GPU 可使用 CPU wheel 與 `--device cpu`。`prepare` 下載指定 revision；其後訓練與推論使用本機檔案。只做 EDA 可安裝 `requirements/eda.txt`，不需 PyTorch 或 BERT 權重。

`prepare` 下載後與每次 `train` 開始前，會依 `src/model_source.py` 內固定的官方版本雜湊，核對 5 個基底檔案。可單獨離線檢查，不載入模型或使用 GPU：

```powershell
.venv/Scripts/python.exe main.py verify-base
```

`--base-model` 仍可指向同一份已核對檔案的其他本機副本；須包含 `config.json`、`model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt`，除此之外只允許 `.cache/`。缺檔、雜湊不符或多出 tokenizer／adapter 等檔案，會在 GPU 初始化及建立實驗目錄前被拒絕。請用 `prepare --base-model <新的空目錄>` 重新準備，不修改白名單來接受未知模型。

新訓練的 `run_config.json` 與 checkpoint 的 `training_config.json` 會保存 `base_model_source`（實際路徑、revision、各檔案大小與 SHA-256）及 `training_code_sha256`。舊 checkpoint 繼續可推論，但不會追補為已完成歷史來源驗證。交件時務必保留完整 `src/`（含 `__init__.py`）與 `requirements/`；修正與驗證範圍見 [合規稽核報告](docs/Compliance_Audit.md)。

固定切分位於 `outputs/eda/split.json`：seed 42、8,100／900 筆，按空白正規化文字分組並分層，保存 ID 與資料 SHA-256。後續訓練共用此切分；`train --seed` 只影響訓練。EDA 不相容的資料／參數會被拒絕覆寫，須指定新 `--output-dir`。訓練集長度 P50／P90／P95 = 31／40／43；`max_length=64` 截斷率約 0.02%，採批內動態 padding。

預設訓練為 3 epochs、batch 2／累積 8、AdamW 2e-5、weight decay 0.01、10% warmup／linear decay、未加權 CE。CUDA 預設 FP16 AMP 與梯度檢查點；正式單模型推論目前為 FP32，跨精度差異仍待實測。已完成的加權對照使用 `--class-weighting sqrt-inverse`。

短跑使用每類各 16 筆、1 epoch；其 CSV 不作正式提交：

```powershell
.venv/Scripts/python.exe main.py train --smoke --device cuda --output-dir outputs/smoke_new --checkpoint outputs/smoke_new/checkpoint
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/smoke_new/checkpoint --predictions outputs/smoke_new/predictions.csv
```

## 實驗紀錄維護

每個完整實驗使用獨立的 `outputs/<run_id>/`，將報告放入 `docs/reports/`，在 `experiments/registry.json` 登錄路徑與使用者回報的 Kaggle 分數；未知分數維持 `null`。然後執行：

```powershell
.venv/Scripts/python.exe tools/refresh_experiments.py
.venv/Scripts/python.exe tools/refresh_experiments.py --check
```

工具核對既有 metadata、固定切分、混淆矩陣／Macro-F1、完整 checkpoint 檔案雜湊與 CSV 格式，不重新訓練或推論。`snapshot.json` 中目前程式碼的 hash 僅標示盤點時版本，不能證明歷史訓練使用了相同程式碼。只更新 registry 再重建總表，避免在多份文件重複維護完整分數。

Git 只保存程式、資料、文件、登錄及輕量 snapshot。權重、輸出和 `.venv` 約占 8 GiB，本次完整保留；另行備份模型與 CSV，Git 並不包含它們。交件封裝依 [課程要求與清單](docs/HW3_Work_Plan.md)，不要直接壓縮整個開發工作區。
