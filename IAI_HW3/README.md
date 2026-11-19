# IAI HW3｜Sentiment Analysis

**所有 HW3 工作開始前，先閱讀並遵守 [最高優先工作規範 AGENTS.md](AGENTS.md)；完成後須通過其中的收尾檢核表。**

使用指定的 `google-bert/bert-base-uncased` 微調三分類模型。**先看 [實驗總表](docs/Experiment_Results.md)，下一步依 [92% 改進計畫](docs/Target_92_Plan.md)。** 文件整理於 2026/11/14；歷史報告集中在 `docs/reports/`。

## 目前狀態

| 用途 | 版本 | 已知結果 |
|---|---|---|
| 驗證比較基準 | `outputs/weighted_sqrt/checkpoint/` | 固定驗證 Macro-F1 92.65% |
| 後續新方法共同控制組 | `outputs/weighted_half_epoch_control/checkpoint/` | 半輪評估已完成；最佳權重／CSV 與舊驗證基準相同 |
| Kaggle 單模型候選 | `outputs/weighted_quarter/checkpoint/` | 目前最高使用者回報；分數見實驗總表 |
| 目前預設 checkpoint | `saved_models/checkpoint/` | **早期 baseline，尚未替換成交件版本** |

已完成 12 個完整訓練、2 組集成。[FGSM C2](docs/reports/FGSM_High_Report.md) 驗證未改善，Kaggle 回報亦未超越首選，保留對照 CSV；停止本輪 FGSM 搜尋，保留 weighted_quarter 為首選。下一項為 weighted_quarter 的獨立封裝與乾淨環境推論演練；尚未啟動。

## 目錄與閱讀入口

```text
IAI_HW3/
  AGENTS.md                       工作前必讀規範與收尾檢核表
  main.py                         inspect / eda / prepare / verify-base / train / predict / kfold / pretrain-mlm
  requirements.txt                作業指定的依賴入口，轉入 requirements/train.txt
  src/                            核心程式套件
    training.py, mlm.py, eda.py, preprocessing.py
    model_source.py               指定基底的固定雜湊與離線驗證
  tools/                          分析與維護工具
    analyze_ensemble.py, audit_emoji.py, audit_precision.py, cv_oof.py, refresh_experiments.py
    package_candidate.py, verify_package.py
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
.venv/Scripts/python.exe -m tools.audit_precision --help
```

需要用目前 Kaggle 候選重新推論時，**明確指定 checkpoint**；預測檔必須使用尚不存在的路徑：

```powershell
.venv/Scripts/python.exe main.py predict --device cuda --checkpoint outputs/weighted_quarter/checkpoint --predictions outputs/review_quarter/predictions.csv
```

目前首選 CSV 是 `outputs/weighted_quarter/predictions.csv`；最新探索檔為 [FGSM C2 CSV](outputs/weighted_fgsm_high/predictions.csv)。提交格式固定為 `id,pred`，依 `dataset/benchmark.csv` 原始 ID 順序輸出 1,000 筆；標籤限 `negative`、`neutral`、`positive`。程式拒絕覆寫既有 CSV 及非空訓練目錄。

新對照以新的 run ID 同時指定輸出及 checkpoint，以下只是操作範例，尚未執行：

```powershell
.venv/Scripts/python.exe main.py train --device cuda --eval-interval 0.5 --class-weighting sqrt-inverse --learning-rate 2e-5 --output-dir outputs/weighted_control_v2 --checkpoint outputs/weighted_control_v2/checkpoint
```

每半輪評估已實作並完成一次控制組；上面是操作範例，不是要求再跑相同設定。`--eval-interval` 預設 1.0，相容舊指令；後續新方法明確指定 0.5，評估對齊完整累積窗口並記錄實際進度。較溫和權重 `--class-weighting quarter-inverse` 已實作並完成對照；分層學習率 `--layer-lr-decay 0.9` 已實作並完成對照，預設 1.0 保留均一學習率。FGSM 以 `--fgsm-epsilon` 開啟，預設 0 保留原流程；尺度與短跑見 [校準報告](docs/reports/FGSM_Calibration_Report.md)，最新結果與停止搜尋決策見 [C2 報告](docs/reports/FGSM_High_Report.md)。

**預設值變更（2026/11/19，依據 [速度實測報告](docs/reports/Speed_Check_Report.md)）**：`train` 的評估 batch 預設為 64（`--eval-batch-size` 未指定時；`pretrain-mlm` 仍沿用 `--batch-size`），gradient checkpointing 預設關閉（`--gradient-checkpointing` 開啟，`--no-gradient-checkpointing` 仍可用於舊指令）。微調權重與舊設定位元相同，1 epoch 約 467 秒降到 203 秒。開 FGSM、加長 `--max-length` 或遇到 OOM 時請加 `--gradient-checkpointing`。需要和舊 MLM run 逐位比較驗證 loss 時，不要改 `pretrain-mlm` 的評估 batch（驗證遮罩依評估 batch 物化，改變會使 val MLM loss 不可逐位比較；實測評估 batch 64 還會讓峰值 reserved 超過 4 GB）。

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

預設訓練為 3 epochs、batch 2／累積 8、AdamW 2e-5、weight decay 0.01、10% warmup／linear decay、未加權 CE。CUDA 預設 FP16 AMP 與梯度檢查點；正式單模型推論為 FP32。兩個保留候選已完成 [跨精度檢查](docs/reports/Precision_Audit_Report.md)：驗證與 benchmark 標籤均一致，保留現行精度方式。已完成的加權對照使用 `--class-weighting sqrt-inverse`。`--label-smoothing ε`（預設 0.0，範圍 [0, 1)）只作用於訓練 loss：每筆樣本先算未加權的平滑 CE，再乘以真實類別權重；評估 loss 不變。`--exclude-train-ids 檔案`（每行一個 id 的 UTF-8 文字檔）只從訓練子集移除這些 id（類別權重依過濾後計算），驗證子集不變；重複 id 或不在資料集的 id 會中止，屬於驗證子集的 id 不移除但計數，紀錄寫入 run_config／training_config 的 `exclude_train_ids`（供 CV 每折共用同一份清單）。

`pretrain-mlm` 子命令（`--split` 必填；`--output-dir`、`--checkpoint` 須為新路徑）以該 split 的訓練子集文字做 MLM 續訓（標準 80/10/10 遮罩，預設 10 epochs、lr 5e-5、mask 0.15、warmup 0.06、weight decay 0.01；前處理、tokenizer、max_length、batch／累積同 train），validation 文字只用來算固定遮罩的 MLM eval loss，不選模型、不用標籤與 benchmark，儲存最後一個 epoch 權重，並寫入 `mlm_config.json`（含 `mlm_text_ids`、來源與 hash）、`history.json`、`curves.png`、`summary.json`（完成標記，checkpoint 內另存副本 `mlm_summary.json`）。`train --init-from <MLM checkpoint>` 沿用其 embeddings／encoder 權重，pooler 與分類頭維持原流程（依 seed 新初始化分類頭）；若 base 來源或 dataset hash 不符、MLM 語料與本次驗證 id 有交集、非訓練 id 子集或 MLM 未完成，會在建立任何目錄前中止；紀錄寫入 `init_from`（未使用為 null）。講義未明文規定續訓是否屬於 base model，使用者已知悉並接受此風險。

五折交叉驗證入口（切分、五折訓練與 OOF 評估；細節見 [CV 報告](docs/reports/CV5_Sqrt_Report.md)）：

```powershell
.venv/Scripts/python.exe main.py kfold --k 5 --seed 42 --output-dir outputs/cv5_seed42
bash tools/run_cv5_sqrt.sh   # 5 次訓練後呼叫 python -m tools.cv_oof 產生 OOF 與集成 CSV
bash tools/run_cv5.sh <name> <none|sqrt-inverse|quarter-inverse> [epochs] [train 參數...]   # 通用版（epochs 可選，正整數，預設 3）；第 4 個起的參數原樣轉給 main.py train（例如 --label-smoothing 0.1，需明確給 epochs）；run_cv5_sqrt.sh 為其包裝
bash tools/run_cv5_mlm.sh <name> <none|sqrt-inverse|quarter-inverse> <finetune_epochs> <mlm_epochs> [folds...]   # 每折先 pretrain-mlm 再 train --init-from；5 折皆完成才跑 OOF，只給部分 folds 會跳過 OOF
bash tools/run_full_mlm.sh <name> <mlm_epochs> <finetune_epochs> <seeds...>   # 預設切分 8,100 筆：每個 seed 先 pretrain-mlm 再 sqrt-inverse train --init-from；全部 seed 完成後呼叫 python -m tools.ensemble_checkpoints（至多 3 個 checkpoint 的 FP32 等權 softmax 集成）
bash tools/run_cv5_mlm_shared.sh <name> <none|sqrt-inverse|quarter-inverse> <finetune_epochs> <mlm_epochs>   # 以全部 9,000 筆文字續訓一次（pretrain-mlm --mlm-corpus all-dataset），5 折共用該 checkpoint 微調（train --allow-mlm-val-overlap，非嚴格 OOF）
python -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_quarter_oof --output <新 json>   # 配對 bootstrap 比較兩份 OOF
```

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

候選演練 ZIP、安裝與推論證據由 registry 的 `package_rehearsals` 登錄，總表會連同 hash 核對；重建與包內操作指令見 [封裝報告](docs/reports/Package_Rehearsal_Report.md)。演練不會自動設定 `release_candidate`。

Git 只保存程式、資料、文件、登錄及輕量 snapshot。權重、輸出和 `.venv` 約占 8 GiB，本次完整保留；另行備份模型與 CSV，Git 並不包含它們。交件封裝依 [課程要求與清單](docs/HW3_Work_Plan.md)，不要直接壓縮整個開發工作區。
