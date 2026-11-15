# 候選封裝與乾淨環境推論演練

日期：2026/11/15。狀態：獨立 ZIP、乾淨解壓、新 venv 安裝與離線 GPU 推論均通過，1,000 筆預測與原 CSV 逐位元組相同。

## 目的、範圍與預算

沿用 registry 的 Kaggle 單模型候選 `weighted_lr3e5`，製作獨立 ZIP、解壓至新目錄，建立不共用 site-packages 的 Python 3.12 venv，依封裝 README 安裝依賴後，以封裝中的預設 checkpoint 執行 FP32 推論。完成條件為 1,000 筆 ID／標籤及 CSV SHA-256 均與已登錄原檔相同，封裝檔案 hash 與原來源一致，且未依賴開發目錄的程式或模型。

這是封裝／環境改變後的必要重現檢查，不是新模型實驗。上限：0 次訓練、1 次完整 GPU 推論；只有發現具體問題並修正時才重試。來源／hash／安裝或預測不符即停止後續判定並保留證據。安裝沿用本機 pip 快取中的原始 wheels，以相同 lock 檔重新安裝；不複製既有已安裝環境。

演練根目錄為 `outputs/package_rehearsal_20261115/`；ZIP 暫名 `HW3_rehearsal_weighted_lr3e5.zip`，不是 `HW3_{student_id}.zip` 正式交件包。未指定最終模型或學號，`release_candidate` 保持 `null`；不代為上傳 E3／Kaggle。

## 交付與結果

| 項目 | 結果 |
|---|---|
| 候選 | `weighted_lr3e5`，原最佳 epoch 3；未重訓或更換候選 |
| 演練 ZIP | [HW3_rehearsal_weighted_lr3e5.zip](../../outputs/package_rehearsal_20261115/build/HW3_rehearsal_weighted_lr3e5.zip)，407,968,988 bytes（約 408 MB） |
| ZIP SHA-256 | `6e7476cda021ce00eb32ef1d26ea09c12f34bcb598633fc2fdfe49960da74894` |
| 封裝內容 | 28 個內容檔案＋manifest；逐檔 hash、ZIP 檔名白名單、解壓內容均通過 |
| 安裝 | 全新 Python 3.12.10 venv、39 個 lock 版本套件從既有原始 wheels 離線安裝；`pip check` 通過 |
| 載入隔離 | `include-system-site-packages=false`、停用 user site、清除外部 Python 搜尋路徑；逐項核對 39 個 distribution 位於新 venv，`src.*` 位於解壓包 |
| 推論環境 | Windows、RTX 3050 Laptop GPU、torch 2.9.1+cu126、CUDA 12.6、FP32、batch 2 |
| 推論結果 | 1,000 筆 `id,pred`、ID 原順序、合法標籤、0 筆差異；SHA-256 與原檔相同 |
| CSV SHA-256 | `b6bae1443d4cdc28060906038e8d5ec5750e37225952693a718476ac0a013a7f` |
| 離線證據 | `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，使用全新空 HF cache；推論後 cache 仍空，載入包內預設 checkpoint |
| 成本 | 0 次訓練、1 次完整推論；推論程序約 12.71 秒（含 import／模型載入），依賴安裝約 4 分 40 秒 |

機器可讀結果：[summary.json](../../outputs/package_rehearsal_20261115/summary.json)。安裝明細：[installation.json](../../outputs/package_rehearsal_20261115/installation.json)。實際命令、工作目錄、退出碼及耗時：[commands.json](../../outputs/package_rehearsal_20261115/verification/commands.json)。原始輸出：[推論](../../outputs/package_rehearsal_20261115/verification/predict.log)、[環境](../../outputs/package_rehearsal_20261115/verification/environment.log)、[最後核對](../../outputs/package_rehearsal_20261115/verification/verify_predictions.log)。安裝明細中的全部 wheel SHA-256 已與快取複本 manifest 核對一致。

`build/build_summary.json` 保留剛打包時尚未安裝／推論的階段快照；最終狀態以本輪 `summary.json` 與 registry 登錄為準。原始 checkpoint、原 CSV 及既有精度稽核均未覆寫。

## 封裝與重現

封裝用 [package_candidate.py](../../tools/package_candidate.py) 只複製必要檔案，將目前 Kaggle 單模型候選放到包內的 `saved_models/checkpoint/`。不將開發 `.venv`、其他候選、安裝 wheelhouse 或基底權重放入 ZIP。訓練曲線／歷史、固定切分與指定 revision 的基底準備方法均隨包保留。

重建另一份演練包時使用新的目錄，從 HW3 根目錄執行：

```powershell
.venv/Scripts/python.exe -m tools.package_candidate --output-dir outputs/package_rehearsal_next/build
```

包內的 [README](../../outputs/package_rehearsal_20261115/build/HW3_rehearsal_weighted_lr3e5/README.md) 由 [維護來源](../Package_README.md) 複製，列出建立 venv、線上／離線安裝、推論、hash 核對及重訓同設定的完整指令。本次實際走離線安裝分支；沒有另外測試線上下載安裝、CPU 或其他作業系統。推論後使用包內 `verify_package.py --predictions outputs/predictions.csv` 再驗證全部內容與輸出。

此次沒有新增 GPU 訓練實驗或平台提交，不改寫 Kaggle 分數。研究目標尚未達 92%；學號、最終候選、正式 ZIP 命名、平台公告與收件核對仍待辦，不能將這份演練包視為已交件。

下一個具體工作：依 92% 計畫檢查 optimizer／scheduler、累積、加權 loss、checkpoint 選取與類別／ID 對齊，再實作每 0.5 epoch 評估，建立新控制組。核心完整訓練預算仍未使用。

## 收尾檢核

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：已重讀 AGENTS，變更限 HW3；不重訓、不使用額外資料、不上傳平台。 |
| 目的完成 | 通過：建立 ZIP、新目錄解壓與新環境安裝，正式推論 1,000 筆與原 CSV 相同。 |
| 查重與成本 | 通過：沿用候選、固定切分與既有精度證據；使用既有 wheels，0 次訓練／1 次推論。 |
| 既有工作保全 | 通過：修改前讀取 diff，保留上一輪及 Emoji 更新；全部新產物在獨立演練目錄。 |
| 驗證證據 | 通過：三個工具語法檢查、`pip check`、包內 `inspect`、39 依賴版本／來源、28 檔 hash、離線推論及 CSV 核對通過；未更動核心模型程式，不重跑無關訓練測試。 |
| 實驗可追溯 | 通過：獨立 package rehearsal 已登錄，ZIP／manifest／模型／資料／CSV 與安裝執行證據 hash 核對通過；refresh 及 `--check` 通過。直接讀取 ZIP 也核對全部 29 個項目與 28 份內容 hash；7 個原 checkpoint／9 份原 CSV 不變。 |
| 文件與狀態 | 通過：入口、計畫、交件清單與本報告已同步，8 份 Markdown 共 97 個本機連結存在；diff 無空白錯誤。 |
| 交付誠實 | 通過：演練完成與最終交件分開標示，保留尚未實測的平台／CPU／線上下載限制。 |

檢核：適用項 8／8 通過；不適用：無。此判定僅涵蓋本次候選封裝演練，正式交件仍待完成。
