# HW3｜課程要求與交件清單

本文件只維護課程要求與完成狀態。當前研究排程以 [92% 改進計畫](Target_92_Plan.md) 為準，實驗結果見 [總表](Experiment_Results.md)。2026/11/14 已核對 [原始講義](HW3.pdf) 第 7–15 頁；平台是否有後續公告尚未核驗。

## 規範

| 要求 | 依據／現況 |
|---|---|
| 指定基底 | 只使用 `google-bert/bert-base-uncased`（第 9 頁） |
| 自訂流程 | 自行設計前處理與訓練管線（第 9 頁）；已實作於 `src/preprocessing.py`、`src/training.py` |
| 訓練資料 | 只使用提供資料切出的 train／validation；不得使用 benchmark 或其他額外資料訓練（第 14 頁） |
| 訓練曲線 | 每 epoch 保存訓練與驗證曲線（第 9 頁）；各完整實驗均有 `curves.png` |
| 集成 | 最多 3 個模型（第 9 頁） |
| E3／Kaggle 截止 | 講義記載 2026/11/30（一）23:59（第 7 頁）；交件前核對平台公告 |
| E3 檔名 | `HW3_{student_id}.zip`；名稱錯誤扣 10 分（第 8、13 頁） |
| Kaggle 身分與頻率 | 隊名為學號，帳號綁課堂名單 email，每 24 小時 3 次（第 8 頁） |
| 可執行性 | 助教無法執行為 0 分；需監看 email 並於聯絡後 3 天內回覆（第 14–15 頁） |

| Benchmark Macro-F1 | 講義分數 |
|---|---:|
| ≥ 92% | 100 |
| ≥ 90.5% 且 < 92% | 90 |
| ≥ 89% 且 < 90.5% | 80 |
| < 89% | 0 |

Kaggle 分數是使用者歷史回報，不能據此保證最終成績。講義第 14 頁要求訓練／驗證切分，沒有明示免驗證的全 9,000 筆最終重訓；暫不將全資料重訓列為預設交件流程。

## 六個課程主題的進度

| 主題 | 已完成 | 尚未做／目前決策 |
|---|---|---|
| EDA | 文字群組分層、固定 8,100／900 切分、token 長度盤點；預先固定 seed 142／242 額外切分 | 保留原切分作比較；額外切分尚未訓練，僅用於確認候選 |
| Data Preprocessing | 空白、URL、emoji 對照 | 保留空白正規化；停止擴大已失敗規則 |
| Class Imbalance | 未加權、反平方根與較溫和 n^-0.25 加權 CE | 維持原驗證基準；較溫和權重驗證下降、Kaggle 回報提升，設為 Kaggle 候選 |
| Model Training | 自訂迴圈檢查、累積、AMP、checkpoint 重載、學習率／seed 對照；半輪評估控制組完成 | 分層學習率 B 已完成，驗證未改善；保留 AMP 驗證／FP32 推論與半輪選模流程 |
| Adversarial Training | [FGSM C1／C2 完整訓練與核對](reports/FGSM_High_Report.md) | 兩者驗證未改善，停止本輪 FGSM 搜尋，保留既有候選 |
| Ensemble Learning | 兩組等權集成與錯判分析 | 尚無超越最佳 Kaggle 的證據；保留單模型候選 |

## 交件目錄

依講義第 11 頁，在獨立封裝目錄保留 `main.py`、`README.md`、`requirements.txt` 與指定資料／checkpoint 路徑。`main.py` 會 import `src/`，依賴入口會引用 `requirements/`，因此兩個資料夾也須一併帶入；重訓所需的基底模型與固定切分位置需在交件 README 說明。

```text
HW3_{student_id}/
  dataset/
    dataset.csv
    benchmark.csv
  saved_models/checkpoint/          最終選定的模型、tokenizer、training_config
  main.py
  src/
    __init__.py
    training.py
    eda.py
    preprocessing.py
    model_source.py                固定基底驗證；其他模組會 import
  README.md                        交件版本的安裝與推論指令
  requirements.txt
  requirements/
    train.txt
    eda.txt
    lock.txt
  ...                              訓練曲線與重現所需檔案
```

目前開發目錄的 `saved_models/checkpoint/` 仍是 baseline。日後打包時從 registry 指定的最終候選複製到獨立封裝目錄，再以 hash 核對；不要把現有開發目錄直接當成已完成交件包。

2026/11/15 已對 `weighted_lr3e5` 完成 [候選演練包](reports/Package_Rehearsal_Report.md)：ZIP 解壓、新 venv 安裝與離線 GPU 推論通過，1,000 筆 CSV 與原檔相同。這份包未填學號且尚未定為最終版本，以下正式交件項目仍需依選定版本完成。

## 交件完成條件

- [x] 固定切分、基準、加權對照、完整訓練與預測流程存在。
- [x] 整理 14 個正式實驗候選的 metadata、checkpoint hash 與 1,000 筆 CSV 對應（半輪控制組 CSV 與既有版本相同）。
- [x] 完成兩個保留候選的 AMP／FP32 一致性實測：標籤均相同，採 FP32 正式單模型推論（[證據](reports/Precision_Audit_Report.md)）；新候選 weighted_quarter 僅在[訓練報告](reports/Quarter_Weight_Report.md)記錄驗證集 900 筆 AMP／FP32 檢查（標籤差異 0、F1 均 91.9119%）；未跑 benchmark AMP 比較，亦不屬 precision_audit_20261115。
- [x] 原候選 weighted_lr3e5 已完成獨立封裝與乾淨環境推論演練（非正式交件）。
- [ ] 新候選 weighted_quarter 的獨立封裝與乾淨環境推論演練。
- [ ] 確定最終模型及對應 Kaggle CSV，在 registry 設定 `release_candidate`。
- [ ] 在獨立的 `HW3_{student_id}/` 放入選定 checkpoint、必要程式、資料與曲線。
- [ ] 從乾淨解壓目錄照交件 README 安裝並推論，核對結果與原 CSV。
- [ ] 核對 ZIP 內容、學號、E3 檔名，以及 Kaggle 選定版本。
- [ ] 由使用者確認 E3／Kaggle 收到正確版本；記錄回執及時間。

研究目標、流程正確與交件完成分開判定；現階段完成實驗／檔案盤點、精度核對及候選封裝演練，並非最終交件。
