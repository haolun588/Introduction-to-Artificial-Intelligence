# HW3 文件索引

**工作前必讀：[最高優先工作規範](../AGENTS.md)。工作完成後，逐項核對其收尾檢核表。**

日常只需三個入口：回到 [操作說明](../README.md)、查看 [實驗總表](Experiment_Results.md)、接續 [92% 改進計畫](Target_92_Plan.md)。

| 文件 | 用途 | 更新方式 |
|---|---|---|
| [AGENTS.md](../AGENTS.md) | 所有 HW3 工作的前置規範、作業紅線與收尾檢核 | 使用者修訂規範或課程依據更新時維護 |
| [Compliance_Audit.md](Compliance_Audit.md) | 作業規範稽核、證據範圍與交件缺口 | 有新的合規證據或完成交件檢查時更新 |
| [Experiment_Results.md](Experiment_Results.md) | 分數、候選、checkpoint、CSV 與 hash 對應 | 修改 registry 後以工具重建 |
| [Target_92_Plan.md](Target_92_Plan.md) | 目前判斷、下一步、實驗預算、停止條件 | 只有方向或階段改變時更新 |
| [HW3_Work_Plan.md](HW3_Work_Plan.md) | 課程要求、主題進度與交件清單 | 規範或交件狀態改變時更新 |
| [HW3.pdf](HW3.pdf) | 課程原始講義 | 保留原檔 |
| [registry.json](../experiments/registry.json) | 實驗登錄與 Kaggle 回報來源 | 新實驗／分數回報時更新 |
| [snapshot.json](../experiments/snapshot.json) | 本機產物驗證與完整 hash | 工具產生 |

## 已完成實驗

| 主題 | 報告 |
|---|---|
| 基準 | [GPU 短跑](reports/Baseline_Smoke_Report.md)、[完整基準](reports/Baseline_Report.md) |
| 類別權重 | [反平方根加權 CE](reports/Weighted_CE_Report.md) |
| 前處理 | [URL 正規化](reports/URL_Normalization_Report.md)、[Emoji 盤點](reports/Emoji_Audit_Report.md)、[Emoji 訓練](reports/Emoji_Training_Report.md) |
| 學習率 | [1e-5](reports/Learning_Rate_Report.md)、[3e-5](reports/Learning_Rate_3e5_Report.md) |
| seed 與集成 | [錯判與學習率集成](reports/Ensemble_Error_Analysis_Report.md)、[seed 43](reports/Seed43_Report.md)、[seed 42／43 集成](reports/Seed_Ensemble_Report.md) |

EDA 報告位於本機 [outputs/eda/EDA_Report.md](../outputs/eda/EDA_Report.md)。報告內的重現指令均以 HW3 根目錄為工作目錄；既有輸出路徑不能直接覆寫。

## 2026/11/14 整理紀錄

- 將 11 份實驗報告集中到 `reports/`，同步調整連結。
- README 改為操作入口；工作計畫改為課程要求與交件清單；後續方向集中於 `Target_92_Plan.md`。
- 整理前的 [README](archive/README_2026-11-14.md) 與 [工作計畫](archive/HW3_Work_Plan_2026-11-14.md) 完整保留內容，僅加封存標記並更新相對連結。封存內容不代表現行排程。
- 首次文件整理保留既有程式、資料、模型及實驗輸出路徑；後續程式分類見下節。沒有因此啟動新訓練或更換預設 checkpoint。
- 已有 emoji 程式與測試修改、分析腳本及報告屬整理前的未提交工作；沒有回退或自動提交它們。

整理後驗證：現有 13 項測試通過；9 份 CSV、7 份 checkpoint 與總表一致；18 份 Markdown 的本機連結均存在。總表工具另確認會拒絕重複 ID、缺漏／多餘 CSV 欄位、工作區外路徑與不一致的 Macro-F1。檢查範圍為檔案與程式測試，未重新執行模型推論。

後續每輪新增一份報告和一筆 registry 記錄即可；不再把全部實驗歷史附加到 README。

## 2026/11/14 程式與依賴分類

| 類別 | 現在位置 | 使用方式 |
|---|---|---|
| 核心程式 | `src/` | 根目錄 `main.py` 匯入；保留完整套件 |
| 分析／維護 | `tools/` | `python -m tools.analyze_ensemble`、`python -m tools.audit_emoji`；總表工具原指令仍可用 |
| 測試 | `tests/` | 從 HW3 執行 `python -m unittest discover -p "test_*.py"` |
| 依賴 | `requirements/` | 根目錄 `requirements.txt` 引用 `train.txt`，再引用 `eda.txt`；完整鎖定環境為 `lock.txt` |

共搬移 13 個檔案；根目錄只保留作業入口 `main.py` 與 `requirements.txt`。同步更新 import、以檔案位置推算的 HW3 根路徑、來源 hash 清單及現行文件指令。封存文件保留當時的指令文字；需要重現時以現行 README 的路徑為準。

分類完成檢核（8/8 通過）：

| 項目 | 證據 |
|---|---|
| 規範與範圍 | 先讀 AGENTS，變更限 HW3，保留作業指定的根目錄入口。 |
| 目的完成 | 13 個檔案歸入四類，根目錄 `.py`／`.txt` 僅剩兩個入口。 |
| 查重與成本 | 搬移並調整既有程式；未新增實驗、未重訓 BERT、未安裝依賴。 |
| 既有工作保全 | 搬移前檢視差異，以目前工作副本為準；保留既有未提交修改。 |
| 驗證證據 | 20 項測試、`inspect`、`verify-base`、兩個分析工具的 `--help`、依賴清單離線解析及 diff 檢查通過。 |
| 實驗可追溯 | 索引已更新且 `--check` 通過；前後比較確認資料／切分、9 個候選、7 份模型及 CSV 雜湊與分數相同。 |
| 文件與狀態 | README、交件結構、現行報告指令及 Markdown 本機連結已更新／核對。 |
| 交付誠實 | 完成的是目錄分類與相容性檢查，未宣稱已做完整 BERT 推論或最終交件。 |
