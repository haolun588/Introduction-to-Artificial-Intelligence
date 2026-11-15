# 分層學習率對照 B

2026/11/15。三輪訓練與推論完成，最佳 validation Macro-F1 **92.0268%**，較半輪控制組下降 **0.6220 個百分點**，三類 F1 均下降。使用者於同日回報 Kaggle **0.8989**，未登入平台獨立核驗；保留對照，不更換現有候選。

假設：降低 BERT 底層更新幅度能保留預訓練表示，改善泛化。共同對照為 `weighted_half_epoch_control`，唯一主要變因為 AdamW 各層學習率。encoder 第 11 層 2e-5，第 i 層為 `2e-5 * 0.9^(11-i)`，embeddings 為 `2e-5 * 0.9^12`；pooler 與分類 head 均為 2e-5。所有參數沿用 weight decay 0.01，包含 bias／LayerNorm，不額外變更 decay 規則。

其餘沿用控制組：指定 BERT 基底、固定 8,100／900 切分、seed 42、反平方根類別權重、空白前處理、batch 2／累積 8、max_length 64、warmup 10%、linear scheduler、AMP、梯度檢查點、3 epochs、每半輪評估。預設 decay=1 保留既有單一參數組行為，只有本輪明確指定 0.9。

決策依據：最佳 validation Macro-F1、三類 precision／recall／F1、錯判數與控制組差異；≥0.3 個百分點為後續確認優先門檻，不是顯著性判定。無論驗證是否改善皆產生真實模型 CSV，內容完全相同則不重複提交；Kaggle 在收到回報前維持未知，不代為提交。

預算與停止條件：最多 1 次完整訓練、3 epochs、45 分鐘；測試失敗、來源／切分不符、非有限 loss、重載失敗或超時即停。完成條件：參數無遺漏／重複、正確分層比例與 scheduler 測試、六點曲線、checkpoint 重載、AMP／FP32 驗證、1,000 筆 CSV、報告及 registry／snapshot 核對。唯一輸出 `outputs/weighted_layer_lr/`。啟動後核心完整訓練預算累計 3／9，不自動開始 FGSM 或額外切分訓練。

```powershell
.venv/Scripts/python.exe -u main.py train --device cuda --seed 42 --epochs 3 --eval-interval 0.5 --batch-size 2 --accumulation-steps 8 --learning-rate 2e-5 --layer-lr-decay 0.9 --class-weighting sqrt-inverse --max-length 64 --preprocessing whitespace-v1 --output-dir outputs/weighted_layer_lr --checkpoint outputs/weighted_layer_lr/checkpoint
```

已完成：查重與既有 diff 檢視，沿用控制組與現有精度分析工具；未重訓既有設定。`.venv/Scripts/python.exe -m unittest tests.test_layer_lr tests.test_training tests.test_model_source` 共 20 項通過，包含真實小型 BERT 的參數覆蓋、層級比例、scheduler warmup／decay、預設 AdamW 更新完全一致，以及未知參數拒絕處理。

訓練前 config 對照通過：既有欄位只有 `training_code_sha256` 改變，新增 `layer_lr_decay` 與 `optimizer_groups`。保存全部參數名稱、數量、起始 LR、weight decay，以及每個評估點各組的下一步 LR。12 encoder 層、embeddings 與 head／pooler 共 14 組；最小 LR 約 5.64859e-6。

執行證據：[啟動器](../../outputs/run_layer_lr.py)、[狀態](../../outputs/weighted_layer_lr.execution.json)、[日誌](../../outputs/weighted_layer_lr.log)。啟動器以 2,700 秒硬性限制子程序；核心預算已用 3／9。原 snapshot 備份於 `outputs/pre_layer_lr_snapshot.json`，供收尾核對既有產物不變。

## 結果與對照

| Epoch | Train Macro-F1 | Validation Macro-F1 | 成功／嘗試更新 |
|---:|---:|---:|---:|
| 0.5 | 86.3429% | 87.4109% | 252／254 |
| 1.0 | 91.8464% | 90.4135% | 504／507 |
| 1.5 | 92.3296% | 88.0809% | 758／761 |
| 2.0 | 95.5047% | 90.4554% | 1,011／1,014 |
| 2.5 | 95.9242% | 91.0942% | 1,265／1,268 |
| 3.0 | 96.7265% | **92.0268%** | 1,517／1,521 |

最佳 checkpoint 為 epoch 3；[history](../../outputs/weighted_layer_lr/history.json) 的六個評估點與 [曲線](../../outputs/weighted_layer_lr/curves.png) 已核對，曲線的點數、圖例與座標軸目視正常。train loss 下降而 validation loss 波動，不能以訓練表現取代泛化比較。

| 指標 | 半輪控制組 | 分層學習率 B | 差值（百分點） |
|---|---:|---:|---:|
| Macro-F1 | 92.6488% | 92.0268% | -0.6220 |
| negative F1 | 97.0359% | 96.4981% | -0.5378 |
| neutral F1 | 86.4662% | 85.9316% | -0.5346 |
| positive F1 | 94.4444% | 93.6508% | -0.7937 |
| 錯判筆數 | 44／900 | 49／900 | 多 5 筆 |

| B 類別 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 95.5316% | 97.4843% | 96.4981% | 636 |
| neutral | 87.5969% | 84.3284% | 85.9316% | 134 |
| positive | 96.7213% | 90.7692% | 93.6508% | 130 |

混淆矩陣按 negative／neutral／positive 排列，列為真實、欄為預測：`[[620,13,3],[20,113,1],[9,3,118]]`。neutral precision 略升，但 recall 下降；本輪未支持改善泛化的假設。

[summary](../../outputs/weighted_layer_lr/summary.json) 記錄重載成功、總耗時 **1,259.02 秒（約 20.98 分鐘）**、CUDA peak allocated **1,904.24 MiB**。1,521 次更新嘗試中有 4 次 AMP skip，成功更新與 scheduler 均為 1,517 步。各組每個評估點的實際 LR 均與相同 warmup／linear decay 係數相符；skip 時間差是改變訓練更新後的結果，沒有手動變更其他設定。

## 重載、精度與 CSV

執行 `.venv/Scripts/python.exe -u -c "import runpy; runpy.run_path('outputs/check_layer_lr.py',run_name='__main__')"` 完成 FP32 benchmark 推論、兩次驗證精度推論與設定／scheduler 核對。其 benchmark 指令等同：

```powershell
.venv/Scripts/python.exe main.py predict --device cuda --batch-size 2 --checkpoint outputs/weighted_layer_lr/checkpoint --predictions outputs/weighted_layer_lr/predictions.csv
```

訓練內重載與獨立 AMP 驗證均重現保存的 loss／混淆矩陣。AMP／FP32 各 900 筆，標籤差異 **0**、Macro-F1 同為 92.0268%，錯判同為 49 筆；機率最大差 0.00248966、平均差 0.00002729。證據：[設定與精度比較](../../outputs/weighted_layer_lr/control_comparison.json)、[逐筆機率](../../outputs/weighted_layer_lr/precision/probabilities.csv)、[錯判紀錄](../../outputs/weighted_layer_lr/precision/errors.csv)。未額外執行 benchmark AMP 推論。

[CSV](../../outputs/weighted_layer_lr/predictions.csv) 已核對 1,000 筆、`id,pred`、原始 ID 順序及合法標籤。與共同控制組相差 **16 筆**、目前 Kaggle 首選 weighted_quarter 相差 **21 筆**、原候選 weighted_lr3e5 相差 **25 筆**，且不同於所有既有 CSV。使用者已回報此版本 Kaggle 分數，未代為提交或登入平台核驗。

- CSV SHA-256：`47c39add40d1d68f6a2b4cb85f061add88bec97a5c2cf2c656dab5d4f043e9c1`
- 模型權重 SHA-256：`e6c07817b5ead0bf1b57ad0bde81aecf3f8750977ecd310919ad3accc7c5826d`
- 基底、資料、切分與程式來源：[run config](../../outputs/weighted_layer_lr/run_config.json)；精度 CSV hash 保存於比較證據中。

## 決策與交接

B 未達原定 +0.3 個百分點的確認優先門檻，不追加 decay 搜尋。驗證基準與 Kaggle 首選維持不變，release_candidate 仍為 null；Kaggle 回報亦未改善，驗證超過 92% 不代表課程目標達成。

已完成本輪唯一一次三輪訓練、六點曲線、重載、FP32 benchmark CSV 與 AMP／FP32 驗證。核心完整訓練預算已用 **3／9**，剩餘上限 6 次。下一個具體工作為 C：實作 embedding FGSM，先依訓練 batch 檢查擾動 mask、尺度、有限梯度與 AMP／累積縮放，固定最多兩個 epsilon 後才安排正式對照。本輪未開始 FGSM 或額外切分訓練。

## 訓練完成時的收尾檢核（Kaggle 回報前）

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：已讀 AGENTS，變更限 HW3，指定基底與固定課程切分不變。 |
| 目的完成 | 通過：分層學習率實作、一次三輪訓練、六點曲線、最佳模型重載與 CSV 完成。 |
| 查重與成本 | 通過：沿用控制組與現有工具，只改預定 LR 因素；核心預算 3／9，未追加訓練。 |
| 既有工作保全 | 通過：先看 diff，保留既有未提交修改；收尾 snapshot 比較確認原 9 份 checkpoint、11 個候選及其他歷史產物紀錄完全相同。 |
| 驗證證據 | 通過：20 項相關測試、實際 scheduler 比例、模型重載、900 筆跨精度標籤一致、1,000 筆 CSV 及曲線目視核對。 |
| 實驗可追溯 | 通過：設定／來源 hash 核對、registry 登錄與 refresh／--check，12 runs／10 checkpoints 及全部 CSV 一致。 |
| 文件與狀態 | 通過：README、計畫、交件清單、索引、報告與總表同步；本機連結與 diff 空白檢查通過。 |
| 交付誠實 | 通過：如實記錄下降、Kaggle 未知及未提交；未宣稱達標或正式交件，下一工作尚未啟動。 |

檢核：適用項 **8／8 通過**；不適用：無。

## 2026/11/15 Kaggle 回報登錄

使用者回報 `weighted_layer_lr` 為 **0.8989**，比共同控制組所對應的 weighted_sqrt 0.9049 低 **0.60 個百分點**，比目前首選 weighted_quarter 0.9122 低 **1.33 個百分點**。本機驗證與 Kaggle 回報都未改善，維持停止擴大 decay 搜尋的決策；驗證基準、Kaggle 首選與 release_candidate 不變。

registry 已標示 `user_reported` 與回報日期，未宣稱平台獨立核驗。這次只更新分數與文件，沒有新增訓練、推論或提交；核心預算仍為 **3／9**。下一項仍是 FGSM 實作與短跑校準。

本次登錄檢核：適用項 **8／8 通過**；不適用：無。規範與範圍：已讀 AGENTS，修改限 HW3；目的完成：分數、來源與決策已登錄；查重與成本：沿用既有模型與 CSV；既有工作保全：已看 diff，前後 snapshot 確認模型／CSV／來源紀錄未變；驗證證據：refresh、--check、本機連結及 diff 檢查通過；實驗可追溯：12 runs／10 checkpoints 核對一致；文件與狀態：報告、計畫與總表同步；交付誠實：使用者回報與平台核驗分開記錄，未宣稱達標或交件完成。
