# FGSM C 實作與短跑校準

日期：2026/11/15。狀態：實作與校準完成；尚未執行完整訓練，沒有新的正式提交 CSV。

## 預先固定的目的與預算

- 假設：word embedding 的局部 sign-gradient 擾動可作為後續泛化對照；本次只檢查實作及可執行性，不以短跑 F1 推論泛化提升。
- 對照：沿用半輪控制組，sqrt-inverse、seed 42、2e-5、有效 batch 16、原切分與前處理；唯一主要變因為 clean／adversarial loss 等權的 FGSM。
- 尺度：從固定 train 子集（每類 16 筆）的非 padding／非特殊 token 原始 word embedding RMS，預先選 5% 與 10% RMS 為兩個 epsilon（6 位有效數字）。不依 validation 或 benchmark 分數選幅度。
- 檢查：tiny BERT 的參考梯度／累積測試；指定官方基底上以 train batch 做 AMP／FP32、mask、有限梯度與權重不變檢查；兩個 epsilon 各一次 48 train／48 validation、1 epoch smoke，僅作管線驗證。
- 上限：兩次 smoke、零次完整訓練；GPU 校準及 smoke 合計 15 分鐘。非有限梯度、mask／界限錯誤、checkpoint 重載不一致或超時即停止並記錄；修 bug 才允許有理由的重跑。
- 完成後提供兩個固定 epsilon、命令、設定／來源 hash、時間／顯存及下一項正式對照。既有候選與提交 CSV 保留。

## 方法來源

採 [Goodfellow 等 FGSM 論文](https://arxiv.org/abs/1412.6572) 的 sign-gradient，適配到 word embedding；不是 L2 FGM。AMP 的 `autograd.grad` 回傳梯度先除以 scaler scale，累積窗口完成後才 unscale optimizer，依 [PyTorch 2.9 AMP 範例](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html)。模型參數只由 optimizer 更新；攻擊 delta detached，不永久改寫 embedding table。

## 實作與校準結果

`main.py train --fgsm-epsilon` 預設 0，沿用原本 clean-only 路徑；大於 0 時先算 clean input gradient，再分別 backward 半份 clean loss 與半份 adversarial loss。兩份均除以實際累積窗口樣本數。第二次 embedding lookup 保留 adversarial loss 回傳到 embedding table 的路徑；不回傳穿過 delta，不對模型權重做原地攻擊。非有限 attack gradient 直接停止，不悄悄套用 NaN 的 sign。

固定 train 子集 48 筆的 word embedding RMS = **0.0416185535**。依事前規則固定：

| 設定 | epsilon | 相對 RMS | 用途 |
|---|---:|---:|---|
| C1 | 0.00208093 | 5% | 下一次正式對照 |
| C2 | 0.00416186 | 10% | 第二個預先限定的設定，尚未正式訓練 |

尺度與梯度只使用原切分 train IDs；兩次 smoke 的 validation 只檢查評估／重載，不拿 F1 選 epsilon。原始基底、資料／切分與程式 hash、ID 清單在 [完整校準紀錄](../../outputs/fgsm_precision_diagnostic_20261115/summary.json)。

## 精度診斷與限制

1. [第一次 train-mode 診斷](../../outputs/fgsm_calibration_20261115/summary.json) 做 3 batches × 2 epsilon × AMP／FP32，共 12 組 backward，mask、界限、有限參數梯度、權重版本不變及乾淨推論完全重現皆通過。但梯度符號一致率只有 66.73%–70.39%，此比較含 dropout，不足以單獨歸因於數值精度。
2. 關閉 dropout 的第一次診斷加了相對 L2 < 2% 的檢查，在首個 batch 停止。此嘗試保留於 [未完成紀錄](../../outputs/fgsm_precision_calibration_20261115/summary.json)，`completed=false`。這個 2% 數值等價檢查**未通過**。
3. 修正診斷工具為保存所有 batch 數值及獨立的 2% 判定後，完成相同 train batches 的 [deterministic 診斷](../../outputs/fgsm_precision_diagnostic_20261115/summary.json)。沒有變更 FGSM 訓練程式或兩個 epsilon；補跑目的是分離 dropout 並保留失敗證據，沒有增加完整訓練。

| Train batch | AMP／FP32 梯度符號一致率 | 梯度相對 L2 差異 |
|---|---:|---:|
| 0 | 99.2471% | 5.3756% |
| 1 | 99.9421% | 0.2083% |
| 2 | 99.9295% | 0.1986% |

兩個 epsilon 共用同一 clean gradient，故各 batch 的精度結果相同。關閉 dropout 後差異明顯縮小，支持原診斷混入隨機差異；仍不能宣稱 AMP／FP32 梯度等價，`precision_relative_l2_under_2pct=false`。FGSM 使用符號而非梯度幅度；本次觀察到最多約 0.75% 座標符號不同。決策為保留共同控制組的 AMP 訓練，將此誤差記為方法限制；不據此宣稱模型成效相同，也不為湊到 2% 放寬數值等價判定。後續正式候選仍應檢查 AMP／FP32 驗證標籤與 F1。

測試另以 tiny BERT、dropout=0 比較完整 batch 的獨立 joint-loss 參考梯度與實作的分次 backward／不完整 microbatch，並檢查 FP16 scaler 累積後 unscale 的參數梯度、epsilon=0 clean 梯度、padding／特殊 token mask 與不修改參數。這些是實作正確性檢查，不代表所有真實 BERT 梯度都通過 2% 數值等價檢查。

## 兩次 GPU smoke

環境：RTX 3050 Laptop GPU、PyTorch 2.9.1+cu126、Python 3.12.10；AMP 與 non-reentrant gradient checkpointing 開啟。每次從已驗證的官方基底開始，48 train／48 validation、1 epoch、batch 2、累積 8、半輪評估。smoke 的每類數量相同，因此 sqrt-inverse 權重皆為 1；不平衡加權的縮放另由非等權測試及使用完整 train 權重的真實 batch 診斷覆蓋。

| 項目 | C1 smoke | C2 smoke |
|---|---:|---:|
| Optimizer／scheduler updates | 3／3 | 3／3 |
| 跳過更新 | 0 | 0 |
| 評估次數 | 2 | 2 |
| 選定 epoch | 0.5 | 1.0 |
| 重載驗證 | 通過 | 通過 |
| 訓練及收尾時間（不含載入） | 7.00 秒 | 7.87 秒 |
| Peak allocated VRAM | 1,903.58 MiB | 1,903.58 MiB |

證據：[C1 summary](../../outputs/fgsm_smoke_low_20261115/summary.json)、[C1 config](../../outputs/fgsm_smoke_low_20261115/run_config.json)、[C1 curves](../../outputs/fgsm_smoke_low_20261115/curves.png)；[C2 summary](../../outputs/fgsm_smoke_high_20261115/summary.json)、[C2 config](../../outputs/fgsm_smoke_high_20261115/run_config.json)、[C2 curves](../../outputs/fgsm_smoke_high_20261115/curves.png)。短跑分數只保留在原始紀錄，不列入正式候選比較。

GPU 執行的三次診斷與兩次 smoke 均為秒級執行，未觸及 15 分鐘預算；另有 tiny-model 測試。兩次完成診斷核心運算分別為 2.82 秒、1.72 秒，中途停止的一次也已保留；沒有保存含 Python 啟動與模型載入的精確累計秒數。核心完整訓練仍為 **3／9**。

## 重現與下一項工作

以下從 HW3 目錄執行。既有產物不可覆寫，重現時使用新的輸出路徑。

```powershell
.venv/Scripts/python.exe -m unittest discover -p "test_*.py"
.venv/Scripts/python.exe -m tools.calibrate_fgsm --output-dir outputs/fgsm_calibration_reproduction
.venv/Scripts/python.exe main.py train --smoke --device cuda --eval-interval 0.5 --class-weighting sqrt-inverse --fgsm-epsilon 0.00208093 --output-dir outputs/fgsm_smoke_low_reproduction --checkpoint outputs/fgsm_smoke_low_reproduction/checkpoint
.venv/Scripts/python.exe main.py train --smoke --device cuda --eval-interval 0.5 --class-weighting sqrt-inverse --fgsm-epsilon 0.00416186 --output-dir outputs/fgsm_smoke_high_reproduction --checkpoint outputs/fgsm_smoke_high_reproduction/checkpoint
```

下一項 C1：測試 clean／adversarial 等權 FGSM 是否改善原 900 筆 holdout 的 Macro-F1 及三類 F1。唯一主要變因為 FGSM；其餘與 `weighted_half_epoch_control` 相同，仍使用 sqrt-inverse，**不與 quarter-inverse 或 layer LR 合併**。上限一次 3 epochs、60 分鐘；非有限 attack gradient／loss、無成功 optimizer 更新或重載不一致即停止。若完成，依既有規則產生 FP32 benchmark CSV、核對精度及對照差異，再記錄結果；不得用 Kaggle 回饋更改 epsilon。

下面命令**尚未執行**：

```powershell
.venv/Scripts/python.exe main.py train --device cuda --epochs 3 --eval-interval 0.5 --class-weighting sqrt-inverse --learning-rate 2e-5 --fgsm-epsilon 0.00208093 --output-dir outputs/weighted_fgsm_low --checkpoint outputs/weighted_fgsm_low/checkpoint
```

C2 的 epsilon 已固定，最多再一個完整設定，是否執行由 C1 結果與預算決定。候選、Kaggle 分數與 release_candidate 維持原狀。

## 收尾檢核

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：已讀 AGENTS，修改限 HW3；僅指定基底與課程資料，校準只用 train。 |
| 目的完成 | 通過：FGSM 實作、梯度／mask／累積測試、兩個固定 epsilon 與兩次 smoke；精度差異已量化並記錄限制。 |
| 查重與成本 | 通過：沿用控制組、切分與工具；只增加預定短跑，補診斷為排除 dropout 混淆，完整訓練仍 3／9。 |
| 既有工作保全 | 通過：開始時 Git 乾淨；所有新輸出獨立路徑，既有候選、checkpoint 與 CSV hash 前後一致。 |
| 驗證證據 | 通過：35 項測試、兩次重載及乾淨推論、索引與文件檢查；2% 梯度等價診斷失敗已如實保存，未稱為通過。 |
| 實驗可追溯 | 通過：registry 獨立 calibrations 區、snapshot 校準／smoke hash 與來源／設定；不混入正式 runs。 |
| 文件與狀態 | 通過：README、計畫、課程進度、索引與報告同步；下一項明確為 C1 正式對照。 |
| 交付誠實 | 通過：未以 smoke F1 宣稱泛化改善，未宣稱 AMP／FP32 梯度等價、Kaggle 達標或交件完成。 |

檢核：適用項 8／8 通過；不適用：無。2% 診斷的失敗是已保留的數值差異結果，不是通過項目；它不改變本次 functional 校準結論。
