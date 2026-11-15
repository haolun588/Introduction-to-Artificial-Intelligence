# 半輪評估控制組與訓練迴圈檢查

日期：2026/11/15。狀態：完成一次 3 epochs 控制組、六個評估點、重載／精度核對及正式 CSV；中斷後只補完收尾驗證，未重新訓練。最佳仍為 epoch 3，與舊 `weighted_sqrt` 的權重及 CSV 逐位元組相同。

## 假設、對照與預算

假設：原本僅在整數 epoch 選模，可能漏掉中途較好的驗證 checkpoint。對照沿用 `weighted_sqrt`：指定 BERT 基底、固定 8,100／900 切分、seed 42、原空白前處理、max_length 64、batch 2／累積 8、AdamW 2e-5、weight decay 0.01、10% warmup／linear decay、反平方根加權 CE、AMP 與梯度檢查點。唯一主要變因為評估與最佳 checkpoint 選取頻率由 1 epoch 改成 0.5 epoch；不同時改 loss、optimizer 或資料。

評估對齊至半輪後第一個完整梯度累積窗口，絕不切斷尚未完成的累積。正式每輪 4,050 batches，在 batch 2,032（約 0.501728 epoch）及 4,050 評估，名義 epoch 顯示 0.5／1.0；實際進度也記入 history。每點保存完整 train／validation 指標與曲線；僅當 validation Macro-F1 嚴格提升才更新最佳 checkpoint，同分保留較早者。

決策資訊：六個評估點的驗證表現、最佳點是否落於半輪、整數 epoch 對照、各類 F1、錯判數及新 CSV 與目前 Kaggle 候選的不同筆數。額外選模機會不能直接宣稱對舊流程公平提升；本輪建立後續 A／B／FGSM 共用的新控制組，不宣稱必然改善 Kaggle。

上限：一次完整訓練、最多 3 epochs／45 分鐘；必要的單元／小型 CPU 整合測試先行，通過後直接正式訓練，不另重複 BERT smoke。完整訓練後做一次正式 FP32 benchmark 推論，並對新 checkpoint 補做 AMP／FP32 驗證核對。停止條件：來源／切分不符、測試失敗、非有限 loss、重載不符或超出時間上限；保留已完成產物並先處理原因，不自動開始其他設定。

輸出：`outputs/weighted_half_epoch_control/`，checkpoint 與 CSV 使用全新路徑。此輪占核心 9 次完整訓練預算中的第 1 次；既有模型、演練 ZIP 與 Kaggle 候選均保留，平台分數未知維持 `null`。

## 迴圈檢查要點

- 以當前 source 核對，未找到需要先重跑的既有模型行為 bug。類別權重僅由 train 計算，平均樣本權重為 1；加權 loss 除以整個實際累積窗口的樣本數，涵蓋最後不滿窗口。
- 先 unscale、clip，再 scaler step／update；僅在 scale 未下降時推進 scheduler。計畫每輪 507、總 1,521 次 optimizer 嘗試；舊 `weighted_sqrt` 記錄 1,518 次成功更新，與少量 AMP skip 相容，但舊紀錄沒有逐次 skip 證據，不追補為已驗證原因。新輪會明確記錄嘗試／成功／skip 數與 scheduler 步數。
- 10% warmup 與 linear decay 以預定更新數規劃、以實際成功更新前進；有 skip 時最終 LR 可能未精確到 0，不暗改既有策略。既有 AdamW 對全部參數套用 weight decay，亦保留此設定作控制。
- 固定切分驗證資料 hash、ID 完整分割與正規化文字不洩漏；標籤固定 negative／neutral／positive；選模使用 validation F1，驗證 loss 為未加權 CE。
- 半輪評估需保留原訓練模式與 Torch RNG；新增測試驗證額外評估不改變小型 dropout 模型的訓練結果，並驗證中途最佳模型的選取／重載。

## 已完成實作與測試

新增 `--eval-interval {0.5,1.0}`，預設 1.0 保持既有 CLI 使用方式；本輪及後續新方法明確使用 0.5。`evaluate()` 以 `finally` 恢復模型模式，並保留 CPU／當前 CUDA 裝置 Torch RNG；訓練、驗證 DataLoader 維持獨立 generator。history 記錄名義／實際 epoch、batch 位置、嘗試／成功／skip 次數、scheduler 步數與下一步 LR；選定模型支援非整數 `best_epoch`。

`python -m unittest discover -p "test_*.py"`：27／27 通過。新增覆蓋窗口對齊與短 epoch、評估成功／失敗時的 RNG／模式恢復、小型 dropout 模型全輪／半輪評估的最終參數逐位相同、半輪同分最佳保留及 checkpoint 重載、模擬 overflow skip 的 scheduler 計數。其餘原有 loss、資料切分、來源驗證與前處理測試也通過。上述是小型測試，非新增 BERT 訓練實驗。

正式命令（HW3 根目錄）：

```powershell
.venv/Scripts/python.exe main.py train --device cuda --seed 42 --epochs 3 --eval-interval 0.5 --batch-size 2 --accumulation-steps 8 --learning-rate 2e-5 --class-weighting sqrt-inverse --max-length 64 --preprocessing whitespace-v1 --output-dir outputs/weighted_half_epoch_control --checkpoint outputs/weighted_half_epoch_control/checkpoint
```

啟動器設定 2,700 秒上限，狀態、完整參數與恢復時間記於 [execution.json](../../outputs/weighted_half_epoch_control.execution.json)；訓練輸出保存於 [日誌](../../outputs/weighted_half_epoch_control.log)。中斷時若缺 summary，須先核對程序及 history，不直接另啟同名訓練；本輪 summary 已完成後補驗證。

## 完整訓練結果

| 名義 epoch | Train Macro-F1 | Validation Macro-F1 | 成功更新／嘗試 | 累計 AMP skip |
|---:|---:|---:|---:|---:|
| 0.5 | 86.7388% | 87.2463% | 253／254 | 1 |
| 1.0 | 92.9119% | 90.6718% | 505／507 | 2 |
| 1.5 | 94.8921% | 89.5860% | 759／761 | 2 |
| 2.0 | 96.9204% | 89.7376% | 1,012／1,014 | 2 |
| 2.5 | 97.5356% | 90.6523% | 1,266／1,268 | 2 |
| 3.0 | 98.1060% | **92.6488%** | 1,518／1,521 | 3 |

最佳 checkpoint 為 epoch 3；validation 錯判 44／900，三類 F1 分別為 negative 97.0359%、neutral 86.4662%、positive 94.4444%。三個整數 epoch 的 train／validation 完整指標與成功更新數均重現舊控制組。最終 `model.safetensors` SHA-256 同為 `e03e0bafe1209e66b5dd7e58a0e79633cf820b55dd51f21c934cbcdb649e8d07`。

完整 [history](../../outputs/weighted_half_epoch_control/history.json)、[曲線](../../outputs/weighted_half_epoch_control/curves.png)、[比較證據](../../outputs/weighted_half_epoch_control/control_comparison.json) 已保存；曲線已目視核對六個點、train／validation 圖例及軸標。本輪 1,521 次嘗試中明確記錄 3 次 skip，scheduler 只前進 1,518 步，最後下一步 LR 為 `4.3827611395178965e-08`，符合原先保留的策略。

23 個舊 run config 欄位全部相同，新增欄位只涉及已驗證來源、評估頻率及觀測計數；[設定對照](../../outputs/weighted_half_epoch_control/control_design_check.json) 已保存。此結果支持本輪中途評估沒有改變原訓練軌跡，不能推論所有硬體或後續方法都會逐位重現。

## 中斷恢復與推論

用量中斷後恢復時，原訓練程序已不存在，六個評估點與最佳 checkpoint 已落盤，但最後重載驗證及 summary 未保存，原啟動器狀態仍為 running。保留 [原執行紀錄](../../outputs/weighted_half_epoch_control/recovery/original_execution.json)，用 [本輪收尾腳本](../../outputs/weighted_half_epoch_control/finish_interrupted_run.py) 只載入保存的最佳模型，核對來源、完整 history、選模及 AMP／FP32 驗證後補寫 [summary](../../outputs/weighted_half_epoch_control/summary.json)。沒有再次執行 optimizer 或重跑任何 epoch。

原程序退出碼、全程序總耗時及訓練尖峰記憶體未落盤，維持未知（`null`），不補造為正常退出。三輪 history 所記錄的 epoch 時間合計 **1,175.59 秒（約 19.59 分鐘）**，是已知訓練／評估區間時間，不包含中斷等待或後補驗證，也不當作速度提升證據。

兩種精度的驗證均為 92.6488%、44 筆錯判，逐筆標籤差異 **0**；最大機率差 0.00424320，平均差 0.00002410。AMP loss／混淆矩陣重現保存值。保留 AMP 訓練／驗證、FP32 正式推論；證據見 [precision_summary](../../outputs/weighted_half_epoch_control/precision_summary.json) 與 [逐筆機率](../../outputs/weighted_half_epoch_control/recovery/probabilities.csv)。

正式推論命令：

```powershell
.venv/Scripts/python.exe main.py predict --device cuda --batch-size 2 --checkpoint outputs/weighted_half_epoch_control/checkpoint --predictions outputs/weighted_half_epoch_control/predictions.csv
```

已產生 [1,000 筆 CSV](../../outputs/weighted_half_epoch_control/predictions.csv)，格式、唯一 ID、原始順序與標籤均核對；SHA-256 為 `7330b4e257aa271b8132c4a919df7ba7eb1321f7f0e3d5739055100b74b07324`。與 `weighted_sqrt` 完全相同，與目前 Kaggle 候選 `weighted_lr3e5` 相差 15 筆。因此**不重複提交**；新 run 的平台分數保持 `null`，不將舊回報直接補為新提交紀錄。

## 決策與剩餘預算

此次半輪評估沒有找出優於整數 epoch 的模型，不能稱為性能提升。以 `weighted_half_epoch_control` 設為 registry 的 `comparison_control`，供後續較溫和權重、分層學習率與 FGSM 使用相同選模流程；`validation_reference` 與 `kaggle_candidate` 保留原角色，`release_candidate` 仍為 `null`。

核心完整訓練預算已完成 **1／9**；本輪後段實際執行 2 次驗證精度推論及 1 次 FP32 benchmark 推論，無額外 BERT smoke、無平台提交。下一個具體工作是設定 A：僅將類別權重由 `n^-0.5` 改為 `n^-0.25`，其餘沿用本控制組；本輪尚未實作或執行 A。

## 後續確認切分預先固定

依第一階段原計畫，現在預先指定額外切分 seed **142、242**，使用與主切分相同的 90／10 文字群組分層演算法；不根據任何新方法成績選 seed。只保存 manifest、ID 與 hash，不重做 token EDA、不啟動額外訓練。後續僅對第二階段最有希望的一個方法與本控制組各跑兩份（共 4 次）；若沒有候選則不執行。這些切分來自同一資料集且可能重疊，不是新獨立測試集；訓練 seed 仍為 42，與切分 seed 分開記錄。

兩份切分均為 8,100／900，通過 ID 完整分割、文字群組不跨 train／validation 及各類數量核對；與原 validation 重疊分別 91、103 筆，兩份額外 validation 互相重疊 87 筆。manifest 的完整 SHA-256 由 registry／snapshot 維護，摘要見 [confirmation_splits](../../outputs/confirmation_splits/summary.json)。

## 收尾檢核

| 項目 | 結果與證據 |
|---|---|
| 規範與範圍 | 通過：開始及中斷恢復時均讀 AGENTS；變更限 HW3，使用指定基底及課程切分。 |
| 目的完成 | 通過：迴圈檢查、半輪評估、一次控制組與 CSV 均完成；未宣稱提升分數。 |
| 查重與成本 | 通過：既有證據沿用，控制組為預定單一變因；恢復後只補驗證，未重訓。 |
| 既有工作保全 | 通過：新路徑保存產物，7 份舊 checkpoint 登錄及 9 份原 CSV hash 不變，原演練 ZIP 經索引核對不變；先前修改與 Emoji 回報保留。 |
| 驗證證據 | 通過：27 項測試、六點 history、重載、AMP／FP32 標籤、1,000 筆 CSV 與曲線檢查已通過。 |
| 實驗可追溯 | 通過：新 run／comparison_control／兩份確認切分已登錄；refresh 及 `--check` 通過，10 runs／8 checkpoints／全部 CSV 與恢復證據 hash 一致。 |
| 文件與狀態 | 通過：入口、計畫與交件清單已同步，6 份 Markdown 共 89 個本機連結存在；diff 無空白錯誤。 |
| 交付誠實 | 通過：完整訓練與後補驗證分開記錄；原程序未知值保留 null，沒有宣稱已提交或達 92% Kaggle。 |

檢核：適用項 8／8 通過；不適用：無。本輪控制組工作完成，設定 A 與後續階段仍待辦。
