# BERT 基準流程短跑紀錄

日期：2026/11/12。狀態：訓練、儲存、重載、推論流程已通過；尚未執行完整基準訓練。

## 環境與設定

- Python 3.12.10，PyTorch 2.9.1+cu126，CUDA runtime 12.6。
- NVIDIA GeForce RTX 3050 Laptop GPU，4 GB。
- Transformers 4.57.3；完整環境版本見 `requirements-lock.txt`。
- 指定基底 `google-bert/bert-base-uncased`，revision `86b5e0934494bd15c9632b12f734a8a67f723594`。
- 固定 EDA 切分內每類抽 16 筆，訓練／驗證各 48 筆，seed 42。
- 1 epoch，max length 64，micro-batch 2，累積 8，有效 batch 16。
- AdamW，learning rate 2e-5，weight decay 0.01，未加權 cross-entropy。
- 啟用 FP16 AMP、梯度檢查點、動態 padding。短跑總共 3 個 optimizer steps，10% warmup 向下取整為 0；完整基準則按完整步數計算。

## 實測結果

| 項目 | 結果 |
|---|---:|
| 成功 optimizer 更新 | 3 |
| PyTorch 峰值配置顯存 | 1,902.55 MiB（約 1.86 GiB） |
| Epoch 訓練與兩側評估耗時 | 2.6 秒 |
| 訓練迴圈至儲存、畫圖及重載核對 | 7.21 秒 |
| Train loss / Macro-F1 | 1.06588 / 0.34074 |
| Validation loss / Macro-F1 | 1.09344 / 0.36027 |
| Checkpoint 重載核對 | loss 與混淆矩陣一致 |

耗時不含環境安裝、下載與初始模型載入；顯存是 PyTorch 配置量，不是含桌面程式的整張 GPU 使用量。短跑樣本小、僅更新 3 次，F1 只作流程驗證，不代表完整基準效果，不能據此推估 Kaggle 成績。

## 驗證與產物

- 6 項單元測試通過：固定切分重現性、重複文字衝突、切分比例檢查、尾端不足 batch 的累積梯度等價性、Macro-F1 包含缺席類別、資料雜湊／ID 重疊檢查。
- `pip check` 通過；曲線產生後已檢視。
- 重載短跑 checkpoint 後推論 benchmark 全部 1,000 筆，驗證 `id,pred` 欄名、唯一 ID、原始列順序及合法小寫標籤。
- `outputs/smoke/` 保存 checkpoint、指標、曲線、設定、切分副本、環境、摘要、預測及 `prediction_check.json`。此目錄由 Git 忽略。
- `saved_models/checkpoint/` 尚未放入正式模型。下一步使用完整 8,100／900 切分執行 3 epochs 基準訓練，以完整驗證集 Macro-F1 選模型。

重跑方式及完整訓練指令見 README；重跑需指定新的 output 與 checkpoint 目錄。
