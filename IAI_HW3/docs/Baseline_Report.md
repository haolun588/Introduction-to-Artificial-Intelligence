# 完整 BERT 基準實驗

2026/11/12 完成。最佳模型為第 3 epoch，固定 900 筆驗證集 Macro-F1 **0.915715（91.57%）**。使用者已回報此輪 Kaggle 分數 **0.8973（89.73%）**，比本機驗證低約 1.84 個百分點；此分數來自使用者回報，未由工具登入平台核驗。

## 設定

- 基底：`google-bert/bert-base-uncased`，revision `86b5e0934494bd15c9632b12f734a8a67f723594`。
- 沿用 EDA 固定切分，train 8,100／validation 900，seed 42；相同正規化文字不跨集合。
- 空白正規化，max length 64，動態 padding，micro-batch 2，梯度累積 8，有效 batch 16。
- AdamW、learning rate 2e-5、weight decay 0.01、10% linear warmup／decay、未加權 cross-entropy、3 epochs。
- RTX 3050 Laptop 4 GB；FP16 AMP、梯度檢查點。Python 3.12.10、PyTorch 2.9.1+cu126；其餘版本見 `requirements-lock.txt`。
- 訓練僅使用指定訓練切分；benchmark 只在選好模型後做推論。

## 每輪結果

以下 train／validation 指標均為該 epoch 結束後的 eval 模式評估。

| Epoch | Train loss | Val loss | Train Macro-F1 | Val Macro-F1 | 訓練與評估秒數 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.140705 | 0.227335 | 92.97% | 90.09% | 353.8 |
| 2 | 0.071873 | 0.245893 | 96.71% | 88.78% | 407.1 |
| 3（入選） | 0.049484 | 0.269419 | 98.04% | **91.57%** | 412.8 |

訓練迴圈、評估、儲存、畫圖與重載核對共 1,181.94 秒（19.70 分鐘），不含初始載入與後續 benchmark 推論。PyTorch 峰值配置顯存 1,904.24 MiB（1.86 GiB），不含其他應用程式。原定 1,521 次更新中成功 1,519 次；AMP 略過 2 次，scheduler 隨成功更新前進。

曲線位於 `outputs/baseline/curves.png`，已檢視。

## 最佳模型各類指標

| 類別 | Precision | Recall | F1 | 驗證筆數 |
|---|---:|---:|---:|---:|
| negative | 95.83% | 97.48% | 96.65% | 636 |
| neutral | 86.72% | 82.84% | 84.73% | 134 |
| positive | 95.20% | 91.54% | 93.33% | 130 |

混淆矩陣：列為真實標籤，欄為預測標籤。

| 真實／預測 | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 620 | 13 | 3 |
| neutral | 20 | 111 | 3 |
| positive | 7 | 4 | 119 |

## 驗證與交付

- 最佳模型保存在 `saved_models/checkpoint/`，含模型、tokenizer、類別對照、前處理與訓練設定。
- 訓練完成後重新載入 checkpoint，驗證 loss 與混淆矩陣核對通過。
- `outputs/baseline/predictions.csv` 已輸出全部 1,000 筆 benchmark 推論，欄位 `id,pred`、ID 唯一性、原始列順序與標籤合法性皆通過檢查。
- 預測分布：negative 734、neutral 125、positive 141。無 benchmark 真實標籤，無法在本機計算其分數。
- `outputs/baseline/` 保存 history、summary、環境、run config、split 副本及 `prediction_check.json`；檢查紀錄含預測檔與 checkpoint 的 SHA-256。
- 使用者已提交 Kaggle 並回報 0.8973；E3 提交狀態未確認。輸出及模型目錄由 Git 忽略，交件時須另外納入。

## 判讀與下一項工作

驗證 loss 逐輪上升，但第 3 輪 Macro-F1 最佳；兩個指標衡量的面向不同，依既定 Macro-F1 準則選第 3 輪。Train／validation 差距與 loss 趨勢顯示過擬合風險，不宜只靠增加 epochs。

Neutral 仍是主要弱項，134 筆中有 20 筆被判成 negative。下一項依工作計畫比較訓練集頻數的反平方根加權 cross-entropy，保持切分、seed、前處理及其他超參數一致，另存實驗與 checkpoint。比較完整 Macro-F1 及 neutral 的 precision／recall，不能僅因少數類 recall 增加就認定改善；本輪先保留為未加權基準。
