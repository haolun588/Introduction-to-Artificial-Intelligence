# URL 正規化對照實驗

2026/11/13 完成。**本次不採用 URL 正規化**：最佳驗證 Macro-F1 **91.28%**，相較保留原 URL 的加權模型 **92.65%** 下降 **1.37 個百分點**，三類 F1 皆下降。繼續保留 `outputs/weighted_sqrt/checkpoint/`，其 Kaggle 分數為使用者回報的 0.9049。

## 實驗設定

- 從相同原始 BERT 重新訓練 3 epochs。已核對所有共同 run config 欄位，唯一差異是 `preprocessing` 從 `whitespace-v1` 改為 `url-http-v1`。
- 保留固定 8,100／900 切分、seed 42、反平方根類別權重、max length 64、micro-batch 2、累積 8、AdamW 2e-5、weight decay 0.01、10% warmup／linear decay、AMP 與梯度檢查點。
- 將以 HTTP／HTTPS／www. 開頭的 URL 換成既有單一 token `http`，保留尾端標點，再做空白正規化。不新增 tokenizer token、不改模型 vocabulary。
- 規則是適用本資料的啟發式 regex，不是完整 URL parser；尾端標點可能同時是合法 URL 的一部分。此次只測此規則，不能推論所有 URL 處理方式都無效。
- 訓練資料有 568 筆受到影響。設定保存於 checkpoint；推論自動沿用 checkpoint 的版本，而非 CLI 前處理選項。舊 checkpoint 仍使用 `whitespace-v1`。

## 每輪結果

| Epoch | Train loss | Val loss | Train Macro-F1 | Val Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 0.150802 | 0.240456 | 92.91% | 88.80% |
| 2（入選） | 0.064247 | 0.230953 | 97.26% | **91.28%** |
| 3 | 0.046920 | 0.270057 | 98.11% | 90.75% |

評估 loss 均為未加權 CE。總耗時 1,000.51 秒（16.68 分鐘），不含初始載入與 benchmark 推論。成功更新 1,517 次，AMP 略過 4 次；峰值配置顯存 1,903.90 MiB（約 1.86 GiB）。

## 與原加權模型比較

| 指標 | 保留原 URL | URL → http |
|---|---:|---:|
| Macro-F1 | **92.65%** | 91.28% |
| negative F1 | 97.04% | 96.33% |
| neutral F1 | 86.47% | 83.77% |
| positive F1 | 94.44% | 93.75% |
| neutral precision | 87.12% | 84.73% |
| neutral recall | 85.82% | 82.84% |

第 2 輪最佳模型混淆矩陣（列為真實、欄為預測，順序 negative／neutral／positive）：`[[616,16,4],[21,111,2],[6,4,120]]`。

## 正規化後文字重合與驗證

原始固定切分無相同空白正規化文字跨集合，但將 URL 合併為 `http` 後，有 3 種原本不同的文字變成跨集合相同文字，涉及驗證 ID `2225`、`3345`、`8984`。為維持實驗可比性未重新切分，也沒有使用驗證標籤訓練。此限制記錄於 `preprocessing_audit.json`；另在同時排除這 3 筆的 897 筆驗證資料上，比較兩個既有最佳模型作敏感度檢查：原加權模型 **92.54%**、URL 正規化 **91.16%**，結論不變。完整結果位於 `nonoverlap_comparison.json`，不改變原先模型選擇規則。

- 11 項測試通過，包括 URL／標點處理、無 URL 文字保持不變、舊版相容、未知模式拒絕、重複處理穩定、訓練與推論 token 一致，以及既有切分／加權梯度測試。
- 最佳 checkpoint 重載的 loss 與混淆矩陣核對通過，訓練曲線已檢視。
- `outputs/weighted_url/predictions.csv` 已產生 1,000 筆 `id,pred`，欄名、唯一 ID、原始順序與合法標籤檢查通過。使用者已提交並回報 Kaggle **0.8977（89.77%）**，比原加權模型 0.9049 低 **0.72 個百分點**；此為使用者回報，未由工具登入平台核驗。驗證與 Kaggle 的比較方向一致，保留原加權模型。
- 模型、tokenizer、設定、history、summary、曲線及核對紀錄皆保存在 `outputs/weighted_url/`，由 Git 忽略。
- 已用 SHA-256 確認原始基準與加權模型的權重及預測檔均未改動。

下一項回到保留原 URL 的加權設定，比較較低學習率 `1e-5`，其他條件固定。這是工作計畫中的訓練參數對照；先看固定驗證集，不假設一定能提高 Kaggle。
