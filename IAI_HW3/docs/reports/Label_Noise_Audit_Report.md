# OOF 高信心不一致樣本的錯標抽查

日期：2026/11/16。狀態：已完成（不訓練的後處理分析；產物在 `outputs/label_noise_audit_20261116/`，不寫進 registry 的 runs）。對應 [92% 計畫](../Target_92_Plan.md)「路線調整（2026-11-16，第二次）」順序 2。

## 目的

檢驗 `cv5_sqrt_ep5` 的 OOF 中「高信心卻與標籤不同」的樣本是否多為錯標，以決定是否值得做一組錯標降權（排除）CV。

## 方法

- 來源：`outputs/cv5_sqrt_ep5_oof/oof_predictions.csv`（9,000 筆，SHA-256 `2bd407bd19687c233dc3e6c9839cac63270d221e24909ba9059cd580427b37a0`）。
- 集合定義：D = pred ≠ label（625 筆）；高信心集合 H = D 中 p_pred ≥ 0.90（473 筆）。
- 抽樣（seed 20261116，打亂後盲標）：
  - A 組：H 中 label = neutral，依 p_pred 取前 40。
  - B 組：H 中其他類別隨機 20。
  - C 組（對照）：pred = label 隨機 20，用來估計盲標者可靠度。
- 盲標：盲標者為 Claude opus-reviewer 子代理，只讀 `review_blind.csv`（僅 `review_id` 與文字，不含標籤與模型預測），輸出 `review_annotations.csv`（annotation、confidence、reason）。
- 工具：`tools/audit_label_noise.py`（抽樣與產生檔案）、`outputs/label_noise_audit_20261116/score_annotations.py`（計分）。

## 預先登記的判定規則

- C 組一致率 ≥ 80% 才採信盲標。
- A+B 中 annotation ≠ label 的比例 ≥ 50% 且 |H| ≥ 90 → 值得一組 CV 降權。
- 比例 < 30% → 放棄；介於 30–50% → 再討論。

## 結果

數字取自 `summary.json` 與 `audit_scores.json`。

不一致集合依門檻與 (label → pred) 拆分（|D| = 625，p_pred 門檻以 `summary.json` 為準）：

| 門檻 | 總數 | neg→neu | neg→pos | neu→neg | neu→pos | pos→neg | pos→neu |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部不一致 | 625 | 170 | 57 | 219 | 57 | 63 | 59 |
| p_pred ≥ 0.70 | 566 | 148 | 54 | 202 | 51 | 57 | 54 |
| p_pred ≥ 0.80 | 528 | 136 | 47 | 192 | 49 | 55 | 49 |
| **p_pred ≥ 0.90（H）** | **473** | 119 | 39 | 176 | 42 | 52 | 45 |
| p_pred ≥ 0.95 | 424 | 100 | 34 | 161 | 40 | 50 | 39 |

各組盲標結果：

| 組 | n | annotation = label | annotation = pred | 兩者皆非 |
|---|---:|---:|---:|---:|
| A | 40 | 14（35%） | 24（60%） | 2 |
| B | 20 | 10（50%） | 10（50%） | 0 |
| C | 20 | 20（100%） | 20（100%） | 0 |
| A+B | 60 | 24（40%） | 34（56.7%） | 2 |

A+B 中 annotation ≠ label：**36/60 = 60%**；其中盲標者 high confidence 19 筆（31.7%）、low confidence 17 筆（28.3%）。

(label, pred, annotation) 組合：

| 組 | label | pred | annotation | 筆數 |
|---|---|---|---|---:|
| A | neutral | negative | negative | 15 |
| A | neutral | negative | neutral | 13 |
| A | neutral | negative | positive | 1 |
| A | neutral | positive | negative | 1 |
| A | neutral | positive | neutral | 1 |
| A | neutral | positive | positive | 9 |
| B | negative | neutral | negative | 3 |
| B | negative | neutral | neutral | 8 |
| B | negative | positive | negative | 4 |
| B | negative | positive | positive | 1 |
| B | positive | negative | negative | 1 |
| B | positive | negative | positive | 2 |
| B | positive | neutral | positive | 1 |

## 判定

依預先登記規則**通過**：C 組一致率 100% ≥ 80%（盲標可採信）；A+B annotation ≠ label 為 60% ≥ 50%；|H| = 473 ≥ 90。故值得一組 CV 降權。使用者選擇「排除版本（w = 0）」，預先登記見計畫的 `cv5_sqrt_ep5_drop90`。

## 重要發現與限制

標示「已驗證」為本次數據直接支持；「推測」為尚未檢驗的解釋。

1. 已驗證：A 組有 14 筆原標籤 neutral、盲標亦為 neutral，模型卻以約 0.999 信心判 negative（13 筆）或 positive（1 筆），多為單純詢問（例：詢問第二件行李費、付費優先登機）。推測：模型對提到行李／航班／改票的 neutral 問句有系統性負面偏誤，neutral 弱點不全是錯標。（已由追加檢查修正，見下節「追加檢查：neutral 問句的高信心誤判」。）
2. 已驗證：可疑樣本（annotation ≠ label）中近半為盲標者 low confidence，較多屬模糊而非明顯錯標；只計 high confidence 約 32%（19/60），低於 50% 門檻。判定採預先登記的 60%（含 low confidence），但若只計 high confidence 則落在「再討論」區間，使用者知悉後仍選擇排除版本。
3. C 組皆為模型與標籤一致的容易樣本，盲標可靠度 100% 偏樂觀，不代表在困難樣本上也同樣可靠。
4. 盲標者為 LLM，標註只用於估計比例，**不作為訓練標籤**，也不用來改寫任何標籤（作業紅線）；後續只依 OOF 信心排除訓練樣本，不使用 annotation。
5. 樣本數小（60 筆），比例的抽樣誤差大。
6. 記錄：計分腳本 `ambiguous` 欄原誤定義為 label = pred = annotation，已於同日（2026/11/16）修正為 annotation = "ambiguous" 的筆數並重跑覆寫 `audit_scores.json`（同一輪分析的計分修正，非新實驗）；與舊版逐行比對，只有 C 組 `ambiguous` 由 20 變為 0，其餘數字不變。

## 追加檢查：neutral 問句的高信心誤判（2026-11-16）

目的：檢驗「重要發現」第 (1) 點的推測（模型對 neutral 問句有系統性負面偏誤）。資料來源 `outputs/question_bias_check_20261116/results.json`（腳本 `check_question_bias.py`；只用 `dataset.csv` 與 OOF，未用 benchmark）。以下數字已與 results.json 核對。

**問句 vs 非問句（OOF，cv5_sqrt_ep5）**

| 定義 | 組 | 筆數 | neutral recall | neutral→negative |
|---|---|---:|---:|---:|
| 含 '?' | 問句 | 2,230 | 0.852 | 13.7%（80/582） |
| 含 '?' | 非問句 | 6,770 | 0.748 | 18.5%（139/753） |
| 開頭詞（how/what/why/when/where/is/are/can/could/do/does/will/any） | 問句 | 1,020 | 0.861 | 12.1% |
| 開頭詞 | 非問句 | 7,980 | 0.772 | 17.8% |

**問句中含關鍵詞者的訓練標籤分布（含 '?' 定義）**

| 關鍵詞 | 筆數 | negative 比例 |
|---|---:|---:|
| bag | 212 | 85.4% |
| fee／$ | 86 | 86.0% |
| boarding／priority | 39 | 74.4% |
| rebook 類 | 60 | 93.3% |
| cancel | 234 | 85.9% |

**14 筆（原標籤 neutral、盲標 neutral、模型高信心判錯）的 TF-IDF 近鄰**：前 10 近鄰合計 neg／neu／pos = 89／31／20（negative 64%），平均相似度 0.138。對照組（隨機 14 筆判對的 neutral）為 52／69／19，平均相似度 0.258（含 1 筆近重複文）。

**判讀（推測，非已驗證）**

- (a) 模型對問句的 neutral recall 反而較高，「模型對 neutral 問句有系統性負面偏誤」的推測不成立。
- (b) 訓練資料中提到行李、費用、改票、取消的問句絕大多數被標為 negative，這 14 筆是其鄰域中的少數標法，模型判 negative 是依循資料多數的標註習慣。benchmark 推測與訓練集同一標註來源，若如此，強行改判 neutral 不會提高分數，不值得另外處理。近鄰相似度偏低，此點證據強度屬中等。
- (c) 因此盲標者（LLM）的「標籤正確／可疑」反映的是它自己的標註準則，與資料集標註者的習慣不完全一致；60% 可疑比例同時包含錯標與標註習慣差異。
- (d) 對 `cv5_sqrt_ep5_drop90` 的影響：原先擔心排除這些 neutral 問句會加重偏誤，此風險因 (a)(b) 而降低，預先登記不變。

`results.json` SHA-256：`26945451f7e90d52c2df0abea32a9893e4442f4d15e0e0ac6ff0d249b960dc18`；`check_question_bias.py`：`f4b879067428dd4af122d8aeaec7b7c8a3f06c224cd4b02b623949fd9cced51b`。

## 產物與 SHA-256

目錄：`outputs/label_noise_audit_20261116/`（相對專案根 `IAI_HW3/`）。

| 檔案 | SHA-256 |
|---|---|
| `review_blind.csv` | `0260e5933e5aaf8ab8cd679691fd6fdbe797aafdc4e26f19153e183b32f15e93` |
| `review_key.csv` | `152927f64bcae212ac642c8e5274f083d723bdb1bd9ae884879d0fda5cd0139e` |
| `review_annotations.csv` | `c30792acc093bcf4f137dac434bb484f982cad92151d3bd87781b70ac5630d02` |
| `candidates_high_conf.csv` | `10025261eb682e8b19815642cba390b4fc63cdc9ea9fb81ab007418a907a6546` |
| `audit_scores.json` | `f1946b73710ca852a3447eda9301004b2068ada0096f6b2422e921b3843cae06` |
| `summary.json` | `44a77b9594878b2c5fdd6420865c9609025f9530b55c9e2cb8973201ec746d9e` |
| `score_annotations.py` | `ac1df15dca12f1f36a85c568ff67acda3853862f5eaae6a8795aa1400e190ccb` |
| `tools/audit_label_noise.py` | `78e894cdecfdf219f2083abcb246b5a8876ed79f9fd662e741a5e01b6571e37e` |

`exclude_ids_p090.txt`（473 筆排除清單）由後續訓練前步驟產生，產生後另行記錄其 SHA-256。

## 重現

```powershell
.venv/Scripts/python.exe -m tools.audit_label_noise --output-dir outputs/label_noise_audit_20261116   # 抽樣與產生 review_blind／review_key；輸出目錄須為新的
.venv/Scripts/python.exe outputs/label_noise_audit_20261116/score_annotations.py
```
