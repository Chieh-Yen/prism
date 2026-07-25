# PRISM 8674 — Rebuttal 重寫策略與逐項提案

> 基於:44 頁論文全文(A.1/B/C.1–C.3 逐條驗證)、四位 reviewer 原始評論
> (`raw_review.txt`)、meta-review、以及 937 行 `rebuttal_draft.txt`。
> 本檔是重寫的依據與檢查清單;實際改動已套入 `rebuttal_draft.txt`。

---

## 0. 最關鍵發現:戰場是 SIGNIFICANCE,不是正確性

分項打分:

| Reviewer | Quality | Clarity | Significance | Originality | Rating | Conf |
|---|---|---|---|---|---|---|
| pCi8 | 3 | 3 | **2** | 3 | 3 | **4** |
| G3T9 | 3 | 3 | **2** | 3 | 3 | **4** |
| eQL6 | **2** | **2** | 3 | 3 | 3 | 3 |
| 8VrD | 3 | 3 | 3 | 3 | **4** | 3 |

兩位 conf-4 只有 Significance 給 2(其餘全 3)。pCi8 Questions 寫 N/A(已停止好奇、
進入評判)。他們的潛台詞一致:「這是好看的排序啟發式,我為何不直接跑 benchmark?」

**推論**:純技術正確性實驗(E4/E8/E11)不直接加 significance。要拉 2→3,每段都必須
ladder up 到「PRISM 能做 benchmark/CKA 做不到、而且有用的事」。

- pCi8-W2 原話:「not whether Q4 is actually good enough. You still need to run benchmarks.」
- pCi8-W1:「putting them in a unified framework, not discovering anything new.」
- G3T9-W1:「simply evaluate on the benchmark suites… another layer of descriptive analysis.」

AC 原話:「careful **narrowing of the claims** could move the paper closer to borderline accept」
→ narrowing 是 AC 明講的加分鍵。

---

## 1. 三支柱敘事(貫穿全篇,打 significance=2)

global response 開場 + 每個 significance 相關 thread 第一句都回到:

**PRISM 已從「排序啟發式」跨越成「可操作、免標籤、能做 benchmark/CKA 做不到之事的工具」。**

1. **可操作**:校準後回答「Q4 夠不夠好?」(LOO MAE 0.055 nats;reviewer 自己的兩個
   cell 23.24→0.003、266→0.193 落在 ε=0.1 正確側)。殺 pCi8-W2 / G3T9-W4 / AC-A。
2. **又便宜又可行動**:E12 成本 12–66× + E4 因果介入 + GGUF 閉環(Q6_K→head 軸→保留
   FP16→76.95→3.81,20×)。講成「診斷→行動→驗證」閉環,不是「又一層描述」。殺 G3T9-W1。
3. **嚴格超集 CKA**:Procrustes²=feature arm(App A.1, Eq 23,已驗證);CKA/SVCCA 是其
   旋轉不變近親,已在 PRISM 家族內。額外提供 risk bound + attribution + regularizer。
   殺 pCi8-W3 / G3T9-W3。

外加 global 顯眼區塊「**Claims we narrow**」直接命中 AC 決策準則。

---

## 2. 六個會反噬的風險(全部已在改寫中處理)

1. **CKA「贏 +0.008」會反噬** — CI 蓋 0 還粗體標贏 = overselling,連累全篇可信度。
   → 改「tie + containment + 嚴格超集」。這是代數(App A.1/B),不可反駁。
2. **長度** — 超字數 3–5×。每 thread lead-first(第一句給硬結果+數字);共享項目非主場
   只留 2 句。
3. **七個 [TBD-E2] 全押在還在跑+canary-gated 的 E2** — AC 第二條件無數字的風險。
   → E2 最高優先;備 partial fallback。
4. **校準隱藏成本** — 「先 benchmark 10 個 variant」→ 補 amortization 句(one-time
   per family、跨 bit-width 遷移、未校準排序已有用;never zero labels ever,只是 zero
   labels per new variant)。
5. **McDiarmid bounded-difference 軟肋** — CE loss 原則上無界。→ 補「hidden states
   RMSNorm-bounded ⇒ D_Z 有限 ⇒ b 有限」(App A.3/A.5 機制)。正中 8VrD 頭號 weakness。
6. **precision base-rate 陷阱** — 主打 MAE(base-rate-free),precision 當輔證。

---

## 3. 逐 reviewer 逐項提案(格式:原話關切 → 判斷 → 提案)

### pCi8(conf 4,sig=2,已進評判模式——最需攻)

- **W1 貢獻=整合非發現**:用他自己讚美當武器(「simplex polarization is a real
  contribution」「exact scale/shape split」)。thermometer 比喻保留但先鋪 humility;
  落點=commensurable 是能做排序/歸因/介入的前提。
- **W2 不能說 Q4 夠不夠好**（★整篇最該當開場）:第一句給 punchline(兩個 cell 校準
  結果),slack 統計第二段,補 amortization。
- **W3 Table 3 機械小收益**:先承接他的具體數字(W=I pooled Ω 0.804→B 0.820,+0.016),
  解釋 pooling 異質 γ 稀釋,再 containment;**移除** B_N vs CKA +0.008 賣點。
- **W4 GSM8K 結構限制**:主動升格 first-class limitation 姿態前置;E9 fallback 已備。
- **W5 EWC 沒比/replay 圖方便**:先認+給表,設計論點(weight/data/feature 三家)放第二句。
- **W6 激進量化後 isometry**:已 lead「validity≠tightness」(好);E8 段先給結論再給方法。

### G3T9(conf 4,sig=2)

- **W1 何不跑 benchmark**:三子問逐一對上他的清單(forward passes / reference data /
  cost)。actionability 用**閉環**反駁「another layer of descriptive analysis」。
- **W2 teacher-forced/refset**:free-running 只跑 GSM8K→講成「最難情況」優勢。
- **W3 CKA 直接比 + simpler feature-preserving reg**:containment 重構;點名 feature
  distillation 就是他要的 baseline。
- **W4 bound 鬆=啟發式**:與 pCi8-W2 同素材濃縮。
- **Q1 full FT/RLHF**:lead with head share 0%→50% 命中預測。
- **Q2 mixed/SFT**:E10 已有,誠實劃界。

### eQL6(conf 3,Q=2/Cl=2——ROI 最高、最可能翻盤;語氣要溫暖)

- **W1 notation/appendix**:認錯+具體 promote 清單(CKA 表/negative-cell matrix/Table 22)。
- **W2+Q1 orthogonal 動機**:gauge 論證放第一,用最直白的話(線性 head 吸收可逆 A)。
- **W3 prefactor 主導**（★當場釘死）:用他自己的代數框架——「你的代數對,而且我們證明
  那情形不會發生」:CV 0.63% vs 366× 動態範圍、Spearman 1.000、相同 rank corr 0.770。
- **W4+Q3 對原任務影響**:別說他沒看懂;說「第一欄正是,但我們讓它太容易被忽略」+E2 表。
- **Q2/Q4**:純改寫/澄清,已好。

### 8VrD(conf 3,rating=4——盟友,要餵飽+給辯護彈藥)

- **W1 empirical vs population**:加 RMSNorm-bounded 句,corollary 滴水不漏。
- **W2/Q1 鬆 + 他的兩個 cell**:用他自己 cell 做校準表(心理學最佳),保持;主打 MAE。
- **W3 無 controlled intervention/Table 22 退步/四篇 citation**:E4+E2+gating;Table 22
  退步 cell 主動承認+gating 解釋;SLoRA/two-phase/CLAIM/ArMA 各給一句定位(sequential
  multi-task CIT vs 我們 single-adapter shape drift),不全 defer。
- **W4/Q2 0.831 Llama-only/free-running**:E13 20-cell + E7;「不脫離 context 呈現 0.831」前置。

---

## 4. 反思檢查清單(改完後逐條驗)

- [ ] 每個 reviewer 每條 W/Q 都有對應、且 lead-first
- [ ] significance 相關 thread 都 ladder up 到三支柱之一
- [ ] CKA 全篇口徑統一為 containment+tie(無 +0.008 賣點殘留)
- [ ] 六個風險句全部入文
- [ ] 所有真實數字未被更動(絕不虛構);[TBD-E#] 保留
- [ ] narrowing 區塊顯眼
- [ ] eQL6 語氣溫暖、不指責
- [ ] 8VrD 四篇 citation 各有一句
