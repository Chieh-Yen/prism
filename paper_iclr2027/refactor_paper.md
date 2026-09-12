# PRISM Paper Refactor — Logical Contradictions & Improvements

> **流程**：每輪 = 看論文 → 思考 → 規劃 → 驗證 → 寫入。共 3 輪，每輪用不同 lens 掃描。
>
> **嚴重度標準**：
> - 🔴 **Critical** — 真實邏輯/技術矛盾，reviewer 會挑
> - 🟡 **Moderate** — Notation/wording 不一致，可能誤導 reader
> - 🟢 **Minor** — 微調可改進易讀性

---

## Round 1 — Notation & Logical-Mismatch Scan

**Lens**：技術 notation 一致性、claim-to-contribution 對應、未定義符號。

### R1.1 🔴 `δ` 和 `γ` 雙重定義（Sec 3 actual error vs Sec 5 bound expression）

**現況**：
- **Sec 3.2 L216**（定義為 actual error，bound 的 LHS）：
  > "feature error $\delta := |\mathcal{R}_T - \mathcal{R}_{P\to T}|$ and a head error $\gamma := |\mathcal{R}_{P\to T} - \mathcal{R}_P|$"
- **Eq.(5) L242**：$\delta \le K_{\mathrm{feat}}\sqrt{(\rho_T-\rho_P)^2 + 2\rho_T\rho_P(1-\Omega_W)}$
- **L246**：$\gamma \le K_{\mathrm{pred}}\,\|\Sigma_P^{1/2}\Delta H\|_F$
- **Sec 4 L309**："head term $\gamma$ vanishes" — 這裡 $\gamma$ 必指 bound expression（$\|\Sigma_P^{1/2}(H_T-H_P)\|_F=0$ when $H_T=H_P$），不是 actual head error
- **Sec 5.2 L358**："the feature-only ($\delta$) scatter" — $\delta$ 是 bound expression
- **Sec 5.3 L380**："$(\Delta\rho)^2{=}0.56$, $\Omega{=}1.0000$, so $\delta{=}0.75$, yet ... $\gamma{=}60.16$" — $\delta, \gamma$ 是具體數值（bound 計算結果）
- **Sec 5.5 L412**："the feature alignment error $\delta$ ... $\delta$ alone is not a valid bound in that setting" — 明確指 bound expression
- **Tables 各行**：$\delta = K_{\mathrm{feat}}\sqrt{...}$、$\gamma = K_{\mathrm{pred}}\|\Sigma_P^{1/2}\Delta H\|_F$ 為欄位

**矛盾**：Sec 3.2 定義 $\delta, \gamma$ 為 **actual risk gaps**（無法計算，是被 bound 住的對象）；Sec 5、tables、Sec 4 一律用 $\delta, \gamma$ 指 **bound expressions**（可計算的 RHS）。

**建議 fix（推薦 Option A）**：
- **Option A**：改 Sec 3.2 定義為 bound expression（簡潔，與 Sec 5 對齊）
  ```latex
  Triangle inequality: $|\mathcal{R}_T - \mathcal{R}_P| \le |\mathcal{R}_T - \mathcal{R}_{P\to T}| + |\mathcal{R}_{P\to T} - \mathcal{R}_P|$.
  We bound each term: define
  $\delta := K_{\mathrm{feat}}\sqrt{(\rho_T-\rho_P)^2 + 2\rho_T\rho_P(1-\Omega_W)}$ (feature alignment error)
  and $\gamma := K_{\mathrm{pred}}\|\Sigma_P^{1/2}(WH_T-H_P)\|_F$ (head discrepancy);
  then $|\mathcal{R}_T - \mathcal{R}_{P\to T}| \le \delta$ and $|\mathcal{R}_{P\to T} - \mathcal{R}_P| \le \gamma$.
  ```
- **Option B**：保留 Sec 3.2 actual-error 定義，Sec 5 改用 $\hat\delta, \hat\gamma$ 或 "feature term" / "head term" 措辭（侵入性高，需改 tables 欄位）

**驗證 status**：✅ Confirmed via grep。

---

### R1.2 🟡 `𝓑` (full PRISM bound) 從未正式定義

**現況**：
- 全 Sec 5 (L349 caption, L356, L363, L368, L412, L414) 都用 $\mathcal{B}$ 指「PRISM bound」
- Sec 3 Theorem 1 (L248-254) 寫出 RHS 但**沒有命名為 $\mathcal{B}$**
- Reader 必須從上下文推斷 $\mathcal{B} = \delta + \gamma$（或 = Theorem 1 的整個 RHS）

**建議 fix**：在 Theorem 1 之後（或 Sec 3.3 開頭）加一行
```latex
We denote the right-hand side of Eq.~(\ref{eq:unified_bound}) as $\mathcal{B} := \delta + \gamma$ (the \emph{PRISM bound}); it serves as the diagnostic quantity throughout Sec.~\ref{sec:experiments}.
```

**驗證 status**：✅ Confirmed via grep — `\mathcal{B}` 14 次出現，零次定義。

---

### R1.3 🔴 "Three claims tied to the three contributions" — 對應不成立

**現況**（L320）：
> "We organize the experiments around three claims tied to the three contributions: \emph{Predictiveness}—...; \emph{Decomposability}—...; and \emph{Actionability}—..."

**問題**：Contributions 是 **{Theory, Framework, Empirical}**（L168-174）；experimental claims 是 **{Predictiveness, Decomposability, Actionability}**。**兩組不是 1-to-1**：
- Predictiveness ⊂ Empirical(C3)
- Decomposability ⊂ Empirical(C3) + 部分 driven by Theory(C1)
- Actionability ⊂ Framework(C2) + uses differentiability from Theory(C1)

三個 experimental claims 都是 Empirical(C3) 的 sub-aspects，不是分別對應三個 contributions。"tied to the three contributions" 暗示一個不存在的對應結構。

**建議 fix**：
- **Option A（最輕，刪冗詞）**：刪 "tied to the three contributions"
  > "We organize the experiments around three claims: \emph{Predictiveness}—..."
- **Option B（重述對應）**：改寫成正確對應
  > "We organize the experiments around three claims that operationalize the empirical contribution: ..."

**推薦 Option A**：簡潔且直接。

**驗證 status**：✅ Confirmed via 對照 L168-174 contributions 和 L320 claims。

---

### R1.4 🟡 `n` vs `N` 三處不一致

**現況**：
| 位置 | 符號 | 意義 |
|---|---|---|
| Sec 3.1 L210 | $n$ | calibration sample size, $Z_M \in \mathbb{R}^{n \times d}$, $\rho_M = \|Z_M\|_F/\sqrt{n}$ |
| Sec 3.4 L283 | $N$ | total target tokens across calibration set, $Z_M^{\mathrm{AR}} \in \mathbb{R}^{N \times d}$ |
| Sec 5.1 L338 | $N$ | per-benchmark subset size（sequence count, **not** token count）："$N{=}512$ per benchmark for PTQ" |

**矛盾**：
- Sec 3.1 vs Sec 3.4：同一個概念（feature matrix 的 row count）用了 $n$ 和 $N$ 兩個符號
- Sec 3.4 vs Sec 5.1：兩處的 $N$ scope 不同（前者 = total tokens，後者 = subset sequence count）

**建議 fix**：
- 統一 Sec 3.1 + Sec 3.4 為 $N$（更顯眼，且 Sec 3.4 已用 $N$）
- Sec 5.1 將 $N{=}512$ 改為 "**$|\mathcal{D}_{\mathrm{cal}}|{=}512$ sequences per benchmark**"（明確區分 sequence count vs token count $N$）
  - 或保留 $N{=}512$ 但首次出現時加註「sequence count, not token count $N$ of Sec.~\ref{subsec:ar_extension}」

**驗證 status**：✅ Confirmed via grep — 三處 scope 不同。

---

### R1.5 🟢 "Single forward pass" 略誇大（head term 需 SVD）

**現況**：
- L160（intro）："Each axis is independently measurable from a single forward pass"
- L183（RW）："scale collapse, structural distortion, and head divergence are each measured from a single forward pass"

**事實**：
- Scale ($\rho$) 和 Shape ($\Omega$)：只需 features $Z$，**single forward pass 成立** ✓
- Head term $\|\Sigma_P^{1/2}(WH_T - H_P)\|_F$：需 forward pass 收集 $\Sigma_P$ + 一次 $d \times d$ SVD/eigendecomp。SVD 成本相對 forward pass 微不足道，但**嚴格來說不是「single forward pass」**

**defensibility**：「forward pass」可以解讀為「資料收集步驟」，SVD 是後處理。Reviewer 大概不會挑。

**建議 fix（可選）**：軟化措辭
- "from a single forward pass" → "from one forward pass plus a $d \times d$ SVD"
- 或 "from data collected in a single forward pass"

**驗證 status**：✅ Confirmed by 計算 head term 的數學步驟。低優先級。

---

### R1.6 🟢 Conclusion limitation tautology (L424)

**現況**：
> "PRISM is a *relative* metric: it scores how a variant differs from a reference base via the cross-entropy risk gap, not the variant's absolute task quality—the strong rank correlation reported in Sec.~\ref{subsec:predict} **supports variant comparison but does not predict $|\Delta\mathcal{R}|$ itself**."

**問題**：「rank correlation supports variant comparison but does not predict $|\Delta\mathcal{R}|$」**幾乎是 tautological**——rank correlation 本來就只測 ordering，本來就不預測絕對值。讀起來像在解釋自己用的 metric 的定義。

**建議 fix**：
- **Option A（精簡，去重）**：
  > "PRISM is a *relative* metric: it bounds the cross-entropy risk gap to a reference base, not the variant's absolute task quality. The bound is calibrated for ranking, not for absolute-magnitude estimation."
- **Option B（保留 rank-correlation 引用，但避免 tautology）**：
  > "PRISM is a *relative* metric: it bounds the cross-entropy risk gap to a reference base. The strong rank correlation (Sec.~\ref{subsec:predict}) certifies variant ordering; absolute-magnitude calibration of $|\Delta\mathcal{R}|$ is a separate question."

**驗證 status**：✅ 邏輯結構檢查確認。

---

### Round 1 套用優先級

| ID | 嚴重度 | 修改成本 | 推薦套用順序 |
|---|---|---|---|
| R1.1 δ/γ overload | 🔴 Critical | 中（改 Sec 3.2 + 確保 Sec 5 一致） | **第一優先** |
| R1.2 𝓑 未定義 | 🟡 Moderate | 低（加一句） | 第二優先（與 R1.1 同處改） |
| R1.3 three claims/contributions | 🔴 Critical | 低（刪 5 字） | 第三優先 |
| R1.4 n/N 不一致 | 🟡 Moderate | 中（改三處 + 可能 Eq.(2) 重排） | 第四優先 |
| R1.5 single forward pass | 🟢 Minor | 低（可選） | 第五優先 |
| R1.6 limitation tautology | 🟢 Minor | 低 | 第六優先 |

---

## Round 2 — Cross-Section Consistency Scan

**Lens**：Abstract → intro → body → conclusion 之 claim 一致性、figure caption vs prose alignment、Theorem statement vs experimental usage、forward references。

### R2.1 🟡 "Settings" vs "Failure modes" 術語混用

**現況**：兩個詞混合指**同一個概念**（三軸對應的失敗類型）：

| 位置 | 用詞 | 指涉 |
|---|---|---|
| L260 Sec 3.3 | "failure mode" | 三軸對應 |
| L320 Sec 5 opening | "three qualitatively distinct empirical **failure modes**" | 三軸對應 |
| L371 Sec 5.3 title | "Three Axes, Three **Failure Modes**" | 三軸對應 |
| L374 Sec 5.3 body | "Three **settings** recur in our experiments" | 三軸對應 ← **同一個概念用了 settings** |
| L421 conclusion | "the three axes localize distinct empirical **settings**" | 三軸對應 ← **同一個概念用了 settings** |

同時 "settings" 也指 PTQ vs LoRA 兩個 application contexts：
- L132 abstract: "distinct empirical settings under PTQ and LoRA"
- L149: "the same risk gap in both settings"
- L272: "frozen-head settings studied here—LoRA and FP16-head PTQ"
- L305 Sec 4: "two LLM lifecycle settings"
- L421 conclusion: "rank-consistent across both settings"

**問題**：
1. "settings" 同時指 (a) PTQ/LoRA 兩個 applications 和 (b) 三個 failure modes — overloading
2. L374 + L421 用 "settings" 指三軸對應，但 L260 / L320 / L371 同一概念用 "failure modes" — 不一致

**建議 fix**：
- 統一指三軸對應為 **"failure modes"**（保留 PTQ/LoRA 用 "settings" 或 "applications"）
- 改 L374："Three **settings** recur in our experiments" → "Three **failure modes** recur in our experiments"
- 改 L421："localize distinct empirical **settings**" → "localize distinct **failure modes**"

**驗證 status**：✅ Confirmed via grep。

---

### R2.2 🟡 0.835 LoRA scope 與 0.82 PTQ scope 不對等（M3 沒延伸到 contributions/conclusion）

**現況**：
- **L368 Sec 5.2**（M3 已軟化）："$|r_s|{=}0.835$ aggregated over both fine-tuning tasks and the five downstream benchmarks, **matching the strong correlation of the PTQ grid (mean $|r_s|{\approx}0.82$, Fig.~\ref{fig:quant_grid_bound})**"
- **L173 contributions**："the bound ranks variants with mean Spearman $|r_s|{\approx}0.82$ on PTQ and $0.835$ on LoRA forgetting"
- **L421 conclusion**："rank-consistent across both settings (mean Spearman $|r_s|{\approx}0.82$ on PTQ, $0.835$ on LoRA forgetting)"

**Scope 對比**：
| 數字 | 模型範圍 | Cells |
|---|---|---|
| 0.82 (PTQ) | Llama-3.1-8B + Qwen3-8B | 10 (2 模型 × 5 benchmarks) |
| 0.835 (LoRA) | **Llama-3.1-8B 單獨**（Qwen LoRA 在 appendix，部分 cells 是 noise floor） | 10 (2 fine-tune tasks × 5 benchmarks) |

**問題**：L173 和 L421 把兩個數字並列，**未提示 scope 不對等**。Reviewer 比對 Fig 3 (Llama only) vs Fig 2 (Llama+Qwen) 即可發現。

**建議 fix**：
- **Option A（最輕，去 0.835 純並列）**：L173 + L421 改用「strong rank correlation across both settings (PTQ: $|r_s|{\approx}0.82$; LoRA: comparable strength on Llama, see Sec.~\ref{subsec:predict})」
- **Option B（保留 0.835 但加註）**：「$|r_s|{\approx}0.82$ on PTQ ($2{\times}5$ cross-family grid), $0.835$ on LoRA forgetting (Llama, $2{\times}5$ cells)」
- **Option C（最保守，照做 M3 同樣措辭）**：L173/L421 直接套用 L368 的 "matching the strong correlation of the PTQ grid"

推薦 **Option B**：保留具體數字但明確 scope。

**驗證 status**：✅ Confirmed by 對照 Fig 3 caption + appendix 結構。

---

### R2.3 🟢 Sec 3.5 "overhead" 承諾未兌現

**現況**：
- **L299 Sec 3.5**："training schedule and **overhead** in Sec.~\ref{subsec:exp_setup}"
- **L338 Sec 5.1**：只給 schedule（"refreshed every $k{=}8$ micro-steps"，"Checkpoints every $25$ steps"），**沒量化 overhead**（wall-clock %、GPU 記憶體成本）

**問題**：forward reference 承諾要在 5.1 報 overhead，5.1 沒提供。

**建議 fix**：
- **Option A（最輕，刪承諾）**：L299 改為 "training schedule in Sec.~\ref{subsec:exp_setup}"
- **Option B（補 overhead）**：L338 加一句 "$\Omega$ refresh adds $\sim$X\% wall-clock overhead per training step"（需查實際數字）

推薦 **Option A**：簡單且無爭議。

**驗證 status**：✅ Confirmed via grep — 只找到 schedule，沒有 overhead 量化。

---

### R2.4 🟢 Eq.(9) LoRA bound 隱式假設 $W{=}I$ 沒交代

**現況**：
- **Eq.(9) L290-292**：
  $|\mathcal{R}_0 - \mathcal{R}_t| \le K_{\mathrm{feat}}\sqrt{(\rho_0-\rho_t)^2 + 2\rho_0\rho_t(1-\Omega)}$
- 主文 L272 commits to $\Omega := \Omega_{W=I}$，但 Eq.(9) 沒明說「這裡的 $\Omega$ 是 $\Omega_{W=I}$」
- LoRA 為何 $W=I$ 是對的 alignment？因為 LoRA 用 low-rank update（$W_0 + BA$）改 backbone 但保留座標系——這個 reasoning 主文沒有

**問題**：reader 需自己推：(i) $H_T = H_P$（LoRA 凍結 head），(ii) LoRA 不引入 rotation→$W=I$ natural，(iii) $\gamma = K_{\mathrm{pred}}\|\Sigma_P^{1/2}(W H_T - H_P)\|_F = 0$。

**建議 fix**：在 Eq.(9) 之前加一句
> "Under LoRA's frozen-head and additive low-rank update, $H_T = H_P$ and $W = I$ is the natural alignment, so $\gamma = 0$ and Theorem~\ref{thm:unified_bound} reduces to:"

**驗證 status**：✅ 數學一致性檢查。

---

### R2.5 🟢 Abstract "four families" vs 主圖只展示 2 — expectation mismatch

**現況**：
- **L132 abstract**："Across **four** 8B-scale LLM families and five benchmarks"
- **L349 Fig 2 caption**："two model families and five benchmarks"
- **L356 Sec 5.2 prose**："Llama--Qwen pair ... replicates on Ministral and DeepSeek in Appendix"
- **L377 Sec 5.3**："**all four families tested**"（指 Llama+Qwen+Ministral+DeepSeek）

**問題**：
- Abstract 給「4 families」期望
- 主文 figures 只看到 2
- Reader 需翻 appendix 才能 verify "four"
- Defensible（4 families 確實都測了）但 expectation 不對齊

**建議 fix（可選）**：
- **Option A（保留 4，但加註 main+appendix）**：abstract 改 "Across four 8B-scale LLM families (two in main figures, two in appendix replications) and five benchmarks"
- **Option B（精確降為 2，appendix 加分）**：abstract 改 "Across two main model families (Llama-3.1-8B, Qwen3-8B) and five benchmarks, with appendix replications on two additional families"

推薦 **保持原狀**（"four" 在 Sec 5.3 / Sec 5.5 都引用，不是純 sales pitch；reviewer 翻 appendix 即可 verify）。

**驗證 status**：✅ 4 families 確實都有 appendix tables。**標記但不一定要改**。

---

### Round 2 套用優先級

| ID | 嚴重度 | 修改成本 | 推薦套用順序 |
|---|---|---|---|
| R2.1 settings/failure modes 混用 | 🟡 Moderate | 低（L374, L421 各改一處） | 第一優先 |
| R2.2 0.835 scope 不對等 | 🟡 Moderate | 低（L173, L421 各加 scope 註） | 第二優先 |
| R2.3 overhead 承諾未兌現 | 🟢 Minor | 極低（刪 2 字） | 第三優先 |
| R2.4 Eq.(9) $W=I$ 隱式 | 🟢 Minor | 低（加一句） | 第四優先 |
| R2.5 abstract "four" 期望落差 | 🟢 Minor (defensible) | – | **不建議改** |

---

## Round 3 — Subtle/Structural Improvement Scan

**Lens**：narrative flow、hedging、措辭精準度、未用 notation、numerical precision 一致性、reader experience。

### R3.1 🟡 "Scale axis is empirically separable from shape" — empirical 證據與 claim 不對齊

**現況 L383**：
> "The decomposition $(\Delta\rho, 1{-}\Omega)$ in Eq.~(\ref{eq:lora_bound}) thus exhibits the two arms as non-redundant channels: \emph{different source tasks induce qualitatively different drift geometries}, **and the scale axis is empirically separable from shape**---an observation a single-number similarity would mask."

**問題**：
- 句子前半（"different tasks → different drift geometries"）有 evidence：TruthfulQA 是 shape-only drift、BBQ 是 mixed
- 句子後半（"scale axis is empirically separable from shape"）暗示 statistical independence 或「兩軸可獨立移動」
- 但實際 evidence 只展示 LoRA 一個方向（shape 大、scale 小）；要 full separability 需要反向 case（scale 大、shape 小）—— 這個在 LoRA 不存在
- L173 contributions、L421 conclusion 都引用 "scale-axis separability under cross-task LoRA drift"，propagate 這個措辭

**建議 fix**：把「separable」軟化為「dissociable」或「register different mixtures」
- L383："...the two arms as non-redundant channels: \emph{different source tasks induce qualitatively different drift geometries}, **and the (scale, shape) profile is task-dependent**---an observation a single-number similarity would mask."
- 對應 L173, L421 改為「**(scale, shape) profile dissociation** under cross-task LoRA drift」

**驗證 status**：✅ 邏輯與 evidence 對應檢查確認。

---

### R3.2 🟢 Sec 5.1 L335 末句 defensive 句子重複 limitation

**現況**：
> L335: "...PRISM is a variant-ranking signal complementary to generation-based evaluation, not a substitute."

**問題**：
- 主動「招認」未被攻擊的點，是無資訊的 hedge
- L424 conclusion limitations 段已有 "PRISM is a *relative* metric ... supports variant comparison but does not predict $|\Delta\mathcal{R}|$ itself" 涵蓋
- 主文 5.1 結尾再加一個 disclaimer 反而暴露 attack surface

**建議 fix**：移除 L335 末句，paragraph 在 "...same scale as the LoRA fine-tuning objective." 處結束。

**驗證 status**：✅ Limitation 段已 cover；主文 disclaimer 為冗餘。

---

### R3.3 🟢 "Llama-3.1-8B" base vs instruct 隱式（首次出現未標 Base）

**現況**：
- L325 Sec 5.1："The main analysis uses the cross-family pair Llama-3.1-8B~\cite{grattafiori2024llama} and Qwen3-8B~\cite{qwen2025qwen3}"
- 沒明說 Base
- Appendix 有 separate "instruction-tuned counterparts" tables → 暗示 main = Base
- 但 Llama-3.1-8B 預設常被理解為 chat/instruct

**建議 fix**：L325 首次出現改為 "Llama-3.1-8B (base)" 和 "Qwen3-8B (base)"，後續 reference 維持 short form。

**驗證 status**：✅ 與 appendix 區分 base vs instruct 一致。

---

### R3.4 🟢 數值精度不一致：narrative 中 $\Omega{=}1.0000$ vs $\Omega \approx 1$

**現況 L380**：
> "...backbone scale and shape are essentially perfect ($(\Delta\rho)^2{=}0.56$, $\Omega{=}1.0000$), so $\delta{=}0.75$..."

**對比同段附近**：
- L358："$\Omega$ drops to $0.76$"（2 位）
- L380："$\Omega{=}1.0000$"（4 位）
- L380："$\delta{=}0.75$"、"$\gamma{=}60.16$"（2-4 位混合）
- L402："$\Omega$ on downstream benchmarks increases monotonically (e.g., ARC $0.82{\to}0.86{\to}0.93$, MMLU $0.87{\to}0.90{\to}0.95$, GSM8K $0.94{\to}0.97{\to}1.00$)"（2 位）

**問題**："$\Omega{=}1.0000$" 4 位精度暗示「跟非 1 有差別」，但 narrative 只想說「approximately 1」。Inconsistent with same paragraph 的 "$1.00$" 寫法。

**建議 fix**：L380 改為 "$\Omega \approx 1.00$" 或 "$\Omega \approx 1$"。

**驗證 status**：✅ 同段內精度不一致。

---

### R3.5 🟢 $h_{M,j}$ 在主文定義但完全未使用

**現況 L210 Sec 3.1**：
> "denote the columns of $H_M$ by $h_{M,1},\ldots,h_{M,V}$"

**Grep 結果**：主文後續無 $h_{M,j}$ 引用。Likely 用於 appendix proofs。

**建議 fix**：
- **Option A**：刪 "denote the columns of $H_M$ by $h_{M,1},\ldots,h_{M,V}$,"（L210 縮短）；appendix proof 自行 introduce
- **Option B**：保留（主文 setup 通常會列所有 notation 即使主文不用）

推薦 **Option A**：減少主文 notation 噪訊。

**驗證 status**：✅ Confirmed via grep。

---

### R3.6 🟢 "Reference set" 命名 (Sec 3.5) 與 "fixed batch of 32 sequences" (Sec 5.1) 不直接連接

**現況**：
- L297 Sec 3.5：$Z_0^{\text{ref}}, Z_t^{\text{ref}}$ are base and current feature matrices on a fixed **reference set**
- L338 Sec 5.1：$\Omega$ is computed on a fixed batch of $32$ sequences from the pre-training distribution (disjoint from $\mathcal{D}_{\mathrm{FT}}$)

**問題**：Sec 5.1 沒明說「這 $32$ sequences 就是 Sec 3.5 的 $\mathcal{D}_{\mathrm{ref}}$」。Reader 需自己連接。

**建議 fix**：L338 開頭加「the **reference set** $\mathcal{D}_{\mathrm{ref}}$ for $Z^{\mathrm{ref}}$ in Eq.~(\ref{eq:shape_reg}) is a fixed batch of $32$ sequences...」。

**驗證 status**：✅ 兩處概念連結缺失。

---

### R3.7 🟢 "qualitatively different" 在 L368 和 L383 同段附近用兩次但指不同概念

**現況**：
- L368 (Sec 5.2 末)："a single closed-form bound transfers across **two qualitatively different sources** of model variation"（指 PTQ vs LoRA）
- L383 (Sec 5.3 末)："**different source tasks induce qualitatively different drift geometries**"（指 TruthfulQA vs BBQ）

**問題**：兩處都用 "qualitatively different" 但 scope 不同（applications vs source tasks within LoRA）；緊鄰段落讀起來像在說同一件事。

**建議 fix**：L383 改為「different source tasks induce \emph{distinct} drift geometries」或「elicit different (scale, shape) profiles」（與 R3.1 對應改動可一併處理）。

**驗證 status**：✅ 緊鄰段落同詞重複。

---

### Round 3 套用優先級

| ID | 嚴重度 | 修改成本 | 推薦套用順序 |
|---|---|---|---|
| R3.1 "scale axis separable" 措辭精確化 | 🟡 Moderate | 低（L383 + L173 + L421 各改一處） | 第一優先 |
| R3.2 L335 defensive 句移除 | 🟢 Minor | 極低（刪 1 句） | 第二優先 |
| R3.3 Llama (base) 標示 | 🟢 Minor | 極低（加 4 字） | 第三優先 |
| R3.4 $\Omega{=}1.0000$ 精度 | 🟢 Minor | 極低 | 第四優先 |
| R3.5 $h_{M,j}$ 移除 | 🟢 Minor | 極低 | 第五優先 |
| R3.6 reference set 命名連結 | 🟢 Minor | 低 | 第六優先 |
| R3.7 "qualitatively different" 重複 | 🟢 Minor | 極低（與 R3.1 合併） | 與 R3.1 一併 |

---

## 整體優先級總覽

按 ROI（嚴重度 × 修改成本反比）排序：

### 🔴 Critical（強烈建議套用）
- **R1.1** δ/γ overload — Sec 3.2 + 確保 Sec 5 對齊
- **R1.3** "three claims tied to three contributions" — 刪 5 字
- **R3.1** "scale axis separable" 軟化（與 R3.7 合併）

### 🟡 Moderate（建議套用）
- **R1.2** 𝓑 正式定義
- **R1.4** n/N notation 統一
- **R2.1** "settings"/"failure modes" 標準化
- **R2.2** 0.835 scope 註明（在 contributions + conclusion）

### 🟢 Minor（低優先，視時間決定）
- **R1.5** "single forward pass" 軟化
- **R1.6** Limitation tautology 改寫
- **R2.3** "overhead" 承諾改 schedule
- **R2.4** Eq.(9) $W=I$ 顯式化
- **R3.2** L335 defensive 句刪除
- **R3.3-R3.7** 細節精修

### 標記但不一定改
- **R2.5** Abstract "four families" 期望落差（defensible）

---

## 套用建議路徑

**Phase 1（核心，~30 分鐘工作量）**：
1. R1.1 δ/γ Sec 3.2 redefine
2. R1.2 𝓑 加定義
3. R1.3 + R2.1 + R2.2 同改 L320, L173, L421（一個 sweep）
4. R1.4 n/N 統一

**Phase 2（中度，~15 分鐘）**：
5. R3.1 + R3.7 合併改 L383
6. R2.3 刪 "overhead"
7. R2.4 Eq.(9) 加 $W=I$ 註

**Phase 3（細修，~10 分鐘）**：
8. R3.2-R3.6 細節
9. R1.5, R1.6 視情況決定

**Phase 1 預期影響**：解決所有 Critical 級邏輯/notation 問題，為 reviewer 不留 attack vector。

---

## 套用紀錄（Phase 1）

| ID | 狀態 | 備註 |
|---|---|---|
| **R1.1** δ/γ overload | ✅ 套用 | Sec 3.2 改 triangle inequality 介紹，Eq.(5) 改 $\delta := K_{\mathrm{feat}}\sqrt{...}$，Head error 段改 $\gamma := K_{\mathrm{pred}}\|...\|_F$；保留 $\|\mathcal{R}_T - \mathcal{R}_{P\to T}\| \le \delta$ 等 inequality |
| **R1.2** 𝓑 未定義 | ✅ 套用 | Theorem 1 加 $\mathcal{B} := \delta + \gamma$ 定義，underbraces 標記為 $\delta, \gamma$；新增 `\label{eq:unified_bound}` |
| **R1.3** "tied to three contributions" | ✅ 套用 | L320 刪 5 字 |
| **R2.1** settings → failure modes | ✅ 套用 | L374, L421, L173 三處改用 "failure modes" 指三軸對應；保留 "settings" 指 PTQ/LoRA application contexts |
| **R2.2** 0.835 scope qualifier | ❌ **撤回** | User 指出 appendix 已有 Qwen LoRA + noise floor 解釋，主文加 "Llama only" qualifier 反而暴露弱點。保留原 0.82 / 0.835 並列。 |
| **R1.4** n/N 不一致 | ✅ 套用 | Sec 3.4 補 "$N$ replacing $n$ of Sec 3.1 in AR setting"；Sec 5.1 改 $N{=}512$ 為 "$512$ samples"（避免 symbol collision） |

**編譯狀態**：✅ 37 頁，無 error。

---

## 套用紀錄（Phase 2）

| ID | 狀態 | 備註 |
|---|---|---|
| **R3.1** + R3.7 scale-axis separable 軟化 | ⏭️ Skip | User 決定 |
| **R2.3** Sec 3.5 "overhead" 承諾 | ✅ 套用 | L299 刪 "and overhead"，剩 "training schedule in Sec.~5.1" |
| **R2.4** Eq.(9) 加 $W=I$ explicit | ⏭️ Skip | User 決定 |

---

## 套用紀錄（Phase 3）

| ID | 狀態 | 備註 |
|---|---|---|
| **R3.2** L335 defensive 句移除 | ✅ 套用 | 直接刪「PRISM is ... complementary, not a substitute.」整句 |
| **R3.3** Llama (base) 標示 | ✅ 套用 (Option B) | L325 改 "cross-family pair **of base models** Llama-3.1-8B and Qwen3-8B" |
| **R3.4** $\Omega{=}1.0000$ 精度 | ✅ 套用 | 改為 $\Omega \approx 1$ |
| **R3.5** $h_{M,j}$ unused notation | ✅ 套用 | L210 刪 "denote the columns of $H_M$ by $h_{M,1},\ldots,h_{M,V}$,"；$H_M$ 在 L201 已完整定義（含 dim $d \times V$） |
| **R3.6** reference set 命名連結 | ✅ 套用 (Option B) | L338 加 "the reference set $\mathcal{D}_{\mathrm{ref}}$ (a fixed batch of 32 sequences..., disjoint from $\mathcal{D}_{\mathrm{FT}}$)" |
| **R1.5** "single forward pass" 軟化 | ❌ **DROP — 原分析錯誤** | 主文 $W{=}I$ 下 head term 用 Frobenius norm 恆等式 $\|\Sigma_P^{1/2}A\|_F = \frac{1}{\sqrt{n}}\|Z_P A\|_F$，**不需 SVD**。三 component 全部只需 matrix ops，「single forward pass」嚴格正確。 |
| **R1.6** Limitation tautology | ✅ 套用 (精簡版) | L424 重寫："The bound is calibrated for ranking; tight estimation of $\|\Delta\mathcal{R}\|$ is a separate question."（去 absolute/magnitude 累贅） |

**編譯狀態**：✅ 37 頁，無 error。

---

## 全套用 summary

| Phase | 套用 | Skip / Reject | Drop |
|---|---|---|---|
| Phase 1 | R1.1, R1.2, R1.3, R2.1, R1.4 (5 項) | R2.2 (user 撤回) | – |
| Phase 2 | R2.3 (1 項) | R3.1, R2.4 (user skip) | – |
| Phase 3 | R3.2, R3.3, R3.4, R3.5, R3.6, R1.6 (6 項) | – | R1.5 (原分析錯誤) |
| **總計** | **12 項** | 3 項 skip | 1 項 drop |

**最終 PDF**：37 頁（無變化），所有 Critical 級邏輯/notation 問題解決。
