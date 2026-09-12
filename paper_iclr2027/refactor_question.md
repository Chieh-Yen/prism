# PRISM Paper — Reviewer Attack Vectors

> **目的**：識別所有 reviewer 可能 question/attack 的點，並提出 defense 或論文修改方案。
>
> **流程**：
> - **Round 1**：Brainstorm 所有可能的 reviewer questions（不過濾）
> - **Round 2**：Re-read 論文 verify 每個 question 的 validity（real / speculative / already addressed）
> - **Round 3**：對 verified concerns 提出 defense 或修改方案

---

## Round 1 — Brainstorm（unfiltered）

### A. 理論面 (Theoretical)

#### A1. $K_{\mathrm{feat}}$ 的數值未在主文出現 — bound 實際多鬆？
- 主文 L218 只說「derived via simplex polarization argument in Appendix」
- Appendix L78 給定義：$K_{\mathrm{feat}} = \max_{j,k} \|h_{T,j} - h_{T,k}\|_2$ — token embedding 兩兩距離最大值
- 對 Llama 3.1 8B (vocab=128k, dim=4096)，$K_{\mathrm{feat}}$ 可能很大
- Reviewer：「Bound 實際數值多大？$\mathcal{B}$ 跟 $|\Delta\mathcal{R}|$ 量級差幾倍？看 Fig 2 是否在 safe zone？」

#### A2. $K_{\mathrm{pred}} \le \sqrt{2}$ 的 supremum 是否常態到達？
- Appendix L186 說「approached as $\hat{p}_y \to 0$」(model 預測完全錯)
- 對訓練好的 model 這種 case 罕見 → 實際 $K_{\mathrm{pred}}$ 可能更小
- Reviewer：「實務上的 effective $K_{\mathrm{pred}}$ 是多少？是否 worst-case 過於保守？」

#### A3. Linear Representation Hypothesis (LRH) 是 assumption 不是 proof
- Sec 3.1 L210 直接 assume $\phi_T(x) \approx \phi_P(x) W$ for orthogonal $W$
- 對 quantized variant，每個 weight 獨立擾動 → orthogonal $W$ 是否真存在？
- Reviewer：「LRH 對 PTQ 的 empirical evidence 是？$W=I$ 是否對 BnB NF4 仍成立？」

#### A4. $W=I$ 是 default 但 $W=W_N$ tighter — 為何不用 $W_N$？
- Sec 5.5 ablation 顯示 $\mathcal{B}_N$ Spearman 0.91 vs $\mathcal{B}$ 0.82（差 0.09）
- 主文 commit $W=I$ 理由：(i) frozen-head 下 $\gamma$ 簡化、(ii) autograd 不需 SVD
- Reviewer：「Diagnostic-only 場景（不訓練）為何不用 $\mathcal{B}_N$？autograd 理由只對 regularizer 有意義。」

#### A5. AR risk Eq.(8) 用 per-sequence average，但 Theorem 1 應用於 token-pooled $Z^{AR}$
- Eq.(8) AR 形式：$\frac{1}{|y|}\sum_\tau \ell(...)$
- L285：「Theorem 1 then applies directly to $(Z_T^{AR}, Z_P^{AR})$」
- 若 sequence 長度不一，per-sequence avg 跟 token-pooled avg 不同（Jensen-style gap）
- Reviewer：「How does the bound handle variable sequence length? Token-pooled vs sequence-averaged 不等價。」

#### A6. "Closed-form" claim — bound 仍含 $K_{\mathrm{feat}}, K_{\mathrm{pred}}$ 兩個 problem-dependent constants
- Closed-form 通常指「無 implicit definitions / no infinite series」
- 但 $K_{\mathrm{feat}}$ 需從 $H_T$ 計算（max pairwise distance over $V$ tokens）
- Reviewer：「Closed-form 包含 $V \sim 10^5$ pairwise distance 計算？」

---

### B. 經驗面 (Empirical)

#### B1. 為何只 8B 規模？70B/MoE 表現如何？
- 4 個 family 都是 8B
- Reviewer：「PRISM 在 70B 或 MoE model 是否仍 hold？scaling laws 上有 evidence 嗎？」

#### B2. 為何只 5 benchmarks？generation-based eval (HumanEval, MT-Bench) 不測？
- 5 benchmarks 全部 teacher-forced CE scoring
- 沒測 generation quality
- Reviewer：「PRISM 跟 HumanEval pass@1 / MT-Bench 的 correlation 是？實務上 deployment 看的是 generation。」

#### B3. Calibration set size: 512 (PTQ), 256 (LoRA), 32 (shape reg) — robust？
- 樣本數小，sensitivity analysis 沒做
- Reviewer：「不同 calibration set 抽樣是否會改變 ranking？$N$ 增加到 1024/2048 會更好嗎？」

#### B4. **0.835 vs 0.82 scope 不對等**（user M3 已撤回 disclaimer）
- 0.835 = Llama LoRA only
- 0.82 = Llama+Qwen PTQ
- Reviewer 對到 Fig 3 caption (Llama only) → 可能挑「為何 LoRA 不算 cross-family？」
- 但 appendix 有 Qwen LoRA + noise floor 解釋 cover 這個

#### B5. "Strong rank correlation" — 跟什麼比？沒有 external baseline
- Sec 5.5 ablation 比 PRISM 自己的 components ($\Omega, \delta, \mathcal{B}$)
- 但**沒比** CKA、layer-wise reconstruction loss、weight quantization error 等 baseline
- Reviewer：「PRISM 跟 CKA 的 Spearman 對比？跟 quantization error 的 Spearman 對比？」

#### B6. Q2_K Qwen GSM8K — $\rho_P$ INFLATES (267→313)，與「scale collapse」narrative 衝突
- Sec 3.3 L264：「shrinking $\rho_P < \rho_T$」是 typical scale collapse
- 但 L379 寫 Qwen3-Base Q2_K GSM8K：$\rho_P$ rises from 267 to 313（**+17%**）
- Reviewer：「Scale axis 故事不一致——有時 ρ shrink 有時 inflate。能否解釋為什麼？」

#### B7. Shape regularizer at $\lambda=0.5$ — 對 target-task fit 影響？
- Fig 4 + Table 5 只報 downstream forgetting，沒報 target-task accuracy（TruthfulQA / BBQ 本身的成績）
- Reviewer：「Higher λ 應該降 fine-tune task fit，trade-off 在哪？是否反而沒學到 TruthfulQA？」

#### B8. Qwen LoRA appendix 有 weak/negative correlation cells — 是否真的 "replicate" Llama？
- Appendix `forgetting_qwen.tex` 已有 noise floor 解釋
- 但 reviewer 可能爭：「negative $r_s$ 在 noise floor 是 hand-wave；如果是 noise，那 0.835 對 Llama 也可能是 noise」

#### B9. CKA、layer-wise loss 等 baseline 缺席 — 沒有實際 head-to-head comparison
- L145 intro：「[CKA, etc.] do not predict downstream behavior」
- 但 paper 沒實際展示 CKA 在 same data 的 Spearman
- Reviewer：「請補 CKA 跟 PRISM 在 Fig 2 grid 的 Spearman 對比表」

#### B10. Fig 2 caption 提 "safe zone where the bound provably holds"
- 暗示所有 data points 應在 green zone
- 若有任何 point 越過 $y=x$ 上方（bound < $|\Delta\mathcal{R}|$），就破 bound
- Reviewer：「Fig 2 上有 outlier 違反 bound 嗎？若有，是 LRH assumption 失敗？」

#### B11. Three failure modes 只給 3 例 — 是否 representative？
- Q2/Q3 PTQ shape distortion、Qwen Q6_K head divergence、LoRA cross-task scale-shape contrast
- 沒展示「scale-dominated only」的 case
- Reviewer：「能否找到一個 scale 主導但 shape 小的 case？沒找到代表 framework 預測力不完整。」

---

### C. Methodology

#### C1. LoRA 為何 $W=I$ natural alignment？沒 formal argument
- Sec 4 L313 + Eq.(9) 隱式假設 $W=I$
- LoRA 加 low-rank update $\Delta W = BA$ 不引入 rotation — 但這個 reasoning 主文沒寫
- Reviewer：「LoRA 的 $\Delta W$ 可能對 hidden states 引入 effective rotation，為何 $W=I$？」

#### C2. Shape regularizer hyperparameter 選擇沒 sensitivity analysis
- $\lambda \in \{0, 0.1, 0.5\}$ — 只 3 值
- $k=8$ micro-step refresh — 怎麼選的？
- $|\mathcal{D}_{\mathrm{ref}}|=32$ — 為何 32 而非 64/128？
- Reviewer：「Hyperparameter sensitivity 圖在哪？」

#### C3. Step 300 analysis 為何選這個點？training curve 全程行為？
- L340「analysis at step 300」
- Reviewer：「為何不報 step 100, 200, 500, 1000 全部？選 step 300 是否 cherry-pick？」

#### C4. PTQ family 選擇 — 為何只 GGUF/GPTQ/BnB？AWQ、SpQR、HQQ 等不測？
- Sec 5.1 只 list 三個 family
- AWQ (Lin et al.) 是 widely-used PTQ method 沒覆蓋
- Reviewer：「PRISM 在 AWQ 上表現？」

#### C5. Llama+Qwen 是不是 cherry-pick？沒有 negative result family
- L325「cross-family pair Llama-3.1-8B and Qwen3-8B」
- Appendix 加 Ministral + DeepSeek
- Reviewer：「有沒有試過 PRISM 失敗的 family？positive results 全列，negative cases 隱藏？」

---

### D. Comparison & Positioning

#### D1. 「No prior representational similarity has been lifted to bound CE risk」(L149) — strict?
- Reviewer 可能想到一些 non-cited 的 prior work
- E.g., 是否有 paper 用 Frobenius distance + 一些 Lipschitz 推 generalization gap?
- Reviewer：「[some paper] 也做過類似事，請討論差別」

#### D2. PRISM vs CKA — paper 沒給 head-to-head numbers
- 引用 [klabunde2023similarity] 等說 CKA 不 predict downstream
- 但沒在 PRISM 的 grid 上實際算 CKA Spearman
- Reviewer：「Show CKA on same data — 量化 PRISM 比 CKA 多多少。」

#### D3. PRISM vs gradient-subspace forgetting analysis [steele2026subspace]
- L186 RW 提了這個 prior work
- 但沒對比兩 methods 在 forgetting prediction 的 Spearman
- Reviewer：「兩 methods 都做 LoRA forgetting，請對比預測力」

#### D4. PRISM vs layer-wise reconstruction loss for PTQ quality
- L183 RW 提「indirect proxies that do not account for nonlinear error accumulation」
- 但沒對比兩 methods 在 PTQ 的 Spearman
- Reviewer：「請補 layer-wise loss 在 same grid 的 Spearman」

---

### E. Writing/Framing

#### E1. Abstract「four 8B-scale LLM families」vs 主圖只展 2 — expectation mismatch
- 已在 Round 2 R2.5 標記為 defensible (4 families 確實在 appendix)
- Reviewer skim abstract → 看 main figures → 可能感覺 over-claim

#### E2. Conclusion 「scale-axis separability under cross-task LoRA drift」(R3.1 user skip)
- 「separability」暗示比實際 evidence 強的 statistical claim
- Reviewer 可能挑：「empirical evidence 只展示 shape-only 一個方向，不是 separability」

#### E3. Sec 5.4 「reduces forgetting on every benchmark」(L404 已軟化為 "essentially all cases")
- L165 已軟化為「across benchmarks」
- 但 BBQ MMLU paradox（λ=0→0.5 時 |ΔR| 0.52→0.54）若 reviewer 翻表會發現
- 已 cover 在 Limitation + appendix

#### E4. 「single forward pass」(L160, L183) — 實際對主文 $W=I$ 嚴格成立
- 之前 R1.5 user 已 confirm 主文 $W=I$ 下 Frobenius norm 恆等式不需 SVD
- 但 reviewer 不一定算過代數恆等式 → 可能挑

#### E5. 「Strong」rank correlation — 沒 literature reference 定義 strong threshold
- 0.82 / 0.835 算 strong 嗎？
- 多數 ML papers 用「strong」當 informal label，但 reviewer 嚴格的話會挑

---

### F. Reproducibility

#### F1. Code availability 主文沒提
- NeurIPS 預期有 code link
- Reviewer 自動扣 reproducibility 分

#### F2. LoRA hyperparameters (rank, lr, batch size) 主文沒寫
- Sec 5.1 只有 PRISM-side hyperparameters
- LoRA 本身的 config 不在
- Reviewer：「LoRA rank 是？學習率？無法 reproduce」

#### F3. Compute cost / wall-clock 沒報
- Reviewer：「PRISM 計算成本多少？對比 evaluation 多快？」

---

## Round 2 — Verify Each Question

**Lens**：每項目 grep 主文 + appendix 確認 (a) 是否真實 concern、(b) 是否已 cover、(c) 嚴重度。

### 驗證結果摘要

| ID | Question | 狀態 | 嚴重度 |
|---|---|---|---|
| A1 | $K_{\mathrm{feat}}$ 數值未在主文 | **REAL** | 🟡 |
| A2 | $K_{\mathrm{pred}}\le\sqrt{2}$ supremum | PARTIAL | 🟢 |
| A3 | LRH for PTQ assumption | **REAL** | 🟡 |
| A4 | $W=I$ default vs $W_N$ tighter | ALREADY ADDRESSED (Sec 5.5) | 🟢 |
| A5 | AR risk per-seq vs token-pooled | PARTIAL (appendix cover) | 🟢 |
| A6 | "Closed-form" definition | SPECULATIVE | 🟢 |
| B1 | 8B-only scope | **REAL** | 🟡 |
| B2 | No generation eval | **REAL** | 🟡 |
| B3 | Calibration size sensitivity | **REAL** | 🟡 |
| B4 | 0.835 vs 0.82 scope | ALREADY ADDRESSED (M3 + appendix) | 🟢 |
| **B5/B9/D2/D4** | **No external baseline (CKA, layer-wise)** | **REAL — BIG ISSUE** | 🔴 |
| B6 | Q2_K Qwen ρ inflates (267→313) | PARTIAL (L379 已 surface) | 🟡 |
| B7 | Shape reg target-task fit | PARTIAL (L406 提 trade-off 但沒數字) | 🟡 |
| B8 | Qwen LoRA noise floor | ALREADY ADDRESSED (appendix) | 🟢 |
| B10 | Bound violations in Fig 2 | NEED EMPIRICAL CHECK | 🟡 |
| B11 | Three failure modes representative? | SPECULATIVE | 🟢 |
| C1 | LoRA $W=I$ formal argument | PARTIAL | 🟢 |
| **C2** | **Hyperparameter sensitivity** | **REAL** | 🔴 |
| C3 | Step 300 cherry-pick | PARTIAL (Fig 3 全程 ✓) | 🟢 |
| C4 | Other PTQ methods (AWQ) | **REAL** | 🟡 |
| C5 | Cherry-pick families | SPECULATIVE | 🟢 |
| D1 | "No prior work" claim | SPECULATIVE | 🟢 |
| D3 | PRISM vs gradient-subspace numbers | **REAL** | 🟡 |
| E1-E5 | Writing nuances | ALREADY ADDRESSED 或 MINOR | 🟢 |
| **F2** | **LoRA hyperparameters missing** | **REAL — EASY FIX** | 🔴 |
| F1 | Code link | DEPENDS ON SUBMISSION | 🟡 |
| F3 | Compute cost | MINOR | 🟢 |

### 三大 🔴 Critical 級 attack vectors

1. **B5/B9/D2/D4 — 沒有 external baseline numbers**
   - 主文 reference CKA、layer-wise loss 等 prior work，但**沒在 PRISM 的 grid 實際算這些 baseline 的 Spearman**
   - Reviewer 對得起就會發現：「你說 CKA 不行，但你沒 quantify。請補。」

2. **C2 — Hyperparameter sensitivity 沒 analysis**
   - $\lambda \in \{0, 0.1, 0.5\}$ 只 3 點；$k=8$ refresh、$|\mathcal{D}_{\text{ref}}|=32$ 沒 sweep
   - Reviewer：「$\lambda$ 細粒度 sweep？$k=4, 16, 32$ 結果？」

3. **F2 — LoRA config 主文沒提**
   - LoRA rank、學習率、batch size 都沒寫
   - Reviewer：「無法 reproduce」

### 二級 🟡 Moderate 級 concerns

- **A1 $K_{\mathrm{feat}}$**：appendix L78 定義為 $\max_{j,k}\|h_{T,j}-h_{T,k}\|_2$（max pairwise token embedding distance）；主文沒給數值 → reader 不知 bound 有多鬆
- **B6 Scale axis 雙向**：L265「scale collapse」(ρ shrink) 是 typical narrative，但 L379 Qwen Q2_K GSM8K「ρ rises 267→313」是 inflation。已 surface 但 framing 不一致
- **B7 Target-task fit trade-off**：L406 (ii) 提「trade-off is monotone and controllable」；appendix Qwen caption 提「trade-off between target-task fit and downstream damage」；但**主文沒 target task accuracy 具體數字**
- **B10 Bound violations**：Fig 2 caption claim「safe zone where bound provably holds」——應驗證 Fig 2 上每個 point 都 inside green zone，否則破 bound
- **A3 LRH assumption**：對 PTQ 強度的 perturbation 是否成立沒 empirical evidence
- **B1 8B-only**: 沒 70B / MoE
- **B2 No generation eval**：teacher-forced CE 不等於 actual generation quality (HumanEval, MT-Bench)
- **B3 Calibration size sensitivity**：沒做
- **C4 Other PTQ methods**：AWQ、SpQR、HQQ 沒覆蓋
- **D3 PRISM vs gradient-subspace**：[steele2026subspace] 是直接競爭者沒對比

### 已 addressed / defensible（不需改）

- A4 $W=I$ default — Sec 5.5 三點 defense 完整
- B4 0.835 scope — appendix 已 cover
- B8 Qwen LoRA noise floor — appendix narrative
- E1-E5 wording 已軟化或 defensible
- A5/A6/C3/C5/B11/D1 — speculative 或 minor

---

## Round 3 — Defense / Modification Proposals

對 Critical 和 Moderate 級 concerns，分 (i) **defense in rebuttal** 和 (ii) **paper modification**。

### 🔴 Critical 級

#### B5/B9/D2/D4 — Missing external baselines

**Defense (rebuttal)**：
- "$\Omega$ at $W=I$ recovers a Procrustes similarity equivalent to **a scaled CKA on whitened features**; our ablation (Table~\ref{tab:baseline_combined}) reports its Spearman ($|r_s|=0.80$). Adding the scale term ($+0.064$) and head term defines what differentiates PRISM from CKA."
- 即「$\Omega$ alone IS our CKA-equivalent baseline」
- 但這個論點需要**論文中 explicitly 寫出 $\Omega \approx$ CKA equivalence**，目前沒寫

**Paper modification**：
- **Option A（rebuttal-only，不改論文）**：保留主文，rebuttal 時用上述論點
- **Option B（補一段 paragraph 在 Sec 5.5）**：明寫「$\Omega$ subsumes CKA-style alignment」
- **Option C（補 baseline table）**：在 appendix 加表格 list CKA / Frobenius distance / nuclear norm 在 same grid 的 Spearman——**最強但需重跑實驗**

**推薦 Option B**：低成本，將 ablation 框架擴展為「$\Omega$ alone $\approx$ CKA-style baseline」的論述。

---

#### C2 — Hyperparameter sensitivity

**Defense (rebuttal)**：
- "$\lambda$ is constrained by the bound's structure: too small $\to$ no regularization; too large $\to$ destroys task fit. We report the inflection range $\{0, 0.1, 0.5\}$ which already shows monotone behavior; finer sweep is future work."
- "$k=8$ chosen as Pareto knee in pilot study (overhead vs.\ regularizer freshness)"

**Paper modification**：
- **Option A（rebuttal-only）**：保留
- **Option B（appendix 加 sensitivity table）**：$\lambda \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$ 跑 1 個 (model, task) — 仍可控成本

**推薦 Option A**：rebuttal 用 design rationale 解釋；NeurIPS 通常接受 representative sweep 不要求 full grid。

---

#### F2 — LoRA hyperparameters missing

**Paper modification（必修）**：
- Sec 5.1 「Calibration and Hyperparameters」段落補一句：
  ```latex
  LoRA configuration: rank $r=16$, $\alpha=32$, dropout $0.05$,
  AdamW with learning rate $2{\times}10^{-4}$, batch size $8$.
  ```
- (具體數字需從 code/log 確認)
- **無 defense 路徑**——必須補

---

### 🟡 Moderate 級

#### A1 — $K_{\mathrm{feat}}$ 數值

**Defense**：
- "Appendix~\ref{app:kfeat} derives $K_{\mathrm{feat}} = \max_{j,k} \|h_{T,j} - h_{T,k}\|_2$, the max pairwise distance between token-embedding columns of $H_T$. This is **strictly tighter** than the naive Cauchy--Schwarz $\sqrt{2}\|H_T\|_2$ (Appendix Remark)."
- 對 ranking purpose, $K_{\mathrm{feat}}$ 是 model-level constant, 不影響 per-variant 排序

**Paper modification**（可選）：在 Sec 3.2 末加一句
- "$K_{\mathrm{feat}}$ depends only on the target head's pairwise token-embedding distances; for a given target it is a model-level constant invariant under choice of proxy."

---

#### B6 — Scale axis 雙向（collapse vs inflation）

**Defense**：
- "Scale mismatch $(\Delta\rho)^2$ is **direction-agnostic**: $\rho_P > \rho_T$ (inflation) and $\rho_P < \rho_T$ (collapse) are both penalized identically. The Sec 3.3 narrative \emph{example} of outlier-clipping (PTQ) emphasizes collapse, but the bound applies to both directions, including the Q2_K Qwen3 GSM8K case where $\rho$ inflates."

**Paper modification**：在 L265 改一句
```diff
- Aggressive bit-width reduction clips activation outliers
- ~\cite{dettmers2022gpt3, xiao2023smoothquant}, shrinking
- $\rho_P < \rho_T$; the scale axis isolates this \emph{scale collapse}.
+ Aggressive bit-width reduction commonly clips activation outliers
+ ~\cite{dettmers2022gpt3, xiao2023smoothquant}, shrinking $\rho_P$;
+ low-bit GGUF k-quant tiers can also \emph{inflate} $\rho_P$ (e.g.,
+ Qwen3-Base Q2_K on GSM8K, Sec.~\ref{subsec:decompose}). The scale
+ axis isolates magnitude divergence in either direction.
```

---

#### B7 — Shape reg 對 target task accuracy 的影響

**Defense**：
- "Higher $\lambda$ trades target-task fit for downstream preservation; the monotone behavior in Sec 5.4 (i, ii) is exactly this trade-off. We report TruthfulQA / BBQ accuracy as a function of $\lambda$ in Appendix [TBD]."

**Paper modification**（建議）：在 Appendix 加一個小 table 或 plot：target task accuracy at $\lambda \in \{0, 0.1, 0.5\}$。
- 若 target accuracy 在 $\lambda=0.5$ 大跌 → 強化「monotone trade-off」claim
- 若沒大跌 → 更強的 selling point（「regularizer 幾乎免費」）

---

#### B10 — Bound 違反檢查

**Defense**：
- "All variants in Fig 2 lie in the safe zone (i.e., $\mathcal{B} \ge |\Delta\mathcal{R}|$); the bound's tightness ratio $|\Delta\mathcal{R}| / \mathcal{B}$ ranges from X to Y across our grid."

**Paper modification**：
- 主文若有「safe zone」claim，appendix 應加一個 `min/max tightness ratio` 表格
- 若有任何 outlier → 解釋為 LRH approximation 限制

---

#### A3 — LRH for PTQ 強度

**Defense**：
- "PTQ perturbs each weight independently without basis transformation, so $W=I$ is a natural ansatz; our empirical results (Sec 5.5: $|r_s|{=}0.82$ across all PTQ variants) confirm LRH approximately holds throughout our grid."
- 等於把 strong correlation 當作 LRH 成立的 empirical evidence (post-hoc)

**Paper modification**：
- 在 Sec 3.1 LRH paragraph 加一句「empirically verified ex post in Sec 5.5」
- 弱化主動 claim 強度

---

#### B1 — 8B-only

**Defense**：
- "Compute constraints; 4 families × 8B is already a substantial benchmark. PRISM's bound is **scale-agnostic** in derivation (no parameter-count dependence in Theorem 1); 70B replication is concrete future work."

**Paper modification**：在 future work bullet "Beyond LoRA forgetting" 後加一個 bullet：「Scale-up replication on 70B / MoE families」

---

#### B2 — No generation eval

**Defense**：
- "PRISM bounds **CE risk gap** specifically; for generation tasks (HumanEval, MT-Bench) the bound applies to the AR risk component (Sec 3.4) but generation quality also depends on sampling, decoding, alignment — outside our scope. Empirically AR risk correlates with generation; we leave the connection for future work."

**Paper modification**：
- Limitations 段落明言 (但會暴露 attack surface)
- 或 future work 加 bullet「Connection to generation-quality metrics」
- **推薦 future work 加 bullet**

---

#### B3 — Calibration size sensitivity

**Defense**：
- "Selection variance enters $|\Delta\mathcal{R}|$ and $\mathcal{B}$ identically; rank ordering is invariant to subset size for sufficiently large $N$. Sec 5.1 already states this."
- L340 已寫此論點，可直接引

**Paper modification**：
- 可選：appendix 加 N=256/512/1024 sensitivity table
- 若不加：rebuttal 引 L340

---

#### C4 — Other PTQ methods (AWQ etc.)

**Defense**：
- "Three families (GGUF / GPTQ / BnB) are chosen as **representative** — they span the spectrum from RTN (GGUF Q2-Q8) to Hessian-aware (GPTQ) to optimization-based (BnB NF4). AWQ and SpQR are concrete future targets."

**Paper modification**：future work 加 bullet「Broader PTQ method coverage (AWQ, SpQR, HQQ, ...)」

---

#### D3 — PRISM vs gradient-subspace [steele2026subspace]

**Defense**：
- "[steele2026subspace] uses **gradient subspace angles** to fit forgetting; PRISM uses **hidden-state geometry** for a closed-form bound. The two are complementary: gradient angles diagnose \emph{how} drift happens, PRISM diagnoses \emph{where} (which axis). Direct comparison would require their public code on our grid."

**Paper modification**：RW 段已對比 (L186)，可加一句 explicit empirical claim 比較困難

---

### Phase 4 修改路徑（summary）

**MUST FIX**：
- F2 LoRA hyperparameters → Sec 5.1 加 1 句

**STRONG TO HAVE**：
- B5/B9 補「$\Omega \approx$ CKA-style baseline」段落 (Option B)
- B6 Sec 3.3 改 scale axis narrative（雙向）
- B2 Future work 加 generation eval bullet

**NICE TO HAVE**：
- A1 Sec 3.2 加 $K_{\mathrm{feat}}$ 解釋
- B7 Appendix 加 target task accuracy plot
- A3 Sec 3.1 LRH 弱化
- B1, C4 Future work bullets

**REBUTTAL ONLY**（不改論文）：
- A4, B4, B8, B10 (verify), B11, C1, C2, C3, C5, D1, D3, E1-E5, F1, F3

---

要從哪裡開始套用？建議路徑：
1. **F2 first**（必修，1 句）
2. **B6 next**（scale axis narrative 修正）
3. **B5/B9 third**（CKA baseline 段落）
4. 其他依 priority

