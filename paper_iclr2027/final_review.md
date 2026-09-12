# PRISM Paper — Final Review (10 Rounds)

Date: 2026-05-03
Source: `paper/neurips_2026.tex` + appendices + figure/table captions
Method: 10-round structured review. Rounds 1–5 = correctness. Rounds 6–10 = reviewer-style concerns.

---

## Status Tracker

| Round | Focus | Status |
|---|---|---|
| 1 | Math correctness (Thm 1, Prop 1, proofs, definitions) | DONE |
| 2 | References (citation existence, accuracy, formatting) | DONE |
| 3 | Logic flow (claim ↔ evidence, internal consistency) | DONE |
| 4 | Writing flow (transitions, terminology stability) | DONE |
| 5 | Experimental claims (numbers, methodology) | DONE |
| 6 | Reviewer concern: Novelty / contribution boundary | DONE |
| 7 | Reviewer concern: Empirical strength / coverage | DONE |
| 8 | Reviewer concern: Method design / limitations | DONE |
| 9 | Reviewer concern: Completeness / missing analysis | DONE |
| 10 | Reviewer concern: Presentation / reviewer-pleaseness | DONE |

---

## Round 1 — Math Correctness

### Verified correct
- **Theorem 1** (Eq. 13, line 258): $|\mathcal{R}_T-\mathcal{R}_P| \le K_{\mathrm{feat}}\sqrt{(\Delta\rho)^2 + 2\rho_T\rho_P(1-\Omega_W)} + K_{\mathrm{pred}}\|\Sigma_P^{1/2}(WH_T-H_P)\|_F$ — derivation in Appendix `proof_of_the_unified_risk_bound.tex` is sound (5 steps: triangle inequality, K_feat via simplex polarization, geometric decomposition, K_pred via simplex norm, assembly).
- **Proposition 1** (Eq. 11, line 230): the algebraic identity for $\frac{1}{n}\|Z_T-Z_PW\|_F^2$ uses $\|Z_PW\|_F^2 = \|Z_P\|_F^2$ (orthogonal $W$) correctly. The "complete the square" step (proof Eq. 16 → Eq. 17) is right.
- **K_feat bound** (Appendix Step 2): the simplex polarization derivation $\nabla_z\ell = \sum_{j\neq y}\hat{p}_j(h_{T,j}-h_{T,y})$ correctly gives $\|\nabla_z\ell\|_2 \le \max_{j,k}\|h_{T,j}-h_{T,k}\|_2$ using $\sum_{j\neq y}\hat{p}_j \le 1$ and triangle inequality.
- **K_pred bound**: $\|\hat{p}-e_y\|_2 \le \sqrt{2-2\hat{p}_y} \le \sqrt{2}$ derivation correct.
- **Definitions consistent**: $\rho_M = \|Z_M\|_F/\sqrt{n}$, $\Sigma_P = Z_P^\top Z_P/n$, $\Omega_W = \mathrm{Tr}(Z_T^\top Z_PW)/(\|Z_T\|_F\|Z_P\|_F)$ — used identically in main text and appendix.
- **AR Extension** (Sec 3.4): per-sequence length normalization $1/|y|$ matches the AR risk definition; Theorem 1 applies to stacked $Z^{\mathrm{AR}}$ matrices correctly.
- **LoRA bound specialization** (Eq. 19): under frozen head ($\gamma{=}0$), the bound reduces to $K_{\mathrm{feat}}\sqrt{(\rho_0-\rho_t)^2 + 2\rho_0\rho_t(1-\Omega)}$ — only valid at $W{=}I$ since $\Omega$ (no subscript) is the trace form. Consistent with main text default.
- **Nuclear-form tightness** (Appendix `tightness_of_nuclear_norm.tex`): $\Omega_F \le \Omega_N$ via $\|x\|_2 \le \|x\|_1$ on non-negative singular values. Correct.

### Potential issues
- **None major identified.** The proofs are self-contained and rigorous.
- **Minor notational note**: $\Delta H = WH_T - H_P$ is defined in main text Sec 3.2 (line 250) and figure caption, but the symbol's first use in Theorem 1 (line 258) writes the formula explicitly without naming $\Delta H$. Consistent but slightly noisy — readers see both notations.
- **Minor**: Sec 5.3 paragraph header "Head divergence" uses formula $\|\Sigma_P^{1/2}(WH_T - H_P)\|_F$ (line 275) but body discusses just "head term" — minor stylistic inconsistency, no math error.

---

## Round 2 — References

### Verified standard citations (sample)
- GPTQ `frantar2022gptq`, BnB/LLM.int8 `dettmers2022gpt3`, LoRA `hu2022lora`, SmoothQuant `xiao2023smoothquant` — standard ✓
- CKA `kornblith2019similarity`, SVCCA `raghu2017svcca`, generalized shape metrics `williams2021generalized`, decodability bound `harvey2024what` — standard ✓
- LIH `park2023linear`, relative representations `moschella2022relative`, Platonic Hypothesis `huh2024platonic` — standard ✓
- EWC `kirkpatrick2017overcoming`, scaling laws `kaplan2020scaling, hoffmann2022training`, weak-to-strong `burns2023weak` — standard ✓
- Benchmarks: MMLU `hendrycks2021measuring`, ARC `clark2018think`, TriviaQA `joshi2017triviaqa`, SQuAD `rajpurkar2016squad`, GSM8K `cobbe2021training`, TruthfulQA `lin2022truthfulqa`, BBQ `parrish2022bbq` — all standard ✓
- Procrustes refs in appendix: `schonemann1966generalized`, `gower2004procrustes` — standard ✓
- `ethayarajh2019contextual` (anisotropy, in `general_orthogonal_alignments.tex`) — standard ✓

### ⚠️ Verify
- **`steele2026subspace`** (line 191, Related Work): cited as "a recent geometric analysis empirically fits LoRA forgetting to gradient-subspace angles". 2026 paper — verify this exists, is correctly named, and accurately characterized. Reviewer may check if this work overlaps with PRISM more than acknowledged.
- **`polo2024tinybenchmarks`** and **`perlitz2024efficient`** (line 194): efficient evaluation citations — verify these exist and are characterized correctly.
- **`grattafiori2024llama`** (Llama-3.1-8B paper): verify exact arxiv id / venue.

### Citation formatting
- All citations use `\cite{...}` consistently.
- Numeric `compress` natbib option set; no formatting issues.

---

## Round 3 — Logic Flow & Internal Consistency

### Claim ↔ evidence map (verified)

| Claim location | Claim | Evidence | Status |
|---|---|---|---|
| Abstract | "PRISM achieves mean Spearman $r_s{\approx}0.82$ on PTQ" | Sec 5.2: "mean $r_s{\approx}0.82$ across the 2×5 grid" | ✓ |
| Abstract | "0.83 on LoRA forgetting" | Sec 5.2: "$r_s{=}0.831\pm 0.0722$" | ✓ |
| Abstract | "shape regularizer outperforms experience replay" | Sec 5.4: trace -19% vs replay -9% on TruthfulQA | ✓ |
| Sec 5.3 | "Q6_K on MMLU: δ=2.61, γ=85.08" | Table `tab:qwen_decomposition_all_bound` (Appendix) | ⚠️ verify table values |
| Sec 5.5 | "$\mathcal{B}_N$ at $r_s{=}0.91$" | Table 3 baseline_combined | ✓ |
| Discussion | "$r_s{\approx}0.82$ on PTQ, 0.831 on LoRA" | repeats Sec 5.2 | ✓ |

### Issues

- **Sec 5.3 "Head divergence" example specificity**: cites Q6_K MMLU with $\delta{=}2.61$, $\gamma{=}85.08$, and Q8_0 with $\delta{=}0.51$, $\gamma{=}26.81$. Verify these exact numbers in Table `tab:qwen_decomposition_all_bound` (Appendix) — currently not visible without checking that specific table.
- **Sec 5.3 "Scale-axis separability"**: cites Qwen3-Base Q2_K on GSM8K $\rho_P$ rising from 267→313 alongside shape drift. Verify in Appendix table.
- **TruthfulQA table** (Sec 5.4): cites mean numbers 0.84/0.76/0.68 — should match `tab:reg_compact_llama_truthfulqa`.

### Logical consistency
- ✓ Abstract → Intro → Theorem → Experiments — consistent narrative chain.
- ✓ Sec 4 "Applications" subsections (Quantization at $W=I$; LoRA $\gamma=0$) — correctly derived from Theorem 1 specializations.
- ✓ Sec 5.5 ablation closer "design-driven, not predictiveness-driven" — supported by both $\mathcal{B}$ ($r_s=0.82$) and $\mathcal{B}_N$ ($r_s=0.91$) being "strong"; the framing is now consistent with Fig 1 caption + Sec 3.3 paragraph.

---

## Round 4 — Writing Flow & Terminology Stability

### Issues found

#### W1. Intro Para 5 still contains "PRISM failure" wording (line 167)
Current: "So a **PRISM failure** is not merely a flag but a *direction*..."
- We fixed this in the Abstract ("the dominant axis points to a direction") but Intro Para 5 still has the ambiguous phrasing.
- **Fix**: rewrite to "So the dominant axis points to a *direction* for remediation, not just a flag" (parallel with Abstract).

#### W2. K_feat technical aside in Intro Para 5 (line 167)
Current: "A simplex polarization argument keeps the feature-side Lipschitz constant $K_{\mathrm{feat}}$ informative at LLM vocabulary scale."
- Too technical for Intro; already covered in Sec 3.2 + Appendix.
- Identified earlier as B2 candidate; not yet removed.
- **Fix**: delete the sentence.

#### W3. Sec 5.3 "Head divergence" paragraph header still says "touch \texttt{lm\_head}" (line 389)
- Fig 1 caption was updated to "quantize \texttt{lm\_head}" for precision.
- Sec 5.3 header inconsistent.
- **Fix**: change "tiers that touch \texttt{lm\_head}" → "tiers that quantize \texttt{lm\_head}".

#### W4. Sec 4 "Applications" overlaps Sec 3.5 + Intro (lines 309-320)
- Quantization paragraph: PTQ + W=I + γ trigger conditions — already in Sec 3.3 + Intro.
- LoRA paragraph: γ=0 + backbone drift — already in Sec 3.5.
- **Optional fix**: consolidate Sec 4 (delete or reduce to single tightening sentence).

#### W5. Discussion para 1 repeats Abstract + Contributions (line 434)
- "We presented PRISM... Theory... Framework... Empirical..." parallels Abstract + Contributions almost word-for-word.
- **Fix**: remove or compress to one transitional sentence.

#### W6. Sec 5.3 "Three such failure modes recur in our experiments." (line 384)
- Filler line, the next paragraphs already enumerate them.
- **Fix**: delete.

#### W7. Sec 6 mitigations list duplicates Sec 5.3 (line 437)
- "(per-channel outlier smoothing for scale, Hessian-aware reconstruction for shape, FP16-\texttt{lm\_head} retention for head)" appears in Sec 5.3 paragraph "From diagnosis to remediation" verbatim.
- **Fix**: in Sec 6, refer back to Sec 5.3 instead of repeating.

#### W8. Future work paragraph (line 441)
- "We describe future directions in detail in Appendix..." is wordy.
- **Fix**: "Future directions (Appendix...): beyond LoRA forgetting; diagnostic applications; beyond LLMs."

### Terminology stability
- ✓ "trace form $\Omega$", "nuclear form $\Omega_N$", "PRISM bound $\mathcal{B}$" — used consistently after recent updates.
- ✓ "feature alignment error $\delta$", "head discrepancy $\gamma$" — consistent.
- ⚠️ "PRISM failure" (Intro) vs "dominant axis points to..." (Abstract) — needs unification (W1 above).

---

## Round 5 — Experimental Claims & Methodology

### Numbers consistency (cross-section)

- ✓ $r_s \approx 0.82$ (PTQ) appears in: Abstract, Sec 5.2, Sec 5.5, Discussion
- ✓ $r_s \approx 0.83$ (LoRA) appears in: Abstract, Sec 5.2 ($0.831\pm 0.0722$), Discussion (0.831)
- ✓ $\mathcal{B}_N$ at $r_s{=}0.91$ in: Sec 5.5, Fig 1 caption (after recent edit)
- ✓ Spearman gap $\sim 0.09$ between $\mathcal{B}$ and $\mathcal{B}_N$ — Sec 5.5 mentions this

### Methodology checks

| Aspect | Description | Status |
|---|---|---|
| Calibration sample size | 512 (PTQ), 256 (LoRA forgetting) | Sec 5.1 specifies; mentions robustness to subset choice |
| Reference set for regularizer | 32 pre-training sequences | Sec 5.1 + Sec 5.4 consistent |
| Hyperparameter sweep | trace λ ∈ {0.01,0.05,0.1,0.5,1.0}; replay λ ∈ {0.001..0.1} | Sec 5.1 specifies |
| LoRA setup | rank-32 attention, AdamW, lr 1e-5, bf16 | Sec 5.1 specifies |
| Step | analysis at step 300 | Sec 5.1 specifies |
| Hardware | RTX 5090 32GB | Sec 5.1 specifies |
| Teacher forcing | single forward pass over gold span | Sec 5.1 specifies |

### Methodology concerns

#### M1. "matching each benchmark's scoring rule" claim (line 342)
- The current text says "matching each benchmark's scoring rule and producing deterministic per-sample residuals". The "matching scoring rule" claim is vague — different benchmarks may use different scoring (multiple-choice accuracy vs F1 vs exact-match). Teacher-forced CE doesn't really "match" each benchmark's scoring rule.
- **Reviewer may push back**: "Did you actually use each benchmark's official scoring? Or are all reduced to per-token CE?"
- **Fix**: clarify or weaken — e.g., "All risks are computed teacher-forced (gold span scored in a single forward pass) for variant ranking; aggregate accuracy reported in benchmark units when applicable."

#### M2. λ choice for regularizer (line 415)
- "trace at λ=1.0 ... replay at λ=0.01" — chosen as "each method's sweep optimum under the identical evaluation protocol".
- Reviewer concern: optimum chosen on which set? If on test set, that's leakage. If on train, how is "downstream best" defined?
- Sec 5.1 says "checkpoints every 25 steps, analysis at step 300" — implies a fixed step, but optimum vs ΔR-best may still need clarification.

#### M3. Statistical significance
- $r_s = 0.831 \pm 0.0722$ vs $0.820 \pm 0.0471$ — comparable, but no significance test reported.
- Comparison shape regularizer vs replay-CE: difference 0.68 vs 0.76 mean ΔR — no error bars or significance test.

#### M4. Number coverage in main text
- Only Llama MMLU table appears in main body (Table 1).
- Other 9 main-grid cells (Llama 4 other benchmarks + Qwen3 5 benchmarks + 6 instruct/Ministral/DeepSeek) all in appendix.
- Reviewer may say: "Main paper feels light on data — too much hidden in appendix."

---

## Round 6 — Reviewer Concern: Novelty / Contribution Boundary

### Likely reviewer questions

#### N1. "How is PRISM different from CKA / SVCCA / generalized shape metrics?"
- **Paper's response**: Sec 2 Para 1 says representational similarities don't link to risk; Sec 3 lifts Procrustes residual into closed-form CE bound on the deployed head.
- **Reviewer pushback**: "Your $\Omega = \mathrm{Tr}(Z_T^\top Z_P)/(\|Z_T\|_F\|Z_P\|_F)$ is mathematically a specific cosine-style similarity. Why couldn't I compute CKA + a separate scale measure and get the same axes?"
- **Strength of paper's reply**: The exact Procrustes residual decomposition into scale + shape (Prop 1) is genuinely new framing, but the *individual quantities* are not novel. The novelty is the *risk-bound lift* + decomposition + actionability.
- **Recommended counter**: emphasize "no prior CKA-based work has lifted to a CE risk bound" + decomposition is exact (not heuristic).

#### N2. "Compared to harvey2024what (decodability bound), what's new?"
- **Paper's response**: "decodability bound routes through whitened kernels and newly-trained linear probes rather than the deployed prediction heads."
- **Reviewer pushback**: "harvey2024what bounds risk on the actual prediction surface (probed). Why is your 'deployed head' framing materially different?"
- **Recommended counter**: PRISM's bound is computable in a single forward pass without retraining a probe; covariance-weighted head term captures *actual* head divergence (not probe-fitted). This is a deployment-friendly property.

#### N3. "Theoretical novelty is moderate; main contribution is empirical?"
- The bound itself uses standard techniques (triangle inequality, Lipschitz, Jensen, Procrustes). Novelty is the *combination*: simplex-polarization $K_{\mathrm{feat}}$ + covariance-weighted head term + axis decomposition.
- **Risk**: a theoretical reviewer may say "no individual step is novel; the assembly is engineering."
- **Counter**: the assembly enables (i) actionable diagnosis, (ii) differentiable regularizer — both empirical novel.

#### N4. "Contribution 2 says 'unified diagnostic for PTQ + LoRA'. Is this 'unified' or two separate applications?"
- The bound is one formula but practical use differs (PTQ engages all 3 axes; LoRA only 2 with γ=0).
- "Unified" might be over-claimed if the head term vanishes in LoRA.
- **Mitigation**: the same Theorem applies to both — the *bound structure* is unified even if some terms vanish.

---

## Round 7 — Reviewer Concern: Empirical Strength / Coverage

### E1. Model-scale concern
- **Only 8B models tested.** Reviewer will ask: does PRISM generalize to 70B / 405B?
- **Mitigation**: the math is scale-agnostic; near-isometry assumption likely *strengthens* at larger scale (better-trained features). Add a sentence about scale generalization argument.

### E2. Benchmark diversity
- **5 benchmarks: ARC, MMLU, SQuAD, TriviaQA, GSM8K.** All English, all teacher-forced.
- **Missing**: code generation, chat-style, multilingual, long-context. Reviewer may say "benchmarks are narrow."
- **Mitigation**: scope clarification — 5 benchmarks span knowledge / reading comprehension / multi-step reasoning. Free-running generation is acknowledged as future work in Sec 6.

### E3. Baseline comparison: only experience replay
- **Sec 5.4** compares against replay-CE only. Reviewer will ask about EWC, MAS, Online EWC, LoRA-specific baselines (e.g., Wise-FT).
- **Mitigation**: explain replay is the most natural matched-data-budget baseline (uses same $\mathcal{D}_{\mathrm{ref}}$); other methods either change the optimization geometry (EWC needs Fisher) or are out of scope.
- **Risk**: a reviewer expecting EWC head-to-head may downgrade. Could add EWC to appendix.

### E4. PTQ coverage
- 3 backends (GGUF, GPTQ, BnB). Missing: AWQ, SqueezeLLM, SpinQuant, more recent low-bit methods.
- **Mitigation**: GGUF/GPTQ/BnB cover the deployment-popular set; AWQ/SpinQuant follow similar patterns.

### E5. LoRA fine-tune source diversity
- Only 2 sources: TruthfulQA, BBQ.
- **Reviewer**: "What about Alpaca-style instruction tuning, math fine-tuning, code fine-tuning?"
- **Counter**: the 2 chosen tasks elicit *different drift geometries* (Sec 5.3 scale-axis separability via TruthfulQA/BBQ contrast) — this was a deliberate design choice for axis-localization, not coverage.

### E6. Statistical significance
- $r_s = 0.83$ vs $0.82$ — no significance test. Standard deviations reported but no $p$-values, confidence intervals, or paired tests across cells.
- **Reviewer**: "Are these differences meaningful?"
- **Mitigation**: add bootstrap CI or paired Wilcoxon between $\mathcal{B}$ and $\delta$/$\Omega$ rankings.

### E7. Replication: cross-family vs same-family
- "Cross-family Llama--Qwen" pair is featured. But the LIH assumption (Sec 3.1) is about *same-family encoders*. How does cross-family work?
- **Risk**: reviewer points out an apparent contradiction.
- **Mitigation**: clarify that the bound holds *for any orthogonal $W$*; the LIH assumption motivates *why* an orthogonal $W$ exists. Cross-family may have larger $\delta$ but bound still valid.

---

## Round 8 — Reviewer Concern: Method Design / Limitations

### D1. Bound looseness (paper acknowledges)
- Sec 6: "Tight absolute estimation of $|\Delta\mathcal{R}|$ is a complementary problem we leave to future work."
- $K_{\mathrm{feat}}$ can be large (per main text discussion of "informative at LLM vocabulary scale"); even with simplex polarization, the bound is $\sim 10\times$ the empirical $|\Delta\mathcal{R}|$ in some cells (visible in Fig 2 where points are far below $y{=}x$).
- **Reviewer**: "If the bound is 10× off in absolute terms, it's not useful as a bound, only as a ranking score."
- **Counter (in paper)**: ranking is the calibrated use case; Spearman shows it works.

### D2. Choice of $W=I$ vs $W=W_N$
- $W=W_N$ achieves $r_s=0.91$ (better than $W=I$'s 0.82) per Sec 5.5.
- The paper defends $W=I$ choice via autograd-compatibility + frozen-head simplification.
- **Reviewer concern**: "If $W_N$ is 0.09 Spearman better, why not adopt it as default?"
- **Counter**: $W_N$ requires per-step SVD (cost) and inflates $\gamma$ (head term) under near-frozen head. Trade-off explicitly discussed in Sec 5.5 + Sec 3.3 + Appendix C.

### D3. Asymmetry: $K_{\mathrm{feat}}$ uses $H_T$ not $H_P$
- Definition $K_{\mathrm{feat}} = \max_{j,k}\|h_{T,j}-h_{T,k}\|_2$ uses target's head columns. Why not $H_P$ or symmetric?
- **Reviewer**: "Is this a typo? Should it be $H_T$ + $H_P$ symmetric?"
- **Justification (in proof)**: the hybrid risk $\mathcal{R}_{P\to T}$ uses $H_T$ for the feature-side delta, so $K_{\mathrm{feat}}$ depends on $H_T$. Documented but worth flagging.

### D4. Assumption: $\phi_T(x) \approx \phi_P(x) W$ is approximate
- Sec 3.1 says "we assume the two backbones lie in a common family: there exists an orthogonal $W \in \mathcal{O}(d)$ such that $\phi_T(x) \approx \phi_P(x)\, W$ for all $x$."
- The bound *does not require* this approximation; the proof works for any $\phi_T, \phi_P$.
- **Risk**: reviewer may think the assumption is needed; clarify it's only motivation.
- **Fix**: rephrase as "we *expect* (per LIH/relative-rep) that..." rather than "we assume".

### D5. Calibration sample size sensitivity
- 512/256/32 samples — but no robustness study showing rankings stable across sizes.
- **Reviewer**: "What if I use 50 samples? 1000?"
- **Mitigation**: "$|\Delta\mathcal{R}|$ spans 1-2 orders of magnitude" claim hand-waves robustness; no plot.

### D6. K_pred upper bound √2
- Tight at $\hat{p}_y \to 0$ (model totally wrong).
- For typical well-trained models, $\hat{p}_y$ is moderate, so $K_{\mathrm{pred}} < \sqrt{2}$.
- The bound uses worst-case $\sqrt{2}$ → another source of looseness.

### D7. Cross-entropy only
- The bound is for CE loss specifically. Generalization to MSE / 0-1 / contrastive losses requires re-derivation.
- Acceptable scope but worth noting.

---

## Round 9 — Reviewer Concern: Completeness / Missing Analysis

### C1. Failure modes of PRISM
- When does the bound *fail* to predict ranking? No discussion.
- **Suggested addition**: identify cells where $r_s$ is low (e.g., GSM8K Q2_K — outlier?). Acknowledge.

### C2. Computational cost
- "Single forward pass per variant" claim — but extracting $Z_T, Z_P$ for $n=512$ samples is non-trivial in memory.
- No comparison: how much faster is PRISM vs full benchmark eval?
- **Reviewer**: "Why use PRISM instead of just running the benchmark?"
- **Counter**: PRISM applies *cross-benchmark* — once you compute $Z_T, Z_P$, you can evaluate against multiple risk gaps. But this isn't articulated clearly.

### C3. Sensitivity to $\lambda$ (regularizer)
- Sec 5.4 reports trace at $\lambda=1.0$, replay at $\lambda=0.01$ — sweep optima.
- No plot of $\lambda$ sweep curves; reader can't verify the claim.
- **Suggested**: appendix sweep plot.

### C4. What if the LIH assumption fails?
- Cross-architecture (Llama vs Qwen) may have weaker isometry.
- No diagnostic for "is LIH satisfied here?"
- **Suggested**: report empirical $\Omega$ values across model pairs as a sanity check.

### C5. Decomposability claim is anecdotal
- Sec 5.3 cites specific cells (Q2_K MMLU, Q6_K MMLU, etc.) as evidence for failure modes.
- No systematic statistic across all cells (e.g., "shape > scale by ≥1 OOM in X% of low-bit cells").
- **Suggested**: aggregate decomposition statistics in appendix.

### C6. Negative results
- No "where PRISM doesn't work" section.
- Even in main text, only confirmed-works cases shown.
- **Risk**: reviewer suspects cherry-picking.

### C7. Comparison to scaling-law-based extrapolation
- Sec 2 Related Work mentions scaling laws as alternative; no head-to-head comparison.
- **Reviewer**: "How does PRISM rank PTQ variants vs a scaling-law extrapolation that knew bit-width?"
- **Counter**: scaling laws don't predict variant-vs-base degradation (paper claims). But empirical comparison would strengthen.

### C8. Connection to Eq. 8 / safety / ethics
- Memory mentions "safety" experiments earlier (forgetting_safety logs) — not in paper.
- This is fine if descoped; but reviewer may ask "what about catastrophic forgetting of safety alignment?"

---

## Round 10 — Reviewer Concern: Presentation / Reviewer-Pleaseness

### P1. Abstract quality (after recent edits)
- ✓ Strong opening (problem + gap)
- ✓ Insight sentence (LLM-specific structures)
- ✓ Contribution sentence (PRISM = closed-form bound + 3 axes)
- ✓ Advantage (dominant axis = direction; differentiable → regularizer)
- ✓ Experiments (specific Spearman + axes + replay comparison)
- **Minor**: "experience replay" technical name OK; could reviewers unfamiliar with continual-learning recognize it? Probably yes.

### P2. Figure 1 (concept figure)
- Visual centering issues earlier addressed. Caption now precise (γ vanishing condition tied to W=I).
- ⚠️ Caption is still long (~7 lines). Could reviewers skim it cleanly?
- ✓ The 3-axis decomposition + W=I/W_N message is now explicit.

### P3. Section 4 ("Applications") feels redundant
- 8 lines that mostly restate Sec 3.5 + Intro. Reviewer may flag as filler.
- **Recommended**: Compress to 3-4 lines, or merge into Sec 5.1.

### P4. Discussion section weakness
- Para 1 (lines 434) almost word-for-word repeats Abstract+Contributions.
- Scope and Limitations is the most valuable part.
- **Recommended**: cut Para 1 entirely; lead Discussion directly with Scope+Limitations.

### P5. Math density in Theorem 1 statement (line 256-258)
- The full bound formula on one line is dense. Two `\underbrace`s help.
- ✓ Acceptable for theorem-style audience.

### P6. Writing style consistency
- Hyphenation: "post-training" used consistently; "fine-tuning" used; "low-bit"; "frozen-head" — all hyphenated correctly.
- Abbreviations: PTQ, LoRA, CE — defined on first use.

### P7. Table formatting
- Table 1 (Llama MMLU): heatmap-like shading — reviewer-friendly.
- Table 2 (regularization): compact form, $\Omega$ + $|\Delta\mathcal{R}|$ side-by-side.
- Table 3 (ablation): 2-block structure ($W=I$ vs $W=W_N$).
- ✓ Consistent visual style across tables.

### P8. Citation density
- ~30 citations — reasonable for a NeurIPS paper.
- ⚠️ "steele2026subspace" is the only forward-dated cite; verify exists.

### P9. Acknowledgments / NeurIPS checklist
- Checklist is `\input{checklist.tex}` — should be filled out completely. Reviewer-meta checks this.

### P10. "Three contributions" framing in Sec 1
- Contribution items 1-3 each have **bold** lead-ins. Visually clear.
- Item 3 is one massive sentence with semicolons (predictiveness; decomposability; actionability) — reads OK but dense.

---

## Summary: Top Action Items

### Must-fix (correctness / reviewer-blocking)
1. **W3** (Round 4): Sec 5.3 "touch lm_head" → "quantize lm_head" (consistency with Fig 1).
2. **W1** (Round 4): Intro Para 5 "PRISM failure" wording — fix consistent with Abstract.
3. **M1** (Round 5): "matching each benchmark's scoring rule" claim — clarify or weaken.
4. **R2 verify**: `steele2026subspace` citation — confirm it exists and is correctly characterized.

### Should-fix (clarity / reviewer-pleaseness)
5. **W2** (Round 4): delete K_feat sentence in Intro Para 5.
6. **W4** (Round 4): consolidate Sec 4 (Applications redundancy).
7. **W5/W7** (Round 4): cut Discussion para 1 + dedupe mitigations list.
8. **W6** (Round 4): delete Sec 5.3 filler line.
9. **W8** (Round 4): tighten Future Work paragraph.

### Nice-to-have (reviewer concern preemption)
10. **E6** (Round 7): add statistical significance test for $r_s$ comparisons.
11. **E3** (Round 7): mention EWC explicitly in Sec 5.4 framing (why replay-only is the right choice).
12. **D5** (Round 8): add calibration-size robustness mini-table in appendix.
13. **C1/C5/C6** (Round 9): aggregate decomposition statistics + acknowledge negative cells.
14. **C3** (Round 9): $\lambda$ sweep curves in appendix.

### Minor / optional
15. Trim Fig 1 caption further if page-budget tight.
16. **P4**: reorganize Discussion to lead with Scope+Limitations.

---
