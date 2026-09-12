# AI Review Rebuttal & Action Plan

This document goes through every point raised in `ai_review.md` (Google PAT), verifies whether it is grounded in the current paper, and proposes a response (fix / reframe / push back).

**Verification methodology**: Each claim was cross-checked against `neurips_2026.tex`, the appendix `.tex` files, the actual table outputs in `paper/tables/quantization/`, and `paper/paper.bib`. Findings are labelled:

- ✅ **Confirmed**: the issue is real and should be fixed
- ⚠️ **Partially valid**: the substance has merit but the wording / framing in the review needs adjustment
- ❌ **Refuted**: the claim is incorrect (likely a model hallucination)

Severity:

- 🔴 **Critical** — mathematical or factual error that would damage credibility
- 🟡 **Major** — substantive concern about claim/evidence alignment
- 🟢 **Moderate / Minor** — clarity, formatting, or typo

---

## 🔴 Critical Issues

### C1. Figure 1 caption: $\gamma$-vanishing condition is mathematically wrong

**Reviewer claim**: The Fig 1 caption uses $W^{*} = \arg\min_W \|Z_T - Z_P W\|_F$ (Procrustes-optimal) and states $\gamma$ "vanishes when $H_T = H_P$ (frozen-head LoRA)". This is wrong: $\gamma = K_{\mathrm{pred}} \|\Sigma_P^{1/2}(W^{*} H_T - H_P)\|_F$, so $H_T = H_P$ alone does not imply $\gamma = 0$ unless $W^{*} = I$.

**Verification** (`paper/figures/prism_geometric_decomposition.tex`, line 170):
> "Under Procrustes alignment $W^{*}{=}\arg\min_{W\in\mathcal{O}(d)}\|Z_T{-}Z_PW\|_F$ ... $\gamma$ measures $\Delta H{=}W^{*}H_T{-}H_P$ ..., vanishing when $H_T{=}H_P$ (frozen-head LoRA)."

✅ **Confirmed**. $\Delta H = W^{*} H_T - H_P$. If $H_T = H_P$, then $\Delta H = (W^{*}-I) H_T$, which vanishes only when $W^{*} = I$. The main text (Sec 3.3) correctly notes this: "at $W = I$, LoRA and FP16-head PTQ ... preserve $H_T = H_P$ so the head term vanishes."

**Action (proposed)**: Two clean options:
- **(a)** Replace $W^{*}$ in Fig 1 with $W = I$ (since the main text uses $W = I$ as default and the failure-mode examples in the caption are also under $W = I$).
- **(b)** Keep $W^{*}$ but rephrase: "vanishing when $H_T = H_P$ \emph{and} $W^{*} = I$ (e.g., frozen-head LoRA where Procrustes returns identity)".

Recommend (a) — cleaner and consistent with the main-text default.

---

### C2. App C.3 (general orthogonal alignments): "isotropic features → head term independent of $W$" is wrong

**Reviewer claim**: With $\Sigma_P \approx \lambda I$,
$$\gamma(W) = K_{\mathrm{pred}} \cdot \sqrt{\lambda} \cdot \|W H_T - H_P\|_F,$$
which still depends on $W$ through the cross-term $\mathrm{Tr}(H_P^\top W H_T)$ in the squared norm expansion. So the claim "head term becomes independent of $W$" is algebraically false.

**Verification** (`appendix/general_orthogonal_alignments.tex`, line 83):
> "If $\Sigma_P \approx \lambda I$, the head term becomes independent of $W$ and the joint objective reduces to a standard Procrustes problem."

✅ **Confirmed wrong**. $\|W H_T - H_P\|_F^2 = \|H_T\|_F^2 + \|H_P\|_F^2 - 2 \mathrm{Tr}(H_P^\top W H_T)$ — the last term IS a function of $W$.

**Action (proposed)**: Rephrase to:
> "If $\Sigma_P \approx \lambda I$, the head term reduces to $K_{\mathrm{pred}} \sqrt{\lambda}\|W H_T - H_P\|_F$, and the joint objective minimizes $K_{\mathrm{feat}} \sqrt{\|Z_T - Z_P W\|_F^2/n} + K_{\mathrm{pred}}\sqrt{\lambda}\|W H_T - H_P\|_F$. Each term individually admits a closed-form Procrustes minimizer (different in general); the sum is not Procrustes."

---

### C3. App C.3: "frozen-head LoRA → head term vanishes for any $W$" is wrong

**Reviewer claim**: Same logic. With $H_T = H_P$, $\gamma(W) = K_{\mathrm{pred}}\|\Sigma_P^{1/2}(W - I) H_T\|_F$, which vanishes only when $W = I$ (or $W$ acts as identity on the column space of $H_T$, a measure-zero condition).

**Verification** (`appendix/general_orthogonal_alignments.tex`, line 92):
> "**LoRA ($H_T = H_P$):** The head term vanishes for any $W$, so the joint problem reduces to standard Procrustes."

✅ **Confirmed wrong**.

**Action (proposed)**: Rephrase to:
> "**LoRA ($H_T = H_P$):** The head term becomes $K_{\mathrm{pred}}\|\Sigma_P^{1/2}(W - I) H_T\|_F$, which vanishes at $W = I$. Since $W = I$ is a feasible point in $\mathcal{O}(d)$, the bound at $W = I$ is always available and is the form we use throughout the LoRA experiments and the regularizer of Sec 3.5. The Procrustes-optimal $W_N$ would in general inflate $\gamma$ for an arbitrary $W_N \neq I$."

---

### C4. App A.4: "strictly tighter than the spectral bound" is overstated

**Reviewer claim**: The proposed covariance-weighted head bound uses Jensen's inequality ($\mathbb{E}\|X\|_2 \le \sqrt{\mathbb{E}\|X\|_2^2}$), which is an upper-bounding step. So in some configurations (e.g., $\Sigma_P = I$ and rank-1 $\Delta H$ aligned with the most-active direction), the spectral bound can be tighter.

**Verification** (`appendix/proof_of_the_unified_risk_bound.tex`, line 214):
> "This is strictly tighter than the spectral bound $\gamma \le K_{\mathrm{pred}} \|W H_T - H_P\|_2 \cdot \mathbb{E}[\|z\|_2]$, which assumes worst-case alignment between features and head error."

⚠️ **Partially valid**. Our bound captures the active-subspace effect (vanishes if $\Delta H$ lies in null$(\Sigma_P)$), which the spectral bound does not. But "strictly tighter" overstates because of the Jensen's inequality step — in the isotropic + rank-1-aligned worst case, the bounds coincide and our Jensen step can introduce slack.

**Action (proposed)**: Soften to:
> "This is sharper than the spectral bound $\gamma \le K_{\mathrm{pred}} \|W H_T - H_P\|_2 \cdot \mathbb{E}[\|z\|_2]$ in the typical anisotropic regime, where the covariance weighting suppresses head-error directions outside the active subspace (the spectral bound assumes worst-case alignment of $\Delta H$ with the data-dominant direction). The two coincide in the isotropic limit; our form remains the natural one because it is autograd-compatible and admits the direct Procrustes specialization at $W = I$."

---

### C5. App B (and similar): Cauchy–Schwarz misattribution

**Reviewer claim**: The inequality $\|Z_T^\top Z_P\|_F = \sqrt{\sum \sigma_i^2} \le \sum \sigma_i = \|Z_T^\top Z_P\|_*$ is justified "by Cauchy–Schwarz" — but this is not Cauchy–Schwarz; it is the basic $\ell_2 \le \ell_1$ inequality for non-negative vectors.

**Verification** (`appendix/tightness_of_nuclear_norm.tex`, lines 28–32 and 42):
> "By the Cauchy--Schwarz inequality applied to the vectors $(\sigma_1, \ldots, \sigma_d)$ and $(1, \ldots, 1)$: $\sqrt{\sum_i \sigma_i^2} \le \sum_i \sigma_i$ ..."
> "since $\|A\|_F \le \mathrm{Tr}(A)$ for any positive semidefinite $A$, by Cauchy--Schwarz on the eigenvalues"

✅ **Confirmed** (the inequality holds; the attribution is wrong). C–S applied to $(\sigma)$ and $(1,\ldots,1)$ actually yields $\|\sigma\|_1 \le \sqrt{d}\|\sigma\|_2$ — the reverse direction. The actual fact used is: for non-negative reals, $\sum a_i^2 \le (\sum a_i)^2$ (cross terms are $\ge 0$).

**Action (proposed)**: Replace "by Cauchy–Schwarz" with the correct justification. Two options:
- "since for non-negative reals $\sum_i \sigma_i^2 \le (\sum_i \sigma_i)^2$ (cross terms are non-negative)"
- "since $\|x\|_2 \le \|x\|_1$ for any vector with non-negative entries"

---

## 🟡 Major Issues

### M1. Sec 5.4 hyperparameter selection: "sweep $|\Delta\mathcal{R}|$-best" is data leakage

**Reviewer claim**: Reporting "each at its sweep $|\Delta\mathcal{R}|$-best" selects $\lambda$ by maximizing the very downstream metric used as the test outcome. This is in-sample model selection on the test set.

**Verification** (`neurips_2026.tex`, lines 418, 426):
> "...the latter two share a $32$-sample reference set and are each method's sweep $|\Delta\mathcal{R}|$-best."
> "compare our trace against the replay baseline (each at its sweep $|\Delta\mathcal{R}|$-best)..."

✅ **Confirmed legitimate concern**. Both methods are tuned on the same metric they're scored on; this advantages whichever has more usable hyperparameters within its sweep grid. While the comparison is fair-by-symmetry (both methods get the same treatment), the reported numbers do not reflect generalization to a held-out validation split.

**Action options** (in order of effort):
- **(a) Reframe** (cheapest, no new experiments): Acknowledge the in-sample tuning explicitly: "$\lambda$ is selected per method to minimize downstream $|\Delta\mathcal{R}|$ on the same evaluation suite; a held-out validation split for $\lambda$ selection is left to future work. The point of this comparison is the *mechanism* contrast (replay re-fits data; trace contracts shape geometry), not absolute gains under blind selection."
- **(b) Rerun** with held-out validation: split benchmarks into a tuning subset (e.g., 2 of 5) for $\lambda$ selection and a test subset (3 of 5) for reporting. More robust but requires new computation.
- **(c) Report intrinsic-metric selection**: select $\lambda$ to maximize $\Omega$ on $\mathcal{D}_{\mathrm{ref}}$ (an intrinsic metric, no leakage), then show downstream $|\Delta\mathcal{R}|$. Cheap and rigorous.

Recommend **(c)** as primary fix and **(a)** as caveat in text.

---

### M2. Overstated regularizer generalization

**Reviewer claim**: The claim "shape regularizer suffices to reduce downstream forgetting across benchmarks" is overstated; supplementary tables show the trace regularizer occasionally **increases** mean forgetting (Llama BBQ; Qwen TruthfulQA).

**Verification** (Sec 5 contributions, line 188):
> "...and a mild shape regularizer reduces forgetting across downstream benchmarks (\emph{actionability})."

(Plus appendix tables show the cited counter-examples — Llama BBQ and Qwen TruthfulQA, where trace's mean $|\Delta\mathcal{R}|$ is competitive but not always best.)

⚠️ **Partially valid**. Looking at the actual numbers in the compact tables:
- **Llama TruthfulQA**: $|\Delta\mathcal{R}|$ no-reg=0.84, replay=0.76, trace=0.68 ✓ (trace best)
- **Llama BBQ**: $|\Delta\mathcal{R}|$ no-reg=0.179, replay=0.241, trace=0.195 — trace ≈ no-reg, replay worst
- **Qwen TruthfulQA**: no-reg=0.263, replay=0.282, trace=0.270 — trace ≈ no-reg
- **Qwen BBQ**: no-reg=0.112, replay=0.110, trace=0.112 — all near baseline noise floor

The "improvement" is real and large only on Llama TruthfulQA. On the other three, baseline forgetting is small (already at noise floor for Qwen BBQ), so trace doesn't worsen but doesn't help meaningfully.

The **qualitative** claim (trace lifts $\Omega$ where replay doesn't) does hold across all four. But the **quantitative** "reduces forgetting across downstream benchmarks" is overstated.

**Action (proposed)**: Reframe to a more precise version. Rewrite the contribution claim:
> "...and a mild shape regularizer reduces forgetting *where forgetting is measurable* (large gains on Llama TruthfulQA; matched-baseline behavior on tasks already at the noise floor); the qualitative mechanism (trace lifts $\Omega$, replay does not) holds across all evaluated $(\text{model}, \text{FT-task})$ pairs."

Or shorter: "...and a mild shape regularizer reduces forgetting where forgetting is measurable, while consistently lifting backbone shape preservation $\Omega$ above the replay baseline."

---

### M3. Empirical vs. claimed FineWeb-Edu correlation

**Reviewer claim**: The text says FineWeb-Edu retains $|r_s|\approx 0.92$, but tables show 0.15–0.65.

**Verification** (`appendix/quantization_exp.tex`, line 39):
> "...FineWeb-Edu retains $|r_s|\approx 0.92$ while WikiText is weaker ($|r_s|\approx 0.54$)."

Actual FineWeb-Edu $r_s$ across the 7 model variants:
| Model | $r_s$ |
|-------|-------|
| Llama base | 0.17 |
| Mistral base | 0.32 |
| DeepSeek base | 0.24 |
| Qwen base | 0.65 |
| Llama instruct | 0.15 |
| Mistral instruct | 0.41 |
| Qwen instruct | 0.62 |
| **Mean** | **0.36** |

✅ **Confirmed**. The claim of $\approx 0.92$ is **not supported** by any table. Mean is ~0.36; range 0.15–0.65.

**Action (proposed)**: Rewrite the appendix paragraph. The actual story is more interesting:
> "Among MC/QA/reasoning benchmarks the bound consistently achieves strong $|r_s|$. Language-modeling benchmarks (WikiText, FineWeb-Edu) are weaker (mean $|r_s|\approx 0.4$), suggesting that the dense per-token averaging interacts with LM scoring distributions to attenuate the rank signal — likely because these benchmarks have narrower $|\Delta\mathcal{R}|$ spread (variants differ less, so noise dominates)."

This both fixes the false claim and provides a credible mechanism.

---

### M4. Empirical vs. claimed MC/QA range; negative GSM8K correlation

**Reviewer claim**: The text says MC/QA/reasoning benchmarks achieve $|r_s| \in [0.79, 0.95]$, but GSM8K is often below this range and Qwen-instruct GSM8K is **−0.57** (negative correlation).

**Verification**: Actual GSM8K $r_s$:
| Model | $r_s$ |
|-------|-------|
| Llama base | 0.51 |
| Mistral base | 0.48 |
| DeepSeek base | 0.45 |
| Llama instruct | 0.45 |
| Qwen base | 0.68 |
| Qwen instruct | **−0.57** |
| Mistral instruct | 0.83 |
| **Range** | **−0.57 to 0.83** |

✅ **Confirmed**. GSM8K is largely below $0.79$, and Qwen-instruct GSM8K is negatively correlated.

**Action (proposed)**: Two parts.
1. **Fix the range claim**: Replace "$|r_s| \in [0.79, 0.95]$" with a more honest "median $|r_s| \approx 0.8$ across MC/QA, with reasoning benchmarks (GSM8K) systematically weaker (median $\approx 0.5$)".
2. **Acknowledge the negative case**: Add a sentence in the appendix's per-benchmark variability paragraph (already addresses this for Qwen forgetting). For PTQ Qwen-instruct GSM8K, the appendix should note: "the single negative-$r_s$ case (Qwen-instruct GSM8K, $r_s = -0.57$) involves a $|\Delta\mathcal{R}|$ range too small for ranking to be meaningful — the GSM8K FP16 baseline differs from variants by ≤ 0.01 in many cases."

If the FP16-vs-Q8 baseline differences for that cell are indeed at noise floor, the negative correlation is meaningless rather than damaging. We should verify this with the data.

---

### M5. Misleading aggregation of Spearman correlations

**Reviewer claim**: Reporting mean **absolute** Spearman correlation $\overline{|r_s|}$ masks negative-correlation cases (the GSM8K Qwen-instruct example above).

**Verification** (Sec 5.2, line 389):
> "...with mean Spearman $|r_s|=0.831 \pm 0.0722$ over the $2 \times 5$ downstream cells, comparable to the PTQ grid ($|r_s|=0.820 \pm 0.0471$, Fig.~\ref{fig:quant_grid_bound})."

⚠️ **Partially valid**. For the **main 2×5 PTQ grid** (Llama, Qwen on 5 benchmarks), all $r_s$ values are positive (no negative cases in the main figure — Qwen-instruct is in the appendix). So $\overline{|r_s|}$ and $\overline{r_s}$ agree.

For the **appendix replications** (Mistral, DeepSeek, instruct variants), there is at least one negative case (Qwen-instruct GSM8K, −0.57). For those tables, $\overline{|r_s|}$ does mask the sign.

**Action (proposed)**: Two parts.
1. In the main text Sec 5.2, the mean is computed on the main 2×5 grid where all $r_s \ge 0$, so $\overline{|r_s|}$ is unambiguous. **Add a one-sentence clarification**: "Since all per-cell $r_s$ in the main grid are positive, $|r_s|$ and $r_s$ coincide; the appendix discusses isolated negative cases on extended replications."
2. In appendix language-modeling discussion, **report both** $\overline{r_s}$ and $\overline{|r_s|}$ and note where they diverge.

---

### M6. Tables show unscaled $\delta, \gamma$ that don't sum to $\mathcal{B}$

**Reviewer claim**: In Equations 5–6 (delta, gamma), $\delta$ and $\gamma$ include the Lipschitz constants $K_{\mathrm{feat}}$, $K_{\mathrm{pred}}$, so $\delta + \gamma = \mathcal{B}$. But in Tables 1, 9–15, the $\delta$ and $\gamma$ columns do not sum to the displayed $\mathcal{B}$ — they appear to report unscaled geometric residuals.

**Verification**: Looking at Table 1 row Llama MMLU FP16: $\delta = 2.4695$, $\gamma = 0$, $\mathcal{B} = 6.4546$. Indeed $\delta + \gamma = 2.4695 \ne 6.4546 = \mathcal{B}$. The ratio $6.4546 / 2.4695 \approx 2.61$, which would be the implicit $K_{\mathrm{feat}}$.

Looking at the Q6_K MMLU row of Qwen all_bound: $\delta = 0.7540$, $\gamma = 60.1581$, $\mathcal{B} = 87.6841$. Sum $= 60.91$, but $\mathcal{B} = 87.68$. So $\mathcal{B} = K_{\mathrm{feat}} \cdot 0.754 + K_{\mathrm{pred}} \cdot 60.16$. With $K_{\mathrm{pred}} = \sqrt{2}$: $\sqrt{2} \cdot 60.16 = 85.07$. Then $\mathcal{B} - 85.07 = 2.61 \approx K_{\mathrm{feat}} \cdot 0.754$, giving $K_{\mathrm{feat}} \approx 3.46$. (Or some other constant.)

✅ **Confirmed**. The tables show **unscaled geometric residuals** $\sqrt{(\rho_T-\rho_P)^2 + 2\rho_T\rho_P(1-\Omega)}$ and $\|\Sigma_P^{1/2}\Delta H\|_F$, while $\mathcal{B}$ adds the Lipschitz scaling. This is real notational inconsistency.

**Action (proposed)**: Two options:
- **(a) Add table-caption clarification** (cheapest): "Columns $\delta$ and $\gamma$ report the *unscaled* geometric residuals $\sqrt{(\rho_T-\rho_P)^2 + 2\rho_T\rho_P(1-\Omega)}$ and $\|\Sigma_P^{1/2}\Delta H\|_F$; $\mathcal{B} = K_{\mathrm{feat}}\delta + K_{\mathrm{pred}}\gamma$ adds Lipschitz constants $K_{\mathrm{feat}}, K_{\mathrm{pred}}$ defined in Theorem 1."
- **(b) Rename columns** in tables to $\widetilde\delta, \widetilde\gamma$ (with "$\widetilde{}$" denoting unscaled) and add a one-line key.

Recommend **(a)** — minimal change, reader-friendly.

---

## 🟢 Moderate / Minor Issues

### D1. AR risk: per-sequence average ≠ flattened-token mean

**Reviewer claim**: Eq 7 (`eq:ar_risk`) defines per-sequence average loss; line 620 ("stacking all tokens into a single matrix $Z^{\mathrm{AR}}$") computes a uniform mean over all tokens. These weight long sequences differently.

**Verification** (`neurips_2026.tex`, lines 309-316; `appendix/ar_extension.tex`):
✅ **Confirmed mathematically**. The per-sequence average is $\mathbb{E}_c [\frac{1}{|y|}\sum_\tau \ell_\tau]$; the flattened mean is $\frac{1}{N_{\text{total}}} \sum_{\text{all tokens}} \ell$. These coincide only when all sequences have equal length, or when we explicitly normalize per-sequence before pooling.

**Action (proposed)**: Add a one-sentence caveat in Sec 3.4 (AR extension): "Equation (\ref{eq:ar_risk}) uses per-sequence length normalization; when applying Theorem 1 to the stacked matrix $Z_M^{\mathrm{AR}}$, we use per-sequence-normalized aggregation to match Eq.~(\ref{eq:ar_risk}). With variable-length sequences, the unweighted token-level average gives a different (length-weighted) quantity."

### D2. Empirical vs. population risk in Theorem 1

**Reviewer claim**: Eq 2 defines $\mathcal{R}_M$ as a population expectation, but the proof substitutes empirical averages (Eq 20 in App A.3) without flagging the generalization gap.

**Verification**: Eq 2 uses $\mathbb{E}_{(x,y)\sim\mathcal{D}}$; the bound terms $\rho_T, \rho_P, \Omega, \Sigma_P$ are defined as empirical statistics on a calibration sample of size $n$.

⚠️ **Partially valid**. In practice, the bound is evaluated empirically on the same calibration sample where $\mathcal{R}_T - \mathcal{R}_P$ is also evaluated — so it is really an *empirical* bound. The population-vs-empirical distinction is not explicit.

**Action (proposed)**: Either (a) restate Theorem 1 with $\mathcal{R}_M$ = empirical risk on the calibration sample (cleanest, no generalization claim needed); or (b) keep population statement and add an assumption "we treat the empirical statistics as $n$-sample estimates of the population quantities and ignore the $O(1/\sqrt{n})$ generalization gap; tightening this is left to future work."

Recommend **(a)** for honesty; the bound is most useful as an empirical instrument.

### D3. Add $K_{\mathrm{feat}}$ formula to main text

**Reviewer claim**: The intro promises "tight Lipschitz via simplex polarization" but the main text only points to the appendix.

**Verification** (Sec 3.2 line 234):
> "The cross-entropy loss is Lipschitz in features with constant $K_{\mathrm{feat}}$ (simplex polarization, Appendix~\ref{app:kfeat})."

⚠️ **Partially valid**. We deliberately moved the derivation to the appendix to keep the main text tight. But citing only the appendix without giving the value is unsatisfying given the intro's emphasis.

**Action (proposed)**: Add a parenthetical with the value: "$K_{\mathrm{feat}} \le \sqrt{2}\,\sigma_{\max}(H)$ (simplex polarization, Appendix~\ref{app:kfeat})" — assuming this matches the appendix derivation; need to verify exact form.

### D4. Notation: dot in Eq 1 vs no dot in Eq 2

**Verification**:
- Eq 1: `f_M(x) = softmax(phi_M(x) · H_M)` (with `\cdot`)
- Eq 2: `ell(phi_M(x) H_M, y)` (no `\cdot`)

✅ **Confirmed inconsistency**. Trivial fix.

**Action**: Remove the `\cdot` in Eq 1 (`phi_M(x) H_M`) for consistency.

### D5. Eq 2 sum range

**Reviewer claim**: $\sum_j e^{v_j}$ should be $\sum_{j=1}^V$ for clarity.

**Verification**: Yes, range omitted.

✅ **Confirmed minor**. Trivial fix: `\sum_{j=1}^{V}`.

### D6. Sec 6 mitigations parenthetical mismatch

**Reviewer claim**: Sec 6 lists "per-channel outlier smoothing, Hessian-aware reconstruction, FP16-lm_head retention" as mitigations for "scale and head" axes, but Hessian-aware reconstruction (per Sec 5.3) addresses **shape** distortion.

**Verification** (`neurips_2026.tex`, line 445):
> "The shape regularizer of Sec.~\ref{subsec:action} addresses the shape axis directly; protocol-level mitigations for the scale and head axes (per-channel outlier smoothing, Hessian-aware reconstruction, FP16-\texttt{lm\_head} retention) are a research follow-up the diagnostic enables."

(Sec 5.3 line ~404):
> "Each dominant axis suggests a different remediation---per-channel outlier smoothing for scale collapse, Hessian-aware reconstruction for shape distortion, and FP16-\texttt{lm\_head} retention for head divergence."

✅ **Confirmed contradiction**. Hessian-aware reconstruction is explicitly mapped to shape distortion in Sec 5.3 but bundled with scale+head mitigations in Sec 6.

**Action (proposed)**: Rewrite Sec 6 sentence:
> "The shape regularizer of Sec.~\ref{subsec:action} addresses the shape axis directly via training; complementary protocol-level mitigations targeting each axis (per-channel outlier smoothing for scale, Hessian-aware reconstruction for shape, FP16-\texttt{lm\_head} retention for head) are a research follow-up the diagnostic enables."

### T1. "Sections 4 and 4" rendering

**Reviewer claim**: A `\ref{...}` to two paragraph-level labels (`subsec:quantization` and `subsec:forgetting`) inside Sec 4 renders as "Sections 4 and 4".

**Verification** (`appendix/proof_of_the_unified_risk_bound.tex`, line 20):
> "Sections~\ref{subsec:quantization} and~\ref{subsec:forgetting} show that the two arms of this split..."

Both labels are inside `\paragraph{}` blocks under Sec 4, so `\ref` returns "4" for both.

✅ **Confirmed**.

**Action (proposed)**: Two options:
- **(a)** Promote `\paragraph` to `\subsection` — adds visual structure; requires checking page budget.
- **(b)** Rephrase to avoid double-ref: "Section~\ref{sec:applications} (Quantization and LoRA Forgetting paragraphs) shows..."

Recommend **(b)** — cheapest.

### T2. Bibliography casing & author parsing

**Reviewer claim**: Several bib entries render incorrectly:
- "Klimov Oleg" (should be "Oleg Klimov")
- "Gptq" (should be "GPTQ")
- "llama 3" (should be "Llama 3")
- "Svcca" (should be "SVCCA")

**Verification**:
- `paper.bib` line 45: `Oleg, Klimov` — author name parsed wrong (BibTeX expects "Last, First" or "First Last"; this becomes "Klimov O.").
- Title fields are correctly cased in source (`SVCCA: ...`, `GPTQ: ...`, `The Llama 3 ...`), but `plainnat` bibstyle lowercases titles by default unless words are wrapped in `{}`.

✅ **Confirmed**. Two distinct fixes needed:
1. **Line 45 author fix**: `Oleg, Klimov` → `Klimov, Oleg`.
2. **Title brace protection**: Wrap all-caps and proper nouns in `{}`:
   - `title={SVCCA: ...}` → `title={{SVCCA}: ...}`
   - `title={GPTQ: ...}` → `title={{GPTQ}: ...}`
   - `title={The Llama 3 Herd ...}` → `title={The {Llama 3} Herd ...}`

**Action**: Patch the `.bib` file with these brace protections.

### T3–T5. App C.2 minor issues

**Reviewer claims**:
- T3: "symmetric" should be "symmetric positive semi-definite (SPSD)" for the SVD identity claim
- T4: $\Sigma_P$ called "covariance" in some places, "second-moment" in others
- T5: undefined `T/Q` notation in App C.2

**Verification**: These are in `appendix/general_orthogonal_alignments.tex`. T3 and T5 reviewer-cited line numbers don't match exactly to current file (the labels `T/Q` may be from earlier draft), but T4 is real:
- Line 226 in main: "the empirical second-moment matrix $\Sigma_P$"
- Multiple places in appendix and intro discussion call it "covariance weighting"

✅ **T4 confirmed**. T3 and T5 need closer reading of `appendix/general_orthogonal_alignments.tex` to verify; the file as currently written may already have been updated.

**Action**: Standardize on "second-moment matrix" everywhere (most accurate, since features are not mean-centered).

### T6–T7. Missing variants in appendix tables

**Reviewer claims**:
- T6: Table 11 (Qwen3-8B-Base) omits GPTQ
- T7: Table 15 (Qwen3-8B-Instruct) omits Q3_K_M and Q2_K

**Verification**: From `appendix/model_quantization.tex` (Tables 6–8, GPTQ checkpoints):
- Qwen3-8B-Base uses `AlphaGaO/Qwen3-8B-GPTQ` (4-bit, marked $^\ddagger$ as community / non-canonical)
- Qwen3-8B (instruct) uses `JunHowie/Qwen3-8B-GPTQ-Int8` (8-bit)

So GPTQ-Qwen3 entries DO exist in the catalogue. If they are missing from the per-model decomposition tables (Tables 11/15), it's a coverage gap in the actual evaluation runs (the GPTQ variant may have failed loading or been excluded).

**Action**: Audit the source CSV (`paper/exp_result/quantization/quantization_merged_slim.csv`) for Qwen GPTQ rows. If absent, either (a) re-run those configs, or (b) add a footnote to the affected tables: "GPTQ entries for Qwen3-8B are listed in Tables 6–8 but not yet evaluated in this version; included in the next data refresh."

### T8. GPTQ bit-width contradiction

**Reviewer claim**: Sec 5.1 says "GPTQ (4-bit, second-order reconstruction)" but Table 6 lists `JunHowie/Qwen3-8B-GPTQ-Int8` and Table 15 reports a "GPTQ-8bit" variant.

**Verification**:
- Sec 5.1 (line 346): "**GPTQ}**~\cite{frantar2022gptq} (4-bit, second-order reconstruction)"
- `appendix/model_quantization.tex` Table `tab:gptq_awq`: lists various GPTQ checkpoints, including INT8 variants for Qwen.

⚠️ **Partially valid**. We did extend GPTQ to 8-bit for Qwen because no 4-bit checkpoint was available for that family. The Sec 5.1 text understates this.

**Action (proposed)**: Update Sec 5.1: "**GPTQ}** (second-order reconstruction; 4-bit for Llama/Mistral/DeepSeek, 8-bit for Qwen due to checkpoint availability)".

### T9. Unused models in tables

**Reviewer claim**: Tables 6, 7, 8 list "Qwen2.5-7B" and "Qwen2.5-7B-Inst", absent from primary target list.

**Verification**: Need to look at `model_quantization.tex` Tables 6/7/8.

**Action**: Either remove unused rows from the appendix tables, or add a note "Qwen2.5 entries are listed for completeness; not used in the main evaluations."

### T10. Broken cross-reference "Tables ??, 14, and 15"

**Verification**: Searched for literal `Tables ??` in source — not found. The `??` rendering happens when `\ref{}` cannot resolve (stale `.aux` file or missing label). The most likely candidate is the appendix prose at line 36 of `quantization_exp.tex`, which references three instruct tables. The current state (after our recent reference cleanup):
> "Tables~\ref{tab:llama_instruct_decomposition_ext_bound}, \ref{tab:mistral_instruct_decomposition_all_bound}, and~\ref{tab:qwen_instruct_decomposition_all_bound}"

All three labels are now defined in the `\input`'d tables (verified during the reference audit just done). So this issue is **likely already resolved** post-cleanup. A clean re-build will confirm.

**Action**: Verify after `pdflatex` re-run that no `??` appears in the output PDF.

### T11. Table 3 "Wins/10" sums to 11 / 17

**Reviewer claim**: Wins counted with ties; should be noted in caption.

**Verification**: Need to inspect the current `baseline_combined.tex` or the relevant ablation table.

**Action**: Add caption note: "ties counted for each tied metric, so column sums may exceed 10."

### T12. Caption wording "ext task group" / "all task group"

**Reviewer claim**: This is internal-jargon; should be "extended tasks" / "all tasks".

**Verification**: Confirmed — this matches what we discussed in an earlier session (proposal to fix `gen_latex_table.py` caption template). The fix proposal exists but was not finalized.

**Action**: Apply the previously-proposed fix to `gen_latex_table.py` TASK_GROUPS (add `caption_label` field) and regenerate tables.

### T13. NeurIPS Checklist Q5 = TODO

**Verification** (`paper/checklist.tex`): Q5 ("Open access to data and code") has both Answer and Justification as `\answerTODO{}` / `\justificationTODO{}`.

✅ **Confirmed**.

**Action**: Fill in (likely `\answerYes{}` with code-release plan).

---

## Summary of Recommended Actions (in priority order)

### Must-fix before submission
1. **C1**: Fix Fig 1 caption (replace $W^{*}$ with $W = I$).
2. **C2, C3**: Rewrite App C.3 isotropic and frozen-head claims.
3. **C4**: Soften "strictly tighter" claim in App A.4 to "sharper in anisotropic regimes".
4. **C5**: Replace "Cauchy–Schwarz" attribution with correct $\ell_2 \le \ell_1$ justification.
5. **M3, M4**: Correct false correlation claims in appendix (FineWeb-Edu and MC/QA range).
6. **M5**: Add clarifying sentence about $|r_s|$ vs $r_s$ in main grid + report both in appendix.
7. **M6**: Add table-caption clarification on unscaled $\delta, \gamma$ vs $\mathcal{B}$.
8. **D6**: Fix Sec 6 mitigation list (Hessian-aware reconstruction → shape).
9. **T13**: Fill in NeurIPS Checklist Q5.

### Should-fix (improves credibility)
10. **M1**: Address Sec 5.4 hyperparameter selection (option (c) intrinsic-metric selection, or option (a) explicit caveat).
11. **M2**: Reframe regularizer generalization claim to "where forgetting is measurable".
12. **D2**: Restate Theorem 1 over empirical risk (or add explicit assumption).
13. **D1**: Add per-sequence-normalization caveat in App D (AR extension).
14. **T1**: Rephrase "Sections~\ref{subsec:quantization} and~\ref{subsec:forgetting}" in proof appendix.
15. **T2**: Bibliography casing fixes.

### Nice-to-have (clarity / polish)
16. **D3**: Add explicit $K_{\mathrm{feat}}$ formula in main text.
17. **D4, D5**: Eq 1/2 notation consistency.
18. **T3, T4, T5**: SPSD condition; second-moment vs covariance terminology.
19. **T6, T7, T8, T9**: Appendix table coverage and bit-width text.
20. **T10**: Find broken cross-ref.
21. **T11, T12**: Table 3 ties note + caption wording fix.

---

## Notes on the Reviewer's Methodology

The AI reviewer is **mostly accurate** — the math errors (C1–C5) are real and the empirical mismatches (M3–M5) are verifiable from the data. A few claims (e.g., M2 "overstated generalization") are partially valid: the qualitative mechanism does hold, even where the quantitative gain is small.

The reviewer's strength is granular cross-checking of specific numerical claims against tables. Its weakness is occasional over-confidence in math typography that turned out to be correct in spirit but informally stated (T3: "symmetric" instead of "SPSD" is technically loose but mathematically harmless if $Z_T^\top Z_P$ has all-real eigenvalues).
