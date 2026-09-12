# NeurIPS Reviewer Audit — PRISM

**Reviewer profile**: Strict CS/ML researcher, faculty rank. Three-round review.

---

## Round 1: Issue Inventory by Category

### A. Theoretical / Mathematical Issues

**A1. K_feat dependence**
- The Lipschitz constant $K_{\mathrm{feat}} = \max_{j,k}\|h_{T,j} - h_{T,k}\|_2$ is target-only. For variant ranking *within* a fixed target, K_feat is constant (good). But the headline Spearman $r_s = 0.820$ pools variants across multiple targets (Llama, Qwen, Ministral, DeepSeek), each with a different K_feat. Cross-target pooling assumes K_feat scaling preserves rank order — needs explicit justification.

**A2. AR extension residual gap**
- Main Eq. 8 defines $\mathcal{R}^{\mathrm{AR}}$ as per-sequence-mean (with $1/|y|$). Appendix corollary stated for single sequence ($|y| \times d$). Multi-sequence experimental setup uses stacked $Z^{\mathrm{AR}} \in \mathbb{R}^{N \times d}$ (per-token uniform aggregation) — an implicit extension that is left unaddressed. Tight only when sequence lengths uniform.

**A3. Bound tightness not quantified**
- Figures show "below y=x line" (validity), but tight\textsubscript{ness} (gap between $\mathcal{B}$ and $|\Delta\mathcal{R}|$) is not reported. Scatter plots in log-log scale visually obscure tightness ratio.

**A4. Procrustes decomposition novelty**
- Prop. 1 (exact $(\Delta\rho)^2 + 2\rho_T\rho_P(1-\Omega)$ split) is essentially the size-and-shape distance from Williams 2021 generalized shape metrics. Appendix L18 acknowledges this. Reviewer may push back: "what's actually new here beyond CKA/Procrustes literature?" Theory contribution narrows to the *Lipschitz lifting* (K_feat tight constant + bound assembly).

**A5. K_feat computation cost**
- For V ≈ 128K (Llama), $V^2$ pairwise distances = 1.6e10 entries. Computed once per target. Need to verify supplementary code actually does this, or uses approximation.

**A6. K_pred ≤ √2 universal — nontrivial?**
- This is a standard CE Lipschitz bound. Paper presents as a contribution but it's well-known.

### B. Experimental Issues

**B1. Spearman 0.82 — is this even good?**
- No baseline comparison: simple $\|Z_T - Z_P\|_F$ or $1 - \mathrm{CKA}$ might also give $r_s \approx 0.7$–$0.8$ on the same grid. Without this, the "predictiveness" claim is undertested. Most damning: paper criticizes CKA but never reports CKA's Spearman on the same task.

**B2. Single-seed LoRA**
- LoRA forgetting reported on single training trajectory per setting. No error bars on $|\Delta\mathcal{R}|$ values, so the "trace beats replay by 19% vs 9%" claim has no statistical floor.

**B3. Calibration sample sizes**
- 512 (PTQ), 256 (LoRA). Small. For long-tail benchmarks (TriviaQA, GSM8K), this may underrepresent. Reviewer can attack as "noise-floor of evaluation may exceed measured |ΔR| differences".

**B4. λ selection is post-hoc**
- "Each method's sweep $|\Delta\mathcal{R}|$-best" — both trace and replay swept and best chosen. Cherry-picking risk; more honest is to fix λ a priori or to report the full sweep.

**B5. Replay baseline is one-of-many**
- Only compared to in-context-replay-CE on $\mathcal{D}_{\mathrm{ref}}$. EWC, RecAdam, LoRA dropout, MeZO are obvious omissions. "Outperforms experience replay" overstates if only one variant of replay is compared.

**B6. Qwen3 regularizer is null result**
- Appendix shows trace gives no benefit on Qwen3 (mean baseline forgetting too small). Paper reframes this as "regime-dependence" but main-text headline ("trace outperforms replay") is Llama-only. Mild selective reporting.

**B7. GSM8K outlier (r_s = 0.405)**
- Acknowledged but not analyzed deeply. The signal-to-noise explanation is plausible but per-token vs per-sequence aggregation could also play a role (per A2). Reviewer may want both explanations addressed.

**B8. RTX 5090 single-machine**
- Reproducibility narrowed. Different GPU could give slightly different bf16 numerical noise → different |ΔR|.

### C. Logical / Conceptual Issues

**C1. "Independently measurable axes"**
- The bound's three axes are independent in the bound's algebraic structure. But "independently *measurable*" as failure modes requires more: showing that each axis's value correlates uniquely with a specific behavioral failure. Currently only qualitative case studies (Sec 5.3). No factor-loading analysis or controlled experiment.

**C2. Decomposability ≠ causality**
- Saying "shape dominates at Q2" describes the bound's composition, not what *causes* downstream forgetting. The paper occasionally slips toward causal language.

**C3. "Doubles as training-time regularizer"**
- Validated only on frozen-head LoRA. The "doubles" framing implies general applicability; but for full SFT or quantization-aware training, the bound's structure may differ.

**C4. "Vanishing γ for frozen-head LoRA"**
- Requires both H_T = H_P AND W = I. Only the recent M8 fix made this chain explicit. Main text still presents this as a single property of the choice $W=I$.

**C5. Predictiveness vs Decomposability vs Actionability framing**
- Three claims framing is clean but conflates "the bound predicts |ΔR|" with "the decomposition is meaningful." The latter requires more evidence than the former.

### D. Citation / Reference Issues

**D1. None outstanding** (verified all 32 entries; harvey2024what just enriched with PMLR vol.)

### E. Internal Inconsistencies

**E1. Spearman precision inconsistent**
- Abstract: $r_s{=}0.820$; main text: $0.820$, $0.831$; figures: "$\approx 0.82$"; ablation: $0.804$, $0.868$. Mixing 2 vs 3 decimals.

**E2. K_feat on T or P — convention not in main text**
- Decomposition routes through $\mathcal{R}_{P\to T}$ (uses $H_T$). Main text doesn't explain this choice. Just-fixed M2 puts the explanation only in appendix.

### F. Conclusion / Discussion Overreach

**F1. "Three diagnostic axes correspond to distinct empirical failure modes"**
- True for the case studies shown, but the corresponding-to claim is empirical and may not generalize beyond the specific (model, quant scheme, fine-tune task) combinations tested.

**F2. Mean Spearman framing**
- Pooling 0.91 (Llama MMLU) with 0.41 (GSM8K) into a mean of 0.82 hides the GSM8K outlier. Median or per-cell distribution would be more honest.

### G. Reproducibility / Documentation

**G1. Code-paper alignment unstated**
- Supplementary `prism/experiments/` provided. But mapping from code modules to figures/tables not documented.

**G2. λ sensitivity**
- For shape regularizer, no λ-sensitivity curve. Reviewer would expect one.

### H. Misleading or Imprecise Claims

**H1. "Same family of bounds for any W"**
- Vacuous — the bound is informative only when γ is small or zero. For arbitrary W, γ(W) can dominate, making the bound near-trivial.

**H2. "Closed-form" claim for shape regularizer**
- $1 - \Omega$ is differentiable but the SVD inside Procrustes-optimal alignment is not closed-form per-step. Paper uses W=I for differentiability, which is fine but shouldn't be packaged as "closed form" of joint optimum.

**H3. "Substantially looser" naive Cauchy-Schwarz claim**
- Paper says K_feat\textsubscript{naive} = √2 ‖H_T‖_2 is "substantially looser." Quantification needed (how much looser? 5×? 100×? The K_feat empirical table in proof appendix may answer this — needs cross-reference).

### I. Other

**I1. Notation creep**
- Uses $W$, $W_N$, $\Omega$, $\Omega_W$, $\Omega_N$, $\delta$, $\delta_W$, $\gamma$, $\gamma_W$, $\mathcal{B}$, $\mathcal{B}_W$ — several only defined deep in appendix. Main text use should be consistent.

---

## Round 1 Severity Triage

**REJECT-RISK (any one could justify a 4 or below)**:
- B1 (no Spearman baseline against CKA/Frobenius)
- B2 (single seed → no error bar on regularizer claim)
- C1 (decomposition's "independence" not quantitatively validated)
- A1 (cross-target pooling without K_feat normalization argument)

**WEAKEN (would push reviewer score down by 1)**:
- A3 (tightness not quantified)
- A4 (theory contribution narrower than headline)
- B5 (only one replay variant)
- F1, F2 (overclaiming + mean hides outlier)

**MINOR (defensible but should fix)**:
- A2, A6, B3, B4, B6, B7, B8, C2, C3, C4, C5, E1, E2, G1, G2, H1, H2, H3, I1

---

## Round 2: Verification + New Findings

### Issue-by-issue verification

**A1 (K_feat cross-target pooling) — DEMOTED to non-issue.**
Verification: The headline $r_s = 0.820$ is the mean of *per-cell* Spearman across the 2×5 (model, benchmark) grid. Within each cell, K_feat is constant (single target model). Pooling is not on the rank scale, so K_feat differences across cells don't perturb Spearman within. My initial concern was wrong.

**A2 (AR multi-sequence aggregation) — REAL but small.** Confirmed via code (base.py:248: per-sequence mean; extractors.py:201-212: per-token uniform Z). Just-fixed appendix only treats single sequence; multi-sequence implicit. Reviewer ≤5% likely to spot.

**A3 (tightness not quantified) — REAL.** Confirmed: Sec 5.2 only asserts validity (below y=x). No B/|ΔR| ratio reported. Could add one row "median bound/measure ratio = X" without re-running.

**A4 (Procrustes novelty) — REAL.** Appendix L18 explicitly disclaims Procrustes-distance novelty and frames PRISM contribution as the *Lipschitz lifting* + *empirical decomposability*. This is a defensive move but a reviewer may still feel theory is incremental.

**A5 (K_feat compute cost) — confirmed feasible.** Appendix Table tab:lipschitz_constants reports actual K_feat values (Mistral 0.93 → Qwen3 3.46). Computed once per model, V² pairwise distances tractable for V=128K-152K with appropriate batching. Non-issue.

**A6 (K_pred ≤ √2 not novel) — REAL minor.** Standard CE Lipschitz constant. Should be presented as a known fact, not as a contribution.

**B1 (no baselines for Spearman) — STRONG REJECT-RISK.** Verified: ablation Table 4 (`baseline_combined`) compares only Ω vs δ vs B (within-PRISM components). No external baselines. The paper criticizes CKA/SVCCA but never reports their Spearman on the same task. **Reviewer can write: "I want to see CKA, linear-CKA, RSA, simple Frobenius-distance, and ‖Σ_P^{1/2}(W H_T - H_P)‖ alone. If a 1-line baseline gets r_s ≈ 0.78, the bound's value-add is minimal."**

**B2 (single-seed LoRA) — STRONG REJECT-RISK.** Verified: Sec 5.1 hyperparameter list contains no `n_seeds` or `error bar` mention. The shape-regularizer claim "trace -19% vs replay -9%" rests on single-trajectory comparison. Without per-seed variance, the 10pp gap may be within noise.

**B3 (sample sizes 512, 256) — REAL MEDIUM.** Confirmed. Defensible as compute trade-off but small.

**B4 (post-hoc λ selection) — REAL MEDIUM.** Confirmed: "each method's sweep |ΔR|-best." Mitigated slightly by symmetric sweep design (5 values each side, range matched to scale), but cherry-pick risk remains.

**B5 (one replay variant) — REAL MEDIUM.** Confirmed. EWC, MeZO, RecAdam, L2-SP, learning-without-forgetting all absent. Defensible scope but limits the "outperforms" claim.

**B6 (Qwen3 null) — REAL MEDIUM.** Confirmed in `forgetting_qwen.tex` and `regularization_task_dependence.tex`. Main text cites Qwen replication generally but headline numbers are Llama-only. Not strictly selective reporting since appendix discloses fully, but main text could acknowledge the regime-dependence.

**B7 (GSM8K outlier) — REAL.** Confirmed: r_s ≈ 0.41 with mean |ΔR| ≈ 0.019. Already explained as low-SNR in Table tab:gsm8k_outlier. Adequate disclosure, but main-text mean of 0.82 is inflated by averaging in this cell as if it were equally informative.

**B8 (single GPU) — non-issue.** Standard for academic submissions.

**C1 (axes' independence not validated) — STRONG REJECT-RISK.** Verified: Sec 5.3 has only qualitative case studies. No quantitative test like "controlled scale-only perturbation moves only the scale axis" or factor-loading analysis. **Reviewer can attack: "the three-axis decomposition is a property of your bound's algebraic structure, not a discovery about how risk decomposes. Show a controlled experiment where the dominant axis predicts the dominant cause."**

**C2 (decomposition vs causality) — REAL.** Confirmed across multiple sentences ("scale collapse", "shape distortion drives", "head divergence inflates"). Wording slips toward causal where the math only supports correlational.

**C3 ("doubles as regularizer" overclaim) — REAL minor.** Validated only on frozen-head LoRA. The framing implies broader applicability that isn't tested.

**C4 (γ=0 chain) — JUST FIXED via M8.** Defensive; no longer attackable.

**C5 (three-claim framing conflation) — REAL minor.** Predictiveness ≠ Decomposability ≠ Actionability are distinct empirical claims; the paper sometimes uses one to support another.

**E1 (Spearman precision inconsistency) — REAL minor.** Confirmed: 0.82, 0.820, 0.820±0.0471, $\approx 0.82$ all appear. Standardize to 3 decimals + SEM where reported.

**E2 (K_feat=H_T convention) — JUST FIXED via M2 in appendix; main text still implicit.** Could add one sentence to main text Sec 3.2 too.

**F1, F2 (overclaim + mean) — REAL.** Mean Spearman without distribution shape is a real limitation. Adding a violin plot or per-cell distribution figure would address.

**G1 (code-paper alignment unstated) — REAL but typical.** Many NeurIPS papers omit this. Add a 1-page "code-to-figure mapping" table to supplementary README.

**G2 (λ sensitivity) — REAL.** No sensitivity curve. Trace at λ=1.0 is the optimum but no neighborhood reported.

**H1 ("any W" claim) — REAL minor.** True but undermotivated — only $W=I$ and $W=W_N$ are studied.

**H2 ("closed-form" wording) — REAL minor.** $1-\Omega$ at $W=I$ is closed-form; the joint $W_{\mathrm{opt}}$ optimization is not. Wording care needed.

**H3 (naive bound "substantially looser") — VERIFIED.** Appendix Table tab:lipschitz_constants shows K_feat ≈ 1–3.5; naive √2 ‖H‖_2 for Llama H is much larger (operator norm of a 4096×128256 weight matrix is typically O(10) at least). So "substantially looser" is correct; just needs explicit number ("~10× looser" or similar) for credibility.

### NEW issues found in Round 2

**N1. Theorem 1 absolute value issue.**
Re-reading the proof Step 1 (line 33-37): triangle inequality on absolute values gives $|R_T - R_P| \le |R_T - R_{P\to T}| + |R_{P\to T} - R_P|$. Step 2 then bounds the first term (δ) and Step 4 the second (γ). All correct. But notice $\delta$ as defined at line 220 has *no absolute value*: $\delta = K_{\mathrm{feat}}\sqrt{...}$. Is $\delta \ge |R_T - R_{P\to T}|$ or only $\ge R_T - R_{P\to T}$?
   - Verifying via Step 2 (line 81-86): the Lipschitz inequality $|g_T(z_2,y) - g_T(z_1,y)| \le K_{\mathrm{feat}}\|z_2 - z_1\|$ has absolute value on LHS. ✓
   - Then Cauchy-Schwarz step gives $|\mathbb{E}[\ell_T] - \mathbb{E}[\ell_{P\to T}]| \le K_{\mathrm{feat}}\sqrt{\mathbb{E}[\|...\|^2]}$. ✓
   - So $\delta$ correctly bounds $|R_T - R_{P\to T}|$ even without the absolute-value notation on δ itself. **Non-issue**.

**N2. Empirical |ΔR| sign and bound direction.**
The bound $\mathcal{B}$ is symmetric in T and P (uses |·|). But experimentally, the paper plots $|\Delta\mathcal{R}|$ which is also symmetric. Consistent. Non-issue.

**N3. Spearman SEM formula not stated.**
"$0.820 \pm 0.0471$ (SEM)" — standard error of the mean over 10 cells = std/sqrt(10). With 10 samples, this assumes Gaussianity which Spearman per-cell is not. Better to report bootstrap CI. Minor methodology nitpick.

**N4. "Different drift geometries" claim is one of the strongest selling points.**
Sec 5.3 paragraph "Scale-axis separability": "TruthfulQA-FT drives ρ_P > ρ_T on all five benchmarks, while BBQ-FT produces ρ_P < ρ_T on ARC/MMLU." This is a real, interesting finding. But it's based on ONE training run per task. Without n=multiple-seed validation, the "qualitatively different drift geometries" claim is undertested. Same B2 concern compounds here.

**N5. Frozen-head LoRA assumption may not hold for all LoRA practice.**
Sec 5.4 line 168: "LoRA fine-tuning... even when the lm_head stays frozen." But many LoRA recipes (e.g., LoRA+LM-head, Q-LoRA with adapter on head) do not keep the head frozen. The bound's γ=0 claim hinges on this. Should be more upfront in the abstract/intro that the analysis is for frozen-head LoRA specifically.

**N6. Theorem 1's alignment $W$ is data-independent.**
The bound is stated "for any $W \in \mathcal{O}(d)$." Choosing $W = W_N$ requires an SVD computation on $Z_T^\top Z_P$ — a per-checkpoint cost. The paper occasionally hand-waves the cost of $W_N$ as "requires per-step SVD" but doesn't compare to PRISM (W=I) compute cost. Minor.

**N7. Reference set $\mathcal{D}_{\mathrm{ref}}$ size = 32.**
Sec 5.1: "32 pre-training sequences disjoint from $\mathcal{D}_{\mathrm{FT}}$." 32 sequences for computing the regularization target. Tiny. Reviewer concern: regularizer based on 32-sample $\Omega$ may have high variance.

**N8. Calibration set sharing.**
"PRISM and |ΔR| are evaluated on fixed held-out subsets shared across all variants of a base." Same data on both sides → potential coupled-noise concern in r_s estimation (if the data has a particular bias, both sides shift together, inflating correlation). The 512/256 sample size doesn't fully decouple this.

---

## Round 3: Final Verdict + Top-5 Reject-Grounds

### Reviewer score prediction (NeurIPS 1-10 scale)

Without further fixes: **5 (borderline; weak accept lean reject)**.
With Top-5 fixes addressed: **6-7 (weak accept; possibly accept)**.

### Top 5 reject-grounds, ranked by reviewer probability

#### #1 — Missing baselines for Spearman (B1)

**Severity**: Reject-risk. **Probability of being raised**: ~80%.

The single most damning gap. The paper's headline claim is "the bound predicts |ΔR| with $r_s = 0.82$." Without showing that this is meaningfully better than:
- Frobenius distance $\|Z_T - Z_P\|_F$
- linear CKA / RBF CKA
- $\sqrt{(\Delta\rho)^2 + (1-\Omega)}$ (PRISM minus K_feat scaling)

…a reviewer can dismiss the entire predictiveness contribution. Suggested fix: 1-row addition to ablation Table 4 reporting CKA-only and $\|Z_T - Z_P\|_F$-only Spearman on the same 10 cells. Compute is trivial (already have features). This is the SINGLE most impactful fix.

#### #2 — Single-seed regularizer comparison (B2)

**Severity**: Reject-risk. **Probability of being raised**: ~60%.

Trace cuts |ΔR| by 19%, replay by 9%. Without per-seed variance, the 10pp gap could be noise. Suggested fix: re-run trace + replay + no-reg on Llama TruthfulQA with 3 seeds; report mean ± std. Even if the 19% drops to 14% ± 4%, the claim survives. Compute cost: ~3× existing experiment, ~12 hours on RTX 5090.

#### #3 — Decomposability is qualitative-only (C1)

**Severity**: Weakens "framework" contribution. **Probability of being raised**: ~50%.

Sec 5.3 reads: "shape dominates at low-bit, scale separates LoRA tasks, head dominates at GGUF k-quant." All true descriptively. But the paper claims independence of axes and that they "localize qualitatively distinct failure modes." A reviewer wants:
- Controlled perturbation: artificially induce shape-only drift; verify the shape axis dominates the bound while empirical |ΔR| follows.
- Or factor analysis: across all variants, regress |ΔR| onto the three axes and report partial R² for each.

Without this, "decomposability" reads as bound algebra, not empirical claim.

#### #4 — Theoretical novelty narrowly framed (A4)

**Severity**: Weakens "theory" contribution. **Probability of being raised**: ~40%.

Williams 2021 generalized shape metrics already define the size-and-shape distance $d_1$ identical to PRISM's scale+shape decomposition. Park 2023 LRH motivates orthogonal alignment. Schönemann 1966 gives the Procrustes solution. PRISM's actual theory contribution = Lipschitz lifting + tight K_feat. This is incremental; the paper's framing should center on the *empirical bound + diagnostic + regularizer* package rather than positioning as a theoretical breakthrough.

#### #5 — Replay is a one-of-many baseline (B5)

**Severity**: Weakens actionability claim. **Probability of being raised**: ~30%.

"Outperforms experience replay" is one comparison. EWC, MeZO, RecAdam, L2-SP are obvious omissions for a forgetting paper. Defensible scope choice but a reviewer will note. Suggested fix: at least one additional baseline (EWC is cheap to add).

### Other notable concerns (not reject-tier but worth fixing)

- **F2** (mean Spearman hides GSM8K outlier): add per-cell distribution figure or violin plot.
- **A3** (tightness not quantified): add B/|ΔR| ratio summary.
- **N5** (frozen-head LoRA assumption): make this scope explicit in abstract.
- **N7** (32-sample reference set): justify or test sensitivity.

### Strengths the paper should lean into

The strong points that should be preserved/emphasized:

1. **Cross-family empirical breadth**: 7 model families (Llama, Qwen, Ministral, DeepSeek + 3 instruct counterparts) × 5 benchmarks × 3 PTQ schemes. This is rare for a theory-flavored paper. Mention upfront.

2. **Both PTQ + LoRA in one framework**: usually treated as separate problems. Unifying them is a real contribution.

3. **Bound holds empirically**: All variants below y=x line in Fig 2. Theoretical guarantee that ALSO survives empirical test. Stress this.

4. **Shape regularizer is novel**: $1-\Omega$ as a training-time penalty (vs replay/EWC's parameter-space penalties) is a new mechanism. Even if margin shrinks under multi-seed, the *mechanism* claim survives.

### Summary

The paper has a coherent and empirically substantiated thesis. The reject-risk is concentrated in two areas: **(a) missing baseline comparisons that could undercut the predictiveness claim**, and **(b) single-seed reporting that weakens the regularizer comparison**. Both are addressable without re-deriving anything.

The decomposability claim is the third axis of critique: it sells the framework as more than rank correlation, but the supporting evidence is qualitative case studies.

**If submitting as-is**: borderline accept/reject (~5/10).
**With CKA/Frobenius baseline + 3-seed regularizer + per-cell Spearman distribution**: comfortable accept (~7/10).

The minimum credible fix package:
1. Add 2-row CKA + Frobenius baseline to ablation table (1 hour).
2. Run shape regularizer with 3 seeds on Llama TruthfulQA (~12 hours).
3. Add per-cell Spearman distribution plot (1 hour).
4. Add EWC as a third forgetting baseline (~6 hours).

Total cost: ≈ 1 day of compute + 4 hours of writing. Outsized impact on reviewer scores.
