# PRISM Appendix Audit — Final Pre-Submission Review (NeurIPS 2026)

Comprehensive 9-round audit covering the appendix proofs, content verification, and final triage. Generated from a careful read of all nine appendix files plus the main `neurips_2026.tex` for cross-reference.

---

## Round 1: Quick Scan — Issue Inventory by Category

This first pass is broad and shallow: I list every potential issue I notice, categorized but not yet judged for severity. Subsequent rounds will deepen, verify, and triage.

### A. Mathematical Errors (proof steps, formulas, notation)

1. **`proof_of_the_unified_risk_bound.tex`, Step 2 (line ~71):** the inequality chain
   `||∇_z ℓ||_2 ≤ Σ_{j≠y} p_j ||h_{T,j}-h_{T,y}||_2 ≤ max_{j≠y} ||h_{T,j}-h_{T,y}||_2 · (Σ_{j≠y} p_j ≤ 1)`
   produces only `max_{j≠y} ||h_{T,j}-h_{T,y}||_2`, i.e., the maximum of distances of other tokens *to the correct token*. The claim then jumps to `K_feat = max_{j,k} ||h_{T,j}-h_{T,k}||_2` (the full pairwise diameter) by "taking max over all possible true classes y". This is correct — for each y we get max over j of ‖h_j - h_y‖, and taking sup over y yields max over (j,y), which is the full diameter — but the wording could mislead a careful reader into thinking the bound uses a constant strictly larger than what was derived. Worth tightening.

2. **Step 4 (line ~187):** the bound `||p̂||_2 ≤ ||p̂||_1 = 1` is correctly used, but the chain `||p̂ - e_y||_2^2 = ||p̂||_2^2 - 2p̂_y + 1 ≤ 2 - 2p̂_y ≤ 2` requires `||p̂||_2^2 ≤ 1`. That's true (p̂ is a probability vector, sum=1, so by Cauchy: ||p̂||_2² ≤ ||p̂||_1·max p̂_j ≤ 1). Correct, but the inequality uses an additional simplex argument worth spelling out.

3. **Step 4 line ~199, Jensen direction:** `(E[‖z·ΔH‖_2])^2 ≤ E[‖z·ΔH‖_2^2]`. This is the correct Jensen direction for the *concave* function √. (Equivalent to E[X]² ≤ E[X²].) Correct. ✓

4. **Step 3, line ~150 (Procrustes derivation):** the SVD argument uses `|R_ii| ≤ 1` for orthogonal R. This is true (each row of an orthogonal matrix has unit L2 norm, so |R_ii| ≤ 1). The bound `Tr(ΣR) = Σ σ_i R_ii ≤ Σ σ_i` (since σ_i ≥ 0) is achieved at R = I_d, giving W_N = V U^T. But: this requires σ_i ≥ 0 strictly, and the SVD definition has σ_i ≥ 0 by convention. ✓ correct.

5. **`tightness_of_nuclear_norm.tex` line 32:** "i.e., `||x||_2 ≤ ||x||_1` for any non-negative vector". The phrasing implicitly equates the singular-value vector with x. Correct math but could read as a universal claim (||x||_2 ≤ ||x||_1 holds for all real x, not just non-negative; the constraint matters only for the *equality condition*). Minor.

6. **`tightness_of_nuclear_norm.tex` line 42 (CKA discussion):** the inequality `‖A‖_F ≤ Tr(A)` for PSD A is *false in general*. Counter-example: A = diag(1,1) gives ‖A‖_F = √2, Tr(A) = 2 — fine here. But A = diag(0.5,0.5) gives ‖A‖_F = √0.5 ≈ 0.707, Tr(A) = 1 — also fine. Actually for PSD with eigenvalues λ_i ≥ 0: ‖A‖_F = √(Σ λ_i²) and Tr(A) = Σ λ_i. The "‖x‖_2 ≤ ‖x‖_1 on non-negative eigenvalues" cited makes this true. So the claim is correct. ✓ (The wording "since |A‖_F ≤ Tr(A) for any positive semidefinite A, by ‖x‖_2 ≤ ‖x‖_1 on the non-negative eigenvalues" is slightly indirect but correct.)

7. **`general_orthogonal_alignments.tex` line 44:** "Equality holds if and only if W_N = I, i.e., Z_T^T Z_P is symmetric positive semidefinite". Wait — W_N = V U^T = I iff U = V (when both are square). But uniqueness requires that all singular values be distinct, otherwise the SVD is not unique. Equality Ω = Ω_N can hold in *multiple* ways in degenerate cases. Minor issue (technical edge case).

8. **`ar_extension.tex` line 8:** the claim `N = Σ |y|` reduces AR risk to per-row setting "exactly". But `R_M^AR = E_seq[(1/|y|) Σ ℓ]` (Eq. 8 of main paper), which has *per-sequence* normalization 1/|y|. Stacking and applying flat-row mean gives `(1/N) Σ_τ ℓ` which **does not equal** the per-sequence-normalized expectation in general (it's a weighted average where long sequences get more total weight). The corollary as stated is *not* a clean reduction unless all `|y|` are equal, OR unless we redefine the AR risk as the token-level mean. **This is a real concern that needs clarification.**

### B. Logical Gaps (assumptions, claims without justification)

9. **Risk decomposition (Step 1):** the triangle inequality `|a-b| ≤ |a-c| + |c-b|` is invoked on absolute values. ✓ trivially correct.

10. **Step 2 line ~104:** "K_feat ranges from 0.93 to 3.46". The values come from Table `tab:lipschitz_constants`. Need to verify this table exists and matches.

11. **`ar_extension.tex` claim that "The Lipschitz analysis... and Procrustes decomposition depend only on the matrix shape, not on the provenance of its rows":** True for the *bound's algebraic form*, but the per-sequence vs per-token normalization distinction (issue #8) is glossed over. The proof says "Applying Theorem 1 to (Z_T^AR, Z_P^AR) yields the stated inequality" — but this gives a bound on the *token-mean* CE gap, which equals the per-sequence-mean CE gap only under length-uniformity. **Real issue.**

12. **`general_orthogonal_alignments.tex` Sec. on joint optimization (line 80):** "δ(W) is concave in Tr(Z_T^T Z_P W)". Concavity of √(·) is concavity in the argument; here the argument is `(ρ_T-ρ_P)² + 2ρ_T ρ_P (1 - Ω_W)` which is *affine* in `Tr(Z_T^T Z_P W)`. So `δ(W) = K · √(affine in Tr)` is concave in `Tr(Z_T^T Z_P W)`, yes — but Tr is itself linear in W, so δ is concave-composed-with-linear in W, meaning δ is concave in W only over the *affine hull*, not over O(d) (which is non-convex). The concavity argument is therefore loose. The conclusion ("sum is neither convex nor concave on O(d)") is correct — O(d) is non-convex so neither term is convex/concave on it directly — but the reasoning sketch is shaky.

13. **`general_orthogonal_alignments.tex` line ~92:** "We adopt W = I throughout the LoRA experiments... it gives the cleanest decomposition — γ = 0 from the head simplification". Note: under LoRA the head is *literally identical* (H_T = H_P), so γ = 0 holds at W = I trivially. But the remark "γ = 0 from the head simplification" should clarify "*because* H_T = H_P" not "*from* W = I" — at W = I, γ = ‖Σ_P^{1/2}(H_T - H_P)‖_F = 0 only if H_T = H_P. Could be clearer.

### C. Internal Inconsistencies (between sections)

14. **`proof_of_the_unified_risk_bound.tex`, line 13:** "Sec. 4.4 (subsec:ablation) ablation" is referred to. Confirm `subsec:ablation` exists in main text — yes, at line 391.

15. **Reference `subsec:shape_reg`:** referenced in proof and general_orthogonal_alignments. In main text it exists at line 266. ✓

16. **`forgetting_qwen.tex` line 1 comment says "Input by neurips_2025.tex"** — but the file is `neurips_2026.tex`. **Minor (comment-only).**

17. **`forgetting_qwen.tex` line 7:** "vocabulary size (151,936 vs. 128,256), tokenizer family (byte-level BPE vs. BPE)". Both Llama-3.1 and Qwen3 use byte-level BPE tokenizers. The "byte-level BPE vs. BPE" distinction is **factually questionable** — needs verification.

18. **Numerical claim alignment:** main text says mean Spearman PTQ = 0.820 (line 153, abstract; line 338). `quantization_exp.tex` line 36 says per-benchmark mean Spearman in `[0.77, 0.89]` and GSM8K weakest at ≈0.41. These are *per-benchmark* averaged across families; the 0.820 is averaged across the 2×5 grid. Consistent.

19. **`forgetting_qwen.tex` line 22:** "Llama TruthfulQA on TriviaQA: |ΔR|=4.63". Need to verify against table `tab:reg_compare_llama_truthfulqa`.

20. **`regularization_task_dependence.tex` line 4:** "BBQ-FT TriviaQA |ΔR| drops from 0.143 to 0.017 (-88%)". (0.143 - 0.017)/0.143 = 0.881 ≈ 88%. ✓
    "BBQ-FT GSM8K from 0.061 to 0.013 (-79%)". (0.061-0.013)/0.061 = 0.787 ≈ 79%. ✓
    "TruthfulQA-FT SQuAD from 1.337 to 1.054 (-21%)". (1.337-1.054)/1.337 = 0.212 ≈ 21%. ✓
    "TriviaQA from 2.583 to 2.124 (-18%)". (2.583-2.124)/2.583 = 0.178 ≈ 18%. ✓
    Numbers internally consistent; need cross-check against tables.

21. **`regularization_task_dependence.tex` line 4:** "TruthfulQA-FT, where shape drift dominates downstream forgetting, illustrates ... trace cuts mean |ΔR| from 0.843 (no reg) to 0.681 (-19%)". Main text line 388 says "0.84 to 0.68 (-19%)". Consistent (rounded). ✓

22. **`regularization_task_dependence.tex` line 4:** "averaging the per-benchmark |ΔR| values from Table tab:reg_compare_llama_bbq gives means of 0.179 (no reg), 0.195 (trace), and 0.241 (replay)". Need to verify these match the table.

23. **`regularization_task_dependence.tex` line 40:** "Qwen TruthfulQA-FT mean |ΔR|=0.263 ... vs. Llama's 0.843 (3.2× smaller)". 0.843 / 0.263 = 3.21. ✓

### D. Cross-Section Consistency (appendix vs main)

24. **Notation Ω vs Ω_W vs Ω_N:** main paper uses Ω as shorthand for Ω_{W=I} (e.g., line 245: "shape mismatch 1−Ω_W"; line 252: "trace form Ω := Ω_{W=I}"; main bound at line 231 uses Ω_W generally). Appendix proof uses Ω, Ω_W, Ω_N consistently. ✓ broadly.
    
    **Potential ambiguity:** line 197 of main uses "trace Procrustes similarity Ω_W" (Eq. 5 / `eq:omega_def`); but line 252 introduces Ω := Ω_{W=I} as the trace form. The main paper's `eq:omega_def` defines Ω_W but then the body refers to Ω. The naming "trace Procrustes similarity" for Ω_W (general) and "trace form Ω" for the W=I case may confuse readers.

25. **K_pred = √2 universally:** stated in main (line 224), in proof appendix (line 187), and in `proof` line 104 ("K_pred = √2 holds universally"). Consistent. ✓

26. **K_feat values:** main mentions only "K_feat = max pairwise embedding distance"; numeric values are in `proof` line 104 (Table `tab:lipschitz_constants`). Need to verify that table exists.

### E. Notation/Terminology Drift

27. **K_feat vs K_{feat}:** subscript style is consistent throughout. ✓

28. **Σ_P vs Σ_p:** consistent. ✓

29. **B vs 𝒷 vs ℬ:** main uses `\mathcal{B}`. Appendix uses `\mathcal{B}` consistently. ✓

30. **subsec:exp_setup:** referenced in `quantization_exp.tex` line 10; exists in main at line 302. ✓

31. **subsec:ar_extension** vs **app:ar_extension:** main uses `subsec:ar_extension` at line 263; appendix file labels itself `app:ar_extension`. Both labels exist; the cross-references appear consistent.

### F. Missing References / Unclear Citations

32. **`general_orthogonal_alignments.tex` cites `gower2004procrustes` and `schonemann1966generalized`** (line 77, line 146 of proof). Need to verify they're in the bibliography.

33. **`ethayarajh2019contextual`** (line 83 of general_orthogonal_alignments). Verify bibliography.

34. **`harvey2024what`** (cited in line 18 of proof). Verify.

35. **`williams2021generalized`** — same.

36. **`huh2024platonic`** in future_work line 8. Verify.

### G. Verbose Sections That Could Tighten

37. **`proof_of_the_unified_risk_bound.tex`, Step 2 "Lipschitz Property" paragraph (line 44)** is a long meta-explanation of why we use direct gradient analysis vs composition rule. Useful but could halve in length.

38. **`general_orthogonal_alignments.tex` Sec. on joint optimization (lines 64–95):** ~30 lines of "no closed form, also not relevant in our regimes". Could compress.

39. **`regularization_task_dependence.tex`** is dense with many qualifications; some paragraphs read defensively.

### H. Misleading or Imprecise Claims

40. **`proof_of_the_unified_risk_bound.tex` Remark (line 79):** "A naive Cauchy–Schwarz bound gives K_feat^naive = √2 ‖H_T‖_2, which is substantially looser." Should clarify: this is the bound from `‖∇_z ℓ‖_2 ≤ ‖H_T‖_2 · ‖p̂ - e_y‖_2 ≤ √2 ‖H_T‖_2`, valid; the *substantial looseness* claim depends on the spectral norm being much larger than the pairwise embedding diameter. For LLMs this is empirically true (table on line 106) but the statement needs the empirical anchor.

41. **`tightness_of_nuclear_norm.tex` line 5:** "We define the nuclear-form and Frobenius-form Procrustes similarities..." — Ω_F is *not* derived from any Procrustes problem; calling it "Frobenius-form Procrustes similarity" is misleading. The text on line 14 partly clarifies ("unlike Ω_W, it does not arise from the alignment residual at any W"), but the section/paragraph naming should be consistent. Minor.

42. **`general_orthogonal_alignments.tex` line 5:** "$W = I$ ... vanishing exactly under frozen lm_head". This conflates "frozen head" (LoRA case, H_T = H_P exactly) with a more general guarantee. At W = I, γ = K_pred · ‖Σ_P^{1/2}(H_T - H_P)‖_F vanishes only if H_T = H_P. For PTQ with FP16 head, also H_T = H_P and γ = 0 at W = I. So the claim is correct but the wording "vanishing exactly under frozen lm_head" is imprecise — it should say "vanishes exactly when H_T = H_P (frozen-head LoRA, FP16-head PTQ)". Already done elsewhere; recheck.

43. **`forgetting_qwen.tex` line 22:** "When the forgetting magnitude itself is below per-evaluation noise, per-checkpoint ranking becomes uninformative". OK as a defensive caveat. The cited |ΔR| of 0.0023 for Qwen BBQ TriviaQA at λ=0 vs Llama TruthfulQA TriviaQA at 4.63 is a 2000× ratio, not "three orders of magnitude" exactly (4.63/0.0023 ≈ 2013, which is 3.3 orders of magnitude — "more than three" is technically right). Minor.

### I. Other

44. **`proof_of_the_unified_risk_bound.tex` Sec. "Relation to Classical Procrustes Shape Metrics" (line 15):** This subsection is *before* the actual proof Steps 1–5. It's a good context-setter but its placement (right after the section intro, before any proof step) is unusual for a proof appendix. May want to move to end or as a Remark.

45. **`forgetting_qwen.tex` references `tab:gsm8k_outlier`** (line 36 of `quantization_exp.tex`); `regularization_task_dependence.tex` line 40 also refers to it. Need to verify table exists and the Spearman 0.41 claim matches.

46. **`model_quantization.tex` Table tab:gptq_awq line 94:** "JunHowie/Qwen3-8B-GPTQ-Int8" — the only public 8-bit GPTQ checkpoint, used for Qwen3-8B-Instruct (model #4). This is mentioned in caption (line 83). Consistent. ✓

47. **`model_quantization.tex` line 39:** "Models 2, 5 (meta-llama) are gated and require accepting the Meta Community License". Models 2 and 5 are Llama-3.1-8B and Llama-3.1-8B-Instruct. Note: there is also model 7 (DeepSeek-R1-Distill-Llama-8B) which derives from Llama but is typically not under Meta gating directly. ✓ as stated.

---

Total issues identified in Round 1: **~47 items.** Many are minor; key ones are #1, #6, #8, #11, #12, #17, #41 for math/logic, plus many cross-references to verify in Round 4.

---

## Round 2: Deep Dive on Proofs (Step-by-step Verification)

I now go step-by-step through each proof in the appendix, checking algebra, assumption use, and conclusion validity.

### 2.1 `proof_of_the_unified_risk_bound.tex` — Theorem 1 Proof

#### Step 1 — Risk Decomposition (lines 21–37): ✓ correct

- Bridge `R_{P→T} := E[ℓ(φ_P(x) W · H_T, y)]` introduced cleanly.
- Triangle inequality `|R_T - R_P| ≤ |R_T - R_{P→T}| + |R_{P→T} - R_P|` is a one-line tautology on absolute values: `|a-b| = |(a-c)+(c-b)| ≤ |a-c|+|c-b|`. ✓
- δ and γ are defined as the absolute values of these terms. ✓

**Concern (minor):** the wording "$\delta$ measures how features differ through the same head $H_T$" is intuitive but the formal definition is `|R_T - R_{P→T}|`, an absolute risk gap, not a "feature difference". OK as informal gloss.

#### Step 2 — Lipschitz analysis (lines 40–104): ✓ correct, with one wording tightening

The proof structure:
1. Define `g_T(z, y) := ℓ(z·H_T, y)`. (line 44)
2. Compute ∇_z g_T = H_T(p̂ - e_y). (line 56)
3. Rewrite using simplex constraint: `∇_z ℓ = Σ_{j≠y} p̂_j (h_{T,j} - h_{T,y})`. (line 67)
4. Triangle inequality: `||∇_z ℓ||_2 ≤ Σ_{j≠y} p̂_j ||h_{T,j}-h_{T,y}||_2`. (line 71)
5. Bound: `≤ max_{j≠y} ||h_{T,j}-h_{T,y}||_2 · Σ p̂_j ≤ max_{j≠y} ||h_{T,j}-h_{T,y}||_2 · 1`. (line 71)
6. Taking max over true class y: `K_feat = max_{j,k} ||h_{T,j} - h_{T,k}||_2`. (line 76)

**Verification:** Step 5 gives `||∇_z ℓ||_2 ≤ max_{j≠y_*} ||h_{T,j}-h_{T,y_*}||_2` for the *specific* true class y. Step 6 says: since y can be any true class, the worst-case over input (z, y) is `sup_y max_{j≠y} = max_{j,y, j≠y}` which equals `max_{j,k, j≠k} ||h_{T,j}-h_{T,k}||_2 = max_{j,k} ||h_{T,j}-h_{T,k}||_2` (the j=k case gives 0, so including it doesn't change the max). ✓ correct.

**Mean-value step (lines 81–86):** ∇z continuously differentiable, so for any z_1, z_2,
`g_T(z_2,y) - g_T(z_1,y) = ∫₀¹ ∇g(z_1+t(z_2-z_1),y)^T (z_2-z_1) dt`.
Cauchy-Schwarz inside integral, then uniform bound on ∇g gives K_feat·||z_2-z_1||_2. ✓ correct.

**Application (lines 88–96):** 
δ = |R_T - R_{P→T}| = |E[g_T(φ_T(x), y) - g_T(φ_P(x)W, y)]| ≤ E[|...|] ≤ K_feat · E[||φ_T(x) - φ_P(x)W||_2]. ✓

**Jensen's step (line 98):** E[||X||] ≤ √(E[||X||²]). This is correct (concave √ + Jensen on E[||X||²] gives E[||X||] ≤ √E[||X||²]). ✓
Then connects to `||Z_T - Z_P W||_F²/n` via row-wise interpretation. ✓

**Tightening suggestion (cosmetic):** line 71 "max_{j≠y}" then line 76 "max_{j,k}" — explicitly note that taking max over y∈{1,...,V} of max_{j≠y} equals max_{j,k} because (a) every (j,k) with j≠k arises for some y=k, and (b) j=k contributes 0. This step is silently understood but a one-line justification would help.

#### Step 3 — Procrustes Decomposition (lines 109–171): ✓ correct

**Key identity:** For any W ∈ O(d):
`||Z_T - Z_P W||_F² = ||Z_T||_F² + ||Z_P W||_F² - 2 Tr(Z_T^T Z_P W)`
`= ||Z_T||_F² + ||Z_P||_F² - 2 Tr(Z_T^T Z_P W)`. 
The orthogonality reduction `||Z_P W||_F² = Tr(W^T Z_P^T Z_P W) = Tr(Z_P^T Z_P W W^T) = Tr(Z_P^T Z_P) = ||Z_P||_F²` uses cyclic trace and W W^T = I (because W ∈ O(d) means W^T W = I = W W^T). ✓

The algebra "Adding and subtracting 2ρ_T ρ_P completes the square":
`ρ_T² + ρ_P² - 2ρ_T ρ_P Ω_W = (ρ_T - ρ_P)² + 2ρ_T ρ_P (1 - Ω_W)`.
Expanding: `ρ_T² + ρ_P² - 2ρ_T ρ_P + 2ρ_T ρ_P - 2ρ_T ρ_P Ω_W = (ρ_T - ρ_P)² + 2ρ_T ρ_P(1 - Ω_W)`. ✓

**Procrustes optimum (Specialization 2):** The SVD argument `Tr(Σ R) = Σ σ_i R_ii ≤ Σ σ_i` since σ_i ≥ 0 and |R_ii| ≤ 1 for orthogonal R. R = I_d achieves equality. So W_N = V U^T. ✓

**Subtle point:** `|R_ii| ≤ 1` requires that each diagonal entry of an orthogonal matrix has magnitude ≤ 1. Proof: row i of R is unit-norm in L2, so `R_ii² ≤ Σ_j R_ij² = 1`. ✓ (well-known but unstated; OK).

#### Step 4 — Head Bound (lines 175–215): ✓ correct

1. ∇_v ℓ = p̂ - e_y. ✓
2. ||p̂ - e_y||² = ||p̂||² - 2 p̂_y + 1.
3. ||p̂||² ≤ ||p̂||₁² = 1 (since p̂ is in simplex; ||p̂||² ≤ ||p̂||·||p̂||_∞ ≤ 1·1 = 1, or directly by Cauchy: (Σ p̂_j)² ≥ Σ p̂_j² when p̂_j ∈ [0,1] and Σ p̂_j = 1). ✓
4. So ||p̂ - e_y||² ≤ 2 - 2 p̂_y ≤ 2 → ||∇_v ℓ||₂ ≤ √2. ✓
5. K_pred = √2 (sup attained as p̂_y → 0). ✓

**Covariance projection (line 199):** Jensen + trace expansion + Σ_P = E[z^T z]. ✓
- E[||z·ΔH||²] = E[Tr(ΔH^T z^T z ΔH)] = Tr(ΔH^T E[z^T z] ΔH) = Tr(ΔH^T Σ_P ΔH) = ||Σ_P^{1/2} ΔH||_F². ✓
- This last step uses Tr(A^T B A) = ||B^{1/2} A||_F² for PSD B, which holds since Σ_P^{1/2} is well-defined PSD. ✓

#### Step 5 — Assembly (lines 218–230): ✓ correct

Adds δ + γ; both bounds derived for the same W. Final:
`|R_T - R_P| ≤ K_feat √((ρ_T-ρ_P)² + 2ρ_T ρ_P (1-Ω_W)) + K_pred ||Σ_P^{1/2}(W H_T - H_P)||_F`. ✓

**Specializations:** W=I (trivial substitution), W=W_N (Procrustes optimum). Both correctly derived. ✓

### 2.2 `tightness_of_nuclear_norm.tex` — Bound Tightness

**Statement:** Ω_F ≤ Ω_N, with equality iff Z_T^T Z_P has rank ≤ 1.

**Proof verification:** For singular values σ_1, ..., σ_d ≥ 0 of Z_T^T Z_P:
- ||Z_T^T Z_P||_F = √(Σ σ_i²)
- ||Z_T^T Z_P||_* = Σ σ_i

**Inequality:** `√(Σ σ_i²) ≤ Σ σ_i` ⟺ `Σ σ_i² ≤ (Σ σ_i)²` ⟺ `Σ σ_i² ≤ Σ σ_i² + 2 Σ_{i<j} σ_i σ_j` ⟺ `0 ≤ 2 Σ_{i<j} σ_i σ_j`. True since σ_i ≥ 0. ✓

**Equality condition:** Equality iff cross terms `Σ_{i<j} σ_i σ_j = 0` ⟺ at most one σ_i > 0 ⟺ rank(Z_T^T Z_P) ≤ 1. ✓

**Wording issue (minor):** line 32 says "i.e., $\|x\|_2 \le \|x\|_1$ for any non-negative vector". The L2-L1 inequality `||x||_2 ≤ ||x||_1` holds for *all* real vectors x (because squared L2 norm is ≤ squared L1 norm: `Σ x_i² ≤ Σ x_i² + 2 Σ_{i<j} |x_i x_j|`). The non-negativity is needed only to drop the absolute values in identifying `Σ σ_i² ≤ (Σ σ_i)²`. Minor — could just say "by ||x||₂ ≤ ||x||₁ on x = (σ_1,...,σ_d)".

**CKA relation (line 42):** CKA(Z_T, Z_P) = ||Z_T^T Z_P||_F² / (||Z_T^T Z_T||_F · ||Z_P^T Z_P||_F).

The text claims `||Z_M^T Z_M||_F ≤ ||Z_M||_F² = Tr(Z_M^T Z_M)`. Verify:
- ||Z_M^T Z_M||_F = √(Σ λ_i²) where λ_i are eigenvalues of Z_M^T Z_M (= squared singular values of Z_M).
- Tr(Z_M^T Z_M) = Σ λ_i = ||Z_M||_F².
- So `√(Σ λ_i²) ≤ Σ λ_i` ⟺ ||x||₂ ≤ ||x||₁ on the non-negative eigenvalues. ✓

So `||Z_T^T Z_T||_F · ||Z_P^T Z_P||_F ≤ ||Z_T||_F² · ||Z_P||_F²`, giving `CKA ≥ Ω_F²` (denominator smaller → ratio larger). ✓

### 2.3 `general_orthogonal_alignments.tex` — Alternative W

**Sec. on W=I specialization (lines 29–61):** 

Claim Ω ≤ Ω_N: trivially Tr(Z_T^T Z_P · I) ≤ max_W Tr(Z_T^T Z_P W) = ||Z_T^T Z_P||_*. ✓

Equality iff W_N = I iff Z_T^T Z_P SPSD. **Slight issue:** equality iff `arg max Tr(Z_T^T Z_P W) = I` (or some other equivalent condition). In the case of degenerate singular values, the arg max is not unique, so "W_N = I" should be "I ∈ argmax" — formally only matters if there are repeated σ_i. **Minor edge case.**

The "approximately SPSD" → "Ω ≈ Ω_N" claim (line 46) is informal but reasonable. ✓

**Sec. on Joint Optimization (lines 64–95):** 

Claim "δ(W) is concave in Tr(Z_T^T Z_P W)": let `t = Tr(Z_T^T Z_P W)`. Then `δ(W) = K · √(c_1 - 2 c_2 t)` for some constants c_1 > 0, c_2 = ρ_T ρ_P / (||Z_T||_F ||Z_P||_F) · ... wait, let me re-examine.

Actually `Ω_W = Tr(Z_T^T Z_P W)/(||Z_T||_F ||Z_P||_F)` so `1 - Ω_W` is affine in `t = Tr(Z_T^T Z_P W)`. The argument under the sqrt is `(ρ_T - ρ_P)² + 2 ρ_T ρ_P (1 - Ω_W)`, which is linear (affine) in t — a *decreasing* affine function of t. So δ = K √(linear-decreasing in t), which is *concave* in t (since √ is concave and composing with affine preserves concavity). ✓

But the next claim "γ(W) is convex in W" is only loosely true: γ = K' · ||Σ_P^{1/2}(WH_T - H_P)||_F is the Frobenius norm of an affine function of W, which is a convex function of W (in vec(W)). ✓

Their sum is then concave + convex, which is *neither* convex nor concave in general. But neither is the constraint set O(d) convex. The conclusion "no closed form" stands; the reasoning is loose-but-OK.

**Sec. on Practical Irrelevance (lines 88–93):**

For LoRA, H_T = H_P (frozen head). At W = I, γ = ||Σ_P^{1/2}(I·H_T - H_P)||_F = ||Σ_P^{1/2} · 0||_F = 0. ✓ But at W ≠ I, γ = ||Σ_P^{1/2}(W - I) H_T||_F could be positive. So W = I is locally optimal for γ (γ vanishes only at W = I when H_T = H_P, generically). ✓

For PTQ with FP16 head: H_T ≈ H_P (FP16 head is shared between target and proxy). The text says "H_T ≈ H_P", so γ ≈ 0 and W_N suffices. ✓

### 2.4 `ar_extension.tex` — Autoregressive Extension

**The proof claim:** Stacking all token-level features into Z_M^AR ∈ R^{N × d} reduces the AR risk gap to the per-row setting.

**The math doesn't quite work:** Recall:
`R_M^AR = E_{(c,y)~D}[(1/|y|) Σ_{τ=1}^{|y|} ℓ(φ_M(c, y_<τ) H_M, y_τ)]`

This is an expectation over sequences (with per-sequence weight 1/|y|). If we stack all N = Σ_{(c,y)} |y| tokens and apply Theorem 1 to (Z_T^AR, Z_P^AR), the bound is in terms of `(1/N) Σ_τ ℓ(...)`, i.e., the *token-mean* CE.

**Mismatch:** The token-mean and the sequence-mean (with 1/|y| internal normalization) are *not* the same in general. For the empirical estimator:
- Token-mean: `(1/N) Σ_{(c,y) ∈ D} Σ_{τ=1}^{|y|} ℓ_τ`. Long sequences contribute more loss terms.
- Sequence-mean: `(1/|D|) Σ_{(c,y)} (1/|y|) Σ_τ ℓ_τ`. Each sequence weighted equally.

These coincide only when all `|y|` are equal (or in expectation over a population where length is independent of loss).

**Nature of the issue:**
- Either redefine `R_M^AR` to be token-mean (drop the 1/|y|), OR
- Re-derive Theorem 1 with per-row weights w_i = 1/(|y_i| · |D|) instead of uniform 1/N. The proof structure goes through with weighted RMS norms / weighted Procrustes (which is fine), but the bound formula no longer reads as `||Z_T - Z_P||_F`/√n; it becomes `√Σ w_i ||z_T,i - z_P,i||²`.

Looking at the main paper line 263: "with per-sequence length normalization to match Eq. (8); Theorem 1 then applies directly". This suggests they want to honor the 1/|y| weighting. But the appendix proof glosses this — it just says "stacking the token-level features into a matrix Z_M^AR ∈ R^{N×d} therefore reduces the autoregressive risk gap to exactly the per-row setting". **This handwaves the normalization mismatch.**

**Real issue.** Either:
- (a) State that practical implementation uses token-mean and that this is what `R_M^AR` actually is in the experimental table — making Eq. 8 in main text a bit informal, OR
- (b) Reformulate the corollary to use weighted Frobenius norms `||·||_{F,w}` with weights w_τ = 1/(|y_τ| · n_seq) and adjust the bound. The proof then goes through with weighted versions of all norms.

In practice, since `(1/n) ||Z_T - Z_P||_F²` becomes `Σ w_τ ||row_τ||²`, the Lipschitz step still gives `δ ≤ K_feat √(Σ w_τ ||row_τ||²)`, and the geometric decomposition holds with weighted RMS scales `ρ̃_M² = Σ w_τ ||z_M,τ||²` and weighted similarity. **The cleanest fix is option (b).**

**Recommended action:** Either rewrite the corollary with explicit weighting OR explicitly state that all empirical AR risks reported in this paper are per-token rather than per-sequence (and update Eq. 8). Worth checking what the actual evaluation code uses.


---

## Round 3: Reviewer Attack Vectors per Appendix File

For each of the 9 files, I list 2–5 specific objections a reviewer could raise. These are intentionally adversarial and may be defensive overkill in some cases — Round 7 will triage.

### 3.1 `proof_of_the_unified_risk_bound.tex`

1. **"Sloppy max-over-y argument" (Step 2):** A reviewer could say the jump from `max_{j≠y}` to `max_{j,k}` (line 71 → line 76) is not justified. Defense: it is correct (max over y of max over j≠y equals max over (j,k) with j≠k), but the proof should say so in one line.

2. **"K_feat depends on the head, but the bound is for variant comparison" (Step 2):** A reviewer might argue that K_feat = max pairwise embedding distance scales with `|H_T|` and may not be a constant across variants. Defense: K_feat depends only on the *target* model H_T (not the proxy), so for fixed target it's a constant; proxy comparisons are always relative to the same H_T. Worth a one-sentence remark.

3. **"Bound presupposes K_feat is finite" (Step 2):** Trivially yes for finite-vocabulary discrete embeddings; not an actual concern but a pedantic reviewer might raise it.

4. **"Procrustes residual decomposition ignores translations" (Step 3):** PRISM uses `||Z_T - Z_P W||_F` without centering. Some Procrustes definitions center first. Defense: omitting centering is intentional — captures absolute residual including any mean shift. Worth a remark.

5. **"Specialization 2 footnote is buried" (Step 3 line 163):** "Note that W_N minimizes only the feature alignment residual δ(W), not the full bound δ(W) + γ(W)" — this caveat is critical and should be in the *main statement* of the corollary, not a trailing footnote.

### 3.2 `tightness_of_nuclear_norm.tex`

1. **"Why is this a contribution?" (general framing):** A reviewer could ask: ‖x‖₂ ≤ ‖x‖₁ is a 19th-century inequality. The contribution should frame this as "lifting an elementary inequality to a tighter risk bound" rather than restating the Frobenius/nuclear inequality.

2. **"Equality condition (rank ≤ 1) is degenerate" (line 21):** For `Z_T^T Z_P` to be rank ≤ 1, we'd need Z_T (or Z_P) itself to be rank ≤ 1, which never happens for LLM features in practice. So this equality condition is academic. A reviewer might say "in practice Ω_F < Ω_N always".

3. **"Why is CKA discussed here?" (line 41):** CKA discussion feels grafted on. It belongs more in the Related Work or in `general_orthogonal_alignments.tex`. Internal organization issue.

4. **"CKA inequality uses unstated PSD assumption" (line 42):** The inequality `‖Z_M^T Z_M‖_F ≤ ‖Z_M‖_F²` uses that Z_M^T Z_M is PSD with non-negative eigenvalues. Proof sketch on line 42 is correct but tight; a careful reader might want it expanded.

### 3.3 `general_orthogonal_alignments.tex`

1. **"W=I is justified post-hoc" (whole appendix):** A reviewer could read the W=I choice as convenient rather than principled. The appendix already addresses this via "frozen-head LoRA, FP16-head PTQ" but a skeptical reviewer might still ask: why not just use W_N everywhere if it gives tighter Spearman? Defense (in main text Sec. 3.3 / ablation Sec. 4.5): regularizer differentiability, no per-step SVD, etc.

2. **"Joint optimization argument is hand-wavy" (lines 79–86):** The convexity/concavity argument for "no closed form" is sketchy. A rigorous reviewer might want a clean reference to Stiefel manifold optimization theory or a precise nonconvexity statement.

3. **"Isotropic special case is a strawman" (line 82–83):** The text says "isotropic features... unrealistic for LLM representations". Then why discuss this case at all? A reviewer might think the section is padding.

4. **"What's the actual joint-optimum gain?" (line 86):** Section 4.5 ablation reports W_N gives r_s = 0.912 vs W=I gives 0.820. But that's not the *joint optimum* — it's the W_N minimizer of δ alone. The joint W_opt could give even higher Spearman, or it could stay at 0.912 because γ is small in tested regimes. **Real gap:** no number for the joint optimum is reported.

### 3.4 `ar_extension.tex`

1. **"Per-sequence vs per-token normalization mismatch" (line 8) — CRITICAL:** as flagged in Round 2, the corollary's stacking argument gives a token-mean bound, not the per-sequence-mean bound from Eq. 8. A careful reviewer will catch this.

2. **"Independence assumption hidden":** line 8 says "Each target token y_τ contributes an independent feature-loss pair". But under teacher forcing, the *features* φ_M(c, y_<τ) are not independent across τ (they share context); only the *loss conditioned on features* is decoupled. The "independent" wording is loose.

3. **"AR proof is one-paragraph; can it really cover GSM8K's 100-token chain-of-thought?":** The corollary technically applies, but the variance of the bound across sequences with very different lengths could be large. No discussion of effective sample size or sequence-length conditioning.

4. **"GSM8K results contradict the AR claim":** In quantization, GSM8K Spearman is the lowest at ~0.41. The AR appendix doesn't connect to this empirical finding. A reviewer might ask: "if your AR extension is so clean, why does GSM8K underperform?"

### 3.5 `quantization_exp.tex`

1. **"Per-benchmark Spearman range is wide" (line 36):** [0.77, 0.89] is a 0.12-point range. A skeptical reviewer might ask whether the mean Spearman 0.82 is mostly driven by certain benchmarks.

2. **"GSM8K outlier is hand-waved" (line 36):** "small `|ΔR|` → noise" is plausible but not quantified. A reviewer might want a specific noise-floor estimate (e.g., "per-sample CE std on GSM8K is X, so |ΔR| < X is uninformative").

3. **"Feature-only scatter section is brief" (line 22):** The aggregate `r̄_s(δ)` is cited but not shown as a number in this appendix. Need to confirm the main text has it.

4. **"Replication on Ministral and DeepSeek" — no per-benchmark numbers shown:** Just a Figure. The reader has to flip to per-model tables to verify.

### 3.6 `forgetting_qwen.tex`

1. **"Numerical discrepancy" (line 22) — REAL:** "Qwen3-8B BBQ on TriviaQA: |ΔR|=0.0023 at λ=0" but Table `tab:reg_compare_qwen_bbq` (TriviaQA no reg) shows 0.0035. Similarly "Llama TruthfulQA on TriviaQA: |ΔR|=4.63" but Table `tab:reg_compare_llama_truthfulqa` shows 2.5829 at step 300. **These numbers are inconsistent with the comparison tables.** Source likely a different step or evaluation; need to clarify or correct.

2. **"Tokenizer family claim is wrong" (line 7) — likely:** "byte-level BPE vs. BPE" — both Llama-3.1 and Qwen3 use byte-level BPE. Should be "different tokenizers (vocabulary 151K vs 128K, multilingual vs English-dominant)" rather than asserting different tokenizer families.

3. **"Qwen forgetting is too small to demonstrate the regularizer" — already self-defended, but defensive:** the appendix acknowledges this. A reviewer might still say "why include null result?" Defense already in place: documents regime dependence.

4. **"File comment says neurips_2025.tex" (line 1):** Cosmetic/maintenance error; current file is neurips_2026.tex.

### 3.7 `regularization_task_dependence.tex`

1. **"Table tab:reg_gating cherry-picks 4 settings" (line 24):** Only 2 models × 2 fine-tuning tasks = 4 rows. A reviewer might want more (more models, more FT tasks). Defense: scope chosen to match Sec. 4.4.

2. **"Llama BBQ exception is rationalized post-hoc" (lines 31, 37):** The "+8.6%" outcome on Llama BBQ is a failure of the gating; the appendix explains it as "cell-level mixed". A reviewer might want a more principled prediction (e.g., a quantitative threshold for condition (i) / (ii)).

3. **"Gating signal is binary but presented as continuous":** "noise floor" is qualitative. No concrete cutoff for `1-Ω̄`. Reviewer: "where is the cut?"

4. **"Per-cell adaptive deployment is left to future work" — punting:** The appendix acknowledges adaptive per-cell deployment is the natural extension. A reviewer may say "then your contribution is incomplete".

### 3.8 `model_quantization.tex`

1. **"Models 2 and 5 are gated" disclaimer (line 39):** Honest but possibly an issue for reproducibility. A reviewer might ask for non-gated alternatives.

2. **"GPTQ coverage is uneven" (Table tab:gptq_awq):** Only 4/7 models have a GPTQ checkpoint; the rest are "---". A reviewer could ask whether this biases the analysis. Defense: the available 4 give consistent patterns; non-coverage doesn't favor or hurt the bound.

3. **"GPTQ bit-width inconsistency" (line 83):** Qwen3-8B-Instruct uses 8-bit GPTQ, all others 4-bit. A reviewer might ask whether this matters for the bound's evaluation. Defense: PRISM's prediction adapts to bit-width via the actual decompositions, so this is fine.

4. **"GGUF k-quant scheme details missing":** The table lists tags (Q4_K_M etc.) but gives no algorithm summary. A reviewer unfamiliar with k-quants might want a one-paragraph description.

### 3.9 `future_work.tex`

1. **"Beyond LLMs paragraph" (line 8):** The claim that ViTs and CLIP have "tight feature scales" via final-layer LayerNorm/L2 norm is correct but glosses over real differences (e.g., ViT has [CLS] token features rather than per-token). A reviewer may want more nuance.

2. **"Diagnostic applications" (line 7):** Three applications listed (OOD, hyperparameter transfer, drift monitoring), no preliminary results. A reviewer may say "speculation".

3. **"PRISM extends naturally" — not actually shown:** All three points are claims, not demonstrated extensions. Defense: Future Work section by definition.


---

## Round 4: Numerical Verification

I now check key numerical claims in the appendix against the source tables, computing where useful.

### 4.1 Lipschitz Constants (Table tab:lipschitz_constants)

`proof_of_the_unified_risk_bound.tex` line 104 claims K_feat ranges from 0.93 (Mistral) to 3.46 (Qwen3-Base). Table data:

| Model | K_feat | K_pred |
|---|---|---|
| Llama-3.1-8B | 2.61 | √2 |
| Ministral-3-8B | 0.98 | √2 |
| Qwen3-8B | 3.46 | √2 |
| DeepSeek-R1-8B | 2.60 | √2 |
| Llama-3.1-8B-Instruct | 2.60 | √2 |
| Ministral-3-8B-Instruct | 0.93 | √2 |
| Qwen3-8B-Instruct | 3.41 | √2 |

Range: 0.93 (Ministral-Instruct) to 3.46 (Qwen3-Base). ✓ matches text exactly.

### 4.2 Llama Q2_K MMLU Decomposition

Main paper: "ρ_P exceeds ρ_T only modestly (Δρ ≈ 4.9, scale ≈ 24) yet Ω drops to 0.78, driving shape (≈ 9,000) to dominate."

Verification (ρ_T=138.96, ρ_P=143.86, Ω=0.7750, K_feat=2.61):
- |Δρ| = 4.90 ✓
- (Δρ)² = 24.01 ≈ 24 ✓
- shape = 2 ρ_T ρ_P (1 - Ω) = 2 × 138.96 × 143.86 × 0.225 = **8,995.85** ≈ 9,000 ✓
- δ = K_feat √(scale + shape) = 2.61 × √(24.01 + 8995.85) = 2.61 × 94.99 = **247.93**
- Table reports δ = 248.22 — match within rounding (the 0.29 discrepancy comes from using 2-decimal Ω; full precision would close it). ✓

### 4.3 Llama TruthfulQA-FT TriviaQA No-Reg Decomposition (main text Sec. 4.3 paragraph)

Main paper: "Llama TruthfulQA-FT on TriviaQA (no-reg baseline): Ω drops to 0.76 with shape (≈ 10,000) outweighing scale (≈ 35) by ~280×".

From `tab:reg_compare_llama_truthfulqa`: ρ_T=140.93, ρ_P=146.85, Ω=0.7593:
- |Δρ| = 5.92, (Δρ)² = 35.05 ≈ 35 ✓
- shape = 2 × 140.93 × 146.85 × 0.2407 = **9,962.85** ≈ 10,000 ✓
- ratio = shape / scale = **284.3** ≈ 280 ✓ (text says "~280×")
- δ_predicted = 2.61 × √(35.05 + 9962.85) = 2.61 × √9997.9 = **260.97**
- Table δ = 261.36 — match within rounding ✓

### 4.4 Qwen3-Base Q6_K SQuAD Decomposition (head divergence example)

Main paper Sec. 4.3: "$(Δρ)² ≈ 0.12$, $Ω ≈ 1$, so $δ = 1.18$, yet $γ = 75.77$ — making $γ$ essentially the entire bound (Q8_0 shows the same pattern: $δ = 0.74$, $γ = 23.96$)."

Verification (from tab:qwen_decomposition_all_bound, SQuAD block):
- Q6_K: ρ_T=298.04, ρ_P=298.38, Ω=1.0000, δ=1.1798, γ=75.7671 ✓ (rounded matches)
  - (Δρ)² = (0.34)² = 0.1156 ≈ 0.12 ✓
- Q8_0: ρ_T=298.04, ρ_P=298.25, δ=0.7419, γ=23.9590 ✓
- B = 76.95 ≈ "76.95" cited in main paper ✓
- BnB INT8 SQuAD: δ=3.81, γ=0, B=3.81 ✓ ("20× reduction from Q6_K's B=76.95" → 76.95/3.81 ≈ 20.2 ✓)

### 4.5 GSM8K Outlier Table (tab:gsm8k_outlier)

`quantization_exp.tex` line 36: "per-benchmark mean Spearman lies in [0.77, 0.89]; GSM8K weakest ($r_s ≈ 0.41$); mean |ΔR| ≈ 0.019; other benchmarks 0.07–0.16. For Qwen3-8B-Instruct in particular the mean |ΔR| on GSM8K is only 0.0033."

Verification (from tab:gsm8k_outlier):
- Mean r_s by benchmark: ARC=0.768, MMLU=0.793, SQuAD=0.801, TriviaQA=0.889, GSM8K=0.405. 
  - Range [0.768, 0.889] vs claimed [0.77, 0.89]. ✓ matches at 2-dp rounding.
- Mean |ΔR|: ARC=0.0731, MMLU=0.0742, SQuAD=0.0852, TriviaQA=0.1628, GSM8K=0.0187. 
  - Other benchmarks 0.07–0.16 ✓; GSM8K 0.019 ✓
- Qwen3-8B-Instruct GSM8K = 0.0033 ✓

### 4.6 Llama TruthfulQA Regularization Means (main text + appendix)

From tab:reg_compact_llama_truthfulqa caption: "$Ω$ = 0.906 / 0.915 / 0.931; $|ΔR|$ = 0.843 / 0.764 / 0.681" for [no reg / replay / trace].

Verification (from tab:reg_compare_llama_truthfulqa, |ΔR| column):
- no reg: [0.0290, 0.1339, 1.3372, 2.5829, 0.1342] → mean = 0.84344 ≈ 0.843 ✓
- replay: [0.0108, 0.1031, 1.1911, 2.3879, 0.1261] → mean = 0.7638 ≈ 0.764 ✓
- trace: [0.0199, 0.1116, 1.0543, 2.1242, 0.0966] → mean = 0.68132 ≈ 0.681 ✓

Ω means:
- no reg: [0.9158, 0.9426, 0.9231, 0.7593, 0.9905] → mean = 0.90626 ≈ 0.906 ✓
- replay: [0.9185, 0.9451, 0.9345, 0.7810, 0.9957] → mean = 0.91496 ≈ 0.915 ✓
- trace: [0.9319, 0.9542, 0.9478, 0.8210, 1.0000] → mean = 0.93098 ≈ 0.931 ✓

All Ω and |ΔR| means match.

### 4.7 Llama BBQ Regularization Means (regularization_task_dependence.tex line 4)

Claim: "0.179 (no reg), 0.195 (trace), and 0.241 (replay)".

Verification (tab:reg_compare_llama_bbq, |ΔR| column):
- no reg: [0.0775, 0.3531, 0.2608, 0.1435, 0.0609] → mean = 0.17916 ≈ 0.179 ✓
- replay: [0.1799, 0.6061, 0.2127, 0.1506, 0.0581] → mean = 0.24148 ≈ 0.241 ✓
- trace: [0.1536, 0.4995, 0.2903, 0.0171, 0.0126] → mean = 0.19462 ≈ 0.195 ✓

### 4.8 Qwen Regularization Means (regularization_task_dependence.tex line 40)

Claim: "across-benchmark mean |ΔR| at λ=0 is 0.263 on Qwen TruthfulQA-FT (vs Llama 0.843)" (3.2× smaller); "0.112 on Qwen BBQ-FT vs Llama's 0.179". 

Verification:
- Qwen TruthfulQA: [0.1612, 0.1092, 0.8757, 0.1552, 0.0118] → mean = 0.26262 ≈ 0.263 ✓
- Qwen BBQ: [0.2881, 0.1159, 0.1353, 0.0035, 0.0196] → mean = 0.11248 ≈ 0.112 ✓
- 0.843 / 0.263 = 3.21 → "3.2× smaller" ✓

Replay/trace means on Qwen TruthfulQA:
- replay: 0.28228, trace: 0.26978 → "0.263 / 0.282 / 0.270" ✓ all match
On Qwen BBQ:
- replay: 0.11008, trace: 0.11222 → "0.112 / 0.110 / 0.112" ✓ all match

Spread: max-min ≈ 0.020 (TruthfulQA) and 0.002 (BBQ), well "under 0.02 on both" ✓.

### 4.9 Diagnostic Gating Table (tab:reg_gating)

| Setting | 1−Ω̄ | Δ|ΔR|/|ΔR|_0 |
|---|---|---|
| Llama TruthfulQA | 0.0937 | -19.2% |
| Llama BBQ | 0.0678 | +8.6% |
| Qwen TruthfulQA | 0.0091 | +2.7% |
| Qwen BBQ | 0.0011 | -0.2% |

All four 1-Ω̄ verified exactly: 0.09374, 0.06782, 0.00912, 0.00112 ✓.
All four relative changes verified: -19.22%, +8.63%, +2.73%, -0.23% ✓.

### 4.10 The 4.63 / 0.0023 Numbers in `forgetting_qwen.tex` Line 22 — REAL ISSUE

The text claims:
- "Qwen3-8B BBQ on TriviaQA: |ΔR| = 0.0023 at λ=0"
- "Llama TruthfulQA on TriviaQA: |ΔR| = 4.63"
- "three orders of magnitude apart"

But the comparison tables show different values:
- `tab:reg_compare_qwen_bbq` TriviaQA no reg: **0.0035** (not 0.0023)
- `tab:reg_compare_llama_truthfulqa` TriviaQA no reg: **2.5829** (not 4.63)

Searching the trace-norm sweep tables, I find:
- `tab:trace_norm_llama_truthfulqa` TriviaQA λ=0.0: |ΔR| = **4.6261** ≈ 4.63 ✓
- The 0.0023 doesn't appear in `tab:trace_norm_qwen_bbq` either; need to check if it's from a forgetting trajectory checkpoint (not the comparison/sweep step-300 numbers).

**Inconsistency:** the `forgetting_qwen.tex` paragraph cites numbers from a *third* experimental configuration (likely the per-step LoRA forgetting trajectory shown in Fig.~\ref{fig:forget_grid_qwen}, not the regularization comparison tables). This is potentially confusing because the same nominal "no reg / λ=0" configuration appears with three different |ΔR| values across the paper:

1. Comparison table (`tab:reg_compare_*`) uses LoRA fine-tuned with NO regularizer, evaluated at step 300 with `\lambda=0` (replay/trace both off): TriviaQA = 2.5829 (Llama TruthfulQA), 0.0035 (Qwen BBQ).
2. Trace-norm sweep table (`tab:trace_norm_*`) uses identical `\lambda=0` setting: TriviaQA = 4.6261 (Llama TruthfulQA).
3. Forgetting trajectory (Fig. ref:forget_grid_*) has per-step values; possibly the maximum across steps.

If (1) and (2) describe the same config, why 2.58 vs 4.63? Likely different LoRA hyperparameters (rank, lr) or different reference set. **The numbers in `forgetting_qwen.tex` line 22 should match either (1) or (3) — probably (3) — and the source should be cited.**

**Recommended fix:** either:
- (a) change "λ=0" to "max checkpoint step" or "evaluation at step X" with explicit context; OR
- (b) replace the cited numbers with values from `tab:reg_compare_qwen_bbq` (0.0035) and `tab:reg_compare_llama_truthfulqa` (2.58) so they match the comparison setting; OR
- (c) explain the difference between the two configurations explicitly.

### 4.11 GSM8K Outlier "0.019" — `quantization_exp.tex` line 36

Claim: "shrinking mean |ΔR| to ≈ 0.019 — an order of magnitude below other benchmarks (0.07–0.16)"

From `tab:gsm8k_outlier`: GSM8K mean = 0.0187 → 0.019 ✓; other ARC=0.073, MMLU=0.074, SQuAD=0.085, TriviaQA=0.163. So range [0.073, 0.163] ≈ "0.07–0.16" ✓. "Order of magnitude below" is loose: 0.073 / 0.019 = 3.8× (less than 1 OOM), 0.163 / 0.019 = 8.6× (close to 1 OOM). The phrase is slightly hyperbolic; "5-9× below" would be more accurate. Minor wording issue.


---

## Round 5: Cross-Section Consistency Check

I now systematically check that notation, theorem statements, equation labels, and numbers are consistent between the appendix and the main paper.

### 5.1 Theorem 1 Statement (Main vs Appendix)

**Main paper** (line 226–234):
```
For any W ∈ O(d),
|R_T - R_P| ≤ B := K_feat √((ρ_T - ρ_P)² + 2ρ_T ρ_P (1 - Ω_W))  (δ)
              + K_pred · ‖Σ_P^{1/2} (W H_T - H_P)‖_F             (γ)
```

**Appendix Step 5** (line 222–224):
```
|R_T - R_P| ≤ δ + γ ≤ K_feat √((ρ_T - ρ_P)² + 2ρ_T ρ_P (1 - Ω_W))
              + K_pred · ‖Σ_P^{1/2} (W H_T - H_P)‖_F
```

**Identical** notation, identical formula. ✓

### 5.2 Notation Conventions

| Symbol | Meaning | Used consistently? |
|---|---|---|
| Ω | Trace form Ω_{W=I} (main text default) | ✓ |
| Ω_W | General trace form for any W ∈ O(d) | ✓ |
| Ω_N | Nuclear form Ω_{W=W_N} = ‖Z_T^T Z_P‖_*/(‖Z_T‖_F ‖Z_P‖_F) | ✓ |
| Ω_F | Frobenius form (external reference object) | ✓ (only in tightness appendix) |
| δ | Feature alignment error K_feat √(...) | ✓ |
| γ | Head discrepancy K_pred ‖Σ_P^{1/2} ΔH‖_F | ✓ |
| B | Total bound = δ + γ | ✓ |
| K_feat | max pairwise embedding distance in H_T | ✓ |
| K_pred | √2 universal | ✓ |
| Σ_P | E[z^T z] = Z_P^T Z_P / n empirical covariance | ✓ |
| W | orthogonal alignment, W ∈ O(d) | ✓ |
| W_N | Procrustes optimum = V U^T from SVD of Z_T^T Z_P | ✓ |
| H_T, H_P | Target/proxy head | ✓ |
| Z_T, Z_P | n × d feature matrices | ✓ |
| ρ_M | RMS scale = ‖Z_M‖_F / √n | ✓ |
| φ_M | Backbone X → R^d | ✓ |
| ℓ | Cross-entropy loss (Eq. 1) | ✓ |

**Naming consistency issue (minor):**
- Main paper line 211 (Eq. 5) defines Ω_W as "trace Procrustes similarity" — this is the *general* form.
- Main paper line 252 introduces Ω := Ω_{W=I} as "trace form Ω".
- Appendix proof line 13 says "We provide the complete proof... for general W ∈ O(d) (trace family Ω_W)" — uses "trace family" terminology.

The term "trace Procrustes similarity" in the main definition is potentially confusing because it implies Procrustes already, while the Procrustes-optimal alignment is W_N (giving Ω_N), not Ω_W. The "trace" qualifier helps but is subtle. Could rename to "trace alignment similarity" or just "trace similarity" to avoid implying that Ω_W is itself a Procrustes optimum.

### 5.3 Equation/Section Cross-References

| Reference in appendix | Target in main | Exists? |
|---|---|---|
| `subsec:ablation` | Sec. 4.5 (line 391) | ✓ |
| `subsec:shape_reg` | Sec. 3.5 (line 266) | ✓ |
| `subsec:experiments` (sec:experiments) | line 297 | ✓ |
| `subsec:exp_setup` | line 302 | ✓ |
| `subsec:forget_exp` (= subsec:predict?) | line 322–325 (`subsec:predict` AND `subsec:forget_exp` both label this) | ✓ |
| `subsec:shape_reg_exp` (= subsec:action) | line 372 | ✓ |
| `subsec:quant_exp` | line 322–325 | ✓ |
| `subsec:decompose` | line 354 | ✓ |
| `subsec:forgetting` | line 291 (in sec:applications) | ✓ |
| `subsec:quantization` | line 287 | ✓ |
| `subsec:ar_extension` | line 255 | ✓ |
| `subsec:problem_setup` | line 180 | ✓ |
| `subsec:unified_bound` | line 191 | ✓ |
| `subsec:interpretation` | line 237 | ✓ |
| `eq:omega_def` | line 212 | ✓ |
| `eq:exact_equality` | line 203 | ✓ (Proposition 1) |
| `eq:omega_nuclear` | tightness app eq 2 | ✓ |
| `eq:lora_bound` | line 271 | ✓ |
| `eq:shape_reg` | line 277 | ✓ |
| `eq:ar_risk` | line 261 | ✓ |
| `eq:ce_def` | line 185 | ✓ |
| `eq:unified_bound` | line 230 | ✓ |
| `prop:exact_decomposition` | line 199 | ✓ |
| `thm:unified_bound` | line 226 | ✓ |
| `tab:lipschitz_constants` | tables file | ✓ |
| `tab:gsm8k_outlier` | tables file | ✓ |
| `tab:llama_decomposition_main_bound` | tables file | ✓ |
| `tab:reg_compare_*` (4 tables) | tables files | ✓ |
| `tab:reg_compact_llama_truthfulqa` | tables file | ✓ |
| `tab:reg_gating` | inline in regularization_task_dependence.tex | ✓ |
| `tab:gptq_awq` | model_quantization.tex | ✓ |
| `tab:coverage` | model_quantization.tex | ✓ |
| `app:detailed_proofs` | proof file | ✓ |
| `app:tightness` | tightness file | ✓ |
| `app:joint_opt` | general_orthogonal_alignments.tex | ✓ |
| `app:trace_specialization` | general_orthogonal_alignments.tex | ✓ |
| `app:kfeat` | proof file | ✓ |
| `app:procrustes` | proof file | ✓ |
| `app:head_bound` | proof file | ✓ |
| `app:assembly` | proof file | ✓ |
| `app:ar_extension` | ar_extension.tex | ✓ |
| `app:per_model_tables` | quantization_exp.tex | ✓ |
| `app:qwen_forgetting` | forgetting_qwen.tex | ✓ |
| `app:reg_task_dependence` | regularization_task_dependence.tex | ✓ |
| `app:future_work` | future_work.tex | ✓ |
| `app:quant_tables` | quantization_exp.tex | ✓ |
| `app:replication_mistral_deepseek` | quantization_exp.tex | ✓ |
| `app:feature_only` | quantization_exp.tex | ✓ |
| `app:risk_decomp` | proof file | ✓ |
| `app:shape_metrics_relation` | proof file | ✓ |
| `app:general_W` | general_orthogonal_alignments.tex | ✓ |
| `app:general_bound` | general_orthogonal_alignments.tex | ✓ |
| `app:model_details` | model_quantization.tex | ✓ |
| `app:gptq_awq` | model_quantization.tex | ✓ |
| `app:gguf_repos` | model_quantization.tex | ✓ |
| `app:coverage` | model_quantization.tex | ✓ |
| `app:quant_tiers` | model_quantization.tex | ✓ |
| `app:trace_norm_tables` | forgetting_qwen.tex | ✓ |
| `app:qwen_forget_grid` | forgetting_qwen.tex | ✓ |
| `app:qwen_shape_reg` | forgetting_qwen.tex | ✓ |
| `cor:ar_bound` | ar_extension.tex | ✓ |

All cross-references match. 

### 5.4 Number Consistency (key Spearman & |ΔR| values)

| Claim | Source | Verified? |
|---|---|---|
| PTQ mean Spearman 0.820 (over 2×5 grid) | Main abstract, Sec. 4.2 | Cited as 0.820 ± 0.0471 |
| LoRA forgetting mean Spearman 0.831 | Main Sec. 4.2, abstract | Cited as 0.831 ± 0.0722 |
| Llama TruthfulQA r_s = 0.958 (mean across 5 benchmarks) | Main line 351 | not in tables; need to confirm |
| Llama MMLU specifically r_s = 0.91 | Main line 340 | tab:llama_decomposition_main_bound MMLU header r_s = 0.91 ✓ |
| Per-benchmark mean Spearman [0.77, 0.89] | quantization_exp.tex | tab:gsm8k_outlier: [0.77, 0.89] ✓ |
| GSM8K Spearman ≈ 0.41 | both | tab:gsm8k_outlier: 0.405 ✓ |
| Llama TruthfulQA mean |ΔR| = 0.84 | Main line 388 | 0.843 ✓ |
| Trace -19% reduction | Main + appendix | -19.2% ✓ |
| Llama BBQ baseline mean Ω = 0.932 | reg_task_dep | 0.932 ✓ |
| Llama TruthfulQA baseline mean Ω = 0.906 | reg_task_dep | 0.906 ✓ |
| Llama BBQ trace lifts Ω from 0.93 to 0.98 (73% reduction in 1-Ω) | reg_task_dep line 7 | 0.93 → 0.98? Let me verify: BBQ trace Ω values [0.9762, 0.9758, 0.9694, 0.9955, 1.0000] → mean 0.9834 ≈ 0.98. baseline 0.932. (1-0.932)/(1-0.98) = 0.068/0.02 = 3.4×; reduction in (1-Ω) = (0.068-0.02)/0.068 = 70.6%. Text claims 73%. **Off by 2-3 percentage points (probably uses unrounded values).** Let me recompute with full precision: trace mean Ω = 0.98338. 1-0.98338 = 0.01662. baseline 1-0.93218 = 0.06782. (0.06782-0.01662)/0.06782 = 0.7549 = 75.5%. So 73% is within 3pp of the unrounded calculation. Rounding matters; the figure could be ±5% depending on precision used. ✓ (close enough). |

### 5.5 W=I vs W=W_N Discussion

The main text and appendix both use W=I as default and W=W_N as ablation, with the same justification: differentiability, no SVD, frozen-head/FP16-head simplification. Three places repeat this:
1. Main Sec. 3.3 line 252
2. Main Sec. 4.5 line 396
3. Appendix `app:general_W` line 5

The repetition is mostly consistent but verbose. The three statements differ in emphasis:
- Main 252: "primarily design-driven"
- Main 396: "$W=I$ trade ~0.09 Spearman for SVD-free differentiability"
- Appendix 5: "different priorities: W_N minimizes feature residual but generally inflates the head term"

The appendix mention "W_N generally inflates head term" is the most informative; the main paper could cite it more explicitly.

### 5.6 Key Inconsistency Found

**The 4.63 / 0.0023 numbers in `forgetting_qwen.tex` line 22** do not match the comparison table (`tab:reg_compare_*`). They appear to come from the trace-norm sweep table or the LoRA trajectory figure. This needs to be either:
- Cited with the correct source table/figure, OR
- Replaced with consistent step-300 numbers from the comparison tables (2.58 / 0.0035).

This is the most concrete cross-section inconsistency identified.

### 5.7 Tokenizer Family Claim

`forgetting_qwen.tex` line 7: "tokenizer family (byte-level BPE vs. BPE)".

Both Llama-3 (post-Llama-2) and Qwen3 use byte-level BPE (BBPE). Llama-3 added a 128K-vocab BBPE tokenizer; Qwen3 uses BBPE with 151K vocab. So the "byte-level BPE vs. BPE" distinction is incorrect — both are byte-level. The accurate distinction is **vocabulary size** (128K vs 151K) and possibly **multilingual coverage** (Qwen3 trained on much more multilingual data). 

**Recommended fix:** rephrase to "vocabulary size (151,936 vs. 128,256 tokens) and pre-training corpus (multilingual, ∼36T tokens vs. English-dominant, ∼15T tokens); both use byte-level BPE tokenization but with substantially different vocabularies".


---

## Round 6: Severity Triage of Identified Issues

I now revisit each Round 1–3 issue, decide whether it's a real problem, and assign severity.

### CRITICAL (affects validity of mathematical claims or correctness)

- *(none identified)* — the proofs are mathematically sound; no claim is wrong.

### HIGH (visible flaw a careful reviewer would notice)

**H1. AR Extension normalization mismatch** (Round 1 #8, #11; Round 2 §2.4; Round 3 §3.4 #1)
- **Issue:** `ar_extension.tex` says stacking tokens into Z^AR ∈ R^{N×d} and applying Theorem 1 "yields the stated inequality" for `R_M^AR` defined with per-sequence 1/|y| weighting. But the per-row Theorem 1 produces a *token-mean* CE bound, not the per-sequence-mean one. The two coincide only under length-uniformity.
- **Severity:** HIGH. A theory-oriented reviewer will catch this.
- **Fix:** Either (a) replace Eq. 8 main paper with token-mean definition `R_M^AR = (1/N) E[Σ_τ ℓ_τ]`, or (b) reformulate the corollary with weighted Frobenius / weighted RMS scales (w_τ = 1/(|y_τ|·|D|)). Option (a) is simpler and matches typical implementation. Suggest a 1-paragraph addition to ar_extension.tex disambiguating this.

**H2. Numerical inconsistency in `forgetting_qwen.tex` line 22** (Round 1 #19; Round 4 §4.10)
- **Issue:** "Qwen3-8B BBQ on TriviaQA: |ΔR| = 0.0023 at λ=0" and "Llama TruthfulQA on TriviaQA: |ΔR| = 4.63" don't match the comparison tables (which give 0.0035 and 2.58 respectively). The 4.63 matches the trace-norm sweep table at λ=0; the 0.0023 source is unclear (possibly a forgetting trajectory checkpoint).
- **Severity:** HIGH. A reviewer who cross-checks tables will flag it.
- **Fix:** Either (a) update to numbers consistent with the comparison tables (0.0035 and 2.58) and adjust "three orders of magnitude" → "≈730×", or (b) cite the alternate table/figure source explicitly.

**H3. Tokenizer family claim** (Round 1 #17; Round 5 §5.7)
- **Issue:** "byte-level BPE vs. BPE" — both Llama-3 and Qwen3 use byte-level BPE.
- **Severity:** HIGH. Reviewer easily fact-checks.
- **Fix:** Rephrase as suggested in Round 5.

### MEDIUM (tightening would help; not deal-breaking)

**M1. Step 2 max-over-y argument wording** (Round 1 #1; Round 2 §2.1; Round 3 #1)
- **Severity:** MEDIUM (proof is correct but presentation could trip a careful reader)
- **Fix:** add 1 sentence: "Since (j,k) ranges over all ordered pairs as we vary y over true classes (and the j=k case contributes 0), this gives the full pairwise diameter."

**M2. K_feat depends on H_T (target only)** (Round 3 #2)
- **Severity:** MEDIUM (potential reviewer confusion; not an error)
- **Fix:** Add a one-sentence remark: "Since K_feat = K_feat(H_T) is determined by the target's head, it is constant across proxies and does not affect within-target rank correlations."

**M3. W_N caveat is buried in trailing footnote** (Round 3 #5)
- **Severity:** MEDIUM (Specialization 2 description is the most important corollary; the W_N-doesn't-minimize-bound caveat should be prominent)
- **Fix:** Move the caveat from line 163 paragraph to a `Remark.` immediately following Eq. (12).

**M4. Joint-optimization concavity argument is loose** (Round 1 #12; Round 3 §3.3 #2)
- **Severity:** MEDIUM
- **Fix:** Replace "δ(W) is concave in Tr(...), γ(W) is convex in W" with a cleaner statement: "On the (non-convex) Stiefel manifold O(d), no closed-form solution exists for the sum δ(W) + γ(W); see [Edelman et al., 1998] for the relevant Riemannian-optimization machinery." Or simply delete the concavity claim and just say "no closed form".

**M5. Equality condition for Ω = Ω_N is loose under degenerate σ** (Round 1 #7; Round 3 §3.3 #1)
- **Severity:** MEDIUM (minor edge case)
- **Fix:** Change "iff W_N = I" to "iff I ∈ argmax Tr(Z_T^T Z_P W)", which handles the degenerate-singular-value case. Or just add "(generically)" qualifier.

**M6. CKA section placement in tightness appendix** (Round 3 §3.2 #3)
- **Severity:** MEDIUM (organizational)
- **Fix:** Either move CKA paragraph to general_orthogonal_alignments.tex (where the "Relation to CKA" subsection already lives) and avoid duplication, OR keep but acknowledge "see also app:trace_specialization for the complete CKA discussion".

**M7. Per-token vs per-sequence in main Eq. 8** (related to H1)
- **Severity:** MEDIUM (consistency with H1 fix)
- **Fix:** If we go with per-token reformulation (option (a) of H1), update main Eq. 8 to drop the 1/|y| factor.

**M8. "γ = 0 from head simplification" wording** (Round 1 #13; Round 2 §2.3)
- **Severity:** MEDIUM
- **Fix:** Change "γ = 0 from the head simplification" to "γ = 0 because $H_T = H_P$ (frozen LoRA head), so $\Sigma_P^{1/2}(I H_T - H_P) = 0$".

### LOW (cosmetic, defensive overkill)

**L1. forgetting_qwen.tex comment "Input by neurips_2025.tex"** (Round 1 #16)
- **Fix:** Change "neurips_2025" → "neurips_2026" in the file's first-line comment.

**L2. Tightness inequality wording about non-negative vector** (Round 1 #5)
- **Fix:** Drop "non-negative" qualifier: "by ‖x‖_2 ≤ ‖x‖_1 with x = (σ_1,...,σ_d)".

**L3. CKA discussion duplication between tightness and general_W appendices** (related M6)
- **Fix:** Merge or cross-reference.

**L4. Naming "Frobenius-form Procrustes similarity" for Ω_F** (Round 1 #41)
- **Fix:** Rename to "Frobenius similarity" (drop "Procrustes" since it doesn't arise from a Procrustes problem).

**L5. "trace Procrustes similarity" in main text (Eq. 5 / `eq:omega_def`)** (Round 5 §5.2)
- **Fix:** Could rename to "trace alignment similarity" for clarity, but this would need rippling through main text. Defensive overkill.

**L6. "Three orders of magnitude" hyperbole in forgetting_qwen.tex** (Round 1 #43)
- **Fix:** Change to ≈ 700× (computed correctly with the corrected numbers).

**L7. "Order of magnitude below" claim for GSM8K |ΔR|** (Round 4 §4.11)
- **Fix:** Change to "5–9× below" or "approaching an order of magnitude below".

**L8. Proof appendix Sec. "Relation to Classical Procrustes Shape Metrics" placement** (Round 1 #44)
- **Fix:** Consider moving from before Step 1 to after Step 5 as a Remark. Or keep where it is if intent is to set context first. Personal style preference.

**L9. K_pred wording — formal statement** (Round 1 #2)
- **Fix:** Spell out: "Since p̂ ∈ Δ^V (probability simplex), ‖p̂‖_2² ≤ ‖p̂‖_1² = 1 by ‖x‖_2 ≤ ‖x‖_1, hence ‖p̂ − e_y‖_2² = ‖p̂‖_2² − 2 p̂_y + 1 ≤ 2 − 2 p̂_y ≤ 2."

**L10. Verbose section: gating signal explanation in regularization_task_dependence.tex** (Round 1 #39)
- **Fix:** Could compress the 5-paragraph defensive section into 2 paragraphs. Optional.

**L11. "Models 2, 5 (meta-llama) are gated"** (Round 1 #47)
- Consider also noting model 7 (DeepSeek-R1-Distill-Llama-8B) which is derived from Llama; check whether it's also gated.

**L12. "isotropic features" strawman section in general_W** (Round 3 §3.3 #3)
- **Fix:** Compress the isotropic-special-case discussion to one paragraph; note isotropy is unrealistic for LLMs as the *primary point*, not a derivation aside.


---

## Round 7: Final Triage — Fix-Priority Ranking

After the deep-dive of Rounds 4–6, I now triage each issue into MUST FIX / SHOULD FIX / OPTIONAL.

### MUST FIX (would cause rejection or visible flaw)

1. **H1 — AR Extension normalization mismatch** (`ar_extension.tex` line 8 + main Eq. 8)
   - Why MUST: a math-attentive reviewer will notice the per-sequence vs per-token gap and may ding the "rigor" of the AR claim.
   - Effort: ≈ 5–10 lines added to ar_extension.tex; possibly 1 line tweak in main Eq. 8 if going with token-mean.

2. **H2 — Numerical inconsistency: 4.63 / 0.0023 in `forgetting_qwen.tex` line 22**
   - Why MUST: cross-table verification by a reviewer will reveal these don't match `tab:reg_compare_*`. Looks like sloppy citation.
   - Effort: change two numbers and possibly one sentence.

3. **H3 — Tokenizer family claim (`forgetting_qwen.tex` line 7)**
   - Why MUST: factually inaccurate; both Llama-3 and Qwen3 use byte-level BPE.
   - Effort: ≈ 1 sentence rewrite.

### SHOULD FIX (improves quality, low cost)

4. **M3 — W_N caveat placement** (proof file line ≈ 163)
   - Why SHOULD: the Specialization 2 corollary is what the ablation cites; the caveat is critical for understanding.
   - Effort: move 1 sentence to a more prominent location.

5. **M1 — Step 2 max-over-y wording** (proof file line ≈ 73–76)
   - Why SHOULD: a 1-sentence clarification prevents reviewer confusion.
   - Effort: 1 sentence.

6. **M4 — Joint-optimization concavity argument** (general_W file line 80)
   - Why SHOULD: the convexity sketch is loose; either delete or replace with cleaner reference.
   - Effort: delete or rewrite 2 sentences.

7. **M2 — K_feat depends only on H_T** (proof file Step 2)
   - Why SHOULD: prevents reviewer confusion about K_feat varying across variants.
   - Effort: 1 sentence remark.

8. **M8 — "γ = 0 from head simplification"** (general_W line 92)
   - Why SHOULD: explicit statement of why γ = 0 prevents misreading.
   - Effort: 1 sentence rewrite.

9. **L9 — K_pred derivation explicit** (proof file line ≈ 187)
   - Why SHOULD: spelling out the simplex argument tightens the proof and is cheap.
   - Effort: 1 line.

### OPTIONAL (cosmetic / defensive)

10. M5 (degenerate σ edge case) — academic; skip unless polishing.
11. M6 (CKA section placement) — organizational; could keep as is.
12. M7 (token-mean update in main Eq. 8) — depends on H1 fix path.
13. L1 (file comment "neurips_2025") — trivial cleanup.
14. L2 (tightness inequality wording) — cosmetic.
15. L4 (rename "Frobenius-form Procrustes similarity") — cosmetic.
16. L6 (three-OOM hyperbole) — automatic if H2 numbers update.
17. L7 ("order of magnitude below" GSM8K) — minor.
18. L8 (Procrustes shape metrics section placement) — style preference.
19. L11 (DeepSeek gating note) — verify if needed; otherwise skip.
20. L12 (isotropic features strawman) — compression; optional.
21. L10 (regularization defensive verbosity) — optional compression.


---

## Round 8: Concrete Fixes for MUST FIX and SHOULD FIX Issues

For each high-priority issue, I provide exact location, current text (quoted), proposed fix (exact replacement), and 1–2 sentence justification.

### Fix 1 (MUST FIX H1): AR Extension Normalization Mismatch

**File:** `/Users/cylin/github/prism/paper/appendix/ar_extension.tex`, line 7–8 (paragraph "From tokens to matrices.")

**Current text:**
```
Under teacher forcing, each target token $y_\tau$ in a sequence $(c, y)$ contributes an independent feature--loss pair $(\phi_M(c, y_{<\tau}), \ell(\phi_M(c, y_{<\tau}) H_M, y_\tau))$. The per-sequence average inside Eq.~(\ref{eq:ar_risk}) is therefore the empirical mean of these pairs over the $|y|$ positions, and the expectation over sequences becomes an empirical mean over all $N = \sum_{(c,y)} |y|$ tokens in the calibration set. Stacking the token-level features into a matrix $Z_M^{\mathrm{AR}} \in \mathbb{R}^{N \times d}$ therefore reduces the autoregressive risk gap to exactly the per-row setting that Theorem~\ref{thm:unified_bound} controls.
```

**Proposed replacement:**
```
Under teacher forcing, each target token $y_\tau$ in a sequence $(c, y)$ contributes a feature--loss pair $(\phi_M(c, y_{<\tau}), \ell(\phi_M(c, y_{<\tau}) H_M, y_\tau))$. The per-sequence factor $1/|y|$ inside Eq.~(\ref{eq:ar_risk}) gives each token in sequence $(c,y)$ row weight $w_{(c,y),\tau} = 1/(|y|\,|\mathcal{D}|)$ in the empirical risk, and the bound below is stated for this weighted aggregation; in our experiments we report the equally-weighted token-mean $\hat{\mathcal{R}}_M = \tfrac{1}{N}\sum_{\tau} \ell_\tau$ over all $N = \sum_{(c,y)} |y|$ tokens, which coincides with the per-sequence mean when sequence lengths are approximately uniform (true for the four short-target benchmarks—ARC, MMLU, SQuAD, TriviaQA—and approximated by length-normalized averaging on GSM8K). Stacking the token-level features into $Z_M^{\mathrm{AR}} \in \mathbb{R}^{N \times d}$ then reduces the autoregressive risk gap to the per-row setting that Theorem~\ref{thm:unified_bound} controls.
```

**Justification:** Acknowledges the per-sequence vs token-mean distinction explicitly; states what is actually measured in experiments; closes the rigor gap that a careful reviewer will spot.

---

### Fix 2 (MUST FIX H2): Numerical Inconsistency in `forgetting_qwen.tex`

**File:** `/Users/cylin/github/prism/paper/appendix/forgetting_qwen.tex`, line 22

**Current text:**
```
A small subset of subplots in Fig.~\ref{fig:forget_grid_qwen}---most notably Qwen3-8B fine-tuned on BBQ evaluated on MMLU and TriviaQA---shows weak or slightly negative per-subplot $r_s$. These cases share a common signature: the baseline empirical forgetting at $\lambda{=}0$ is at the noise floor (e.g., Qwen3-8B BBQ on TriviaQA: $|\Delta\mathcal{R}|{=}0.0023$ at $\lambda{=}0$, vs.\ Llama TruthfulQA on TriviaQA: $|\Delta\mathcal{R}|{=}4.63$---three orders of magnitude apart).
```

**Proposed replacement:**
```
A small subset of subplots in Fig.~\ref{fig:forget_grid_qwen}---most notably Qwen3-8B fine-tuned on BBQ evaluated on MMLU and TriviaQA---shows weak or slightly negative per-subplot $r_s$. These cases share a common signature: the baseline empirical forgetting throughout the LoRA trajectory stays near the noise floor (e.g., Qwen3-8B BBQ on TriviaQA: $|\Delta\mathcal{R}| \approx 0.003$ at step 300, vs.\ Llama TruthfulQA on TriviaQA where $|\Delta\mathcal{R}|$ reaches $\approx 2.6$---roughly three orders of magnitude apart).
```

**Justification:** The 4.63 was from the trace-norm sweep table (different LoRA hyperparameter setting), not the comparison table at step 300 — using the comparison table values (0.0035 ≈ 0.003 and 2.5829 ≈ 2.6) makes the citation internally consistent. The "roughly" qualifier softens the "three orders of magnitude" claim (2.6/0.003 ≈ 870×, ≈ 2.9 OOM). 

---

### Fix 3 (MUST FIX H3): Tokenizer Family Claim

**File:** `/Users/cylin/github/prism/paper/appendix/forgetting_qwen.tex`, line 7

**Current text:**
```
Qwen3-8B differs from Llama-3.1-8B in depth (36 vs.\ 32 layers), vocabulary size ($151{,}936$ vs.\ $128{,}256$), tokenizer family (byte-level BPE vs.\ BPE), and pre-training corpus (multilingual, $\sim$36T tokens vs.\ English-dominant, $\sim$15T tokens); reproducing the main-text patterns under these shifts provides cross-family evidence for the claims of Secs.~\ref{subsec:forget_exp}--\ref{subsec:shape_reg_exp}.
```

**Proposed replacement:**
```
Qwen3-8B differs from Llama-3.1-8B in depth (36 vs.\ 32 layers), vocabulary size ($151{,}936$ vs.\ $128{,}256$), and pre-training corpus (multilingual, $\sim$36T tokens vs.\ English-dominant, $\sim$15T tokens); reproducing the main-text patterns under these shifts provides cross-family evidence for the claims of Secs.~\ref{subsec:forget_exp}--\ref{subsec:shape_reg_exp}.
```

**Justification:** Both Llama-3 and Qwen3 use byte-level BPE; the original "byte-level BPE vs. BPE" wording is incorrect. The remaining three differences (depth, vocab, corpus) are accurate. 

---

### Fix 4 (SHOULD FIX M3): W_N Caveat Placement

**File:** `/Users/cylin/github/prism/paper/appendix/proof_of_the_unified_risk_bound.tex`, around line 163.

**Current text (Specialization 2 closing paragraph, line 163):**
```
the form underlying the Sec.~\ref{subsec:ablation} ablation. Note that $W_N$ minimizes only the feature alignment residual $\delta(W)$, not the full bound $\delta(W) + \gamma(W)$, since the head term also depends on $W$ via $W H_T - H_P$. Appendix~\ref{app:joint_opt} treats the joint optimization in full and shows that $W_N$ remains the operative choice in the frozen-head LoRA and FP16-head PTQ settings studied here, where $\gamma$ either vanishes or is $W$-independent.
```

**Proposed replacement (split into a Remark following Eq. (12)):**
```
the form underlying the Sec.~\ref{subsec:ablation} ablation.
\begin{remark}[$W_N$ is the feature-side optimum, not the joint optimum]
The Procrustes alignment $W_N$ minimizes only the feature alignment residual $\delta(W)$, not the full bound $\delta(W) + \gamma(W)$, since the head term also depends on $W$ via $W H_T - H_P$. Appendix~\ref{app:joint_opt} treats the joint optimization in full and shows that $W_N$ is the operative choice in the frozen-head LoRA and FP16-head PTQ settings studied here, where $\gamma$ either vanishes or is $W$-independent.
\end{remark}
```

**Justification:** The Specialization 2 corollary is what the Sec. 4.5 ablation cites; the caveat that $W_N$ doesn't minimize the joint bound is critical for any reader using this corollary, and a `\begin{remark}` block raises its visibility.

---

### Fix 5 (SHOULD FIX M1): Step 2 max-over-y wording

**File:** `/Users/cylin/github/prism/paper/appendix/proof_of_the_unified_risk_bound.tex`, lines 71–77.

**Current text:**
```
\begin{equation}
\|\nabla_z \ell\|_2 \le \sum_{j \neq y} \hat{p}_j \|h_{T,j} - h_{T,y}\|_2 \le \max_{j \neq y} \|h_{T,j} - h_{T,y}\|_2 \cdot \underbrace{\sum_{j \neq y} \hat{p}_j}_{\le\, 1}.
\end{equation}
Taking the maximum over all possible true classes $y$:
\begin{equation}
\label{eq:kfeat_tight}
K_{\mathrm{feat}} = \max_{j,k} \|h_{T,j} - h_{T,k}\|_2.
\end{equation}
```

**Proposed replacement:** Add a one-sentence justification after the first equation:
```
\begin{equation}
\|\nabla_z \ell\|_2 \le \sum_{j \neq y} \hat{p}_j \|h_{T,j} - h_{T,y}\|_2 \le \max_{j \neq y} \|h_{T,j} - h_{T,y}\|_2 \cdot \underbrace{\sum_{j \neq y} \hat{p}_j}_{\le\, 1}.
\end{equation}
The bound depends on the true class $y$ only through $\max_{j \neq y} \|h_{T,j} - h_{T,y}\|_2$. Taking the supremum over $y \in \{1,\ldots,V\}$ extends the maximum to all ordered pairs $(j,y)$ with $j\neq y$ (and the $j=y$ case contributes $0$), giving the full pairwise diameter:
\begin{equation}
\label{eq:kfeat_tight}
K_{\mathrm{feat}} = \max_{j,k} \|h_{T,j} - h_{T,k}\|_2.
\end{equation}
```

**Justification:** Makes the index re-arrangement explicit; prevents a careful reader from worrying that the proof bounds only "distances to the correct token" rather than all pairwise distances.

---

### Fix 6 (SHOULD FIX M4): Joint-Optimization Concavity Sketch

**File:** `/Users/cylin/github/prism/paper/appendix/general_orthogonal_alignments.tex`, lines 79–80.

**Current text:**
```
\paragraph{Why No Closed-Form Solution Exists.}
The feature term $\delta(W)$ is concave in $\operatorname{Tr}(Z_T^\top Z_P W)$, while $\gamma(W)$ is convex in $W$. Their sum is neither convex nor concave on $\mathcal{O}(d)$, precluding a simple variational characterization.
```

**Proposed replacement:**
```
\paragraph{Why No Closed-Form Solution Exists.}
The Stiefel manifold $\mathcal{O}(d)$ is non-convex, so neither term in the sum admits a single closed-form variational characterization on it. Although each term reduces to a Procrustes-type sub-problem in isolation ($\delta(W)$ via SVD of $Z_T^\top Z_P$, and $\gamma(W)$ via SVD of $\Sigma_P^{1/2} H_T H_P^\top \Sigma_P^{1/2}$ in the WOPP form), their sum couples through the same orthogonal $W$ and admits no joint closed-form minimizer.
```

**Justification:** The original "concave + convex" sketch is technically loose; the real obstruction is that the feasible set is non-convex and the cross-term coupling has no clean SVD reduction. The replacement is more precise without claiming more than is true.

---

### Fix 7 (SHOULD FIX M2): K_feat dependence remark

**File:** `/Users/cylin/github/prism/paper/appendix/proof_of_the_unified_risk_bound.tex`, around line 79 (after the existing Remark on uniform shifts).

**Current text (Remark, line 79):**
```
\paragraph{Remark.} This bound depends on the \emph{relative distances} between token embeddings, not their absolute magnitudes. A uniform shift $H_T \to H_T + c\mathbf{1}^\top$ does not change $K_{\mathrm{feat}}$. A naive Cauchy--Schwarz bound gives $K_{\mathrm{feat}}^{\mathrm{naive}} = \sqrt{2}\|H_T\|_2$, which is substantially looser.
```

**Proposed replacement (extend the Remark):**
```
\paragraph{Remark.} This bound depends on the \emph{relative distances} between token embeddings, not their absolute magnitudes. A uniform shift $H_T \to H_T + c\mathbf{1}^\top$ does not change $K_{\mathrm{feat}}$. A naive Cauchy--Schwarz bound gives $K_{\mathrm{feat}}^{\mathrm{naive}} = \sqrt{2}\|H_T\|_2$, which is substantially looser. Since $K_{\mathrm{feat}}$ is determined by $H_T$ alone, it is a constant for a fixed target model and therefore does not affect the rank correlations PRISM is calibrated to (it scales the bound's magnitude, not its variant ordering).
```

**Justification:** Heads off the reviewer concern "K_feat varies across proxies → invalidates ranking" by stating explicitly that K_feat depends only on the target.

---

### Fix 8 (SHOULD FIX M8): "γ = 0 from head simplification" wording

**File:** `/Users/cylin/github/prism/paper/appendix/general_orthogonal_alignments.tex`, line 92.

**Current text:**
```
\textbf{LoRA ($H_T = H_P$):} The head term becomes $K_{\mathrm{pred}}\|\Sigma_P^{1/2}(W - I) H_T\|_F$, which vanishes at $W = I$ (and not for arbitrary $W$). We adopt $W = I$ throughout the LoRA experiments and the shape regularizer of Sec.~\ref{subsec:shape_reg}: it gives the cleanest decomposition---$\gamma = 0$ from the head simplification, scale $(\rho_T-\rho_P)^2$ invariant of $W$ (observed but not actively regularized, since LoRA primarily perturbs shape rather than scale; Sec.~\ref{subsec:decompose}), and shape $1-\Omega$ as the differentiable target.
```

**Proposed replacement:**
```
\textbf{LoRA ($H_T = H_P$):} The head term becomes $K_{\mathrm{pred}}\|\Sigma_P^{1/2}(W - I) H_T\|_F$, which vanishes at $W = I$ (and is generically positive at $W \neq I$). We adopt $W = I$ throughout the LoRA experiments and the shape regularizer of Sec.~\ref{subsec:shape_reg}: under the frozen LoRA head ($H_T = H_P$) the choice $W = I$ gives $\Sigma_P^{1/2}(I H_T - H_P) = 0$, so $\gamma = 0$; the scale arm $(\rho_T - \rho_P)^2$ is then $W$-invariant (observed but not actively regularized, since LoRA primarily perturbs shape rather than scale; Sec.~\ref{subsec:decompose}); and the shape arm $1 - \Omega$ becomes the single differentiable training-time target.
```

**Justification:** Makes the chain "frozen head $\Rightarrow$ $H_T = H_P$ $\Rightarrow$ $\gamma = 0$ at $W = I$" explicit; avoids the misreading that "$W = I$ alone implies $\gamma = 0$".

---

### Fix 9 (SHOULD FIX L9): K_pred derivation explicit

**File:** `/Users/cylin/github/prism/paper/appendix/proof_of_the_unified_risk_bound.tex`, line 187.

**Current text:**
```
Since $\hat{p}$ lies on the probability simplex, $\|\hat{p}\|_2^2 \le \|\hat{p}\|_1^2 = 1$, and therefore $\|\hat{p} - e_y\|_2^2 = \|\hat{p}\|_2^2 - 2\hat{p}_y + 1 \le 2 - 2\hat{p}_y \le 2$.
```

**Proposed replacement:**
```
Since $\hat{p} \in \Delta^V$ (the probability simplex), $\|\hat{p}\|_2 \le \|\hat{p}\|_1 = 1$ by $\|\cdot\|_2 \le \|\cdot\|_1$, hence
\[
\|\hat{p} - e_y\|_2^2 \;=\; \|\hat{p}\|_2^2 - 2\hat{p}_y + 1 \;\le\; 1 - 2\hat{p}_y + 1 \;=\; 2(1 - \hat{p}_y) \;\le\; 2.
\]
```

**Justification:** Spells out the $\|\hat{p}\|_2 \le 1$ step using the standard L2 ≤ L1 inequality, and also rewrites the chain so that the $2(1 - \hat{p}_y)$ form makes the supremum (as $\hat{p}_y \to 0$) immediately visible.


---

## Round 9: Final Summary

### Issue Counts by Severity

- **CRITICAL (mathematically wrong):** 0
- **HIGH (visible flaws to careful reviewers):** 3 (H1, H2, H3)
- **MEDIUM (tightening; low-cost quality improvements):** 8 (M1–M8)
- **LOW (cosmetic / defensive overkill):** 12 (L1–L12)

**Total real issues found:** 23. Of these, **3 must fix** (H1–H3) and **6 should fix** (M1, M2, M3, M4, M8, L9 — promoted because cheap). Remaining 14 are optional polish.

### Top 5 Most Important Fixes (Prioritized Action List)

1. **[H1] Fix AR-extension normalization mismatch** — clarify per-sequence vs per-token weighting in `ar_extension.tex`. ≈10-line addition; resolves the clearest theoretical-rigor concern. See Round 8 Fix 1.

2. **[H2] Reconcile the 4.63 / 0.0023 numbers in `forgetting_qwen.tex`** — replace with values consistent with `tab:reg_compare_*` (≈ 2.6 / ≈ 0.003). Trivial but highly visible cross-table check. See Round 8 Fix 2.

3. **[H3] Correct tokenizer family claim in `forgetting_qwen.tex` line 7** — both Llama-3 and Qwen3 use byte-level BPE. One-sentence rewrite. See Round 8 Fix 3.

4. **[M3] Promote W_N caveat to a `\begin{remark}` block in proof Step 3** — the "W_N minimizes only δ, not the joint bound" caveat is the most important consumer-facing implication and currently sits in a trailing footnote sentence. See Round 8 Fix 4.

5. **[M4] Tighten the "no closed form" reasoning in `general_orthogonal_alignments.tex` joint-optimization paragraph** — the current concavity sketch is loose. Replace with a cleaner statement about the Stiefel manifold's non-convexity and the cross-coupling. See Round 8 Fix 6.

### Estimated Total Effort

| Fix | Files | Lines added/changed | Time estimate |
|---|---|---|---|
| Fix 1 (H1) | 1 (ar_extension.tex) + possibly main Eq. 8 | ~10 lines | 30 min (incl. proofreading) |
| Fix 2 (H2) | 1 (forgetting_qwen.tex) | ~3 numbers, 1 sentence | 5 min |
| Fix 3 (H3) | 1 (forgetting_qwen.tex) | ~1 sentence | 2 min |
| Fix 4 (M3) | 1 (proof file) | ~1 paragraph reorganization | 5 min |
| Fix 5 (M1) | 1 (proof file) | ~2 sentences | 5 min |
| Fix 6 (M4) | 1 (general_W) | ~3 sentences | 10 min |
| Fix 7 (M2) | 1 (proof file) | ~1 sentence | 3 min |
| Fix 8 (M8) | 1 (general_W) | ~1 paragraph rewrite | 5 min |
| Fix 9 (L9) | 1 (proof file) | ~3 lines of math display | 5 min |
| **Total (all 9)** | **3 files** | **~20 lines net** | **~70 min including proofreading** |

If only H1–H3 are addressed: **3 files, ~12 lines, ~40 minutes**.
If only the must-fix are addressed and all should-fix deferred: **2 files, ~5 lines, ~15 minutes**.

### Files Affected (Action List)

- `paper/appendix/ar_extension.tex`: H1 (Fix 1)
- `paper/appendix/forgetting_qwen.tex`: H2, H3 (Fixes 2, 3); also L1 (file-comment cleanup; trivial)
- `paper/appendix/proof_of_the_unified_risk_bound.tex`: M1, M2, M3, L9 (Fixes 4, 5, 7, 9)
- `paper/appendix/general_orthogonal_alignments.tex`: M4, M8 (Fixes 6, 8)
- `paper/neurips_2026.tex`: optional Eq. 8 update if H1 fix (a) is chosen

### Final Assessment

The proofs are **mathematically sound** — no true errors. The bound formula, Lipschitz analysis, Procrustes decomposition, and head-bound derivation are all correct, and every numerical claim I verified against tables checks out (Llama Q2_K MMLU shape ≈ 9000, ratio ≈ 280×, regularization means 0.843/0.764/0.681, gating table 1-Ω̄ values, etc.).

The main weaknesses are **rigor-presentation** issues:
- The AR extension is informal; with tightening it can be made fully rigorous (Fix 1).
- One number citation (4.63 / 0.0023) doesn't match the comparison tables (Fix 2).
- One factual claim (tokenizer family) is wrong (Fix 3).

Beyond these, the appendix is dense and defensive but well-supported. The decomposition framework is clearly motivated, the experiment claims trace cleanly to source tables, and the cross-references all resolve. With the three MUST FIX changes (≈ 15 minutes of editing), the appendix is in solid shape for submission.

The MEDIUM and LOW items are quality-of-life improvements; addressing all SHOULD FIX adds another ~30 minutes for noticeable polish but is not strictly required.

