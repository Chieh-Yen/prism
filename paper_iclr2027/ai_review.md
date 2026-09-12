NeurIPS 2026 Conference Program Chairs added a new a LLM Feedback edit × 2  •  an hour ago
LLM Feedback by Program Chairs
Feedback: Hello!

You requested a review of your paper submitted to NeurIPS using the Google Paper Assistant Tool (PAT). The resulting AI Feedback can be found below. Note that this feedback is posted automatically, and is only visible to authors. Importantly, the feedback will **not** be used in the review process. Reviewers, area chairs, and program committee members will **not** have access to the PAT feedback.

Disclaimer: Please note that the models used by the PAT pipeline are not infallible; they may hallucinate and make mistakes. Authors should treat the generated feedback with the same critical eye they would apply to a human review.


# Summary

The paper introduces PRISM (Proxy Risk Inference via Structural Mapping), a geometric framework designed to bound the cross-entropy risk gap between a target Large Language Model (LLM) and its proxy variants (e.g., quantized or LoRA fine-tuned models). By leveraging the structural decoupling of the non-linear backbone and the linear prediction head, the authors decompose the Procrustes alignment residual into three independently measurable diagnostic axes: scale mismatch, shape mismatch, and covariance-weighted head divergence. This decomposition provides a closed-form upper bound on the cross-entropy risk gap, which empirically correlates well with actual performance degradation across various benchmarks and model families. Furthermore, the framework utilizes the differentiable shape mismatch term as a training-time regularizer to mitigate catastrophic forgetting during parameter-efficient fine-tuning.

# Strengths

* High Novelty and Significance: Deriving a closed-form upper bound on the cross-entropy risk gap of deployed, frozen prediction heads is a highly novel theoretical contribution. It elegantly bridges the gap between scalar representational similarity metrics (like CKA or SVCCA) and downstream task performance.

* Actionable Diagnostics: By exactly decomposing the Procrustes residual into scale, shape, and head divergence axes, the framework transforms a generic scalar similarity score into a highly actionable diagnostic tool. This allows practitioners to isolate specific failure modes, such as scale collapse from outlier clipping versus structural shape distortion from low-bit quantization.

* Training-time Regularization: The utilization of the differentiable shape mismatch term (trace Procrustes similarity) as an active training-time regularizer to mitigate catastrophic forgetting during LoRA fine-tuning is an innovative and practical extension of the diagnostic framework.

* Mathematical Rigor in Feature Bounding: The use of a 'simplex polarization' argument to sharpen the Lipschitz constant 
 is a strong mathematical refinement. By proving that the constant depends strictly on the relative Euclidean distance between token embeddings rather than their absolute magnitudes, the authors successfully prevent the bound from becoming vacuously loose for modern LLMs with large vocabularies.



# Weaknesses

* Loose Absolute Bound / Focus on Relative Ranking: While the theoretical derivation is elegant, the authors explicitly acknowledge that the bound remains numerically loose for large vocabulary sizes, effectively serving as a relative ranking metric rather than a tight absolute risk bound. The authors might consider discussing if there are any future pathways to tighten the absolute numerical estimation of the risk gap.

* Limitation to Teacher-Forced Evaluation: The PRISM bound relies heavily on teacher-forced evaluation, which does not account for compounding errors or exposure bias in open-ended, multi-step autoregressive generation. The authors might consider clarifying this limitation in the main text and discussing how divergent token predictions in free-running generation might affect the Procrustes alignment.

* Data Leakage in Regularizer Hyperparameter Tuning: In Section 5.4, the shape regularizer and the replay-CE baseline are evaluated 'each at its sweep 
-best'. Selecting the optimal hyperparameter 
 by maximizing performance directly on the test set cross-entropy risk gap constitutes data leakage. Please consider selecting the optimal hyperparameter using a disjoint validation set or an intrinsic metric to properly assess generalization.

* Overstated Generalization of the Shape Regularizer: The claim that the shape regularizer 'suffices to reduce downstream forgetting across benchmarks' (Lines 56-57) appears somewhat overstated. The supplementary tables indicate that the regularizer occasionally increases mean forgetting relative to the `no reg` baseline (e.g., Llama-3.1-8B on BBQ in Table 17 and Qwen3-8B on TruthfulQA in Table 18). The authors might consider revising the text to accurately reflect these nuances.

* Contradictory Claims on Benchmark Correlations: There appear to be discrepancies between the textual claims and the appendix tables regarding benchmark correlations. The text claims FineWeb-Edu retains 
 and MC/QA benchmarks achieve 
, but the tables show FineWeb-Edu ranging from 0.15 to 0.65, and GSM8K occasionally falling below the claimed range (e.g., 0.45 or -0.57). Please consider correcting these textual claims to align with the empirical data.

* Misleading Aggregation of Rank Correlations: Reporting the mean absolute Spearman correlation (
) in Section 5.2 masks instances where the bound is inversely correlated with risk (e.g., 
 for GSM8K in Table 15). The authors might want to report the mean of the raw 
 values and discuss the implications of negative correlations.

* Mathematical Inaccuracies in Joint Optimization Claims: In Appendix C.3, the claims that if 
, 'the head term becomes independent of 
' and that for frozen-head LoRA, 'the head term vanishes for any 
' appear mathematically incorrect. The squared Frobenius norm maintains a linear dependence on 
, and the term only vanishes if 
 acts as the identity matrix on the column space of 
. Please consider revising these statements for mathematical precision.

* Strictly Tighter Claim for Head Bound: In Appendix A.5, the claim that the proposed covariance-weighted head bound is 'strictly tighter' than the spectral bound is inaccurate because the proposed bound uses Jensen's inequality, which can result in a looser bound in isotropic scenarios. Please consider softening this claim.

* Mathematical Contradiction in Figure 1 Caption: The caption for Figure 1 states that under the optimal Procrustes alignment 
, the head discrepancy term 
 vanishes when 
. However, this is generally only true for the identity alignment 
, not necessarily for the optimal alignment 
. The authors might consider revising the caption to clarify this distinction.

* Inconsistent Notation in Decomposition Tables: In Equations 5 and 6, 
 and 
 are defined to include their respective Lipschitz constants so that 
. However, in the experimental tables (e.g., Table 1), the 
 and 
 columns seem to report unscaled geometric residuals that do not sum to 
. The authors might consider clarifying this distinction in the column headers or table captions.



# Potential Issues And Suggestions

### [Introduction and Related Work] (Pages: 1-3)
### Potential Mistakes and Improvements

- **Mathematical Contradiction in Figure 1 Caption:** The caption for Figure 1 states that under the Procrustes alignment 
, the head discrepancy term 
 measures 
, and adds that this term is "vanishing when 
 (frozen-head LoRA)." This is mathematically incorrect. If 
 is the optimal orthogonal alignment for the features, it will generally not be the identity matrix (
). Consequently, even if the prediction heads are identical (
), the term 
 will not equal zero. The main text (Section 3.3, Line 145) correctly notes that the head term vanishes for frozen heads specifically under the identity alignment (
). It is recommended to revise the Figure 1 caption to clarify that 
 vanishes when 
 under the identity alignment, rather than the optimal alignment 
.

- **Population vs. Empirical Risk Definition:** In Section 3.1, Equation 2 defines the cross-entropy risk 
 as an expectation over the data distribution 
. However, the quantities used to construct the bound in Theorem 1 (such as 
, 
, 
, and 
) are explicitly defined as empirical statistics calculated on a calibration sample of size 
. Because Theorem 1 bounds the population-level risk gap 
 using these finite-sample empirical terms, there is a mathematical discrepancy. The proof in Appendix A.3 (Equation 20) substitutes the sample average directly for the expectation. It would improve theoretical precision to clarify whether Theorem 1 formally bounds the *empirical* cross-entropy risk on the calibration set (meaning 
 acts as the empirical distribution), or to explicitly state the assumption that the empirical terms perfectly represent their population expectations without a generalization gap.

- **Clarity of 
 in the Main Text:** The introduction explicitly highlights that "the naive constant scales with the head's full spectral norm" as a key technical obstacle (Line 33), implying the paper provides a specific, tighter bound to overcome it. However, the resolution to this issue is omitted from the main text; Section 3.2 only references "
 (simplex polarization, Appendix A.3)". Providing the explicit formula for the sharpened Lipschitz constant (e.g., 
) briefly in the main text would improve clarity and immediately substantiate the core claim made in the introduction without requiring the reader to consult the appendix.

### Minor Corrections and Typos

- **Notation Inconsistency in Equations 1 and 2:** Equation (1) uses a dot operator for the matrix multiplication: 
. However, Equation (2) uses standard contiguous matrix multiplication notation without the dot: 
. Removing the dot in Equation (1) would make the notation consistent across both definitions.

- **Equation 2 Notation:** The set over which the index 
 is summed in the log-sum-exp term (
) is omitted. While the context implies the sum is over the vocabulary dimension 
, explicitly denoting it as 
 would ensure notational completeness.

### [PRISM Framework and Theoretical Bounds] (Pages: 3-6, 13-20)
### 1. Potential Mistakes and Improvements

*   **Incorrect Mathematical Claims in Appendix C.3 (Joint Optimization):** The discussion regarding the head discrepancy term 
 and the joint optimization of 
 contains two mathematically incorrect statements:
    *   **Line 603:** The text asserts that if 
, "the head term becomes independent of 
." This is algebraically false. Under this assumption, the head term evaluates to 
. The squared Frobenius norm expands to 
, which explicitly maintains a linear dependence on the orthogonal alignment 
.
    *   **Line 612:** The text claims that for frozen-head LoRA (
), "the head term vanishes for any 
." Setting 
 yields 
. This expression does not vanish for an arbitrary orthogonal matrix 
; it only vanishes if 
 acts as the identity matrix on the column space of 
. Consequently, the joint optimization does not trivially reduce to standard Procrustes.

*   **"Strictly Tighter" Claim for Head Bound (Appendix A.5):** On Line 515, the paper claims that the proposed covariance-weighted head bound is "strictly tighter" than the spectral bound 
. This claim is mathematically incorrect because the proposed bound uses Jensen's inequality (
). Because of this upper-bounding relaxation, the proposed bound can be looser in isotropic scenarios. For example, if 
 and 
, the proposed bound evaluates to 
, whereas the spectral bound evaluates to 
. Since 
, the spectral bound is tighter in this specific case. 

*   **Misattribution of Norm Inequalities to Cauchy-Schwarz (Appendix B):** On Lines 539–540 and 551–552, the paper invokes the Cauchy-Schwarz inequality to justify bounding an 
 norm or Frobenius norm from above by an 
 norm or trace. Applying Cauchy-Schwarz to the vector of singular values 
 and a vector of ones 
 yields the reverse dimensional bound: 
. The utilized inequality (
) is mathematically valid for non-negative singular values because 
, but this follows from basic properties of 
 norms for non-negative numbers, not Cauchy-Schwarz.

*   **Sequence-Level vs. Token-Level Expectation Mismatch (Appendix D):** Equation (7) defines the autoregressive risk 
 as the expectation over sequences of the *per-sequence average* loss (i.e., a macro-average: 
 
). However, Lines 620–624 state that stacking all tokens into a single matrix 
 reduces the risk gap "exactly to the per-row setting that Theorem 1 controls." Applying Theorem 1 to this flattened matrix computes a uniform mean over all tokens, which inherently weights sequences proportionally to their lengths. This algebraically differs from Equation (7) unless all sequences in the calibration set share the exact same length.

*   **Empirical vs. Population Risk Bounding (Section 3.1 & Appendix A.3):** In Equation (2), the risk 
 is defined as an expectation over the population distribution 
. However, in Equation (20), the derivation transitions directly from the population expectation 
 to exact algebraic equality with the finite-sample empirical mean 
 
. For this bound to be formally rigorous without introducing a generalization gap, 
 must be scoped explicitly as the *empirical* risk evaluated over the calibration sample.

### 2. Minor Corrections and Typos

*   **Notation Inconsistency for 
 and 
:** In Equations 5 and 6, 
 and 
 are defined to incorporate the Lipschitz constants 
 and 
, leading to the relationship 
. However, in all experimental tables (e.g., Table 1), the columns labeled 
 and 
 report the *unscaled* geometric residuals (such that 
 in the table values). Harmonizing this notation or adding a clarification to the table captions would resolve the ambiguity.

*   **Symmetric vs. Symmetric Positive Semi-Definite Condition (Appendix C.2):** Line 582 asserts that for the SVD 
, the equality 
 "holds when 
 is symmetric." This condition formally requires the matrix to be strictly symmetric *positive semi-definite* (SPSD). A symmetric matrix with negative eigenvalues will yield 
 to absorb the negative signs.

*   **Covariance vs. Second-Moment Terminology:** Line 114 defines 
 correctly as the "empirical second-moment matrix." However, throughout the rest of the text (e.g., Lines 49, 127), it is referred to as a "covariance" weighting. Because the proxy features are not mean-centered, calling it a second-moment projection is mathematically more precise.

*   **Undefined Notation (Appendix C.2):** Line 584 references the approximation "
". The subscripts 
 and 
 are undefined in this context. To align with the nomenclature established throughout the paper, this should read 
.

### [Empirical Evaluation and Applications] (Pages: 5-10, 19-36)
### 1. Potential Mistakes and Improvements

*   **Data Leakage in Regularizer Hyperparameter Tuning:** Section 5.4 evaluates the shape regularizer and the replay-CE baseline "each at its sweep 
-best" (Line 265). Selecting the optimal hyperparameter 
 by maximizing performance directly on the downstream evaluation metric (the test set cross-entropy risk gap) constitutes data leakage. To properly assess generalization, the optimal hyperparameter should be selected using a disjoint validation set or an intrinsic metric (e.g., target 
 retention) rather than the final test results.

*   **Overstated Generalization of the Shape Regularizer:** The text claims the shape regularizer "suffices to reduce downstream forgetting across benchmarks" (Lines 56-57). However, the supplementary tables indicate this effect is highly inconsistent. While Llama-3.1-8B on TruthfulQA improves, the regularizer *increases* mean forgetting relative to the `no reg` baseline for Llama-3.1-8B on BBQ (Table 17) and Qwen3-8B on TruthfulQA (Table 18). Claims regarding the regularizer's general efficacy should be revised to accurately reflect these negative results.

*   **Contradictory Claims on Benchmark Correlations:** In Appendix F.3 (Lines 685–688), the text claims that MC/QA/reasoning benchmarks consistently achieve 
 and that FineWeb-Edu retains 
. Both statements contradict the provided tables:
    *   FineWeb-Edu correlations range from 0.15 to 0.65 across Tables 9–15 and do not approach 0.92.
    *   The GSM8K reasoning benchmark frequently falls below the claimed range, scoring 0.51 (Table 9), 0.48 (Table 10), 0.68 (Table 11), 0.45 (Table 13), and -0.57 (Table 15).
    These textual claims should be corrected to match the empirical data.

*   **Misleading Aggregation of Rank Correlations:** In Section 5.2 (Line 222), the paper reports the "mean Spearman 
". Averaging the absolute values of correlation coefficients masks cases where the bound is inversely correlated with risk (e.g., 
 for GSM8K in Table 15). A diagnostic bound that negatively correlates with risk represents a failure mode in ranking. The authors should report the mean of the raw 
 values and discuss instances of negative correlation.

*   **Inconsistent Notation in Decomposition Tables:** In Equations 5 and 6, the feature alignment error 
 and head discrepancy 
 are defined to include their respective Lipschitz constants (
 and 
), ensuring the PRISM bound satisfies 
. However, in Tables 1 and 9–15, the 
 and 
 columns do not sum to 
 (e.g., Table 1 FP16: 
). The tables appear to report unscaled geometric distances instead. The column headers or table captions should be updated to clarify this distinction.

*   **Missing Quantization Variants in Appendix Tables:** 
    *   Table 11 (Qwen3-8B-Base) entirely omits the GPTQ variant, despite it being listed in Tables 6 and 8 and plotted in Figure 2.
    *   Table 15 (Qwen3-8B-Instruct) systematically omits the GGUF `Q3_K_M` and `Q2_K` variants across all benchmarks, despite Table 5 stating these tiers were applied universally.

### 2. Minor Corrections and Typos

*   **Contradictory GPTQ Bit-Widths:** Section 5.1 (Line 186) states the GPTQ protocol uses 4-bit precision, but Table 6 lists an INT8 checkpoint (`JunHowie/Qwen3-8B-GPTQ-Int8`) for Qwen3-8B, and Table 15 reports a "GPTQ-8bit" variant.

*   **Unused Models in Tables:** Tables 6, 7, and 8 list "Qwen2.5-7B" and "Qwen2.5-7B-Inst", which are absent from the primary target models list (Table 4) and have no corresponding evaluation tables.

*   **Broken Cross-Reference:** Line 683 contains a broken LaTeX reference ("Tables ??, 14, and 15 respectively"); this should point to Table 13.

*   **Table 3 "Wins / 10" Sums:** The number of wins in Table 3 sums to 11 for the top block and 17 for the bottom block. Given there are only 10 cells, this implies ties occurred, which should be explicitly noted in the caption.

*   **Table Captions:** The phrases "— ext task group" and "— all task group" in the captions of Tables 9–15 appear to be typos and should likely read "extended tasks" and "all tasks".

### [Conclusion, Limitations, and Bibliography] (Pages: 9-13)
**Verification Summary:**
The mathematical derivations provided in Appendix A (Page 13)—specifically the decomposition of the hybrid risk via the triangle inequality (A.1, A.2) and the derivation of the Lipschitz constant 
 via simplex polarization (A.3)—have been verified as logically sound and mathematically correct. They successfully yield the tighter bound required to support the claims made in the main text. Additionally, the Scope and Limitations section transparently and accurately bounds the claims of the framework (explicitly noting its design as a relative ranking metric rather than a tight absolute estimator).

### 1. Potential Mistakes and Improvements:

*   **Mismatch in listed mitigations (Page 9, Section 6):** In Lines 301-303, the text states, "protocol-level mitigations for the scale and head axes (per-channel outlier smoothing, Hessian-aware reconstruction, FP16-lm_head retention) are a research follow-up...". This introduces a slight logical contradiction. As established earlier in Section 5.3 (Lines 256-257), "Hessian-aware reconstruction" is specifically a remediation for *shape distortion*, whereas outlier smoothing and FP16-lm_head retention map to the scale and head axes, respectively. Consider either revising the text to "mitigations for the scale, shape, and head axes" or removing "Hessian-aware reconstruction" from the parenthetical list if the intent is to strictly highlight mitigations *other* than the shape regularizer mentioned in the first half of the sentence.

### 2. Minor Corrections and Typos:

*   **Typo in Cross-Reference (Page 13, Line 425):** There is a typo in the text: "Sections 4 and 4 show that the two arms of this split...". This should likely be updated to reference the distinct sections intended (e.g., Sections 4 and 5).

*   **Bibliography Formatting:** Several entries in the bibliography contain casing or naming errors, likely due to BibTeX parsing artifacts:
    *   Ref: "Klimov Oleg" should likely be reversed to "Oleg Klimov".
    *   Ref: "Gptq" should be capitalized as "GPTQ".
    *   Ref: "llama 3" should be capitalized as "Llama 3".
    *   Ref: "Svcca" should be capitalized as "SVCCA".

### [Administrative and NeurIPS Checklist] (Pages: 35-43)
### Potential Mistakes and Improvements

*   **Incomplete Checklist Entry:** In the NeurIPS Paper Checklist, Question 5 ("Open access to data and code", Lines 867–868) has both the Answer and Justification left as `[TODO]`. This section should be completed with the appropriate response prior to publication. For example, the authors might reference their plan to release code as supplementary material, which is currently stated in the justification for Question 4 (Line 829).

### Minor Corrections and Typos

*   None identified.
