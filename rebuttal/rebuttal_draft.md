================================================================================
PRISM (Submission 8674), Rebuttal draft, filled 2026-07-24
================================================================================
Legend / 貼文規則(2026-07-27 重整:本檔只留要貼的內容)

  貼:  每則的 "> ..." blockquote(= 該 weakness/question 的精簡摘要,貼在回覆
        最前面,讓 reviewer 一眼知道在答哪一條)
        + blockquote 之後的全部內文(2026-07-30 起不再有 "Response:" 標籤)
        + 結尾的 "In sum:" 收尾句(thread 級總結,給只掃結尾的人看)

  不貼:"[原文 ...]:" 的逐字引文    = 一一對應檢查用(改由上面的 "> " 摘要代替)
        "···· ORIGINAL ... ····"   = reviewer/AC 原文備查
        本 Legend

  合併規則:weakness 與強相關的 question 合併成一則(標頭如 "W2 + Q1"),
        兩邊原文並列、答案只寫一次,確保兩者都被完整回答。

  內部註記(數字出處、決策理由、待辦、assembly checklist)已全部移到
        rebuttal/internal_notes.md,貼文前對照該檔的 checklist。

All numbers below are REAL (computed from existing repo data; sources:
rebuttal_exp/out/{E1,E3,E4,E6,E7,E8,E9,E11,E12,E13,eql6w3}/ and the submitted PDF).
================================================================================

================================================================================
GENERAL RESPONSE (to all reviewers and the Area Chair)
================================================================================

We thank the reviewers and the Area Chair for a careful and constructive
assessment, and for recognizing the paper's core contributions: the exact,
closed-form scale/shape decomposition, the vocabulary-independent Lipschitz
constant that keeps the bound informative at LLM scale, the principled link
between representation geometry and cross-entropy risk, and the shape term as a
differentiable training-time regularizer. The reviews converge on a fair and
useful message: the framework is technically sound, but its practical
significance and the completeness of its evaluation should be demonstrated
rather than asserted. We have taken this to heart.

Claims we narrow in the revision (as the meta-review suggests), conceding what
is genuinely limited and substantiating what is not:
  - Theorem 1 -> a bound on the empirical risk gap over the reference sample,
    with a concentration corollary for the population version;
  - the raw bound is scoped to ranking; only the calibrated bound is claimed to
    answer absolute thresholds;
  - the reference set is a small slice of the diagnosed task's own validation
    data; predicting one distribution's degradation from another
    distribution's features (unpaired use) is outside the derivation and not
    claimed;
  - Corollary 1 (autoregressive) -> teacher-forced-only, with trajectory shift
    an explicit limitation (and a new free-running experiment probing it);
  - validated claims scoped to PTQ and frozen-head LoRA; full SFT/RLHF is the
    joint-alignment regime (App. C.3) we leave to future work, with a first
    base-vs-instruct data point verifying validity there.

What the rebuttal establishes, each measured rather than asserted:
  - **cost**: one forward pass per variant against greedy decoding on the same 256
    prompts, 8.9 s versus 556.7 s (62.6x);
  - **free-running**: $r_s$ +0.947 ± 0.015 on self-generated text against
    +0.958 ± 0.011 teacher-forced, and a teacher-forced bound orders free-running
    gaps at +0.958 ± 0.011;
  - **diagnostic reference**: 8 sequences order the 12 variants as well as 512 do
    ($r_s$ 0.932 either way), with seed agreement on the ordering 0.981 to 0.998;
  - **regularizer reference**: 8 sequences already deliver the benefit, and four
    disjoint draws at each of n = 8/16/32 keep downstream forgetting within 5.9
    points of one another;
  - **against similarity scores**: on identical features PRISM ties CKA and holds at
    a small reference (0.932 ± 0.016 at n = 8) where SVCCA collapses (0.083 ± 0.036);
  - **against forgetting baselines**: at matched plasticity the shape penalty leads
    on both axes among the methods that learn the task, 0.680 ± 0.016 downstream and
    0.872 ± 0.005 on the target task, ahead of replay, L2-SP and EWC in every seed;
    layer-freezing's lower gap is bought at target loss 0.924;
  - **ranking versus absolute values**: the raw bound is scoped to ranking, since
    slack is large and not constant (median 1597x) while the gaps themselves span
    one to two orders of magnitude; per-cell calibration then recovers nats, at
    leave-one-out MAE 0.055 with precision >= 0.8 at 0.1 nats in 49/55 cells.

A closing word on the larger goal. All four reviews, and the meta-review, credit
the same core: a timely problem, a decomposition more informative than any single
similarity score, and a principled link between geometry and risk. The reason we
pursue exactly that combination is practical: **the ecosystem now produces
post-trained variants far faster than it can afford to evaluate them, and PRISM
is our attempt at the missing instrument, a measurement that turns "re-run the
benchmark suite for every variant" into one forward pass per variant, three
commensurable numbers, and a remediation direction**, so that the community's
time goes into building models rather than repeatedly grading them. This review process has made
the instrument materially sharper: the calibration, the empirical-risk
restatement, and the stronger baselines exist because the reviewers and the AC
asked exactly the right questions. The directions we could not fully pursue
within the rebuttal window (sequential continual learning, the joint-alignment
regime of full SFT/RLHF, a trajectory-shift term for free-running generation) are
stated in the revision as the explicit roadmap of this line of work.

We believe these results, together with the narrowed framing (PRISM as a
calibrated, generation-free ranking-and-attribution instrument with an explicitly
empirical-risk guarantee), directly address the five points of the meta-review,
and we respectfully ask the AC and reviewers to reconsider in light of the new
evidence. If any point still appears only partially met, we would welcome the
chance to complete it during the discussion period.

Thank you, to all four reviewers and the AC, for the time and care these reviews
reflect; the paper is genuinely better for it.

================================================================================

Meta Review (AC)
AC LN7U (17 Jul 2026, modified 24 Jul 2026), Preliminary: Borderline Reject, "What could change the recommendation?"

···· ORIGINAL META-REVIEW — verbatim from OpenReview (for 一一對應檢查; internal, DO NOT POST) ····
[Summary / Metareview]
The paper proposes PRISM, a framework for diagnosing degradation in variants of LLM
post-training, including quantized and LoRA-adapted models. The main technical claim is that
the cross-entropy risk gap between a target-base model and a proxy-variant model can be
upper-bounded by a closed-form geometric quantity decomposed into three interpretable
components: scale mismatch, shape mismatch, and head divergence. The authors argue that
these components correspond to different failure modes and that the differentiable shape
component can thus serve as a regularizer against catastrophic forgetting.

[Strong Points]
- Interesting and timely problem: All reviewers recognize that diagnosing post-trained LLM
  variant degradation is an important problem. PRISM attempts to go beyond measuring
  degradation and decomposes drift into interpretable components.
- Conceptually appealing decomposition: Reviewers generally find the decomposition into
  scale, shape, and head terms intuitive and potentially useful. pCi8 emphasizes that the
  scale/shape split is mathematically clean and exact, and G3T9 and 8VrD note that this
  decomposition is more informative than a single representation similarity score such as
  CKA or SVCCA.
- Connection between geometry and risk: Several reviewers appreciate the connection between
  representation geometry and cross-entropy risk. G3T9 identifies this as a key strength,
  and 8VrD highlights the hybrid risk construction that separates backbone drift from head
  drift.
- Promise as an empirical ranking tool: The reported rank correlations for PTQ and LoRA
  forgetting suggest that PRISM may be useful for ranking variants when direct evaluation is
  expensive. pCi8 and G3T9 both note the low computational cost and the potential practical
  value of using a small number of forward passes rather than a full benchmark suite.
- Shape regularization: The use of the differentiable shape term as a training-time
  regularizer is viewed positively in principle. pCi8, eQL6, and 8VrD all mention this as an
  attractive aspect of the paper, though they differ on how convincing the evidence is.

[Overall recommendation]
My preliminary recommendation is hovering around Borderline Reject. A strong and convincing
rebuttal with additional experiments and careful narrowing of the claims could move the
paper closer to borderline accept, but based on the current reviews, the weaknesses outweigh
the strengths.
(對應方式:五個 "What could change the recommendation?" 條件 = A–E 標頭的 [原文 A]–[原文 E];
六個 Weak Points 併入對應條件,標頭寫成 "A + W-1 + W-6a" 等,原文並列於同一標頭下,答案只寫
一次;W-4 與 W-5 各跨兩條件,全文引一次、另一處只引相關片段並註明;W-2 與 W-6b 不屬五條件,
放在 Point E 之後。)
···· END ORIGINAL META-REVIEW ····

--------------------------------------------------------------------------------
GLOBAL RESPONSE TO THE META-REVIEW

We thank the AC for distilling four reviews into five testable points: that
structure is what allowed this rebuttal to answer with measurements rather than
prose. **Against the bar the meta-review set, "concrete evidence rather than only
clarification", all five points are answered with new measurements, and the
claims are narrowed exactly where the meta-review invites narrowing.** The
strengths all four reviews credit and the narrowed claims are in our GENERAL
RESPONSE; the two experiments carrying most of the load are given once, up front.

| point | what was delivered |
|:--|:--|
| A. empirical or population; how loose; any threshold | restated on the empirical gap + concentration corollary; slack located, two of three sources are cell constants that cannot reorder; calibrated to a leave-one-out MAE of 0.055 nats against 0.082 for predict-the-mean, with precision >= 0.8 at a 0.1-nat threshold in 49/55 cells |
| B. what reference data the bound needs, how little, domain | needs only a small slice of the diagnosed task's own validation data, a scope the revision now states (the invited narrowing); 8 sequences order the variants as 512 do; 1/62.6x the compute of one benchmark run; the regularizer's own reference moves retention by at most 5.9 points across sizes 8-32 and disjoint draws |
| C. stronger baselines, both settings | ties CKA/SVCCA on identical features while adding what they cannot give (certified bound, axis attribution, trainable objective); shape penalty leads EWC, L2-SP and replay in every seed at matched plasticity; layer-freezing under-learns the task (target loss 0.924 against the shape penalty's 0.872) |
| D. free-running generation, or limit the claims | both: run ($r_s$ +0.947 free-running vs +0.958 teacher-forced), and Corollary 1 restated as teacher-forced-only |
| E. failures and mixed results | 18/20 LoRA cells positive; both flagged cells explained by stated mechanisms (noise-floor gap / non-monotone gap); Table 22 read as the gating validation it is, its one genuine exception owned |

------------------------------------------------------------------------------
### A + W-1 + W-6a: usefulness of the bound, and empirical vs population risk

[原文 A]: Clarify usefulness of the bound: The authors should explicitly state whether the
   result should be understood as an empirical/calibration-set bound or a population-risk
   statement. They should also quantify the looseness of the bound and explain whether it
   can support any operational threshold beyond ranking.
[原文 W-1]: Loose risk bound: Reviewers pCi8, G3T9, and 8VrD all note that the proposed
   bound appears useful mainly as a ranking signal, not as an operationally meaningful
   certificate of degradation. This weakens the framing of PRISM as a risk diagnostic
   rather than a calibrated heuristic for ordering variants.
[原文 W-6a]: Clarity and theoretical-framing issues: ... whether the stated risk guarantee
   should be framed more carefully as applying to empirical risk on calibration samples
   rather than population risk. (The presentation half of W-6 is answered after Point E.)

> **Point A (+ Weak Points 1, 6a).** Empirical or population bound? How loose?
> Any threshold beyond ranking?

Addressed in all three parts: the theorem is restated on the empirical gap with a
concentration corollary recovering the population version, the looseness is
quantified and located, and a per-cell calibration yields an operational
screening threshold (a filter, not a certificate).

(i) The guarantee, restated. The reviewers are right: Theorem 1 is written for the
population risk gap, but its proof works with a finite reference sample and carries
no generalization term. The revision therefore states it as a bound on the
empirical risk gap over the reference sample, which is the quantity every
experiment in the paper already reports, and adds a concentration corollary for the
population version: the per-sample loss gap is bounded (hidden states are
RMSNorm-bounded, so the aligned feature difference is bounded), so McDiarmid's
inequality recovers the population statement at an additive cost that shrinks as
1/sqrt(N) in the reference size N. The restated theorem and its proof are written,
and we can post them verbatim in this thread.

(ii) How loose it is, quantified and located. We measure looseness as slack, the
ratio of the bound to the risk gap it bounds, over all 570 (family, benchmark,
variant) cells of the paper's five benchmarks:

| looseness, located | value | consequence |
|:--|--:|:--|
| median slack | 1597x (IQR [673, 3966]) | the raw bound is not an estimate |
| slack spread | log10 sd 0.76 (~6x cell to cell) | no single correction rescales it |
| Lipschitz step (worst-case K vs observed) | at most ~2.5x | not where the looseness lives |
| remainder (alignment residual + triangle + Jensen steps) | 140x to 3000x | the dominant source |
| cell constants: $K_{\mathrm{feat}}$; $K_{\mathrm{pred}} \le \sqrt{2}$ (App. A.5); prefactor $\rho_T \rho_P$ | within-cell CV 0.63% | exactly (K) or nearly (prefactor) shared: cannot materially reorder |

**Two of the three slack sources are constants within a (model, benchmark) cell,
and a factor that is nearly the same for every variant inflates every bound by
nearly the same multiple, so it cannot change their order**; that is why the paper
reports rank correlations within cells: the ordering is what survives the loose
constants.

(iii) An operational threshold does exist, per cell. We fit a monotone (isotonic)
map from the bound to the measured gap inside each cell and evaluate it
leave-one-out, so no variant is ever scored by a map that has seen it; a monotone
family is right precisely because the constants in (ii) preserve order. Across the
55 (model, benchmark) cells the held-out error is 0.055 nats, against 0.082 for
predicting the cell's mean gap (each cell holds 10-12 variants, so we present
this as a proof of concept). On the two cells the reviews single out as extreme,
plus the deployment-relevant middle one:

| variant (Llama-MMLU) | bound B | measured gap | calibrated prediction | call at 0.1 nats |
|:--|--:|--:|--:|:--|
| `Q8_0` | 23.24 | 0.0002 | 0.003 | within / within, correct |
| `Q4_K_M` | 96.75 | 0.036 | 0.055 | within / within, correct |
| `Q2_K` | 266.09 | 0.366 | 0.193 | outside / outside, correct |

Reading "predicted gap below the tolerance" as an accept/reject rule, **precision
is at least 0.8 in 49 of the 55 cells (21 of them at 1.0), and in all 55 cells at a
0.5-nat tolerance**; the six weaker ones are all multiple-choice benchmarks, where
the measured gaps sit so close together that any threshold rule is fragile. The
limit is worth stating plainly: the map must be fitted on already-measured variants
of the same cell, so this amortizes benchmarking rather than replacing it, and it
is a filter rather than a certificate. Derivations in pCi8 W2 and 8VrD W1/W2+Q1.

------------------------------------------------------------------------------
### B + W-4a: the reference set: what the bound licenses, and size/domain sensitivity

[原文 B]: Demonstrate benchmark-independent use: The authors should show that PRISM remains
   predictive when computed on a small, benchmark-independent reference set, including
   sensitivity to reference-set size and domain.
[原文 W-4]: Narrow experimental setting: Reviewers G3T9 and 8VrD emphasize that PRISM is
   evaluated largely under teacher-forced feature extraction and benchmark-aligned
   reference data. It remains unclear how well the method transfers to benchmark-
   independent calibration sets, free-running generation, reasoning-heavy tasks, or
   variants that drift farther from the base model.
   (Quoted in full here; its free-running half is answered under Point D, and its
   last two items in (iv) below.)

> **Point B (+ Weak Point 4).** Predictive on a small benchmark-independent
> reference set? Sensitivity to size and domain? Transfer to free-running generation,
> reasoning tasks, farther drift?

Addressed by explicit scope narrowing plus in-scope measurement: the reference the
bound needs is the diagnosed task's own data, so benchmark-independent use is
scoped out, the narrowing the meta-review invites. Size is measured on both
reference sets below, cost is measured, the two remaining W-4 items are answered,
and on domain we separate the two halves rather than answer only the easy one.

(i) What the reference set is, and what we do not claim. The bound's reference,
which we will call the **diagnostic set** (Sec. 5.1's fixed held-out subsets), is
a small slice of the diagnosed task's own validation data, read once, with no
decoding and no grading: 512 sequences per benchmark in the quantization study
and 256 in the LoRA study, which any evaluation already has on hand. The bound is
input-conditioned: computed on that slice, it bounds the risk gap on that same
data. Computing it on generic text in order to predict a benchmark's degradation
would break that pairing and turn the guarantee into a distribution-transfer
question the derivation does not address. We therefore do not claim
benchmark-independent use, and the revision states that scope in the Limitations.

(ii) Sensitivity to size, measured. Three fresh seeds, 12 Llama variants, 5
benchmarks, with the ground-truth gap held at the full 512 sequences:

| reference slice | bound vs full-slice gap ($r_s$) | seed agreement on the ordering |
|:--|--:|--:|
| 8 sequences | **+0.932 ± 0.016** | 0.981 |
| 32 | +0.931 ± 0.009 | 0.992 |
| 128 | +0.933 ± 0.011 | 0.993 |
| 512 (paper size) | +0.932 ± 0.011 | 0.998 |

**Eight sequences already order the twelve variants as well as 512 do.** The
requirement is small and stable, and at that slice the similarity baselines do not
hold up, which the four-score table under Point C shows. The paper's other reference set is the
**regularization reference $D_{\mathrm{ref}}$** (32 held-out sequences of the
fine-tuned task, Sec. 5.4), what the trace penalty reads during fine-tuning. The
revision adopts both names throughout and corrects Sec. 5.1, which calls
$D_{\mathrm{ref}}$ "pre-training sequences" in error.

$D_{\mathrm{ref}}$ has its own ablation, run at the paper's operating point so the numbers sit
beside Table 2's 0.681: four disjoint draws at each of n = 8 / 16 / 32 give mean
downstream forgetting 0.690 ± 0.019, 0.685 ± 0.011 and 0.676 ± 0.008, with all
twelve runs 15.2% to 21.1% below the no-regularization 0.843, so **the whole
size-and-draw grid moves retention by at most 5.9 points.** Two controls: rerunning
the paper's own draw reproduces it to 0.0016 nats, and the spread shrinks as the
reference grows, the signature of sampling variation. **On the domain half we have no
controlled answer, and the bound licenses none.** Preserving shape on generic text
and expecting benchmark retention is the same distribution-transfer step that (i)
scopes out, so we treat a generic reference as an untested heuristic rather than a
claim. The matched test, a reference drawn from a different structured-QA task, is
follow-up (8VrD Q3).

(iii) What that buys, measured on identical prompts (256 GSM8K prompts, one RTX
5090, model load excluded on both sides):

| pipeline | what it actually runs | compute |
|:--|:--|--:|
| PRISM | one teacher-forced pass, no decoding, no grading | **8.9 s** |
| benchmark, greedy decode | ~209 generated tokens per prompt, then parse and score | 556.7 s (**62.6x**) |
| benchmark, self-consistency maj@8 | ~eight times the above | 4453.6 s (501x) |

The gap widens with generation length, because what PRISM avoids is the
autoregressive decoding and the answer grading rather than the validation data.
Itemized in G3T9 W1 and W2.

(iv) The two remaining W-4 items. Reasoning-heavy tasks: GSM8K is kept as a
first-class limitation with its mechanism quantified (Point E below), and a
tested span-level mitigation fails and is reported as failing (pCi8 W4). Variants
farther from the base: we ran the decomposition directly across the full
post-training gap, base checkpoint as reference (target), SFT/instruct
counterpart as diagnosed variant (proxy):

| pair (base -> instruct) | cells where the bound holds | median head share of B | backbone drift vs own `Q2_K` level |
|:--|--:|--:|:--|
| Llama-3.1-8B -> Instruct | 5 / 5 | 52% | at that level: drift 0.224, `Q2_K` 0.225 |
| Qwen3-8B-Base -> Qwen3-8B | 5 / 5 | 48% | 14x beyond: drift 0.254, `Q2_K` 0.018 |

Validity and decomposability persist at this distance, with the head term engaging
at about half of B (pooled median 50%) exactly as the open-head regime predicts;
calibrated tightness there is App. C.3's regime and stays future work.

------------------------------------------------------------------------------
### C + W-3 + W-5a: stronger baselines, for diagnostics and for regularization

[原文 C]: Add stronger baselines: For diagnostics, direct comparisons to CKA, SVCCA, and
   simpler feature-preserving scores are needed. For regularization, stronger
   continual-learning baselines such as EWC, SLoRA, CLAIM, ArMA, or layer-freezing/replay
   variants would substantially improve the evaluation.
[原文 W-3]: Incomplete empirical comparisons: The paper positions PRISM against
   representation similarity metrics and catastrophic-forgetting methods, but reviewers
   G3T9, pCi8, and 8VrD argue that the evaluation lacks sufficiently strong or direct
   baselines in both the diagnostic/ranking setting and the regularization setting.
[原文 W-5]: Unconvincing regularization results: Reviewers eQL6, pCi8, and 8VrD find the
   shape regularizer promising but not yet fully validated. The evaluation uses limited
   baselines, shows mixed results across settings, and does not clearly establish the
   tradeoff between reducing forgetting and preserving new-task performance.
   (Quoted in full here; its "mixed results" half is answered under Point E.)

> **Point C (+ Weak Points 3, 5).** CKA/SVCCA and simpler feature scores for the
> diagnostic; EWC, SLoRA, CLAIM, ArMA, layer-freezing for the regularizer.

Addressed on both halves: the similarity baselines are compared on identical features
(a statistical tie, with three capabilities they lack), and three regularizer
families are added at matched plasticity (the shape penalty leads every one that
learns the task).

(i) Diagnostic baselines, on identical features. Every score below is computed on the
same features from one forward pass, with the risk gap recomputed from that same pass,
so no row enjoys a different extraction; the n=8 column holds the target gap at 512
and varies only the slice the score sees.

| score (identical features, 12 Llama variants, 3 fresh seeds) | slice n=8 | slice n=512 |
|:--|--:|--:|
| 1-CKA | +0.903 ± 0.015 | +0.931 ± 0.008 |
| 1-SVCCA | +0.083 ± 0.036 | +0.941 ± 0.014 |
| our feature arm $\delta_N$ (Procrustes) | +0.924 ± 0.007 | +0.944 ± 0.012 |
| PRISM $B_N$ (full certified bound) | +0.932 ± 0.016 | +0.932 ± 0.011 |

This is the like-for-like comparison the point asks for. **At full size the bound is
no worse than the similarity scores on ranking, and no better**: paired bootstraps put
the full bound against 1-CKA at +0.001 [-0.004, +0.007] here, and the feature arm's
small edge on this Llama rerun (+0.013) disappears once Qwen is included
(-0.002 [-0.038, +0.011]), so we claim parity, not an advantage. That parity is
expected, for an algebraic reason: our feature arm IS the Procrustes size-and-shape
distance, so Procrustes is the feature half of our own bound rather than an external
competitor, and App. A.1 states that our contribution is not that distance but its
split into a scale term and a shape term plus its lifting to a risk bound.

Two things that comparison does not reach. **First, the small-slice regime, where the
bound wins outright**: with eight reference sequences it is the highest of the four
scores (0.932 against 1-CKA's 0.903) while SVCCA collapses to 0.083, which is also the
size half of Point B. **Second, ranking is one of three outputs, and the only one a
similarity score can produce at all.** None of them bounds the risk gap, so none
certifies anything; none attributes the gap to an axis, which is what the paper's
diagnoses rest on; and none names the quantity the Sec. 5.4 regularizer penalizes.
Qwen3-8B-Base `Q6_K` on SQuAD has near-perfect backbone geometry ($\Omega$ close to 1)
yet its head term is 75.77 out of a bound of 76.95, while BnB INT8 on the same model
leaves the head untouched ($\gamma = 0$) at a bound of 3.81. That 20x difference is set
entirely by whether the protocol quantizes `lm_head`, and it is invisible to any
similarity score.

(ii) Regularizer baselines, at matched plasticity. As the meta-review asks, we add
layer-freezing (LoRA on the top-K layers, K in {4, 8, 16}) alongside EWC with a
diagonal Fisher and L2-SP, all under the paper's exact protocol. Each method is
reported at the sweep configuration whose target-task loss comes closest to the
shape run's (matched plasticity), so that no baseline can look good merely by
under-training the task it was asked to learn.
TruthfulQA, three seeds (42/43/44); downstream forgetting is the mean risk gap over
the five held-out benchmarks:

| method (TruthfulQA, 3 seeds) | downstream forgetting | target-task loss |
|:--|--:|--:|
| no regularization | 0.815 ± 0.042 | 0.950 ± 0.008 |
| replay, lambda 0.01 | 0.771 ± 0.027 | 0.899 ± 0.007 |
| L2-SP, lambda 0.01 | 0.763 ± 0.019 | 0.901 ± 0.003 |
| EWC, lambda 0.1 | 0.751 ± 0.015 | 0.905 ± 0.010 |
| **shape penalty, lambda 1** | **0.680 ± 0.016** | **0.872 ± 0.005** |
| layer-freeze, top 16 | 0.404 ± 0.017 | 0.924 ± 0.001 |

**The shape penalty is the lowest on forgetting among the methods that learn the
task to a comparable level, it beats each of them in all three seeds individually
by +0.050 to +0.157, and it carries the lowest target-task loss in the table**, so
the reduction is not bought with plasticity. Layer-freezing reaches a much lower
gap only by learning the task less, at target loss 0.924 against the shape run's
0.872, which is exactly the comparison matched plasticity is there to expose. Of the four remaining
named methods, SLoRA, two-phase continual instruction tuning, CLAIM and ArMA all
target the sequential continual-instruction setting, a stream of tasks rather than
the single fine-tuning we study, so a fair comparison means moving to their
benchmarks; App. I already lists continual learning among the paper's future
directions, and the revision cites all four methods there. Detail in pCi8 W3/W5,
G3T9 W3 and eQL6 W4+Q3.

------------------------------------------------------------------------------
### D + W-4b: free-running generation

[原文 D]: Free-running generation: The authors should either provide free-running
   generation experiments or clearly limit the claims to teacher-forced comparisons.
[原文 W-4b]: ... It remains unclear how well the method transfers to ... free-running
   generation ... (W-4 quoted in full under Point B.)

> **Point D (+ Weak Point 4).** Free-running experiments, or limit the claims to
> teacher forcing.

We ran it, and on the hardest case for a teacher-forced bound. GSM8K is the
generation-heaviest of the five benchmarks, so it is where trajectory shift can do
the most damage: each of the 12 variants greedily generates its own continuation,
averaging 76 tokens per prompt (72-81 across subsets), so its early mistakes
compound into its own later context, precisely the regime the autoregressive
corollary did not cover. Both models are then scored on those generated
trajectories. Five independent 100-prompt subsets (seeds 42-46):

| statistic (12 variants, mean ± sd over 5 subsets) | value |
|:--|--:|
| $r_s$(bound, gap), teacher-forced | +0.958 ± 0.011 |
| $r_s$(bound, gap), free-running | +0.947 ± 0.015 |
| rank agreement, teacher-forced vs free-running bound | +0.959 ± 0.010 |
| **cross: teacher-forced bound vs free-running gap** | **+0.958 ± 0.011** |

**The last row is the operational one: a practitioner ranks once on reference text
and the ordering carries over to self-generated output.** We also take the
alternative the point offers, since the two are not exclusive: Corollary 1 is
restated as teacher-forced-only, with trajectory-distribution shift named as an
explicit limitation. The bound itself applies to any feature rows (App. D), so the
restriction is a property of the protocol rather than of the theory, and the
protocol is chosen for what the instrument is for: a diagnostic has to be cheaper
than the benchmark it replaces, and teacher forcing is what keeps it to one
deterministic pass per variant, with no decoding and no sampling to control for.
We therefore claim the teacher-forced scope and report the free-running result as
evidence that the ordering survives it. Per-subset ranges in 8VrD W4+Q2.

------------------------------------------------------------------------------
### E + W-5b: failures, mixed results, and where the regularizer does not help

[原文 E]: Analyze failures and mixed results: The authors should discuss weaker
   correlations, such as those noted for GSM8K or Qwen-BBQ, and cases where the regularizer
   does not help or worsens forgetting.
[原文 W-5b]: ... The evaluation ... shows mixed results across settings, and does not
   clearly establish the tradeoff between reducing forgetting and preserving new-task
   performance. (W-5 quoted in full under Point C.)

> **Point E (+ Weak Point 5).** Weak correlations (GSM8K, Qwen-BBQ); cases where
> the regularizer does not help or worsens forgetting; the
> forgetting-versus-plasticity tradeoff.

All three parts are answered below, and the ingredients were already in the submitted
appendices (App. F.3's GSM8K quantification, App. G.1's low-signal mechanism, App. H's
gating validation): what the revision changes is their placement, promoting them to
the main text.

(i) The weak correlations. All 20 (model, fine-tuning task, benchmark) LoRA cells
move into the main text with their aggregate: **mean $r_s$ +0.71, median +0.93, and
18 of 20 cells positive.** The two flagged cells fail to rank for different,
now-stated reasons (Table 21 plus a per-checkpoint recomputation on the paper's
round):

| cell (Qwen3-8B, BBQ fine-tune) | $r_s$ | what the panel records |
|:--|--:|:--|
| TriviaQA | -0.66 | gap at the noise floor (0.0035 at lambda 0 vs 0.288 on ARC, no trend across checkpoints): nothing to rank |
| MMLU | -0.34 | gap real but non-monotone (peaks near step 75, partially recovers by 300) while accumulated drift grows: rank correlation structurally depressed |

**Ranking is uninformative there by construction, and the bound holds at every
checkpoint in both cells.** On
GSM8K under quantization the correlation is genuinely low (0.41 pooled across
families), and for a quantified reason: the mean gap there is about 0.019 nats
(Table 10), an order of magnitude below the other benchmarks, which leaves the least
signal and makes the cell the most sensitive to how the mismatch is attributed. That
sensitivity is measured, and the paper's design rule rests on it: at the analysis
default $W = I$ the head rotation stays inside the shape term and GSM8K ranks +0.51 on
Llama and +0.68 on Qwen, while the Procrustes alignment $W = W_N$ absorbs that
rotation so the head term carries the ordering and the same cells rank +0.97 and
+0.96, the largest gain of any benchmark in either family. **This is why we
recommend $W_N$ when the goal is an ordering and $W = I$ for axis analysis and the
regularizer**; both are certified, so the choice costs no validity. Under LoRA
fine-tuning, which does move GSM8K, the bound ranks it at +0.97 at $W = I$ too (Fig.
3, TruthfulQA row); the tested mitigation is in pCi8 W4.

(ii) Where the regularizer does not help, or hurts. Our own caption made this too easy to
misread, and the fault is ours. **Table 22 is a gating-validation table, not a results
table over four settings**: its rows are ordered by decreasing mean shape drift
$1-\bar{\Omega}$, the drift available to repair, and its last column records what
PRISM's diagnosis says to do:

| setting (ordered by $1-\bar{\Omega}$) | $1-\bar{\Omega}$ | trace effect | gating verdict |
|:--|--:|--:|:--|
| Llama TruthfulQA | 0.0937 | **-19.2%** | shape-driven, apply |
| Llama BBQ | 0.0678 | +8.6% | cell-level mixed |
| Qwen TruthfulQA | 0.0091 | +2.7% | at noise floor, skip |
| Qwen BBQ | 0.0011 | -0.2% | at noise floor, skip |

What makes the two Qwen rows a prediction rather than an excuse is that the column
deciding them is measured before any intervention: it is one minus the baseline
$\Omega$, and Qwen's baseline $\Omega$ is already 0.991 and 0.999, against Llama's
0.906 and 0.932, so there is essentially nothing to repair. Their +2.7% and -0.2%
then sit inside the +-3% band the appendix itself defines as neutral, on cells where
every method lands within 0.02 of every other. Three of the four settings therefore
match the prediction directly.

**Llama BBQ (+8.6%) is the one genuine exception**, and it is a condition-(ii) failure
rather than a shape-drift failure: condition (i), enough drift to repair, is satisfied
at 0.0678, but condition (ii), that the drift be accompanied by proportional risk-gap
growth, fails on ARC and MMLU, so the penalty has no target there. On the two
Llama-BBQ benchmarks where condition (ii) does hold, TriviaQA and GSM8K, the penalty
gives -88% and -79%. The honest reading is therefore granular: at the setting level,
where Table 22 applies the gate, this row does mislead; at the per-benchmark level,
where the two conditions are actually stated, the gate separates the cells correctly.
The revision promotes this analysis to the main text and presents the regularizer as
axis-targeted with its gating rule explicit, not as a universal win. Decomposed in
8VrD W3.

(iii) The forgetting-versus-plasticity tradeoff. This is why every baseline in
Point C is reported at the config closest to the shape run's target-task loss
rather than at its own best setting: it is the only comparison that separates a
method which forgets less from one which simply learns less. The table there shows
the shape penalty holding the lowest target-task loss while also forgetting least,
and layer-freezing achieving a lower gap purely by giving up plasticity.

------------------------------------------------------------------------------
Weak Points 2 and 6b, outside the five points

> **Weak Point 2.** Are the scale/shape/head axes actionable, or coarse descriptive
> categories? Some failure modes are already expected.
> **Weak Point 6b.** Clarity: notation used before definition, and unclear
> motivation for restricting alignment to orthogonal maps.
> (Empirical-versus-population half: Point A.)

Weak Point 2, actionability: each axis points at a distinct remediation, and for two
of the three the loop closes with a measurement rather than a suggestion. On the head
axis it closes protocol-side: PRISM attributes `Q6_K`'s SQuAD
degradation to the head axis ($\gamma = 75.77$ of $B = 76.95$, Qwen3-8B-Base), and the
protocol acting on that diagnosis (BnB INT8, head unquantized) removes it at a 20x
lower bound, which is a comparison across existing protocols rather than an
intervention we ran. On the shape axis it closes through training instead: the diagnosis says
shape dominates LoRA forgetting, and penalizing it directly (Eq. (8)) cuts the gap
from 0.843 to 0.680 (Point C). Per-channel smoothing for the scale axis is a
direction we name but do not evaluate, and eQL6 Q2 states that scope. Controlled
single-axis interventions confirm the attribution is causal (Llama-3.1-8B, $W = I$ default; max |term - control| within each family,
MMLU / TriviaQA):

| intervention family | scale term | shape term | head term |
|:--|--:|--:|--:|
| scale-only (final-norm rescale) | **1.9e4 / 2.0e4** | 6.9e-2 / 7.5e-2 | 0 / 0 |
| rotation-only (norm-preserving) | 1.3e-4 / 6.6e-6 | **7.7e2 / 7.9e2** | 0 / 0 |
| head-only (RTN `lm_head`) | 0 / 0 | 0 / 0 | **5.4e2 / 4.6e2** |

**Each family moves only its own term, with own-axis response at least 2.6e5x the
largest cross-axis leakage, and the bound holds in 26 of 26 configs (identity
control: $B$ = 0.00/0.08).** The terms also follow the laws the theory predicts,
scale as $(\alpha-1)^2$ and shape as $\theta^2$ to within about 1%, with $\gamma$
rising monotonically over head bit-width (10, 40, 210, 541 at 8/6/4/3 bits), and the
perturbations do real damage ($|\Delta\mathcal{R}|$ up to 1.39 nats at
$\alpha = 0.5$). So the axes are not labels attached after the fact: each one responds
only to its own perturbation and does so on the functional form the derivation gives it.

Weak Point 6b, clarity: a notation table, first-use definitions, and Section 5's key
results moved into the main text. The orthogonal restriction has a reason we should
have stated, namely identifiability: because the head is linear, features and head are
only defined jointly up to an invertible map, since $Z_P H_P$ and
$(Z_P A)(A^{-1} H_P)$ are the same model, so a discrepancy measured after an
unrestricted alignment can always be pushed through $A$ into the head and fitted away.
$O(d)$ is exactly the set of maps that leaves the geometry the head reads out
untouched, and it is also what makes Prop. 1's exact scale-shape split available,
since a general linear map carries scale and shear of its own. Detail: pCi8 W1,
eQL6 W1/W2+Q1.

------------------------------------------------------------------------------
What this adds up to, in the meta-review's own structure: (A) the guarantee is
stated on the quantity we measure, with a concentration corollary back to the
population version, and a per-cell calibration turns the bound into a working
threshold; (B) the reference requirement is eight to a few hundred sequences of
the task's own data, at 1/62.6 of one benchmark run, the regularizer's own reference
moves retention by at most 5.9 points across a size-and-draw grid, and unpaired use
is scoped out;
(C) the bound ties CKA/SVCCA on identical features while carrying a certified
bound, an axis attribution and a trainable penalty they cannot, and that penalty
leads four forgetting baselines at matched plasticity; (D) the ranking survives
free-running generation, and the corollary is restated to its teacher-forced
scope; (E) the flagged cells are explained by stated mechanisms and the one
genuine gating exception is owned. Every negative result in this rebuttal, the
failed span mitigation, the two low-signal cells, and the Llama-BBQ exception, is
reported by us rather than left to be found, and much of the underlying analysis
was already in the submitted appendices. We ask the AC to reconsider in this
light, and would welcome completing any point still judged partial during the
discussion. We would especially value the AC's view on whether the explicitly
input-conditioned scope resolves Point B, given that benchmark-independent
transfer remains outside the claim.

Independent of the outcome, this cycle has made the paper markedly better: the
empirical restatement, the per-cell calibration, and the matched-plasticity
protocol each began as a point raised in these reviews. We are grateful for the
care behind them.

================================================================================

Reviewer #1, pCi8 (3: Borderline reject, conf 4), Q3/C3/S2/O3
================================================================================

···· ORIGINAL REVIEW — verbatim from OpenReview (for 一一對應檢查; internal, DO NOT POST) ····
[Summary]
PRISM is a diagnostic tool for post-training LLM variants — quantized, LoRA fine-tuned, or
distilled. The core problem it solves is pretty simple: when you quantize a model, you want
to know how much you broke it and where, without running a full benchmark suite. Existing
similarity metrics like CKA and SVCCA can tell you that a variant has drifted from the base,
but not why or which component is responsible. PRISM derives a closed-form upper bound on
the cross-entropy risk gap between a base model and its variant, decomposed into three
independently measurable axes: scale mismatch (activation magnitude collapse), shape
mismatch (distortion of the relative geometry between token representations), and head
divergence (the prediction head itself drifting). The key technical insight is that the
linear lm_head plus near-isometric backbone structure of modern LLMs lets you express this
bound tightly — avoiding the naive Lipschitz constant that would blow up with vocabulary
size — and the Procrustes residual gives you an exact scale/shape decomposition for free.
Empirically it ranks PTQ variants with Spearman rs = 0.820 and LoRA forgetting checkpoints
at rs = 0.831 across two model families and five benchmarks, using only a single forward
pass per model with no labels. The shape term is also differentiable, so it doubles as a
regularizer during LoRA fine-tuning that outperforms experience replay at mitigating
catastrophic forgetting.

[Strengths]
- Exact decomposition. The scale/shape split is an exact identity, not an approximation.
  The math is clean and the derivation is self-contained.
- Tighter Lipschitz constant. The simplex polarization trick is a real contribution — the
  naive spectral bound scales with vocabulary size and becomes useless at V~10⁵, this one
  doesn't.
- Three axes are actually distinct. Empirically they do capture different failure modes
  rather than just being correlated proxies of the same thing.
- Low compute cost. One forward pass per model, no labels needed. Useful in practice when
  you're choosing between a dozen quantization variants and don't want to run full
  benchmarks on all of them.
- Shape regularizer is a natural byproduct. You get a differentiable training objective for
  free from the bound, and it actually outperforms replay in the experiments.
···· END ORIGINAL REVIEW ····

We thank the reviewer for a careful and specific review. The strengths already
credit the two technical cores (the simplex-polarization constant and the exact
scale/shape identity); the responses below build the practical-significance case
on them, with a measurement for each weakness rather than a restatement: the
contribution restated as commensurable per-variant attribution (W1); the
looseness located and then calibrated, so the bound answers "is Q4 actually good
enough" directly, Q4_K_M predicting 0.055 nats against 0.036 measured, at 1/62.6
the measured cost of one benchmark run on the same prompts (W2); the mixed head
protocols behind the small Table 3 delta, with the GGUF-only correlation at
to 0.943 and 8 reference sequences already ranking as 512 do (W3); GSM8K kept as
an honest limitation, including a mitigation we tested and report as failing
(W4); EWC, L2-SP and layer-freezing added, the shape penalty leading in all
three seeds at matched plasticity (W5); and near-isometry scoped to tightness
alone, measured at under a fifth of the residual even at 2 bits (W6). We address
each in turn:

> **W1.** The three findings were already known; the contribution is unification, not discovery.

We are grateful that the reviewer identifies the simplex-polarization Lipschitz
constant and the exact scale/shape identity as real contributions: those are the
technical core, and neither existed before this paper. Building on them, and
agreeing with the reviewer's framing, our claimed contribution is not the discovery
of the three phenomena but the **first quantitative instrument that makes them
commensurable: all three axes are expressed in the same units, namely each one's
contribution to a certified CE-risk-gap bound.** Prior work could say that low-bit
quantization distorts feature geometry and that LoRA shifts activation scale, but
not, for one given variant, *which* axis accounts for its degradation and by how
much. That common scale is what did not exist.

Three things follow: per-variant axis attribution mapped to a remediation
direction, ranking from a single teacher-forced pass over a small slice of the
task's own validation data, and a differentiable training objective (W2, W3, W5).

Sec. 5.3's Q6_K case is that statement in practice: on Qwen3-8B-Base/SQuAD (Table
12) the head axis alone accounts for 75.77 of a bound of 76.95, so the backbone
geometry is demonstrably not what to fix there. Taking the reviewer's point, the revision states
the contribution in the Introduction as this commensurable, per-variant attribution
across the three axes, rather than as the discovery of the phenomena themselves.

> **W2.** Loose by orders of magnitude and calibrated only for ranking: it says Q2
> is worse than Q4, not whether Q4 is good enough. Benchmarks still needed for that.

We agree the bound is loose, and that is exactly why the paper claims an ordering
rather than a value. Rather than defend the number, we measured where the looseness
sits. Slack, the ratio of the bound to the risk gap it bounds, has a median of
1597x over the 570 (family, benchmark, variant) cells of the paper's five
benchmarks (near-zero-gap rows excluded), and it separates into two measured steps:
replacing the worst-case Lipschitz constant with the largest sensitivity actually
observed accounts for at most ~2.5x; **the remainder, the alignment residual plus
the triangle and Jensen steps, which this analysis does not further separate,
accounts for the remaining 140x to 3000x and is the dominant source.** Slack
also narrows where degradation is real: from ~4500x at Q8_0 to ~600x at Q2_K.

**What matters for the ranking claim is that those constants are fixed within a
(model, benchmark) cell.** K_feat depends only on the target model's head, K_pred
<= sqrt(2) universally (App. A.5), and the prefactor rho_T rho_P varies across the
variants of one cell with a median coefficient of variation of 0.63%. The K
constants are exactly shared, the prefactor nearly so, and a factor this close to
common cannot materially reorder a cell, which is why the paper reports
within-cell rank correlations rather than absolute values; what does vary per
variant is the alignment residual and the remainder slack, which is why the
correlations sit near 0.9 rather than 1.0 and why no single multiplicative
correction exists.

To test the reviewer's own question, whether Q4 is "actually good enough" and not
only better than Q2, we fit a monotone map from bound to measured gap inside each
cell, holding out one variant at a time. **Q4_K_M on Llama-MMLU maps to a predicted 0.055 nats against a measured
0.036, both under the 0.1-nat tolerance**; over the 55 (model, benchmark) cells the
held-out error runs 33% below predicting the cell mean, and read as an
accept/reject rule at 0.1 nats, precision is 0.75 on this cell and 0.8 or better in
49 of the 55. The map is fitted on already-measured variants of the same cell, so
this amortizes benchmarking rather than replacing it: once a few variants in a cell
are measured, the bound flags which of the rest are safe.

So the last word does come from a benchmark run, as the reviewer says. What changes
is the cost of getting there, and the reason is simply that PRISM reads text under
**teacher forcing** while a benchmark must **generate** it. On the same 256 GSM8K
prompts, one RTX 5090, model load excluded from both sides:

| pipeline, same 256 GSM8K prompts | what it actually runs | compute |
|:--|:--|--:|
| PRISM | one teacher-forced pass, no decoding, no grading | **8.9 s** |
| benchmark, greedy decode | ~209 generated tokens per prompt, then parse and score | 556.7 s (**62.6x**) |
| benchmark, self-consistency maj@8 | ~eight times the above | 4453.6 s (501x) |

The gap is structural, growing with generation length, so a practitioner can screen
every candidate under teacher forcing and spend the decoding budget only on the
finalists.

> **W3.** Table 3 shows only a small gap between Omega alone and the full bound: a
> lot of machinery for a modest ranking gain.

The +0.016 is real, and it is small for an identifiable reason: **the variants
pooled in Table 3 do not share a head protocol.** At W = I the six GGUF k-quant
tiers quantize lm_head, so their head term is nonzero, while the GPTQ and
BitsAndBytes variants keep lm_head in FP16, so their head term is zero by
construction (the paper notes this pooling effect at L282-285). Averaging one
metric over both kinds necessarily understates what the head arm contributes, and
the pooling explains the size of the delta: restricted to the GGUF tiers alone
(the only regime where the head arm engages under a common protocol), the
identity bound's mean correlation on Llama is **0.943**, against 0.828 pooled.
Because the variant subset changes with the restriction, we read this as a
protocol-controlled robustness check rather than a per-component gain. The W = I
/ W_N split itself is the paper's prescribed design
choice (Sec. 3.3): W_N absorbs the head rotation that W = I leaves in place, so we
recommend W_N whenever the goal is a correct ordering, and W = I for analysis and
the regularizer. We keep the mixed W = I pool as the headline because that is the
family a practitioner choosing among backends faces: the paper offers a diagnostic
instrument, not a leaderboard number.

**And the ranking delta is the smallest of what the decomposition produces: its
other outputs never appear in a Spearman at all.** (i) The three axes each map to a
distinct remediation, so the bound says which one to fix rather than only that a
variant drifted; W1 gives the concrete case. (ii) The shape term is differentiable,
which turns the diagnosis into a trace-norm penalty that leads every task-keeping
baseline in every seed (W5). (iii) The whole measurement runs under teacher
forcing, 62.6x cheaper than one benchmark evaluation on the same prompts (W2). A
similarity score supplies none of these, and none of CKA/SVCCA/Procrustes bounds
the risk gap at all.

The bound is also frugal where it matters operationally: it needs very little
reference data. Three fresh seeds, 12 Llama variants, 5 benchmarks, with the
ground-truth gap held at the full 512 sequences:

| reference slice | bound vs full-slice gap (r_s) | seed agreement on the ordering |
|:--|--:|--:|
| 8 sequences | **+0.932 ± 0.016** | 0.981 |
| 32 | +0.931 ± 0.009 | 0.992 |
| 128 | +0.933 ± 0.011 | 0.993 |
| 512 (paper size) | +0.932 ± 0.011 | 0.998 |

**Eight sequences already rank the variants against the full-slice risk as well as
512 do**, and the seeds agree with one another on that ordering at 0.981 with eight
sequences, rising to 0.998 at 512. The reference requirement is therefore both
small and stable.

> **W4.** GSM8K's correlation is far lower than every other benchmark, and the long
> chain-of-thought explanation makes it structural rather than an edge case, more
> relevant as models move toward reasoning.

We agree, and the paper already documents it as a first-class
limitation. App. F.3 reports GSM8K as the weakest benchmark (r_s about 0.41 across
families) and quantifies the reason: long teacher-forced chain-of-thought spans
dilute per-token loss, so quantization moves GSM8K's mean gap by only about 0.019
nats against 0.07 to 0.16 elsewhere (Table 10). **That is a property of the
cross-entropy target itself, inherited by any CE-based proxy including plain
perplexity screening**: at gaps this small, per-variant differences approach
measurement noise, and the low correlation reflects how little degradation there is
left to rank.

What we can separate further is how much of that number is the score side rather
than the task side. Recomputed under the Procrustes alignment on the paper's own
round, on Llama:

| Llama-3.1-8B | 1-Omega_I | B_I | 1-Omega_N | B_N |
|:--|--:|--:|--:|--:|
| GSM8K | 0.480 | **0.510** | 0.480 | **0.965** |
| other four (range) | 0.790-0.965 | 0.811-0.972 | 0.790-0.979 | 0.790-0.979 |

On GSM8K the shape term is saturated (Omega = 1.0 to float precision for 11 of the
12 variants, so its ordering is mostly ties), and at W = I the head term is exactly
zero for every variant that keeps lm_head in FP16, so B_I has almost nothing left
to order with (0.510). The feature arm delta is identical under both alignments
(r_s +0.776), so the entire B_N gain is the head arm, which stays nonzero at W_N
because it also measures how far the backbone rotated. The ordering signal
therefore survives under W_N (0.965, the arm we recommend for ranking; W3), but the
measured gaps behind it still average only ~0.03 nats on this Llama cell (0.019
across families), so we do not claim this
makes the absolute drift readable. Provenance note: this is a recomputation on the
paper's round; our fresh float64 round computed only the W_N arm.

We also tested the natural mitigation and report that it does not work: restricting
features and losses to the final-answer span ("####" onward, ~3 tokens) leaves the
ordering unreliable (r_s +0.57 on Llama; per-sample losses turn near-binary, and
the strongest-degradation variant Q2_K even inverts). So we claim no fix; the
revision states the limitation as structural for any low-signal CE target.

What the data does support is placing the weakness on perturbation size rather than
on reasoning per se: under LoRA fine-tuning on TruthfulQA, which does move GSM8K
(mean gap 0.134 on the no-regularization run, Table 2), PRISM ranks its forgetting
at r_s = +0.97 (Fig. 3, TruthfulQA row), while the BBQ row's weaker GSM8K
correlation (+0.48) sits in the same small-gap regime (App. H). The boundary is
therefore the magnitude of the risk gap, which the revision states explicitly:
PRISM's ordering is reliable when a task's gap is meaningfully nonzero, and GSM8K
under PTQ sits below that. We thank the reviewer for pressing on this; the revision
adds the discussion.

> **W5.** EWC is the standard forgetting baseline, cited but never compared against;
> comparing only to small-scale replay is convenient.

Agreed. We added EWC (diagonal Fisher), L2-SP and layer-freezing, the meta-review's
suggestion, all under the identical protocol (LoRA, lr 1e-5), and **each baseline
is reported at the sweep config whose target-task loss comes closest to the shape
run's (matched plasticity)**.
Downstream forgetting is the mean risk gap over the five held-out benchmarks:

| method (TruthfulQA, 3 seeds) | downstream forgetting | target-task loss |
|:--|--:|--:|
| no regularization | 0.815 ± 0.042 | 0.950 ± 0.008 |
| replay, lambda 0.01 | 0.771 ± 0.027 | 0.899 ± 0.007 |
| L2-SP, lambda 0.01 | 0.763 ± 0.019 | 0.901 ± 0.003 |
| EWC, lambda 0.1 | 0.751 ± 0.015 | 0.905 ± 0.010 |
| **shape penalty, lambda 1** | **0.680 ± 0.016** | **0.872 ± 0.005** |
| layer-freeze, top 16 | 0.404 ± 0.017 | 0.924 ± 0.001 |

**The shape penalty forgets least among the methods that learn the task to a
comparable level, beats each of them in all three seeds by +0.050 to +0.157, and
carries the lowest target-task loss in the table.** Layer-freezing's much lower
gap comes from learning the task less (0.924 against 0.872), which is exactly
what matched plasticity is there to expose.

On why replay was the original choice: it is the controlled comparison, same 32
reference sequences, same schedule, same compute as the trace penalty (Sec. 5.4).
This remains a feasibility demonstration that the diagnosed axis is actionable, not
a claim to beat every method.

> **W6.** Does near-isometry still hold after aggressive quantization, when feature
> geometry is already severely distorted? Two citations are given, with no
> discussion.

The alignment affects only how *loose* the bound is, never whether it holds.
**Theorem 1 is a family of bounds indexed by W, and the only condition is that W be
orthogonal: every W in O(d) gives a valid inequality (Thm. 1, L132; proof App.
C.1).** An orthogonal W is a rotation or a reflection and nothing more, so it
re-orients the proxy's feature cloud against the target's without changing a single
length, distance or angle inside it. **The alignment is an exact isometry by
construction; what is empirical is only how well such a map can be made to match
the two clouds, and that governs tightness alone.** Near-isometry is therefore not
a precondition of the setup; the revision makes this distinction explicit where the
paper cites [12, 13].

On the reviewer's empirical question, we measured what the restriction costs
exactly where geometry is most distorted: in the top-r principal subspace the
diagnostic reads (r <= 256, 77-90% explained variance), lifting the restriction to
an unrestricted linear map shrinks the feature residual by under a fifth on
average even at 2 bits (mean 18.3% for Q2_K), and by at most 0.29 in any single
cell (Llama-3.1-8B, MMLU and SQuAD). Within the measured high-variance subspace,
the restriction is therefore cheap as well as optional. Which W to use is, in the paper's own words,
"primarily design-driven" (L145-152): W = I for analysis and the regularizer, W_N
when the goal is ordering (W3); both are certified, so the choice costs nothing in
validity.

**In sum:** the three answers share one shape. We locate the looseness instead of
disputing it: two of its three sources are cell constants, so they cannot reorder,
and a within-cell calibration turns what remains into nats (W2). On identical
features the bound ranks as well as CKA or SVCCA while carrying a valid upper bound,
an axis attribution and a trainable objective that they cannot (W3). And the
near-isometry assumption is measured rather than assumed: it affects only tightness,
by under a fifth of the residual even at 2 bits (W6). We hope these speak to the
practical-significance concern. May we ask whether the calibrated screening rule
and the measured cost gap address that concern? We are glad to engage further on
any point that remains.

================================================================================
Reviewer #2, G3T9 (3: Borderline reject, conf 4), Q3/C3/S2/O3
================================================================================

···· ORIGINAL REVIEW — verbatim from OpenReview (for 一一對應檢查; internal, DO NOT POST) ····
[Summary]
This paper proposes PRISM, a framework for diagnosing performance degradation for
post-trained LLM models. The key idea is to compare the hidden representations and output
heads of a base model and its finetuned model, and to decompose the resulting cross-entropy
risk gap into three interpretable components: scale mismatch, shape mismatch, and head
divergence. The authors derive a closed-form upper bound on the risk difference and show
that the resulting PRISM score correlates with empirical degradation across quantization
and LoRA fine-tuning settings. Beyond post-hoc diagnosis, the paper further uses the
differentiable shape component as a regularizer during frozen-head LoRA training, aiming to
mitigate catastrophic forgetting without relying on experience replay.

[Strengths]
- The paper addresses how to how degradation from post-training can be explained in theory,
  rather than merely detecting that their performance has dropped. The proposed
  decomposition into scale, shape, and head components is intuitive and provides a more
  diagnostic view than single-number representation similarity metrics such as CKA or SVCCA.
- A key strength is the connection between representation geometry and cross-entropy risk.
  It gives a principled justification for using geometric drift as a proxy for model
  degradation. The decomposition also helps localize different failure modes, such as
  representation distortion in low-bit quantization or head divergence in GGUF-style
  quantization.
- The empirical evaluation is reasonably broad, covering multiple model families,
  quantization backends, bit-widths, and LoRA forgetting scenarios. The rank correlations
  reported for both PTQ and LoRA settings suggest that PRISM is useful as a model-ranking
  and diagnostic tool.
···· END ORIGINAL REVIEW ····

We thank the reviewer for pressing on cost and actionability, the two questions
that most sharpen the paper. The four-part cost question in particular (forward
passes, reference data, realistic comparison) is the checklist a practitioner
would actually run, and answering it item by item is what produced the measured
cost table below. Summary of what we added: a measured wall-clock cost
table (one greedy decode 556.7 s vs PRISM's 8.9 s teacher-forced pass on the same
GSM8K prompts, 62.6x) and controlled single-axis interventions showing each axis
is causally actionable (W1); a reference-set ablation on three fresh seeds, an
explicit input-conditioned scope statement, and a free-running experiment (W2); a
same-features CKA/SVCCA comparison and weight-space plus architectural
regularizer baselines (W3); a per-cell calibration that turns the loose bound
into an operational threshold (W4); and scope-plus-evidence answers on full
fine-tuning, RLHF, and multi-task SFT (Q1, Q2).

1) W1 [原文]: The motivation for the proposed framework is not fully convincing to me. In
   many practical settings, one could simply evaluate the post-training models on the
   relevant benchmark suites. The paper argues that PRISM provides a more diagnostic
   alternative, but it is not entirely clear why decomposing the degradation into scale,
   shape, and head terms provides a sufficiently detailed explanation of the failure. These
   axes are interpretable at a high level, but they still seem relatively coarse and do not
   necessarily identify actionable root causes beyond another layer of descriptive analysis.
   If the intended advantage is computational efficiency over direct evaluation, the paper
   should provide a more careful cost analysis, including how many forward passes are
   required, what reference data are used, and how the cost compares to standard benchmark
   evaluation under realistic settings.

> **W1.** Why not just benchmark? Are the axes actionable beyond description?
> Cost analysis: forward passes, reference data, realistic comparison with
> standard benchmark evaluation.

**The use case first.** PRISM is a screening tool for variant decay, not a
replacement for evaluation. A concrete instance from our own grid (Sec. 5.1):
Llama-3.1-8B, Qwen3-8B and Ministral-3-8B each ship a dozen quantized variants
across GGUF, GPTQ and BitsAndBytes at bit-widths 2-8, and the deployment question
under a fixed GPU budget is how low each family's quantization can go on the
tasks that matter: Q4, or all the way down to Q2? Benchmarking every (family,
variant, task) cell answers this by exhaustion, hundreds of decode-and-grade runs
at a measured 62.6x the compute of PRISM's teacher-forced pass each (item 1),
each returning a single risk number with no reason attached. PRISM screens the
same grid with one forward pass per variant, returns the within-cell ordering
(r_s 0.82 PTQ / 0.83 LoRA), attributes each failure to scale, shape or head with
a mapped fix (item 2), and, once a few variants of a cell are benchmarked,
calibrates the rest into predicted gaps (W4: Q4_K_M on Llama-MMLU predicts 0.055
nats against 0.036 measured, so Q4 clears a 0.1-nat bar there). The decode budget
then goes to the one or two finalists. Where no public benchmark exists at all
(proprietary or safety-sensitive deployments) PRISM still applies, needing
reference text rather than a scorer; absent even reference answers, the models'
own continuations serve (r_s +0.947, W2).

(1) Cost analysis, itemized as asked; model loading, a one-time cost both sides
pay, is excluded throughout.

**Forward passes.** One teacher-forced pass per model over the reference set;
ranking K variants costs K+1 passes, since the base is extracted once and reused.
A benchmark run instead needs O(generated tokens) autoregressive passes per
prompt: about 209 per prompt on our measured GSM8K run below.

**Reference data.** Each reference sequence is one held-out validation item of the
task being diagnosed, read as its prompt followed by its reference answer, with
the features and the per-token CE both taken from that answer region (Sec. 5.1,
L198). We use 512 sequences per benchmark for PTQ and 256 for LoRA forgetting, a
slice any evaluation already has, and 8 sequences already give the same variant
ranking as 512 (W2).
**PRISM never parses, matches or scores those tokens, because its target quantity
is the CE gap between two models on identical tokens, in which correctness plays
no part. What it removes is the decoding and the grading, not the validation
data.**

**Compute**, measured on identical prompts (256 GSM8K prompts, one RTX 5090, two
independent runs agreeing to 0.02%):

| pipeline (256 GSM8K prompts) | forward passes | decodes? | parses + grades? | compute (s) | x vs PRISM |
|---|---|---|---|---|---|
| PRISM, teacher-forced (full)      | 1 TF over the prompts | no  | no  | 8.9    | 1x    |
| PRISM, screening (32 prompts)     | 1 TF over the prompts | no  | no  | ~1.1   | -     |
| Benchmark, greedy decode          | O(tokens) AR / prompt | yes | yes | 556.7  | 62.6x |
| Benchmark, self-consistency maj@8 | 8x decode (sampled) | yes | yes | 4453.6 | 501x  |

[a] Compute only, same 256 prompts, model load excluded on both sides; GSM8K is the
    generation-heaviest benchmark, so this is the high end. The 62.6x greedy figure
    is the headline; maj@8 is the optional upper end, and the 32-prompt screening
    row is a reference-size data point, not a term in the multiple.

**So one greedy decode costs a measured 62.6x PRISM's teacher-forced pass on the
same prompts, before any output parsing or scoring, and the gap is structural: it
widens with generation length.**

(2) The axes are actionable, and causally so. The assessment already credits
that the decomposition "helps localize different failure modes", so the open
doubts are coarseness and description. On coarseness, each axis maps one-to-one to
a distinct remediation: scale to per-channel outlier smoothing, shape to
Hessian-aware reconstruction (PTQ) or trace regularization (training), head to
keeping lm_head in FP16 (Sec. 5.3). On description, that map closes into a
diagnose, act, verify loop: PRISM attributes Q6_K SQuAD degradation to the head
axis (the head term gamma = 75.77 of B = 76.95, Qwen3-8B-Base), the
protocol acting on that diagnosis (BnB INT8, head unquantized) removes it, and the
fix is verified at B = 3.81, a 20x reduction. **For these constructed single-axis
perturbations the attribution is causal rather than descriptive: on Llama-3.1-8B,
final-norm rescaling, norm-preserving rotation and lm_head-only RTN each move only
their own term under the W = I default, with own-axis response at least 2.6e5x
the largest cross-axis leakage, and the bound holds in 26 of 26 cells.**

**In sum:** one teacher-forced pass per variant at 1/62.6 of a greedy-decode run,
plus a causal axis attribution that a benchmark score cannot give.

2) W2 [原文]: The empirical evaluation appears incomplete in several important dimensions.
   The experiments rely on teacher-forced feature extraction, but it remains unclear whether
   the same conclusions hold in free-running generation. In addition, the PRISM bound seems
   to be computed using benchmark inputs themselves, so it is unclear how well the method
   works with a small benchmark-independent reference set. Additionally, an ablation for the
   number and type of reference samples would be important to understand the sample
   efficiency and robustness of the method.

> **W2.** Three gaps: teacher-forced extraction only, so does it hold under
> free-running generation? The bound is computed on the benchmark inputs, so how
> would a small benchmark-independent reference set do? And an ablation over the
> number and type of reference samples.

All three dimensions get a direct treatment: a free-running experiment (r_s
+0.947 ± 0.015 against +0.958 ± 0.011 teacher-forced, item 3), a reference-sample
ablation on three fresh seeds (8 sequences already rank the variants as 512 do,
r_s 0.932 either way, item 2), and, on benchmark-independence, a precise
statement of what the bound licenses instead of an oversold claim (item 1).

(1) Benchmark reference: what the bound licenses. The bound is
input-conditioned: its guarantee holds when the feature and the risk are
**paired** on the same data (a per-instance bound, App. A/D). **Computing the
bound on generic text to predict a benchmark's degradation would unpair them, a
distribution-transfer question the derivation does not cover, so we do not claim
benchmark-independent use**; the revision states this scope in the Limitations.
What the diagnostic does need is deliberately minimal: a small slice of the
diagnosed task's own validation items, each read once as its prompt plus
reference answer (features and CE from the answer region, Sec. 5.1; itemized
under W1); nothing is parsed, matched against gold, or scored for accuracy, and
in this default protocol nothing is generated either.

(2) Reference-sample ablation, re-run for this rebuttal on the diagnosed task's own data:
three new seeds (43/44/45), sizes 8/32/128/512, 12 Llama variants, 5 benchmarks,
720 cells. The decision-relevant question is whether a small slice orders the
variants the way the full slice does, so we correlate the score computed at size n
against the risk gap measured at n=512 for the same seed (the sizes are nested, so
the small slice is a subset of the target set).

| reference size | 1-CKA | 1-SVCCA | PRISM B_N |
|:--|--:|--:|--:|
| 8   | 0.903 ± 0.015 | 0.083 ± 0.036 | 0.932 ± 0.016 |
| 32  | 0.932 ± 0.010 | 0.821 ± 0.100 | 0.931 ± 0.009 |
| 128 | 0.934 ± 0.013 | 0.894 ± 0.001 | 0.933 ± 0.011 |
| 512 | 0.931 ± 0.008 | 0.941 ± 0.014 | 0.932 ± 0.011 |

**Eight sequences already rank the variants as well as 512 do (0.932 either way),
and at a fixed size the ordering is seed-stable: the pairwise Spearman between the
three seeds' own orderings is 0.981 at n=8, rising to 0.998 at n=512.** What a small
slice does not support is measuring the gap on that same slice: against its own n=8
gap the correlation is only 0.768, so a small slice is enough to order variants, not
to quantify them. SVCCA is the one that breaks, for a reason worth stating: canonical
correlation is invariant to invertible maps inside each subspace, so when the slice
has fewer rows than the feature dimension the two subspaces align perfectly and every
variant scores the same. At n=8, 1-SVCCA is numerically zero on arc, mmlu, squad and
triviaqa (median -1.2e-06), separating no variant at all; the exception is GSM8K, whose
long answers give ample rows, where SVCCA works at every size. CKA degrades
gracefully, and the bound does not degrade. On the type of the samples: the
reference is the diagnosed task's own validation split by construction, so size
is the free parameter this ablation varies, and a different data type is not a
knob but a different guarantee, the scope stated in item (1).

(3) Free-running generation. We test on GSM8K specifically because it is the
generation-heaviest, reasoning-heavy benchmark where trajectory shift is most
severe, the hardest case for a teacher-forced bound. On a 100-prompt subset each
of the 12 variants greedily generates its own continuations (mean 76
tokens/prompt across subsets) and both models are scored on those trajectories: across 5 subsets
(seeds 42-46), rs = +0.947 ± 0.015 free-run vs +0.958 ± 0.011 teacher-forced, rank
agreement +0.959 ± 0.010. This mode does decode, so it is the answer-availability
fallback rather than the cheap path. Independently we restate Corollary 1 as
teacher-forced-only and add trajectory shift as an explicit limitation (the bound
applies to any feature rows, App. D; the restriction is protocol, not theory).

**In sum:** the reference requirement is a small slice of the task's own
validation data (8 sequences already fix the ordering), the protocol extends to
free-running generation at r_s +0.947, and the one thing we do not claim,
benchmark-independent transfer, is now stated as explicit scope.

3) W3 [原文]: The comparison to existing baselines is also limited. Since PRISM is
   positioned as an improvement over existing representation similarity methods such as CKA
   and SVCCA, the paper should more directly compare against these metrics in the same
   ranking and diagnostic settings. Similarly, for the regularization experiments, it would
   be useful to compare the proposed shape regularizer against simpler feature-preserving
   regularizers or other representation regularization methods, not only experience replay.

> **W3.** Compare directly against CKA and SVCCA in the same ranking and diagnostic
> settings, and compare the shape regularizer against simpler feature-preserving
> regularizers, not only replay.

Agreed on both, and both now exist.

(1) Ranking, on identical features: linear-CKA, SVCCA and our feature arm
(exactly the Procrustes distance) computed on the same re-extracted features,
same variants, same |dR| targets (Llama/Qwen, the full variant sets).
Mean Spearman r_s over the five benchmarks:

| score (identical re-extracted features) | Llama | Qwen3 | mean r_s |
|:--|--:|--:|--:|
| 1-CKA | +0.924 | +0.882 | +0.903 |
| 1-SVCCA | +0.924 | +0.878 | +0.901 |
| our feature arm delta_N (= the Procrustes distance) | +0.930 | +0.873 | +0.901 |

This is a statistical dead heat and we say so rather than claim a win: a paired
bootstrap over variants (5000 resamples) puts our feature arm at -0.002 versus
1-CKA (95% CI [-0.038, +0.011]) and -0.000 versus 1-SVCCA ([-0.014, +0.019]). The
heat is robust three ways: the largest per-cell gap between our feature arm and
either score is 0.036 over the ten (family, benchmark) cells; dropping any one
benchmark moves the pooled paired difference by at most 0.006; and two
extractions differing 3.2x in token budget (Llama; 3.9x on Qwen) reproduce every
cell to four decimals.
One provenance disclosure: the paper's Table 3 W_N block reads as a cumulative
climb (Omega_N +0.806 -> B_N +0.912), but re-run with float64 accumulation and no
omega clamp, on Llama all five scores land within 0.014 of one another, and three
quarters of the paper's W_N advantage traces to GSM8K, where the shape term sits
at the numerical floor and the head term carries the order. We do not lean on
that climb; the same-features tie above is the comparison we stand on.

They tie because they read the same geometry: delta_N^2 = (rho_T - rho_P)^2 +
2 rho_T rho_P (1 - Omega_N) (App. A.1), and CKA/SVCCA are rotation-invariant
readings of the same agreement; CKA relates to our shape core only via the
inequality CKA >= Omega_F^2 and is not alignment-derived, so it is not
substitutable into the residual (App. B). Our contribution was never the
Procrustes distance itself but (a) its explicit split into a scale arm and a
shape arm, which Sec. 4 shows dominate under different lifecycle settings, and
(b) its lifting to a functional-risk bound (Theorem 1). That structure buys what
a similarity score cannot give: a certified bound with axis attribution (e.g.
B_I on extraction-QA, SQuAD +0.811/+0.955), the Sec. 5.3 diagnoses, and
robustness to reference size: on the
three-seed size run (table under W2), **B_N holds at r_s 0.932 from 8 sequences
up while SVCCA collapses to 0.083 at n=8**, so the parity above is measured where
SVCCA is at its best.

(2) Regularizers. On the "simpler feature-preserving regularizer" first: the
simplest one, an L2 match of reference features, decomposes by Prop. 1 into
exactly our scale arm plus our shape arm, and under frozen-head LoRA the scale
arm is nearly inert (rho_0 ~ rho_t; Table 2 caption), so in this regime it nearly
coincides with the shape penalty rather than forming a distinct family. The
genuinely distinct families are therefore weight-space (L2-SP, EWC) and
architectural
(layer-freezing), which we add, all protocol-identical, each at the sweep config
closest to the shape run's target loss. On TruthfulQA over three seeds
(42/43/44, mean ± sd), downstream
forgetting is 0.680 ± 0.016 for the shape penalty at lambda 1 against 0.751 ± 0.015
(EWC), 0.763 ± 0.019 (L2-SP), 0.771 ± 0.027 (replay) and 0.815 ± 0.042 (no-reg); the
shape penalty wins in all three seeds individually and carries the lowest target-task
loss (0.872) of any method here, so it is not trading plasticity. Layer-freezing
reaches 0.404 (top 16) only at target loss 0.924, i.e. by trading plasticity for
retention.

4) W4 [原文]: Finally, the proposed risk bound appears extremely loose in absolute scale,
   as shown in Table 1. While the paper demonstrates that the bound has good rank
   correlation with empirical degradation, this mainly supports its use as a heuristic
   ranking score, not necessarily as a practically meaningful upper bound.

> **W4.** Extremely loose in absolute scale (Table 1): good rank correlation
> supports a heuristic ranking score, not a practically meaningful upper bound.

The raw bound is indeed loose in absolute scale; we quantify it rather than
dispute it: slack B/|dR| median 1597x, IQR [673, 3966] over 570 cells (the
paper's five benchmarks), non-constant (log10 sd ~0.76), hence ranking-only for
uncalibrated B (as the paper scopes: Abstract, Sec. 6). To move beyond a
heuristic: (i) Theorem 1 is restated as an empirical-risk-gap bound with a
concentration corollary to the population version (a guarantee, precisely
scoped); (ii) **a one-time per-(family, benchmark) isotonic calibration turns B
into a predictor of |dR| with a leave-one-out error of 0.055 nats, i.e. the error
on a variant the map was not fitted on, and precision >= 0.8 (at
epsilon = 0.1 nats) in 49/55 cells** (an operational threshold, not just an
ordering; concretely, Q4_K_M on Llama-MMLU predicts 0.055 nats against 0.036
measured, both under the 0.1-nat tolerance). Slack attribution shows the Lipschitz
step contributes only 0.13-0.40 dex, and slack tightens at low bit-widths where
degradation is real.

5) Q1 [原文]: Can PRISM be applied beyond PTQ and frozen-head LoRA, such as to full
   fine-tuning or RLHF/RL-trained models? Since these settings may change both the backbone
   and the prediction head more substantially, it would be helpful to clarify whether the
   proposed decomposition remains meaningful when the variant is much farther from the base
   model.

> **Q1.** Beyond PTQ and frozen-head LoRA: full fine-tuning or RLHF, where both
> backbone and head change and the variant sits much farther from the base?

Mathematically the decomposition survives unchanged. Theorem 1 needs only a
matched architecture, feature dimension, and output vocabulary, and is agnostic
to how the variant was produced, so full fine-tuning and RLHF fall under the
general form, with the head term engaging whenever the output head changes
(App. C.1/D); nothing in the derivation assumes a frozen head or a particular
training recipe, and the frozen-head form is a simplification we exploit in
Sec. 3, not a condition the bound needs. What changes far from the base is
tightness, not validity: the farther the variant drifts, the less optimal any
single orthogonal alignment becomes, so the absolute bound loosens, while the
within-cell ordering is expected, as an empirical extrapolation rather than a
theorem, to persist wherever drift exceeds the noise floor (the SNR regime of
App. F.3).

That far regime is exactly what the paper defers rather than claims. App. C.3
derives the joint feature-head alignment for "models with both rotated features
and divergent heads (e.g., full-parameter SFT)" and leaves its calibration to
future work, and App. I lists the extension first among future directions: beyond
LoRA forgetting, to full SFT, distillation and continual learning, where backbone
drift is substantially larger and the shape and scale terms serve as per-step
drift monitors. We keep validated claims scoped to PTQ and frozen-head LoRA; the
extension stands as stated theory plus declared scope.

6) Q2 [原文]: How does PRISM behave when the model is fine-tuned on mixed or multi-task
   data? The current experiments seem to focus on variants derived from already post-trained
   models and relatively controlled fine-tuning settings. It would be useful to know whether
   similar correlations and diagnostic patterns hold when applying PRISM from the original
   base model to SFT models, or when fine-tuning jointly on heterogeneous datasets.

> **Q2.** Mixed or multi-task fine-tuning: from the original base model to SFT
> models, or joint fine-tuning on heterogeneous data?

The mathematical frame is the same as Q1: the bound is agnostic to the recipe, so
mixed-task SFT changes nothing in validity. What heterogeneous data changes is
the empirical drift pattern, and that is the familiar generalization question
rather than a property of the bound: a mixture spreads drift unevenly across
tasks, so there is no single mixture-level score to certify, and we do not define
one. PRISM is input-conditioned by construction, so it is applied per evaluation
task and reads that task's own drift (Sec. 3.4, App. D); under multi-task
fine-tuning, that per-task reading is exactly the quantity one wants to monitor.

On evidence, two pairings must be kept distinct. What the paper already contains
(Tables 15, 17 and 16; App. F.3) is PTQ diagnosis on SFT products: the
instruction-tuned counterparts of Llama, Ministral and Qwen, themselves the
result of large-scale heterogeneous multi-task SFT (and RLHF), are each
diagnosed against their own FP16 reference and reproduce the same patterns as
the base families. That shows the diagnostics are stable when the model under
study comes from heterogeneous post-training; it is not yet the pairing the
question poses. For that pairing we ran the decomposition directly as a
rebuttal-run consistency check, in the same orientation as the question ("from
the original base model to SFT models") and as the paper's LoRA-forgetting
setting: the base checkpoint serves as the reference (target), and its SFT
(instruct) counterpart is the diagnosed variant (proxy). The bound holds in all
10 (pair, benchmark) cells, the head term engages at a median 50% of B, as the
open-head regime predicts, and backbone drift sits at or beyond each family's
own Q2_K level. The
conservative conclusion we draw: the diagnostic patterns are stable for models
produced by heterogeneous post-training, per-task application remains
well-defined under mixed data, and calibrated tightness for the base-to-SFT
pairing sits in the App. C.3 regime of Q1 and stays future work.

**In sum:** the measured cost table, the reference-set ablation, the free-running
check, the causal single-axis interventions, and the same-features CKA/SVCCA and
regularizer comparisons are measurements, not restatements, and Q1/Q2 are
answered with the theory's actual scope, the paper's instruct-family
replications, and a direct base-to-SFT consistency check. We hope these resolve
the motivation and evaluation concerns; we would particularly value knowing
whether the cost table and the controlled interventions resolve the motivation
and actionability points, and we are glad to follow up on anything that remains
during the discussion.

================================================================================
Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3
================================================================================

···· ORIGINAL REVIEW — verbatim from OpenReview (for 一一對應檢查; internal, DO NOT POST) ····
(Note: eQL6's items are math-heavy; several inline symbols did not survive the OpenReview
OCR and are marked [.] below — see the original PDF for the exact notation.)
[Summary]
The paper proposes PRISM, a framework to analyze different failure modes in post-trained
variants (e.g., quantized, LoRA fine-tuned) of pre-trained LLMs. The authors analyze the
population risk gap between a "target" model (e.g., full-precision pre-trained base model)
and "proxy" model (e.g., quantized model) under an orthogonal transformation of the proxy
model's backbone features and classifier head. They derive a bound on the risk that
decomposes into three terms: backbone scale mismatch, backbone shape mismatch, and
classifier head discrepancy. From this, the authors propose a shape regularizer in the
objective loss to align pre- and post-trained model backbone features. Finally, the authors
empirically demonstrate the derived bound scales with the observed risks, the decomposed
terms correspond to different failure modes, and the proposed regularizer alleviates the
risk gap between pre- and post-trained models.

[Strengths]
- Diagnosing why and how post-trained models significantly deviate from their base models
  is important.
- Decomposing the proxy-target risk gap into interpretable terms, and associating each term
  with different failure modes, are insightful observations.
- The authors' proposed regularizer appears to mitigate the risk gap between the base and
  fine-tuned models.
···· END ORIGINAL REVIEW ····

We thank the reviewer, and we take the clarity feedback seriously: we treat it
as our presentation to fix, not the reviewer's reading. Concretely: a notation
table and first-use definitions, with the key Section 5 results moved into the
main text (W1); an explicit identifiability argument for orthogonal alignment (W2/Q1); and
a demonstration on the paper's own data that the shape term is driven by
1-Omega, not the norm prefactor (W3), which is the sharpest technical concern.

1) W1 [原文]: The writing and communication is sometimes not very clear. The authors
   commonly use notation that has not yet been defined, e.g., under the summary of main
   contributions in the Introduction, they state [.] represents "shape" without specifying
   what [.] and [.] represent, and what "shape" means in that context. Furthermore, in
   Section 5, results that are central to the authors' arguments are left to the appendix,
   making it slightly difficult to verify the written claims there.

> **W1.** Unclear writing: notation used before it is defined ("shape" in the
> contributions list), and Section 5 leaves results central to the argument in the
> appendix.

Agreed on both counts. We will (i) define rho_T, rho_P (RMS feature scale of
the reference feature matrices) and Omega (normalized Procrustes alignment,
with 1-Omega the dimensionless shape residual) at first use in the
Introduction, each with a one-line intuition; (ii) bring the results the Sec. 5
argument rests on into the main text: the full 20-cell LoRA matrix and the
gating table (currently App. H, Table 22) move up from the appendix, and the new
same-features CKA/SVCCA comparison is added alongside them; (iii) add a
notation table. Thank you for flagging these.

2) W2 + Q1 [原文]:
   [W2] In the theoretical analysis, the motivation for considering orthogonal
   transformations for proxy-target alignment is not very clear.
   [Q1] (See Weakness 2) Could the authors clarify why the linear / platonic
   representation hypotheses motivates studying orthogonal transformations specifically to
   analyze target-proxy model risk?

> **W2 + Q1 (Q1 = "See Weakness 2").** Why restrict proxy-target alignment to
> orthogonal transformations, and why should the linear / platonic representation
> hypotheses motivate that specifically?

The load-bearing reason is the linear head, not the representation hypotheses. Our
own presentation hides that: L112-115 motivates the restriction by empirical
isometry, **which says why orthogonality is cheap, not why it is necessary.** The
necessity came from the workflow the tool serves. We need to compare many quantized
and fine-tuned variants of one base quickly, from a single teacher-forced pass each
rather than a benchmark suite, and be told which axis to repair when a variant
decays. **That asks two things of the quantity: the damage must not be fittable
away, and whatever remains must divide into parts one can act on.** Steps (1) and
(2) are those two requirements; (3) is where the hypotheses come in.

(1) Why we restrict to O(d): identifiability. Because the head is linear, a
model's features and head are only defined jointly, up to an invertible map: Z_P H_P
and (Z_P A)(A^-1 H_P) are the same proxy model. **So a feature discrepancy measured
after an unrestricted invertible alignment is not identifiable: the damage can
always be pushed through A into the head and fitted away.** O(d) is exactly the set
of linear maps that leaves the geometry the head reads out untouched: an orthogonal
W only re-orients the proxy's feature cloud, preserving every norm, distance and
angle inside it, so what it leaves is distortion the head genuinely has to absorb.

(2) What the restriction buys: two axes that can be read separately. Over O(d) the
mean squared feature residual splits with no approximation into `(rho_T - rho_P)^2`
plus `2 rho_T rho_P (1 - Omega_W)` (Prop. 1, L125), and the rho are measured RMS
feature scales rather than fitted parameters, which is what lets scale mismatch and
shape mismatch be attributed separately. Under a general linear map the identity is
unavailable, since the map itself carries scale and shear; larger groups do have
closed-form solutions, just not a residual that still means anything.

(3) Where the hypotheses enter, and all they are needed for: they make the
restriction cheap. The linear/platonic results say that related models tend to align
well under near-orthogonal maps, which is why a single orthogonal W is sufficient in
practice, i.e. why the bound is tight rather than vacuous. Quantitatively: on
Llama-3.1-8B, in the top-r principal subspace the diagnostic reads (r <= 256,
77-90% explained variance), lifting the restriction to an unrestricted linear map
shrinks the feature residual by under a fifth on average even at 2 bits, and by at
most 0.29 in any single cell. **Validity does not depend on any of this: Theorem 1 holds for any
W in O(d) (L132, App. C.1), so if the hypotheses were false the bound would merely
loosen, never break.**

One distinction is worth making explicit, since it is the likeliest residue of the
confusion: the paper chooses at two levels. The group is the natural maximal
linear family that keeps the attribution identifiable, by (1) and (2).
Which member of it we use is not; it is a design parameter the family hands us for
free: W = I for analysis and the regularizer (differentiable, head term exactly
zero when the head is untouched), W_N when the goal is a correct ordering. The
paper calls that choice "primarily design-driven" (L145-152) because both are
certified and differ only in tightness and in which term carries the mismatch.

3) W3 [原文]: In the experiments, to me it seems the shape mismatch term dominating the
   risk bound is due to the scale of the feature matrices themselves, rather than the actual
   misalignment [.]. As an extreme example, consider [.] with [.] and [.], i.e., the feature
   matrices and head weights blow up in magnitude, but stay on the same order before and
   after post-training (the second part is supported by Tables 1 and 18 in my opinion). Then,
   both the shape mismatch and head discrepancy terms are still [.], but [.]. If [.] exactly,
   then [.], but we still have [.]. From Table 1, [.] is typically close to [.], and [.], so
   to me it seems any "large" [.]'s are due to the [.] product being large.
   (Math heavily OCR-garbled; the Response below restates the reviewer's algebra in full,
   i.e. "if feature energies blew up, Z-tilde = cZ, the shape/head terms would scale as
   c^2 and the prefactor would dominate.")

> **W3.** The shape term may dominate because of feature magnitude rather than
> misalignment: scaling features and head weights by c scales both terms as c^2
> while staying the same order before and after post-training (Tables 1, 18), so a
> large bound would reflect a large rho_T rho_P product.

The reviewer's algebra is exactly right: if feature energies blew up
(Z-tilde = cZ, large c), the shape and head terms would scale as c^2 and the
prefactor would dominate. We show, on the paper's own data, that this case does
not arise among variants of a shared backbone. The shape term
`rho_T*rho_P*(1-Omega)` is (up to constants) the aligned residual energy
`||Z_T - s Z_P O||^2_F`: the prefactor carries physical units (logits scale with
feature norm x head norm), 1-Omega is the dimensionless geometric part. Every
comparison holds T fixed and varies P over the same backbone, so **within a
(family, benchmark) cell `rho_T*rho_P` varies with median CV 0.63% (p90 7.9%),
essentially constant, while 1-Omega spans a median 366x dynamic range**;
Spearman(shape term, 1-Omega) has median 1.000 (min 0.988) over the 52 cells
with Omega variance (of 55), and the two give identical rank correlations
against |dR| (median 0.770). So the ordering and the diagnostic signal come
from 1-Omega; the prefactor sets units, not ranking. This is also why the
inflation the reviewer worries about could not materially reorder even if it were
larger: the cell's other constants are shared too (K_feat depends only on the
target head, K_pred <= sqrt(2) universally), and a factor common, or nearly
common, to every variant of a cell inflates all bounds together, which is exactly
why the paper reports within-cell rank correlations rather than absolute values.

Two boundary cases complete the answer. The premise that rho_T stays close to
rho_P is model-dependent rather than guaranteed, and the paper itself contains
the exception: Qwen3-Base Q2_K on GSM8K, where rho_P jumps from 267 to 313
(|Delta rho| = 46), is called out in Sec. 5.3 as a scale-axis outlier, exactly
the channel the decomposition separates so that it can be flagged. And the
divergent-norm limit (feature energies blowing up relative to each other) is
that same pathology taken to the extreme; RMSNorm-bounded hidden states keep
feature energies finite in practice, so it cannot arise silently. We
additionally report the dimensionless `delta/(rho_T*rho_P)` alongside delta in
the revision; rankings are unchanged.

4) W4 + Q3 [原文]:
   [W4] Although the proposed regularization helps close the pre-trained &
   fine-tuned model risk gap, it's unclear how this regularization impacts the fine-tuned
   model's performance on the original downstream task.
   [Q3] (See Weakness 4) Under the proposed regularization objective, does the
   fine-tuned model perform just as well on the original downstream task compared to the
   baselines (no regularization and replay-CE)?

> **W4 + Q3 (Q3 = "See Weakness 4").** How does the shape regularization affect the
> fine-tuned model on its own downstream task (the task it is fine-tuned on), and
> does it match the no-regularization and replay-CE baselines there?

It does not degrade the target task, and the check is already in the paper: we
will make it unmissable and add numbers.

(1) Versus no-reg and replay-CE, the two baselines Q3 names. The first column of
Fig. 4 (Fig. 8 for Qwen), "Fine-tune Dataset", plots the signed risk change on
the fine-tuning task itself for all three configurations (no-reg / replay /
trace) across training. Since the y-axis is Delta R relative to the base, below
zero means the model is improving on its own task, and all three trajectories
descend below zero together and essentially coincide through step 300: the shape
penalty buys its reduction in downstream forgetting without giving up
target-task learning at the operating points used. We recognize this column is
easy to miss and will label it explicitly.

(2) Versus the newly added baselines, numerically, over three seeds (42/43/44,
mean ± sd). **Target-task loss: shape penalty 0.872 ± 0.005, replay 0.899 ± 0.007,
L2-SP 0.901 ± 0.003, EWC 0.905 ± 0.010, layer-freeze 0.924 ± 0.001 (top 16). The
shape penalty has the lowest target-task loss of every method compared, while also
having the lowest downstream forgetting among the methods that keep the task
(0.680 ± 0.016).** So it does not trade target-task performance for retention; it is
better on both axes at once. Layer-freezing's low forgetting is bought exactly here,
at target loss 0.924. This is also why every baseline in
this comparison is reported at the sweep config whose target loss comes closest to
the shape run's: comparing at matched plasticity is the only way the forgetting
numbers mean anything.

(3) The revision adds this as a numeric column (target-task loss beside
downstream forgetting) for every method, and if a trade-off does appear for any
baseline we report its lambda sweep, tracing the stability-plasticity frontier
rather than a single point.

5) Q2 [原文]: Could the authors also clarify the meaning of "Shape preservation admits two
   lifecycle-specific instances: Hessian-aware reconstruction at PTQ time, and ... " from
   Section 5.3?

> **Q2.** Clarify Section 5.3's "Shape preservation admits two lifecycle-specific
> instances: Hessian-aware reconstruction at PTQ time, and ...".

Thank you for flagging this sentence. It compressed two different things and
overstated the PTQ half, so we will rewrite it.

**What we meant.** The quantity $1-\Omega$ measures how much the feature
directions disagree, independently of size, and the shape term of the bound is
that disagreement weighted by the two scales, $2\rho_T\rho_P(1-\Omega)$. Whether it
can be reduced directly depends on when one acts. In GPTQ the model is
not trained end to end: the quantization decisions are discrete and are taken
layer by layer against a calibration objective, so our end-to-end shape term is
not a quantity that objective can act on directly. During LoRA training it is.
Those are the two "instances": two different kinds of method, not one method with
two settings.

**Why the PTQ half is not shape-specific.** Take one layer with weights $M$
($d_{\mathrm{out}}\times d_{\mathrm{in}}$), quantized weights $\widehat{M}$, and
calibration inputs $X$ ($d_{\mathrm{in}}\times n_c$) for $n_c$ tokens, and write
$E = M - \widehat{M}$. GPTQ targets the output error

$$\|EX\|_F^2 = \mathrm{tr}\left(E \, XX^\top E^\top\right),$$

whose Hessian in each row of $\widehat{M}$ is proportional to $XX^\top$; that is
what "Hessian-aware" refers to. Now write that layer's outputs as
$Y = X^\top M^\top$ and $\widehat{Y} = X^\top \widehat{M}^\top$, both
$n_c \times d_{\mathrm{out}}$, with scales $\rho = \|Y\|_F / \sqrt{n_c}$ and
$\widehat{\rho} = \|\widehat{Y}\|_F / \sqrt{n_c}$. Since
$\|Y - \widehat{Y}\|_F^2 = \|EX\|_F^2$, that error is exactly what Proposition 1's
identity splits in two, the identity being algebraic and so holding for any two
matrices of the same shape (at $W = I$; the factor $1/n_c$ does not change what is
minimized):

$$\frac{1}{n_c} \|Y - \widehat{Y}\|_F^2 = \left(\rho - \widehat{\rho}\right)^2 + 2 \rho \widehat{\rho} \left(1 - \Omega(Y, \widehat{Y})\right),$$

where $\Omega$ is the normalized inner product of Eq. (3) at $W = I$. GPTQ therefore
targets a joint
residual containing both components. **Minimizing a sum does not by itself make
either part fall, so it may affect the local shape component only indirectly, and
a single scalar residual cannot say which component dominates.** This is analysis of a published
objective, not a measurement of ours.

**The LoRA half is different** because the model is still training, so the objective
is ours to design rather than inherited. There, $1-\Omega$ is computed from the
current
features and, wherever the feature norms are nonzero, differentiated without an
SVD, which is one reason we use the
identity alignment $W = I$ for the intervention, together with the frozen-head
simplification it gives (Sec. 3; $W = W_N$ needs a per-step SVD). Eq. (8)
therefore penalizes $1-\Omega$ itself rather than a
residual in which it is mixed with scale.

**What we used, and what we meant.** The GPTQ and GGUF variants we diagnose are
publicly released pre-quantized checkpoints; BitsAndBytes is applied on the fly
from the full-precision checkpoint, without calibration data. We never run or
modify GPTQ calibration. Our intention was narrower than the sentence suggests:
GPTQ's calibration phase optimizes an objective that could be changed, so the
decomposition points at a possible link to shape there, though we did not pursue
it. The sentence also closes Sec. 5.3, which reports diagnosis; the one
intervention we evaluate is the regularizer of Sec. 5.4. The scale- and head-axis remedies
there are likewise suggested directions, not controlled interventions. We will replace the original sentence with:

> "Each dominant axis suggests a remediation direction, though only shape
> regularization is evaluated here as an intervention. At PTQ time, GPTQ-style
> Hessian-aware reconstruction [frantar2022gptq] affects shape only indirectly,
> since it minimizes a layerwise error containing both a scale and a shape component
> (Prop. 1 at $W = I$); at LoRA training time, $1-\Omega$ can be penalized directly
> (Eq. (8), Sec. 5.4). Per-channel outlier smoothing [xiao2023smoothquant] for the
> scale axis and FP16-`lm_head` retention for the head axis are suggested as
> protocol-level remediation directions but are not evaluated here as controlled
> interventions."

**Follow-up.** The separated version already exists here at the other lifecycle
stage, as Eq. (8). Whether it carries into calibration is the open question, and two
attempts differ in kind: adding a penalty $\mu(1-\Omega)$ on top of the joint
residual only moves the shape coefficient from $2\rho\widehat{\rho}$ to
$2\rho\widehat{\rho}+\mu$ by the identity above, so it still optimizes a mixture,
whereas minimizing $1-\Omega$ under a scale budget separates the axes. Either needs calibration rerun under
matched configurations, outside this diagnostic study; the revision lists it in
Sec. 6.

6) Q4 [原文]: In Section 5.5, the Spearman [r_s] is computed between each risk gap bound
   term, and what other variable? Is it the empirical risk gap?

> **Q4.** In Section 5.5, Spearman is computed between each bound term and what? The
> empirical risk gap?

Against the empirical cross-entropy risk gap: for each variant we compute the
quantity |dR|, the benchmark-measured CE gap between target and proxy under the fixed
evaluation protocol (defined in Sec. 5.1; Table 1's caption states it as
"r_s(B, |dR|)", and Fig. 2 plots B against |dR| with per-subplot r_s
annotated), and r_s is computed between each bound
term and |dR| across all variants within a (family, benchmark) cell; Table 3
aggregates over the 10 cells. One nuance, since the reviewer asks precisely:
Eq. (1) defines the population risk, and every reported quantity, the bound
and |dR| alike, is its empirical counterpart on the fixed held-out subsets of
Sec. 5.1; the revision restates Theorem 1 at that empirical level, with a
concentration corollary recovering the population version. We will state both
explicitly in the Sec. 5.5 text and every table caption.

**In sum:** the four substantive changes are the notation and presentation fixes
(W1), the identifiability argument for orthogonal alignment with the hypotheses' role
scoped to tightness (W2 + Q1), the prefactor analysis on the paper's own data
showing the shape term is driven by 1-Omega and not by feature magnitude (W3),
and the target-task numbers showing the penalty costs no plasticity (W4 + Q3).
W3 in particular forced an analysis that made the paper better: the prefactor
study it prompted now anchors how the shape term should be read, and the
dimensionless variant it suggested enters the revision. We hope these resolve
the quality and clarity concerns; we would value knowing whether the
identifiability argument resolves the orthogonal-alignment motivation, and we are
glad to clarify anything further.

================================================================================
Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3
================================================================================

···· ORIGINAL REVIEW — verbatim from OpenReview (for 一一對應檢查; internal, DO NOT POST) ····
[Summary]
This paper proposes a PRISM method for diagnosing drift between LLMs and their post-training
variants. The analysis introduces a hybrid model that applies the target head to aligned
surrogate features. This decomposes the cross-entropy risk gap into feature terms and head
terms. Feature residuals are further precisely decomposed into scale mismatch and shape
mismatch, while the head term is weighted by the covariance of the surrogate features.
Experiments cover PTQ variants and frozen-head LoRA checkpoints. The authors also use a
differentiable shape term as a regularization term to prevent forgetting.

[Strengths]
- The derivation is clear and easy to understand. The hybrid risk construction method
  separates the backbone network and head drift, with pairwise embedding diameters superior
  to the general spectral norm Lipschitz bound, and covariance weighting restricting head
  differences to the direction of data activity. Its innovation lies in combining these
  elements to form a deployed head risk diagnostic model.
- Scale/shape/head decomposition is more informative than a single representation similarity
  score. This paper also effectively distinguishes between identity alignment that preserves
  frozen head simplification and feature-optimal Procrustes alignment.
- Transforming differentiable shape terms into training objectives is conceptually
  attractive and helps address the significant shape drift problem in Llama-TruthfulQA
  scenarios.

[References cited in the Weaknesses]
[1] SLoRA: Balancing Plasticity and Forgetting in Large Language Models for Continual
    Learning. ACL, 2026.
[2] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in
    Large Language Models. Findings ACL, 2026.
[3] CLAIM: Mitigating Catastrophic Forgetting in Continual Instruction Fine-tuning Large
    Language Models. IEEE TCDS, 2026.
[4] ArMA: Mitigating Catastrophic Forgetting using Attention-Regularized Model Averaging in
    Continual Fine-tuning Large Language Models. IEEE TAI, 2026.

[Limitations (reviewer's field)]
This paper acknowledges the limitations of ranking risk and absolute risk, as well as the
teacher-enforced approach. It should also discuss reference set sensitivity, free-run
generation, head variation variants, and feature computation costs.
···· END ORIGINAL REVIEW ····

We thank the reviewer for the precise and technically careful reading. The
response adds exactly what was asked: the empirical-risk restatement plus a
McDiarmid corollary (W1); controlled single-axis interventions (W3); a
calibration analysis worked out on the reviewer's own two example cells (W2/Q1);
and a free-running experiment (W4/Q2). The two cells the review flags lack a
rankable signal, for two different reasons now stated; they are reported and
explained rather than hidden.

1) W1 [原文]: Empirical Risk vs. Overall Risk: Formula (1) defines risk as the expectation
   of D, but subsequent proofs use a finite sample feature matrix without introducing a
   generalization term. For now, this guarantee seems to apply only to empirical risk on
   calibration samples. The theorem should be restated accordingly, or supplemented with a
   finite sample argument.

> **W1.** Eq. (1) defines risk as an expectation, but the proofs use a finite-sample
> feature matrix with no generalization term, so the guarantee appears to cover only
> empirical risk. Restate the theorem, or add a finite-sample argument.

The reviewer is right, and we fix it exactly as suggested, both ways.

(1) Theorem 1 is restated as a bound on the empirical risk gap over the
reference sample, also the quantity our diagnostic use case targets (all
experiments evaluate on fixed reference sets; **every step of the proof (the
per-sample triangle inequality, pointwise Lipschitz/simplex-polarization,
Jensen on the empirical mean, and the exact finite-sample identity of
Prop. 1) already holds at the sample level**). (2) New corollary: under i.i.d.
sampling and an explicit almost-sure bound on the per-sample loss gap (with
explicit constant `b <= K_feat*D_Z + sqrt(2)*C_z*||W H_T - H_P||_op`, whose
empirical analogues the pipeline measures), McDiarmid's inequality
yields `|population gap - empirical gap| <= b*sqrt(2 ln(2/delta)/N)` w.p.
1-delta, giving the population version at an additive concentration cost. The
constant b is finite by construction: hidden states are RMSNorm-bounded, so D_Z
(a bound on the aligned feature difference) and C_z are finite and no
unbounded-CE pathology arises. For
sequences, N counts i.i.d. sequences (per-sequence token-averaged losses), not
tokens. The restated theorem and proof are written; we are happy to post them
in this thread. No experimental conclusion changes, since all reported
quantities were already empirical.

2) W2 + Q1 [原文]:
   [W2] Tightness and Practical Value of the Bound: The bound is very loose. For example,
   B=266.09 for Llama-MMLU Q2_K, while |Delta R|=0.3658; B=23.24 for Q8_0, while
   |Delta R|=0.0002. The paper should quantify additive or multiplicative relaxation and
   clarify whether the bound supports any operational threshold other than ordination.
   [Q1] Besides the ordering, could the authors provide an error analysis regarding the
   magnitude of the bounds?

> **W2 + Q1 (Q1 asks for the error analysis W2 wants).** Very loose, e.g. Llama-MMLU
> Q2_K gives B = 266.09 against |dR| = 0.3658. Quantify the relaxation, analyse the
> bound magnitudes and not only the ordering, and say whether any operational
> threshold is supported.

Quantified directly, on all 570 (family, benchmark, variant) cells over the
paper's five benchmarks. Slack B/|dR|: median 1597x, IQR [673, 3966], and not
constant (log10 sd ~0.76; the reviewer's two cells, ~727x near the lower
quartile and ~10^5x in the upper tail, are both within the observed range),
which is exactly why we claim only ordering from the raw bound. Attribution by
derivation step: the Lipschitz-constant relaxation contributes only 0.13-0.40
dex (simplex polarization already keeps K_feat tight, which bears out the
reviewer's observation that the pairwise-diameter step improves on the
spectral-norm bound: that step is not where the looseness lives); the dominant
slack is the alignment/triangle remainder; and median slack shrinks with bit-width
(Q8_0: 3.65 dex -> Q2_K: 2.78 dex), relatively tightest where degradation is
real. Operational threshold: per-(family, benchmark) leave-one-out isotonic
calibration of B -> |dR| achieves aggregate leave-one-out MAE 0.055 nats, the
error on a held-out variant (vs 0.082 predict-the-mean), and the rule
"calibrated prediction < 0.1 nats" reaches
precision >= 0.8 in 49/55 cells (21 at 1.0; the six weaker cells are all
multiple-choice benchmarks, ARC/MMLU, where the |dR| dynamic range is
compressed).

**Concretely, the reviewer's own two cells (both Llama-3.1-8B/MMLU): the same
absolute-scale-uninformative bounds, after the one-time per-cell calibration,
land on the correct side of an epsilon = 0.1 nat tolerance:**

| Variant | $\mathcal{B}$ | $\lvert\Delta\mathcal{R}\rvert$ | Slack $\mathcal{B}/\lvert\Delta\mathcal{R}\rvert$ | $\hat{r}$ (nats) | $\hat{r}<\varepsilon$ / truth |
|:--|--:|--:|:--:|--:|:--:|
| `Q8_0` | 23.24 | 0.0002 | $\sim 10^{5}\times$ | 0.003 | within / within ✓ |
| `Q4_K_M` | 96.75 | 0.036 | $\sim 2718\times$ | 0.055 | within / within ✓ |
| `Q2_K` | 266.09 | 0.366 | $\sim 727\times$ | 0.193 | outside / outside ✓ |

We add that cell's operating characteristic rather than only its favourable rows,
since the reviewer is likely to check it: Leave-one-out MAE 0.040 nats and precision 0.75 at
eps = 0.1, one of the six cells of 55 below 0.8. All four of its errors have
calibrated predictions within 1.2 MAE of the tolerance, and every call whose
prediction sits further out than that band is correct, so the
failure mode is a bounded ambiguous band rather than an arbitrary error: the
calibrated bound decides the clear variants and flags the borderline ones for
measurement. Both analyses become a new section/table, the band is stated with
them, and uncalibrated B is explicitly scoped to ranking.

3) W3 + Q3 + Q4 [原文]:
   [W3] The evidence for actionability is mixed. The three proposed failure modes are
   illustrated through selected cases, but there are no controlled interventions that change
   only activation scale, backbone geometry, or the output head. The regularizer clearly
   improves over no regularization only for Llama-TruthfulQA. Table 22 reports increased
   forgetting on Llama-BBQ (+8.6%) and Qwen-TruthfulQA (+2.7%), with almost no change on
   Qwen-BBQ. Experience replay is therefore not enough as the sole strong baseline.
   Comparisons with SLoRA [1], replay or layer freezing from two-phase continual fine-tuning
   [2], CLAIM [3], and ArMA [4] would make the result more convincing. Multiple training
   seeds, source-task performance, and sensitivity to the 32-sequence reference set are also
   needed.
   [Q3] How sensitive is the shape regularizer to the size of the reference set and the
   domain?
   [Q4] Can this regularizer be compared to stronger forgetting baseline methods?

> **W3 + Q3 + Q4 (Q3 and Q4 restate two of W3's items).** Actionability rests on
> selected cases, with no controlled interventions isolating scale, geometry or the
> head. The regularizer clearly wins only on Llama-TruthfulQA, and Table 22 shows
> more forgetting on Llama-BBQ (+8.6%) and Qwen-TruthfulQA (+2.7%), so replay alone
> is too weak a sole baseline: SLoRA, two-phase CIT, CLAIM, ArMA (Q4). Also multiple
> seeds, source-task performance, and 32-sequence reference sensitivity (Q3).

Every item asked for is now measured: controlled single-axis interventions (1),
the mixed Table 22 cells read through the gating analysis that table exists to
validate, rather than hidden (2), and matched stronger baselines plus seeds,
source-task performance, and reference-set sensitivity (3).

(1) Controlled interventions (added): into one base model we inject (a) pure
scale change (final-norm rescale, alpha in {0.5..2}), (b) pure geometric
distortion (norm-preserving rotation `expm(theta*A)`, token norms preserved
exactly), (c) head-only perturbation (RTN-quantized lm_head at {8,6,4,3}
bits, backbone untouched). Each intervention is a real forward-pass change, so
the same run reports the measured |dR| and verifies the bound; all terms are
read under the paper's W = I analysis default, the gauge in which a rotation
against the fixed head is genuine functional damage. Selectivity matrix
(Llama-3.1-8B; max |term - control| within each family, MMLU / TriviaQA):

| intervention family | Scale term | Shape term | gamma |
|---|---|---|---|
| scale-only (alpha 0.5-2.0) | **1.9e4 / 2.0e4** | 6.9e-2 / 7.5e-2 | 0 / 0 |
| rotation-only (theta <= 0.4 rad) | 1.3e-4 / 6.6e-6 | **7.7e2 / 7.9e2** | 0 / 0 |
| head-only (RTN 8 -> 3 bits) | 0 (exact) | 0 (exact) | **5.4e2 / 4.6e2** |

**Each family moves its own term by >= 2.6e5x the largest cross-axis leakage,
and the bound holds in 26/26 configs (identity control: B = 0.00/0.08).**
The terms also track their theoretical laws: Scale follows (alpha-1)^2 and
Shape follows theta^2 to within ~1%, and gamma rises monotonically with head
quantization (10 -> 40 -> 210 -> 541 for 8/6/4/3 bits). The interventions
cause real measured damage (|dR| up to 1.39 nats at alpha = 0.5, 0.42 at
3-bit head), so for these constructed perturbations the attribution is causal,
not correlational.

(2) The Table 22 cells: we would first note that the reviewer's own strengths already
state the positive half of our gating thesis, that the shape objective "helps
address the significant shape drift problem in Llama-TruthfulQA scenarios"; the
gating analysis simply completes that observation by explaining the "only". 
One thing we should have made clearer in the paper, and the fault is ours: **Table 22
is not a four-setting results table but a gating-validation table.** Its rows are
ordered by decreasing 1-Omega-bar, the shape drift available to repair, and the claim
it tests is a dose-response one: does our own diagnosis predict where the penalty will
help? The verdict column carries that, and two of the four rows read "at noise floor
=> skip".

| setting (ordered by 1-Omega-bar) | 1-Omega-bar | trace effect | our gating verdict |
|:--|--:|--:|:--|
| Llama TruthfulQA | 0.0937 | **-19.2%** | shape-driven, apply |
| Llama BBQ | 0.0678 | +8.6% | cell-level mixed |
| Qwen TruthfulQA | 0.0091 | +2.7% | at noise floor, skip |
| Qwen BBQ | 0.0011 | -0.2% | at noise floor, skip |

So on the two Qwen rows the framework says not to apply the penalty at all, because
their baseline Omega is already 0.991 and 0.999 (against Llama's 0.906 and 0.932),
leaving essentially no drift to repair. **Their +2.7% and -0.2% sit inside the ±3%
band the appendix itself defines as neutral, and every method there is within 0.02 of
every other, so we read them as confirming the gating signal rather than as increased
forgetting.** Three of the four settings match the prediction directly (App. H).

**Llama BBQ is the one genuine partial exception, and it is a condition-(ii) failure
rather than a shape-drift failure.** Its aggregate 1-Omega-bar = 0.0678 does exceed the
noise floor, so condition (i) holds; but the drift is concentrated on ARC and MMLU,
where substantial shape movement is NOT accompanied by proportional |dR| growth, so the
penalty has no proportional target. On the two Llama-BBQ benchmarks where condition
(ii) does hold, TriviaQA and GSM8K, trace yields **-88% and -79%**. The honest
reading is therefore granular: at the setting level, where Table 22 applies the
gate, this one row does mislead; at the per-benchmark level, where the two
conditions are actually stated, the gate separates the cells correctly. We promote
this analysis to the main text, present the regularizer as axis-targeted with the
gating rule explicit rather than as a universal win, and per-benchmark gating
(adaptive per-cell deployment, driven by the decomposition online) is the natural
next step.

(3) Baselines and scope, which is also Q4. A framing point first, since it fixes
which baselines are right: the paper's contribution is the diagnostic framework,
and the shape regularizer (Sec. 5.4) demonstrates that the diagnosed axis is
actionable, not a bid for state-of-the-art continual learning. The bar is whether
acting on the diagnosis helps against matched baselines. Replay was not arbitrary:
in this single-adapter, single-task setting it is the matched comparison (same
reference data, schedule, compute). We agree, and add the families matched to this setting, weight-space (EWC, L2-SP) and
architectural (layer-freezing), all protocol-identical, each at the sweep config
closest to the shape run's target loss (matched plasticity). TruthfulQA, three
seeds (42/43/44, mean ± sd); downstream forgetting is the mean risk gap over the
five held-out benchmarks:

| method | downstream forgetting | target-task loss |
|:--|--:|--:|
| no regularization | 0.815 ± 0.042 | 0.950 ± 0.008 |
| replay, lambda 0.01 | 0.771 ± 0.027 | 0.899 ± 0.007 |
| L2-SP, lambda 0.01 | 0.763 ± 0.019 | 0.901 ± 0.003 |
| EWC, lambda 0.1 | 0.751 ± 0.015 | 0.905 ± 0.010 |
| **shape penalty, lambda 1** | **0.680 ± 0.016** | **0.872 ± 0.005** |
| layer-freeze, top 16 | 0.404 ± 0.017 | 0.924 ± 0.001 |

The shape penalty forgets least among the methods that learn the task comparably;
layer-freezing's lower gap is bought at target loss 0.924, trading plasticity for
retention. The four cited methods
(SLoRA, two-phase CIT, CLAIM, ArMA) are specialized for sequential
continual-instruction tuning (a task stream), whereas we mitigate forgetting
within a single LoRA run; a fair comparison means moving to their benchmarks
rather than adding a baseline here. App. I already lists continual learning as
future work; the revision cites all four there.

(4) Reference-set sensitivity, which is Q3, on both sides. On the
regularizer side we ran the ablation at the paper's own operating point
(Llama-TruthfulQA, trace at lambda 1.0, lr 1e-5, seed 42, step 300), so its
numbers sit beside Table 2's 0.681 on the same scale. We varied only what the
question asks about, which sequences form D_ref: four draws per size, disjoint
windows of one fixed shuffle of the task's own held-out split, verified pairwise
disjoint. One correction: Sec. 5.1 calls D_ref "32 pre-training sequences", a
wording error. The code draws, and every number here was produced with, 32
held-out sequences of the fine-tuned task, disjoint from its training split; the
revision fixes it.

| reference size | mean downstream forgetting |
|:--|--:|
| n = 8 | 0.690 ± 0.019 |
| n = 16 | 0.685 ± 0.011 |
| **n = 32** (the paper's setting) | **0.676 ± 0.008** |

**All twelve runs fall in 0.666-0.716, i.e. 15.2% to 21.1% below the
no-regularization 0.843: the whole size-and-draw grid moves retention by at most
5.9 points, so the specific 32-sequence set is not load-bearing. The spread also
shrinks as the reference grows (sd 0.019, 0.011, 0.008), the signature of sampling
variation rather than of sensitivity to the data used.** Three checks. Rerunning
the paper's own draw here gives 0.6829 against 0.6813, a 0.0016-nat
reproduction gap, so the draw spread is a real effect about ten times that floor,
and small. The paper's 0.681 sits inside the n=32 range and is slightly worse than
the four-draw mean 0.676, so it is a typical draw, not a favourable one. And the
0.016 seed spread in (3) already contains draw variation, since each seed draws its
own reference set; isolating the draw gives 0.008, so this knob is smaller than the
seed variation already reported.

On the domain half of Q3 we have no controlled answer, and the bound licenses none:
preserving shape on generic text and expecting benchmark retention is the same
distribution-transfer step we scope out for the diagnostic, so a generic reference
is an untested heuristic rather than a claim. The matched test, a reference from a
structured-QA task outside the five benchmarks, is follow-up in the revision.

On the diagnostic side, three new seeds on the task's own data: against the gap on
the full 512-sequence slice, B_N reaches rs 0.932 ± 0.016 at 8 sequences and
0.932 ± 0.011 at 512, seed agreement on the ordering rising from 0.981 (n=8) to
0.998 (n=512). On domain, the diagnostic bound is input-conditioned by
construction, reading the diagnosed task's own validation items, a scope the
revision states in the Limitations.

(5) Multiple seeds and source-task performance, W3's remaining two items, both in
(3)'s table. **Every method is reported over all three seeds (mean ± sd),
and the shape penalty's advantage holds in each seed separately rather than only
on average.** Source-task performance is the target-task-loss column, where the
shape penalty is also the best (0.872 ± 0.005); that column defines the
matched-plasticity selection, so no method is credited for low forgetting bought by
under-training.

4) W4 + Q2 [原文]:
   [W4] Some conclusions are broader than the evidence. The reported LoRA mean correlation
   of r_s=0.831 is based on the Llama cells, whereas the Qwen-BBQ appendix includes
   correlations of -0.34 and -0.66. These failures should be reflected more clearly in the
   main discussion. Also, the autoregressive corollary evaluates both models on the same
   teacher-forced prefixes. It does not address the setting where the models generate
   different early tokens and therefore encounter different later contexts. A free-running
   experiment, or an additional term accounting for trajectory distribution shift, is needed
   for that claim.
   [Q2] How does PRISM perform on generated trajectories when surrogate errors affect
   subsequent context?

> **W4 + Q2 (Q2 asks for the free-running evidence W4 needs).** Two claims outrun
> the evidence: the LoRA mean rs = 0.831 rests on the Llama cells while Qwen-BBQ
> contains -0.34 and -0.66, which belong in the main discussion; and the
> autoregressive corollary scores both models on shared teacher-forced prefixes, so
> it misses models that diverge early and meet different later contexts (Q2).

(1) Agreed on prominence, and we move these two cells into the main text; we also
want to be precise about what they record, because the two mechanisms differ and
neither is a bound failure. The cells are Qwen3-8B fine-tuned on BBQ, evaluated
on MMLU (rs = -0.34) and on TriviaQA (rs = -0.66) in Fig. 7; Table 21 plus a
per-checkpoint recomputation on the paper's round show what each panel records.
**On TriviaQA the target side has nothing to rank: the fine-tune barely moves the
benchmark (|dR| = 0.0035 at lambda = 0 against 0.288 on ARC inside the same
trajectory, with per-checkpoint gaps within 0.002-0.03 and no trend), so the
checkpoints are statistical ties, and a rank correlation over ties is noise
(App. G.1). On MMLU the gap is real but non-monotone: it peaks near step 75 and
partially recovers by step 300, while accumulated backbone drift, and with it the
bound, grows through training, so a per-checkpoint rank correlation is
structurally depressed; the bound holds at every checkpoint.** So the two panels
read "benchmark robust to this fine-tune" and "forgetting that partially heals
while drift accumulates", not "bound failure"; Fig. 7's purpose is the
cross-family replication of the drift-tracking claim, and these panels are the
same instrument meeting a no-signal cell and a non-monotone cell, not a
scoreboard on which a negative entry is a failed case. The revision states the
two mechanisms separately where App. G.1 currently groups them.

We report it with the aggregate it qualifies: the revised main text carries the full
20-cell matrix, **mean rs +0.71, median +0.93, and 18 of 20 cells positive**, with
the two Qwen-BBQ exceptions visible, so 0.831 never appears without that context.
For ranking use, W_N is the sharper alignment where the shape term is at the noise
floor and close to neutral elsewhere: recomputed per benchmark, three quarters of the
Table 3 climb (Omega_N 0.806 to B_N 0.912) comes from GSM8K alone, the other four benchmarks moving
by at most 0.06; LoRA freezes
the head, so gamma = 0 at the W = I default and both alignments are certified,
though this does not change the two cells above, whose rank signal is degenerate
for the reasons already stated.

(2) Free-running, which is also Q2. We test on GSM8K, and the choice is forced
rather than convenient: of the five benchmarks it is the only one whose answers are
long enough for trajectory shift to exist at all, since ARC and MMLU answers are a
single token and SQuAD and TriviaQA average about 3 to 4. Each of the 12 variants
greedily generates its own continuation (mean 76 tokens per prompt across
subsets, so its errors
compound into its own context, exactly the regime the corollary did not cover) and
both models are then scored on those trajectories, with the generated tokens as
their own next-token targets, so the model-versus-model gap needs no reference
answers. Over 5 independent 100-prompt subsets (seeds 42-46, 12 variants each;
greedy decoding is deterministic, so a seed varies only the prompt subset):

| statistic (12 variants, mean ± sd over 5 subsets) | value | per-subset range |
|:--|--:|--:|
| rs(bound, gap) teacher-forced | +0.958 ± 0.011 | 0.944 to 0.972 |
| rs(bound, gap) free-running | +0.947 ± 0.015 | 0.923 to 0.965 |
| rank agreement, teacher-forced vs free-running bound | +0.959 ± 0.010 | 0.944 to 0.972 |
| **cross: teacher-forced bound vs free-running gap** | **+0.958 ± 0.011** | 0.944 to 0.972 |

**The last row is the operational one: the bound computed on reference text already
predicts the degradation the variant shows on its own generated trajectories, at
+0.958.** (Its summary statistics coincide with the teacher-forced row's because
the five per-subset values happen to form the same set in a different order;
the underlying per-subset pairs differ.) So a practitioner ranks once on the
reference slice and the ordering carries over to free-running use. The spread
across subsets is at most 0.042 on any row, so this is not a single-subset
artifact. Independently, Corollary 1 is restated
as teacher-forced-only with trajectory-distribution shift an explicit limitation:
the bound applies to any feature rows (App. D), so the restriction is protocol
rather than theory. The Limitations section adds reference-set sensitivity,
free-running generation, head-varying variants and feature-extraction cost, as
suggested.

5) Limitations [原文]: This paper acknowledges the limitations of ranking risk and absolute
   risk, as well as the teacher-enforced approach. It should also discuss reference set
   sensitivity, free-run generation, head variation variants, and feature computation costs.

> **Limitations.** Should also discuss reference-set sensitivity, free-running
> generation, head-varying variants, and feature computation cost.

Agreed, all four are added to the Limitations: (i) reference-set sensitivity
with the new size/domain ablations; (ii) free-running generation with the new
subset experiment and the restated teacher-forced-only corollary; (iii)
head-varying variants (full SFT/RLHF), already identified in App. C.3 as the
joint-alignment regime we scope out and list as future work; the revision
adds a first base-vs-instruct data point verifying bound validity in that
regime; (iv) feature-extraction cost, quantified in the new cost table (one
forward pass per variant over the reference set; no decoding, no grading;
measured on GSM8K on identical prompts, one greedy decode 556.7 s vs PRISM's 8.9 s
teacher-forced pass, a 62.6x gap, with model load excluded from both sides).
On (iv) we also make the data side explicit, since it bears on cost: the
reference sequences are validation items read as prompt plus reference answer,
with features and CE taken from that gold span (Sec. 5.1), so what the
measurement removes relative to a benchmark run is the decoding and the grading,
not the validation data itself.

**In sum:** each point above is answered with a measurement rather than a
clarification (the restated theorem, the calibration on the reviewer's own two
cells, the 26/26 causal interventions, the free-running result, the matched
baselines, and the reference size-and-draw ablation). 
Your careful reading materially improved the paper. We would particularly 
value knowing whether the empirical-risk restatement and the scoped free-running 
test address your main concerns about scope; if anything remains, 
we would be glad to address it during the discussion.
