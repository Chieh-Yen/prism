# PRISM (8674) Rebuttal Proposal: concrete edits on `rebuttal_draft.txt`

> Every proposal below is a concrete change to the CURRENT `rebuttal_draft.txt`,
> cited as **FROM** (exact current text, with line anchor) and **TO** (proposed
> replacement or insertion), each with the reviewer-psychology reason it moves a
> score. Ordered by return on the Borderline-Reject -> Borderline-Accept decision.
>
> **The situation in one breath.** This is a *significance* fight, not a
> correctness one: pCi8 and G3T9 (both conf 4) rate Quality/Clarity/Originality
> = 3 and only Significance = 2; eQL6 (conf 3, Q=2/Cl=2) is the cheapest flip;
> 8VrD (4) is the ally to convert into an advocate; and the AC makes the call
> and told us the price ("narrowing + concrete evidence"). The draft's *content*
> is already strong. The remaining gains are in *delivery*: make the AC's
> decision a checklist, defend against skimming, and hand each reviewer a
> low-effort reason to raise the score. Nothing below invents a number; it
> re-packages what is already true so a busy reviewer cannot miss it.

---

## P1 (INSERT, highest ROI): an AC-facing condition-by-condition checklist

**Why.** The AC decides, listed five conditions A-E, and set the bar "concrete
evidence rather than only clarification." A tickable table lets the AC justify
BR -> BA as a checklist and proves every condition carries a *measurement*, not a
promise. This is the single highest-leverage addition in the whole rebuttal.

**FROM** (draft L242-247, the global response currently ends here):

```
Summary, positioning sentence to close the global response:
"We believe these results, together with the narrowed framing (PRISM as a
calibrated, label-free ranking-and-attribution instrument with an explicitly
empirical-risk guarantee), directly address the five points of the
meta-review, and we respectfully ask the AC and reviewers to reconsider in
light of the new evidence."
```

**TO** (insert this table immediately BEFORE the positioning sentence, so the
response ends on the invitation):

```
Every condition is now answered with a NEW measurement, not a restatement:

| AC condition | New evidence (experiment) | Headline result |
|---|---|---|
| A. usefulness of the bound | isotonic calibration + slack decomposition | LOO MAE 0.055 nats; the two cited cells calibrate to 0.003 / 0.193 nats |
| B. benchmark-independent | generic-reference ranking (done) + size/stability ablation [E3B] | WikiText rs +0.506 vs +0.725 (done); size/stability [E3B] |
| C. stronger baselines | CKA/SVCCA on identical features (done); EWC/L2-SP/KD/layer-freeze [E2] | bound ties CKA (CIs cover 0) and does strictly more; baselines [E2] |
| D. free-running generation | greedy self-generation subset | rs +0.944 free vs +0.958 TF; TF bound predicts free-run at +0.951 |
| E. failures / mixed results | full 20-cell matrix + GSM8K SNR + honest span-fix negative | mean +0.71 / median +0.93; only 2 negative cells, both at the noise floor |

(Honesty guard: the [E3B] and [E2] cells above must not be posted until those
runs land; at posting, either fill the number or move the row's headline to "in
the revision" so the checklist never asserts an un-run result.)

Summary, positioning sentence to close the global response:
"We believe these results, together with the narrowed framing (PRISM as a
calibrated, label-free ranking-and-attribution instrument with an explicitly
empirical-risk guarantee), directly address the five points of the
meta-review, and we respectfully ask the AC and reviewers to reconsider in
light of the new evidence."
```

---

## P2 (INSERT x4, high ROI): a 3-line "what we added" opener per reviewer

**Why.** Reviewers skim. A short, bolded summary of the concrete NEW items at the
TOP of each reviewer's thread does three jobs: it survives skimming, it primes
the score before the reviewer hits the details, and for pCi8 it front-loads the
re-engagement hook (their strongest new result) without reordering the numbered
weaknesses. Keep each to three lines; length is the enemy.

**Length is net-neutral, not additive.** The opener lets you DELETE the
redundant framing sentences that currently repeat inside the individual items
(e.g., "as the meta-review suggests, we added...", "Agreed on both, and both now
exist..."). Add the opener, then cut those; the thread gets shorter and reads
top-down. This also directly serves the per-comment word limit.

**P2a: pCi8.** **FROM** (draft L251-253):

```
================================================================================

1) W1: three diagnostic conclusions already known; contribution is a unified
```

**TO** (insert the opener between the divider and "1) W1"):

```
================================================================================

Summary of what we added for the reviewer's points: a calibration analysis that
answers "is Q4 good enough" on the exact looseness the reviewer flagged (W2); a
same-features CKA/SVCCA comparison that quantifies what the machinery buys (W3);
EWC / L2-SP / feature-distillation / layer-freezing baselines (W5); and a direct
measurement of isometry violation vs bit-width (W6). We also report, candidly,
one mitigation that did NOT work (GSM8K final-answer span, W4).

1) W1: three diagnostic conclusions already known; contribution is a unified
```

**P2b: G3T9.** **FROM** (draft L413-415):

```
================================================================================

1) W1: why not just run benchmarks? axes coarse / not actionable; needs cost
```

**TO**:

```
================================================================================

Summary of what we added: a wall-clock cost table (PRISM 144 s/variant vs 1671 s
to evaluate the suite, 12x, label-free; W1); controlled single-axis
interventions showing each axis is causally separable and actionable (W1); a
same-features CKA/SVCCA comparison (W3); a benchmark-independent reference and a
free-running experiment (W2); and base-vs-instruct and multi-task extensions
(Q1, Q2).

1) W1: why not just run benchmarks? axes coarse / not actionable; needs cost
```

**P2c: eQL6.** **FROM** (draft L583-585):

```
================================================================================

1) W1: notation used before definition (rho_T rho_P (1-Omega) in the Intro);
```

**TO**:

```
================================================================================

We take the clarity feedback seriously and treat it as our presentation to fix,
not the reviewer's reading. Concretely: we add a notation table and move the key
Section 5 results into the main text (W1); we spell out the gauge argument for
orthogonal alignment (W2/Q1); and we show on the paper's own data that the shape
term is driven by 1-Omega, not the norm prefactor (W3), which is the reviewer's
sharpest technical concern.

1) W1: notation used before definition (rho_T rho_P (1-Omega) in the Intro);
```

**P2d: 8VrD.** **FROM** (draft L719-721):

```
================================================================================

1) W1: Eq. (1) defines population risk; proofs use finite-sample matrices;
```

**TO**:

```
================================================================================

We thank the reviewer for the precise reading. The response adds exactly what
was asked: the empirical-risk restatement plus a McDiarmid corollary (W1);
controlled single-axis interventions (W3); a calibration analysis worked out on
the reviewer's own two example cells (W2/Q1); and a free-running experiment
(W4/Q2). Negative cells are reported and explained, not hidden.

1) W1: Eq. (1) defines population risk; proofs use finite-sample matrices;
```

---

## P3 (FROM->TO, high ROI): name the new experiments in the very first sentence

**Why.** The AC's bar is "concrete evidence rather than only clarification."
"new experiments" is vague and reads as clarification; naming them in sentence
one satisfies the bar before the reader reaches section A.

**FROM** (draft L19-24):

```
We thank the AC and reviewers. Since submission we ran new experiments; below
we address each of the five points with concrete evidence and narrow the claims
where asked. One framing first, because it organizes everything: PRISM has
moved from a ranking heuristic to an operational, label-free instrument that
does three things benchmark evaluation and representation similarity (CKA/SVCCA)
cannot.
```

**TO**:

```
We thank the AC and reviewers. Since submission we added a calibration analysis,
a same-features CKA/SVCCA comparison, controlled single-axis interventions, a
free-running-generation experiment, and a measured GPU-cost table; below we
address each of the five points with this new evidence and narrow the claims
where asked. One framing first, because it organizes everything: PRISM has moved
from a ranking heuristic to an operational, label-free instrument that does
three things benchmark evaluation and representation similarity (CKA/SVCCA)
cannot.
```

---

## P4 (FROM->TO): a one-line "Net" close for the two Significance=2 reviewers

**Why.** pCi8 and G3T9 must move Significance 2 -> 3. Give each a single closing
line that names the axis and the two results that touch it, so the reviewer has
a ready-made justification. Do NOT presume the score; invite it. This is the
score-raise enabler.

**Register is deliberate.** The posted line says "we hope this addresses the
concern," never "please raise your score." Explicitly asking for a score reads
as pressure and backfires with conf-4 reviewers. The reviewer must feel they
reached the conclusion; our job is only to make the conclusion easy. (The blunt
"I raise my score because X" sentence belongs in our internal strategy notes,
not in the thread.)

**P4a: pCi8.** **FROM** (draft L407-409, end of the W6 Note; insert the Net line
between the W6 response and its Note, i.e., after L405 "...not gauge artifact."):

```
quantization most feature damage is not linearly alignable at all, so the
isometry-aligned residual is genuine degradation, not gauge artifact.

Note: ✅ 指路(L112–115, App C.1/C.2)+ E8 定稿已填(控制組 0.084;Q8/Q4/Q2/NF4
```

**TO**:

```
quantization most feature damage is not linearly alignable at all, so the
isometry-aligned residual is genuine degradation, not gauge artifact.

Net: the calibration (W2) shows the bound supports an absolute tolerance, not
only ranking, and the same-features comparison (W3) shows the decomposition buys
a risk certificate and an attribution that a similarity score cannot. We hope
these speak directly to the practical-significance concern.

Note: ✅ 指路(L112–115, App C.1/C.2)+ E8 定稿已填(控制組 0.084;Q8/Q4/Q2/NF4
```

**P4b: G3T9.** **FROM** (draft L452-454, end of W1 (3); insert the Net line after
"...made explicit in the Introduction." and before the Note):

```
(3) PRISM is label-free, so it applies where no benchmark exists
(private domains). Positioning is made explicit in the Introduction.

Note: 指路(L258–262, L249–257)+ E4 ✅ 定稿(26/26 bound holds,選擇性 ≥2.6e5×)
```

**TO**:

```
(3) PRISM is label-free, so it applies to proprietary or safety-sensitive
deployments where no public benchmark exists. Positioning is made explicit in
the Introduction.

Net: the measured 12-66x cost gap and the causal single-axis interventions
together answer "why not just run benchmarks": PRISM is materially cheaper AND
tells you which axis to fix, which a benchmark score does not. We hope this
addresses the significance concern.

Note: 指路(L258–262, L249–257)+ E4 ✅ 定稿(26/26 bound holds,選擇性 ≥2.6e5×)
```

(Note P4b also tightens the hand-wavy "private domains" into "proprietary or
safety-sensitive deployments where no public benchmark exists.")

---

## P5 (FROM->TO): trim the G3T9 cost paragraph so its headline survives

**Why.** G3T9-W1 is their #1 concern and the cost answer is strong, but the
E12b decode detail (token-length, ceiling caveat) is in-the-weeds and buries the
headline. Keep the honesty (lower-bound) and the number; drop the weeds. Shorter
is more likely read by a conf-4 reviewer.

**FROM** (draft L432-437):

```
additionally requires labels. The GSM8K component is measured, not assumed:
one greedy decode to natural EOS vs PRISM's exact teacher-forced call on
identical prompts, model, and batch = 556.7 s vs 8.9 s (>= 62.6x; 13% of
generations were still running at the 1024-token ceiling, so the decode
side is understated and the ratio is a lower bound; mean natural CoT length
208.8 tokens/prompt vs 102 gold tokens).
```

**TO**:

```
additionally requires labels. The GSM8K component is measured, not assumed: on
identical prompts, one greedy decode costs 556.7 s vs 8.9 s for PRISM's exact
teacher-forced call, at least 62.6x (a conservative lower bound; 13% of
generations had not reached EOS at the cap).
```

---

## P6 (FROM->TO): make pCi8-W2 open on the reviewer's word "good enough"

**Why.** pCi8 asked zero questions (disengaged) and their W2 verbatim is "cannot
say whether Q4 is good enough." The current lead is strong but starts with a
meta-comment ("The reviewer's exact question,..."). Open on the answer itself so
a skimming reviewer sees the payoff in the first four words.

**FROM** (draft L279-285):

```
The reviewer's exact question, "is Q4 good enough, not just better than Q2", 
is now answerable. Take the two Llama-3.1-8B/MMLU cells that most dramatize the
looseness: Q8_0 (raw bound 23.24 vs true |dR| 0.0002, slack ~10^5x) and Q2_K
(266.09 vs 0.366, ~727x). A one-time per-(family, benchmark) leave-one-out
isotonic calibration B -> |dR| maps those same bounds to 0.003 and 0.193 nats, 
correctly placing Q8_0 well within an epsilon = 0.1 tolerance and Q2_K clearly
outside it, despite the raw bounds being uninformative in absolute scale.
```

**TO**:

```
"Is Q4 good enough", not just "better than Q2", is now answerable. Take the two
Llama-3.1-8B/MMLU cells that most dramatize the looseness: Q8_0 (raw bound 23.24
vs true |dR| 0.0002, slack ~10^5x) and Q2_K (266.09 vs 0.366, ~727x). A one-time
per-(family, benchmark) leave-one-out isotonic calibration B -> |dR| maps those
same bounds to 0.003 and 0.193 nats, correctly placing Q8_0 well within an
epsilon = 0.1 tolerance and Q2_K clearly outside it, despite the raw bounds
being uninformative in absolute scale.
```

(Also removes the two trailing spaces on L279 and L283, an AI-writing tell.)

---

## P7 (FROM->TO, small): one-line-each on the four continual-learning methods for 8VrD

**Why.** 8VrD cited SLoRA / two-phase / CLAIM / ArMA with full references. The
current text (C and 8VrD-W3) says only that they are "sequential
continual-instruction settings." A conf-3 ally will not die on this, but naming
the distinction crisply shows we read their references and prevents "you dodged
my citations." The 8VrD-W3 response already does this well; the gap is in the
GLOBAL C response, which is the one the AC reads.

**FROM** (draft L165-167):

```
[TBD-E2, baseline table]. SLoRA, two-phase CIT, CLAIM, and ArMA address
sequential continual-instruction settings; we discuss all four in Related Work
and will add matched-setting comparisons in the revision.
```

**TO**:

```
[TBD-E2, baseline table]. SLoRA, two-phase CIT, CLAIM, and ArMA target the
SEQUENTIAL multi-task continual-instruction setting (a stream of tasks), whereas
our regularizer targets single-adapter shape drift diagnosed by the bound; a
matched comparison requires their sequential benchmarks, which we add in the
revision. All four are now cited.
```

---

## P9 (FROM->TO, content gap): reaffirm the paper's headline regularizer result so the honest gating does not read as a retreat

**Why.** The abstract and Sec. 5.4 claim the shape regularizer "outperforms
experience replay." The rebuttal's honest gating story (helps on shape-driven
cells, mixed elsewhere) is correct, but as written it can be misread as
*conceding the regularizer does not work*. A senior reviewer will re-read the
abstract. Reaffirm the actual aggregate result (Table 2 / Fig. 4: on
Llama-TruthfulQA trace cuts mean downstream forgetting by 19% vs replay's 9%)
FIRST, then present gating as the *explanation of the per-cell variation*, not
as a walk-back. Concede the scope (clearly wins on Llama-TQA), keep the win.

**FROM** (draft L224-228):

```
(3) Regularizer failures: Table 22 is, by construction, our diagnostic-gating
analysis: its "gating verdict" column already marks Llama-BBQ "cell-level
mixed" and both Qwen settings "at noise floor => skip"; trace still yields
per-benchmark wins on shape-driven cells (BBQ-FT TriviaQA -88%, GSM8K -79%)
and lifts Omega 0.93->0.98 where replay leaves it flat.
```

**TO**:

```
(3) Regularizer: the paper's aggregate result stands (Table 2 / Fig. 4: on
Llama-TruthfulQA the trace penalty cuts mean downstream forgetting by 19% vs
replay's 9%). The gating analysis explains the per-cell variation the reviewers
flag rather than contradicting the aggregate. Table 22 is, by construction, that
diagnostic-gating analysis: its "gating verdict" column already marks Llama-BBQ
"cell-level mixed" and both Qwen settings "at noise floor => skip"; trace still
yields per-benchmark wins on shape-driven cells (BBQ-FT TriviaQA -88%, GSM8K
-79%) and lifts Omega 0.93->0.98 where replay leaves it flat.
```

(The identical reaffirmation should also open 8VrD-W3 (2), whose FROM at draft
L798 currently begins "(2) Negative cells: we address the reviewer's exact
figures." Prepend one clause: "The aggregate win on Llama-TruthfulQA stands
(Table 2; -19% vs -9%); the negative cells are the per-cell variation, not a
reversal." Same move, same honesty, protects the headline in the ally's thread.)

---

## P8 (polish sweep, low ROI but zero risk)

Small wording/whitespace fixes that remove AI-writing tells and tighten flow.
Apply as a batch; none changes meaning.

- **L115** (`[TBD-E3B, ` has a trailing space) and **L630** (`p90 7.9%), ` trailing
  space): strip trailing spaces.
- **L78** the wrap `ranking only. Decomposing the slack: the Lipschitz-constant step`
  runs long; rebreak for readability (cosmetic).
- **pCi8-W1** (L267-269): "much as fever was known before the thermometer" is a good
  hook; keep, but ensure it appears only here and in the global spine (it does),
  so it reads as a frame, not a tic.

---

## Convergence

Judged converged against the following, which this proposal now meets:

1. **Every item is a concrete FROM->TO on the current draft**, not abstract advice;
   each cites exact text and a line anchor. [P1-P9]
2. **The highest-leverage move is an addition, not a defense**: the AC checklist
   (P1) turns the decision into a tick-box that proves the "concrete evidence"
   bar. This is where a BR->BA decision is actually won.
3. **Skim-defense and score-priming** are installed at the top of all four threads
   (P2, net-length-neutral because the openers let you cut in-item repetition),
   with the two Significance=2 reviewers additionally given a closing,
   deliberately non-presumptuous score-raise line (P4).
4. **The AC's stated price is paid in sentence one** (P3) and in the checklist (P1).
5. **Delivery is tightened where density was hiding the headline** (P5, P6) and the
   one place the AC-facing text under-answered the cited references is fixed (P7).
6. **The one content gap is closed** (P9): the paper's headline regularizer result
   is reaffirmed so the honest gating story reads as explanation, not retreat.
   This was the only place the current draft could be misread as conceding a
   central claim.
7. **No new claims, no overselling, no new numbers.** Every edit re-packages
   content already verified in the draft; the tie-with-CKA framing, the honest
   GSM8K negative, and the calibration numbers are untouched. The one honesty
   guard (P1) prevents the checklist from asserting the still-pending [E2]/[E3B]
   results.

Sober note: this proposal maximizes the odds of the flip; it cannot guarantee it.
Two things remain outside the text and are tracked in the draft's own checklist:
E2 must land so condition C has numbers rather than [TBD], and the GSM8K-artifact
upgrade waits on the E1 REDO. With those two in hand, P1-P4 give the AC and the
swing/ally reviewers everything they need to move the paper to acceptable.
Converged.
