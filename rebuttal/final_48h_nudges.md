# T-48h 收尾貼文包(2026-08-02;v3 = v2 精簡版 + 標題/粗體導航)

規則:零新數字(每個數字都已在已貼回覆中);每則 ~1500 字元(30 秒導讀,
不是第三份 rebuttal——每 bullet 一個結論 + 一個數字,論證留在 thread 裡);
不寫 "raise your score",寫 "reflected in your final assessment"。
時程:T-48h 貼 1) 8VrD → 2) pCi8 → 3) G3T9;**AC/global 導讀留到 T-24h~T-12h
再貼**(AC 在窗口關閉後才行動,晚貼零損失,且可吸收最後一天的 reviewer 發展、
確保作者整理是 thread 的收尾;硬底線 T-12h,不要更晚)。eQL6 條件觸發(見末)。
reviewer nudge 貼出後,T-24h 前只回答不再主動貼;AC 導讀是唯一例外。

--------------------------------------------------------------------------------
## 1. 8VrD thread(最優先;advocate)

Title 欄:Summary of Responses Before the Discussion Closes

**A one-screen map of the responses to your review**

Your review shaped the largest share of the new experiments in this rebuttal. With the window closing soon, a short map from each point to its posted answer; the details and tables are in the comments above, and nothing here is new.

- **W1** (empirical vs population): Theorem 1 restated on the empirical gap, McDiarmid corollary added; the offer to post it verbatim stands.
- **W2 + Q1** (looseness, error analysis): slack quantified on all 570 cells (median 1597x); your two example cells calibrate to the correct side of the 0.1-nat tolerance, with that cell's weak spots reported too (MAE 0.040, precision 0.75).
- **W3 + Q3 + Q4** (interventions, Table 22, baselines, reference sensitivity): single-axis interventions, each moving only its own term (>= 2.6e5x selectivity, bound holds 26/26); Table 22 as the gating validation, the Llama-BBQ exception owned (the gate still separates its cells: -88%/-79%); EWC, L2-SP and layer-freezing at matched plasticity over three seeds; the reference size-and-draw grid moves retention by at most 5.9 points, with a WikiText domain probe reported as a single-seed observation.
- **W4 + Q2** (Qwen-BBQ cells, free-running): the full 20-cell matrix moves to the main text (18/20 positive), both negatives mechanistically explained; rollout r_s +0.947 vs +0.958 teacher-forced, and the teacher-forced bound orders rollout gaps at +0.958.
- **Limitations**: all four items added.

If these resolve your concerns, we would be grateful if that is reflected in your final assessment. If anything remains, there is still time and we will answer promptly.

--------------------------------------------------------------------------------
## 2. pCi8 thread

Title 欄:Summary of Responses Before the Discussion Closes

**A one-screen map of the responses to your review**

With the window closing soon, a short map from each weakness to its posted answer; details are in the comments above, and nothing here is new.

- **W1** (phenomena already known): the contribution restated as commensurable per-variant attribution (which axis, by how much, in one unit); Introduction revised accordingly.
- **W2** (loose; "is Q4 actually good enough?"): the looseness located (median 1597x; Lipschitz step at most ~2.5x), and the calibration answers your question directly: Q4_K_M predicts 0.055 vs 0.036 measured, both under a 0.1-nat bar, at 1/62.6 the measured cost of a benchmark run.
- **W3** (Omega alone vs full bound): the small delta is a head-protocol mixing effect (GGUF-only 0.943 vs 0.828 pooled), and the machinery's other outputs (axis attribution, the differentiable penalty, the 62.6x cost gap) never appear in a Spearman.
- **W4** (GSM8K structural): kept as a first-class limitation (mean gap ~0.019 nats vs 0.07-0.16 elsewhere); the tested mitigation fails and is reported as failing; the boundary is gap magnitude, not reasoning (W_N ordering recovers to 0.965; under LoRA, where the gap is 0.134, it ranks at +0.97).
- **W5** (EWC): EWC, L2-SP and layer-freezing added at matched plasticity; the shape penalty leads every task-keeping baseline in all three seeds (0.680 ± 0.016) with the lowest target loss (0.872).
- **W6** (near-isometry): affects tightness only, never validity; measured, an unrestricted alignment shrinks the residual by under a fifth even at 2 bits.

If these resolve the practical-significance concern, we would be grateful if that is reflected in your final assessment; if anything remains, we would value the chance to address it before the window closes.

--------------------------------------------------------------------------------
## 3. G3T9 thread

Title 欄:Summary of Responses Before the Discussion Closes

**A one-screen map of the responses to your review**

With the window closing soon, a short map from each point to its posted answer; details are in the comments above, and nothing here is new.

- **W1** (why not benchmark; cost; actionability): cost measured on identical prompts (PRISM 8.9 s vs 556.7 s greedy decode, 62.6x), itemized as you asked; the diagnose-act-verify loop closes (head case: 75.77 of 76.95 attributed, fix verified at a 20x lower bound); interventions hold 26/26.
- **W2** (teacher-forcing; benchmark inputs; reference ablation): 8 sequences rank the variants as 512 do (r_s 0.932 at both, three fresh seeds); the input-conditioned scope is stated as the narrowing the meta-review invites; the rollout-conditioned test reaches +0.947 vs +0.958 teacher-forced.
- **W3** (CKA/SVCCA; simpler regularizers): on identical features, a statistical tie at full size, stated as parity rather than a win, with PRISM highest at n=8 (0.932 vs 0.903 and 0.083); the simplest feature-preserving penalty nearly coincides with the shape penalty (Prop. 1), so the distinct families added are weight-space and architectural, at matched plasticity.
- **W4** (loose bound): slack quantified (median 1597x) and calibrated per cell to MAE 0.055 nats, precision >= 0.8 in 49/55 cells.
- **Q1 + Q2** (full FT/RLHF; mixed-task): validity is recipe-agnostic, calibrated tightness deferred (App. C.3); the base-to-instruct sanity check holds in all 10 cells with the head term engaging as predicted.

If these resolve the motivation and evaluation concerns, we would be grateful if that is reflected in your final assessment; if anything remains, we would value the chance to address it before the window closes.

--------------------------------------------------------------------------------
## 4. Global / AC thread(**T-24h~T-12h 再貼**;貼前檢查是否有 reviewer 新回應要先處理)

Title 欄:Closing Index: Evidence Against the Five Meta-Review Points

**Before the discussion closes: a one-screen index of the evidence**

As the discussion window closes, we offer a short index of what was delivered against the meta-review's five points. Every number below was posted earlier in this thread; nothing here is new.

| point | headline result | where |
|:--|:--|:--|
| **A.** empirical or population; how loose; any threshold | Theorem restated on the empirical gap + concentration corollary; slack located (median 1597x; Lipschitz step at most ~2.5x); LOO-calibrated predictor: MAE 0.055 nats, precision >= 0.8 at 0.1 nats in 49/55 cells | "Point A" comment |
| **B.** reference data: how little, what domain | 8 sequences rank the variants as well as 512 (r_s 0.932 at both); the regularizer's reference moves retention by at most 5.9 points across sizes and disjoint draws; input-conditioned scope stated as the invited narrowing | "Point B" |
| **C.** stronger baselines | Same-features CKA/SVCCA: parity at full size, PRISM highest at n=8 (0.932 vs 0.903 and 0.083); shape penalty leads EWC, L2-SP and replay in every seed at matched plasticity (0.680 vs best baseline 0.751), with the lowest target-task loss (0.872) | "Point C" |
| **D.** free-running, or narrow | Both: rollout-conditioned r_s +0.947 vs +0.958 teacher-forced, cross +0.958; Corollary 1 restated for fixed shared trajectories | "Point D" |
| **E.** failures and mixed results | All 20 LoRA cells reported (mean +0.71, median +0.93, 18/20 positive); both negative cells mechanistically explained; the one genuine Table 22 exception owned | "Point E" |

One development since those posts: reviewer eQL6's follow-up led to a theoretical sharpening we adopted. The diagnosis is provably invariant under paired orthogonal reparameterization, and explicitly coordinate-relative beyond O(d); a formal remark and an explicit limitation are both committed to the revision. We believe the framework is more precisely scoped for it.

[發展插槽:貼出前檢查最後一天的 thread。若有 reviewer 正面確認,僅在屬實時加一行,
例:"Reviewer [X] has since confirmed that [W?/Q?] are resolved." 若有新問題,先在
該 thread 答完再貼本索引,讓這份整理是 thread 的收尾;無發展則整段刪除。]

If any point is still judged partial, there is time remaining and we would welcome the chance to complete it. Thank you again for a review process that has concretely improved the paper.

--------------------------------------------------------------------------------
## 5. eQL6(條件觸發:T-24h 仍無回覆才貼;否則不貼)

Title 欄:Closing Note on the Coordinate-Dependence Exchange

Thank you again for the exchange on coordinate dependence. As the window closes soon: we hope the O(d)-equivariance boundary, the explicit GL(d) limitation, and the native-coordinate scope resolve the ambiguity you identified; if anything remains unclear, we are glad to address it today. If your concerns are resolved, we would be grateful if that is reflected in your final assessment.

--------------------------------------------------------------------------------
## 貼文前 checklist
- [ ] 每則數字 grep 一次已貼回覆(必須全部命中;零新數字)
- [ ] 每 bullet ≤2 行:結論 + 一個數字;不重演論證(論證在 thread 裡)
- [ ] Title 欄照各節「Title 欄:」行填;body 從粗體標題行開始
- [ ] "final assessment" 句每個 thread 全程只出現這一次;不加 sincerely/署名;
      respectfully 不再使用(global response 已用過一次)
- [ ] T-48h:8VrD → pCi8 → G3T9;**AC 導讀留到 T-24h~T-12h**(硬底線 T-12h),
      貼前處理發展插槽;eQL6 條件觸發(T-24h 仍無回覆才貼)
- [ ] AC 導讀貼出前,若有未回答的 reviewer 新問題,先答完再貼(讓索引收尾)
- [ ] reviewer nudge 之後、AC 導讀之外,不再主動貼文,只回答
- [ ] 任何 reviewer 回覆後數小時內回應(即使只是致謝 + 修訂承諾確認)
