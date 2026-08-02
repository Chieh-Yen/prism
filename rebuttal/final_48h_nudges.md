# T-48h 收尾貼文包(2026-08-02)

規則:零新數字(每個數字都已在已貼回覆中);每則 ≤2000 字元;不寫 "raise your
score",寫 "reflected in your final assessment";T-24h 後不再主動貼文,只回答。
貼文順序:1) 8VrD → 2) pCi8 → 3) G3T9 → 4) AC/global 導讀。eQL6 條件觸發(見末)。

--------------------------------------------------------------------------------
## 1. 8VrD thread(最優先;advocate)

**A one-screen map of the responses to your review, before the discussion closes**

Your review shaped the largest share of the new experiments in this rebuttal; with the window closing soon, a short map from each point to its posted answer (all numbers already appear in this thread):

- W1 (empirical vs population): Theorem 1 restated on the empirical gap; McDiarmid corollary with explicit constant added; the offer to post the restated theorem verbatim stands.
- W2 + Q1 (looseness, error analysis): quantified on all 570 cells (median 1597x, IQR [673, 3966]); your own two cells, after one-time per-cell calibration, land on the correct side of the 0.1-nat tolerance, and we reported that cell's full operating characteristic (LOO MAE 0.040, precision 0.75, one of the six weaker cells) rather than only its favourable rows.
- W3 + Q3 + Q4 (interventions, Table 22, baselines, reference sensitivity): controlled single-axis interventions: each family moves only its own term (>= 2.6e5x selectivity), bound holds 26/26; Table 22 read as the gating validation with the Llama-BBQ exception owned; EWC/L2-SP/layer-freezing added at matched plasticity over three seeds; the size-and-draw grid moves retention by at most 5.9 points, so the 32-sequence set is not load-bearing.
- W4 + Q2 (Qwen-BBQ cells, free-running): the full 20-cell matrix moves to the main text (18/20 positive) with both negative cells mechanistically explained; the rollout-conditioned experiment reaches r_s +0.947 vs +0.958 teacher-forced across five subsets.
- Limitations: all four items you listed are added.

If these resolve your concerns about scope and evidence, we would be grateful if that is reflected in your final assessment. If anything remains, there is still time and we will answer promptly.

--------------------------------------------------------------------------------
## 2. pCi8 thread

**A one-screen map of the responses to your review, before the discussion closes**

With the discussion window closing soon, a short map from each weakness to its posted answer (all numbers already appear in this thread):

- W1 (phenomena already known): the contribution is restated as the first quantitative instrument making the three axes commensurable (for one given variant: which axis, by how much, in one unit); the Introduction is revised accordingly.
- W2 (loose; "is Q4 actually good enough?"): the looseness is located (median slack 1597x; the Lipschitz step accounts for at most ~2.5x), and the per-cell calibration answers your question directly: Q4_K_M on Llama-MMLU predicts 0.055 nats against 0.036 measured, both under a 0.1-nat bar, at 1/62.6 the measured compute of one benchmark run on the same prompts.
- W3 (Omega alone vs full bound): the small pooled delta is a head-protocol mixing effect; restricted to the GGUF tiers where the head arm engages, the bound reaches 0.943 against 0.828 pooled; and 8 reference sequences already rank the variants as 512 do.
- W4 (GSM8K structural): agreed and kept as a first-class limitation with the mechanism quantified (mean gap ~0.019 nats, an SNR floor inherited by any CE-based proxy); the natural span-level mitigation we tested fails and is reported as failing.
- W5 (EWC): added, with L2-SP and layer-freezing, all at matched plasticity; the shape penalty leads every task-keeping baseline in all three seeds (0.680 ± 0.016) while also holding the lowest target-task loss (0.872).
- W6 (near-isometry under aggressive quantization): it affects tightness only, never validity, and the cost is measured: an unrestricted alignment shrinks the residual by under a fifth even at 2 bits.

If these resolve the practical-significance concern, we would be grateful if that is reflected in your final assessment; if anything remains, we would value the chance to address it before the window closes.

--------------------------------------------------------------------------------
## 3. G3T9 thread

**A one-screen map of the responses to your review, before the discussion closes**

With the discussion window closing soon, a short map from each point to its posted answer (all numbers already appear in this thread):

- W1 (why not just benchmark; cost; actionability): measured on identical prompts: PRISM 8.9 s vs 556.7 s greedy decode (62.6x; maj@8 501x), itemized as you asked (passes, reference data, realistic comparison); the diagnose-act-verify loop is concrete (head-axis case: 75.77 of 76.95 attributed, protocol fix verified at 20x lower bound), and single-axis interventions confirm each axis responds only to its own perturbation (bound holds 26/26).
- W2 (teacher-forcing; benchmark inputs; reference ablation): 8 sequences rank the variants as 512 do (r_s 0.932 both, three fresh seeds); the input-conditioned scope is stated explicitly as the narrowing the meta-review invites; the rollout-conditioned test reaches r_s +0.947 vs +0.958 teacher-forced.
- W3 (CKA/SVCCA; simpler regularizers): compared on identical features: a statistical tie at full size, which we state as parity rather than a win, with PRISM highest at the 8-sequence slice (0.932 vs 0.903 and 0.083); the regularizer table adds EWC, L2-SP and layer-freezing at matched plasticity, with the shape penalty leading every task-keeping baseline in all three seeds.
- W4 (loose bound): slack quantified (median 1597x) and calibrated per cell to MAE 0.055 nats with precision >= 0.8 at 0.1 nats in 49/55 cells.
- Q1 + Q2 (full FT/RLHF; mixed-task): scope stated precisely (validity is recipe-agnostic; calibrated tightness is App. C.3's regime and future work), with a base-to-instruct sanity check where the bound holds in all 10 cells and the head term engages as predicted.

If these resolve the motivation and evaluation concerns, we would be grateful if that is reflected in your final assessment; if anything remains, we would value the chance to address it before the window closes.

--------------------------------------------------------------------------------
## 4. Global / AC thread(reviewer 懶人包之後貼)

**Before the discussion closes: a one-screen index of the evidence**

With the discussion window closing in about 48 hours, we offer a short index of what was delivered against the meta-review's five points. Every number below was posted earlier in this thread; nothing here is new.

| point | headline result | where |
|:--|:--|:--|
| A. empirical or population; how loose; any threshold | Theorem restated on the empirical gap + concentration corollary; slack located (median 1597x; Lipschitz step at most ~2.5x); LOO-calibrated predictor: MAE 0.055 nats, precision >= 0.8 at 0.1 nats in 49/55 cells | "Point A" comment |
| B. reference data: how little, what domain | 8 sequences rank the variants as well as 512 (r_s 0.932 at both); the regularizer's reference moves retention by at most 5.9 points across sizes and disjoint draws; input-conditioned scope stated as the invited narrowing | "Point B" |
| C. stronger baselines | Same-features CKA/SVCCA: parity at full size, PRISM highest at n=8 (0.932 vs 0.903 and 0.083); shape penalty leads EWC, L2-SP and replay in every seed at matched plasticity (0.680 vs best baseline 0.751), with the lowest target-task loss (0.872) | "Point C" |
| D. free-running, or narrow | Both: rollout-conditioned r_s +0.947 vs +0.958 teacher-forced, cross +0.958; Corollary 1 restated for fixed shared trajectories | "Point D" |
| E. failures and mixed results | All 20 LoRA cells reported (mean +0.71, median +0.93, 18/20 positive); both negative cells mechanistically explained; the one genuine Table 22 exception owned | "Point E" |

One development since those posts: reviewer eQL6's follow-up led to a theoretical sharpening we adopted. The diagnosis is provably invariant under paired orthogonal reparameterization, and explicitly coordinate-relative beyond O(d); a formal remark and an explicit limitation are both committed to the revision. We believe the framework is more precisely scoped for it.

If any point is still judged partial, there is time remaining and we would welcome the chance to complete it. Thank you again for a review process that has concretely improved the paper.

--------------------------------------------------------------------------------
## 5. eQL6(條件觸發:T-24h 仍無回覆才貼;否則不貼)

Thank you again for the exchange on coordinate dependence. As the window closes soon: we hope the O(d)-equivariance boundary, the explicit GL(d) limitation, and the native-coordinate scope resolve the ambiguity you identified; if anything remains unclear, we are glad to address it today. If your concerns are resolved, we would be grateful if that is reflected in your final assessment.

--------------------------------------------------------------------------------
## 貼文前 checklist
- [ ] 每則數字 grep 一次已貼回覆(必須全部命中;零新數字)
- [ ] 各則 <2000 字元、表格在 OpenReview 預覽正常
- [ ] 粗體標題放進 OpenReview 的 Title 欄位,內文從第一句開始
- [ ] "about 48 hours" 改成貼文當下的實際剩餘時間
- [ ] "final assessment" 句每個 thread 全程只出現這一次;不加 sincerely/署名;
      respectfully 不再使用(global response 已用過一次)
- [ ] 順序:8VrD → pCi8 → G3T9 → global;eQL6 條件觸發(T-24h 仍無回覆才貼)
- [ ] T-24h 後只回覆、不再主動貼
- [ ] 任何 reviewer 回覆後數小時內回應(即使只是致謝 + 修訂承諾確認)
