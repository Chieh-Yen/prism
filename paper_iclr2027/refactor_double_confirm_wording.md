# Wording & Compression Audit (Double Confirm)

逐段掃描全文，依 **易讀性 ROI × 字數削減** 排序所有改進機會，目標 PDF 削減 ~20 行（保證易讀性 ≥ 現況）。

> **原則**：
> - 易讀性 > 字數削減
> - 只刪「冗詞、贅字、重複資訊、奇怪用字」
> - 任何改動若可能降低易讀性，**寧可放棄**
> - 已重複出現的資訊（在 conclusion / appendix）優先刪除主文版本

---

## A. Tier 1：高 ROI 大塊壓縮（每處 2-3 行）

### A1. Sec 5.2 L358 末句（TriviaQA Q2_K 例子）— 移除 ⭐️
**現況**：
> "As a concrete example: at the extreme Q2\_K TriviaQA case, $\rho_P$ exceeds $\rho_T$ only mildly ($\Delta\rho{\approx}1.3$), but $\Omega$ drops to $0.76$, driving the shape-mismatch arm to dominate the bound---a separation invisible to any scalar similarity metric."

**為何可刪**：
- Sec 5.3 「Shape distortion」段已有更 dramatic 的 examples（Llama-Q2_K MMLU: 24 vs 9000，Ministral-Q2_K MMLU: 9 vs 11000）
- 這個 TriviaQA 例子的 take-away 「shape dominates」已被 Sec 5.3 完整 cover
- "a separation invisible to any scalar similarity metric" 也在 Sec 5.3 末段重複表達

**節省**：~3 行
**易讀性影響**：✅ 反而提升（去重複）

---

### A2. Sec 5.3 L386「From diagnosis to remediation」段 — 大幅精簡 ⭐️
**現況**（3 句）：
> "Each dominant axis suggests a different remediation---per-channel outlier smoothing~\cite{xiao2023smoothquant} for scale collapse, Hessian-aware reconstruction for shape distortion, and FP16-\texttt{lm\_head} retention for head divergence. The bound thus turns from a scalar score into an axis-level diagnostic with actionable structure. We turn the differentiable shape arm into a training-time regularizer in Sec.~\ref{subsec:action}."

**為何可精簡**：
- 「per-channel outlier smoothing / Hessian-aware reconstruction / FP16-lm_head retention」三項 remediation **完全重複** Conclusion L420 limitations 段
- 「The bound thus turns from a scalar score into an axis-level diagnostic with actionable structure」是 marketing 語調，conclusion 已強調

**建議改寫**（兩個版本）：

**Option A2-a（最簡，僅保留 transition）**：
> "We turn the differentiable shape arm into a training-time regularizer in Sec.~\ref{subsec:action}."

刪除 3 句中的前 2 句，只留 transition。Conclusion limitations 段已完整列舉三個 protocol-level mitigations。

**Option A2-b（保留 axis-remediation pairing 但 cross-ref）**：
> "Each dominant axis suggests a matched protocol-level mitigation (discussed in Sec.~\ref{sec:conclusion}); we instead pursue the differentiable shape arm as a training-time regularizer in Sec.~\ref{subsec:action}."

**推薦 A2-a**：最徹底去重複，readers 翻 conclusion 就能找到 mitigations。

**節省**：~3 行（A2-a），~2 行（A2-b）
**易讀性影響**：✅ 提升（去重複，更直接 transition 到 5.4）

---

### A3. Sec 5.5 L410 末段冗長句 — 拆短 ⭐️
**現況**（單一巨大從句）：
> "...the full bound $\mathcal{B}_N$ attains the strongest aggregate ranking ($|r_s|{=}0.91$, $8/10$ cells)---**consistent with the head term being beneficial when the alignment can accommodate the non-identity rotations between $H_T$ and $H_P$, and with the non-monotonicity observed under $W{=}I$ tracing to the alignment rather than the bound design itself**."

**問題**：兩個 "consistent with" 子句連續、又長又抽象

**建議改寫**：
> "...the full bound $\mathcal{B}_N$ attains the strongest aggregate ranking ($|r_s|{=}0.91$, $8/10$ cells). The head term thus benefits the bound **when the alignment can absorb the rotation between $H_T$ and $H_P$**; the non-monotonicity at $W{=}I$ traces to the alignment, not the bound design."

**節省**：~2 行
**易讀性影響**：✅ 提升（兩短句 vs 一長從句）

---

## B. Tier 2：中度壓縮（每處 1 行）

### B1. Intro L160 三軸範例改 colon-arrow 結構 ⭐️
**現況**：
> "scale-dominated failures localize the loss to outlier-channel corruption, shape-dominated to distorted feature geometry between tokens, and head-dominated to head perturbation with the backbone intact."

**建議改寫**：
> "scale-dominated failures point to outlier-channel corruption, shape-dominated to distorted token-pair geometry, and head-dominated to head perturbation alone."

**節省**：~1 行（縮短形容詞片語）
**易讀性影響**：✅ 維持（簡化形容詞但保留意思）

---

### B2. Intro Para 1 L139-141 兩處冗詞
**現況**：
- "a new engineering bottleneck has emerged **downstream of pre-training**"
- "each of which **must be compared before a deployed model is selected**"

**建議改寫**：
- "a new engineering bottleneck emerges **after pre-training**"
- "each **requiring comparison before deployment**"

**節省**：~1 行
**易讀性影響**：✅ 提升（去 awkward "downstream of"，去被動語態）

---

### B3. Intro Para 2 L145 合併兩 claim
**現況**：
> "These scores, however, do not pin down downstream behavior, and a single number still flags drift without identifying an axis for diagnosis."

**建議改寫**：
> "These scores neither predict downstream behavior nor identify an axis for diagnosis."

**節省**：~1 行
**易讀性影響**：✅ 提升（neither/nor 結構更直接，去 "pin down" 口語）

---

### B4. Sec 5.1 L338 calibration 句壓縮
**現況**：
> "Because every variant sees the same subset, selection variance enters $|\Delta\mathcal{R}|$ and the bound identically and cannot affect their ranking; the orderings themselves span 1--2 orders of magnitude in $|\Delta\mathcal{R}|$ (e.g., Q2\_K vs.\ Q8\_0)."

**建議改寫**：
> "Selection variance enters $|\Delta\mathcal{R}|$ and the bound identically across variants, so it cannot affect ranking; orderings span 1--2 orders of magnitude in $|\Delta\mathcal{R}|$ (e.g., Q2\_K vs.\ Q8\_0)."

**節省**：~1 行
**易讀性影響**：✅ 提升（去 "Because" 從句，主動語態）

---

### B5. Sec 3.5 L294 過度宣稱句精簡 ⭐️
**現況**（earlier sentence.md 標記過 over-claim）：
> "Constraining shape should therefore stabilize target-task learning and suppress catastrophic forgetting on downstream tasks at once."

**問題**：「stabilize target-task learning」是 over-claim（regularizer 反而可能限制 target task fitting）

**建議改寫**：
> "Constraining shape should therefore curb backbone drift and suppress catastrophic forgetting on downstream tasks."

**節省**：~1 行
**易讀性影響**：✅ 提升（去 over-claim，更精準描述）

---

## C. Tier 3：細節精修（每處 < 1 行）

### C1. Sec 5 opening L320 去冗詞
**現況**：「around three claims **tied to the three contributions**: ...」
**建議**：「around three claims: ...」
**節省**：~5 字，少 0.3 行

### C2. Sec 5.2 LoRA L368 末句精簡
**現況**：「a single closed-form bound transfers across two **qualitatively different** sources of model variation」
**建議**：「the same bound transfers across two **distinct** sources of model variation」
**節省**：~3 字

### C3. Conclusion limitations L420 去 "themselves"
**現況**：「are **themselves** a focused empirical follow-up the diagnostic enables」
**建議**：「are a focused empirical follow-up the diagnostic enables」
**節省**：1 字

### C4. Intro Para 3 L149 (i) 收緊
**現況**：「the **established probe-based route** evaluates a freshly-trained linear classifier **instead of** the model's own head」
**建議**：「the established **probe-based methods** evaluate a freshly-trained classifier, **not** the model's own head」
**節省**：~3 字

### C5. RW L180 Para 1 去 "with the latter's...already carrying"
**現況**：「with the latter's squared Procrustes distance **already carrying** the nuclear norm」
**建議**：「the latter's squared Procrustes distance **already carries** the nuclear norm」
**節省**：~2 字（去 "with" 子句結構）

### C6. Sec 5.3 L380 末句精簡
**現況**：「The decomposition makes this protocol-level switch read off directly, identifying not just the magnitude of degradation but its dominant channel.」
**建議**：「The decomposition exposes this protocol-level switch directly, identifying not just the magnitude but the dominant channel.」
**節省**：~3 字

---

## D. 預期削減估計

| Tier | 改動 | 行數削減 | 累計 |
|---|---|---|---|
| A1 | 5.2 末句移除 | 3 | 3 |
| A2 | 5.3 remediation 段精簡 | 3 | 6 |
| A3 | 5.5 W_N 句拆短 | 2 | 8 |
| B1 | Intro 三軸範例 | 1 | 9 |
| B2 | Intro Para 1 冗詞 | 1 | 10 |
| B3 | Intro Para 2 合併 claim | 1 | 11 |
| B4 | Sec 5.1 calibration 句 | 1 | 12 |
| B5 | Sec 3.5 過度宣稱句 | 1 | 13 |
| C1-C6 | Tier 3 微調 | 累計 ~3 | **16-17** |

**預估總計：16-17 行 PDF 削減**（接近但未必達 20 行目標）

如要再衝 20 行，可考慮：

### Bonus 候選（風險較高，需 user 決策）

**E1. Conclusion limitations 段壓縮**
- 「PRISM is a *relative* metric: ... — the strong rank correlation reported in Sec 5.2 supports variant comparison but does not predict |ΔR| itself.」 第二從句可進一步壓縮
- 但這段是 limitations 防禦核心，動之需慎重

**E2. Sec 3.4 AR extension 段濃縮**
- 描述 corollary 應用，但細節在 Appendix D
- 可考慮把第二長句拆掉

---

## D'. 第二輪掃描：補充大塊壓縮候選（Tier 1 擴增）

> 為了衝 30+ 行目標，重新掃描補充以下高 ROI 候選。

### A4. Sec 4 L305 整段移除 ⭐️⭐️
**現況**：
> "The Unified Risk Bound (Theorem~\ref{thm:unified_bound}; Fig.~\ref{fig:prism_concept}) applies to two LLM lifecycle settings---post-training quantization and frozen-head LoRA fine-tuning---which differ in which term of the bound dominates the risk gap."

**為何整段可移除**：
- Section title 已寫「Applications: Post-Training Quantization and LoRA Forgetting」明示兩 settings
- 兩個 paragraph headers（L307「Quantization quality estimation」+ L311「Geometric monitoring of catastrophic forgetting」）名字直接 deliver 內容
- 「differ in which term of the bound dominates the risk gap」這個 claim 在每個 subsection body 都更具體陳述（Quantization 段說 γ vanishes vs GGUF k-quant；LoRA 段說 γ vanishes 因 frozen head）

**節省**：~2 行
**易讀性影響**：✅ 維持（資訊已在 title + subsection headers 提供）

---

### A5. Sec 3.3 L272 「PRISM admits a family」段壓縮 ⭐️
**現況**（兩長句）：
> "The family $\{\Omega_W\}$ of Prop.~\ref{prop:exact_decomposition} includes the trace form $\Omega_{W=I}$ and the nuclear form $\Omega_N := \Omega_{W=W_N}$ (Appendix~\ref{app:tightness}); a related Frobenius similarity $\Omega_F$, tied to CKA, is discussed there as an external comparison. The main text commits to $\Omega := \Omega_{W=I}$ because the frozen-head settings studied here---LoRA and FP16-head PTQ---keep $H_T = H_P$ in most configurations, so the head term simplifies at $W = I$ and leaves the risk gap tractable; the $W = W_N$ counterpart is reported as a Sec.~\ref{subsec:ablation} ablation."

**建議改寫**：
> "The family $\{\Omega_W\}$ of Prop.~\ref{prop:exact_decomposition} includes the trace form $\Omega := \Omega_{W=I}$ (used throughout) and the Procrustes-optimal nuclear form $\Omega_N := \Omega_{W=W_N}$ (Appendix~\ref{app:tightness}; reported as a Sec.~\ref{subsec:ablation} ablation). We adopt $\Omega$ because the frozen-head settings studied here keep $H_T = H_P$, simplifying the head term at $W = I$."

**節省**：~2 行
**易讀性影響**：✅ 維持（去重複的 "the frozen-head settings studied here" 描述）

---

### A6. Sec 3.4 L283 AR sentence 壓縮 ⭐️
**現況**：
> "Under teacher forcing, all token-level features $\phi_M(c, y_{<\tau})$ are extracted in a single forward pass and collected into a matrix $Z_M^{\mathrm{AR}} \in \mathbb{R}^{N \times d}$ (with $N$ the total number of target tokens across the calibration set); Theorem~\ref{thm:unified_bound} then applies directly to $(Z_T^{\mathrm{AR}}, Z_P^{\mathrm{AR}})$, unifying point-wise classification, language modeling, short-horizon QA, and multi-step reasoning under a single bound. The full corollary is in Appendix~\ref{app:ar_extension}."

**為何可壓縮**：
- 「unifying point-wise classification, language modeling, short-horizon QA, and multi-step reasoning under a single bound」——這四類 benchmark 在 Sec 5.1 又重複列出
- 「(with $N$ the total number of target tokens across the calibration set)」可省

**建議改寫**：
> "Under teacher forcing, token-level features $\phi_M(c, y_{<\tau})$ collect into $Z_M^{\mathrm{AR}} \in \mathbb{R}^{N \times d}$, and Theorem~\ref{thm:unified_bound} applies to $(Z_T^{\mathrm{AR}}, Z_P^{\mathrm{AR}})$ unchanged. Full corollary: Appendix~\ref{app:ar_extension}."

**節省**：~2 行
**易讀性影響**：✅ 維持（benchmark enumeration 重複，可省；技術內容仍完整）

---

### A7. Sec 5.5 L408 first paragraph「Two observations」結構壓縮 ⭐️
**現況**：
> "Two observations keep $\mathcal{B}$ the right \emph{default} metric. First, it is the certified upper bound of Theorem~\ref{thm:unified_bound} and is strictly required whenever $\gamma{>}0$; $\delta$ alone is not a valid bound in that setting. Second, $\mathcal{B}$ has the smallest Llama--Qwen gap ($\approx 0.015$ vs.\ $0.026$ and $0.042$ for $\delta$ and $\Omega$), so its ranking performance is the most robust across the two families tested."

**建議改寫**（合併兩 observations）：
> "Two reasons keep $\mathcal{B}$ as default: it is the certified upper bound of Theorem~\ref{thm:unified_bound} (required when $\gamma{>}0$), and it has the smallest Llama--Qwen gap ($\approx 0.015$ vs.\ $0.026$ and $0.042$ for $\delta$ and $\Omega$)---the most robust across families."

**節省**：~2 行
**易讀性影響**：✅ 維持（First/Second 的列表結構在這裡其實沒必要，分號連接更緊湊）

---

### A8. Sec 5.3 L380 末句移除 ⭐️
**現況**：
> "The decomposition makes this protocol-level switch read off directly, identifying not just the magnitude of degradation but its dominant channel."

**為何可移除**：
- 前文已具體展示「INT8 vs Q6_K → γ=0 vs γ=60.16」這個 dramatic contrast
- 這句是 meta-commentary，重複前文的 take-away
- 「magnitude vs channel」對比在 Sec 5.3 開頭「The decomposition contributes the next layer」已概括

**節省**：~1.5 行
**易讀性影響**：✅ 維持（去 meta-commentary，前文具體例已 deliver）

---

### A9. Sec 5.2 LoRA L368 末句移除 ⭐️
**現況**：
> "Together, the PTQ and LoRA results show that a single closed-form bound transfers across two qualitatively different sources of model variation."

**為何可移除**：
- 同段前文已說「comparable to the 0.82 obtained on the PTQ grid」——已 deliver「跨兩 settings」訊息
- 是 meta-summary 句，paper-wide 的 narrative arc 已透過 contributions + conclusion 強調

**節省**：~1.5 行
**易讀性影響**：✅ 維持（去重複，前文具體數字對比已強過 abstract claim）

---

### A10. Future Work bullets 壓縮 ⭐️
**現況**：3 bullets 各 ~3-4 行

**建議**：
- Bullet 1（Beyond LoRA forgetting）：移除「---where backbone drift is substantially larger---」破折號子句（~1 行）
- Bullet 2（Diagnostic applications）：濃縮 3 application 描述（~1 行）
- Bullet 3（Beyond LLMs）：兩短句合併成一句（~1 行）

**節省**：~3 行
**易讀性影響**：✅ 維持（仍清楚列出 3 application + 3 future work）

---

### A11. Related Work L180 末句精簡 ⭐️
**現況**：
> "Our formulation differs in retaining activation scale via $(\Delta\rho)^2$, modeling head mismatch through a covariance-weighted term, and lifting the Procrustes residual into a closed-form upper bound on cross-entropy risk evaluated on each model's deployed head."

**為何可精簡**：
- 三 differentiator 跟 contributions bullet 1 完全重複
- RW 末句的功能是「跟 prior 區別」，不需要展開全部技術細節

**建議改寫**：
> "Our formulation differs by retaining activation scale, modeling head mismatch through covariance weighting, and lifting the residual into a closed-form CE risk bound on each model's deployed head."

**節省**：~1 行
**易讀性影響**：✅ 維持（去 "via (Δρ)²" 重複數學，跟 contributions 不雙重 sell）

---

### A12. Sec 5.3 第一段 L374 末句省略
**現況**：
> "...the dominant axis of Theorem~\ref{thm:unified_bound} reads off directly from the per-variant numbers and corresponds to a distinct empirical failure mode. **Three settings recur in our experiments.**"

**為何可省**：
- 「Three settings recur」是 lead-in 句，後面三個 paragraph headers（Shape distortion / Head divergence / Scale-axis separability）已自帶 enumeration
- 句子本身不帶資訊

**建議**：移除「Three settings recur in our experiments.」
**節省**：~0.5 行
**易讀性影響**：✅ 維持

---

## D''. 重估總削減量

| 改動 | 行數削減 |
|---|---|
| **Tier 1 原版**（A1+A2+A3） | 8 |
| **Tier 1 新增**（A4+A5+A6+A7+A8+A9+A10+A11+A12） | 12-15 |
| **Tier 2**（B1-B5） | 5 |
| **Tier 3**（C1-C6） | 3 |
| **Bonus**（E1, E2） | +2-3 |
| **保守總計** | **~30-34 行** |

衝 50 行的可能性：
- 若同時 cut 一些 conclusion / RW 的更深層段落，理論上可達 35-40 行
- 但 ≥ 40 行需要犧牲一些重要 contribution claim，**不建議**

**現實 target：30-35 行（保守）/ 35-40 行（積極）**

---

## E'. 補充：Whole-paragraph 候選

### P1. Sec 5.3「From diagnosis to remediation」整段移除（與 A2 結合）
- 經 A2 後此段只剩 1 個 transition 句
- 可以乾脆完全 remove，讓 Sec 5.4 自然開始
- Sec 5.4 開頭已自帶「The decomposition of Sec 5.3 suggests an immediate intervention」做承接
- **節省：~3 行**

### P2. Sec 4 L305 paragraph 整段移除（A4 升級版）
- 已在 A4
- **節省：~2 行**

---

## F'. 套用建議（最終版）

按 ROI 排序：

**Phase 1：核心去重（最安全）**
- A1（5.2 末句）+ A2（5.3 remediation→1 句 or 0 句）+ P1（fold transition 入 5.4）

**Phase 2：Sec 5 + Sec 3 內部壓縮**
- A3（5.5 W_N 句）+ A7（5.5 Two observations）+ A8（5.3 末句）+ A9（5.2 末句）
- A5（3.3 PRISM family）+ A6（3.4 AR sentence）

**Phase 3：Intro + Conclusion 微調**
- B1-B5（intro 5 處）+ A10（FW bullets）+ A11（RW 末句）+ A12（5.3 lead-in）

**Phase 4（細節）**
- C1-C6 微調

預期 PDF 削減 **30-35 行**，可能衝到 40 行。**易讀性全程維持**。

---

## E. 全文掃出的 weird 用詞清單（NOT 建議改）

這些雖偏 informal 但已在 ML 接受範圍，**易讀性無損，不建議動**：
- "factor cleanly into"（Para 4）
- "doubles as"（Conclusion）
- "lifts into"（多處數學 idiom）
- "pin down"（Para 2）— B3 已順帶處理
- "non-monotone variance"（Sec 5.5）— 數學 jargon 在 context OK
- "in lockstep with"（appendix）— informal but vivid

---

## F. 套用順序建議

**第 1 波（最高 ROI）**：A1 + A2 + A3（共 8 行）
- 三處都是「重複資訊或冗長從句」直接砍
- 易讀性反而提升（去重複、拆長句）

**第 2 波（次高 ROI）**：B1-B5（共 5 行）
- Intro + Sec 5.1 + Sec 3.5 的句子精簡
- 每處 ~1 行，累計可觀

**第 3 波（細節）**：C1-C6（共 ~3 行）
- 字級微調

**第 4 波（如需衝 20 行）**：Bonus 候選 E1/E2
- 需 user 決策

---

## G. 總結：可削減 ~16-17 行（保守估計）

執行 A + B + C 三 tier 全部後：
- ✅ 易讀性提升（去重複、去 over-claim、拆長句）
- ✅ 字數削減 ~150 字
- ✅ PDF 預估削減 ~16-17 行
- ⚠️ 若要 ≥ 20 行，需考慮 Bonus E1/E2

要從 Tier 1 開始套用嗎？或者你想先 review 個別建議？
