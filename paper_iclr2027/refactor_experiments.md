# Experiments Section Rewrite Plan

> **目標**：以易讀性 > 故事性精準傳達實驗最重要內容，總行數比現在 **少 10 行**。
>
> **流程**：本檔案會多次迭代——Round 1 列分析、Round 2 細修建議、Round 3 確認改寫方案。

---

## Round 1 — 內容重要性分析

### 核心 message（必保）

| 主題 | Key claim | Section |
|---|---|---|
| **Predictiveness** | $\mathcal{B}$ rank-tracks $\|\Delta\mathcal{R}\|$ — PTQ $0.82\pm0.05$、LoRA $0.831\pm0.07$ | 5.2 |
| **Decomposability** | 三 axes 對應三 distinct failure modes（shape/head/scale-shape contrast） | 5.3 |
| **Actionability** | Differentiable shape arm 成 regularizer，monotone reduce forgetting | 5.4 |
| **Ablation** | $\Omega$ alone 已 carry 大部分 signal；$\mathcal{B}$ 跨 family gap 最小 | 5.5 |

### 必保的具體數字

- 0.82 PTQ + 0.831 LoRA Spearman + SEM
- Q2 shape >> scale **2-3 orders of magnitude**（Llama-Q2_K MMLU: 24 vs 9000）
- Qwen3 Q6_K: $\gamma=60.16$ >> $\delta=0.75$（head dominates）
- BnB INT8 同 Qwen: $\gamma=0$, $\delta=13.2$（protocol-level switch）
- LoRA TruthfulQA: $\Omega$ ARC 0.82→0.93, MMLU 0.87→0.95, GSM8K 0.94→1.00
- Ablation: $\Omega$ 0.80, $\delta$ 0.87, $\mathcal{B}$ 0.82, $\mathcal{B}_N$ 0.91

### 必保的 figures / tables

- Fig 2 (PTQ grid): 主 evidence
- Fig 3 (LoRA grid): 主 evidence
- Fig 4 (shape reg): 主 evidence
- Table 1 (Llama decomposition): 數值佐證
- Table 5 (Llama TruthfulQA trace norm): regularizer 數據
- Table 4 (baseline_combined ablation): component-wise

---

## Round 1 — 句子層級必要性 audit

### Sec 5 intro (L322)
> "We organize the experiments around three claims: \textbf{Predictiveness}—...; \textbf{Decomposability}—...; and \textbf{Actionability}—...."

**評估**：✅ 必保。Trio 結構建立。

### Sec 5.1 Models (L327-328)
- 主句：Llama+Qwen main, others appendix → ✅ 必保
- "differing in pre-training corpus, depth, vocabulary, and tokenizer family" → ⚠️ 細節已在 appendix `forgetting_qwen.tex` 引用——**可省**

### Sec 5.1 Quantization Protocols (L331)
- ✅ 已 compact，無 cut 空間

### Sec 5.1 Fine-Tuning Tasks (L333-334)
- "factual grounding" / "Bias Benchmark for QA, social-context reasoning" → 可省，task name 自帶 hint
- "two tasks chosen to elicit different drift geometries (Sec.~\ref{subsec:decompose})" → forward ref 可保
- **可省 1 行**

### Sec 5.1 Benchmarks and Scoring (L337)
- "Following Eq.~(\ref{eq:ar_risk}), five benchmarks: ..." → "Following Eq.X" 可省
- "matching each benchmark's scoring rule by construction and producing deterministic per-sample residuals on the same scale as the LoRA fine-tuning objective" → 後半子句冗長，可壓
- **可省 ~1 行**

### Sec 5.1 Calibration and Hyperparameters (L340)
- "Selection variance enters $|\Delta\mathcal{R}|$ and the bound identically across variants, so it cannot affect ranking; the orderings span 1--2 orders of magnitude in $|\Delta\mathcal{R}|$ (e.g., Q2\_K vs.\ Q8\_0)" → defensive justification of subset choice
- 對 reviewer 攻擊 calibration size 有用，但偏 defensive
- **保留**（已是 R2.3 之前留下的；移除會弱化 calibration 邏輯）
- "All experiments use a single NVIDIA RTX 5090 (32~GB)" → ✅ 必保（Q8 checklist 引用）

### Sec 5.2 Quantization paragraph (L358)
- 主句：Spearman 0.82 across 2x5 grid + appendix replication → ✅ 必保
- "Two patterns hold consistently: (i) bit-width drives the bound monotonically...; (ii) family ordering is preserved across benchmarks—..."
  - (i) 重要：bit-width → bound 單調 → 直接視覺 evidence
  - (ii) 描述 GPTQ vs BnB cluster pattern → 細節，**可大幅壓縮或移除**
- **可省 ~2 行**（壓縮 (ii)）

### Sec 5.2 Table ref + concrete example (L360)
- "Table~\ref{tab:llama_decomposition_main} reports..." → ✅ 表 reference 必保
- "the Qwen3-8B counterpart, the remaining Llama benchmarks, the feature-only ($\delta$) scatter, parallel language-modeling results, and Ministral/DeepSeek decompositions are in Appendices~..." → 6 個 appendix 引用，**可壓縮為 2-3 個**
- "**As a concrete example**: at the extreme Q2\_K TriviaQA case, $\rho_P$ exceeds $\rho_T$ only mildly ($\Delta\rho{\approx}1.3$), but $\Omega$ drops to $0.76$, driving the shape-mismatch arm to dominate the bound—a separation invisible to any scalar similarity metric." → **與 Sec 5.3 Q2/Q3 shape distortion 段重複（更 dramatic 數字在 5.3）**
- **可省 ~3 行**（移除 concrete example + 壓縮 appendix list）

### Sec 5.2 LoRA forgetting (L370)
- 開頭 + Spearman 數字 + comparable to PTQ → ✅ 必保
- "Because the head is frozen, $\gamma{=}0$ and $\mathcal{B}$ reduces to backbone scale and shape drift" → ✅ 必保（解釋為何 $\gamma=0$）
- "Crucially, the correlation holds \emph{benchmark-by-benchmark}—each of ARC, MMLU, SQuAD, TriviaQA, GSM8K (including GSM8K, whose distribution differs substantially from the fine-tuning data)—consistent with backbone geometry being the locus of the forgetting signal across distinct evaluation families." → **整段可壓**——5 benchmark 列表已在 caption；GSM8K 特例可保
- "Together, the PTQ and LoRA results show that a single closed-form bound transfers across two qualitatively different sources of model variation." → 主動結論句，與 Sec 5.3 lead-in 重複，**可省**
- **可省 ~2 行**

### Sec 5.3 Lead-in (L376)
- "A scalar correlation alone would already match prior work on representational similarity. The decomposition contributes the next layer: the dominant axis of Theorem~\ref{thm:unified_bound} reads off directly from the per-variant numbers and corresponds to a distinct empirical failure mode. Three such failure modes recur in our experiments."
- 第一句是 RW 對比；第二句是核心 framing；第三句是 enumeration trigger
- 可壓為 **1 句**："The decomposition reads the dominant axis of Theorem 1 from per-variant numbers; three failure modes recur."
- **可省 1 行**

### Sec 5.3 Shape distortion (L378-379)
- "At Q2 and Q3 across all four families tested..., the shape term outweighs the scale term by 2-3 orders of magnitude (e.g., Llama-Q2\_K on MMLU: 24 vs 9000; Ministral-Q2\_K on MMLU: 9 vs 11000)" → **兩 examples**，可保留一個（Llama 更熟悉）
- "Scale drift is measurable across the PTQ grid but does not cleanly dominate any single variant; the most scale-affected case is Qwen3-Base Q2\_K on GSM8K..." → 處理 B6 ρ inflation case
- **可省 ~1 行**（去 Ministral example）

### Sec 5.3 Head divergence (L381-382)
- 細數字（δ, γ）必保
- "Q8\_0 shows the same pattern: δ=0.15, γ=19.0" → 第二例；可考慮省
- "The decomposition makes this protocol-level switch read off directly, identifying not just the magnitude of degradation but its dominant channel." → meta-commentary；前文具體例已 deliver
- **可省 ~1.5 行**

### Sec 5.3 Scale-axis separability (L385)
- 已 short；R3.1 user 決定 skip 軟化
- ⚠️ 「scale axis is empirically separable」用詞風險仍在，但本 round 不討論這個（focus 字數）

### Sec 5.3 From diagnosis to remediation (L387-388)
- "Each dominant axis suggests a different remediation—per-channel outlier smoothing for scale collapse, Hessian-aware reconstruction for shape distortion, FP16-lm_head retention for head divergence."
- **完全與 Conclusion limitations (L426) 重複**：「protocol-level mitigations for the scale and head axes (per-channel outlier smoothing, Hessian-aware reconstruction, FP16-lm_head retention)」
- "The bound thus turns from a scalar score into an axis-level diagnostic with actionable structure." → marketing 語調，前文已 deliver
- "We turn the differentiable shape arm into a training-time regularizer in Sec.~\ref{subsec:action}." → transition 句，可保留（折衷只保 transition）
- **可省 ~3 行**（保 transition only）

### Sec 5.4 motivation (L404)
- "The decomposition of Sec.~\ref{subsec:decompose} suggests an immediate intervention: since shape drift dominates LoRA forgetting and $1{-}\Omega$ is differentiable in $Z_t$, penalizing it during fine-tuning should preserve downstream task knowledge." → 完整論述
- "Fig.~... and Table~... answer the question raised by Sec.~\ref{subsec:shape_reg}: \emph{does minimizing $(1{-}\Omega)$ during fine-tuning actually suppress forgetting?}" → 自問自答，可省問句
- "For Llama-3.1-8B fine-tuned on TruthfulQA, Table~... reports the decomposition at step $300$: as $\lambda$ rises from $0.0$ to $0.5$, $\Omega$ on downstream benchmarks increases monotonically (e.g., ARC $0.82{\to}0.86{\to}0.93$, MMLU ..., GSM8K ...), and downstream $|\Delta\mathcal{R}|$ drops on every benchmark." → 數據展示，**必保**
- **可省 ~1 行**（去自問自答）

### Sec 5.4 patterns (L406)
- (i) consistency across benchmarks
- (ii) monotone & controllable trade-off
- (iii) λ=0.1 already eliminates substantial fraction of forgetting
- (ii) 與 (iii) 部分 overlap：「monotone」對應「λ↑ → 強 preservation」，「λ=0.1 already」是「monotone 起手就有效」
- 可 merge (ii)+(iii) 為一個 pattern：「monotone trade-off; even λ=0.1 already eliminates substantial fraction」
- **可省 ~1 行**

### Sec 5.5 W=I block (L414)
- "Adding the scale term to form the feature alignment error $\delta$ lifts the mean by $+0.064$ to $0.87$ and wins the plurality of cells, making scale collapse the single largest per-component contributor." → ✅ key finding
- "Adding the covariance-weighted head term to form the full bound $\mathcal{B}$ brings the aggregate mean slightly down to $0.82$: because our suite mixes protocols that quantize the output embedding ($\gamma{>}0$ for GGUF k-quant tiers) with protocols that keep it in FP16 ($\gamma{\equiv}0$ for GPTQ and BnB), the $\gamma$ term injects non-monotone variance across the pooled scatter." → 解釋為何 head term hurts under W=I → ✅ 必保但 verbose
- "Two observations keep $\mathcal{B}$ the right \emph{default} metric. First, it is the certified upper bound of Theorem~\ref{thm:unified_bound} and is strictly required whenever $\gamma{>}0$; $\delta$ alone is not a valid bound in that setting. Second, $\mathcal{B}$ has the smallest Llama--Qwen gap ($\approx 0.015$ vs.\ $0.026$ and $0.042$ for $\delta$ and $\Omega$), so its ranking performance is the most robust across the two families tested." → 兩 reasons defending B as default → 可壓兩 reasons 為一段子
- **可省 ~2 行**

### Sec 5.5 W_N block (L416)
- "Under the Procrustes-optimal alignment $W{=}W_N$ ..., the cumulative pattern reverses cleanly..." → key finding
- "The head term thus benefits the bound \emph{when the alignment can absorb the rotation between $H_T$ and $H_P$}; the non-monotonicity at $W{=}I$ traces to the alignment, not the bound design." → ✅ 必保
- "The two blocks together support Theorem~\ref{thm:unified_bound} at $W_N$ and quantify the trade-off at $W{=}I$: the trace specialization sacrifices $\sim$$0.09$ Spearman for autograd compatibility (no SVD per step), the head-term simplification under frozen \texttt{lm\_head}, and consistency with the regularizer of Sec.~\ref{subsec:shape_reg}." → 三 reasons 列舉，可壓
- "We continue to recommend $\mathcal{B}$ as our default and $\mathcal{B}_N$ as a post-hoc reference when the additional SVD cost is acceptable." → recommendation
- **可省 ~1 行**

---

## Round 1 — 預估省行統計

| 位置 | 操作 | 預估省行 |
|---|---|---|
| Sec 5.1 Models | 去 family-difference list | 0.5 |
| Sec 5.1 Fine-Tuning Tasks | 去 task descriptions | 0.5 |
| Sec 5.1 Benchmarks | 壓尾子句 | 1 |
| Sec 5.2 Quantization (ii) | 壓 family ordering 描述 | 1.5 |
| Sec 5.2 table refs + Q2_K example | 壓 appendix list + 移除 example | 3 |
| Sec 5.2 LoRA closing | 去 benchmark-by-benchmark + closing summary | 2 |
| Sec 5.3 Lead-in | 壓為 1 句 | 1 |
| Sec 5.3 Shape distortion | 去 Ministral example | 1 |
| Sec 5.3 Head divergence | 去 Q8_0 重例 + meta-commentary | 1.5 |
| Sec 5.3 Remediation | 保 transition only | 3 |
| Sec 5.4 motivation | 去自問自答 | 1 |
| Sec 5.4 patterns | merge (ii)+(iii) | 1 |
| Sec 5.5 W=I block | 壓 two observations | 2 |
| Sec 5.5 W_N block | 壓 trade-off 三項列舉 | 1 |
| **總計** | | **~20 行** |

**緩衝**：~20 行候選 → 選 ~12-15 行落實，留 5-8 行緩衝。

---

## Round 1 — 推薦的 12 個改動（優先級排序）

按 **「省行 × 易讀性提升 / 風險」** 排序：

### 🔴 高優先級（高省行 + 改善易讀性）

1. **Sec 5.3 Remediation 段只保 transition**（省 3 行）
   - 完全與 Conclusion limitations 重複；移除無資訊損失
2. **Sec 5.2 移除 Q2_K TriviaQA example**（省 ~2 行）
   - Sec 5.3 shape distortion 段已有更 dramatic 數字
3. **Sec 5.2 壓縮 appendix list**（省 ~1 行）
   - 5+ 個 appendix 引用可合併

### 🟡 中優先級（明確省行 + 適度改善）

4. **Sec 5.2 LoRA closing summary 移除**（省 1 行）
   - "Together, the PTQ and LoRA results show..." 與 Sec 5.3 lead-in 重複
5. **Sec 5.3 Lead-in 壓為 1 句**（省 1 行）
6. **Sec 5.5 Two observations 壓縮**（省 ~2 行）
7. **Sec 5.3 Shape distortion 去 Ministral example**（省 1 行）

### 🟢 低優先級（細修）

8. Sec 5.1 Models 去 family-difference 子句（省 0.5 行）
9. Sec 5.1 Benchmarks 壓尾子句（省 1 行）
10. Sec 5.4 去自問自答（省 1 行）
11. Sec 5.4 merge patterns (ii)+(iii)（省 1 行）
12. Sec 5.5 W_N block 三項壓縮（省 1 行）

---

## Round 2 — 高優先級改動的具體 before/after 草案

### 改動 1：Sec 5.3 Remediation 段完全移除（省 3 行）

**Before** (L387-388):
```
Each dominant axis suggests a different remediation---per-channel outlier
smoothing~\cite{xiao2023smoothquant} for scale collapse, Hessian-aware
reconstruction for shape distortion, and FP16-\texttt{lm\_head} retention
for head divergence. The bound thus turns from a scalar score into an
axis-level diagnostic with actionable structure. We turn the differentiable
shape arm into a training-time regularizer in Sec.~\ref{subsec:action}.
```

**After**: 整段移除。

**Risk**：失去 Sec 5.3 → Sec 5.4 的顯式 transition。
**Mitigation**：Sec 5.4 開頭「The decomposition of Sec.~\ref{subsec:decompose} suggests an immediate intervention...」已是自然 transition，reader 不會迷路。
**Risk on duplicate avoidance**：Conclusion L426 已完整列舉同 3 mitigations，移除 5.3 不損失資訊。

**省行**：3 行

---

### 改動 2：Sec 5.2 移除 Q2_K TriviaQA concrete example（省 2 行）

**Before** (L360 後半):
```
... and Ministral/DeepSeek decompositions are in
Appendices~\ref{app:feature_only}--\ref{app:per_model_tables}.
As a concrete example: at the extreme Q2\_K TriviaQA case, $\rho_P$
exceeds $\rho_T$ only mildly ($\Delta\rho{\approx}1.3$), but $\Omega$
drops to $0.76$, driving the shape-mismatch arm to dominate the bound---a
separation invisible to any scalar similarity metric.
```

**After**: 移除 "As a concrete example: ..." 至段尾整句。

**Risk**：失去早期具體 example 引發興趣。
**Mitigation**：Sec 5.3 shape distortion 段有更 dramatic 同類 example（Llama-Q2_K MMLU: 24 vs 9000）；reader 在 Sec 5.3 仍會 see 具體 example。

**省行**：2 行

---

### 改動 3：Sec 5.2 壓縮 appendix list（省 1 行）

**Before** (L360 前半):
```
Table~\ref{tab:llama_decomposition_main} reports the per-variant
decomposition for Llama-3.1-8B on MMLU and TriviaQA; the Qwen3-8B
counterpart, the remaining Llama benchmarks, the feature-only
($\delta$) scatter, parallel language-modeling results, and
Ministral/DeepSeek decompositions are in
Appendices~\ref{app:feature_only}--\ref{app:per_model_tables}.
```

**After**:
```
Table~\ref{tab:llama_decomposition_main} reports the Llama decomposition
on MMLU and TriviaQA; per-model and per-benchmark extensions
(Qwen3-8B, feature-only $\delta$ scatter, language modeling,
Ministral/DeepSeek) are in Appendices~\ref{app:feature_only}--\ref{app:per_model_tables}.
```

**省行**：1 行（list 從 5 項分散改為 1 個 parenthetical group）

---

### 改動 4：Sec 5.2 LoRA closing summary 移除（省 1 行）

**Before** (L370 末):
```
... The Qwen3-8B replication (Appendix~\ref{app:qwen_forgetting})
reproduces both patterns. Together, the PTQ and LoRA results show
that a single closed-form bound transfers across two qualitatively
different sources of model variation.
```

**After**: 移除最後一句 "Together, the PTQ and LoRA results show..."

**Risk**：失去 PTQ + LoRA 統一性的明確 statement。
**Mitigation**：Conclusion L423 Empirical theme 已 cover ("rank-consistent across both settings")；Sec 5.3 lead-in 也會建立此 framing。

**省行**：1 行

---

### 改動 5：Sec 5.3 Lead-in 壓為 1 句（省 1 行）

**Before** (L376):
```
A scalar correlation alone would already match prior work on
representational similarity. The decomposition contributes the next
layer: the dominant axis of Theorem~\ref{thm:unified_bound} reads
off directly from the per-variant numbers and corresponds to a
distinct empirical failure mode. Three such failure modes recur
in our experiments.
```

**After**:
```
The decomposition reads the dominant axis of
Theorem~\ref{thm:unified_bound} from per-variant numbers; three
distinct failure modes recur.
```

**Risk**：失去與 prior work（scalar similarity）的對比。
**Mitigation**：Intro L145、RW L180 已 establish CKA 等 scalar similarity 不夠用的對比；Sec 5.3 不需重述。

**省行**：1.5 行

---

### 改動 6：Sec 5.5 Two observations 壓縮（省 2 行）

**Before** (L414 後半):
```
Two observations keep $\mathcal{B}$ the right \emph{default} metric.
First, it is the certified upper bound of Theorem~\ref{thm:unified_bound}
and is strictly required whenever $\gamma{>}0$; $\delta$ alone is not
a valid bound in that setting. Second, $\mathcal{B}$ has the smallest
Llama--Qwen gap ($\approx 0.015$ vs.\ $0.026$ and $0.042$ for $\delta$
and $\Omega$), so its ranking performance is the most robust across
the two families tested.
```

**After**:
```
$\mathcal{B}$ remains the default: it is the certified upper bound
required whenever $\gamma{>}0$ ($\delta$ alone is not), and it has
the smallest Llama--Qwen gap ($\approx 0.015$ vs.\ $0.026$ and $0.042$
for $\delta$ and $\Omega$)---most robust across families.
```

**Risk**：去 "two observations" 列表結構。
**Mitigation**：兩 reasons 用分號連接後一樣清楚；本就只有 2 點不用編號。

**省行**：2 行

---

### 改動 7：Sec 5.3 Shape distortion 去 Ministral example（省 1 行）

**Before** (L378-379):
```
At Q2 and Q3 across all four families tested..., the shape term
$2\rho_T\rho_P(1{-}\Omega)$ outweighs the scale term $(\Delta\rho)^2$
by two to three orders of magnitude (e.g., Llama-Q2\_K on MMLU:
$24$ vs.\ $\sim$$9{,}000$; Ministral-Q2\_K on MMLU: $\sim$$9$ vs.\
$\sim$$11{,}000$)---consistent with low-bit PTQ corrupting the
relational structure of the feature manifold rather than its global
scale.
```

**After**:
```
At Q2 and Q3 across all four families tested..., the shape term
$2\rho_T\rho_P(1{-}\Omega)$ outweighs the scale term $(\Delta\rho)^2$
by two to three orders of magnitude (e.g., Llama-Q2\_K on MMLU:
$24$ vs.\ $\sim$$9{,}000$)---consistent with low-bit PTQ corrupting
the relational structure of the feature manifold rather than its
global scale.
```

**Risk**：reader 不見「跨 family」具體 evidence。
**Mitigation**：「across all four families tested」phrase 已宣稱跨 family；appendix 表有完整數字 verify。

**省行**：1 行

---

### 改動 8：Sec 5.3 Head divergence 去 Q8_0 重例 + meta-commentary（省 1.5 行）

**Before** (L382 末):
```
... yet the quantized output embedding alone contributes $\gamma{=}60.16$
---making $\gamma$ essentially the entire bound (Q8\_0 shows the same
pattern: $\delta{=}0.15$, $\gamma{=}19.0$). Under BnB INT8 the same
Qwen3-Base keeps $\gamma{\equiv}0$ by construction, leaving $\delta{=}13.2$
as the sole error source---a qualitatively different decomposition
determined entirely by which protocol quantizes \texttt{lm\_head}.
The decomposition makes this protocol-level switch read off directly,
identifying not just the magnitude of degradation but its dominant channel.
```

**After**:
```
... yet the quantized output embedding alone contributes $\gamma{=}60.16$,
making $\gamma$ essentially the entire bound. Under BnB INT8 the same
Qwen3-Base keeps $\gamma{\equiv}0$ by construction, leaving $\delta{=}13.2$
as the sole error source---a qualitatively different decomposition
determined entirely by which protocol quantizes \texttt{lm\_head}.
```

**移除**：(i) Q8_0 重例 (parenthetical), (ii) "The decomposition makes this protocol-level switch read off directly..." 這個 meta-commentary (前文具體例已 deliver)。

**省行**：1.5 行

---

## Round 2 — 預估總省行

| # | 改動 | 省行 | 累計 |
|---|---|---|---|
| 1 | Sec 5.3 Remediation 完全移除 | 3 | 3 |
| 2 | Sec 5.2 Q2_K TriviaQA example 移除 | 2 | 5 |
| 3 | Sec 5.2 appendix list 壓縮 | 1 | 6 |
| 4 | Sec 5.2 LoRA closing summary 移除 | 1 | 7 |
| 5 | Sec 5.3 Lead-in 壓為 1 句 | 1.5 | 8.5 |
| 6 | Sec 5.5 Two observations 壓縮 | 2 | 10.5 |
| 7 | Sec 5.3 Shape distortion 去 Ministral example | 1 | 11.5 |
| 8 | Sec 5.3 Head divergence 去 Q8_0 重例 + meta | 1.5 | 13 |

**達標**：~13 行 ≥ 目標 10 行，**留 3 行 buffer**。

可選擴充（若覺得想再 cut）：
- 9. Sec 5.1 Models 去 family-difference 子句 (0.5 行)
- 10. Sec 5.1 Benchmarks 壓尾子句 (1 行)
- 11. Sec 5.4 去自問自答 (1 行)
- 12. Sec 5.4 merge patterns (ii)+(iii) (1 行)

---

## Round 2 — 風險矩陣

| 改動 | 易讀性影響 | 故事性影響 | 整體 |
|---|---|---|---|
| 1 (Remediation 移除) | ✅ 提升（去 marketing 詞） | ⚠️ 失去 axis-mitigation 對應 | ✅ 淨正（Conclusion 已 cover） |
| 2 (Q2_K example 移除) | ✅ 提升（去重複） | ✅ 仍有 Sec 5.3 example | ✅ 淨正 |
| 3 (appendix list 壓縮) | ➖ 中性 | ➖ 中性 | ✅ 純省行 |
| 4 (LoRA closing 移除) | ✅ 提升（去 marketing） | ⚠️ 失去 unification claim | ✅ 淨正（Conclusion 已 cover） |
| 5 (Lead-in 壓縮) | ✅ 提升（直接） | ⚠️ 失去與 RW 對比 | ✅ 淨正（Intro 已 cover） |
| 6 (Two observations 壓縮) | ✅ 提升（兩 reasons 一句講完） | ➖ 中性 | ✅ 淨正 |
| 7 (Ministral example 移除) | ➖ 中性 | ⚠️ 微失去跨 family evidence | ⚠️ 邊際（看 reviewer）|
| 8 (Q8_0 重例移除) | ✅ 提升 | ➖ 中性 | ✅ 淨正 |

**所有改動皆 net positive**。改動 7 唯一邊際，可選擇保留 Ministral example 改用其他方式達標。

---

## Round 2 — 改動順序建議

按依賴性 + 安全性排序執行：

**Phase A（最安全，無互相依賴）**：
1. 改動 1 (Remediation 移除)
2. 改動 4 (LoRA closing 移除)
3. 改動 8 (Head divergence 重例 + meta 移除)

→ 累計省 ~5.5 行

**Phase B（內容壓縮）**：
4. 改動 2 (Q2_K example 移除)
5. 改動 3 (appendix list 壓縮)
6. 改動 6 (Two observations 壓縮)

→ 累計省 ~10.5 行（已達標）

**Phase C（可選微調，看是否需要）**：
7. 改動 5 (Lead-in 壓縮)
8. 改動 7 (Ministral example) — 風險較高，可保留

---

## Round 3 — 待 user 確認

要從哪 Phase 開始？或者對個別改動有想 skip 的？
