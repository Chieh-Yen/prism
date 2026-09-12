# Sec 5.4 Replay-Comparison Rewrite Plan

## 改寫範圍

| File | Line(s) | 修改 |
|---|---|---|
| `neurips_2026.tex` | L340 | Sec 5.1 hyperparam — 加 replay sweep description |
| `neurips_2026.tex` | L399 | Fig 4 caption — 改 3-config (no reg / replay / trace) |
| `neurips_2026.tex` | L403 | `\input` table_trace_norm → table_compare |
| `neurips_2026.tex` | L405-407 | Sec 5.4 narrative — 改寫成 trace vs replay 對比 + interpretation |
| `appendix/forgetting_qwen.tex` | L29-31 | Qwen Fig caption — 同 Fig 4 改寫 |
| `appendix/forgetting_qwen.tex` | L34 | Qwen narrative — 改為 3-config replication |
| `appendix/forgetting_qwen.tex` | L37-39 | Subsection title + 表 intro |
| `appendix/forgetting_qwen.tex` | L41,43,45 | 三 `\input` table_trace_norm → table_compare |

Label remap：所有 `tab:trace_norm_X` → `tab:reg_compare_X`

---

## 數字 anchor（來自新 data）

- Llama TruthQA mean \|ΔR\|: no reg `0.84` → replay `0.76` (-9%) → **trace `0.68` (-19%)**
- Llama BBQ mean Ω: no reg `0.93` → replay `0.93` (flat) → **trace `0.98`** (mechanism punchline)

---

## 草稿

### Sec 5.1 update (L340)
```
For the regularization comparison (Sec.~\ref{subsec:action}), the trace-norm
shape penalty sweeps λ∈{0.01, 0.05, 0.1, 0.5, 1.0} and a replay-CE baseline
sweeps λ∈{0.001, 0.005, 0.01, 0.05, 0.1}, both on the same reference set
D_ref (32 sequences from pre-training distribution, disjoint from D_FT,
refreshed every k=8 micro-steps). Checkpoints every 25 steps, analysis at
step 300. All experiments use a single NVIDIA RTX 5090 (32 GB).
```

### Fig 4 caption (L399) — 簡潔版
```
\textbf{Shape regularization vs.\ replay-CE on Llama-3.1-8B.} LoRA
fine-tuning on TruthfulQA (top) and BBQ (bottom) under three matched-budget
settings: \emph{no reg}, \emph{replay} (λ=0.01), \emph{trace} (λ=1.0,
Eq.~\ref{eq:shape_reg}); the latter two share a 32-sample reference set and
are each method's sweep |ΔR|-best. Trace cuts downstream |ΔR| further than
replay across benchmarks; full decomposition in Table~\ref{tab:reg_compare_llama_truthfulqa}.
Qwen3-8B replication: Appendix~\ref{app:qwen_forgetting}.
```

### Sec 5.4 narrative (L405-407)
**Para 1 (motivation + setup)**:
```
The decomposition of Sec.~\ref{subsec:decompose} suggests an immediate
intervention: shape drift dominates LoRA forgetting, and (1-Ω) is
differentiable. We add a trace-norm penalty λ(1-Ω(Z_0^ref, Z_t^ref)) on a
32-sample reference set D_ref. To isolate shape-preservation from data
re-fitting, we compare against a replay-CE baseline that uses the same
D_ref (cross-entropy on D_ref, swept across λ).
```

**Para 2 (results + interpretation + Qwen pointer)**:
```
Fig.~\ref{fig:shape_reg_combined_llama} and Table~\ref{tab:reg_compare_llama_truthfulqa}
report each method at its sweep |ΔR|-best (replay λ=0.01, trace λ=1.0)
against no regularization on Llama-3.1-8B. Trace cuts mean downstream |ΔR|
on TruthfulQA from 0.84 (no reg) to 0.68 (-19%); replay only reaches 0.76
(-9%). On Llama BBQ the contrast is qualitative: trace lifts Ω from 0.93 to
0.98 while replay leaves Ω flat at 0.93. Mechanism: replay reduces
forgetting indirectly by re-fitting reference data, leaving Ω unchanged;
trace contracts the shape arm of the bound at the source. Preserving
shape geometry during training thus outperforms gradient replay at matched
budget, consistent with Sec.~\ref{subsec:decompose}'s identification of
shape as the dominant LoRA-forgetting axis. The Qwen3-8B replication and
BBQ per-variant tables are in Appendix~\ref{app:qwen_forgetting}.
```

### Appendix Qwen Figure caption
```
\textbf{Shape regularization vs.\ replay-CE on Qwen3-8B (replication of
Fig.~\ref{fig:shape_reg_combined_llama}).} Same three settings as Llama
(\emph{no reg}, \emph{replay} λ=0.01, \emph{trace} λ=1.0). Qwen3-8B's
baseline forgetting is already small, leaving little room for improvement;
both methods stay near baseline. Per-variant tables:
Tables~\ref{tab:reg_compare_qwen_truthfulqa} and~\ref{tab:reg_compare_qwen_bbq}.
```

### Appendix narrative (L34)
```
Figure~\ref{fig:shape_reg_combined_qwen} reproduces the main-text Llama
comparison (Fig.~\ref{fig:shape_reg_combined_llama}) on Qwen3-8B. Qwen's
baseline forgetting is already small (mean |ΔR|<0.27 across both
fine-tuning tasks vs. Llama's 0.84 on TruthfulQA), leaving little room for
improvement; the trace-vs-replay gap narrows accordingly but the qualitative
mechanism (trace lifts Ω, replay does not) holds. Per-variant tables:
Tables~\ref{tab:reg_compare_qwen_truthfulqa} and~\ref{tab:reg_compare_qwen_bbq}.
```

### Appendix subsection title + intro (L37-39)
```
\subsection{Additional Regularization-Comparison Tables}
\label{app:reg_compare_tables}

Tables~\ref{tab:reg_compare_llama_bbq}--\ref{tab:reg_compare_qwen_bbq} give
the full per-(model, fine-tuning task) decomposition at step 300 for the
three settings (no reg, replay λ=0.01, trace λ=1.0). The main text
(Table~\ref{tab:reg_compare_llama_truthfulqa}) showed the Llama-TruthfulQA
combination; this appendix completes the 2×2 grid.
```

### Appendix `\input` lines (L41,43,45)
```
\input{tables/regularization/table_compare_llama_bbq}
\input{tables/regularization/table_compare_qwen_truthfulqa}
\input{tables/regularization/table_compare_qwen_bbq}
```

---

## 執行順序

1. Sec 5.1 (L340) — replay setup
2. Fig 4 caption (L399) — 3-config
3. L403 input table change
4. L405-407 narrative rewrite
5. Appendix figure caption + narrative + subsection + 3 inputs
6. Compile + verify
