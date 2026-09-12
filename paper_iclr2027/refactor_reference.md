# Cross-Reference Audit: Tables and Figures in PRISM Paper

逐一 audit 所有透過 `\input` / `\includegraphics` 載入的 figures 和 tables，確認 reference 狀態，識別 orphans，並為 appendix orphans 提出最小化 discussion 建議。

> **原則**：
> - 主文不加新 reference（避免 over-claim）
> - Appendix orphans → 在 appendix 內加一句 discussion 句使其合法化
> - 真正多餘的 table/figure → 標註可考慮移除

---

## 1. 完整 Inventory

### 1.1 Figures（共 8 個）

| # | Label | 定義位置 | 內容 |
|---|---|---|---|
| F1 | `fig:prism_concept` | `figures/prism_geometric_decomposition.tex` | PRISM 概念圖 |
| F2 | `fig:quant_grid_bound` | `neurips_2025.tex` L350 | Llama+Qwen PTQ 散點圖 |
| F3 | `fig:forget_grid_llama` | `neurips_2025.tex` L364 | Llama LoRA forgetting 散點圖 |
| F4 | `fig:shape_reg_combined_llama` | `neurips_2025.tex` L397 | Llama shape regularization |
| F5 | `fig:quant_grid_bound_extra` | `appendix/quantization_exp.tex` L16 | Mistral+DeepSeek PTQ 替換 |
| F6 | `fig:quant_grid_feature` | `appendix/quantization_exp.tex` L28 | Feature-only scatter |
| F7 | `fig:forget_grid_qwen` | `appendix/forgetting_qwen.tex` L16 | Qwen LoRA forgetting 散點圖 |
| F8 | `fig:shape_reg_combined_qwen` | `appendix/forgetting_qwen.tex` L28 | Qwen shape regularization |

### 1.2 Tables（共 18 個）

| # | Label | 定義位置 | 內容 |
|---|---|---|---|
| T1 | `tab:baseline_combined` | `tables/quantization/baseline_combined.tex` | Ablation（W=I + W=W_N） |
| T2 | `tab:llama_decomposition_main` | `tables/quantization/table_llama_main.tex` | Llama 主 PTQ decomp |
| T3 | `tab:llama_decomposition_ext` | `tables/quantization/table_llama_ext.tex` | Llama PTQ ext (WikiText/FineWeb) |
| T4 | `tab:mistral_decomposition_all` | `tables/quantization/table_mistral_all.tex` | Ministral 完整 PTQ |
| T5 | `tab:qwen_decomposition_all` | `tables/quantization/table_qwen_all.tex` | Qwen 完整 PTQ |
| T6 | `tab:deepseek_decomposition_all` | `tables/quantization/table_deepseek_all.tex` | DeepSeek 完整 PTQ |
| T7 | `tab:llama_instruct_decomposition_all` | `tables/quantization/table_llama_instruct_all.tex` | Llama-Instruct PTQ |
| T8 | `tab:mistral_instruct_decomposition_all` | `tables/quantization/table_mistral_instruct_all.tex` | Ministral-Instruct PTQ |
| T9 | `tab:qwen_instruct_decomposition_all` | `tables/quantization/table_qwen_instruct_all.tex` | Qwen-Instruct PTQ |
| T10 | `tab:trace_norm_llama_truthfulqa` | `tables/regularization/table_trace_norm_llama_truthfulqa.tex` | Llama TruthfulQA reg |
| T11 | `tab:trace_norm_llama_bbq` | `tables/regularization/table_trace_norm_llama_bbq.tex` | Llama BBQ reg |
| T12 | `tab:trace_norm_qwen_truthfulqa` | `tables/regularization/table_trace_norm_qwen_truthfulqa.tex` | Qwen TruthfulQA reg |
| T13 | `tab:trace_norm_qwen_bbq` | `tables/regularization/table_trace_norm_qwen_bbq.tex` | Qwen BBQ reg |
| T14 | `tab:target_models` | `appendix/model_quantization.tex` L17 | 9 個 target models |
| T15 | `tab:quant_tiers` | `appendix/model_quantization.tex` L50 | 量化 tiers |
| T16 | `tab:gptq_awq` | `appendix/model_quantization.tex` L84 | GPTQ checkpoints |
| T17 | `tab:gguf_repos` | `appendix/model_quantization.tex` L118 | GGUF repos |
| T18 | `tab:coverage` | `appendix/model_quantization.tex` L152 | Backend coverage |

**Total**: 8 figures + 18 tables = **26 labels**

---

## 2. Reference 狀態

### 2.1 已被 reference 的（18）

| Label | 被 reference 處 |
|---|---|
| `fig:prism_concept` | Intro L158, Contributions L169, Sec 3.3 L260, Sec 4 L305 |
| `fig:quant_grid_bound` | Sec 5.2 L356, Appendix Fig 5/6 caption（cross-ref） |
| `fig:forget_grid_llama` | Sec 5.2 L366, Sec 5.3 L380, Appendix forgetting_qwen L15+L19 |
| `fig:shape_reg_combined_llama` | Sec 5.4 L400 |
| `fig:quant_grid_bound_extra` | Sec 5.2 L356（main-text caption），appendix L10 |
| `fig:quant_grid_feature` | appendix L22 |
| `fig:forget_grid_qwen` | appendix L19 |
| `tab:baseline_combined` | Sec 5.2 L356, Sec 5.5 L408 |
| `tab:llama_decomposition_main` | Sec 5.2 L356, Sec 5.1 L338 |
| `tab:qwen_decomposition_all` | Sec 5.3 L375 + L378 |
| `tab:trace_norm_llama_truthfulqa` | Sec 5.4 L400, appendix forgetting_qwen L34 |
| `tab:trace_norm_llama_bbq` | appendix forgetting_qwen L34（範圍 ref） |
| `tab:trace_norm_qwen_bbq` | appendix forgetting_qwen L34（範圍 ref） |
| `tab:target_models` | appendix model_quantization 內部 |
| `tab:quant_tiers` | 同上 |
| `tab:gptq_awq` | 同上 |
| `tab:gguf_repos` | 同上 |
| `tab:coverage` | 同上 |

### 2.2 ❌ Orphan（8 個 — 未直接 reference）

| # | Label | 位置 | 嚴重度 |
|---|---|---|---|
| **O1** | `fig:shape_reg_combined_qwen` | appendix forgetting_qwen.tex L28 | 🔴 high — 完全無 body 討論 |
| **O2** | `tab:trace_norm_qwen_truthfulqa` | appendix forgetting_qwen.tex L38 | 🟡 mid — 被 range "Tables 5-7" 隱含覆蓋 |
| **O3** | `tab:llama_decomposition_ext` | appendix quantization_exp.tex L36 | 🟡 mid — Per-Model 段集體描述 |
| **O4** | `tab:mistral_decomposition_all` | appendix quantization_exp.tex L40 | 🟡 mid — 同上 |
| **O5** | `tab:deepseek_decomposition_all` | appendix quantization_exp.tex L48 | 🟡 mid — 同上 |
| **O6** | `tab:llama_instruct_decomposition_all` | appendix quantization_exp.tex L52 | 🟡 mid — 同上 |
| **O7** | `tab:mistral_instruct_decomposition_all` | appendix quantization_exp.tex L56 | 🟡 mid — 同上 |
| **O8** | `tab:qwen_instruct_decomposition_all` | appendix quantization_exp.tex L60 | 🟡 mid — 同上 |

---

## 3. Orphan 處理建議

### 3.1 🔴 O1: `fig:shape_reg_combined_qwen`（最嚴重）

**現況**：appendix forgetting_qwen.tex L21-29 的 `\subsection{Shape Regularization: Qwen3-8B Replication}` 只 insert figure 沒有 body discussion paragraph。

```latex
\subsection{Shape Regularization: Qwen3-8B Replication}
\label{app:qwen_shape_reg}

\begin{figure}[h]
    \includegraphics{...combined_qwen_truthfulqa_bbq.pdf}
    \caption{...}
    \label{fig:shape_reg_combined_qwen}
\end{figure}

% ← 這裡缺一段 body text discussing fig:shape_reg_combined_qwen

\subsection{Additional Trace-Norm Decomposition Tables}
```

**建議改寫**：在 figure 後加一段討論句

```latex
Figure~\ref{fig:shape_reg_combined_qwen} reproduces the main-text Llama 
result (Fig.~\ref{fig:shape_reg_combined_llama}) on Qwen3-8B: 
λ ∈ \{0.0, 0.1, 0.5\} produces the same monotone reduction in downstream 
|ΔR| across all five benchmarks for both TruthfulQA and BBQ fine-tuning, 
confirming that the regularizer's effect is family-agnostic. Per-variant 
numerical decompositions are in Tables~\ref{tab:trace_norm_qwen_truthfulqa} 
and~\ref{tab:trace_norm_qwen_bbq} below.
```

**好處**：
1. Legitimize `fig:shape_reg_combined_qwen` reference
2. 順便明示 `tab:trace_norm_qwen_truthfulqa`（解決 O2）+ `tab:trace_norm_qwen_bbq`（強化已有 reference）
3. **同時解決 2 個 orphans（O1 + O2）**

### 3.2 🟡 O3-O8: Per-Model Decomposition Tables 集體 orphans

**現況**：appendix quantization_exp.tex L31-60 的 `\subsection{Per-Model Decomposition Tables}` 只有 1 段集體描述，後面 7 個 `\input`，個別表沒有獨立 reference。

```latex
\subsection{Per-Model Decomposition Tables}
\label{app:per_model_tables}

We report the full geometric decomposition...for each evaluated model...
Each table column lists ρ_T, ρ_P, Ω, δ, γ, B, |ΔR|...

\input{tables/quantization/table_llama_ext}
\input{tables/quantization/table_mistral_all}
\input{tables/quantization/table_qwen_all}
\input{tables/quantization/table_deepseek_all}
\input{tables/quantization/table_llama_instruct_all}
\input{tables/quantization/table_mistral_instruct_all}
\input{tables/quantization/table_qwen_instruct_all}
```

**問題**：每個 table 在 list 裡但沒「why this table matters」的句子。

**建議改寫**：在現有段落後加一句 enumerate-style 的 table-by-table summary

```latex
\subsection{Per-Model Decomposition Tables}
\label{app:per_model_tables}

We report the full geometric decomposition...[原段落不變]

Tables are organized as follows: 
\textbf{Llama-3.1-8B} extended benchmarks (WikiText, FineWeb-Edu) in 
Table~\ref{tab:llama_decomposition_ext}; 
\textbf{Ministral-3-8B-Base} (Table~\ref{tab:mistral_decomposition_all}); 
\textbf{Qwen3-8B-Base} (Table~\ref{tab:qwen_decomposition_all}, also 
referenced in Sec.~\ref{subsec:decompose}); 
\textbf{DeepSeek-R1-Distill-Llama-8B} (Table~\ref{tab:deepseek_decomposition_all}); 
and the instruction-tuned counterparts of Llama, Ministral, and Qwen 
in Tables~\ref{tab:llama_instruct_decomposition_all}, 
\ref{tab:mistral_instruct_decomposition_all}, 
and~\ref{tab:qwen_instruct_decomposition_all} respectively.

\input{tables/quantization/table_llama_ext}
[剩下 6 個 \input 不變]
```

**好處**：
1. 一次解決 O3-O8（6 個 orphans）
2. Reader 從 list 可以直接 navigate 到任一 model
3. 符合 user「不必要的 ref 不要加，但需要的話開 sec/subsec 討論」原則——**這就是在現有 subsection 裡加 navigation 句**

---

## 4. 預期完成後狀態

執行兩處修改後：

| 類別 | 處理前 | 處理後 |
|---|---|---|
| 已 reference | 18 / 26 | **26 / 26** |
| Orphans | 8 | **0** |
| 主文新增 ref | 0 | **0**（完全保留主文乾淨） |
| Appendix 新增段落 | 0 | **2 段**（forgetting_qwen + quantization_exp） |

---

## 5. 補充觀察

### 5.1 Range reference 的隱含覆蓋
- `Tables~\ref{tab:trace_norm_llama_bbq}--\ref{tab:trace_norm_qwen_bbq}` 在 forgetting_qwen.tex L34 涵蓋了中間的 `tab:trace_norm_qwen_truthfulqa`
- LaTeX 不會把它算成 explicit ref（reader 仍能理解，但 audit 會 flag）
- **建議 3.1 的改寫直接讓中間表 explicit 出現**，更乾淨

### 5.2 主文已 100% reference
- 主文裡的 4 個 figures + 3 個 tables 都已被 `\ref{}` 引用至少一次
- **完全符合 user「不必要的 ref 不要加」原則**——不需要動主文

### 5.3 「Per-Model Decomposition Tables」subsection 是 reference 集中地
- 7 個 per-model PTQ tables 全部聚在這個 subsection
- 加一段 navigation 句即可一次合法化全部
- 不需要為每個 table 開單獨 subsection（過度結構化）

### 5.4 model_quantization.tex 內 5 個 tables（T14-T18）
- 都在 `\subsection{Target Models}` / `\subsection{Quantization Backends and Bit-Widths}` / `\subsection{Per-Model GPTQ Repositories}` / `\subsection{GGUF Repositories}` / `\subsection{Backend Coverage Summary}` 內被各自 subsection 直接 reference
- **狀態 OK**，不需處理

---

## 6. 建議套用順序

1. **修改 1：appendix/forgetting_qwen.tex** — 在 fig:shape_reg_combined_qwen 後加 1 段討論句（同時 cover O1 + O2）
2. **修改 2：appendix/quantization_exp.tex** — 在 Per-Model Decomposition Tables subsection body 加 1 段 navigation 句（cover O3-O8）

兩處改動，**全部 8 個 orphans 一次解決**，主文完全不動。
