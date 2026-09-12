# Experiments Refactor Plan for `paper/neurips_2025.tex`

審視所有 main + appendix figures/tables 後的重構規劃。每項提案都附 **事實依據** 與 **預期效果**。

---

## A. 現況診斷

### 1. 結構問題

當前 Sec 5 結構：
- 5.1 Setup（長，~1 頁）
- 5.2 Quantization
- 5.3 Forgetting
- 5.4 Regularization
- 5.5 Ablation

**問題**：讀起來像「3 個獨立實驗 + 1 個 ablation」。但論文主訴求是 **「PRISM 是 unified framework」**——這個 narrative 沒在實驗段被強化。

### 2. Figure / Table 統計

| 主文 | 數量 | 問題 |
|---|---|---|
| Figures | 4 | Fig 2 (4×5=20 panels) 太擁擠；Fig 3 只有 Llama，Qwen 在 appendix 失衡 |
| Tables | 3 | table_llama_main 是唯一 per-model decomp 主文；其餘 3 lineages 全在 appendix |

| Appendix | 數量 | 冗餘度 |
|---|---|---|
| Per-model decomp tables | 7 個（base + instruct） | 高——instruct 變體跟 base 高度重複 |
| Regularization tables | 10 個（5 fine-tuning task × 2 model） | 高——LIMA/no_robots/social_iqa 都在 paper 但 main text 未引用 |
| Figures variants (`_rp`, `_rpboth`, `_rplog`) | 16 個 | 是探索性繪圖，main paper 不需要 |

### 3. Narrative 浪費

- 「Which axis fails where」段落（L360）是 **三軸 localization 的核心** 但塞在 paragraph 裡，視覺上完全沒 highlight
- 「TruthfulQA vs BBQ 引出不同 drift geometry」是 forgetting 的關鍵 finding，但目前只用文字提及，沒視覺化
- Ablation 結尾的 W=I vs W_N trade-off 段落很關鍵，但跟 ablation 本體混在一起

---

## B. Reframe Big-Picture

把 Sec 5 從「3 experiments + ablation」改成 **「3 claims under one framework」**：

| 新框架的子節 | 核心 claim | 主要證據 |
|---|---|---|
| **5.2 Predictiveness** | PRISM bound tracks empirical risk gap | Spearman \|r_s\|≈0.79 across PTQ + LoRA |
| **5.3 Decomposability** | 3 axes localize 3 distinct failure modes | shape@Q2/Q3, head@Qwen Q6\_K, scale@BBQ-vs-TruthfulQA |
| **5.4 Actionability** | Bound becomes loss; regularizer suppresses forgetting | λ=0.5 reduces \|ΔR\| on every downstream benchmark |
| **5.5 Ablation** | Each component contributes to ranking | shape→+scale→+head |

優點：直接對應三個 contributions（Sec 1 的三條 bullet），形成「claim → experiment」對應關係，reviewer 一看標題就知道這節要證什麼。

---

## C. 主文 Figures/Tables 具體修改

### Fig 1（concept）— **不動**
- 目前是 conceptual decomposition 圖，已經 self-contained

### Fig 2（quantization grid）— **大改**

**現況**：`figures/quantization/prism_grid_bound.pdf`，4×5=20 個 panel

**問題**：
- 20 panel 平均下來每個 ~1×1 cm，散點看不清
- Reviewer 第一眼只能看到「都散在 y=x 附近」，無法判斷哪個 family 最好或最差
- 4×5 grid 占用半頁版面但只傳達單一訊息：「相關性好」

**提案**：改成 **單一 aggregated scatter + 1 個 inset bar chart**
- **左**: 單一 panel，所有 4 families × 5 benchmarks = 20 cells 全部疊在一張圖上（每個 family 一個 marker shape，每個 PTQ family 一個顏色：GGUF / GPTQ / BnB）
- **右**: 4×5 heatmap or matrix grid，cell 顏色代表「dominant axis」（red=shape, blue=head, green=scale）

**事實依據**：
- L357 主文：「mean Spearman \|r_s\|≈0.79 across the 4×5 grid」
- L360：「Q2/Q3 shape dominates, Qwen3 Q6_K head dominates, Qwen3 Q2_K GSM8K scale-affected」
- 這些 axis dominance 數字都驗證過（前面 wording.md/sentence.md 流程確認過）

**省下空間**：1 個大 grid 圖 → 1 個小 scatter + 1 個小 heatmap，預估 +30% 資訊密度，可能省下 0.3 頁

### Fig 3（forgetting，Llama only）— **大改 + 上升 Qwen**

**現況**：
- Main: `forgetting_grid_bound_llama.pdf`（2 fine-tuning tasks × 5 benchmarks = 10 panels）
- Appendix: Qwen 同樣 10 panels

**問題**：
- 失衡：Llama 在主文，Qwen 只在 appendix。Reviewer 第一印象「實驗只在 1 個 model」
- 10 panels 仍太密
- 「TruthfulQA vs BBQ 引出不同 drift geometry」這個 qualitative finding 完全沒視覺化

**提案**：改成 **2×3 grid，整合 Llama + Qwen**
- 行：Llama / Qwen
- 列：Bound vs |ΔR| (rank consistency) / Drift trajectory in (Δρ, 1-Ω) space (geometry asymmetry) / λ-effect on downstream forgetting (regularization)
- 每個 panel 同時 overlay TruthfulQA + BBQ 兩條 trajectory（不同顏色）

**事實依據**：
- Agent report: TruthfulQA Llama TriviaQA λ=0 時 \|ΔR\|=4.63（最嚴重）
- BBQ Llama Ω≥0.94 even at λ=0（健康得多）
- Qwen 也有對應數據
- Drift geometry contrast 在 L372 文字提及但無圖

**注意**：這個改動較大，需要新繪圖。如果時間緊，可降階為 **2×2**：
- Top: rank consistency (Llama, Qwen)
- Bottom: regularization effect (Llama, Qwen)

### Fig 4（regularization combined）— **小調，整合 Qwen**

**現況**：
- Main: `combined_llama_truthfulqa_bbq.pdf`（Llama only）
- Appendix: `combined_qwen_truthfulqa_bbq.pdf`（Qwen 替換）

**問題**：跟 Fig 3 同樣，Qwen 失衡

**提案**：改成 2×2，行 = Llama/Qwen，列 = TruthfulQA/BBQ
- 維持原有「λ ∈ {0, 0.1, 0.5} 對 5 benchmarks 的 |ΔR|」資訊
- 加入 Qwen 強化 cross-lineage replication 的 visual proof

**事實依據**：
- Llama TruthfulQA: λ=0→0.5 在 5 benchmarks 都降低（main text L387 已有具體數字）
- Qwen 替換確認 in `forgetting_qwen.tex` Fig 3

### Table 3（Llama main decomp）— **改成 cross-family selected examples**

**現況**：`table_llama_main.tex` 只有 Llama-3.1-8B 的 MMLU + TriviaQA

**問題**：
- 4 個 families 但 main text 只 show 1 個的 detailed numbers
- Reviewer 想看其他 lineage 的具體數字會找不到（要翻 appendix 7 個 table）

**提案**：改成 **「3 regimes 各 1 個代表 case」的 3-row table**
- Row 1: shape-dominated example (e.g., Llama-Q2_K MMLU: Δρ², 1-Ω, δ, γ)
- Row 2: head-dominated example (e.g., Qwen3 Q6_K MMLU: Δρ²≈0, δ=0.75, γ=60.16)
- Row 3: scale-affected example (e.g., Qwen3 Q2_K GSM8K: ρ inflates 267→313)

每 row 對應一個 dominant axis，直接呼應 5.3 Decomposability 的 narrative。

**事實依據**：
- Shape-dominated Q2_K 數字：L360 主文已引用
- Head-dominated Qwen Q6_K：L360 已引用 δ=0.75, γ=60.16
- Scale Qwen Q2_K GSM8K：L360 已引用 267→313

### Table 4（regularization Llama TruthfulQA）— **改成 Pareto-style table 或保留**

**現況**：`table_trace_norm_llama_truthfulqa.tex`，5 benchmarks × 3 λ 值

**選項 A（保留現況）**：簡單明瞭
**選項 B（Pareto）**：列出 (target task fitting loss, downstream |ΔR|) trade-off across λ
- 這需要驗證有 target task fitting loss 數據

**建議**：選 A 保留，加上 Qwen TruthfulQA 對照 row（如果空間允許）

### Table 5（baseline_combined ablation）— **小改**

**現況**：6 rows，2 blocks（W=I + W=W_N）

**問題**：W_N block 在 Sec 5.5 占了大量視覺空間，但跟主軸（W=I）關係只是「post-hoc reference」

**提案**：保留 combined table，但把 W_N block 加上 `\rowcolor{gray!10}` 視覺淡化，告訴 reader「這是 ablation reference 而非主要結果」

**事實依據**：W=I 是主文 default，W_N 是 ablation reference（已在 wording.md 流程確認）

---

## D. Appendix 整合

### 1. 砍重複的 instruct tables

**現況**：每個 lineage 有 base + instruct 兩個版本：
- table_llama_all.tex, table_llama_instruct_all.tex
- table_mistral_all.tex, table_mistral_instruct_all.tex
- table_qwen_all.tex, table_qwen_instruct_all.tex

**提案**：只保留 base 版（4 tables），instruct 結果用一段文字總結「instruct variants 結果一致，數據在 supplementary」

**事實依據**：Agent report 顯示 instruct 跟 base 結構完全一樣，只是 model variant 不同

**省下**：3 個 appendix tables，~1.5 頁

### 2. 砍未引用的 regularization tables

**現況**：10 個 regularization table，但主文只引用 `table_trace_norm_llama_truthfulqa.tex`

**Status check**：
- ✓ Llama TruthfulQA — 主文引用
- ✗ Llama BBQ — 未在 main text 引用
- ✗ Llama LIMA / no_robots / social_iqa — 完全沒提及
- ✗ Qwen 全部 5 個 — 只在 forgetting_qwen.tex appendix 提及

**提案**：
- 主文：保留 Llama TruthfulQA + 加 Llama BBQ（強化 cross-task validation）
- Appendix：保留 Qwen TruthfulQA + Qwen BBQ
- **砍掉**：LIMA / no_robots / social_iqa 4 個 Llama + 4 個 Qwen = 8 個

**事實依據**：這些 fine-tuning tasks 不在主文 sceintific narrative 內（L330 只提 TruthfulQA + BBQ）。**需要確認沒在其他地方暗用。**

**省下**：8 個 tables，~2 頁

### 3. 砍 figures variants

**現況**：`prism_grid_bound.pdf` + `prism_grid_bound_rp.pdf` + `prism_grid_bound_rpboth.pdf` + `prism_grid_bound_rplog.pdf`，共 4 個版本

**提案**：保留 canonical version（`prism_grid_bound.pdf`），砍 3 個 `_rp*` variants
**事實依據**：grep 確認 `_rp.pdf`、`_rpboth.pdf`、`_rplog.pdf` 都沒被 \input
**省下**：3 個 figure files（不省 paper 篇幅，但減少 build artifact 干擾）

### 4. WikiText 是否要砍？

**現況**：`table_qwen_all.tex` 包含 WikiText（r_s=0.54），明顯比其他低（其他 0.79-0.92）

**Trade-off**：
- 砍：narrative 更乾淨，避免 reviewer 質疑「為什麼 WikiText 表現這麼差」
- 留：保持 7 benchmarks 完整性，主文 L333 已說「language-modeling results follow the same pattern」（暗示 WikiText/FineWeb-Edu 有點不同）

**建議**：留在 appendix，但在 caption 加一行「WikiText shows weaker correlation (r_s=0.54), discussed in Limitations」並在 conclusion limitations 補一句

**事實依據**：r_s=0.54 來自 agent report，需要 user 進一步驗證實際數字

---

## E. 完整重構後的 Sec 5 結構提案

```
5.1 Experimental Setup（壓縮 1/3，把 teacher-forced + calibration 合併）

5.2 Predictiveness: Bound Tracks Risk Gap
    - Fig 2 (新): aggregated scatter + axis-dominance heatmap
    - Table 3 (新): 3 regimes 各 1 個代表 case
    - 1 段文字 covering both PTQ + LoRA rank consistency

5.3 Decomposability: Three Axes Localize Three Failure Modes
    - 直接展開「which axis fails where」原本 paragraph
    - 引用 Fig 2 右半 heatmap
    - 1 段文字 + Fig 3 (forget grid Llama+Qwen)

5.4 Actionability: From Diagnostic to Regularizer
    - Fig 4 (改): 2×2 整合 Llama + Qwen
    - Table 4: Llama TruthfulQA decomposition (保留)

5.5 Ablation
    - Table 5: combined ablation (W=I 為主，W_N 視覺淡化)
    - 1 段文字（壓縮，把現有 2 段合 1 段）
```

---

## F. 預期收益

| 改動 | 篇幅節省 | Reviewer 印象提升 |
|---|---|---|
| Fig 2 改成 scatter + heatmap | ~0.3 頁 | 高（axis dominance 視覺化） |
| Fig 3/4 整合 Llama+Qwen | ~0.2 頁 | 高（修正 cross-lineage 失衡印象） |
| Table 3 改 3 regimes | 0（換內容） | 高（直接呼應 Sec 5.3 narrative） |
| Sec 5.1 壓縮 | ~0.3 頁 | 中 |
| Appendix 砍 instruct + LIMA/etc | ~3.5 頁 | 中（appendix 不影響 review，但 supplementary 變乾淨） |

預估主文總共省下 ~0.8 頁，**騰出空間給 limitations 段落擴充 + 應對 rebuttal 預留 buffer**。

---

## G. 待 user 確認的關鍵決策

1. **Fig 2 改造**: scatter + heatmap，還是維持 4×5 grid？
2. **Fig 3 改造**: 2×3 整合？還是 2×2 簡化？還是維持 Llama-only 但加 Qwen 副圖？
3. **Table 3 改造**: 3 regimes 範例，還是維持 Llama main decomp + 加其他 family inline？
4. **WikiText**: 留 appendix（加 limitations）還是砍？
5. **Appendix instruct tables**: 全砍還是保留 1-2 個有代表性的？

---

## H. 事實驗證 checklist（執行重構前必查）

- [ ] WikiText r_s=0.54 是否屬實（agent 報的，需查 `table_qwen_all.tex`）
- [ ] Qwen3 Q8_0 head dominance 數字（agent 報 δ=0.15, γ=19.0，需確認）
- [ ] LIMA/no_robots/social_iqa 是否在 main text 任何地方暗用（grep 確認）
- [ ] Pareto 形式 (target task loss vs downstream |ΔR|) 的數據是否存在
- [ ] 整合後 Fig 3/4 需要的源數據是否齊全（Qwen drift trajectory）

要我從 G 區的決策開始一一過？或先執行 H 的事實驗證？
