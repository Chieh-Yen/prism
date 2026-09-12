# Sentence-Level Logic / Narrative Checklist for `paper/neurips_2025.tex`

逐項確認敘述跳躍、邏輯不接、antecedent 模糊、未明說的因果。每項標註：位置、原文、問題、建議。

優先度標記：
- 🔴 **High** — reviewer 真的會卡住或質疑
- 🟡 **Med** — 邏輯能 follow 但路徑陡峭，補一句更順
- 🟢 **Low** — 細節，加分但不必要

---

## A. 邏輯跳躍 / 因果未明說（高優先）

### 1. 🔴 Intro L150 — 兩 facts 直接 lift 到 bound
> "These two structural facts let us **lift the Procrustes feature-alignment residual into an exact closed-form upper bound** on the cross-entropy risk gap."

- 跳躍：linear head + LRH 兩個 facts 直接導出 "Procrustes residual lift 為 CE bound"，但 reader 不知道為何「linear head + LRH」就能讓 Procrustes 變 CE bound
- 缺的中間步：linear head ⇒ CE 對 features Lipschitz；LRH ⇒ orthogonal-only 對齊合理
- 建議補一句：
  > "...let us **bound CE risk by feature alignment** (the linear head makes CE Lipschitz in features) and **restrict alignment to orthogonal Procrustes** (the LRH licenses rotation-only matching), lifting the residual into an exact closed-form upper bound..."
- 或者接受 intro 的 brevity（Sec 3 會詳細展開）

### 2. 🔴 Sec 4 L306 — "preserves the coordinate basis" 為何就 "feature side dominates"
> "PTQ introduces element-wise numerical noise that **preserves the coordinate basis, so the bound's feature side dominates** via $(\Delta\rho)^2$ and $1-\Omega$."

- 跳躍：「保留 basis」→「feature side dominates」之間沒解釋
- 缺的因果：basis 保留意味 proxy features 與 target 在同一 frame，所有偏差都在 feature 軸（scale 或 shape），head 軸只在 protocol 也量化 lm\_head 時才出現
- 建議：
  > "PTQ introduces element-wise numerical noise that preserves the coordinate basis, so **the proxy's features remain in the target's frame** and deviations live entirely on the feature side ($(\Delta\rho)^2$ and $1-\Omega$); the head term contributes only when the protocol also quantizes \texttt{lm\_head}."

### 3. 🔴 Sec 5.5 L397 — "alignment can absorb the non-identity rotations" 機制隱晦
> "consistent with the head term being beneficial *when **the alignment can absorb the non-identity rotations*** between $H_T$ and $H_P$"

- 隱含機制：$W_N$ rotate $H_T$ 來更貼近 $H_P$，所以 head term 的「假性 misalignment」減少
- Reader 看不到這個推導
- 建議明寫：
  > "consistent with the head term being beneficial **because $W_N$ rotates $H_T$ to better match $H_P$**, reducing spurious head misalignment that $W{=}I$ leaves uncorrected."

### 4. 🔴 Sec 5.5 L395 — "γ injects non-monotone variance" 機制未解
> "the $\gamma$ term **injects non-monotone variance** across the pooled scatter."

- 為何 γ 會「注入 non-monotone variance」？機制是：γ 在 GGUF 子集 >0、在其他 protocol ≡0，pooling 起來時 γ 對 ranking 的貢獻就是「有時加訊號、有時加 0」，造成 Spearman 被稀釋
- 建議補：
  > "the $\gamma$ term **contributes only in the GGUF subset and adds zero in others**, so pooling the two regimes dilutes the bound–risk Spearman."

---

## B. Antecedent / 用詞精準度（中優先）

### 5. 🔴 Abstract L131 — "turning A into B" 的 A 是誰？
> "...localize distinct empirical regimes ..., **turning a single-number similarity into a mechanism-level diagnostic**."

- 句子主詞是 PRISM 的成果，但「a single-number similarity」指的其實是 prior art (CKA)，不是 PRISM 本身。語意是「PRISM 取代 CKA」，但句構像「PRISM 把自己變成 diagnostic」
- 建議：
  > "...replacing **single-number similarity scores** with a mechanism-level diagnostic."
- 或：
  > "...**moving beyond single-number similarity** to a mechanism-level diagnostic."

### 6. 🟡 Intro L159 — Ω 捕捉的是 similarity 還是 drift
> "the risk gap is governed by backbone **shape drift**, which **$\Omega$ captures differentiably**."

- 不精確：Ω 捕捉 shape **similarity**；shape **drift** 是 $1-\Omega$
- 建議擇一：
  - "...governed by backbone shape drift, **whose differentiable proxy is $\Omega$**"
  - "...governed by backbone shape drift, which **$1-\Omega$ captures differentiably**"

### 7. 🟡 Intro L158 — "prevent it" 的 it 是 drift 還是 diagnostic
> "The same quantity that diagnoses a variant's drift **can also prevent it**."

- "prevent it" 字面是「prevent the diagnosis」，雖然 reader 會自動讀成「prevent drift」，但嚴格上 metric 本身不 prevent drift，是 derived regularizer 才會
- 建議：
  > "The same quantity that diagnoses drift **can also be used to prevent it**."
- 或更明確：
  > "The same quantity that diagnoses drift **also enables a regularizer that suppresses it**."

### 8. 🟡 Sec 3.1 L206 — "lie in a common lineage" 用詞怪
> "we assume the two backbones **lie in a common lineage**"

- "lineage" 不是 space，"lie in" 搭配怪
- 建議：
  - "we assume the two backbones **share a common lineage**"
  - "the two backbones **belong to a common lineage**"

### 9. 🟡 Sec 3.3 L268 — "in most configurations" 例外情況不明說
> "the frozen-head regimes studied here---LoRA and FP16-head PTQ---**keep $H_T = H_P$ in most configurations**, so the head term simplifies at $W = I$"

- "in most configurations" 引發疑問：例外是哪些？實際指 GGUF k-quant tiers
- 建議顯式點出：
  > "...keep $H_T = H_P$ in most configurations (**the exception being GGUF k-quant tiers that quantize the output embedding**), so..."

---

## C. Imprecise / 過度宣稱（中優先）

### 10. 🔴 Sec 3.5 L290 — "stabilize target-task learning" 過度宣稱
> "Pinning shape should therefore **stabilize target-task learning** and suppress catastrophic forgetting on downstream tasks at once."

- 「pinning shape 會 stabilize target-task learning」其實**值得質疑**：固定 shape 可能反而**傷害** target task 的擬合（限制 backbone 的更新空間）
- 後面實驗也只 validate「downstream forgetting」減少，沒有特別 validate target task 變穩定
- 建議刪掉前半過度宣稱：
  > "Pinning shape should therefore **curb backbone drift** and suppress catastrophic forgetting on downstream tasks."

### 11. 🟡 Sec 5.2 L360 — 為「scale collapse」推薦 remediation 但實驗未觀察到
> "Each dominant axis suggests a different remediation---**per-channel outlier smoothing for scale collapse**, Hessian-aware reconstruction for shape distortion, and FP16-head retention for head divergence"

- 同一段前文說「Scale drift 在我們的 PTQ grid 上沒有 cleanly dominate 任一 variant」
- 但這裡卻為 scale collapse 推薦一個 remediation，邏輯上略不一致（在不存在的問題上開藥方）
- 建議：括號標註 "(less prominent in our grid but the established remedy when scale dominates)"
- 或：刪掉 scale 那項 remediation，只保留有觀察到的兩項

### 12. 🟡 RW L176 — "a different object" 沒解釋為何重要
> "a recent decodability bound reaches downstream via whitened kernels and freshly-optimized linear probes, **a different object than the deployed prediction heads we evaluate**."

- 「不同 object」這件事被丟出來，但沒解釋 reader 為何要 care
- 建議補上影響：
  > "...freshly-optimized linear probes, **a different object than the deployed prediction heads we evaluate---so its bound governs probe-based decodability rather than the deployed model's risk**."

---

## D. 句構 / 修辭（低優先）

### 13. 🟡 Sec 5.3 L372 — "makes a structural observation visible that..."
> "The decomposition **makes a structural observation visible that a unified distance would collapse**: ..."

- 文法 OK 但 "that a unified distance would collapse" 的 that 子句在 "observation" 後面、結構不清晰
- 建議重排：
  > "The decomposition **exposes a structural observation that a unified distance would mask**: ..."

### 14. 🟢 Intro L144 — "pin down downstream behavior"
> "These scores, however, **do not pin down downstream behavior**"

- "pin down" 略口語
- 建議：
  > "These scores, however, **do not determine downstream behavior**"
- 或："**do not predict downstream behavior**"

### 15. 🟢 Sec 5.3 L372 — "supplies the empirical signal for scale-axis separability"
> "...thus shows the two arms as non-redundant channels and **supplies the empirical signal for scale-axis separability** in our experiments."

- 「supplies the empirical signal for X」是 meta-talk，繞口
- 建議：
  > "...and **provides empirical evidence that the scale axis is separately measurable from shape**."

### 16. 🟢 Intro L155 — 三軸描述平行性破壞
> "scale-dominated failures localize the loss to **outlier-channel corruption**, shape-dominated to **distorted feature geometry between tokens**, and head-dominated to **perturbation of the prediction head while the backbone is preserved**."

- 前兩項短，第三項多帶 "while the backbone is preserved"，平行結構不對稱
- 建議簡化第三項：
  > "...and head-dominated to **a perturbed prediction head**."
- 或前兩項也補對應 qualifier（但會更冗長）

### 17. 🟢 Conclusion L407 — "argued for via" 略生硬
> "the per-axis remediations ... **are *argued for* via the bound's structural decomposition**; experimentally verifying that each remediation closes the corresponding axis is a natural next step we do not attempt here."

- "argued for via" 有點生硬
- 建議：
  > "the per-axis remediations ... **are *suggested* by the bound's structural decomposition**; experimentally verifying that each remediation closes the corresponding axis remains future work."

---

## E. 不需要改的（已查過、敘述合理）

- **Intro 五段結構**（diagnostic gap → similarity 不夠 → LLM geometry tractable → PRISM 三軸 → from diagnostic to training）：清晰、層次分明
- **Sec 3.2 Theorem 推導順序**（hybrid risk → triangle → δ + γ → 各自 bound → unified theorem）：標準、正確
- **Sec 5.1 calibration set size 段** (L336)：長但邏輯完整，preempt reviewer 對 Spearman + N=512 的質疑
- **Sec 5.1 teacher-forced scoring** (L342)：長但結構清晰（why teacher-forcing → 三個 benefits → 何時不適用）
- **Sec 5.5 第一段 vs 第二段對比**：W=I 三行 vs W=W_N 三行，平行結構好
- **三個 contributions** (L165, L167, L169)：標準 NeurIPS 寫法，dense 但 OK

---

## 建議處理順序

1. **A 區（4 項）**先看：reviewer 真的會卡住的因果跳躍
2. **B/C 區（8 項）**：精準度問題，影響可信度
3. **D 區（5 項）**：細節 polish

要我從 A1 開始一個一個確認，還是你直接挑要改的？
