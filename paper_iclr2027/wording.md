# Wording Checklist for `paper/neurips_2025.tex`

逐項確認是否需要修改。每項標註：位置（行號）、原文、建議替換、理由。

優先度標記：
- 🔴 **High** — 明顯非 ML 慣用、reviewer 可能皺眉
- 🟡 **Med** — 略文藝/informal 但可接受
- 🟢 **Low** — 細節 polish，可選

---

## A. 文藝詞 / 非 ML 慣用語（高優先）

### 1. 🔴 "kin"（L143, RW）
> "SVCCA, CKA, **and their Frobenius-norm kin**"

- "kin" 在 ML 論文極罕見，偏文學
- 建議 → "and **their Frobenius-norm relatives**" 或 "and **related Frobenius-norm variants**"

### 2. 🔴 "spawns"（L139, intro）
> "A single base checkpoint now **spawns** many post-training variants"

- "spawn" 多用於 process（OS、game），用在 model 偏俚語
- 建議 → "**yields**" / "**produces**" / "**gives rise to**"

### 3. 🔴 "Frobenius-norm kin each reduce" 句式（同 L143）
> "each reduce two activation matrices to a single alignment score"

- "each reduce" 文法上 OK，但連用 "kin" 後讀來累贅
- 替換 #1 後此處自動順暢

### 4. 🔴 "name an axis that diagnosis could act on"（L144）
> "a single number still flags drift without **naming an axis** that diagnosis could act on"

- "name an axis" 動詞搭配怪
- 建議 → "**without identifying an axis** for diagnosis" / "**without pointing to an axis** for diagnosis"

### 5. 🔴 "bound-certified" / "lens"（L132, L153, L160）
> "PRISM from a **diagnostic lens** into an active training signal"（abstract）
> "a **bound-certified, mechanism-level diagnostic**"（intro）
> "post-hoc **lens** into an active constraint"（intro）

- "lens" 是文學比喻，"bound-certified" 是自創複合詞
- 建議：
  - "diagnostic lens" → "**diagnostic tool**" 或直接刪 "lens"
  - "bound-certified" → "**principled**" 或 "**with provable upper bounds**"

### 6. 🔴 "directions along which the data actually lives"（L264）
> "weighted by $\Sigma_P$ so that only **directions along which the data actually lives** can contribute"

- "actually lives" 是口語比喻
- 建議 → "**directions where the data has support**" / "**directions spanned by the data**"

### 7. 🔴 "the alignment can absorb the non-identity rotations"（L397）
> "consistent with the head term being beneficial *when **the alignment can absorb the non-identity rotations*** between $H_T$ and $H_P$"

- "absorb" 是物理/化學比喻
- 建議 → "**when the alignment can accommodate the rotations**" / "**when the alignment can offset the rotations**"

### 8. 🔴 "Pinning shape"（L290）
> "**Pinning shape** should therefore stabilize target-task learning"

- "pin" 在 ML 偶見（pinning weights）但口語感強
- 建議 → "**Constraining shape**" / "**Anchoring shape**" / "**Penalizing shape drift**"

---

## B. 資訊過密 / 複合詞（中優先）

### 9. 🟡 "single-number similarity"（L131 abstract）
> "turning a **single-number similarity** into a mechanism-level diagnostic"

- 不夠 ML 慣用詞
- 建議 → "**scalar similarity**" / "**univariate similarity score**"

### 10. 🟡 "single opaque number"（L129 abstract）
> "reduce geometric mismatch to a **single opaque number**"

- "opaque" 帶情緒，正式論文偏少用
- 建議 → "**a single uninterpretable score**" / 保留 "single number"，刪 "opaque"

### 11. 🟡 "diagnostic gap"（L129 abstract, L137 intro）
- 自創詞，paper 反覆用作 selling point
- 建議：保留（已成文章 keyword），但確保首次出現有定義性脈絡（目前 abstract 第一句就解釋了 → OK）

### 12. 🟡 "mechanism-level diagnostic"（L131, L153, L360）
- 自創複合詞，ML 較少這樣用「-level」
- 建議 → 改為「**axis-level diagnostic**」（呼應論文三軸主題）或「**component-level diagnostic**」
- 注意：用了 3 次，要改全改

### 13. 🟡 "PRISM accommodates a family"（L267）
> "**PRISM accommodates a family** of Procrustes similarities"

- "accommodate" 略 awkward
- 建議 → "**PRISM supports a family** of ..." / "**PRISM admits a family** of ..."（admit 在數學常用）

### 14. 🟡 "freshly-optimized linear probes"（L176, RW）
> "via whitened kernels and **freshly-optimized** linear probes"

- "freshly-optimized" 不是標準 ML 術語
- 建議 → "**newly-trained linear probes**" / "**re-fitted linear probes**" / "**from-scratch linear probes**"

### 15. 🟡 "scale-visible PTQ case"（L360）
> "the most **scale-visible** PTQ case is Qwen3-Base Q2\_K"

- "scale-visible" 是即興複合詞
- 建議 → "the **most scale-affected** PTQ case" / "the PTQ case where **scale drift is most prominent**"

### 16. 🟡 "training-side schedule"（L295）
> "(**training-side schedule** and overhead in Sec.~..."

- "training-side" 不夠標準
- 建議 → "(**training schedule** and overhead in Sec.~..."

---

## C. 細節 polish（低優先）

### 17. 🟢 "downstream of pre-training"（L138）
> "an engineering bottleneck has emerged **downstream of pre-training**"

- "downstream of" 通常指 data flow / pipeline
- 建議 → "**after pre-training**" / "**in the post-training stage**"

### 18. 🟢 "blind trial-and-error"（L140）
> "practitioners resort to **blind trial-and-error**"

- "blind" 略口語
- 建議 → 直接 "**trial-and-error**"（trial-and-error 本身已含「沒有指引」意味）

### 19. 🟢 "factor cleanly into"（L148）
> "Transformer LLMs **factor cleanly into** a non-linear backbone..."

- "cleanly" 口語
- 建議 → "**factor into**" / "**cleanly factor into**"（換語序聽感差別不大，可保留）

### 20. 🟢 "plurality of cells"（L395）
> "wins the **plurality of cells**"

- "plurality" 是政治/法律用語
- 建議 → "**the most cells**" / "**a plurality**"（保留也可，但可換）

### 21. 🟢 "sidestepping KV-cache growth"（L342）
> "**sidestepping** KV-cache growth and sampling overhead"

- "sidestep" 略口語
- 建議 → "**avoiding** KV-cache growth" / "**eliminating** KV-cache growth"

### 22. 🟢 "central pattern"（L372）
> "the forgetting regime's **central pattern**"

- 略文藝
- 建議 → "**main pattern**" / "**primary pattern**"

### 23. 🟢 "locus of the forgetting signal"（L372）
> "consistent with backbone geometry being the **locus of the forgetting signal**"

- "locus" 學術但偏 statistics/genetics 用法
- 建議 → "**source of the forgetting signal**" / "**carrier of the forgetting signal**"

### 24. 🟢 "scatters the PRISM bound"（L351 fig caption）
> "Each subplot **scatters** the PRISM bound..."

- "scatter" 作及物動詞，文法 OK 但稍微少見
- 建議 → "Each subplot **plots** the PRISM bound (x-axis) against ..." / "Each subplot is a **scatter plot of** the PRISM bound vs. ..."

### 25. 🟢 "leaves the risk gap clean to analyze"（L268）
> "leaves the risk gap **clean to analyze**"

- "clean" 口語
- 建議 → "**leaves the risk gap analytically simple**" / "**tractable to analyze**"

### 26. 🟢 "non-trivial stress test"（L330）
> "downstream forgetting on general-knowledge benchmarks ... provides a **non-trivial stress test**"

- "non-trivial" 是 ML/CS 普遍用詞，但 reviewer 偏好具體形容
- 建議 → "**meaningful stress test**" / "**substantial stress test**" / 保留也行

### 27. 🟢 "perfect"（L360）
> "backbone scale and shape are essentially **perfect**"

- "perfect" 形容數字，略口語
- 建議 → "essentially **at the noise floor**" / "essentially **identical**" / "essentially **zero**"

### 28. 🟢 US/UK spelling consistency
- L351, L368: "**colours**" → "**colors**"（NeurIPS 通常 US English）
- L387: "**behaviour**" → "**behavior**"
- 主文 grep 後共有 2 colour + 1 behaviour

---

## D. 不需要改的（已查過、ML 通用）

以下在 ML 論文常見、不需要動：
- "lift / lifts into" — 數學/CS 通用
- "axis / axes" — 三軸論述的核心詞
- "diagnostic" — 已成 paper keyword
- "actionable axes" — 已成 ML 慣用
- "drift" — ML 通用
- "regimes" — 常見
- "lineage" — 本文已建立為 LLM 變體血緣的標準用法
- "instantiates" — CS/ML 接受
- "active subspace" — statistics 標準術語
- "doubles as" — 接受度高
- "grounded entirely in" — 接受
- "agentic workflows" — 已成 LLM 標準

---

## 建議處理順序

1. 先處理 A 區 (8 項)：影響最大
2. 再看 B 區 (8 項)：是否同意調整
3. C/D 視時間決定

要我從 A1 開始一個一個確認，還是你直接挑要改的？
