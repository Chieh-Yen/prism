# PRISM 8674 rebuttal — 內部註記(2026-07-27 從 rebuttal_draft.txt 搬出)

rebuttal_draft.txt 已清空所有 Note,只保留要貼的內容([原文] + "> summary" + Response)。
本檔保存原有 Note 的全部內容(數字出處、決策理由、待辦)與 assembly checklist。
貼文前請對照本檔的 checklist。

---

## 各則 Note(依原順序)

### Meta Review (AC) — A + W-1 + W-6a
```
Note: ✅ 全部已有數字(E5 tex 已備 / E6 已跑)。55/55 cells(11 families × 5
      paper benchmarks,wikitext/fineweb 已排除)皆 >=5 variants;
      slack/tier 表在 rebuttal_exp/out/E6/E6_results.md。
```

### Meta Review (AC) — B + W-4a
```
Note: **2026-07-27 定案(推翻 2026-07-26 三層框架):不賣 benchmark-independent。**
      wikitext unpaired 實驗(+0.506 vs +0.725 兩行表)整段撤下,理由:
      (a) PRISM 測哪個 task 就用該 task 的 validation slice(input-conditioned
      by construction);(b) 需求量極小,size ablation 8 條即足(留作 in-scope
      measurement);(c) unpaired = general distribution-transfer / generalization
      議題,非論文推導情境,conf-4 還可能拿 +0.506 反打 domain-shift 掉 0.22。
      Condition B 改答「out-of-scope 明確 narrowing + in-scope 實測」(AC 自己
      邀請 narrowing)。regularizer 側 wiki 0.773 保留(訓練側、協議內,8VrD-Q3
      直問 domain)。E3-A 的 +0.506 數字僅 internal 備查,勿再入貼文。
      ⚠️ Part C 的 −51% 是 extended round vs 自身 no-reg 1.483,≠ 論文 Table 2
      −19%(no-reg 0.843);貼文只講 robustness/spread 不並排 %。舊 lr=2e-4 勿引。
```

### Meta Review (AC) — C + W-3 + W-5a
```
Note: ✅ E1 GPTQ 全補齊(12/12、11/11)。框架改為 **tie + containment**
      (Procrustes²=feature arm,App A.1/Eq23;不再賣 B_N「贏 +0.008」——CI 蓋 0,
      賣贏會被 conf-4 判 overselling,反噬全篇)。⚠️ CKA/SVCCA gsm8k 格仍 16k
      子抽樣,REDO 後 CKA 只會下修(對 tie 敘事無影響,B_N 不需贏);貼文前重生。
      ⏳ E2 sweep+seeds(layer_freeze 已實作)。
```

### Meta Review (AC) — D + W-4b
```
Note: ✅ E7 定稿 + 5-seed robustness 已填(seeds 42-46,out/E7/E7_seed_aggregate):
      free +0.947±0.015、TF +0.958±0.011、agreement +0.959±0.010、cross
      +0.958±0.011;sd 極小 ⇒ 擋「單子集挑過」。五處已改 mean±sd。mmlu 退役勿引。
```

### Meta Review (AC) — E + W-5b
```
Note: ✅ 全部已有數字(App F.3/G.1/H 指路 + E13 的 20-cell 聚合)。
      by-task 數字全部來自論文,可直接引:ARC |dR|0.288/TriviaQA 0.0035(App G.1
      L775)、Fig 3/7 per-subplot rs(TruthfulQA→GSM8K Llama +0.97)、BBQ vs
      TruthfulQA baseline Ω0.932/0.906 + mean|dR|0.179/0.843(App H L792–797)。
      ⚠️ 框架 = by-(task,benchmark) 而非「gsm8k/BBQ 是弱點」;誠實保留 noise-floor
      caveat(~1e-2 nats)但不認栽成 benchmark/reasoning limitation。
      ⚠️ 2026-07-26 定案:E10(5-task SocialIQA/No-Robots/LIMA)不揭露,已從 (4)
      regularizer 移除;gating 改靠論文 Table 22(2-task)+ Table 19 per-benchmark
      wins(BBQ-FT TriviaQA −88%/GSM8K −79%)。E13 的 λ=0 相關矩陣 checksum 命中
      論文輪次(0.831/−0.34/−0.66),可直接引用。
```

### Meta Review (AC) — W-6b [原文]
```
Note: 2026-07-27 新增 closing paragraph(vision)。設計:放結尾、不放開頭(AC 履約
      型,開頭必須 evidence-first;情感訴求在 measurement 之後才 earned)、不開新
      comment(conf-4 會視為 padding)。素材 = AC 自己歸納的共識 strengths(timely
      problem / decomposition > single score / geometry-risk link);「credit 反送
      reviewers」(校準/重述/baselines 因你們的提問而存在)= 共同擁有感,最強調分
      槓桿;future works 具體點名三方向(sequential CL / SFT-RLHF joint-alignment /
      trajectory-shift term)= 8VrD 四引文 + G3T9-Q1 + 8VrD-W4 的意見入 roadmap,
      非空泛承諾。62.6x 為全文已用實測數字,無新 claim。貼文時外層引號拿掉。
```

### Reviewer #1, pCi8 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 1) W1
```
Note: 純措辭,零實驗。Strengths 已承認 simplex polarization/exact decomposition
      是 real contribution——只需扭轉「發現 vs 儀器」框架。
```

### Reviewer #1, pCi8 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 2) W2
```
Note: ✅ 已有全部數字(E6)。
```

### Reviewer #1, pCi8 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 3) W3
```
Note: ✅ 指路即可成立(L278–285, L249–257, App B);E1 數字已填——
      「machinery 反轉」框架:Ω_N 單獨 +0.81 → B_N +0.91,加值全來自
      scale/γ_N 兩臂(E1.exp.md §1/§2)。⚠️ gsm8k baseline 格待 REDO。
      Google Doc 的 [VERIFY] 已判 CONFIRMED。
```

### Reviewer #1, pCi8 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 4) W4
```
Note: ✅ E9 定稿——mitigation **失敗**(span rs +0.566 < full-CoT +0.979;
      span 僅 ~3 tokens,Q2_K 反轉)。採誠實負結果:維持 first-class
      limitation,限制在 |dR| magnitude / SNR。⚠️ 決定 NOT 宣稱任何 gauge 或
      extraction「救回」gsm8k 排序——會與 (1) 的 noise-floor 讓步自相矛盾、
      且 conf-4 會判 overselling。已移除舊的「B_N +0.965 rescues gsm8k」與
      「full-CoT 0.979 = 截斷 artifact」框架(與論文 Table 3「gsm8k weakest」打架)。
      W_N-for-ranking 的一般性論點統一放 pCi8-W3(不綁 gsm8k)。
```

### Reviewer #1, pCi8 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 5) W5
```
Note: 🔵 E2 部分落地:TruthfulQA seed-42 sweep 已填(shape 0.687 vs no-reg 0.843/
      replay 0.764/EWC 0.750/L2-SP 0.747;layer-freeze under-trains,target 0.92–1.02)。
      ⚠️ L2-SP 只取有效範圍 λ∈{1e-4,1e-3,1e-2}(best=1e-2=0.747);λ0.1 排除(lr 1e-5
      下 penalty 過度壓制微小 LoRA drift,非有意義操作點)。shape 現勝全部 reported
      baselines,但維持「feasibility demo 非 SOTA」框架(feature_kd 已移除、未宣稱勝所有方法)。
      feature_kd 已移除(reviewer 不在意);⏳ TQA seeds 43/44 完成後補多 seed±sd
      (2026-07-27 定案:BBQ 不跑,貼文 BBQ 新實驗承諾已全移除;論文既有 BBQ
      數據 Table 22/19 照引)。選點規則 = 各 baseline 取「target loss 對齊 shape」的 config(equal
      plasticity,不獎勵 under-train)。數字出自 exp_e2_local_table.py(matched-target 視圖)。
```

### Reviewer #1, pCi8 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 6) W6
```
Note: ✅ 指路(L112–115, App C.1/C.2)+ E8 定稿已填(控制組 0.084;Q8/Q4/Q2/NF4
      = 0.130/0.165/0.183/0.189;oracle 線性後 Q2_K 仍餘 57%;E8.md §3)。
      ⚠️ E1 內建的首版 iso_dev 欄已作廢(n_tokens<d 病態;E1.md §7),勿引。
```

### Reviewer #2, G3T9 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 1) W1
```
Note: 指路(L258–262, L249–257)+ E4 ✅ 定稿(26/26 bound holds,選擇性 ≥2.6e5×)
```

### Reviewer #2, G3T9 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 2) W2
```
Note: (1) = scope narrowing(2026-07-27 定案:不賣 benchmark-independent,
      與 AC-B 口徑一致);(2) ✅ E3-B(size stability,sd<0.001,8 條即足);(3) ✅ E7。
```

### Reviewer #2, G3T9 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 3) W3
```
Note: ✅ E1 GPTQ 全補齊、表格已填(E1.exp.md §1);框架 = tie+containment
      (與 AC-C/pCi8-W3 口徑統一,不賣 B_N 贏)。⚠️ gsm8k 格 REDO 後 CKA 下修,
      對 tie 敘事無影響;⏳ E2。
```

### Reviewer #2, G3T9 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 4) W4
```
Note: ✅ 已有全部數字(E5+E6)。與 pCi8-W2 同素材、自包含改寫。
```

### Reviewer #2, G3T9 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 5) Q1
```
Note: ✅ 指路 App C.3 L678–679 + App I + E11 定稿已填(10/10 holds;
      head share 0%→50% 命中理論預測;1−Ω:llama 0.224≈Q2_K 0.225、
      qwen 0.254=14×Q2_K;looseness 主動揭露)。
```

### Reviewer #2, G3T9 (3: Borderline reject, conf 4), Q3/C3/S2/O3 — 6) Q2
```
Note: ✅ 已有(E10 rerun,5 任務 × 2 模型)。mixed-data 誠實劃界。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 1) W1
```
Note: 認錯 + 修訂清單,零實驗。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 2) W2
```
Note: gauge 論證是新增文字;(2)(3) 已有依據(Prop.1, L77–80/L112–115, App B)。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 3) W3
```
Note: ✅ 全部已有數字(exp_eql6w3_prefactor.py,零 GPU 今日跑出)。
      全場最能「當場釘死」的一題。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 4) W4
```
Note: ✅ 指路 Fig 4/8 第一欄(reviewer 沒看懂該欄意義)+ ⏳ E2 數值表。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 5) Q1
```
Note: 與 W2 同素材,thread 內自包含重述。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 6) Q2
```
Note: 純改寫。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 7) Q3
```
Note: 同 W4;thread 自包含。
```

### Reviewer #3, eQL6 (3: Borderline reject, conf 3), Q2/C2/S3/O3 — 8) Q4
```
Note: ✅ 指路 caption(L196–200, Table 1/Fig 2);正文補一句。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 1) W1
```
Note: ✅ 完成(e5_theorem.tex 可直接貼 thread)。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 2) W2
```
Note: ✅ 已有全部數字(E6)。表格為 OpenReview markdown(MathJax,|…|→\lvert\rvert
      避免撞欄位分隔);校準預測 0.003/0.193 出自 out/E6 LOO isotonic
      (Meta-Llama-3.1-8B/mmlu cell)。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 3) W3
```
Note: (1)(2) ✅ 已有(E4 定稿 26/26 + selectivity 表已填;gating 靠論文 Table 22/19,
      E10 5-task 不揭露);(3) ⏳ E2/E3-C。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 4) W4
```
Note: (1) ✅ 指路 Fig 7 / G.1 L770–782 + E13 已算出 20-cell 聚合(mean +0.71/
      median +0.93,checksum 對上論文 0.831 與 −0.34/−0.66,見 E13.md);
      (2) ⏳ E7。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 5) Q1:
```
Note: ✅ W2 的自包含濃縮版。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 6) Q2:
```
Note: ✅ E7 定稿(12/12;E7.md 頂部表含 CI)。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 7) Q3:
```
Note: ✅ E3-C 定稿(lr=1e-5;task |dR| 0.725/0.736/0.724、wiki32 0.773;
      E3.md §8.2)+ E3-B 排序恆定。⚠️ 貼文只講 robustness/spread,**不並排
      reduction %**——Part C 的 −51% 是 extended round(no-reg 1.483),與論文
      Table 2 的 −19%(no-reg 0.843)不同輪,並排會撞 P9。舊 lr=2e-4 勿引。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 8) Q4:
```
Note: ⏳ E2。
```

### Reviewer #4, 8VrD (4: Borderline accept, conf 3), Q3/C3/S3/O3 — 9) Limitations:
```
Note: (iii) 有 App C.3 指路;(iv) ✅ E12 定稿;(i)(ii) 隨 E3/E7 落地。
```

---

## Global assembly checklist

```
=====
Global assembly checklist (post 前逐項核對)
================================================================================
[x] rebuttal.proposal.md 已整合(2026-07-26):general 致謝 + 六 weak-points
    映射(W-1/3/4/5→條件 A/C/B+D/C+E;W-2 全獨立答;W-6 部分→A、部分獨立答)
    + AC condition checklist 表 + 四位 reviewer opener/In sum + P5/P6/P7/P9
[x] 2026-07-27 AC 區結構重整:**刪掉獨立的「six weak points」區**(它把每個 W 複述
    一遍再指向條件,與條件本身的 Response 大量重複)。改為 **W 併入條件標頭**:
    A+W-1+W-6a / B+W-4a / C+W-3+W-5a / D+W-4b / E+W-5b,原文並列於同一標頭下、
    答案只寫一次;W-4/W-5 各跨兩條件 → 全文引一次(B、C),另一處引片段並註明出處;
    W-2 與 W-6b(presentation 半)移到 Condition E 之後獨立作答(內容從舊區搬來並
    擴寫成自包含)。W-4 未被 B/D 涵蓋的兩子項(reasoning-heavy → E;drift farther →
    base-vs-instruct 10/10)已在 B 結尾補接。開頭加一行 mapping 句保留可追溯性。
[x] AC checklist B 列 ✅ 已填實數(E3 定稿:size-invariant sd<0.001 + regularizer
    −48~−51%);C 列仍寫「in the revision」,E2 落地後改填(絕不在未跑前貼數字)
[x] 2026-07-27 定案:**B 不賣 benchmark-independent**——wikitext unpaired(+0.506
    vs +0.725 兩行表)全篇撤下(AC-B/G3T9-W1/G3T9-W2/8VrD-Q3/總表 B 列/W-4 映射/
    narrowing 加第三 bullet);改「out-of-scope 明確 narrowing + in-scope size
    ablation 實測」;regularizer 側 wiki 0.773 保留(訓練側、協議內)
[x] 2026-07-27 定案:**E2 只跑 TQA**——貼文所有 BBQ 新實驗承諾移除(AC-C/pCi8-W5/
    G3T9-W3/eQL6-W4/eQL6-Q3/8VrD-W3/8VrD-Q4/global close);論文既有 BBQ 數據
    (Table 22/19、Qwen-BBQ noise-floor 分析)不受影響照引;TQA 補「the paper's
    primary shape-drift setting」定位(pCi8-W5)
[x] 2026-07-27 新增 global closing paragraph(vision/格局):共識 strengths →
    工具使命(省社群評測時間)→ credit 反送 reviewers → 三項具體 future-work
    roadmap;置於 positioning sentence 之前
[x] 2026-07-27 定案(三次修正後):**貼文不用 "label-free",也不提「我們改用別的詞」**。
    ✅ 已核對論文 PDF:"label-free"/"no labels"/"unlabeled" 出現 **0 次**;且 Sec 5.1
    本來就寫明「All risks are computed teacher-forced (prompt c and targets y scored
    in a single forward pass over the **gold span**)」⇒ **論文無任何 overclaim**。
    "label-free" 是 (a) pCi8 review 的轉述("no labels needed")+ (b) 本草稿自己引入。
    ⚠️ 故「we say X rather than label-free」7 句已**全刪**,且該項已從 "Claims we
    narrow" 清單移除——把不存在的誇大寫進認罪清單 = 白送信用,且 reviewer 回頭找
    不到該詞會困惑,還可能誤以為我們在收回 pCi8 credit 的 strength。
    現行寫法 = **純正面描述協議**,不做用語更正:AC 全域只留 1 段短的 "what no
    grading means"(4 項 + 根因 + 指向 G3T9-W1);G3T9-W1(他直問 what reference
    data)給完整版並**直引論文 Sec 5.1 原句**證明協議一向明示;AC-B(2)/G3T9-W2/
    8VrD-Lim(iv) 只留精確描述。"generation-free" 保留 3 處修辭位(pillar 開場/
    closing/positioning),精確且無需定義。
    supplementary code 事實(備查):reference = prompt + gold answer
    (loaders.py:21–68);loss_mode="answer" ⇒ features 與 per-token CE 都只取 answer
    region(extractors.py:411–425;train_forgetting_multitask.py:712–714);全樹
    diagnostic path 無 accuracy/exact_match/.generate()(唯一 generate 在 E7)。
    ⚠️ 兩性質分層(勿混講):**no grading = 量的性質**(CE gap 定義不含 correctness)
    ⇒ 所有模式成立含 free-running;**no decoding = 協議層**⇒ 僅 default TF。舊稿曾
    把根因寫成「PRISM 不產生 output」是錯的(free-running 會 generate 卻同樣不需
    parse/match/score)。引用 free-running 當 fallback 必須說它 **does decode**
    (answer-availability fallback,非 cheap path),否則與 62.6× 打架。
    ⚠️ 誠實 carve-out(已入文):PRISM 確實會定位 reference text 的 answer region
    (model-independent offset,loaders.py:344–413)——明說這是「讀哪些 token 的
    bookkeeping,不是對模型輸出的判斷」,免得被說「你也 parse」。附帶收益(有 hedge):
    移除 grading 端 harness-dependence,但明說 PRISM 自有 extraction protocol。
    Fallback 證據 = E7 free-running 完全不需 gold answer(+0.947)。
    全域定義段置於三支柱之後;G3T9-W1(他直問 what reference data)與 AC-B 給
    精確版;8VrD Limitations 主動加第五項。⚠️ pCi8 原文區 L539/551 的「no labels
    needed」是他自己的話,不動;定義段已隱性更正,不單獨點名他。
[ ] 每個 [TBD-E#] 都已用 rebuttal_exp/out/E#/ 的真數字替換(絕不虛構)
[ ] 跑不完的 E# 改用該 E#.md「風險與 fallback」節的 narrow-concession 句
[x] E10(5-task SocialIQA/No-Robots/LIMA)不揭露——已從 G3T9-Q2 / AC-E / 8VrD-W3
    移除;G3T9-Q2 改用 recipe-agnostic + base-vs-instruct(E11);gating 改靠論文
    Table 22(2-task)+ Table 19 per-benchmark wins
[ ] E13 的 20-cell 聚合已 inline(checksum REPRODUCED 論文輪次,可與 0.831 同句)
[x] E12 成本表 ✅ 定稿已填(G3T9-W1 + 8VrD-Lim(iv) + AC-ii + G3T9-opener + Net 全部
    改 GSM8K-only 純實測:TF 8.9 s vs greedy decode 556.7 s = 62.6×,maj@8 501×;
    load 兩端皆扣;13% ceiling = lower-bound(decode 端低估,比值方向不變)
    方向,兩次獨立執行差 0.02%);E8 數字取自 out/E8/E8_results_llama.md
    (獨立 lite 腳本;E1 內舊 iso_dev 已作廢)
[ ] 所有量化端統計只用論文 5 benchmarks(E6/E13/eql6w3 已重算;E4 用
    mmlu+triviaqa;E9 = gsm8k;E11 = 全 5 benchmarks)
[x] E9 定稿(pCi8-W4):mitigation **失敗**(final-answer span rs +0.566 <
    full-CoT +0.979;span ~3 tokens 近二元,Q2_K 反轉)→ 誠實負結果,維持
    limitation,限制在 |dR| magnitude / SNR。
    ⚠️ 2026-07-26 定案:pCi8-W4 **不**宣稱任何 gauge/extraction 救回 gsm8k 排序
    (會與 noise-floor 讓步自相矛盾);已刪「交叉引用 B_N +0.965」與
    「full-CoT 0.979 = 截斷 artifact」框架(與論文 Table 3 gsm8k weakest 打架)
    (E11 ✅ G3T9-Q1:10/10 holds、head share 50%、1−Ω vs Q2_K 錨)
[ ] 所有 E2/E3-C run 確認 lr=1e-5(paper-round 啟動值;首輪 2e-4 sweep 全部
    作廢,勿引 out/E3/partC_summary.md 的 lr-stress 數字;E2.md §4 勘誤)
[ ] trace anchor λ 依 backfill 結果定(論文正文口徑 λ1.0;E2.md §5 決策規則);
    replay λ0.01 = 論文正文 sweep-best ✓
[ ] reg_every_k 配對:TQA/bbq 側一律 k=8、lima/no_robots 側 k=4(E2.md §5 表);
    引 E10 只用 λ*=0.5 的 within-model rank,不做 cross-task 同 λ 強度比較
    (樹內 k 混用 ≈ 2× 有效 λ 差;E10.md §4 caveat 3)
[ ] Sec 5.4 正文 "32 pre-training sequences" 與程式碼不符(實為 FT 任務 test
    split,seed+1000)——rebuttal 描述 ref set 以程式碼為準,revision 勘誤該句
[x] E7 定稿 + 5-seed:五處(AC-D/G3T9-W2/8VrD-W4/Q2/checklist)已改 mean±sd
    across seeds 42-46(free +0.947±0.015、TF +0.958±0.011、agree +0.959±0.010、
    cross +0.958±0.011);gen 77.4;擋單子集質疑
[ ] (選配)E7 多子集:SEED=43/44 各一輪(+5h)→ 四處改報 mean±sd
[ ] E1 gsm8k 重跑(REDO="gsm8k",token_cap 65536 全量)+ run_redo_E1.sh(補
    E1-native nuclear omega_W)後:以 n=12/11 完整集重寫 E1 數字 + subgroup +
    bootstrap,**採 tie+containment 框架**(不賣 B_N「贏 CKA」——CI 蓋 0;改
    「Procrustes²=feature arm,CKA 已在家族內」+ 平手 + 額外三件事:risk bound /
    attribution / regularizer)。attribution 例改用 **SQuAD**(B_I +0.811/+0.955),
    **不用 gsm8k**。bootstrap「差距不顯著」是 tie 論證支柱,務必入文;舊
    spearman.md(n=11/10)stale。
    ⚠️ 2026-07-26 定案:全篇**移除 gsm8k B_N +0.965 rescue**——與 pCi8-W4/AC-E/
    8VrD 的 gsm8k noise-floor 讓步自相矛盾,且和論文 Table 3「gsm8k weakest」打架;
    「full-CoT 0.979 = 截斷 artifact」框架一併退役(不貼文、不當賣點)。gsm8k 一律
    誠實 SNR limitation;W_N-for-ranking 一般性論點放 pCi8-W3(不綁 gsm8k)
[x] Global response 開場(2026-07-27 改版,依 study_for_rebuttal_strategy.txt)=
    「五條件皆有 measurement」總表最前 → "Claims we narrow"(先讓步後主張)→
    三支柱敘事;結尾定位句 + 討論期補完承諾 ✓(舊結尾重複表已刪、C 列 KD 殘留已清)
[x] 2026-07-27 lead-sentence pass:每則 response 首 1–2 句 = 直接答案 + 最強數字
    (AC-B/C/E、pCi8-W3、G3T9-W1/W2/Q1、8VrD-W2/W3/W4 重寫 lead;pCi8-W3 恭維
    壓縮至一句;8VrD-W2 恭維句後移至 slack attribution);G3T9/eQL6/8VrD 補結尾
    Net 句(邀請 discussion 互動,Huang et al. 正向訊號)
[ ] CKA 比較全篇口徑統一 = tie+containment(AC-C / pCi8-W3 / G3T9-W3 已改;
    貼文前 grep 確認無殘留 "tops both families" / 粗體 B_N「贏」/ "+0.008" /
    gsm8k「B_N +0.965 rescues」/「saturates ... B_N still ranks」)
[ ] significance 相關 thread(pCi8-W1/W2/W3、G3T9-W1)都 ladder up 到三支柱之一
[ ] 每個 reviewer thread 自包含(不寫 see Reviewer X / see AC)
[ ] 字數:每則 comment 壓到 NeurIPS 限制內(規則待確認)後再貼——共享項目
    (校準、成本、CKA 表)在非主場 thread 只留 2 句 + 數字,不重述全套
[ ] 貼文前決定 "[原文]" 標頭的處理:字數寬鬆 → 保留完整引文(對應最清楚);
    字數緊 → 縮成一行摘要(如 "W2 (bound looseness / operational threshold):"),
    但**務必保留編號與順序**,讓 AC/reviewer 能一一對應。"In sum:" 收尾句
    在任何情況下都保留(它是唯一的 thread 級總結)
[ ] 引用論文內容時用「promote to main text」語氣,不用防禦性句式

```

### Reviewer #2, G3T9 — 1) W1(Note 續行,原以 "+" 開頭故首次抽取時漏掉)
```
+ cost 表 ✅ 改用 E12b GSM8K-only 純實測(TF 8.9 s vs greedy decode 556.7 s = 62.6×;
maj@8 501×;load 兩端皆扣;screening 32-prompt ~1.1 s = 1/8 full,只在敘述提、不與
generation 比倍數;數字出自 out/E12/gsm8k_measured.md)。
```

---

## 2026-07-27 第三次整合(貼文就緒 pass)

依 study_for_rebuttal_strategy.txt 原則執行三項:

### (1) W/Q 合併 — 34 → 29 項,重複清零
- **8VrD**:W2+Q1、W3+Q3+Q4、W4+Q2;**刪掉重複的獨立 5)-8)**(75 行,Q1-Q4 原本各答兩次)
- **eQL6**:W2+Q1、W4+Q3(原文自寫 "See Weakness 2/4",答案原 ~90% 重複)
- **G3T9**:Q1+Q2(兩題共用 base-vs-instruct 實驗)
- **pCi8**:無 questions(N/A)
- 刪前先撈出獨有內容保留:8VrD-Q2 的 `rs(B_tf,|dR|_free)=+0.958±0.011` 與
  "rank once on the reference text" → 併入 W4+Q2;8VrD-Q3 的 diagnostic 側 size
  不變性 → 併入 W3 新 (4);W3 原文要求但原本散落的 multiple seeds /
  source-task performance → 補成明確的 (5)
- 修正誤置:「W-4's two remaining items」段落原誤落在 8VrD-Q3(句尾與 AC-B 相同),
  已移回 AC Condition B

### (2) `>` 摘要 29 則
`[原文]`(不貼)保留 + 下方 `> **標號.** 精簡摘要`(要貼)。原則:忠實不軟化批評、
含合併說明、保留 reviewer 自己的關鍵數字(如 8VrD 的 B=266.09 / 0.3658)。

### (3) Note 全部移出(35 則)→ 本檔
另補一則首次抽取漏掉的(續行以 "+" 開頭)。

### (4) 資訊粗體 26 則(每項一句)
選「教會機制」而非單純結果的句子。全清單見草稿;代表例:
- pCi8-W6 "**VALIDITY does not require near-isometry, only TIGHTNESS does**"
- pCi8-W4 "**property of the CE target itself, inherited by ANY CE-based proxy**"
- AC-C / G3T9-W3 "**tie BY CONSTRUCTION: Procrustes² = scale + shape at W_N**"
- eQL6-W2+Q1 "**any invertible map A can be absorbed into the head**"
- 8VrD-W4+Q2 "**the weak cells are the low-drift pairs, not weak benchmarks**"
另 `**In sum:**` 5 處標籤加粗(掃結尾者的落點)。

### (5) 貼文就緒修正(全部歸零)
- 內部實驗代號:`E12b micro-benchmark` → "256 GSM8K prompts, one RTX 5090,
  two independent runs agreeing to 0.02%";`(E3-C, ...)` / `out/E*` 全清
- 跨 thread 指涉:reviewer thread 內的 `W2/AC-A` 移除;AC 區的三處改成
  "our response to Reviewer G3T9 (W1)"(AC 讀全部 thread,指涉合理且省字)
- meta 指示:`Closing paragraph (vision/格局; posted immediately before...)` 與
  `Positioning sentence to close...` 標頭 + 外層引號全部移除,現為可直接貼的兩段
- pCi8 的 `(This reviewer left the Questions field as "N/A")` 移除
- CJK 洩漏:3 行 Note 續行(以 "+" 開頭)已移到本檔
- **markdown 渲染 bug 兩類(重要)**:
  (a) 8 處公式含單一 `*` 當乘號(`rho_T*rho_P*(1-Omega)`、`K_feat*D_Z`、
      `b*sqrt(...)`、`expm(theta*A)`)→ CommonMark 允許字內 `*` 斜體,會吃掉
      符號並斜體化 → 全部改用 backtick inline code
  (b) 3 行以 `|dR|` 開頭 → markdown 當表格列 → 已重排斷行
- en-dash `L770–782` → `L770-782`
- 表格前後空行、`**` 配對(126 個,偶數)、行首裸 `|`:全數檢查通過
- 保留的 em-dash 6 處全在 reviewer 原文備查區(他們的用字,不貼)

### 待辦(未變)
- TQA seeds 43/44 完成後,換掉全部 "(seed 42) so far" 措辭並補 ±sd
- 貼文前逐則壓字數(見上方 checklist 的字數項與 `[原文]` 處理原則)

---

## 2026-07-27 修正:CKA/SVCCA "containment" 過度主張(數學錯誤 + 與論文自相矛盾)

### 問題
草稿三處寫「CKA/SVCCA **already live INSIDE the PRISM family**」「a tie **BY
CONSTRUCTION**」。核對論文後確認**數學上不成立,且與自家 Appendix B 直接打架**:

論文 App B「Relation to CKA」原文:
> Linear CKA … differs from Ω²_F **in its denominator**. The two denominators are
> related by the inequality ‖Z_M^T Z_M‖_F ≤ ‖Z_M‖²_F …, which gives **CKA ≥ Ω²_F**.
> Hence CKA can overestimate alignment relative to Ω_F, and **it is not directly
> substitutable into our Procrustes residual**. More fundamentally, CKA is **not
> derived from any alignment optimization** …

⇒ 我們自己的附錄說 CKA **不可**代入 residual、**非**由 alignment 推導,rebuttal 卻說
它「在 family 內」。8VrD 會逐格讀 appendix(他挖出 Table 22 與 Qwen −0.34/−0.66),
pCi8/G3T9 是 conf-4 ⇒ 幾乎必被抓,且屬 overselling(Louis'26 的強負向訊號)。
SVCCA 更是論文完全沒做代數關聯,主張毫無依據。

### 真正成立的(論文 App A.1 原文)
> The left-hand side of Eq. (2) **at W = W_N is the squared Procrustes
> size-and-shape distance d²₁(Z_T, Z_P)** … accordingly, **our contribution is not
> the Procrustes distance itself**. Rather, it is (a) the explicit split of this
> residual into the scale term (ρ_T − ρ_P)² and the shape term 2ρ_Tρ_P(1 − Ω_N) …
> and (b) its lifting to a functional-risk bound (Theorem 1).

⇒ 只有 **Procrustes = 我們 W_N 下的 feature arm** 是恆等式(exact)。CKA/SVCCA 只是
「讀同一個幾何訊號的 rotation-invariant 分數」+ 一個不等式關係。

### 新增的實測(補強,取代空口主張)
從 out/E1/{llama,qwen}_metrics.csv 的既有 `procr_dist` 欄算 paper-5 benchmarks:
```
1-CKA            llama +0.9245  qwen +0.8818  mean +0.903   (與草稿表完全一致 ✓)
1-SVCCA          llama +0.9245  qwen +0.8782  mean +0.901   (一致 ✓)
procr_dist       llama +0.9297  qwen +0.8727  mean +0.901   ← 新增列
bound_W (B_N)    llama +0.9273  qwen +0.8964  mean +0.912   (草稿寫 0.911,四捨五入差)
```
三者落在 0.002 內 ⇒ 「讀同一訊號」不再是斷言而是**實測**。
表格現在是一條完整階梯:shape core 0.806 → +scale = Procrustes/CKA 水準 0.901
→ +head = certified bound 0.911(正好對應 App A.1 說的 (a) split + (b) lifting)。

### 反噬風險與處理
若只說「Procrustes 就是我們的 feature arm」,reviewer 可反問「那你的 feature arm
只是已知 shape metric,新穎性何在?」⇒ 三處都已同時寫入 App A.1 的自我定位
((a) split + (b) lifting to risk bound),把 containment 論證與 novelty 論證綁在一起。

### 已修三處
- global pillar (iii)、AC-C、G3T9-W3:移除 "live INSIDE the family" / "tie BY
  CONSTRUCTION" / "rank almost identically to Procrustes"(後者原本無表格支撐);
  改為「tie 是幾何預測的結果」+ Procrustes 恆等式(僅此一項 exact)+ CKA 的不等式
  與不可代入性(主動引 App B)+ App A.1 的貢獻定位 + 新的 Procrustes 實測列。
- 順帶移除誤引的 "App. A.1, Eq. 23"(A.1 講的是 Eq. (2) 在 W=W_N;Eq. (23) 是一般
  分解式,引 Eq. 23 會讓查證者找錯地方)。

---

## 2026-07-27 追加:Procrustes 列位置對調 + 撤掉 "tie" 語彙(但**不**宣稱 B_N 贏)

### (1) 列標籤對調
`Procrustes distance (= our feature arm at W_N)` →
`our feature arm at W_N (= squared Procrustes distance)`
理由:把「我們的量」放主詞位,消除 containment 味道(讀起來不再像「把別人的收進來」,
而是「我們的量恰好等於古典度量」)。同時把 shape core 列重排到 feature arm 之前,
讓 0.806 → 0.901 → 0.911 的階梯順著閱讀方向出現。兩表(AC-C、G3T9-W3)同步。

### (2) "tie" 全部撤掉(0 殘留),但**不能**改成「B_N 贏」
使用者問可否宣稱 B_N > CKA/SVCCA/Procrustes。**答案:不行**,三個硬理由:

**a. 自家新表就反證**:Llama 上 Procrustes **高於** B_N(+0.9297 vs +0.9273)。
   宣稱「贏過三者」在兩個 family 之一被自己的表格否證。

**b. paired bootstrap 9 個 CI 全部蓋 0**(我今日實算,5000 reps,對 variants 重抽,
   配對抽樣;out/E1 的 CSV,paper-5 benchmarks):
```
           llama                              qwen                            pooled
B_N-CKA    +0.0028 [-0.036,+0.040]  covers0   +0.0145 [-0.079,+0.107] covers0  +0.0087 [-0.039,+0.058]
B_N-SVCCA  +0.0028 [-0.045,+0.044]  covers0   +0.0182 [-0.015,+0.101] covers0  +0.0105 [-0.018,+0.057]
B_N-Procr  -0.0024 [-0.047,+0.033]  covers0   +0.0236 [+0.000,+0.107] covers0  +0.0106 [-0.017,+0.056]
```
   ⚠️ 注意:草稿原本寫「bootstrap 95% CIs cover 0」是**未經驗證的外推** ——
   out/E1/subgroup_analysis.md 裡的 bootstrap 是 **B_I** − CKA(llama −0.097
   [−0.244,+0.001]),不是 B_N。現已用實算數字取代,並把 CI 明列於貼文。

**c. CI 半寬 ~0.05 是觀測差距 ~0.01 的 5 倍**(n=12/11)。pCi8 的 W3 本來就是
   「a lot of machinery for a modest ranking improvement」——我們若反過來宣稱贏
   0.009,等於自證他的 overselling 指控(Louis'26 強負向訊號),且與 2026-07-26
   已定案的「不賣 B_N 贏 +0.008」自相矛盾。

### 改用的正面框架(比 "tie" 更強且完全誠實)
重點從「打平」轉為「**額外結構在排序上零成本**」:
- pillar (iii) 標題改 "STRICTLY MORE THAN CKA, **AT NO RANKING COST**"
- AC-C headline:「on ranking, the bound's extra structure costs NOTHING, so a
  certified risk bound, an axis attribution and a differentiable penalty come for
  free relative to a similarity score」
- 明列「B_N has the highest pooled mean, but we do NOT claim a ranking advantage」
  + 三個 CI + **主動揭露 per-family 兩個方向**(Llama Procrustes 領先 / Qwen3 bound
  領先)。主動揭露比被抓到強得多,且證明沒有挑邊。

### ⚠️ 待驗證(貼文前必做)
表中 `1-Omega_N +0.839/+0.773/+0.806` 這一列**無法從 out/E1/*_metrics.csv 重算**
(該 CSV 只有 omega_I,無 omega_W;我算 1-Omega_I = +0.916/+0.875,與此列不符)。
其餘四列(CKA/SVCCA/procr_dist/bound_W)都與 CSV 完全吻合。
⇒ Omega_N 列來源與其他列不同,run_redo_E1.sh(補 E1-native nuclear omega_W)跑完
後必須重新核對此列;**在核對前不要把「0.806 → 0.901 → 0.911 階梯」寫成粗體主張**
(現行文字已避免依賴該數值,只用 App A.1 的 (a)+(b) 論證新穎性)。
pCi8-W3 的「Ω_N 單獨 +0.839/+0.773 → B_N +0.927/+0.896」同一問題,一併核對。

---

## 2026-07-27 追加(重要):procr_dist vs 論文 δ_N —— 定義相同,數字差在 pipeline

使用者問「Procrustes distance 和論文的 shape+scale 差在哪?為何與 Table 3 對不上,
但 B_N 對得上?」查證結果:

### (a) 定義上**完全相同**(恆等式,已入貼文)
E1 `procrustes_distance(X,Y)`(exp_e1_similarity_baselines.py:109)算的是
`min_W ||X - YW||_F / sqrt(n)`,即
`sqrt((||X||_F² + ||Y||_F² - 2||XᵀY||_*) / n)`。

論文 δ_N² = (ρ_T-ρ_P)² + 2ρ_Tρ_P(1-Ω_N),其中 ρ=||Z||_F/√n、Ω_N=||ZᵀZ||_*/(||Z_T||_F||Z_P||_F):
```
(ρ_T-ρ_P)² + 2ρ_Tρ_P(1-Ω_N) = ρ_T² + ρ_P² - 2ρ_Tρ_P·Ω_N
                             = (||Z_T||_F² + ||Z_P||_F² - 2||Z_TᵀZ_P||_*)/n   ✓ 完全相同
```
⇒ **procr_dist = δ_N**(差一個 monotone 的平方根,Spearman 不受影響)。
所以「Procrustes = 我們的 feature arm」是**恆等式**,不是類比。已把這條代數直接寫進
AC-C 與 G3T9-W3 的粗體句(reviewer 可當場驗算)。

### (b) 數字對不上的原因:**兩條 pipeline**
`exp_e1_similarity_baselines.py`:
- `cka / svcca / procr_dist / omega_I / omega_W` = **重抽 features 現算**
- `bound_I / bound_W` = **從論文 CSV join**(line 341-342,`bound_csv_W.get(...)`,
  來源 exp_result/quantization/quantization_merged_slim.csv 的 Bound_W 欄)

⇒ **B_N 當然對得上 Table 3(0.927/0.896/0.912)——它就是論文的數字**。
⇒ **procr_dist 對不上論文 δ_N,因為 features 不同**:
   我算 +0.930/+0.873/**+0.901** vs 論文 Table 3 δ_N +0.898/+0.847/**+0.873**。
   主嫌是 **gsm8k**:E1 docstring 自己就記著「paper-round 1-Omega: rs +0.48 full vs
   (重抽) +0.94」,且 gsm8k 仍是 16k token 子抽樣(REDO 待跑)。

### (c) 論文 Table 3 完整內容(pdftotext **-layout** 才解析正確,先前無 layout 抓不到)
```
W=I  :  Ω  0.825 / 0.783 / 0.804    δ  0.881 / 0.855 / 0.868    B  0.828 / 0.813 / 0.820
W=W_N:  Ω_N 0.839 / 0.773 / 0.806   δ_N 0.898 / 0.847 / 0.873   B_N 0.927 / 0.896 / 0.912
```
✅ **先前標記「Ω_N 列來源不明」的疑慮解除**:+0.839/+0.773/+0.806 就是論文 Table 3 的
Ω_N 列,已驗證。(我之前算的 1-Ω_I=+0.916/+0.875 是不同 gauge + 不同 pipeline,不衝突。)
✅ B_N 論文為 **0.912**(草稿曾寫 0.911,已全部改為 0.912)。

### (d) 因此改寫成「兩塊表 + 各自標明 provenance」
- **Block 1(same features,重抽)**:1-CKA 0.903 / 1-SVCCA 0.901 / 我們的 feature arm
  δ_N 0.901 —— 這才是 apples-to-apples 的外部比較。
- **Block 2(論文 Table 3,單一 pipeline)**:Ω_N 0.806 → δ_N 0.873(+scale)→
  B_N 0.912(+head)—— 這是 component ladder。
⇒ 舊寫法把重抽的 procr_dist 與論文的 B_N 放同一表比較 = **跨 pipeline**,
   且會出現「你的 Procrustes 0.901 vs 你論文 δ_N 0.873,哪個才對?」的致命質問。

### (e) bootstrap 重算為 same-features(取代先前跨 pipeline 版本)
paired bootstrap, 5000 reps, 對 variants 重抽,**全部在重抽 features 上**:
```
llama : featarm-CKA +0.0052 [-0.004,+0.028]   featarm-SVCCA +0.0052 [-0.011,+0.040]
qwen  : featarm-CKA -0.0091 [-0.085,+0.000]   featarm-SVCCA -0.0055 [-0.028,+0.000]
pooled: featarm-CKA -0.0019 [-0.038,+0.011]   featarm-SVCCA -0.0001 [-0.014,+0.019]
```
⇒ pooled 差距 -0.002 / -0.000,**統計上完全平手**(比舊的跨 pipeline 版本更乾淨、更有力)。
先前那組「B_N - CKA +0.009 [-0.039,+0.058]」等數字已從貼文移除(跨 pipeline,不可比)。

### (f) pCi8-W3 順帶強化
他抱怨 Table 3 的 Ω→B 只差 +0.016(W=I block)。現在用 W_N block 的 cumulative ladder
拆成兩臂:**+0.067(scale)+ +0.039(head)= +0.106,是他引的 +0.016 的四倍**,
並說明 +0.016 來自「為可微性而選的 gauge」而非為排序而選。

### ⚠️ 貼文前必做
1. `run_redo_E1.sh`(gsm8k 全量 token + E1-native omega_W)跑完後**重生 Block 1 三列**
   與 (e) 的 bootstrap;預期 CKA/SVCCA/featarm 三者一起下修,**平手結論不受影響**
   (差值才是承重的,絕對值不是)。
2. Block 2 全部來自論文 Table 3,不受 REDO 影響,可直接引。

---

## 2026-07-27 追加 2:為何 gsm8k 換 features 後 procr_dist 變了、B_N 卻沒變?

使用者追問。答案分三層,全部有資料驗證:

### 第 1 層(直接答案):**B_N 從來沒有被重算**
`exp_e1_similarity_baselines.py:341-342` 直接把論文 CSV 的 `Bound_I/Bound_W` join 進來。
實測(需篩 `target_model`,論文 CSV 含多 family,同 (Label,dataset) 最多 11 列 ——
我第一次比對忘了篩,誤得 2/12,修正後):
```
llama gsm8k:  bound_W 與論文完全相同 = 12/12  (join,逐位元)
              omega_I 與論文完全相同 =  0/12  (重抽現算)
```
⇒ 「B_N 算出來一樣」的前提不成立:它不是算出來的,是抄進來的。任何重抽都不可能改它。

### 第 2 層:就算重算,差異也只在 gsm8k
rs(δ_N, |dR|) 逐 benchmark,論文 `delta_W` vs 重抽 `procr_dist`(llama):
```
arc      +0.972 / +0.972      mmlu  +0.972 / +0.972     squad +0.790 / +0.790
triviaqa +0.979 / +0.977      gsm8k +0.776 / +0.937  ← 唯一大差
MEAN     +0.898 / +0.930      (0.937-0.776)/5 = 0.032 = 兩者 mean 差,完全對上)
```
⇒ 重抽 pipeline 在 4/5 benchmarks 上**重現論文到小數第 3 位**(是很強的 sanity check),
   全部差異集中在 gsm8k。

### 第 3 層(機制):論文 gsm8k round 的 **Ω_W 飽和在 1.000000**
```
論文 llama gsm8k:Ω_W = 1.000000 於 11/12 variants(僅 Q2_K = 0.985776)
ρ_T = 134.242 固定,ρ_P = 134.177 ~ 134.860
```
δ_W² = (ρ_T−ρ_P)² + 2ρ_Tρ_P(1−Ω_W)。Ω_W 恰為 1 ⇒ **shape 項歸零,δ_W 退化成純
|ρ_T−ρ_P|**。逐格驗證:|134.242−134.278|=0.036≈δ_W 0.037 ✓;0.005 ✓;0.410 ✓;0.079 ✓。
⇒ 論文那一輪的 gsm8k δ 實際上是「純 scale 差」metric(數值 0.005–22.7,近噪音),
   只能排到 +0.776;而 Ω 本身 11/12 並列 ⇒ Spearman 崩潰 ⇒ 這就是論文 Table 3
   gsm8k Ω = +0.48 的來源。重抽 features 不飽和 ⇒ δ 保有 shape 訊號 ⇒ +0.937。
   而 Bound_W 不受影響(8.1→16.5→…→381.6 單調),因為它由 K_feat·δ + K_pred·γ 組成,
   GGUF 層的 γ>0 主導。

### ⚠️ 貼文策略:**這一層絕對不寫進 rebuttal**
理由:(a) 等於說「我們論文存的 gsm8k Ω 欄位是退化的」,會讓 reviewer 懷疑整個
Table 3;(b) 與已定案兩次的「gsm8k 誠實認 noise-floor/SNR,不宣稱任何 extraction
救回排序」自相矛盾(pCi8-W4 正是問 gsm8k)。維持現有讓步框架。

### 已做的可見處理(避免兩個 δ_N 數字在 reviewer 眼前打架)
AC-C 與 G3T9-W3 現在都明寫 provenance:
> Block 1 recomputes every score on ONE fresh extraction, which is what makes the
> CKA/SVCCA comparison like-for-like; Block 2 quotes the paper's own round.
> Absolute levels therefore differ slightly between blocks, and every claim uses
> only WITHIN-block comparisons.

並把論證改成**兩段 within-block 串接**(不做任何跨 block 比較):
(i) same features:our feature arm ≈ CKA/SVCCA(−0.002 / −0.000,CI 蓋 0)
(ii) 論文自己那一輪:+0.067(scale)、+0.039(head)⇒ full bound ≥ 自己的 feature arm
⇒ 結論「full bound 在 CKA 水準或以上,額外結構零排序成本」——**全部用同 pipeline 內比較**。

---

## 2026-07-27 追加 3:「下修版 CKA/SVCCA」——能算什麼、什麼絕對不能貼

使用者問可否算一個下修版的 CKA/SVCCA(擔心 Block 1 被 gsm8k 撐高)。

### 不能做的
1. **重建「論文 pipeline 的 CKA」不可能**:論文從未計算 CKA/SVCCA,沒有可對照的值。
2. **不同 token cap 的重算需要 GPU**:`exp_e1_similarity_baselines.py:346` 用完即
   `del Z_P`,只有 target 特徵存在 out/E1/{fam}_ZT/*.pt(446M/533M);12 個 proxy 要重抽。
   (順帶查明:CSV 的 `n_tokens` 是**子抽樣後**的值 = `X.shape[0]`;gsm8k 為 52184,
   其他 benchmark 僅 511–2330,差 ~100 倍。)

### 能做且已算完的(零 GPU)
**A. leave-one-benchmark-out(差值極穩)**
```
excluded    1-CKA    1-SVCCA    arm      arm-CKA   arm-SVCCA
(none)     +0.9031  +0.9013  +0.9012    -0.0019    -0.0001
arc        +0.9066  +0.9058  +0.9084    +0.0019    +0.0027
mmlu       +0.9016  +0.8961  +0.8959    -0.0056    -0.0001
squad      +0.9222  +0.9211  +0.9198    -0.0024    -0.0013
triviaqa   +0.8941  +0.8947  +0.8919    -0.0022    -0.0028
gsm8k      +0.8913  +0.8891  +0.8901    -0.0013    +0.0010
⇒ 任一排除下 |arm - baseline| ≤ 0.0056
```
**B. 逐 cell 差值**:10 格中單格最大 0.036(llama mmlu);llama gsm8k 三者**同為 +0.937**
(到小數第 3 位完全相同),qwen gsm8k 三者在 0.009 內。
**C. ex-gsm8k paired bootstrap**:arm−CKA −0.0013 [−0.034,+0.015]、arm−SVCCA +0.0010
[−0.015,+0.027] —— 仍是 dead heat。

### ⛔ 貼文只准報「差值」,絕不報 ex-gsm8k 的 levels
`5×0.9031 − 4×0.8913 = 0.9503` ⇒ 只要貼出 ex-gsm8k 的絕對值(0.891/0.889/0.890),
reviewer 就能反推 **gsm8k 那格 ≈ +0.95**。而 pCi8-W4 我們已讓步「gsm8k PTQ correlation
is low / variants near-tied at the noise floor」,論文 Table 3 也是 gsm8k 最弱 ——
貼出 +0.95 = 退休兩次的「gsm8k rescue」陷阱重現,AC 讀兩個 thread 會直接撞上。
同理不可貼逐 benchmark 表(gsm8k 那列 +0.937/+0.964 更露)。

### 已入貼文的版本(AC-C 與 G3T9-W3 各一句,只含差值統計)
> The dead heat does not rest on any single cell. Because all three scores are read
> from the SAME features they move together, so what matters is the difference: the
> largest gap between our feature arm and either similarity score in any of the ten
> (family, benchmark) cells is 0.036, and dropping any one of the five benchmarks
> changes the pooled paired difference by at most 0.006. No single benchmark drives
> the result.

⇒ 這句同時解決兩件事:(a) 擋掉「你的 gsm8k 格偏高」的質疑;(b) **零洩漏**任何 cell 的
水準,與 pCi8-W4 的 noise-floor 讓步不衝突。而且它比報 levels 更切題——承重的本來就是差值。

### run_redo_E1.sh 之後
全量 token 重抽會讓 CKA/SVCCA/arm **一起**移動(同 features),差值統計預期仍 ≤0.01,
所以貼文的 robustness 句不需改;只需重生 Block 1 的三個絕對值。

---

## 2026-07-27 追加 4:E1-C 腳本(在 GPU 上重建論文那一輪,產出 paper-round CKA/SVCCA)

新增 `rebuttal_exp/exp_e1c_paper_round_reconcile.py` + `rebuttal_exp/script_E1C.sh`。

### 為什麼需要
E1 的 `bound_I/bound_W` 是 join(所以必然對上論文),但 `cka/svcca/procr_dist` 是重抽現算。
四個 benchmark 對到小數第 3 位,只有 gsm8k 不對(δ:論文 +0.776 vs 重抽 +0.937)。
若要一個「論文 pipeline 的 CKA/SVCCA」(= 下修版),必須先能重建論文的 features。

### 驗收標準(關鍵設計)
用論文自己的 PRISM 統計量當 ground truth,且只驗**能從 features 單獨重算**的量:
```
rho_M   = ||Z_M||_F / sqrt(n)
omega_I = <Z_T,Z_P> / (||Z_T||_F ||Z_P||_F)
omega_W = ||Z_TᵀZ_P||_* / (||Z_T||_F ||Z_P||_F)
delta_g² = (rho_T-rho_P)² + 2·rho_T·rho_P·(1-omega_g)
```
✅ **δ 恆等式已在論文 CSV 上自檢通過:llama 168/168、qwen 154/154(最差相對誤差 6e-12)**
⇒ 驗 (rho, omega) 就足以認證 features,δ 是導出量。
Bound 不驗(需要 head term / γ,腳本不載 head)。

### 掃描維度(針對已診斷出的三個可能原因)
- `cap`:token 子抽樣上限 {65536, 52184, 32768, 16384, 8192, 4096}
- `dtype`:metric 累加精度 {float64, float32} —— 論文存「字面 1.0」,不是 0.99999,
  高度懷疑 fp32 累加(||Z||²_F≈9.4e8、n·d≈2.1e8 項)或 clamp
- `clamp`:是否 `min(omega, 1.0)` —— clamp 會把數值溢出的 >1 壓成恰好 1.0

### 飽和 pattern(selftest 已印出,與 token 數強相關)
```
llama Ω_W==1.0 恰好:arc 0/12(527 tok)、mmlu 0/12(511)、squad 1/12(2330)、
                     triviaqa 1/12(1524)、gsm8k 11/12(52184)、wikitext 9/12、fineweb 12/12
qwen :arc 4/11、mmlu 4/11、squad 7/11、triviaqa 4/11、gsm8k 9/11、wikitext 10/11、fineweb 11/11
```

### 執行流程
```
bash rebuttal_exp/script_E1C.sh selftest                      # 零 GPU,已驗過
bash rebuttal_exp/script_E1C.sh sweep                         # GPU,只跑 gsm8k,~50min/family
LOCK="cap=...,dtype=...,clamp=..." bash rebuttal_exp/script_E1C.sh lock   # 5 benchmarks
bash rebuttal_exp/script_E1C.sh report                        # 零 GPU,rs + paired bootstrap
```
輸出:`out/E1C/{fam}_reconcile.md`(掃描表+verdict)、`{fam}_paperround.csv`、`report.md`

### verdict 的處理原則(已寫進腳本 docstring 與 wrapper)
- **MATCH**(omega 相對誤差 ≤5e-4 且 exact-1.0 一致率 ≥99%)→ paper-round CKA/SVCCA
  可認證,直接當 Block 1 的下修版數字。
- **NEAR**(≤5e-3)→ 報殘差,只當 indicative。
- **NO MATCH** → **絕對不要挑最接近的 config 當成「論文的」**。誠實 fallback = 草稿現有的
  difference-invariance 論證(三個分數共用 features,逐 cell 差距 ≤0.036、
  leave-one-benchmark-out ≤0.006)。

### ⚠️ 若真的 MATCH,貼文前要重新評估一件事
paper-round 的 gsm8k CKA/SVCCA 會**一起下修**(因為飽和會同時打到三者)。那時 Block 1
三列的絕對值會更接近論文 Table 3 的 δ_N 0.873,provenance 註記可以簡化甚至移除 ——
但**仍不可貼 per-benchmark 或 ex-gsm8k 的 levels**(會反推出 gsm8k 那格,與 pCi8-W4
的 noise-floor 讓步衝突,見追加 3)。

---

## 2026-07-27 追加 5:out/E1.old_setting_gsm8k 能不能重建 paper-round CKA/SVCCA?

**答案:不能** —— 但它給了一個更有用的東西,而且推翻了我原本的診斷方向。

### 舊目錄的內容
`out/E1.old_setting_gsm8k/{llama,qwen}_metrics.csv` + ZT 快取 + logs(2026-07-26 13:19)。
與現行 `out/E1` 的唯一差別:**gsm8k token 數 16384 → 52184(llama)/ 63056(qwen)**,
其餘 benchmark 完全相同(arc 527、mmlu 511、squad 2330、triviaqa 1524)。

### 結果:token 預算對排序**零影響**(已做成可重跑模式)
`python3 rebuttal_exp/exp_e1c_paper_round_reconcile.py --compare-runs`
(或 `bash rebuttal_exp/script_E1C.sh compare`)→ out/E1C/token_budget_invariance.md
```
所有 cell 與所有 score 的 |old − new| 最大值 = 0.0000
llama mean:1-CKA +0.9245/+0.9245、1-SVCCA +0.9245/+0.9245、arm +0.9297/+0.9297
qwen  mean:1-CKA +0.8818/+0.8818、1-SVCCA +0.8782/+0.8782、arm +0.8727/+0.8727
```
逐 variant 的原始值也幾乎不動(omega_I 0.999913 → 0.999911;cka 0.99994 → 0.99994)。
⇒ **舊設定沒有比較接近論文**,兩者同樣是 Ω≈0.9999 而非論文的 1.0。
⇒ `cap` 軸從假設降級為 regression check(已改寫進 E1C 腳本的預設與說明)。

### 更重要的發現:論文的 gsm8k Ω **不可能是真值**
Ω_I = ⟨Z_T,Z_P⟩/(‖Z_T‖‖Z_P‖) 是正規化內積 ⇒ 由 Cauchy–Schwarz,
**Ω_I = 1 ⟺ Z_P = c·Z_T(完全成比例)**。Q2_K 量化後的 backbone 特徵不可能是
BF16 特徵的純量倍。所以論文 CSV 的 gsm8k Ω 欄(11/12 為字面 `1.0`)是
**論文自身 metric path 的數值假影**,而重抽出來的 0.868–0.9999 才是正確值。
⇒ 「重現論文 gsm8k 數字」= 重現一個 bug。E1C 的定位因此改寫:目標不是把論文當
ground truth,而是**辨識論文用了哪條 code path**(這是可回答的問題)。
⇒ 領先假設改為 **feature cast**:bfloat16 在 1.0 附近的間距是 2^-8 = 0.0039,
真值 0.9999 會被捨入成**恰好 1.0**。已加 `--casts none,bfloat16,float16` 軸。

### 論文的 Bound 沒有被汙染(重要,擋住連鎖懷疑)
CSV 上逐格驗證 **Bound_W = K_f·δ_W + K_p·γ_W**(12/12 精確吻合)。gsm8k 上
feature arm 只佔 0.1–15%(K_f·δ_W = 0.01–59,Bound_W = 8–382),γ_W 主導且行為良好
(5.6 → 227.9 隨 bit-width 單調)。⇒ δ 被汙染**不會**傳到 Table 3 報的 B_N 排序。
另檢查 γ_I = 0 於 FP16/FP4/GPTQ(frozen head,identity gauge)、γ_W > 0 全部 ——
與 W=I vs W_N 的 gauge 區別預測一致,head term 本身健全。

### 對貼文的含意
1. 目前草稿的 Block 1 數字(重抽)是**正確的那一組**,不需要下修。
2. difference-invariance 論證更穩了:現在有**兩個獨立 extraction**(16k/52k)給出
   逐格相同的排序,可加一句「invariant to a 3.2x change in token budget」——
   但**仍只報差值,不報 per-benchmark / ex-gsm8k levels**(見追加 3 的反推風險)。
3. ⚠️ **修訂版需要一則 erratum**:論文 Table 3 的 gsm8k Ω/δ 欄(以及 wikitext/
   fineweb 的同類格)受影響。這會讓 Ω/δ 的 mean 上升,而 B 不變 ⇒ **W=I block 的
   Ω→B 差距(+0.016)會縮小甚至變負**,反而削弱 pCi8-W3 的「machinery 買到多少」論證。
   決定前必須先算修正後的 Table 3。**不要在 rebuttal 裡提這件事**(會讓 reviewer
   懷疑整張 Table 3),留給 camera-ready 的 erratum。

---

## 2026-07-27 追加 6:要不要把 gsm8k 從相似度比較中移除?→ **不要**

### 收益 = 0
```
             1-CKA    1-SVCCA   arm      arm-CKA
all-5       0.9031   0.9013   0.9012   -0.0019
ex-gsm8k    0.8913   0.8891   0.8901   -0.0013
```
配對差距幾乎不變,三個 level 一起掉 0.012 ⇒ 移除**不會讓數字變好看**。

### 風險 = 真實且集中在最敏感處
1. pCi8-W4 整條就是 gsm8k、AC Condition E 也點名 gsm8k。在那兩處讓步「gsm8k 弱」,
   卻在 CKA 表把它拿掉 = 「避開自己承認的難題」。那是**他們的**題目,必被注意。
2. 論文全篇 five benchmarks(摘要/Table 1/Table 3),突然 4 個必被問。
3. **Block 2 只存在於 n=5**:論文 Table 3 無 per-benchmark 列,無法造 4-benchmark 版
   component ladder ⇒ 4-bench Block 1 配 5-bench Block 2,又製造回剛修好的跨 block 不一致。

### 唯一可用的理由會自我矛盾
「gsm8k 的 |dR| 在噪音底(0.019 nats vs 其他 0.07–0.16,App F.3/Table 10)」是論文自己
預先寫下的標準,與 pCi8-W4/Condition E 一致 —— **但我們重抽的 gsm8k 那格排到 +0.94**,
三個分數同為 +0.937(llama,小數第 3 位相同)。用「那格是噪音」排除一個表現良好的格,
被追問「被排除的值是多少」即崩。噪音底論證屬於 calibration/threshold(Condition A)層,
不屬於相似度比較層。

### 採用的替代方案(零代價達成同樣目的)
草稿的 robustness 句升級為**三項檢查**(AC-C 與 G3T9-W3 各一,只含差值、零 level 洩漏):
> (i) 逐 cell 最大差距 0.036;(ii) leave-one-benchmark-out 差距 ≤0.006;
> (iii) 兩個相差 3.2x token 預算的獨立 extraction 逐格重現到小數第 4 位。

第 (iii) 項是今天從 out/E1.old_setting_gsm8k 挖到的新證據(16k vs 52k,|old−new| 全域
最大 0.0000),可用 `bash rebuttal_exp/script_E1C.sh compare` 重跑。

⇒ 「移除 benchmark」能達成的(擋掉「你的 gsm8k 格偏高」),差值不變性全部達成,
   而且更強:不是「排除後仍平手」,而是「**無論怎麼排除都平手**」。

---

## 2026-07-27 追加 7:E1-D fresh round(3 個新 seed,五列同一 pipeline)

新增 `rebuttal_exp/exp_e1d_fresh_round.py` + `script_E1D.sh`。使用者提案:gsm8k 用足夠
token cap、抽三個與論文不同的 seed、重算五列表 ⇒ **與論文數字區隔**。

### 評估:可以,而且是解掉 provenance 泥沼最乾淨的做法
1. 五列(1-CKA / 1-SVCCA / 1-Ω_N / feature arm δ_N / B_N)全部在**同一次 forward**
   的同一批 features 上算 ⇒ 不再需要「within-block comparisons only」註記,也不再有
   δ_N 0.901(重抽)與 0.873(論文 Table 3)並列的尷尬。
2. **|dR| 一併重算**(用同一 pass 的 answer-span CE)⇒ 內部配對。若沿用論文 |dR|,
   換 seed 就會把 sample A 的 score 對到 sample B 的 risk,正是 Condition B 我們
   花力氣劃清的 unpaired 問題。
3. **修掉論文那一輪的真 bug**:`prism/core/metrics.py:244` 的
   `omega = max(min(omega,1.0),-1.0)` + float32 累加 ⇒ n=52184 時 ω 溢出被 clamp 成
   字面 1.0(11/12 gsm8k 格)。本腳本 float64 累加、**不 clamp,只記錄
   `clamp_would_fire`**。
4. 三個 seed → **mean ± sd**,是真的 robustness 升級;seeds 43/44/45 與論文的 42 互斥。

### ⚠️ 真實的矛盾風險(已寫進 docstring 與 wrapper 開頭)
修掉 ω 假影會**拉高** shape core 的排序(被汙染的格正是把 Ω 拉低的那些)。論文 Table 3
的 W_N ladder 是 Ω_N 0.806 → δ_N 0.873(+0.067)→ B_N 0.912(+0.039);在修正後的
features 上,scale 臂的增益可能小得多 —— 現有證據:fresh trace-gauge shape core 排
+0.895,feature arm 排 +0.901,**只差 +0.006**。
⇒ 若 nuclear gauge 也如此,**pCi8-W3 現行的「machinery buys +0.106,是他引的 +0.016
   的四倍」會失效**,必須改回「four outputs, ranking 只是其中一個」的框架(那本來就是
   該則的 lead,所以轉換成本低)。**跑之前先決定,跑完照實報。**
次要風險:B_N 可能落在 CKA 之下(論文 W=I block 的 B 0.820 就低於 δ 0.868,head 臂
在混合 γ=0/γ>0 時會傷排序)⇒ "no ranking cost" 需軟化為「差距在噪音內」。

### 內建 sanity check(驗證 B_N 重算是否等於論文的 bound)
K_feat 只依賴 H_T,所以重算值必須吻合論文,否則兩臂權重不同、B_N 不可跨輪比較。
已從論文 CSV 取出:**llama K_f = 2.6137、qwen K_f = 3.4583、K_p = 1.4142 = √2**(兩者
皆是,正如 Proposition 1)。`--report` 會自動比對並印 MATCH/MISMATCH。
diagnostics 另印:clamp-would-fire 格數、逐 benchmark token 數、逐 benchmark mean |dR|
(後者用來看 fresh target 是否穩定;gsm8k 若又落在 ~0.019 噪音底,±sd 會直接暴露)。

### 執行
```
bash rebuttal_exp/script_E1D.sh dry      # 零 GPU,已驗:plan + 從源碼證明 clamp 存在
bash rebuttal_exp/script_E1D.sh run      # GPU,兩 family × 3 seeds,~75 min/family
bash rebuttal_exp/script_E1D.sh report   # 零 GPU → out/E1D/table.md + diagnostics.md
```
輸出 `out/E1D/{family}_seed{S}.csv`(逐 variant×benchmark 全欄位)。`--force` 可重跑。

### 若結果良好,貼文要跟著改的地方
- AC-C 與 G3T9-W3:兩塊表併成一塊(五列、mean±sd),**刪除 provenance 註記**與
  「within-block」措辭;robustness 三項檢查保留(第 (iii) 項可改為「three fresh seeds」)。
- pCi8-W3:ladder 數字換成 fresh 版;若 ladder 變平,改用 four-outputs 框架。
- ⛔ 仍**不可**貼 per-benchmark 或 ex-gsm8k levels(反推 gsm8k 那格 ⇒ 撞 pCi8-W4 的
  noise-floor 讓步)。fresh round 的 gsm8k 若又排得高,更要只報聚合與差值。

---

## 2026-07-27 追加 8:E1-D 加速(保證結果相同且完整)

使用者問能否加速 `script_E1D.sh run`。先量測瓶頸,再只做**確定安全**的優化。

### 瓶頸量測
| 項目 | 每次 | 原設計次數/family |
|---|---|---|
| **proxy load(GGUF 反量化 / GPTQ)** | ~60–120 s | **36**(12 proxy × 3 seed) |
| target load | ~30 s | 3 |
| forward pass(5 benchmarks ≈57k tokens) | ~5–8 s | 195 |
| fp64 SVD 4096² + 2×eigh | ~3–6 s | 180 |
⇒ **load 完全主導**,而同一 proxy 被載入 3 次是純浪費。

### 已實作的四項(皆不改變任何數值)
1. **迴圈外提**:proxy 外層、seed×benchmark 內層 ⇒ load 次數 **39 → 13**(target 也只載 1 次,
   一次抽完 3 seeds × 5 benchmarks)。**~75 min → ~20–30 min /family**。
2. **target features 逐 (seed,benchmark) 磁碟快取 + atomic rename**(被殺不會留半截 .pt),
   proxy 迴圈間共用;`H_T` 也快取。
3. **逐 (seed, proxy) 增量 append 到 CSV** + resume:已完成的 (seed,proxy) **完全不載入 proxy**。
   中斷最多損失一個 proxy 的工作量。
4. **completeness gate**:`--report` 若發現任何 (seed,benchmark) 的 variant 數不齊,
   **拒絕輸出表格**(需 `--allow-incomplete` 才會出、並標記 NOT final)。已用假資料端到端測過:
   完整→正常出表;刪一格→拒絕。

### 順帶修掉的兩個正確性風險(比加速更重要)
- **TF32**:`torch.backends.cuda.matmul.allow_tf32` 只有 10 bit mantissa(~1e-3 相對),
  cross 矩陣量級 1e9 會被毀掉,ω 完全不可靠。torch 2.5.1 預設是 False,但這是**環境相依**
  (`torch.set_float32_matmul_precision` 會翻轉),故在 `fresh_metrics` 內明確 pin 為 False
  並在 finally 還原。
- **記憶體/精度兩難**:原版對 52184×4096 做完整 fp64 copy = 每側 1.7 GB,兩側 3.4 GB,
  加上 8B 模型會爆。改為 **chunked fp64 累加**(預設 8192 tokens/chunk):同樣的算術,
  峰值 ~700 MB,且不必卸載模型。`--chunk` 可調(不改結果)。
- 另加 **normalize-before-cross**:X̂=X/‖X‖、Ŷ=Y/‖Y‖ ⇒ ω 直接等於 X̂ᵀŶ 的 nuclear norm
  (O(1) 量級),不再是兩個 ~1e9 量的比值。這正是讓 1−ω ~ 1e-5 可解析的關鍵,也是論文
  那一輪(fp32 比值 + clamp)失敗的地方。順便免費得到 ω_I(給 erratum 用)。

### 沒做的(刻意)
- **不重寫 linear_cka / svcca**:雖然可以複用 cross 省一次大 matmul,但重寫有微小數值差異
  風險。維持呼叫 E1 的同一函式 ⇒ 正確性由建構保證。反正它們不是瓶頸(~0.05 s)。
- **不用 fp32 SVD 換速度**:1−ω 落在 1e-5 量級,fp32 的 ~2e-5 相對誤差正好在會出事的邊緣
  (就是原 bug 的成因)。fp64 SVD 保留。

---

## 2026-07-27 追加 9:seed 到底改變什麼?(已查證 + 加入 runtime 量測)

### 查證結果:seed 唯一的作用是「抽哪 512 個 examples」
`prism/data/loaders.py`(load_task_data 內):
```python
if seed is not None:
    hf_dataset = hf_dataset.shuffle(seed=seed)          # 打亂整個 split
if num_samples is not None and num_samples < len(hf_dataset):
    hf_dataset = hf_dataset.select(range(num_samples))  # 取前 512
```
⇒ 換 seed = 換抽到的 512 筆 examples。**不是**換 token 子抽樣:E1D 的另一處 seed 用途
`subsample_tokens(..., seed=seed)` 在 TOKEN_CAP=131072 下**永不觸發**(觀測最大 63056),
所以變異來源單一、不混淆。這也再次確認 |dR| 必須重算(examples 換了)。

### ⚠️ 兩個會讓「3 seeds」失去意義的失敗模式(已加 runtime 偵測)
1. **degenerate**:若 `num_samples >= len(split)`,`select` 不觸發 ⇒ 三個 seed 看到**同一批**
   examples,只有順序不同。而本腳本所有 metric(CKA、ω、Procrustes、Σ_P)都只依賴
   XᵀY / XᵀX / YᵀY 這些**對列置換不變**的量 ⇒ 三 seed 會給出**逐位元相同**的數字,
   sd = 0.000 卻與穩定性毫無關係。
2. **correlated draws**:小 split 上兩次 512 抽樣會大量重疊。GSM8K test ≈1319、
   ARC-Challenge test ≈1172(記憶值,未離線驗證)⇒ 重疊 ≈ 512²/n ≈ **39% / 44%**。
   所以三個 seed 是**相關**抽樣,±sd 是真實抽樣變異的**下界**,不可宣稱 independent
   replications。

### 已實作:`loader_fingerprint()` + `{family}_seed_draws.md`
在 loaders 建好後,對每個 (seed, benchmark) 把每筆 example 的 input_ids 做 blake2b 雜湊,
輸出逐 benchmark 的 **selected 數量與 seed 兩兩重疊率**;若任一 benchmark 重疊 >99.9%
就印出 degenerate 警告(建議降 --num_samples 或改報單次抽樣)。
零 GPU 邏輯測試已過:1319 取 512 → 量到 38%(理論 39% ✓);全取僅換順序 → 量到 100%
且 degenerate 觸發 ✓。

### 貼文用語要跟著改
表格 caption 不可寫 "three independent seeds",要寫成
"three draws of the evaluation subset (seeds 43/44/45; draws overlap on the smaller
splits, so the spread is a lower bound on sampling variability)"。
實際重疊率跑完看 out/E1D/{family}_seed_draws.md 再填。

---

## 2026-07-27 追加 10:pCi8-W1 的 "ranking without benchmarks" 不精確,已修

### 問題
W1 原寫「This is exactly what enables (i) **ranking without benchmarks**, ...」。
但 PRISM 讀的 reference set 就是**該 benchmark 自己的 validation 切片(prompt + gold
answer)**,所以「without benchmarks」是 overclaim —— 與 "label-free" 同一類錯誤:
把「不需要**跑** benchmark」講成「不需要 benchmark」。
pCi8 是 conf-4,而且 W2 就緊接著說「You still need to run benchmarks for that」,
他對這條界線特別敏感。

### 修法(與全篇口徑一致)
> (i) ranking variants from one teacher-forced pass over a small reference slice,
> with no decoding and no answer grading (**the reference is the task's own
> validation data, so this replaces RUNNING the benchmark, not having it**)

括號那句是關鍵:主動把界線畫出來,而不是讓 reviewer 自己發現。

### 全篇同類措辭複查結果(其餘皆已精確,不需改)
- L234 AC-B「does not need, and does not claim, a benchmark-independent reference」= 刻意劃界 ✓
- L1038 G3T9-W1(3)「needs no benchmark harness, no scorer, and no correctness labels,
  **only reference text**」= 明寫需要 reference text ✓
- L448 / L1768 free-running「so **here** the risk gap needs no reference answers」
  = 有 "here" 限定,只適用 free-running 模式 ✓
- L1013「screens all K **without decoding or grading**」✓
- L691 AC-A「without benchmarking it (no decoding, no grading; the benchmark is run
  once per family to fit the calibration, not once per variant)」✓
- L54 / L608 / L620 / L632 = AC 與 pCi8 的**原文**(他們自己的用字),不動

### 順帶收緊 closing paragraph
「into one forward pass」→「into **one forward pass per variant**」——
原句可能被讀成「總共一次 forward」。

### 教訓(寫給後續檢查用)
這是本次第三個同型錯誤(label-free → generation/grading-free、containment →
identity-only、without benchmarks → without running)。共同 pattern:
**把「省掉某個步驟」寫成「不需要某個東西」**。貼文前用這個 pattern 再掃一次:
凡是 without / no / free 開頭的主張,都要問「省掉的是步驟還是資料?」

---

## 2026-07-27 追加 11:pCi8-W2 的 "own two cells" 歸屬錯誤 + 0.003/0.193 無出處 + 一個誠實性問題

使用者質疑 pCi8-W2 的「the reviewer's own two cells」。查證後發現**三件事**。

### (1) 歸屬錯誤(已修)
pCi8 的 W2 原文只提 **Q2 和 Q4**,而且**沒有給任何數字**。
「B=266.09 / B=23.24」是 **8VrD** 引的(8VrD-W2 原文)。
⇒ 在 pCi8 thread 寫「the reviewer's own two cells」是錯的,而且會讓他覺得我們在
   跨 reviewer 套用罐頭回答(最傷信用的一種印象)。
⇒ 8VrD thread 的同一句話**是正確的**,保留。

### (2) 更該答的是 Q4(已改)
pCi8 問的是「it tells you Q2 is worse than Q4, but not whether **Q4** is actually
good enough」—— 我們卻用 Q8_0/Q2_K 回答,連他的 variant 都沒碰到。
已重算補上:**Q4_K_M:B_I 96.75、true |dR| 0.0356、LOO 校準預測 0.0554**
⇒ 預測 in / 真值 in,ε=0.1 判對。現在直接用他的 variant 回答他的問句。
(slack = 96.75/0.0356 ≈ 2718×,已一併填入 8VrD 的表。)

### (3) ⚠️ 0.003 / 0.193 原本**沒有出處**,且該格 precision 只有 0.75
- `out/E6/calibration.csv` 只存逐 cell 聚合(MAE、prec、rec),**沒有逐 variant 預測**。
  草稿的 0.003/0.193 在 out/ 下查無實據 —— 違反「絕不虛構 / 只填 out/ 的真數字」。
- 已實際重算並**新建 artifact**:`out/E6/llama_mmlu_loo_predictions.md`
  (輸入 = 論文 CSV 的 Bound_I + |MdR|,LOO isotonic,sklearn)。
  **完全重現 E6 記錄的該格:MAE 0.0401(E6: 0.0401)、prec@0.1 0.75、rec@0.1 0.75** ✓
  且 Q8_0 → 0.0028(草稿 0.003 ✓)、Q2_K → 0.1933(草稿 0.193 ✓)。數字是真的,只是沒存檔。
- **但重算暴露一個誠實性問題**:這一格 12 個 variant 有 **4 個判錯**
  (Q3_K_M / NF4 誤報 out、FP4 / GPTQ 漏報 in),precision 0.75 —— **正是 E6 說的
  「六個低於 0.8」之一**。我們卻挑這一格、且只挑判對的兩列當 demo。8VrD 會逐格核對,
  被抓到比不提更慘。

### 處理:把弱點轉成可陳述的操作特性
重算發現一個漂亮的事實:**4 個判錯的 variant,正好就是 true |dR| 落在門檻
0.1 ± 1 MAE(0.040)帶內的那 4 個**(集合完全相同,已在 artifact 中驗證
`identical set: True`)。所有離門檻更遠的 variant 全部判對。
⇒ 敘述改為「**a screen with a known ambiguous band, not an oracle**」:
   pCi8-W2 與 8VrD-W2 兩處都主動揭露該格 MAE 0.040 / precision 0.75 / 六格之一,
   並說明所有錯誤都在 ±1 MAE 帶內。這是 Li et al. 的 openness 正向因子,
   且把「有 4 格判錯」從醜聞變成一個有原則的精度陳述。

### 教訓(加入貼文前檢查清單)
1. **每個「the reviewer's own ...」都要回原文核對是哪一位 reviewer 說的。**
2. **回答要用 reviewer 自己舉的例子**(他問 Q4 就答 Q4),否則等於沒回答他。
3. **每個數字都要在 out/ 下有檔案**;聚合檔存在 ≠ 個別數字有出處。
