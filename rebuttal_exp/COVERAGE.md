# 已有內容盤點:每條 Weakness/Question 對照論文(含 appendix)的覆蓋狀況

目的:區分「論文已交代但 reviewer 沒讀到/沒 get 到」vs「真的缺、需要新材料」。
行號 = 投稿 PDF 的 LaTeX 行號;判定分三級:

- ✅ **已交代**——rebuttal 指路 + 承諾提升可見度即可,新實驗只是錦上添花
- 🟡 **部分交代**——有現成基礎,用既有段落開場、以新數據補完
- ❌ **未交代**——老實承認 + 用 E1–E10 的新材料回應

**語氣鐵則**:指出「已在論文中」時絕不用 "as stated in the paper" 的防禦句式。統一句型:
*"We agree this deserves more prominence. The analysis appears in Appendix X (quoting: …); we will promote it to the main text and additionally [新證據]."*

---

## 0. 總覽

| # | 條目 | 判定 | 論文位置 | 需搭配 |
|---|---|---|---|---|
| pCi8-W1 | 三結論已知,只是統一框架 | 🟡 | L22–25, L230–232 | 措辭強化(溫度計論證) |
| pCi8-W2 | bound 太鬆、答不了「夠不夠好」 | 🟡 | Abstract, L201–204, L299–302, Fig 2 | E6(已有數字) |
| pCi8-W3 | Ω-only vs 完整 PRISM 差距小 | ✅ | **L278–285, L249–257**, App B | E1 加 CKA/SVCCA 欄 |
| pCi8-W4 | GSM8K 相關性低是結構性限制 | ✅ | **App F.3 L742–751, Table 10** | 位置提升;E9 選配 |
| pCi8-W5 | 引 EWC 卻沒比 | 🟡 | L89–94, **L265–267** | E2 |
| pCi8-W6 | 激進量化下 near-isometry 還成立? | ✅ | **L112–115, App C.1–C.2** | E8 選配 |
| G3T9-W1 | 何不直接跑 benchmark;成本;actionability | 🟡 | L94–99, **L258–262**, L249–257 | 成本表(由 log 估) |
| G3T9-W2 | teacher-forcing;benchmark 輸入;ref ablation | 🟡 | L196–200, L201–204, **L302–305, App D L681–682** | E3(A 已有數字)+E7 選配 |
| G3T9-W3 | 要直接比 CKA/SVCCA;比 simpler regularizers | 🟡 | **App B「Relation to CKA」**, Table 3 Ω row | E1 + E2(feature-KD) |
| G3T9-W4 | bound 絕對尺度太鬆 | 🟡 | 同 pCi8-W2 | E6 |
| G3T9-Q1 | full FT / RLHF? | ✅ | **App C.3 L678–679, App I** | E11 選配 |
| G3T9-Q2 | multi-task / base→SFT? | 🟡 | 無;但**已有 5-FT-task 後續數據**(E10 表) | E10 表 + scope |
| eQL6-W1 | 符號未定義;結果在附錄 | ❌ | —(Intro L57–59 確實未定義 ρ,Ω) | 認錯+修訂清單 |
| eQL6-W2/Q1 | 為何限 orthogonal? | 🟡 | L77–80, L112–115, App B L613–615 | 補 gauge 論證(純文字) |
| eQL6-W3 | ρTρP prefactor 主導 shape 項? | ✅ | **App A.1**, Table 1 自身 | **新數字已算出**(見 §2.3) |
| eQL6-W4/Q3 | regularizer 犧牲原任務? | ✅ | **Fig 4 / Fig 8 第一欄「Fine-tune Dataset」** | E2 補數值表 |
| eQL6-Q2 | Sec 5.3 那句話看不懂 | ❌ | — | 認錯+改寫 |
| eQL6-Q4 | rs 是跟什麼算的? | ✅ | **Table 1 caption, Fig 2 caption, L199–200** | 5.5 補一句 |
| 8VrD-W1 | empirical vs population risk | ❌ | —(正當) | **E5(已完成,tex 可貼)** |
| 8VrD-W2/Q1 | 量化 relaxation;operational threshold | 🟡 | Fig 2 safe zone, L299–302 | E6(已有數字) |
| 8VrD-W3(1) | 無單軸控制介入 | ❌ | — | E4 |
| 8VrD-W3(2) | Table 22 負面 cell | ✅✅ | **App H 全章 = 作者版 E10**(見 §2.4) | E10 擴充到 10 cells |
| 8VrD-W3(3) | baselines 不夠強;seeds;source-task;32-seq | 🟡 | L265–267(比較設計);Fig 4 col 1 | E2 + E3-C |
| 8VrD-W4(1) | 0.831 只算 Llama;Qwen 負相關要進主文 | ✅ | **L225–227(明示 scope), Fig 7(負值公開標注), G.1 L770–782(專段診斷)** | 主文全 cell 矩陣承諾 |
| 8VrD-W4(2)/Q2 | teacher-forced corollary;要 free-run | ✅ | **L302–305, App D L681–682** | E7 選配 |
| 8VrD-Q3 | regularizer 對 ref set 敏感度 | ❌ | L205–207 只寫了 protocol | E3-C |
| 8VrD-Q4 | 更強 forgetting baselines | ❌ | — | E2 |
| 8VrD-Lim | 應加列 4 項 limitation | 🟡 | 2/4 已在 Discussion;head-varying 在 C.3 | 補 Limitations 段 |

統計:✅ 9 條、🟡 12 條、❌ 6 條。**近三分之一的火力可以用「指路」解決,不花任何 GPU。**

---

## 1. 使用者五個猜想的核對結果

| 猜想 | 核對 | 說明 |
|---|---|---|
| Regularizer 對原任務其實有 cover | **成立** | Fig 4(Llama)與 Fig 8(Qwen)的**第一欄就叫「Fine-tune Dataset」**——三種配置在 FT 資料集上的 ΔR 軌跡(負值=進步)全程並列,肉眼可見 trace 未犧牲 target task。eQL6 沒看懂第一欄的意義。缺的只是數值表(E2 aggregator 產出) |
| 負面結果其實已誠實呈現 | **成立(大部分)** | −0.34/−0.66 就標在 Fig 7 的子圖上;G.1 L770–782 有專段解釋(noise floor:BBQ→TriviaQA 的 \|ΔR\|≈0.0035,比同軌跡 ARC 的 0.288 低兩個數量級);Table 22 的 +8.6%/+2.7% 本來就是**論文自己報的**。真正成立的批評只剩「主文 0.831 沒把 Qwen cells 納入聚合」→ 承諾主文放全 cell 矩陣 |
| replay 是「血緣對照」而非方便 | **成立** | L265–267:*"To isolate shape preservation from data re-fitting, we compare against a replay-CE baseline that uses the same D_ref"*——同 32 條、同排程、只換 loss 泛函,是控制變因設計;L89–94 更把 EWC(weight-space)/replay(data-space)/shape(feature-space)列為三族。回 pCi8-W5「convenient choice」時先引這兩段,再給 E2 數字 |
| bound 鬆已主動說明、重點是 ranking | **成立** | Abstract 就寫 *"calibrated for variant ranking"*;L201–204(ranking 對 subset 穩健)、L299–302(*"Tight absolute estimation … we leave to future work"*)、Fig 2 的 y=x 虛線+safe zone。三位 reviewer 其實都承認論文有講——他們要的是「量化+操作化」→ E6 已交付 |
| teacher forcing 是特色非缺陷 | **成立** | L196–200(deterministic、單趟 forward、無 decoding 變因);L302–305 明示是 efficiency 選擇且 *"the bound applies unchanged to any (Z_T, Z_P), including features collected along free-running generation trajectories"*;**App D L681–682 更強:"applies to any matrix pair … whatever the origin of the rows"**——理論本來就與特徵來源無關,teacher-forcing 只是評測協議 |

---

## 2. 四個最高價值的「reviewer 沒發現」素材

### 2.1 App H 就是作者版的 E10(對 8VrD-W3(2),分數槓桿最大)

8VrD 引用 Table 22 的 +8.6%/+2.7% 當負面證據,但 **Table 22 的標題就是 "Diagnostic-gating prediction vs. empirical trace effect",且有一整欄 "Gating verdict"**:

> Llama TruthfulQA 1−Ω̄=0.0937 → −19.2% → *shape-driven ⇒ apply*
> Llama BBQ 0.0678 → +8.6% → *cell-level mixed(下段解釋)*
> Qwen TruthfulQA 0.0091 → +2.7% → *at noise floor ⇒ skip*
> Qwen BBQ 0.0011 → −0.2% → *at noise floor ⇒ skip*

配套段落全都在:gating 訊號定義(L814–819:條件 (i) Ω 大幅漂移、(ii) forgetting 為 shape 主導,**both signals available *before* regularization is applied**);Llama-BBQ 的 cell-level 拆解(L803–812:TriviaQA −88%、GSM8K −79% 的 shape-driven wins vs ARC/MMLU 的 shape-drift-不轉化-\|ΔR\| 例外);機制不變(L799–802:trace 把 Ω 0.93→0.98,replay 不動);Qwen 段結論(L855–857:*"the Qwen3-8B numbers … do not contradict the Llama TruthfulQA-FT result; they confirm the regime-dependence the bound itself predicts"*)。

**Rebuttal 打法**:不是辯解,是指出「axis-specificity 分析已經在 Appendix H,我們同意它該進主文」,再用 E10 把證據從 4 個 setting 擴到 **10 個 (model, FT-task) cell**(加了 social_iqa/no_robots/lima)並給 rank 統計(within-model rs:qwen +1.00、llama +0.40)。⚠️ 注意:本機結果樹與論文 Table 18/20 是**不同輪次**(llama-TQA no-reg mean:論文 0.843 vs 本機 1.483;qwen 0.263 vs 0.781)——E10 數字定位為「新一輪獨立重跑的延伸證據」,不要與 Table 22 數字混排(詳見 E10.md §4)。

### 2.2 App F.3 的 GSM8K 噪音下限診斷(對 pCi8-W4)

reviewer 說「論文歸因於 long CoT 稀釋 per-token loss」——但沒看到**量化版**:L742–751 + Table 10 caption:GSM8K 的 mean |ΔR| 只有 **0.019 nats(其他 benchmark 0.07–0.16 的 1/10;Qwen3-8B-Instruct 甚至 0.0033)**,*"at this scale, per-variant differences are dominated by measurement noise rather than the bound's predictive signal"*。這把「結構性限制」重新定性為 **SNR 問題:不是 PRISM 在 reasoning 上失效,是所有 variant 在 GSM8K 的 CE 幾乎沒差、無序可排**——任何以 CE 為標的的 proxy(含 perplexity screening)在此皆然。G.1 L770–782 用同一機制解釋 forgetting 端的負相關 cell,兩處互證。仍要做的:把它從附錄升為 first-class limitation(pCi8 的合理要求)+ E9 answer-span 變體(選配)。

### 2.3 eQL6-W3 可以當場釘死(新數字,零 GPU,今天已算)

App A.1 已寫明 shape 項是 Procrustes size-and-shape distance 的精確拆分(= 對齊後殘差能量,ρTρP 是量綱)。用論文自己的 CSV(77 cells)補上量化(`exp_eql6w3_prefactor.py`,輸出在 `out/eql6w3_prefactor.md`):

- **CV(ρTρP) 跨 variant:中位數 0.47%**(p90 7.9%)——prefactor 在比較中近乎常數
- **(1−Ω) 同 cell 動態範圍:中位數 291×**——排序變異全部來自幾何部分
- **Spearman(shape term, 1−Ω):中位數 1.0000**(min 0.988)
- rs(shape,|ΔR|) 與 rs(1−Ω,|ΔR|) **完全相同**(中位數皆 0.745)

直接填 Google Doc eQL6-W3 草稿的三個 [RESULT NEEDED]。reviewer 的極限例子數學上對,但其前提(norm 相對爆炸)在同 base 的 post-training variants 之間不發生——而且真發生時就是 scale 軸要抓的 pathology。

### 2.4 「bound 對任意 W 成立」三處明文(對 pCi8-W6)

- L112–115:*"the PRISM bound (Theorem 1) holds for any such W, with alignment quality determining how tight the bound is in practice"* ——validity 與 tightness 的區分**主文就有**
- App C.1:對每個 W ∈ O(d) 給出 bound 家族(Eq. 42)
- App C.2:*"As quantization becomes more aggressive … Ω to fall below Ω_N, and this gap reflects genuine asymmetric distortion that a rotation-invariant metric would mask"*——激進量化不是假設破產,而是訊號本身

пCi8-W6 的回應可以三句話收工 + E8(isometry violation vs bit-width 曲線)當甜點。

---

## 3. 其餘條目的位置與用法(補充細節)

### pCi8-W3(Ω-only vs 完整 PRISM)
- **L278–285(Sec 5.5)已自答**:γ>0 只在 GGUF k-quant 引擎、Ω-only 在 backbone 主導 cell 本來就夠;*"B remains our default—it is Theorem 1's certified upper bound and the only valid metric whenever γ>0."*
- **L249–257(Sec 5.3)是 Ω-blind 的實例**:Qwen3-Base Q6_K SQuAD:Ω≈1、δ=1.18,γ=75.77 佔滿 bound;INT8 同模型 γ≡0、B=3.81,**20× 差距完全由「哪個協議量化 lm_head」決定**——Ω-only 對此無感(Google Doc 裡的 [VERIFY] 可判 CONFIRMED,引 Table 12)。
- App B L608–615 給了 CKA≥Ω_F² 的理論關係與「CKA 非對齊推導」的原則性批評 → E1 補數值即完整。

### G3T9-W1(actionability / cost)
- **L258–262 已有 axis→action 對照**:scale→per-channel outlier smoothing [17]、head→FP16-lm_head retention、shape→Hessian-aware reconstruction(PTQ 時)/ trace regularization(訓練時)——G3T9 說「不 actionable」時,這段就是反例,GGUF head 案例(L249–257)是 end-to-end 實證。
- 缺的只有 GPU-hours 成本表:可用既有 log 估(PRISM:每 variant 單趟 forward,512×5 benchmarks ≈ 1–2 min + 載入;benchmark suite:GSM8K 等需 autoregressive decoding,512 題 × 數百 token)。**不需新 GPU 實驗**,rebuttal 直接放表。

### G3T9-Q1 / Q2(full FT / 多任務)
- Q1:App C.3 L678–679 明文 *"comparing models with both rotated features and divergent heads (e.g., full-parameter SFT). We leave this setting to future work"* + App I 第一條 → 已劃界,回應引用 + 說明分解形式上仍適用(同架構同維度)。
- Q2:論文沒有;但 **repo 已有投稿後補跑的 3 個新 FT 任務(social_iqa / no_robots / lima)× 2 models**——E10 的 10-cell 表就是現成的「更多元 FT 任務」證據,可直接回 Q2 的前半;mixed-data 一次 run(E11)選配。

### eQL6-Q4(rs 對什麼算)
- Table 1 caption:*"reports Spearman's rs(B, |ΔR|) across all quantization variants"*;Fig 2/3 caption 同;L199–200 定義 |ΔR|。回應一句話指路 + 承諾在 5.5 正文與所有 caption 重申。

### 8VrD-W4(1)(0.831 的範圍)
- L225–227 原文就是 *"mean Spearman rs = 0.831 ± 0.0722 over the 2×5 downstream cells"*(=Llama grid)+ *"The Qwen3-8B replication (Appendix G) reproduces both patterns in most cells"*——scope 有寫、hedge 有寫、負值有畫(Fig 7)。成立的部分只有「主文聚合統計應含全部 cells」→ 承諾 + 給全矩陣 mean/median。

### 8VrD-Limitations 清單
- 已在 Discussion:teacher-forcing(L302–305)、ranking-vs-absolute(L299–302)。
- 部分在附錄:head-varying variants(App C.3 L678–679)。
- 待補:reference-set 敏感度(E3 結果一併寫入)、feature-extraction cost(與 G3T9-W1 成本表共用)。

---

## 4. 真正的缺口(誠實區,新材料清單)

| 缺口 | 回應材料 | 狀態 |
|---|---|---|
| Theorem 的 empirical/population 精確性(8VrD-W1) | E5:Theorem 1′ + McDiarmid corollary | ✅ tex 已備 |
| 單軸控制介入(8VrD-W3(1)) | E4 | 腳本就緒,~1.5 h |
| CKA/SVCCA **數值**比較(G3T9-W3) | E1 | 腳本就緒,~3–5 h |
| EWC/L2-SP/feature-KD + seeds + target-task 表(pCi8-W5 等) | E2 | 腳本就緒,~20 h |
| ref set size/domain 對 regularizer(8VrD-Q3) | E3-C | 腳本就緒,~2.5 h |
| 符號定義/5.3 句子/notation table(eQL6-W1/Q2) | 修訂承諾(零實驗) | 直接寫 |
| GPU-hours 成本表(G3T9-W1) | 由既有 log 估算成表 | 直接寫 |
| gauge 論證(eQL6-W2:為何不是任意可逆 A) | 純文字(head 可吸收任意 A ⇒ 殘差失義;O(d) 保內積幾何) | 直接寫 |

---

## 5. 對 rebuttal 草稿(Google Doc)的修訂建議

1. **pCi8-W3、pCi8-W4、pCi8-W6、eQL6-W3、eQL6-W4、eQL6-Q4、8VrD-W3(2)、8VrD-W4(1)** 八條:把草稿開頭改為「指路 + 引原文 + 承諾提升可見度」,新實驗數據降為第二段——先證明內容存在,再展示加碼。
2. **8VrD-W3(2) 直接引 Table 22 的 gating verdict 欄與 L855–857**,E10 作為「擴充到 10 cells 的獨立重跑」;絕不把本機 E10 數字與論文 Table 18–21 數字並排(不同輪次)。
3. eQL6-W3 草稿的 [XX] 以 §2.3 四個數字填入;pCi8-W3 的 [VERIFY] 判 CONFIRMED(Table 12 SQuAD 案例)。
4. Global response 的「誠實性修訂」項可以加一句:負面 cell 與 gating 分析**已在投稿版 Appendix G/H**,修訂版將提升至主文——這會顯著改變「作者在藏負面結果」的觀感。
