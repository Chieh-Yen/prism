# Reviewer/AC 性格側寫與回覆策略(8674;2026-07-27)
依據 rebuttal_exp/study_for_rebuttal_strategy.txt(Gao'19 / Huang'23 / Louis'26 / Li'25):
加分訊號 = 補 baselines、non-defensive、明確修 clarity、補理論、evidence-backed、完整不漏。
失敗模式 = 指向 section 不作答、模糊 future work、防衛語氣、低努力、無法說明 novelty。
核心結論:成功 ≈ 避免失敗模式,而非話術說服。discussion 期回覆時沿用本檔。

## AC LN7U(preliminary Borderline Reject)— 最高優先
- 性格:程序化決策者。自訂五條件合約(A–E),明說要 "concrete evidence rather than
  only clarification",且主動提示 "careful narrowing of the claims" 可換 borderline accept。
- 攻略:履約框架。開場第一屏 = 五條件 × 五 measurement 總表;緊接 claims-narrowing
  (先讓步後主張,雙面論證順序);結尾承諾討論期補完(BBQ/seeds)。
- 危險區:賣故事在證據前面;任何一條件只有「說明」沒有「數字」。
- 給 AC 的隱形產品:可直接抄進 meta-review 更新的句子(headline 欄)。

## pCi8(3, conf 4;S=2)— 直率實務派
- 性格:自信、口語("how much you broke it and where");Questions 留 N/A = 已下定論,
  只吃硬證據。conf 4 會抓 overselling。
- 攻略:一致性原則——他的 Strengths 已承認 machinery 是 real contribution,用他自己的
  話反打 S=2;W2 直接回答他的原句("is Q4 good enough");誠實負結果(GSM8K span 失敗)
  會加信任分。恭維最多一句,連引四句原話會被識破。
- 危險區:任何 gsm8k「救回」語句(與 noise-floor 讓步自相矛盾);吹 B_N 贏 CKA。

## G3T9(3, conf 4;S=2)— 清單型審稿人
- 性格:條列四連問(forward passes / reference data / cost / realistic),要 checklist
  式完整回答。禮貌學術腔。
- 攻略:每個子問題可見地逐項回答 + 表格;lead 直接給 62.6×;Q2 用 recipe-agnostic +
  base-vs-instruct 當 data point(不揭露 E10)。
- 危險區:漏答任何一個子問題(他會逐項對);模糊 scope 切割。

## eQL6(3, conf 3;Q=2/C=2 但 S=3/O=3)— 最可能翻分
- 性格:數學型、語氣試探("to me it seems");喜歡 idea、扣分在寫作與品質。
- 攻略:(a) 明確修 clarity(notation 表、結果進正文)= Louis 正向訊號;
  (b) W3 先確認「你的代數是對的」再用論文自己的數據釘死(CV 0.63% vs 366× range);
  (c) 把 clarity 責任攬在自己身上("our presentation to fix")。
- 危險區:讓他覺得被糾正;跳過他的任何一個 (See Weakness N) 交叉引用。

## 8VrD(4, conf 3)— 唯一正分,potential champion
- 性格:逐格核對 appendix(自己挖出 Table 22 +8.6%、Qwen −0.34/−0.66);要具體補充
  (seeds、source-task、32-seq sensitivity、free-running)。
- 攻略:每問一個 measurement、絕不 oversell;用他自己的兩個 cell 做 calibration 示範;
  負 cell「解釋而非隱藏」;把他武裝成 discussion 裡的辯護者。
- 危險區:誇大任何他能親手驗算的數字。

## 全篇規則(已執行於 2026-07-27 lead-sentence pass)
1. 每則 response 首 1–2 句 = 直接答案 + 最強數字(不用 "See ..." 開頭)。
2. 結構:answer → evidence → 具體修訂承諾(將加入哪個 Table/Section)。
3. 不用 em-dash;table cell 內不放 |dR|(用 downstream / \lvert\rvert)。
4. 各 thread 自包含;結尾 Net 句邀請 discussion 互動(Huang'23 正向訊號)。
5. SLoRA/CLAIM/ArMA = scope 論證(sequential ≠ single-task)+ 引用,非模糊 future work。
6. (2026-07-27) 不賣 benchmark-independent:bound 是 input-conditioned by construction,
   unpaired 使用 = distribution-transfer,超出推導 → Condition B 用「明確 narrowing +
   size ablation 實測」回答;wikitext +0.506 撤下(僅 regularizer 側 0.773 保留)。
7. (2026-07-27) E2 只跑 TQA:貼文不承諾任何 BBQ 新實驗;論文既有 BBQ 數據照引。
8. (2026-07-27) 貼文**不用 "label-free"**,也**不寫「我們改用別的詞」**。論文 0 次提到
   label-free,且 Sec 5.1 早就寫明 "over the gold span" ⇒ 無 overclaim 可收回;寫
   retraction = 憑空認罪 + 可能收回 pCi8 credit 的 strength。改為純正面描述協議
   (reference = prompt + gold answer,features/CE 取 answer region),並直引 Sec 5.1。
9. (2026-07-27) 「no grading」要**具名四項**:no output parsing(GSM8K "#### <number>"/
   選項字母/span)、no answer matching+normalization、no per-benchmark scorer、
   no accuracy;根因 = 量是兩模型在相同 token 的 CE gap,correctness 不在定義裡。
10. (2026-07-27) 兩性質**分層**,不可混講:no grading = 量的性質 ⇒ **所有模式成立,含
   free-running**;no decoding = 協議層 ⇒ **僅 default TF**,free-running 刻意 decode。
   引用 free-running 當 fallback 必須說它付 decode 成本(answer-availability fallback,
   非 cheap path)。誠實 carve-out:PRISM 會定位 reference 的 answer region(dataset 端
   offset,非對模型輸出的判斷)。
