# PRISM NeurIPS 2026 (submission 8674) — Rebuttal 實驗包

對應 rebuttal 策略(updated 版,含 AC meta-review 分析)的實驗 **E1–E7, E10**。
現況:pCi8 = 3、G3T9 = 3、eQL6 = 3、8VrD = 4;**AC LN7U preliminary = Borderline Reject**,
但明列五項翻盤條件 A–E——本包每個實驗直接對應其中一項:
**A**→E5+E6、**B**→E3、**C**→E1+E2(含 AC 點名的 layer-freezing)、**D**→E7(或劃界)、**E**→E10+全 cell 誠實呈現。
草稿填寫狀態見 `rebuttal/rebuttal_draft.txt`;已交代內容的指路清單見 [COVERAGE.md](COVERAGE.md)。

## 總結論:全部可行,GPU 總預算 **~32–38 h(約 2 天)**

零 GPU 項目(E5、E6、E10、E3-A、eQL6-W3 prefactor)**已在本機用既有數據跑完**,真實數字寫入各報告。

| 實驗 | AC 條件 / 回應對象 | GPU 成本 | 狀態 | 主要取捨 |
|---|---|---|---|---|
| **E6** slack + calibration | **A**;pCi8-W2, G3T9-W4, 8VrD-W2/Q1 | **0** | ✅ 已跑(論文 5 benchmarks,570 rows):median slack 1597×、K-step 只佔 0.13–0.4 dex、LOO MAE 0.055 nats、prec@0.1≥0.8 於 49/55 | 無 |
| **E10** Table-22 軸對齊 | **E**;8VrD-W3(2) | **0** | ✅ 已跑:**SUPPORTED**(within-model rs +1.00/+0.40) | 與 App H/Table 22 定位區分(E10.md §4) |
| **E5** 定理重述 + corollary | **A**;8VrD-W1 | **0** | ✅ 完成:`e5_theorem.tex` | 無 |
| **E4** 單軸介入 | 堵 "questionable diagnostic utility";8VrD-W3(1) | ~1–1.5 h | 待跑 | 單 model × {mmlu, wikitext} |
| **E1** CKA/SVCCA/Procrustes | **C**(診斷);G3T9-W3, pCi8-W3 | ~3–5 h | 待跑 | 主文 2 families;⚠️ Gemini「有 cached activations 免 forward」是錯的,需重抽特徵 |
| **E3** reference set ablation | **B**;G3T9-W2, 8VrD-Q3 | ~4 h | Part A ✅(same-bench .725 vs wikitext .506);B/C 待跑 | B 只跑 Llama;C 3 sizes × 2 domains |
| **E7** free-run 子集(meta-review 後新增) | **D**;G3T9-W2, 8VrD-W4/Q2 | ~2–2.5 h | 待跑(腳本就緒) | 1 family × 1 benchmark × 100 prompts;備案=劃界(AC 明文接受) |
| **E2** layer-freeze/EWC/L2-SP/feature-KD | **C**(正則化);pCi8-W5, 8VrD-W3/Q4, eQL6-W4 | **~24 h**(+7 h 選配 Qwen) | 待跑(已加 **layer_freeze**) | 見 E2.md §2(含對 Gemini OOM 顧慮的修正:不適用本實作) |

## 建議排程(依 AC 五條件的證據優先序)

```
Day 0(已完成)   E5 / E6 / E10 / E3-A / eQL6-W3:零 GPU 數字鎖定
                 → AC-A、AC-E 核心數字已在手,global response 可先掛
Day 1 上午       E4(~1.5 h)→ 堵 "questionable diagnostic utility"
Day 1 下午       E7(~2.5 h)→ AC-D 實驗側;傍晚起 E1(~5 h,過夜)
Day 1 深夜–Day 2 E2 STAGE=sweep(~10 h;layer_freeze/EWC 先跑)
Day 2–3          E2 STAGE=seeds(~14 h,過夜)
Day 3 上午       E3 PARTS="B C"(~4 h)
Day 3 下午       E2 aggregate + 全部 [TBD] 填入 rebuttal/rebuttal_draft.txt
Day 3–4(緩衝)   重跑 / 選配 E2 STAGE=qwen(~7 h)/ 寫作交叉檢查
```

## Quickstart(GPU 機器,repo root)

```bash
bash rebuttal_exp/script_E4.sh                     # 先跑:decisive 又便宜
bash rebuttal_exp/script_E7.sh                     # AC-D 實驗側
bash rebuttal_exp/script_E1.sh
STAGE=sweep bash rebuttal_exp/script_E2.sh         # 之後 STAGE=seeds / aggregate
PARTS="B C" bash rebuttal_exp/script_E3.sh
# 零 GPU(任何機器,重現已完成項目):
bash rebuttal_exp/script_E6.sh
bash rebuttal_exp/script_E10.sh
PARTS="A" bash rebuttal_exp/script_E3.sh
python3 rebuttal_exp/exp_eql6w3_prefactor.py
```

## 檔案地圖

```
COVERAGE.md               ⭐ 逐條盤點 27 個 W/Q 何者論文已交代(✅9 / 🟡12 / ❌6)+ 指路寫法
E{1,2,3,4,5,6,7,10}.md    各實驗完整報告(目的/可行性/設計/取捨/風險/rebuttal 對應)
script_E{1,2,3,4,6,7,10}.sh 執行入口(E5 無 script:純理論,交付 e5_theorem.tex)
exp_e1_similarity_baselines.py   E1 主程式(GPU)
train_forgetting_baselines.py    E2/E3-C 主程式(GPU;6 種 method 單一入口,含 layer_freeze)
exp_e2_aggregate.py              E2 彙整(零 GPU;剔除未達 step-300 的中斷 run;支援 top{K} 目錄)
exp_e3_refset_ablation.py        E3 A(零 GPU)+ B(GPU;BnB/GPTQ head 依論文慣例 γ≡0)
exp_e4_interventions.py          E4 主程式(GPU;附帶輸出 E5 需要的 sup-residual)
exp_e7_freerun.py                E7 主程式(GPU;proxy 自生軌跡、target 常駐 CPU 交換)
exp_e6_slack_calibration.py      E6(零 GPU,stdlib-only)
exp_e10_axis_alignment.py        E10(零 GPU,stdlib-only;合併三個結果樹)
exp_eql6w3_prefactor.py          eQL6-W3 prefactor 檢查(零 GPU;CV 0.47% vs 1-Ω 291×)
common_quant.py                  E1/E3/E7 共用(variant 解析自論文 CSV、proxy 載入)
e5_theorem.tex                   E5 交付物(Theorem 1' + Corollary,paper-ready)
out/                             所有輸出(md/csv;E6、E10、E3-A、eql6w3 已有結果)
```

## 全域執行注意(策略文件 §8 + meta-review 補充,務必遵守)

1. **不可虛構**:草稿所有 [TBD] 只能填 `out/` 下真跑出的數字;來不及的點用各報告 fallback 句。
2. 每個 reviewer thread 自包含,不寫 "see Reviewer X";**global response 以 AC 五項 A–E 為骨架**。
3. E10 的 axis-specificity 論述已通過數據核對(SUPPORTED);引用時與論文 Table 22 數字分開呈現(不同輪次,E10.md §4)。
4. 8VrD 優先投入:E5(theorem)+ E4(interventions)是 4→5 的最短路徑。
5. **時程(meta-review 07-24 剛修訂,AC 在看 thread)**:global response 越早掛越好;可標 "results by [date]",但掛出的每個數字必須已定案。
6. 定位語言統一為:*"a calibrated, label-free ranking-and-attribution instrument with an explicitly empirical-risk guarantee"*。
