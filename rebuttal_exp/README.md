# PRISM NeurIPS 2026 (submission 8674) — Rebuttal 實驗包

對應 rebuttal 策略(updated 版,含 AC meta-review 分析)的實驗 **E1–E7, E9–E13**(E8 已併入 E1)。
現況:pCi8 = 3、G3T9 = 3、eQL6 = 3、8VrD = 4;**AC LN7U preliminary = Borderline Reject**,
但明列五項翻盤條件 A–E——本包每個實驗直接對應其中一項:
**A**→E5+E6、**B**→E3、**C**→E1+E2(含 AC 點名的 layer-freezing)、**D**→E7(或劃界)、**E**→E10+E13+全 cell 誠實呈現;另 G3T9-W1 成本表→E12、pCi8-W6 isometry→E8(併於 E1)。
草稿填寫狀態見 `rebuttal/rebuttal_draft.txt`;已交代內容的指路清單見 [COVERAGE.md](COVERAGE.md)。

## 總結論:全部可行,GPU 總預算 **~32–38 h(約 2 天)**;選配 E9+E11 另 +~2.5 h

零 GPU 項目(E5、E6、E10、E3-A、eQL6-W3 prefactor)**已在本機用既有數據跑完**,真實數字寫入各報告。

| 實驗 | AC 條件 / 回應對象 | GPU 成本 | 狀態 | 主要取捨 |
|---|---|---|---|---|
| **E6** slack + calibration | **A**;pCi8-W2, G3T9-W4, 8VrD-W2/Q1 | **0** | ✅ 已跑(論文 5 benchmarks,570 rows):median slack 1597×、K-step 只佔 0.13–0.4 dex、LOO MAE 0.055 nats、prec@0.1≥0.8 於 49/55 | 無 |
| **E10** Table-22 軸對齊 | **E**;8VrD-W3(2) | **0** | ✅ 已跑:**SUPPORTED**(within-model rs +1.00/+0.40) | 與 App H/Table 22 定位區分(E10.md §4) |
| **E13** 20-cell 全相關矩陣 | **E**;8VrD-W4(1) | **0** | ✅ 已跑:**REPRODUCED**(llama mean +0.830 vs 論文 0.831;qwen-bbq −0.34/−0.66 命中)→ 全 20 cells mean **+0.71** / median **+0.93** | protocol 由 checksum 定位(answer-only、≤300、無 step-0) |
| **E5** 定理重述 + corollary | **A**;8VrD-W1 | **0** | ✅ 完成:`e5_theorem.tex` | 無 |
| **E4** 單軸介入 | 堵 "questionable diagnostic utility";8VrD-W3(1) | ~1–1.5 h | 待跑 | 單 model × {mmlu, triviaqa}(只用論文 benchmark;triviaqa = E6 中 |ΔR| 範圍最寬) |
| **E1** CKA/SVCCA/Procrustes | **C**(診斷);G3T9-W3, pCi8-W3 | ~3–5 h | ✅ GPTQ 全補齊(**12/12、11/11**,fallback 生效)。**兩-gauge 發現:B_W(旋轉不變、全量)mean +0.927/+0.896 ≥ CKA 兩家;B_I 在 extraction-QA 領先**;⏳ 尚欠 gsm8k REDO(16k 子抽樣 artifact,E1.md §1c)→ 之後 draft 一次定稿 | [load] 計時餵 E12;bootstrap:B−CKA 差距全不顯著 |
| **E8** isometry lite(選配) | pCi8-W6 | ~20 min | 待跑(首版 iso_dev 因 n<d 病態作廢,新 metric 金測通過;重用 E1 快取) | `script_E8.sh`;FP16 列會標 noise-floor |
| **E3** reference set ablation | **B**;G3T9-W2, 8VrD-Q3 | ~10 h(首輪實測修正) | Part A ✅(.725/.506);**B/C 首輪失敗(OOM / lr 2e-4 + 覆寫),腳本已修待重跑**(E3.md §7) | B 只跑 Llama;C 3 sizes × 2 domains、λ=0.5 |
| **E7** free-run 子集(meta-review 後新增) | **D**;G3T9-W2, 8VrD-W4/Q2 | ~2.5 h | ✅ **定稿(12/12,gsm8k)**:rs_tf +0.958 / rs_free **+0.944** / agreement +0.972 / cross +0.951,gen 77.4 tok,CI 全 >+0.75;draft 四處已填 | 選配 SEED=43/44 報 mean±sd;mmlu 舊輪退化已退役 |
| **E2** layer-freeze/EWC/L2-SP/feature-KD | **C**(正則化);pCi8-W5, 8VrD-W3/Q4, eQL6-W4 | **~24 h**(+7 h 選配 Qwen) | 待跑;⚠️ **lr 必須 1e-5**(paper-round 啟動值;此前 2e-4 sweep 作廢,E2.md §4 勘誤;trace anchor 改 λ0.5) | 見 E2.md §2(含對 Gemini OOM 顧慮的修正:不適用本實作) |
| **E12** GPU-cost 表 | G3T9-W1、8VrD-Lim(iv) | **0**(收割 E1/E7 計時 + CPU 計數) | 待 E1/E7 跑完後執行 | decode 端為刻意下界(greedy、單 pass、還需標籤) |
| **E9** GSM8K final-answer span(選配) | pCi8-W4 mitigation | ~1 h | 待跑(腳本+單元測試就緒) | 只 llama × gsm8k;prompt_length 改寫至 `####` 標記,pipeline 其餘不動;兩種結果都有敘事 |
| **E11** base→instruct data point(選配) | G3T9-Q1、8VrD-Lim(iii) | ~1.5 h(2 pairs) | 待跑(腳本就緒) | W=I、γ>0(instruct head);含 bound_holds 驗證 + Q4/Q2 PTQ anchor;單 pair ~40 min 也可交卷 |

## 建議排程(依 AC 五條件的證據優先序)

```
Day 0(已完成)   E5 / E6 / E10 / E13 / E3-A / eQL6-W3:零 GPU 數字鎖定
                 → AC-A、AC-E 核心數字已在手,global response 可先掛
Day 1 上午       E4(~1.5 h)→ 堵 "questionable diagnostic utility"
Day 1 下午       E7(~2.5 h)→ AC-D 實驗側;傍晚起 E1(~5 h,過夜)
Day 1 深夜–Day 2 E2 STAGE=sweep(~10 h;layer_freeze/EWC 先跑)
Day 2–3          E2 STAGE=seeds(~14 h,過夜)
Day 3 上午       E3 PARTS="B C"(~4 h)
Day 3 下午       E2 aggregate + 全部 [TBD] 填入 rebuttal/rebuttal_draft.txt
Day 3–4(緩衝)   重跑 / 選配 E2 STAGE=qwen(~7 h)/ 選配 E9(~1 h)+ E11(~1.5 h)
                 / 寫作交叉檢查
```

## Quickstart(GPU 機器,repo root)

```bash
STAGE=backfill bash rebuttal_exp/script_E2.sh      # ⭐ 最先(~3 h):補 λ1.0 斷尾
                                                   #   兼環境 canary(E2.md §5)
bash rebuttal_exp/script_E4.sh                     # decisive 又便宜
bash rebuttal_exp/script_E7.sh                     # AC-D 實驗側
bash rebuttal_exp/script_E1.sh
STAGE=sweep bash rebuttal_exp/script_E2.sh         # 之後 STAGE=seeds / aggregate
PARTS="B C" bash rebuttal_exp/script_E3.sh
bash rebuttal_exp/script_E9.sh                     # 選配 (~1 h):pCi8-W4 mitigation
bash rebuttal_exp/script_E11.sh                    # 選配 (~1.5 h):G3T9-Q1 data point
# 零 GPU(任何機器,重現已完成項目):
bash rebuttal_exp/script_E6.sh
bash rebuttal_exp/script_E10.sh
bash rebuttal_exp/script_E13.sh
PARTS="A" bash rebuttal_exp/script_E3.sh
python3 rebuttal_exp/exp_eql6w3_prefactor.py
# E1+E7 跑完後(成本表,CPU 分鐘級):
python3 rebuttal_exp/exp_e12_cost_table.py
```

## 檔案地圖

```
AUDIT_2026-07-25.md       ⭐ 全面審查報告:29 條 answerability、E12/E13 缺口補齊、未跑實驗 setting 判定
COVERAGE.md               ⭐ 逐條盤點 27 個 W/Q 何者論文已交代(✅9 / 🟡12 / ❌6)+ 指路寫法
E{1,2,3,4,5,6,7,9,10,11,12,13}.md 各實驗完整報告(目的/可行性/設計/取捨/風險/rebuttal 對應)
script_E{1,2,3,4,6,7,9,10,11,13}.sh 執行入口(E5 無 script:純理論,交付 e5_theorem.tex;
                          E12 直接 python 執行,依賴 E1/E7 的 log)
exp_e1_similarity_baselines.py   E1 主程式(GPU)
exp_e1_subgroups.py              E1 子群分析(零 GPU;γ 混池機制驗證,B vs CKA within-protocol)
exp_e1_report.py                 E1.exp.md 產生器(零 GPU;兩 gauge 全表/機制表/bootstrap)
exp_e8_isometry.py               E8 lite(GPU ~20 min;isometry-restriction cost,重用 E1 快取)
train_forgetting_baselines.py    E2/E3-C 主程式(GPU;6 種 method 單一入口,含 layer_freeze)
exp_e2_aggregate.py              E2 彙整(零 GPU;剔除未達 step-300 的中斷 run;支援 top{K} 目錄)
exp_e3_refset_ablation.py        E3 A(零 GPU)+ B(GPU;BnB/GPTQ head 依論文慣例 γ≡0;OOM 已修)
exp_e3_partc_from_log.py         E3-C 首輪 log 回收(零 GPU;lr-stress 內參,不得引用)
exp_e4_interventions.py          E4 主程式(GPU;附帶輸出 E5 需要的 sup-residual)
exp_e7_freerun.py                E7 主程式(GPU;proxy 自生軌跡、target 常駐 CPU 交換)
exp_e6_slack_calibration.py      E6(零 GPU,stdlib-only;只取論文 5 benchmarks)
exp_e9_answer_span.py            E9(GPU ~1 h;gsm8k final-answer span,prompt_length 改寫)
exp_e11_base_instruct.py         E11(GPU ~1.5 h;base→instruct 全分解,γ>0 + PTQ anchor)
exp_e10_axis_alignment.py        E10(零 GPU,stdlib-only;合併三個結果樹)
exp_e12_cost_table.py            E12 成本表(CPU;三層估計 floor/standard/maj@8,自動吃 E12b 實測)
exp_e12_gsm8k_measure.py         E12b(GPU ~15-20 min;gsm8k decode×3 vs TF×1 實測,natural EOS)
exp_e13_fullmatrix.py            E13 20-cell 全矩陣(零 GPU;checksum 命中論文 0.831/−0.34/−0.66)
exp_eql6w3_prefactor.py          eQL6-W3 prefactor 檢查(零 GPU;論文 5 benchmarks:CV 0.63% vs 1-Ω 366×)
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
