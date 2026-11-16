# Head experiments

Two experiments on the output head (`lm_head`) of GGUF variants. "Backbone" is
everything before the output head. The target is the original model in BF16.
Metrics use `W = I`. Five benchmarks (ARC, MMLU, SQuAD, TriviaQA, GSM8K),
`n = 512`, seed 42.

**Gap.** Gaps are `|R_target - R|` in nats, where `R` is the answer-token
cross-entropy on the 512 sequences.

**Noise floor.** Gaps and differences of gaps below 0.02 nats are treated as
noise. It applies to gaps (which cells enter the interaction of experiment 2) and to
differences of gaps (which pairs are counted, which changes are "beyond
noise"). It is a constant in both analysis scripts (`NOISE_FLOOR`), not an
option.

| file | what it does | GPU |
|---|---|---|
| `head_swap.py` | experiment 1: for each GGUF variant, evaluates the variant, hybrid A (its quantized backbone + the original head) and hybrid B (the original backbone + its quantized head) | yes |
| `head_precision.py` | experiment 2: backbones {original, GGUF Q4_K_M, GGUF Q2_K} x heads {BF16, Q8_0, Q6_K, Q5_0, Q4_0, 3-bit, 2-bit} | yes |
| `analyze_head_swap.py` | numbers of experiment 1 from the JSON written by `head_swap.py` | no |
| `analyze_head_precision.py` | numbers of experiment 2 from the JSON written by `head_precision.py` | no |
| `run_head.sh` | launches the runs below and the two analyses | |

Models: `Qwen/Qwen3-8B-Base` with `mradermacher/Qwen3-8B-Base-GGUF`, and
`meta-llama/Llama-3.1-8B` with `QuantFactory/Meta-Llama-3.1-8B-GGUF`. All four
repositories are loaded at the commits in `configs/model_revisions.yaml`;
remote code is off unless `PRISM_TRUST_REMOTE_CODE=1`.

## Commands

From the supplementary root:

```bash
bash scripts/response/head/run_head.sh                   # everything
STAGE=swap      bash scripts/response/head/run_head.sh   # experiment 1 (20 cells)
STAGE=precision bash scripts/response/head/run_head.sh   # experiment 2 (210 cells)
STAGE=analyze   bash scripts/response/head/run_head.sh   # both analyses, no GPU
```

`STAGE=analyze` runs both analyses and needs the outputs of both experiments.
Outputs go to `out/response/head/`. Knobs of `run_head.sh`: `OUT` (output
directory), `CUDA_GPU` (GPU index, default 0) and `PYTHON` (interpreter,
default `python3`).
The underlying calls are:

```bash
python scripts/response/head/head_swap.py --target Qwen/Qwen3-8B-Base \
    --gguf_repo mradermacher/Qwen3-8B-Base-GGUF --gguf_tpl 'Qwen3-8B-Base.{quant}.gguf' \
    --quants Q6_K,Q4_K_M,Q2_K --out out/response/head/head_swap_qwen3_8b_base.json
python scripts/response/head/head_swap.py --target meta-llama/Llama-3.1-8B \
    --gguf_repo QuantFactory/Meta-Llama-3.1-8B-GGUF --gguf_tpl 'Meta-Llama-3.1-8B.{quant}.gguf' \
    --quants Q6_K --out out/response/head/head_swap_llama_3_1_8b.json

python scripts/response/head/head_precision.py --target Qwen/Qwen3-8B-Base \
    --gguf_repo mradermacher/Qwen3-8B-Base-GGUF --gguf_tpl 'Qwen3-8B-Base.{quant}.gguf' \
    --backbones Q4_K_M,Q2_K --tasks arc,mmlu,squad \
    --out out/response/head/head_precision_qwen3_8b_base_a.json
# ... same with --tasks triviaqa,gsm8k (_b), and both again for Llama-3.1-8B

python scripts/response/head/analyze_head_swap.py \
    out/response/head/head_swap_qwen3_8b_base.json out/response/head/head_swap_llama_3_1_8b.json
python scripts/response/head/analyze_head_precision.py out/response/head/head_precision_*_[ab].json
```

Both GPU scripts need the `gguf` package (in `requirements-response.txt`):
`head_swap.py` and `head_precision.py` load GGUF files through transformers,
which uses it, and `head_precision.py` also uses it for its start-up check of
the quantizers. The analysis scripts use the
Python standard library only.

The analyses need no GPU.

## What is measured

**Experiment 1.** Per cell (model, benchmark, variant), three gaps: `V` for the
variant, `A` for hybrid A, `B` for hybrid B. The script verifies on the
extracted features that hybrid A has exactly the variant's features and hybrid B
exactly the target's.

* Ranking: in hybrid A the head term is 0, so the bound is the feature term
  alone; the script checks whether it orders the variants of each benchmark as
  the gap of hybrid A does, and counts the correctly ordered pairs of variants
  twice: over all pairs, and over the pairs whose hybrid-A gaps differ by at
  least 0.02 nats.

**Experiment 2.** Per cell (model, benchmark, backbone, head): the gap, the
feature term and the head term.

* The Q8_0, Q5_0, Q4_0, 3-bit and 2-bit heads are produced from the target's
  head by the quantizers in `head_precision.py` (blocks of 32, FP16 scale).
  The Q8_0, Q5_0 and Q4_0 quantizers are compared with gguf-py at start-up on
  the first 2,048 rows of the head; the script stores the share of differing
  entries and the largest absolute difference (`quantizer_check` in the
  output). The 3- and 2-bit heads use the Q4_0 scheme
  with fewer levels; they are not GGUF formats and have no reference
  implementation to compare with.
* The head called Q6_K is the `lm_head` weight of the Q4_K_M GGUF file as
  transformers dequantizes it. All heads are run in BF16.
* Pairs: two quantized head settings in the same group whose outcomes differ by
  at least 0.02 nats (the noise floor); a pair is ranked correctly if the larger head term goes
  with the larger outcome (ties in the head term count as wrong). Pairs with the BF16 head are
  left out, since its head term and outcome are 0 by construction.
  * head-only gap: group = (model, benchmark) on the original backbone, outcome =
    the gap;
  * lowering: group = (model, benchmark, quantized backbone), outcome = gap with
    that head minus gap with the BF16 head, i.e. how much restoring the original
    head lowers the gap.
* Interaction: `I = dR(b, h) - dR(b, BF16) - dR(BF16, h)` on signed gaps, for a
  quantized backbone `b` and a quantized head `h`; reported as `|I| / |dR(b, h)|`
  over the cells with `|dR(b, h)| >= 0.02`.

**Bound.** Both analyses evaluate `K_feat * delta + K_pred * gamma >= gap` from
the stored `delta` and `gamma`. `K_feat` (the largest distance between two
columns of the target head, `UnifiedBound.theoretical_K`) and `K_pred = sqrt(2)`
are stored by the GPU scripts.
