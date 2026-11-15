# Scale-axis intervention: outlier-channel preservation on and off

8-bit quantization in bitsandbytes keeps outlier channels in FP16 by default
(`llm_int8_threshold = 6.0`), which is the remedy the paper names for the scale
axis. Raising the threshold to `1e9` quantizes every channel and changes
nothing else. This folder runs both settings on the same sequences and reports
how the measured gap and each term of the bound respond.

| setting name | `llm_int8_threshold` | meaning |
|---|--:|---|
| outliers kept | 6.0 | bitsandbytes default; the paper's BnB INT8 configuration |
| all quantized | 1e9 | no channel exceeds the threshold, so every channel is quantized |

## Files

| file | what it does | needs |
|---|---|---|
| `scale_toggle.py` | For one target model: per benchmark, evaluates the BF16 target once and the 8-bit variant once per threshold, and stores losses and the PRISM decomposition (W = I) per cell | GPU, `bitsandbytes` |
| `head_constants.py` | K_feat and K_pred of one target model, from its output head (`prism.core.bounds.UnifiedBound.theoretical_K`) | GPU recommended |
| `analyze.py` | Recomputes every reported quantity from the stored primitives and prints the same tables for every results file (per benchmark: gaps, term ratios, absolute terms, bound; then a summary), followed by an across-models table | CPU, Python standard library only |
| `run_scale.sh` | Runs the reported configurations for both models, then `analyze.py` | GPU |

## Protocol

- Targets: `Qwen/Qwen3-8B-Base` and `meta-llama/Llama-3.1-8B`, loaded in BF16.
  Variant: the same checkpoint loaded with
  `BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=...)`.
  Revisions are pinned through `configs/model_revisions.yaml`.
- Benchmarks: SQuAD, TriviaQA, MMLU, ARC, GSM8K; the loader's evaluation
  split (the test split of MMLU, ARC and GSM8K; the validation split of SQuAD
  and TriviaQA), the first 512 sequences after a shuffle with seed 42;
  `max_length` 2048; batch size 8. This is the loader and selection of the
  paper's quantization experiments.
- Settings: the reported thresholds are 6.0 and 1e9. `scale_toggle.py` labels
  a threshold in (0, 1e6) as "outliers kept" and any other value as "all
  quantized"; other thresholds were not run.
- Features and loss follow the paper's protocol for these benchmarks: the final
  hidden states at the answer positions of all sequences are stacked into Z,
  and the loss is the mean token cross-entropy at the same positions. The gap
  is `|loss_T - loss_P|` in nats.
- Decomposition under identity alignment: scale term `(rho_T - rho_P)^2`, shape
  term `2 rho_T rho_P (1 - Omega)`, feature term `delta = sqrt(scale + shape)`,
  head term `gamma`. Bound: `K_feat * delta + K_pred * gamma`.
- bitsandbytes does not quantize the output head, so the variant's head equals
  the target's and `gamma` is 0 in every cell.

## Commands

Everything, from the supplementary root:

```bash
bash scripts/response/scale/run_scale.sh
```

Knobs of `run_scale.sh`: `OUT_DIR` (output directory, default
`out/response/scale`), `PYTHON` (interpreter, default `python`) and
`CUDA_VISIBLE_DEVICES` (the scripts use `cuda:0` of the visible devices).

Step by step (shown for Qwen3-8B-Base; replace the target for Llama-3.1-8B):

```bash
mkdir -p out/response/scale
python scripts/response/scale/scale_toggle.py --target Qwen/Qwen3-8B-Base \
    --thresholds 6.0,1e9 --tasks squad,triviaqa,mmlu,arc,gsm8k --n 512 \
    --out out/response/scale/scale_toggle_qwen3_8b_base.json
python scripts/response/scale/head_constants.py --target Qwen/Qwen3-8B-Base \
    --out out/response/scale/head_constants_qwen3_8b_base.json
python scripts/response/scale/analyze.py \
    --results   out/response/scale/scale_toggle_qwen3_8b_base.json \
                out/response/scale/scale_toggle_llama_3_1_8b.json \
    --constants out/response/scale/head_constants_qwen3_8b_base.json \
                out/response/scale/head_constants_llama_3_1_8b.json
```

Environment switches shared with the rest of the package:
`PRISM_TRUST_REMOTE_CODE=1` allows remote model code (off by default; not
needed for these models), `PRISM_UNPINNED=1` ignores the pinned revisions.

## Output format

`scale_toggle.py` writes a JSON list, one record per (benchmark, threshold):

| field | meaning |
|---|---|
| `task`, `target`, `threshold` | benchmark, target model id, `llm_int8_threshold` |
| `outliers_preserved` | `true` for a threshold in (0, 1e6) (outliers kept; 6.0 in the reported runs), `false` otherwise (all quantized; 1e9 in the reported runs) |
| `loss_T`, `loss_P`, `dR` | target loss, variant loss, `abs(loss_T - loss_P)` |
| `rho_T`, `rho_P`, `d_rho` | RMS feature scales and their signed difference |
| `scale_term`, `shape_term`, `omega`, `delta`, `gamma` | terms of the decomposition |
| `secs` | wall-clock seconds for the variant side of the cell |

`head_constants.py` writes `{"target", "K_feat", "K_pred"}`. `analyze.py`
requires the `target` field and checks that it matches the results file the
constants file is paired with.

## Runtime

- `scale_toggle.py`: one GPU. Each model needs five BF16 target passes and
  ten passes of the 8-bit variant; the `secs` field of each cell records the
  time of the variant side. The PRISM contractions run on the GPU and fall
  back to the CPU when the GPU runs out of memory.
- `head_constants.py`: dominated by loading the model; the constant itself is
  computed from the output head alone.
- `analyze.py`: no GPU.
