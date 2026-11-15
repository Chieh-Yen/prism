# LoRA checkpoint series: PRISM bound against 1-CKA and 1-SVCCA

Ranks the checkpoints of unregularized LoRA fine-tuning runs by how much each
has forgotten, and compares three scores computed from one forward pass: the
PRISM bound, 1-CKA and 1-SVCCA.

| file | purpose |
|---|---|
| `run_series.py` | trains the six runs (Llama-3.1-8B; TruthfulQA and BBQ; seeds 42, 43, 44), 300 steps, one evaluation every 25 steps (12 checkpoints), with `PRISM_EVAL_SIMILARITY=1` so that CKA and SVCCA are recorded next to the PRISM terms, and with `--eval_keep_anchor_rows` |
| `analyze.py` | per-cell Spearman r_s of each score against the risk gap, the ten-cell means, and the validity count of the bound |
| `launch_lora_table.sh` | both steps |

## Commands

```bash
# from the supplementary root
bash scripts/response/lora_similarity/launch_lora_table.sh out/lora_series
# or step by step
python scripts/response/lora_similarity/run_series.py --out_root out/lora_series
python scripts/response/lora_similarity/analyze.py   --out_root out/lora_series
```

`run_series.py --dry_run` prints the six training commands. Training needs one
GPU that holds an 8B model in bf16; `analyze.py` reads JSON only and finishes
in about a second on a CPU. Knobs of `launch_lora_table.sh`: the output root
is its first argument, `PYTHON` selects the interpreter (default `python`),
and the GPU is the one `CUDA_VISIBLE_DEVICES` exposes.

## What is computed

- **Cell**: a (fine-tuning task, downstream benchmark) pair. Two tasks times
  five benchmarks (ARC, GSM8K, MMLU, SQuAD, TriviaQA) give ten cells.
- **Training**: `train_forgetting.py --lambda_shape 0 --lambda_replay 0`,
  i.e. the library's default trainer, with no penalty and no anchor set. The
  trainer selects the paged 8-bit AdamW of bitsandbytes when that package can
  be imported, and torch AdamW otherwise. The launcher passes `--eval_keep_anchor_rows`, so the trainer's
  `eval_loss` is computed on the whole held-out slice of the fine-tuning
  task, as in the reported series; `analyze.py` does not read `eval_loss`.
- **Seeds**: `--seed` sets the adapter initialisation, the data order of
  training and the draw of the 256 evaluation sequences of each benchmark, so
  the three seeds of a cell differ in the training run and in the evaluation
  sequences.
- **Gap**: |ΔR|, the absolute difference between a checkpoint's and the base
  model's answer-only cross-entropy on 256 evaluation sequences of the
  benchmark. The answer-only cross-entropy is the mean over the 256 sequences
  of each sequence's mean answer-token loss.
- **Feature rows**: the features are one row per answer token of the 256
  sequences (field `Z_shape` of `base_model` in the metrics file); the
  feature dimension is 4,096.
- **Bound**: the PRISM bound under the identity alignment W = I, on the same
  256 sequences. LoRA leaves the output head frozen, so the head term is zero
  and the bound is the feature term alone.
- **1-CKA, 1-SVCCA**: computed by the trainer on the features the bound uses
  (final layer, base model against checkpoint), with the functions of
  `similarity_baselines.py`. SVCCA keeps, for each model, the principal
  directions that reach 99% of the variance, at most 256. Both scores are
  computed in float32.
- **r_s**: within one run and one benchmark, the Spearman correlation between
  a score and the gap over the 12 checkpoints.
- **mean ± sd**: over the three seeds of a cell; the sd is the sample standard
  deviation (divisor 2).
- **Validity**: bound ≥ gap, checked at each of the 12 × 10 × 3 = 360
  (checkpoint, cell, seed) cases.
