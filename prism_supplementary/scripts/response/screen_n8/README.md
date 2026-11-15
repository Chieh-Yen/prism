# Eight-sequence screen

How well does a score computed from eight reference sequences order the quantized
variants of a model by the risk gap measured on 512 sequences, and how much does the
answer depend on which eight sequences are used?

## Protocol

- **Cells.** Ten: Llama-3.1-8B and Qwen3-8B-Base, each on ARC, MMLU, SQuAD, TriviaQA and
  GSM8K. Variants: 12 for Llama-3.1-8B and 11 for Qwen3-8B-Base, listed in
  `scripts/rounds/data/variant_manifest.csv`. The manifest is the variant list of the
  three-seed round of Appendix F.2 (12 and 11 variants); the screen uses it unchanged.
  It includes three GPTQ repositories, two for Llama-3.1-8B and one for Qwen3-8B-Base.
- **Pools.** For each benchmark, three pools of 512 evaluation sequences, seeds 43, 44 and 45:
  the pools of the three-seed round (`shuffle(seed)`, then the first 512). The risk gap of
  a variant on all 512 sequences of a pool (`gap512`) gives the ordering to be recovered.
- **Reference sets.** 100 sets of eight sequences per pool. Set 0 is the first eight
  sequences of the pool, which is the n = 8 slice of the three-seed round. Sets 1 to 99
  are eight of the 512 drawn uniformly without replacement by
  `numpy.random.default_rng([seed, benchmark index, 2026])`, with benchmark index
  ARC 0, MMLU 1, SQuAD 2, TriviaQA 3, GSM8K 4. The sets are drawn before any variant is
  loaded, and the same set is used for the target and for every variant. This gives 300
  sets per cell, 3,000 in total, and all of them are written out and analysed.
- **Scores on each set.** `1-CKA` (linear CKA), `B_N` (the PRISM bound under the
  Procrustes alignment, `K_feat * delta_N + K_pred * gamma_N`, with `delta_N` the
  feature term and `gamma_N` the head term), and `gap@8` (the risk gap measured on the
  same eight sequences).
- **Statistic.** For each set, the Spearman r_s across the variants between a score and
  `gap512` of the set's pool.

## Files

| file | what it does | needs |
|---|---|---|
| `run_screen.py` | GPU pass for one family: loads the target once and each variant once, extracts answer-token features and answer losses on the three pools of every benchmark, and writes one row per (variant, seed, benchmark, set) to `{family}_screen.csv`, plus `{family}_sets.json` (the index sets) and `{family}_K.json` (K_feat, K_pred) | GPU |
| `analyze_screen.py` | tables and tests from the CSVs: summary over the ten cells, per-cell mean ± sd and worst set (the lowest r_s among the cell's 300 sets, for each score), across-cell and within-cell paired tests, bound coverage; with `--size-round`, the coverage at n = 8, 32, 128, 512 from the three-seed round's CSVs | numpy, scipy |
| `run_screen.sh` | launcher: stages `run`, `gptq`, `analyze` | |

Shared code: `scripts/rounds/common_quant.py` (variant manifest and model loading) and
the `prism` package. Models are loaded at the revisions pinned in
`configs/model_revisions.yaml`; `trust_remote_code` is off unless
`PRISM_TRUST_REMOTE_CODE=1`.

## Commands

From the supplementary root:

```bash
# GPU pass 1 (resumable): the 10 + 10 variants that need no GPTQ loader;
# writes out/response/screen_n8/{llama,qwen}_screen.csv
bash scripts/response/screen_n8/run_screen.sh run

# GPU pass 2 (resumable), in an environment with a GPTQ loader: the three GPTQ
# variants, appended to the same CSVs
bash scripts/response/screen_n8/run_screen.sh gptq
# equivalent to
python3 scripts/response/screen_n8/run_screen.py --family llama --out out/response/screen_n8 \
    --only "BF16 vs GPTQ(Meta-Llama-3.1-8B-gptq-4bit)" "BF16 vs GPTQ(Meta-Llama-3.1-8B-GPTQ)"
python3 scripts/response/screen_n8/run_screen.py --family qwen --out out/response/screen_n8 \
    --only "BF16 vs GPTQ(Qwen3-8B-GPTQ)"

# tables (no GPU)
bash scripts/response/screen_n8/run_screen.sh analyze
# with the checks that use the output directory of scripts/rounds/three_seed_round.py
SIZE_ROUND=out/rounds/three_seed \
    bash scripts/response/screen_n8/run_screen.sh analyze
```

**Two passes.** The GPTQ repositories are loaded through the GPTQ support of
`transformers`, which needs a GPTQ loader: the GPTQModel package (`gptqmodel`) together
with `optimum`. `requirements-response.txt` lists neither, so the screen is run in two
passes that write to the same CSVs: `run` for the variants that need no GPTQ loader, and
`gptq` for the three GPTQ variants in the environment of `requirements-response.txt`
with `gptqmodel==6.0.3` and `optimum==2.2.0` added (see the header of that file).
Knobs of `run_screen.sh`: `OUT` (output directory, default
`out/response/screen_n8`), `CUDA_GPU` (GPU index, default 0), `FAMILIES`
(default `"llama qwen"`), `SIZE_ROUND` (see above) and `PYTHON` (interpreter
of a stage, default `python3`). Without `--only`,
`run_screen.py` walks the whole manifest; a variant
that cannot be loaded is reported, listed again at the end, and the script exits with
status 1. `analyze_screen.py` stops on an incomplete grid (fewer than 12 or 11 variants
in a cell) unless `--allow-incomplete` is given.

A subset of variants can be run with `--only "<label>" ...`; rows are appended to the
same CSV, and finished (variant, seed, benchmark) blocks are skipped on a re-run. A label
that is not in the manifest of the family is an error.

Cost: per family, the GPU pass does one target load and 12 or 11 variant loads, each
followed by a forward pass over 15 pools of 512 sequences and 1,500 set-level
computations. The analysis needs no GPU.

## Definitions and notes

- **Risk gap.** `gap8` and `gap512` are the risk gap on the set's eight sequences and on
  the pool's 512, as in the three-seed round.
- **Set 0 and the pools.** Set 0 holds the same sequences as the n = 8 slice of the
  three-seed round, and the pools are that round's pools. The scores of set 0 and the
  pool's `gap512` are recomputed here in a separate run and are not copied from that
  round.
- **Pools.** The three pools of a benchmark are drawn from the same test split and
  overlap. The 100 sets of a pool are drawn independently of one another from that pool. The within-cell tests take the set as the unit.
- **Standard deviation.** The ± values are the population standard deviation (ddof = 0)
  of r_s over the 300 sets: the spread across sets, not a standard error of the mean.
- **Tests.** Across cells: exact two-sided Wilcoxon signed-rank test on the ten cell
  means. Within a cell: two-sided Wilcoxon signed-rank test on the 300 paired r_s values
  (normal approximation with tie correction, zero differences dropped), Holm-corrected
  over the ten cells, with the paired t-test as a check. r_s is rounded to six decimals
  before testing.
- **1-SVCCA is not scored by the screen.** `run_screen.py` does not compute SVCCA and
  `analyze_screen.py` prints nothing about it. The CSVs of the three-seed round
  (`scripts/rounds/three_seed_round.py`) contain SVCCA at n = 8, 32, 128 and 512
  (column `svcca`; the score is one minus it).
