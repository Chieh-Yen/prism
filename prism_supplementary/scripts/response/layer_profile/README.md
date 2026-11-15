# Layer-wise and module-wise shape-drift profile

The bound is stated at the output: it needs the linear map from features to
logits, which only the final hidden state has. This folder adds a descriptive
profile inside the network. For a BF16 target and a GGUF variant run on the
same sequences, it measures the shape drift `1 - Omega` at four points of
every block: the residual stream after the block and the outputs of its q, k
and v projections.

## Files

| file | what it does | needs |
|---|---|---|
| `layer_profile.py` | Loads the target and one GGUF variant at a time, runs both on the same batches, and accumulates the per-point statistics | GPU that holds two 8B models in BF16 at once |
| `make_table.py` | Peak table, token counts, the v_proj / q_proj peak ratio and the bit-width ordering check | CPU, Python standard library only |
| `run_layer_profile.sh` | Runs the reported configurations, then `make_table.py` | GPU |

## Protocol

- Targets and variants: `Qwen/Qwen3-8B-Base` against `Q6_K`, `Q4_K_M`, `Q2_K`
  from `mradermacher/Qwen3-8B-Base-GGUF`; `meta-llama/Llama-3.1-8B` against
  `Q6_K` from `QuantFactory/Meta-Llama-3.1-8B-GGUF`. The GGUF files are
  de-quantized by `transformers` (`gguf_file=...`) and run in BF16, as in the
  paper's quantization experiments. Revisions are pinned through
  `configs/model_revisions.yaml`.
- Sequences: MMLU test split, the first 128 sequences after a shuffle with
  seed 42; `max_length` 1024; batch size 4. All non-pad tokens are used,
  prompt included (about 14k tokens).
- Read-out points:
  - residual stream: every entry of `hidden_states`. Index 0 is the embedding
    output, index i (1 to L-1) is the stream after block i, and index L is the
    final hidden state, i.e. after block L and the final norm (L = 36 for
    Qwen3-8B-Base, 32 for Llama-3.1-8B);
  - q, k, v: the output of the linear layers `self_attn.q_proj`, `k_proj`,
    `v_proj` in every block (forward hooks). This is the full projection
    output, one vector per token over all heads, taken before it is split
    into heads; for Qwen3-8B-Base it is therefore taken before the per-head
    `q_norm` / `k_norm` of its attention.
- Normalisation: each token vector is divided by its own L2 norm before
  accumulation. In the residual stream a few tokens carry very
  large activations and would otherwise dominate the inner product. The
  RMSNorm in front of every block applies the same per-token rescaling.
- Statistic: `Omega = <Z_T, Z_P> / (||Z_T||_F ||Z_P||_F)` under identity
  alignment. With unit-length token
  vectors this is the mean per-token cosine between target and variant, so
  `1 - Omega = 0` means no drift. The profile carries the shape term only: the
  normalisation sets scale aside (the stored `rho_T` and `rho_P` are 1 by
  construction), and there is no head inside the network.

## Numbering used in the table

Blocks are numbered from 1.

- Residual stream: the peak is taken over the L values after each block
  (`hidden_states` indices 1 to L). Index L is the final hidden state; when the
  peak falls there it is the value at the output and is printed as `(output)`.
  The embedding output (index 0) is not a block; it is stored in the JSON and
  is not part of the search or of the table.
- q, k, v: the JSON stores one value per block with a 0-based index j; the
  reported block is j + 1.
- The reported block is that of the single largest value.
  `make_table.py --profiles` prints every block.

## Commands

Everything, from the supplementary root:

```bash
bash scripts/response/layer_profile/run_layer_profile.sh
```

Knobs of `run_layer_profile.sh`: `OUT_DIR` (output directory, default
`out/response/layer_profile`), `PYTHON` (interpreter, default `python`) and
`CUDA_VISIBLE_DEVICES` (the script uses `cuda:0` of the visible devices).

Step by step:

```bash
mkdir -p out/response/layer_profile
python scripts/response/layer_profile/layer_profile.py --target Qwen/Qwen3-8B-Base \
    --gguf_repo mradermacher/Qwen3-8B-Base-GGUF \
    --gguf_tpl 'Qwen3-8B-Base.{quant}.gguf' \
    --quants Q6_K,Q4_K_M,Q2_K --task mmlu --n 128 \
    --out out/response/layer_profile/layer_profile_qwen3_8b_base.json
python scripts/response/layer_profile/layer_profile.py --target meta-llama/Llama-3.1-8B \
    --gguf_repo QuantFactory/Meta-Llama-3.1-8B-GGUF \
    --gguf_tpl 'Meta-Llama-3.1-8B.{quant}.gguf' \
    --quants Q6_K --task mmlu --n 128 \
    --out out/response/layer_profile/layer_profile_llama_3_1_8b.json
python scripts/response/layer_profile/make_table.py \
    --results out/response/layer_profile/layer_profile_qwen3_8b_base.json \
              out/response/layer_profile/layer_profile_llama_3_1_8b.json
```

`make_table.py --profiles` also prints the full depth profile of every variant.

Environment switches shared with the rest of the package:
`PRISM_TRUST_REMOTE_CODE=1` allows remote model code (off by default; not
needed for these models), `PRISM_UNPINNED=1` ignores the pinned revisions.

## Output format

`layer_profile.py` writes a JSON list, one record per variant:

| field | meaning |
|---|---|
| `target`, `quant`, `task` | target model id, GGUF tag, benchmark |
| `rows` | number of tokens (non-pad positions) |
| `omega` | L + 1 values, one per entry of `hidden_states`, rounded to 6 decimals |
| `rho_T`, `rho_P` | RMS scale of the normalised token vectors (1 by construction), rounded to 4 decimals |
| `modules.{q_proj,k_proj,v_proj}` | the same three lists per projection, L values each, 0-based block index |
| `secs` | wall-clock seconds for the variant (loading included) |

## Runtime

`layer_profile.py` keeps the target and one variant on the GPU together (two
8B models in BF16, about 16 GB of weights each, plus activations). This does
not fit in 32 GB; use a 96 GB GPU. `make_table.py` needs no GPU.
