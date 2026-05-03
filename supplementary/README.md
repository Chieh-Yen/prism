# PRISM — Supplementary Code

Reproducibility package for the paper *PRISM: Decomposing Geometric Drift for
Risk Diagnosis in LLM Variants*. This package is **run-only**: it contains
the experiment pipelines for the two main experiments (PTQ grid in §5.2 and
LoRA forgetting + regularization in §5.4) but no plotting, table generation,
or analysis utilities.


## Layout

```
.
├── README.md                  # this file
├── requirements.txt           # pinned Python dependencies
├── install.sh                 # one-shot environment setup
│
├── run_quantization.sh        # entry: PTQ grid (Sec. 5.2)
├── run_quantization.py        # PTQ Python entry (called by run_quantization.sh)
│
├── run_forgetting.sh          # entry: LoRA reg sweep (Sec. 5.4)
├── run_forgetting_one.sh      # internal: single-config LoRA driver
├── train_forgetting.py        # internal: LoRA training + online PRISM eval
│
├── configs/
│   ├── quantization.yaml          # default PTQ run config
│   └── quantization_matrix.yaml   # model × dataset × bit-width grid
├── scripts/
│   └── quantization_matrix.py     # matrix expansion helper
└── prism/                         # PRISM core library
    ├── core/        bounds.py, metrics.py
    ├── data/        loaders.py
    ├── models/      extractors.py
    └── experiments/ quantization.py, forgetting.py
```


## Hardware

A single **NVIDIA RTX 5090 (32 GB)** is sufficient for every run reported in
the paper. CUDA 12.8 wheels are pinned in `requirements.txt`.


## Installation

### One-shot (recommended)

```bash
bash install.sh
```

### Manual

GPTQModel does not install cleanly via `pip install -r` because (a) its
`setup.py` cannot detect torch from pip's isolated build sandbox, and
(b) its `pyproject.toml` uses the PEP-639 SPDX `license` field that
setuptools < 71 rejects. Install in this order:

```bash
# 1. torch first (so subsequent builds can detect it)
pip install torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2. setuptools >= 71 (required by GPTQModel pyproject.toml)
pip install "setuptools>=71"

# 3. everything else
pip install -r requirements.txt

# 4. GPTQModel last, with build isolation disabled
pip install GPTQModel==5.7.0 --no-build-isolation
```


## Experiment 1 — Quantization Grid (Sec. 5.2)

The PTQ matrix (four base 8B families × three PTQ families across bit-widths
2–8 × five benchmarks) is enumerated by `configs/quantization_matrix.yaml`
and dispatched by `run_quantization.sh`. Each run loads the BF16 base
(target) and the quantized variant (proxy) on a shared $N{=}512$-token
calibration subset, computes
$\rho_T,\,\rho_P,\,\Omega,\,\delta,\,\gamma,\,\mathcal{B}$
in a single forward pass, and writes a per-(model, dataset, method) JSON
record under `results/quantization/`.

```bash
# full grid (≈ 18 h on a single RTX 5090)
bash run_quantization.sh

# subset / debugging
NUM_SAMPLES=128 DRY_RUN=1 bash run_quantization.sh
MODELS="llama_8b_base,qwen3_8b_base" \
DATASETS="mmlu,arc"      bash run_quantization.sh
MULTI_GPU=1              bash run_quantization.sh   # 2 GPUs
```


## Experiment 2 — LoRA Forgetting + Regularization Sweep (Sec. 5.4)

`run_forgetting.sh` is the **one-stop** entry point: it runs
both regularizers, on both models, on both fine-tuning tasks, sweeping
$\lambda$ across the paper-reported ranges plus a $\lambda{=}0$ no-reg
anchor. PRISM forgetting metrics are computed online at every checkpoint,
so there is no separate inference stage.

```bash
# full sweep (≈ 22 h on a single RTX 5090: 4 × (6 replay + 5 shape) = 44 runs)
bash run_forgetting.sh
```

Default $\lambda$ grids match Sec. 5.1 of the paper:

| Regularizer       | $\lambda$ grid                                    |
|-------------------|--------------------------------------------------|
| replay-CE         | `0  0.001  0.005  0.01  0.05  0.1` (λ=0 = no-reg anchor) |
| trace-norm shape  | `0.01  0.05  0.1  0.5  1.0`                      |

Override via env vars:

```bash
REPLAY_LAMBDAS="0 0.01" SHAPE_LAMBDAS="0.5 1.0" \
    bash run_forgetting.sh
```

A re-run skips any (model, task, $\lambda$) combination whose
`prism_forgetting_metrics.json` is already present; the sweep is
idempotent.

### Outputs

Per-(model, task) checkpoint trajectories under

```
checkpoints/forgetting_{shape,replay}_lam<λ>/<model>/<task>/
└── prism_forgetting_metrics.json
```

containing per-step PRISM metrics (Ω, δ, γ, ℬ, |ΔR|) on five downstream
benchmarks.

### Single-config invocation

`run_forgetting_one.sh` is the inner driver called by the wrapper
above; it can be invoked directly for a single configuration:

```bash
# baseline (no regularization)
bash run_forgetting_one.sh

# trace-norm shape regularizer
SHAPE_REG=1 LAMBDA_SHAPE=1.0 bash run_forgetting_one.sh

# replay-CE baseline
REPLAY_REG=1 LAMBDA_REPLAY=0.01 bash run_forgetting_one.sh
```

`SHAPE_REG` and `REPLAY_REG` are mutually exclusive (the script aborts
if both are set).


## Reproducibility Notes

- **Random seeds.** Both pipelines fix HuggingFace `set_seed(42)` before
  model load and data shuffling. PTQ runs are deterministic up to GPU
  non-determinism in matmuls; LoRA runs match across reruns up to
  non-deterministic CUDA kernels (e.g., scaled dot-product attention's
  fused backward).
- **Benchmark splits.** Each evaluation samples a fixed held-out subset
  (512 for PTQ, 256 for LoRA forgetting) seeded by the dataset name, so
  subsets are stable across reruns regardless of dataset version.
- **Quantized-checkpoint sources.** GPTQ checkpoints come from the
  Hugging Face repositories listed in Appendix~F of the paper; GGUF
  artifacts are produced on-the-fly from the BF16 base via `llama.cpp`'s
  `quantize` tool; BnB variants are produced on-the-fly from BF16 via
  the `bitsandbytes` backend.
- **Anonymity.** This package contains no author identifiers (no email,
  no institution, no path-derived names). The Hugging Face cache directory
  defaults to `$HF_HOME` or `$XDG_CACHE_HOME/huggingface` and is never
  written to a hard-coded user path.


## Wall-clock summary (single RTX 5090)

| Experiment                                    | Time      |
|-----------------------------------------------|-----------|
| PTQ grid (Sec. 5.2, full default)             | ≈ 18 h    |
| LoRA forgetting regularization sweep (Sec. 5.4)| ≈ 22 h    |
| **Total**                                     | **≈ 40 h**|
