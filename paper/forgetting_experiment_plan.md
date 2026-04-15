# PRISM Forgetting Experiment Plan

> **Goal**: Empirically validate Eq. 8 of the paper — show that the PRISM geometric
> bound $Z = K_\text{feat} \sqrt{(\rho_0 - \rho_t)^2 + 2\rho_0\rho_t(1 - \Omega)}$
> tracks empirical risk degradation (catastrophic forgetting) when fine-tuning on
> one task and evaluating on others.

---

## 1. Experiment Overview

### 1.1 Core Hypothesis

When a base model $\theta_0$ is fine-tuned on task $A$ to produce $\theta_t$, its
performance on held-out tasks $B, C, D, E$ degrades (forgetting). This empirical risk
increase $|\mathcal{R}_{\theta_0} - \mathcal{R}_{\theta_t}|$ on held-out tasks is
**upper-bounded** by PRISM's geometric bound (Eq. 8), and the bound should
**correlate strongly** (Spearman $\rho > 0.9$) with actual degradation across tasks,
models, and training steps.

Under LoRA fine-tuning with frozen `lm_head` ($H_t = H_0$), the head divergence
term vanishes, isolating forgetting entirely in backbone geometry:

$$\text{Forgetting}(t) \le K_\text{feat} \cdot \sqrt{(\rho_0 - \rho_t)^2 + 2\rho_0\rho_t(1 - \Omega(Z_0, Z_t))}$$

### 1.2 Experiment Matrix

| Dimension       | Values                                          |
|-----------------|-------------------------------------------------|
| **Models**      | LLaMA-3.1-8B, Qwen3-8B                         |
| **Fine-tune tasks** | ARC, MMLU, SQuAD, TriviaQA, GSM8K          |
| **Eval tasks**  | All 5 tasks (cross-evaluation)                  |
| **Training**    | LoRA (frozen lm_head)                           |
| **Hardware**    | 1× RTX 5090 (32 GB VRAM)                       |

**Total runs:**
- Stage 1 (fine-tuning): 5 tasks × 2 models = **10 training runs**
- Stage 2 (PRISM inference): 10 runs × ~12 checkpoints × 5 eval tasks = **~600 PRISM evaluations**

---

## 2. Model Selection & Justification

### 2.1 LLaMA-3.1-8B (`meta-llama/Llama-3.1-8B`)

| Property         | Value                                |
|------------------|--------------------------------------|
| Hidden dim       | 4096                                 |
| Layers           | 32                                   |
| Vocab size       | 128,256                              |
| Native dtype     | bf16                                 |
| GQA heads        | 32 Q / 8 KV                         |
| Base model       | `meta-llama/Llama-3.1-8B`           |

**Why**: Industry standard, most widely benchmarked 8B model. Extensive fine-tuning
documentation from Meta (`llama-recipes`), Unsloth, Axolotl. The original PRISM paper
already includes Llama experiments.

### 2.2 Qwen3-8B (`Qwen/Qwen3-8B-Base`)

| Property         | Value                                |
|------------------|--------------------------------------|
| Hidden dim       | 4096                                 |
| Layers           | 36                                   |
| Vocab size       | 151,936                              |
| Native dtype     | bf16                                 |
| GQA heads        | 32 Q / 8 KV                         |
| Base model       | `Qwen/Qwen3-8B-Base`                |

**Why**: Different architecture family (36 layers vs 32, 151K vocab vs 128K),
providing diversity. Already used in PRISM quantization experiments. The large vocab
size creates interesting memory constraints and tests PRISM's robustness.

### 2.3 Why Not DeepSeek-R1-8B or Ministral-3-8B

- **DeepSeek-R1-8B**: Reasoning-specialized model with CoT baked in. Fine-tuning
  behavior is atypical (may resist standard SFT patterns). Better as a future
  ablation than a primary comparison.
- **Ministral-3-8B**: Multimodal architecture (`Mistral3ForConditionalGeneration`)
  with a vision-language wrapper. The nested `model.model.language_model` backbone
  adds complexity to feature extraction. Mistral-7B-v0.1 would be simpler, but
  overlaps with Llama's architectural class.

---

## 3. Tasks: Datasets, Formats & Best Practices

### 3.1 Task Summary

| Task     | HF Dataset ID                | Train Size | Eval Split  | Format Type    | Metric     |
|----------|------------------------------|------------|-------------|----------------|------------|
| ARC      | `allenai/ai2_arc` (Challenge)| 1,119      | test (1,172)| Multiple-choice| Accuracy   |
| MMLU     | `cais/mmlu` (all)            | 99,842 (aux)| test (14,042)| Multiple-choice| Accuracy |
| SQuAD    | `rajpurkar/squad`            | 87,599     | validation (10,570)| Extractive QA | EM / F1 |
| TriviaQA | `trivia_qa` (rc.nocontext)  | ~87,000    | validation (~11K)| Open-domain QA | EM / F1 |
| GSM8K    | `openai/gsm8k` (main)       | 7,473      | test (1,319)| Math reasoning | Accuracy   |

### 3.2 Fine-Tuning Data Formats

All tasks use the existing `prism/data/loaders.py` formatters for **PRISM evaluation**
(Stage 2). For **fine-tuning** (Stage 1), we use the same base format but wrapped
in instruction-tuning structure.

#### ARC (Multiple-Choice)
```
Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Answer: {answerKey}
```
- **Loss**: Compute on full sequence (including reasoning path)
- **Train split**: `train` (1,119 examples)
- **Epochs**: 5 (small dataset — more epochs needed)
- **Notes**: ARC-Challenge only (harder subset, standard benchmark)

#### MMLU (Multiple-Choice)
```
Question: {question}
A. {choice_0}
B. {choice_1}
C. {choice_2}
D. {choice_3}
Answer: {A/B/C/D}
```
- **Loss**: Full sequence
- **Train split**: `auxiliary_train` (99,842 examples)
- **Epochs**: 1 (large dataset)
- **Notes**: MMLU has minimal dev/few-shot data by design. We use the
  `auxiliary_train` split which contains ~100K training examples across subjects.

#### SQuAD (Extractive QA)
```
Context: {context}
Question: {question}
Answer: {answer_text}
```
- **Loss**: Full sequence (but answer-only masking yields better models)
- **Train split**: `train` (87,599 examples)
- **Epochs**: 1 (large dataset)
- **Max length**: 512 (truncate long contexts)

#### TriviaQA (Open-Domain QA)
```
Question: {question}
Answer: {answer_value}
```
- **Loss**: Full sequence
- **Train split**: `train` (~87K, `rc.nocontext` subset)
- **Epochs**: 1 (large dataset)
- **Notes**: No context provided (open-domain). Multiple valid answer aliases.

#### GSM8K (Math Reasoning)
```
Question: {question}
Answer: {full_chain_of_thought_answer}
```
- **Loss**: Full sequence (CoT supervision is essential)
- **Train split**: `train` (7,473 examples)
- **Epochs**: 3
- **Notes**: The `answer` field in GSM8K contains the full chain-of-thought
  reasoning, not just the final number. Training on CoT is critical.

### 3.3 Data Subsetting Strategy

To keep fine-tuning comparable across tasks with very different dataset sizes:

| Task     | Train samples used | Max steps   | Epochs (approx) |
|----------|--------------------|-------------|------------------|
| ARC      | 1,119 (all)        | 700         | ~10              |
| MMLU     | 8,000 (subsample)  | 1,500       | ~3               |
| SQuAD    | 8,000 (subsample)  | 1,500       | ~3               |
| TriviaQA | 8,000 (subsample)  | 1,500       | ~3               |
| GSM8K    | 7,473 (all)        | 1,400       | ~3               |

**Rationale**: Subsample large datasets to ~8K to keep total steps comparable.
This ensures checkpoint curves are at similar scales across tasks.

---

## 4. Fine-Tuning Configuration (LoRA)

### 4.1 Why LoRA (Not Full SFT)

1. **Hardware constraint**: Full SFT of an 8B model requires ~60+ GB (model + optimizer
   states + gradients). Does NOT fit on a 32GB RTX 5090.
2. **Paper alignment**: Under LoRA, `lm_head` is frozen ($H_t = H_0$), so the head
   divergence term in Eq. 8 vanishes. This isolates forgetting entirely in backbone
   geometry ($\Delta\rho$ and $\Omega$), giving a cleaner experimental signal.
3. **Practical relevance**: LoRA is the dominant fine-tuning method in production.
   Demonstrating PRISM's utility for LoRA forgetting is highly practical.

### 4.2 LoRA Hyperparameters

| Parameter          | LLaMA-3.1-8B        | Qwen3-8B             |
|--------------------|----------------------|----------------------|
| **LoRA rank (r)**  | 32                   | 32                   |
| **LoRA alpha**     | 64                   | 64                   |
| **Target modules** | q,k,v,o,gate,up,down| q,k,v,o,gate,up,down |
| **LoRA dropout**   | 0.05                 | 0.05                 |
| **Learning rate**  | 2e-4                 | 1e-4                 |
| **LR scheduler**   | cosine               | cosine               |
| **Warmup**         | 5% of max_steps      | 5% of max_steps      |
| **Effective batch** | 16 (bs=2 × ga=8)   | 16 (bs=2 × ga=8)    |
| **Max grad norm**  | 1.0                  | 1.0                  |
| **Weight decay**   | 0.01                 | 0.01                 |
| **Precision**      | bf16                 | bf16                 |
| **Optimizer**      | AdamW (8-bit)        | AdamW (8-bit)        |
| **Max length**     | 512                  | 512                  |
| **Grad checkpoint**| Yes                  | Yes                  |

**VRAM estimate** (bf16 base + LoRA + gradient checkpointing):
- LLaMA-3.1-8B: ~18-20 GB → fits on 32GB RTX 5090
- Qwen3-8B: ~20-22 GB (larger due to 151K vocab embedding) → fits

**Why bf16 not fp16**: Both models were pretrained in bf16. Using fp16 risks NaN
losses due to narrower exponent range, especially with Qwen3's large embedding.

### 4.3 Tokenizer Setup

| Model          | Pad token setup                      | Padding side |
|----------------|--------------------------------------|--------------|
| LLaMA-3.1-8B  | `pad_token = eos_token` (id 128001)  | right        |
| Qwen3-8B      | `pad_token = eos_token` (id 151645)  | right        |

### 4.4 References

- **LLaMA**: Meta `llama-recipes` (GitHub), LLaMA 3.1 paper (arXiv:2407.21783),
  Unsloth notebooks, HF PEFT docs
- **Qwen3**: Qwen3 technical report (arXiv:2505.09388), Qwen readthedocs SFT guide,
  LLaMA-Factory Qwen3 configs

---

## 5. Checkpoint Strategy

### 5.1 Checkpoint Intervals

Save a LoRA adapter checkpoint every `save_steps` steps:

| Task     | max_steps | save_steps | Checkpoints | Eval steps |
|----------|-----------|------------|-------------|------------|
| ARC      | 700       | 50         | 14          | same       |
| MMLU     | 1,500     | 100        | 15          | same       |
| SQuAD    | 1,500     | 100        | 15          | same       |
| TriviaQA | 1,500     | 100        | 15          | same       |
| GSM8K    | 1,400     | 100        | 14          | same       |

Each checkpoint saves only the LoRA adapter (~130 MB), not the full model (~16 GB).
Total storage: ~10 runs × 15 ckpts × 130 MB ≈ **~20 GB**

### 5.2 What Gets Saved

At each checkpoint:
1. **LoRA adapter weights** (`adapter_model.safetensors`, `adapter_config.json`)
2. **Training state** (optimizer, scheduler — for resumability)
3. **Validation loss** on the fine-tuning task (logged by Trainer)

### 5.3 Checkpoint Loading for PRISM Inference

For PRISM inference, each checkpoint is loaded as:
```python
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, ...)
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # merge LoRA → full model for feature extraction
```

This ensures PRISM's `LLMExtractor` sees the standard `model.model` backbone.

---

## 6. Two-Stage Pipeline

### Stage 1: Fine-Tuning (`train_forgetting_multitask.py`)

For each (model, task) pair:
1. Load base model in bf16 with gradient checkpointing
2. Apply LoRA adapter (r=32, target all attention + MLP projections)
3. Load task-specific training data with appropriate formatter
4. Train with HuggingFace Trainer, saving checkpoints at regular intervals
5. Log validation loss at each checkpoint step

```
for model in [llama-3.1-8b, qwen3-8b]:
    for task in [arc, mmlu, squad, triviaqa, gsm8k]:
        train(model, task) → checkpoints/{model}/{task}/checkpoint-{step}/
```

### Stage 2: PRISM Forgetting Inference (`run.py` with `configs/forgetting_multitask.yaml`)

For each (model, task_trained, checkpoint, task_eval) combination:
1. Load base model as Target ($\theta_0$)
2. Load checkpoint as Proxy ($\theta_t$) — merge LoRA adapter
3. Run PRISM feature extraction on eval task data
4. Compute geometric metrics: $\Omega$, $\Delta\rho$, feature alignment error, bound
5. Compute empirical risk: LM loss on eval task
6. Record: `{model, task_trained, step, task_eval, omega, delta_rho, bound, loss_base, loss_ft, delta_risk}`

```
for model in [llama-3.1-8b, qwen3-8b]:
    for task_trained in [arc, mmlu, squad, triviaqa, gsm8k]:
        for checkpoint in sorted(glob(checkpoints/{model}/{task_trained}/checkpoint-*)):
            for task_eval in [arc, mmlu, squad, triviaqa, gsm8k]:
                prism_inference(base_model, checkpoint, task_eval) → results/
```

---

## 7. Evaluation & Metrics

### 7.1 Per-Checkpoint Metrics

For every (model, task_trained, step $t$, task_eval) we record:

| Metric                 | Symbol           | Computation                                    |
|------------------------|------------------|-------------------------------------------------|
| Procrustes Similarity  | $\Omega(Z_0, Z_t)$ | Nuclear norm of cross-moment                 |
| RMS feature scale      | $\rho_0, \rho_t$ | $\frac{1}{\sqrt{n}}\|Z\|_F$                  |
| Scale mismatch         | $\Delta\rho$     | $|\rho_0 - \rho_t|$                            |
| Shape mismatch         | $1 - \Omega$     | Feature manifold distortion                     |
| Feature alignment error| $\delta$         | $K_f\sqrt{(\Delta\rho)^2 + 2\rho_0\rho_t(1-\Omega)}$ |
| PRISM bound            | $Z$              | $\delta + \gamma$ (γ=0 under LoRA)             |
| Base model loss        | $\mathcal{R}_{\theta_0}$ | CE loss of base model on task_eval       |
| Fine-tuned loss        | $\mathcal{R}_{\theta_t}$ | CE loss of checkpoint on task_eval       |
| Empirical forgetting   | $|\Delta\mathcal{R}|$ | $|\mathcal{R}_{\theta_t} - \mathcal{R}_{\theta_0}|$ |

### 7.2 Aggregate Analysis

1. **Forgetting Curves**: Plot $\Omega$ and $|\Delta\mathcal{R}|$ vs training step $t$
   for each (model, task_trained, task_eval) — expect both to degrade over training.

2. **Bound Tightness**: Scatter plot of PRISM bound $Z$ vs empirical $|\Delta\mathcal{R}|$
   across all checkpoints, tasks, models. Compute:
   - Spearman's $\rho$ (rank correlation) — target > 0.9
   - % of points where bound holds ($Z \ge |\Delta\mathcal{R}|$)
   - Mean tightness ratio $|\Delta\mathcal{R}| / Z$

3. **Cross-Task Forgetting Matrix**: Heatmap of $|\Delta\mathcal{R}|$ at the final
   checkpoint, rows = task_trained, cols = task_eval. Diagonal = improvement on
   trained task, off-diagonal = forgetting.

4. **Omega Trajectory**: How $\Omega$ evolves on each eval task as fine-tuning
   progresses on different training tasks.

---

## 8. File & Directory Structure

```
prism/
├── train_forgetting_multitask.py     # NEW: Stage 1 — LoRA fine-tuning
├── infer_forgetting_multitask.py     # NEW: Stage 2 — PRISM inference on checkpoints
├── run_forgetting_multitask.sh       # NEW: Orchestrates Stage 1 + Stage 2
├── configs/
│   └── forgetting_multitask.yaml     # NEW: Config for Stage 2
├── checkpoints/
│   └── forgetting_multitask/
│       ├── llama-3.1-8b/
│       │   ├── arc/checkpoint-50/
│       │   ├── arc/checkpoint-100/
│       │   ├── mmlu/checkpoint-100/
│       │   └── ...
│       └── qwen3-8b/
│           ├── arc/checkpoint-50/
│           └── ...
└── results/
    └── forgetting_multitask/
        ├── llama-3.1-8b/
        │   ├── trained_arc/eval_mmlu/prism_results.json
        │   ├── trained_arc/eval_squad/prism_results.json
        │   └── ...
        └── qwen3-8b/
            └── ...
```

---

## 9. Runtime Estimates

### Stage 1: Fine-Tuning (per model)

| Task     | Steps | Estimated time (RTX 5090) |
|----------|-------|---------------------------|
| ARC      | 700   | ~25 min                   |
| MMLU     | 1,500 | ~50 min                   |
| SQuAD    | 1,500 | ~50 min                   |
| TriviaQA | 1,500 | ~50 min                   |
| GSM8K    | 1,400 | ~45 min                   |
| **Total**| —     | **~3.5 hours/model**      |

Both models: **~7 hours**

### Stage 2: PRISM Inference

Each PRISM evaluation (load checkpoint + extract features + compute metrics):
- ~3 min per (checkpoint, eval_task) pair
- Per training run: 14 checkpoints × 5 eval tasks = 70 evaluations × 3 min = ~3.5 hours
- Total: 10 training runs × 3.5 hours = **~35 hours**

**Overall: ~42 hours** (can be reduced by parallelizing Stage 2 across GPUs
or reducing n_eval to 128 samples)

### Optimization: Reduce PRISM Eval Samples

Using `num_samples=128` instead of 256 halves Stage 2 time to ~17 hours.
The paper shows Spearman $\rho$ is robust with 128 samples.

---

## 10. Implementation Plan

### Step 1: `train_forgetting_multitask.py`
- Extend `train_forgetting.py` to support:
  - Multiple tasks (ARC, MMLU, SQuAD, TriviaQA, GSM8K) via `--task` flag
  - LoRA training (via `peft` library)
  - Model selection via `--model` flag
  - Task-specific data formatting (reuse `prism/data/loaders.py` formatters)
  - Configurable subsample size, max_steps, save_steps
  - Validation loss logging at each checkpoint

### Step 2: `infer_forgetting_multitask.py`
- Load base model once as Target
- Iterate over checkpoints:
  - Load LoRA adapter → merge → Proxy
  - For each eval task:
    - Run PRISM feature extraction + metric computation
    - Compute LM loss
  - Free Proxy model memory
- Output: JSON with all metrics per (checkpoint, eval_task)

### Step 3: `run_forgetting_multitask.sh`
- Orchestrate Stage 1 and Stage 2
- Environment variables for GPU selection, model selection, task selection
- Support running specific stages independently

### Step 4: `configs/forgetting_multitask.yaml`
- Template config consumed by `infer_forgetting_multitask.py`
- Overridable via CLI (model, checkpoints, eval task)

### Step 5: Analysis & Plotting
- Script to aggregate all results into a single DataFrame
- Generate: forgetting curves, scatter plots, heatmaps
- Compute Spearman $\rho$, bound hold rate

---

## 11. Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LoRA not full SFT | LoRA r=32 | Fits 32GB GPU; isolates forgetting in backbone geometry (γ=0) |
| 2 models not 4 | LLaMA + Qwen | Sufficient diversity; reduces total compute from 84h to 42h |
| Subsample large datasets | 8K samples | Equalizes training length across tasks |
| Save LoRA adapters | Not merged models | 130MB vs 16GB per checkpoint; merge at inference time |
| bf16 precision | Not fp16 | Both models trained in bf16; fp16 causes NaN with large vocab |
| 8-bit Adam optimizer | bitsandbytes | Reduces optimizer VRAM by 50%; proven equivalent quality |
| All linear LoRA targets | q,k,v,o,gate,up,down | Standard practice; covers all trainable backbone modules |
| Frozen lm_head | LoRA default | Enables clean Eq. 8 validation with γ=0 |

---

## 12. Dependencies

```
# Core
torch>=2.1
transformers>=4.51.0  # required for Qwen3 support
peft>=0.14.0          # LoRA
bitsandbytes>=0.43    # 8-bit optimizer
datasets
accelerate

# Already in prism
tqdm
numpy
scipy
pyyaml
```

---

## 13. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Qwen3 OOM (151K vocab) | Use `max_length=512`, `bs=2`, gradient checkpointing |
| LoRA adapter incompatible with PRISM loader | Merge adapter before feature extraction |
| MMLU has no standard train split | Use `auxiliary_train` (99K examples), subsample to 8K |
| Forgetting signal too weak | Monitor validation loss; if $\Omega$ stays >0.999, increase LR or rank |
| Checkpoint storage overflow | LoRA adapters only (~130MB each); total ~20GB |
| Tokenizer mismatch between train/eval | Reuse same tokenizer from base model throughout |
