# PRISM: supplementary code

Code for the paper *PRISM: A Geometric Risk Bound for Decomposing Drift into
Scale, Shape, and Head*. The package contains

- the pipelines of the paper's experiments: the quantization grid and the
  LoRA checkpoint series (Sec. 4.2), and the forgetting experiments
  (Sec. 4.5, Appendices H.2, H.3, I, J; `scripts/response/forgetting/`);
- the launch scripts of the three-seed round of Appendix F.2 and of a
  fixed-weight regularizer comparison (`scripts/rounds/`);
- the code behind every table of the author response (`scripts/response/`).

No result files are shipped. Every table listed below is produced by running the scripts.


## Layout

```
.
├── README.md
├── requirements.txt              environment of run_quantization.sh and run_forgetting.sh
├── requirements-response.txt     environment of the experiments in scripts/response/
├── install.sh                    installs requirements.txt (and GPTQModel)
│
├── run_quantization.sh           quantization grid (Sec. 4.2)
├── run_quantization.py           runner called by run_quantization.sh
├── run_forgetting.sh             single-seed λ sweeps (anchor-CE, shape regularizer); λ = 0: single-run LoRA checkpoint series (Sec. 4.2)
├── run_forgetting_one.sh         one configuration of the sweep
├── train_forgetting.py           LoRA training with PRISM evaluation at every checkpoint
├── similarity_baselines.py       linear CKA and SVCCA, used by train_forgetting.py
│
├── configs/
│   ├── quantization.yaml         default configuration of run_quantization.py
│   ├── quantization_matrix.yaml  models, benchmarks and variants of the grid
│   └── model_revisions.yaml      pinned Hugging Face revisions
├── prism/                        library
│   ├── core/         metrics.py, bounds.py
│   ├── data/         loaders.py
│   ├── models/       extractors.py
│   └── experiments/  base.py, quantization.py
└── scripts/
    ├── quantization_matrix.py    expands quantization_matrix.yaml
    ├── sweep_table.py            table of the λ sweeps of run_forgetting.sh (no GPU)
    ├── rounds/                   three-seed round of Appendix F.2; fixed-weight regularizer comparison
    │   ├── run_three_seed_round.sh, three_seed_round.py, size_study.py   (Appendix F.2)
    │   ├── similarity_scores.py          CKA, SVCCA, Procrustes distance (Appendix F.1)
    │   ├── common_quant.py               variant manifest and model loading
    │   ├── data/variant_manifest.csv     the 12 + 11 quantized variants (no result columns)
    │   ├── run_regularizer_baselines.sh, regularizer_table.py           (fixed-weight comparison)
    │   ├── train_forgetting_baselines.py trainers: none, trace, replay, l2sp, ewc,
    │   │                                 layer_freeze, lwf, hint_mse
    │   └── train_forgetting_multitask.py training loop imported by the baselines script
    └── response/                 one folder per result of the response, each with a README
        ├── forgetting/           unified protocol of Sec. 4.5 and Appendices H.2, H.3, I, J: selection
        │                         rule, Table 3, L2-SP, EWC, LwF, KD, layer freezing
        ├── lora_similarity/      LoRA checkpoint ranking against 1-CKA and 1-SVCCA
        ├── screen_n8/            eight-sequence screen (two GPU passes, see its README)
        ├── layer_profile/        layer-wise and q/k/v profile
        ├── head/                 head experiments 1 and 2
        └── scale/                outlier-channel preservation on and off
```


## Installation

Two environment files are provided.

| file | for | how to install |
|---|---|---|
| `requirements.txt` | `run_quantization.sh`, `run_forgetting.sh` (the environment those runs used) | `bash install.sh` |
| `requirements-response.txt` | everything under `scripts/response/` (the environment those runs used: Python 3.12.3, torch 2.11.0, transformers 5.2.0, accelerate 1.12.0, peft 0.18.1, datasets 4.5.0, bitsandbytes 0.50.2, gguf 0.17.1), except the GPTQ pass of the eight-sequence screen | see the header of the file |
| `requirements.txt`, with the GPTQ loader that `install.sh` adds | `scripts/rounds/` is to be run in this environment: three of the 23 variants of `scripts/rounds/data/variant_manifest.csv` are GPTQ repositories and need a GPTQ loader; the trainers under `scripts/rounds/` need none | `bash install.sh` |

`install.sh` installs torch first, then `requirements.txt`, then GPTQModel
with build isolation switched off; the reasons are in the header of
`requirements.txt`. `requirements-response.txt` lists no GPTQ loader. The
eight-sequence screen is therefore run in two passes: the 20 variants that
need no GPTQ loader in the `requirements-response.txt` environment, and the
three GPTQ variants in a separate environment that has one
(`run_screen.sh gptq`; details in `scripts/response/screen_n8/README.md`). In
the reported run the three GPTQ variants were scored in such a separate
environment.

Gated repositories (for example the Llama family) need an accepted licence
and a Hugging Face login.


## Where each result comes from

All commands are run from this directory. Each folder's README gives
definitions.

### Response tables

| result | folder | command |
|---|---|---|
| Main forgetting table (Table 3) and the selection rule (four cells, three seeds) | `scripts/response/forgetting/` | `bash scripts/response/forgetting/launch_main_table.sh out/unified` |
| LwF and representation KD (Llama-3.1-8B/TruthfulQA) | `scripts/response/forgetting/` | `bash scripts/response/forgetting/launch_extra_baselines.sh out/unified` |
| Layer freezing (four cells, three seeds) | `scripts/response/forgetting/` | `bash scripts/response/forgetting/launch_layer_freezing.sh out/unified` |
| LoRA checkpoint ranking over three seeds (Sec. 4.2, Appendix F.3) | `scripts/response/lora_similarity/` | `bash scripts/response/lora_similarity/launch_lora_table.sh out/lora_series` |
| Eight-sequence screen | `scripts/response/screen_n8/` | `bash scripts/response/screen_n8/run_screen.sh run`, then `... run_screen.sh gptq` in an environment with a GPTQ loader, then `... run_screen.sh analyze` |
| Single-seed λ sweep (Llama-3.1-8B/TruthfulQA, shape regularizer) | top level | `bash run_forgetting.sh` (this cell only: `MODELS_ORDER=llama TASKS_ORDER=truthfulqa REPLAY_LAMBDAS="" bash run_forgetting.sh`), then `python scripts/sweep_table.py --root checkpoints` |
| Layer-wise and q/k/v profile | `scripts/response/layer_profile/` | `bash scripts/response/layer_profile/run_layer_profile.sh` |
| Head experiment 1 (hybrids, 20 cells) | `scripts/response/head/` | `STAGE=swap bash scripts/response/head/run_head.sh`, then `STAGE=analyze` (prints both analyses and needs the outputs of experiments 1 and 2) |
| Head experiment 2 (head precision, 210 cells) | `scripts/response/head/` | `STAGE=precision bash scripts/response/head/run_head.sh`, then `STAGE=analyze` (prints both analyses and needs the outputs of experiments 1 and 2) |
| Scale intervention | `scripts/response/scale/` | `bash scripts/response/scale/run_scale.sh` |

The λ sweep row refers to single-seed runs outside the unified protocol
(cross-entropy on the anchor set, i.e. anchor-CE, and the shape regularizer;
the library's default trainer at λ = 0): `run_forgetting.sh`
writes one `prism_forgetting_metrics.json` per (model, task, method, λ) under
`checkpoints/forgetting_{shape,replay}_lam<λ>/<model>/<task>/`, with the PRISM
quantities and \|ΔR\| on the five benchmarks at every checkpoint.
`scripts/sweep_table.py` prints the table from these files at the final step
(300):

- target loss is the trainer's evaluation loss (`eval_loss`) on the whole
  held-out slice of the fine-tuning task (163 rows for TruthfulQA). This slice contains the 32 rows the regularizers
  read as their anchor set. `run_forgetting.sh` passes
  `--eval_keep_anchor_rows` to obtain it. The unified protocol
  (`scripts/response/forgetting/`) instead reports the loss on the rows
  outside the anchor set, so target losses of the two are not comparable;
- downstream \|ΔR\| is the unweighted mean of \|ΔR\| over the five
  benchmarks;
- each value is a single run (one seed).

### Paper experiments

| experiment | command | output |
|---|---|---|
| Quantization grid (Sec. 4.2): 7 models × 5 benchmarks × the variants in `configs/quantization_matrix.yaml`, 512 sequences, seed 42 | `bash run_quantization.sh` (subset: `MODELS="llama31_8b_base,qwen3_8b_base" DATASETS="mmlu,arc" bash run_quantization.sh`; `DRY_RUN=1` prints the commands) | `results/quantization/` |
| LoRA checkpoint series, single run (Sec. 4.2): unregularized LoRA with the library's default trainer, two models × two tasks, 300 steps, a checkpoint every 25 steps | `REPLAY_LAMBDAS="0" SHAPE_LAMBDAS="" bash run_forgetting.sh` | `checkpoints/forgetting_replay_lam0/` |
| Forgetting experiments (Sec. 4.5, Appendices H.2, H.3, I, J; Table 3, Figs. 4 and 8): unified protocol, four cells, seeds 42, 43, 44 | the commands below | `out/unified/` |

**Forgetting experiments.** The forgetting results of Sec. 4.5 and Appendices
H, I, J are produced by `scripts/response/forgetting/` (protocol and
definitions in its README): every method in one training loop, and each
weight λ selected by one rule on the target-task loss. In order:

```bash
bash scripts/response/forgetting/launch_main_table.sh         out/unified  # no regularization, anchor-CE, generic replay, shape regularizer
bash scripts/response/forgetting/launch_appendix_baselines.sh out/unified  # L2-SP, EWC
bash scripts/response/forgetting/launch_extra_baselines.sh    out/unified  # LwF, representation KD
bash scripts/response/forgetting/launch_layer_freezing.sh     out/unified  # layer freezing (no weight to select)
python scripts/response/forgetting/eval_target_disjoint.py --out_root out/unified --model llama  # then --model qwen
python scripts/response/forgetting/select_lambda.py  --out_root out/unified
python scripts/response/forgetting/penalty_report.py --out_root out/unified
```

Each launcher trains its job list and then runs `eval_target_disjoint.py`
(target-task loss on the held-out rows outside the anchor set, runs already
evaluated are skipped); the last two commands print the tables from all runs
present. In `train_forgetting.py`, `--lambda_replay` is the weight of the
cross-entropy penalty: anchor-CE with `--replay_source task`, generic replay
with `--replay_source generic`. `run_unified.py --save_steps 25` records the
trajectories of the figures.

`run_forgetting.sh` with its default grids runs single-seed λ sweeps outside this
protocol: cross-entropy on the anchor set (anchor-CE) at λ ∈ {0, 0.001,
0.005, 0.01, 0.05, 0.1} and the shape regularizer at λ ∈ {0.01, 0.05, 0.1,
0.5, 1}, with the library's default trainer at λ = 0 (output
`checkpoints/forgetting_*/`, table by `scripts/sweep_table.py`).

A variant that raises an exception during loading, extraction or the metric
computation is skipped, and the runner continues with the next variant.
Entries of `configs/quantization_matrix.yaml` that are skipped at run time
because they cannot be loaded:

- the eight `gptq:Efficient-ML/*` entries (four under `qwen3_8b_instruct`,
  four under `qwen3_8b_base`): the repository is not recognised as a model by
  `transformers`;
- `Q3_K_M` and `Q2_K` under `qwen3_8b_instruct`: `Qwen/Qwen3-8B-GGUF` has no
  file for these two tags;
- `gptq:RedHatAI/Qwen3-8B-quantized.w4a16`: it needs the `compressed-tensors`
  package, which `requirements.txt` does not install.

`run_quantization.py` prints one `[skipped]` line per variant that produced
no result, and `run_quantization.sh` lists these variants in its final
summary. `run_quantization.sh` needs bash 4 or later.

### Three-seed rounds

| round | command |
|---|---|
| Appendix F.2, reference-set size: seeds 43, 44, 45; sizes 8, 32, 128, 512; both families | `bash scripts/rounds/run_three_seed_round.sh run` (GPU), `bash scripts/rounds/run_three_seed_round.sh table` (no GPU) |
| Fixed-weight regularizer comparison (not the Appendix I table): Llama-3.1-8B/TruthfulQA, seeds 42, 43, 44 | `STAGE=sweep bash scripts/rounds/run_regularizer_baselines.sh`, then `STAGE=seeds`, then `STAGE=table` |
| Appendix I under the unified protocol: four cells, seeds 42, 43, 44 | `bash scripts/response/forgetting/launch_main_table.sh out/unified`, then `launch_appendix_baselines.sh` (L2-SP, EWC), `launch_extra_baselines.sh` (LwF, representation KD) and `launch_layer_freezing.sh` in the same folder, each with `out/unified` |

Notes on the two rounds:

- `size_study.py` prints the table of Appendix F.2 from the CSVs of
  `three_seed_round.py`; `scripts/response/screen_n8/analyze_screen.py
  --size-round out/rounds/three_seed` reads the same CSVs.
- `similarity_scores.py` (Appendix F.1) is not a three-seed round. It computes
  CKA, SVCCA and the Procrustes distance at seed 42 and joins the risk gaps
  and the bounds from the result tables that `run_quantization.py` writes
  (one CSV per model and dataset). `--results-csv` takes those files, a
  directory that holds them, or a glob. The risk gap it joins is the column
  `|AdR|` (the difference of the means over sequences of each sequence's mean
  answer-token loss), and `|dR|` where `|AdR|` is empty. Its functions
  `linear_cka` and `svcca` are the ones the three-seed round uses.
- Fixed-weight comparison, table script. `regularizer_table.py` lists every run it finds,
  so every configuration of the sweep is listed. It prints the mean ± sample
  sd over seeds for the six configurations of the `seeds` stage
  and for every other configuration with at least two seeds, and a "matched"
  block that selects, per method, the configuration whose
  target-task loss is closest to the shape regularizer's among the seed-42 runs
  of the whole sweep. A table covers one model; `--model` selects it when the
  output root holds runs of more than one.
- Fixed-weight comparison, L2-SP. The sweep covers L2-SP at λ ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹};
  the `seeds` stage runs L2-SP at λ = 0.01.
  Appendix I reports L2-SP and EWC under the unified protocol of
  `scripts/response/forgetting/` (last row of the table above; job list
  `jobs_appendix_baselines.txt`, definitions in that folder's README).
- Fixed-weight comparison, target-task loss. The column is the trainer's `eval_loss`.
  `run_regularizer_baselines.sh` passes `--eval_keep_anchor_rows`, so it is
  computed on the whole held-out slice of the fine-tuning task, anchor rows
  included.
- Fixed-weight comparison, resuming. An interrupted L2-SP or EWC run cannot be resumed
  (its reference point, the adapter initialisation, is not stored); the
  trainer stops with a message and the run has to be restarted.
- Model id. `run_forgetting_one.sh` and `run_regularizer_baselines.sh` load
  `meta-llama/Meta-Llama-3.1-8B`, the launchers under `scripts/response/` load
  `meta-llama/Llama-3.1-8B`. `configs/model_revisions.yaml` pins both names
  to the same commit. The `<model>` component of an output path is the
  lower-cased last part of the id (`meta-llama-3.1-8b` and `llama-3.1-8b`);
  `scripts/sweep_table.py` and `regularizer_table.py` read whichever
  directory is present.


## Protocol notes

These notes state how the code behaves where a reader could otherwise assume
something else.

**(a) Anchor set.** The regularized methods read 32 sequences of the
fine-tuning task. They are drawn from the task's held-out slice (the last 20%
of the split; the first 80% is the training slice): the slice is shuffled with
`seed + 1000` and the first 32 rows are taken. At a given seed the anchor set
is the same for every method; each seed has its own draw. Generic replay does
not read the anchor set (its penalty is computed on WikiText rows). The
target-task loss reported for the unified protocol is computed by
`scripts/response/forgetting/eval_target_disjoint.py` on the held-out rows
outside the run's anchor set, for every method. The trainers' own `eval_loss`
is computed on the same rows by default and on the whole held-out slice with
`--eval_keep_anchor_rows`; `run_forgetting.sh`,
`run_regularizer_baselines.sh` and the launcher of the LoRA checkpoint ranking pass that flag, the launcher of
the unified protocol does not. The flag changes the evaluation only.

**(b) Two unregularized references.** `train_forgetting.py --lambda_shape 0
--lambda_replay 0` uses the library's default `Trainer`: no anchor set, no
penalty. This is the unregularized run of `run_forgetting.sh`
(λ = 0; the LoRA checkpoint series of Sec. 4.2) and the trainer of the LoRA
checkpoint ranking (`scripts/response/lora_similarity/`). The unregularized reference of the
unified protocol (`scripts/response/forgetting/`, method `noreg_sameloop`) is
the regularized trainer at `--lambda_shape 1e-9`, so that it shares the
training loop of the other methods. Results of the two references are not
interchangeable.

**(c) Loss scaling.** Under the library versions of
`requirements-response.txt`, with G = 8 gradient-accumulation steps, the
gradient accumulated by the regularized trainer for one optimizer step is
(1/G) · [sum of the G micro-batch cross-entropy gradients + λ · penalty
gradient]. This holds for every regularized method and for the same-loop
reference. "λ · P" in the tables is λ times the logged penalty P. The `loss`
this trainer logs is the sum of the G micro-batch mean cross-entropies of an
optimizer step, without the penalty.

**(d) λ search.** The grid is {1, 5} × 10^k up to 10^4, the same for every
method. Not every grid value was run. The search went upward from scouting
runs at seed 42 along the grid until the target-loss budget was exceeded;
three seeds were then run at the selected value and, for the baselines, at
their least-forgetting value within budget. The job lists in
`scripts/response/forgetting/` are the runs behind the tables, and the folder
README describes the search step by step.

**(e) Standard deviations.** The forgetting tables of the unified protocol and
the eight-sequence screen use the population standard deviation (divisor n).
The LoRA checkpoint-ranking table and the two rounds of `scripts/rounds/` use the sample
standard deviation (divisor n − 1). Each script prints which one it uses.

**(f) Entry points.** The shape regularizer runs through two entry points that
share the training loop: `train_forgetting.py --lambda_shape` and method
`trace` of `scripts/rounds/train_forgetting_baselines.py`. LwF,
representation KD, L2-SP, EWC and layer freezing (`lwf`, `hint_mse`, `l2sp`,
`ewc`, `layer_freeze --same_loop`) run through the baselines script.
`protocol.py` in `scripts/response/forgetting/` records which entry point each
method of the unified protocol uses.

**(g) Remote code and revisions.** No loader enables `trust_remote_code`
unless `PRISM_TRUST_REMOTE_CODE=1` is set. A repository listed in
`configs/model_revisions.yaml` is loaded at the revision given there, and the
loader prints that revision. A repository that is not listed is loaded at its
head, with a warning on stderr; `PRISM_UNPINNED=1` ignores the file for every
repository. For the repositories the response experiments load, the pinned
hashes are the snapshots those runs loaded (listed in the header of the
file). The remaining entries are the revisions that were current when the
file was written. The trainers record the revision in the `experiment` block
of their output.

**(h) Hardware and runtime.** Every experiment of the paper runs on a single
GPU (NVIDIA RTX 5090, 32 GB, or RTX PRO 6000, 96 GB), with PyTorch 2.10 or
2.11 and transformers 5.2 (the two requirements files). The layer profile of the response holds two
8B models in BF16 on one GPU at the same time, which does not fit in 32 GB;
its reported runs used one 96 GB GPU per job. Run times recorded in the output files
of the response experiments: layer profile, 100 to 112 seconds per variant;
head experiments, about 45 and 50 GPU-minutes of per-cell time. The analysis
scripts need no GPU.

**(i) Seeds.** `--seed` of the trainers sets the initial weights of the LoRA
adapter (as of this release; `adapter_init_seeded` in the `experiment` block
of the output), the seed of the library `Trainer` (order of the training
data, dropout), the anchor draw (`seed + 1000`) and the draw of the benchmark
sequences of the PRISM evaluation.

**(j) Optimizer.** The trainers use the paged 8-bit AdamW of bitsandbytes
(`paged_adamw_8bit`) when bitsandbytes can be imported, and torch AdamW
otherwise; weight decay 0.01, cosine schedule with 5% warm-up, gradient
clipping at 1.0. The trainers print the
optimizer and record it, with these settings, the revision, `replay_source`,
`eval_keep_anchor_rows` and the number of evaluation rows, in the
`experiment` block of `prism_forgetting_metrics.json`. A negative λ is
rejected.

**Other points.**

- Benchmark subsets: the loader shuffles the split with the run's seed and
  keeps the first n sequences (512 in the quantization grid, 256 per benchmark
  in the forgetting runs).
- Dataset identifiers are the current repository names
  (`truthfulqa/truthful_qa`, `mandarjoshi/trivia_qa`, ...); dataset revisions
  are not pinned.
- `prism/__init__.py` caps the CPU threads of PyTorch at 8
  (`PRISM_CPU_THREADS` changes it, 0 leaves the PyTorch default).
- GGUF variants are loaded from the repositories in
  `configs/quantization_matrix.yaml`, GPTQ variants from the repositories
  listed there, and bitsandbytes variants are quantized from the target
  checkpoint at load time.
