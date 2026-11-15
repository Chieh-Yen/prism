# Unified forgetting protocol

Compares the shape regularizer with no regularization, anchor-CE, generic
replay, LwF and representation KD when every method is tuned by one rule that
reads only the target-task loss. The same rule is applied to L2-SP and EWC,
the weight-space baselines of Appendix I; the revised Appendix I reports them
under this protocol.

| file | purpose |
|---|---|
| `protocol.py` | shared definitions: cells, seeds, methods, lambda grid, the hyper-parameters every run is launched with, the run directory layout |
| `run_unified.py` | launcher: one training run per (method, model, task, lambda, seed) |
| `eval_target_disjoint.py` | target-task loss of each run on the evaluation rows outside its 32 anchor rows (field `target_loss`, with the row counts) |
| `select_lambda.py` | the selection rule, the tables, and `--emit` (the next runs of the lambda search) |
| `penalty_report.py` | weighted penalty lambda · P at the selected lambda |
| `jobs_main_table.txt`, `jobs_extra_baselines.txt`, `jobs_appendix_baselines.txt` | the runs behind the tables, one `method model task lambda seed` per line |
| `launch_main_table.sh` | main table: trains `jobs_main_table.txt`, evaluates, prints |
| `launch_extra_baselines.sh` | LwF and representation KD on Llama-3.1-8B/TruthfulQA, with lambda · P; needs the Llama-3.1-8B/TruthfulQA reference and shape-regularizer runs of the main table in the same directory |
| `launch_appendix_baselines.sh` | L2-SP and EWC (Appendix I baselines) on the four cells, with lambda · P; needs the reference and shape-regularizer runs of the main table in the same directory |
| `search_lambda.sh` | the lambda search as a loop, from a list of starting runs |

## Protocol

- **Cells**: Llama-3.1-8B and Qwen3-8B-Base, each fine-tuned on TruthfulQA and
  on BBQ; seeds 42, 43, 44.
- **Training** (every run): LoRA r = 32, alpha = 64, dropout 0.05 on the
  attention and MLP projections; 300 optimizer steps; learning rate 1e-5 with
  5% warm-up and cosine decay; batch size 2 with 8 accumulation steps;
  maximum length 512; one checkpoint and one evaluation at step 300. The
  argument list is `protocol.COMMON_ARGS`. Fixed in the trainers and not
  exposed as flags: weight decay 0.01, gradient clipping at 1.0, bf16, and
  the optimizer, which is the paged 8-bit AdamW of bitsandbytes when that
  package can be imported and torch AdamW otherwise.
- **Anchor set**: 32 sequences of the fine-tuning task, drawn from its
  evaluation split with seed `run seed + 1000` (so each seed has its own
  anchor set, shared by all methods of that seed), visited once per optimizer
  step.
- **Methods**

  | name | what is added to the loss | command |
  |---|---|---|
  | `noreg_sameloop` | nothing: the reference, trained in the same loop as the other methods | `train_forgetting.py --lambda_shape 1e-9` |
  | `shape` | λ · (1 − Ω_I) on the anchor set | `train_forgetting.py --lambda_shape λ` |
  | `anchor_ce` | λ · cross-entropy on the anchor set | `train_forgetting.py --lambda_replay λ --replay_source task` |
  | `generic_replay` | λ · cross-entropy on 32 WikiText sequences | `train_forgetting.py --lambda_replay λ --replay_source generic` |
  | `lwf` | λ · KL to the frozen base model's next-token distribution on the anchor set | `train_forgetting_baselines.py --method lwf --lambda_reg λ` |
  | `hint_mse` | λ · MSE to the base model's final-layer features on the anchor set | `train_forgetting_baselines.py --method hint_mse --lambda_reg λ` |
  | `trace` | the shape regularizer as implemented in the baselines script | `train_forgetting_baselines.py --method trace --lambda_reg λ` |
  | `l2sp` | λ · Σ (θ − θ₀)² over the adapter weights, θ₀ their initial values | `train_forgetting_baselines.py --method l2sp --lambda_reg λ` |
  | `ewc` | λ · Σ F · (θ − θ₀)², F the diagonal Fisher estimate | `train_forgetting_baselines.py --method ewc --lambda_reg λ` |

  The reference uses the regularized trainer class with a penalty weight of
  1e-9. The penalty 1 − Ω_I is at most 2, so the weighted penalty is at most
  2e-9 nats; `--lambda_shape 0` would instead select the library's default
  trainer, whose loop differs.
- **L2-SP and EWC**: both penalties are computed on the trainable adapter
  weights, once per optimizer step like the other penalties, and do not read
  the anchor set during training. For EWC the Fisher diagonal is estimated
  before training, at the adapter initialisation, on the 32 anchor sequences:
  the gradient of the mean answer-only cross-entropy of each anchor batch of
  8 sequences is squared element-wise, the squares are averaged over the four
  batches, and the result is divided by its mean over all trainable entries
  (unit mean), so that λ of EWC and λ of L2-SP are on the same scale.
- **Generic replay**: the 32 sequences are rows of WikiText-2 (raw, train
  split). 1,280 rows are drawn after shuffling the split with seed
  `run seed + 1000`, and the first 32 of them with at least 64 tokens are
  kept, truncated at 512 tokens. The rows have no prompt, so the penalty is
  the next-token cross-entropy over all their non-padding tokens; anchor-CE
  scores the answer tokens only. Generic replay does not read the anchor set.
- **Logged quantities**: the trainers evaluate the penalty once per optimizer
  step and log its unweighted mean over each window of 10 optimizer steps, as
  `shape_loss` (shape regularizer and reference),
  `replay_ce` (anchor-CE and generic replay), `lwf`, `hint_mse`, `l2sp` and
  `ewc`, in
  `checkpoint-300/trainer_state.json`. The `loss` the regularized trainers
  log there (`train_loss` in `prism_forgetting_metrics.json`) is the sum over
  the 8 micro-batches of an optimizer step of the micro-batch mean
  cross-entropy, averaged over the logging window, without the penalty; it is
  not on the scale of the loss the library's default trainer logs.
- **Forgetting**: mean over five downstream benchmarks (ARC, GSM8K, MMLU,
  SQuAD, TriviaQA; 256 evaluation sequences each) of |ΔR|, the absolute
  change of the answer-only cross-entropy from the base model to the
  fine-tuned model, at step 300.
- **Target-task loss**: answer-only cross-entropy on the fine-tuning task's
  evaluation rows outside the run's anchor set, token-weighted
  (`eval_target_disjoint.py`, field `target_loss`). It is computed on these
  rows for every method, generic replay included. The trainer's own
  `eval_loss` is not used.
- **Rule**: budget = mean target-task loss of the reference over its three
  seeds + 0.01 nats. Each method is reported at the largest grid value of λ
  at which it has three seeds and its three-seed mean target-task loss is
  within the budget. Forgetting is not read by the rule.
- **Grid**: λ ∈ {1, 5} × 10^k from 10⁻⁴ up to 10⁴, the same for every method.
- **Spread**: "± sd" is the population standard deviation over the three
  seeds (divisor 3).

## Commands

All commands are run from the supplementary root. Training needs one GPU that
holds an 8B model in bf16; runs are executed one after another, finished runs
are skipped, and several launchers with disjoint job lists can run on
different GPUs. The analysis scripts read JSON only and take seconds on a CPU.
Knobs of the four `.sh` launchers: the output root is their first argument,
`PYTHON` selects the interpreter (default `python`), and the GPU is the one
`CUDA_VISIBLE_DEVICES` exposes.

```bash
# main table (251 training runs), least-forgetting comparison, budget sensitivity
bash scripts/response/forgetting/launch_main_table.sh out/unified

# LwF and representation KD on Llama-3.1-8B/TruthfulQA (39 more runs), with lambda · P.
# out/unified must already hold the Llama-3.1-8B/TruthfulQA runs of jobs_main_table.txt
# for noreg_sameloop (3 runs) and shape (16 runs), with their target-task loss; the
# other runs of the main table are not needed for this table.
bash scripts/response/forgetting/launch_extra_baselines.sh out/unified

# Appendix I baselines L2-SP and EWC on the four cells (95 more runs), with lambda · P.
# out/unified must already hold the runs of jobs_main_table.txt for noreg_sameloop
# (12 runs) and for shape and trace (74 runs), with their target-task loss.
bash scripts/response/forgetting/launch_appendix_baselines.sh out/unified

# analysis only, on existing runs
python scripts/response/forgetting/select_lambda.py  --out_root out/unified
python scripts/response/forgetting/penalty_report.py --out_root out/unified

# a single run, or the commands without running them
python scripts/response/forgetting/run_unified.py --out_root out/unified \
    --methods shape --models llama --tasks truthfulqa --lambdas 1000 --seeds 42
python scripts/response/forgetting/run_unified.py --out_root out/unified \
    --jobs scripts/response/forgetting/jobs_main_table.txt --dry_run
```

## How the lambda search was carried out

The job lists are the result of a search, not a full sweep of the grid:

1. The reference was trained with three seeds per cell.
2. Each method started from a few grid values. Some were three-seed runs at
   fixed values that existed before the search; others were single runs at
   seed 42 placed by hand.
3. The loop of `search_lambda.sh` was then iterated until it requested
   nothing. In each round `select_lambda.py --emit` requests, per cell and
   method, seed 42 at the grid value just above the largest value within
   budget; once that value exists and is out of budget, it requests seeds 43
   and 44 at the largest value within budget. Budget membership is always
   judged on the mean over the seeds available, so a value can leave the
   budget when its seeds 43 and 44 arrive, and the boundary then moves.
4. For every method other than the shape regularizer, seeds 43 and 44 were
   also run at the within-budget value with the least forgetting. These runs
   do not enter the rule; they supply the "least-forgetting λ" comparison.

All tables are computed from the runs in the job lists by the rule above,
whatever order they were run in.

L2-SP and EWC were searched by the same loop (steps 3 and 4), from three-seed
runs at fixed values; `jobs_appendix_baselines.txt` holds these runs and the
runs the loop added (95 in total).

Properties of the search:

- The loop only adds the grid value adjacent to the current boundary. Grid
  values below or between a method's starting values, and beyond the first
  out-of-budget value above the boundary, were run only if they were among
  the starting values (the starting values were mostly powers of ten).
  `select_lambda.py` lists, for every cell and method, each λ with its number
  of seeds, mean target-task loss and mean forgetting.
- On Qwen3-8B-Base/BBQ the shape regularizer and anchor-CE are still within
  budget at the top of the grid, λ = 10⁴, and are reported there.
- The shape-regularizer points at λ = 0.1 and λ = 10 were trained with the
  baselines script (`trace`), the others with `train_forgetting.py`. The two
  entry points share the training loop. Every shape point selected at the
  0.01-nat budget is a `train_forgetting.py` run; a `trace` point can be
  selected in the check at another budget.
- The budget is applied to the three-seed mean of the target-task loss, not
  seed by seed.

`search_lambda.sh OUT_ROOT START_JOBS` repeats the procedure from a list of
starting runs of your choice.

## Output

- `select_lambda.py` on the runs of `jobs_main_table.txt` and
  `jobs_extra_baselines.txt` prints, per model and task, the forgetting (mean ±
  population sd over three seeds), the target-task loss and the selected λ of
  no regularization, anchor-CE, generic replay and the shape regularizer,
  followed by the within-budget λ where each baseline forgets least and the
  selections under budgets of 0.005 and 0.02 nats.
- `select_lambda.py` on the runs of `jobs_main_table.txt` and
  `jobs_appendix_baselines.txt` prints the same quantities for L2-SP, EWC and
  the shape regularizer (same rule, grid, budget, seeds and spread).
- `penalty_report.py` prints, per method, the selected λ, λ · P, the forgetting
  and the target-task loss. P is the unweighted penalty averaged over optimizer
  steps 291 to 300 (the last entry of the trainer's log) and over the three
  seeds. It lists the shape regularizer, LwF and representation KD, and L2-SP
  and EWC in a second table. The cross-entropy penalties of anchor-CE and
  generic replay are not listed by it; they are logged under the key
  `replay_ce` in `checkpoint-300/trainer_state.json` of each run.
