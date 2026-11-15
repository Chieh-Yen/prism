#!/usr/bin/env python3
"""Target-task loss on the evaluation rows outside the anchor set.

The 32 anchor sequences of a run are drawn from the fine-tuning task's
evaluation split (``validation[80%:]`` for TruthfulQA, ``test[80%:]`` for BBQ):
the split is shuffled with seed ``run seed + 1000`` and the first 32 rows are
taken.  The shape regularizer, anchor-CE, LwF and representation KD compute
their penalty on these rows during training; the unregularized reference
computes the shape penalty on them with weight 1e-9.  Generic replay does not
read them (its penalty is computed on WikiText rows); L2-SP does not read them
either (its penalty is computed on the adapter weights), and EWC reads them
once, before training, to estimate its Fisher diagonal.  The target-task loss of
the unified protocol is computed on the remaining rows, for every method
alike, generic replay included.

For every finished run below ``--out_root`` this script loads the LoRA adapter
saved at step 300, rebuilds the run's 32 anchor rows, and records

``target_loss``  the answer-only cross-entropy (next-token loss on the answer
                 tokens, prompt tokens masked) of the fine-tuned model on the
                 rows of the evaluation split outside the run's anchor set,
                 token-weighted: the summed loss of the answer tokens of
                 those rows divided by their number;
``n_rows``       number of rows of the evaluation split;
``n_anchor``     number of anchor rows (32);
``n_eval_rows``  number of rows ``target_loss`` is computed on
                 (``n_rows - n_anchor``).

``select_lambda.py`` reads ``target_loss``.  It is not the trainer's
``eval_loss``, which is aggregated by the library and covers the anchor rows
when the trainer is started with ``--eval_keep_anchor_rows``.

The result is a JSON list, one record per run, written to
``<out_root>/target_loss_disjoint_<model>.json``.  Runs already in that file
are skipped, so the script can be re-run as more runs finish.

Example
-------
    python scripts/response/forgetting/eval_target_disjoint.py \\
        --out_root out/unified --model llama
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Set, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import protocol as P

sys.path.insert(0, str(P.SUPP_ROOT))
from prism.experiments.quantization import pinned_revision  # noqa: E402
from train_forgetting import (  # noqa: E402
    AnswerOnlyDataCollator, TASK_CONFIGS, _load_hf_dataset, build_dataset,
)

N_ANCHOR = 32
# --max_length of the runs (protocol.COMMON_ARGS).
MAX_LENGTH = int(P.COMMON_ARGS[P.COMMON_ARGS.index("--max_length") + 1])


def anchor_rows(task: str, seed: int, n_rows: int) -> Set[int]:
    """Row indices, inside the evaluation split, of a run's anchor set."""
    raw = _load_hf_dataset(task, "eval")
    if len(raw) != n_rows:
        raise RuntimeError(f"{task}: evaluation split has {len(raw)} rows, tokenized set {n_rows}")
    rows = (raw.add_column("_row", list(range(len(raw))))
            .shuffle(seed=seed + 1000).select(range(N_ANCHOR))["_row"])
    return set(rows)


@torch.no_grad()
def answer_ce_per_row(model, dataset, collator, device: str,
                      batch_size: int = 8) -> List[Tuple[float, int]]:
    """(summed answer-token loss, number of answer tokens) for every row."""
    per_row: List[Tuple[float, int]] = []
    for start in range(0, len(dataset), batch_size):
        feats = [dict(dataset[j]) for j in range(start, min(start + batch_size, len(dataset)))]
        batch = {k: v.to(device) for k, v in collator(feats).items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = batch["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none").view(shift_labels.shape)
        mask = shift_labels != -100
        per_row += list(zip((loss * mask).sum(1).tolist(), mask.sum(1).tolist()))
    return per_row


def token_mean(rows: List[Tuple[float, int]]) -> float:
    return sum(r[0] for r in rows) / max(sum(r[1] for r in rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, type=Path,
                    help="directory run_unified.py wrote the runs to")
    ap.add_argument("--model", required=True, choices=list(P.MODELS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_file = args.out_root / f"target_loss_disjoint_{args.model}.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else []
    done = {(r["method"], r["lam"], r["seed"], r["task"]) for r in results}
    runs = [r for r in P.iter_runs(args.out_root)
            if r.model == args.model
            and (r.method, r.lam, r.seed, r.task) not in done
            and (r.path / f"checkpoint-{P.MAX_STEPS}").is_dir()]
    print(f"{len(runs)} adapters to evaluate ({len(done)} already in {out_file})", flush=True)
    if not runs:
        return

    base_id = P.MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(base_id, revision=pinned_revision(base_id))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map=args.device,
        revision=pinned_revision(base_id)).eval()
    collator = AnswerOnlyDataCollator(tokenizer)

    datasets = {}
    for run in runs:
        if run.task not in datasets:
            cfg = TASK_CONFIGS[run.task]
            datasets[run.task] = build_dataset(
                run.task, "eval", tokenizer, MAX_LENGTH,
                cfg["max_eval_samples"], run.seed)
        dataset = datasets[run.task]
        anchors = anchor_rows(run.task, run.seed, len(dataset))
        t0 = time.time()
        model = PeftModel.from_pretrained(base, str(run.path / f"checkpoint-{P.MAX_STEPS}")).eval()
        rows = answer_ce_per_row(model, dataset, collator, args.device)
        base = model.unload()
        if hasattr(base, "peft_config"):
            del base.peft_config
        rest = [r for i, r in enumerate(rows) if i not in anchors]
        record = dict(
            method=run.method, lam=run.lam, seed=run.seed, model=run.model, task=run.task,
            n_rows=len(dataset), n_anchor=len(anchors), n_eval_rows=len(rest),
            target_loss=token_mean(rest),
        )
        results.append(record)
        print(f"{run.method} lam{P.lam_tag(run.lam)} seed{run.seed} {run.task}: "
              f"target loss {record['target_loss']:.4f} on {record['n_eval_rows']} of "
              f"{record['n_rows']} rows ({record['n_anchor']} anchor rows excluded)  "
              f"({time.time() - t0:.1f}s)", flush=True)
        out_file.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
