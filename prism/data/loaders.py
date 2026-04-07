"""
Unified task-based data loading.

Provides a single ``load_task_data`` function that returns a PyTorch DataLoader
for any registered task (HuggingFace datasets, image classification, text, …).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ======================================================================
# Row → plain-text formatters  (for structured Q&A datasets)
# ======================================================================
def _format_gsm8k(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer: {row['answer']}"


def _format_mmlu(row: Dict[str, Any]) -> str:
    q = row["question"]
    choices = row["choices"]
    ans = row["answer"]
    if isinstance(ans, int):
        ans_label = ["A", "B", "C", "D"][ans]
    else:
        ans_label = ans
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Question: {q}\n{opts}\nAnswer: {ans_label}"


def _format_arc(row: Dict[str, Any]) -> str:
    q = row["question"]
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    opts = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(len(labels)))
    key = row["answerKey"]
    return f"Question: {q}\n{opts}\nAnswer: {key}"


def _format_triviaqa(row: Dict[str, Any]) -> str:
    q = row["question"]
    ans = row["answer"]["value"]
    return f"Question: {q}\nAnswer: {ans}"


def _format_squad(row: Dict[str, Any]) -> str:
    context = row["context"]
    q = row["question"]
    ans = row["answers"]["text"][0]
    return f"Context: {context}\nQuestion: {q}\nAnswer: {ans}"


_FORMATTERS: Dict[str, Callable] = {
    "gsm8k":    _format_gsm8k,
    "mmlu":     _format_mmlu,
    "arc":      _format_arc,
    "triviaqa": _format_triviaqa,
    "squad":    _format_squad,
}


# Prompt-only extractors (question prefix, no answer) — used to derive
# ``prompt_length`` so that answer-only loss can be computed later.
def _prompt_gsm8k(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer:"


def _prompt_mmlu(row: Dict[str, Any]) -> str:
    q = row["question"]
    choices = row["choices"]
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Question: {q}\n{opts}\nAnswer:"


def _prompt_arc(row: Dict[str, Any]) -> str:
    q = row["question"]
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    opts = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(len(labels)))
    return f"Question: {q}\n{opts}\nAnswer:"


def _prompt_triviaqa(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer:"


def _prompt_squad(row: Dict[str, Any]) -> str:
    return f"Context: {row['context']}\nQuestion: {row['question']}\nAnswer:"


_PROMPT_FORMATTERS: Dict[str, Callable] = {
    "gsm8k":    _prompt_gsm8k,
    "mmlu":     _prompt_mmlu,
    "arc":      _prompt_arc,
    "triviaqa": _prompt_triviaqa,
    "squad":    _prompt_squad,
}


# ======================================================================
# Task registry — maps short names to HuggingFace dataset identifiers
# ======================================================================
TASK_REGISTRY: Dict[str, Dict] = {
    # Vision classification
    "sun397":        {"hf_id": "tanganke/sun397",     "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "stanford-cars": {"hf_id": "tanganke/stanford-cars", "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "resisc45":      {"hf_id": "tanganke/resisc45",   "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "eurosat":       {"hf_id": "tanganke/eurosat",    "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "svhn":          {"hf_id": "ufldl-stanford/svhn",  "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "gtsrb":         {"hf_id": "tanganke/gtsrb",      "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "mnist":         {"hf_id": "ylecun/mnist",         "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "dtd":           {"hf_id": "tanganke/dtd",         "label_key": "label", "image_key": "image", "split_map": {"test": "test"}},
    "cifar10":       {"hf_id": "uoft-cs/cifar10",      "label_key": "label", "image_key": "img",   "split_map": {"test": "test"}},
    "cifar100":      {"hf_id": "uoft-cs/cifar100",     "label_key": "fine_label", "image_key": "img", "split_map": {"test": "test"}},
    # Text / LLM  — plain text defaults to concat over valid sequence
    "wikitext":      {"hf_id": "Salesforce/wikitext",  "hf_subset": "wikitext-2-raw-v1", "text_key": "text", "split_map": {"test": "test"},
                      "z_mode": "concat", "loss_mode": "full"},
    "ptb":           {"hf_id": "ptb-text-only/ptb_text_only", "text_key": "sentence", "split_map": {"test": "test"},
                      "z_mode": "concat", "loss_mode": "full"},
    "c4":            {"hf_id": "allenai/c4",           "hf_subset": "en", "text_key": "text",  "split_map": {"test": "validation"}, "streaming": True,
                      "z_mode": "concat", "loss_mode": "full"},
    # Text / LLM  — LAMBADA uses concat on full sequence (context + answer)
    "lambada":       {"hf_id": "EleutherAI/lambada_openai", "text_key": "text", "split_map": {"test": "test"},
                      "z_mode": "concat", "loss_mode": "full"},
    # Text / LLM  — FineWeb-Edu  (language modeling, curated educational text)
    "fineweb_edu":   {"hf_id": "HuggingFaceFW/FineWeb-Edu-score-2", "text_key": "text", "split_map": {"test": "train"}, "streaming": True,
                      "z_mode": "concat", "loss_mode": "full"},
    # Text / LLM  — structured Q&A
    "mmlu":          {"hf_id": "cais/mmlu",            "hf_subset": "all",           "formatter": "mmlu",  "split_map": {"test": "test"},
                      "z_mode": "last_context_token", "loss_mode": "answer"},
    "arc":           {"hf_id": "allenai/ai2_arc",      "hf_subset": "ARC-Challenge", "formatter": "arc",   "split_map": {"test": "test"},
                      "z_mode": "last_context_token", "loss_mode": "answer"},
    "arc_easy":      {"hf_id": "allenai/ai2_arc",      "hf_subset": "ARC-Easy",      "formatter": "arc",   "split_map": {"test": "test"},
                      "z_mode": "last_context_token", "loss_mode": "answer"},
    # Text / LLM  — TriviaQA/SQuAD use concat over answer region
    "triviaqa":      {"hf_id": "trivia_qa",            "hf_subset": "rc.nocontext",  "formatter": "triviaqa", "split_map": {"test": "validation"},
                      "z_mode": "concat", "loss_mode": "answer"},
    # Text / LLM  — SQuAD  (short-horizon generation, extractive QA)
    "squad":         {"hf_id": "rajpurkar/squad",                                    "formatter": "squad",    "split_map": {"test": "validation"},
                      "z_mode": "concat", "loss_mode": "answer"},
    "gsm8k":         {"hf_id": "openai/gsm8k",        "hf_subset": "main",          "formatter": "gsm8k", "split_map": {"test": "test"},
                      "z_mode": "concat", "loss_mode": "answer"},
}


def get_task_metadata(task_name: str) -> Dict:
    """Return z_mode and loss_mode metadata for a task."""
    meta = TASK_REGISTRY.get(task_name, {})
    default_zm = meta.get("z_mode", "last_token")
    return {
        "z_mode": default_zm,
        "loss_mode": meta.get("loss_mode", "full"),
    }


# ======================================================================
# Wrapper datasets
# ======================================================================
class ImageClassificationDataset(Dataset):
    """Wraps a HuggingFace image-classification dataset into (image_tensor, label)."""

    def __init__(self, hf_dataset, image_key: str, label_key: str, transform: Optional[Callable] = None):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample[self.image_key]
        label = sample[self.label_key]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TextDataset(Dataset):
    """Wraps a HuggingFace text dataset into tokenised batches.

    Supports two modes:
      1. ``text_key`` — read a named column directly (C4, WikiText, LAMBADA …).
      2. ``formatter`` — apply a callable ``row → str`` for structured
         datasets (GSM8K, MMLU, ARC, …).

    When ``prompt_formatter`` is supplied, the dataset additionally stores
    ``prompt_length`` per sample so that downstream code can compute
    answer-only loss by masking the prompt tokens.

    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        text_key: str = "text",
        max_length: int = 512,
        num_samples: Optional[int] = None,
        formatter: Optional[Callable[[Dict], str]] = None,
        prompt_formatter: Optional[Callable[[Dict], str]] = None,
    ):
        if formatter is not None:
            raw = [
                (formatter(row), prompt_formatter(row) if prompt_formatter else None)
                for row in hf_dataset
            ]
        else:
            raw = [
                (row[text_key], prompt_formatter(row) if prompt_formatter else None)
                for row in hf_dataset
            ]
        raw = [(t, p) for t, p in raw if t.strip()]
        if num_samples is not None:
            raw = raw[:num_samples]

        texts = [t for t, _ in raw]
        self.encodings = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.num_samples = len(texts)

        # Prompt lengths (for answer-only loss and last_context_token Z).
        # We verify prefix alignment with the full-text tokenisation via a
        # longest-common-prefix match so that prompt_length is always
        # defined in the full-text token space.  This guards against
        # context-dependent tokeniser divergence (e.g. SentencePiece
        # unigram) and also handles left-padded sequences by including the
        # padding offset in the stored prompt_length.
        if prompt_formatter is not None:
            prompts = [p for _, p in raw]
            prompt_enc = tokenizer(
                prompts, truncation=True, max_length=max_length,
                add_special_tokens=True,
            )
            pad_id = tokenizer.pad_token_id
            prompt_lengths_list: List[int] = []
            n_mismatch = 0
            for i, p_ids in enumerate(prompt_enc["input_ids"]):
                f_ids = self.encodings["input_ids"][i].tolist()
                # Skip leading pad tokens (left-padded sequences)
                f_start = 0
                if pad_id is not None:
                    while f_start < len(f_ids) and f_ids[f_start] == pad_id:
                        f_start += 1
                expected_pl = len(p_ids)
                pl = 0
                for k in range(min(expected_pl, len(f_ids) - f_start)):
                    fk = f_ids[f_start + k]
                    if pad_id is not None and fk == pad_id:
                        break
                    if p_ids[k] == fk:
                        pl = k + 1
                    else:
                        break
                if pl < expected_pl:
                    n_mismatch += 1
                prompt_lengths_list.append(f_start + pl)
            if n_mismatch > 0:
                import warnings
                warnings.warn(
                    f"Tokeniser prefix mismatch in {n_mismatch}/{len(prompts)} "
                    f"samples; prompt_length adjusted via longest common prefix."
                )
            self.prompt_lengths = torch.tensor(
                prompt_lengths_list, dtype=torch.long,
            )
        else:
            self.prompt_lengths = None

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.prompt_lengths is not None:
            item["prompt_length"] = self.prompt_lengths[idx]
        return item


# ======================================================================
# Public API
# ======================================================================
def load_task_data(
    task_name: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    *,
    transform: Optional[Callable] = None,
    tokenizer=None,
    max_length: int = 512,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    """Load a task dataset and return a ready-to-use DataLoader.

    Args:
        task_name:   Key in ``TASK_REGISTRY`` (e.g. ``"wikitext"``).
        split:       Logical split name (``"test"``, ``"train"``).
        num_samples: Limit number of samples (None = use all).
        batch_size:  Batch size.
        num_workers: DataLoader workers.
        transform:   Image transform (for vision tasks).
        tokenizer:   HuggingFace tokenizer (required for text tasks).
        max_length:  Maximum token length for text tasks.
        shuffle:     Whether to shuffle the DataLoader across epochs.
        seed:        Random seed for reproducible dataset-level shuffling
                     before selecting ``num_samples``.  Ignored for
                     streaming datasets (e.g. c4) where order is fixed.
    """
    from datasets import load_dataset  # lazy import to keep startup fast

    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_name}'. Available: {sorted(TASK_REGISTRY.keys())}")

    meta = TASK_REGISTRY[task_name]
    hf_split = meta.get("split_map", {}).get(split, split)

    load_kwargs = {"split": hf_split}
    if "hf_subset" in meta:
        load_kwargs["name"] = meta["hf_subset"]
    if meta.get("streaming"):
        load_kwargs["streaming"] = True

    hf_dataset = load_dataset(meta["hf_id"], **load_kwargs)

    if meta.get("streaming"):
        # Streaming datasets (c4): take first N rows; shuffle not supported.
        rows = list(hf_dataset.take(num_samples or 256))
        from datasets import Dataset as HFDataset
        hf_dataset = HFDataset.from_list(rows)
    else:
        # Shuffle before selecting so the chosen N samples are random but
        # reproducible.  Skip when seed is None to preserve legacy behaviour.
        if seed is not None:
            hf_dataset = hf_dataset.shuffle(seed=seed)
        if num_samples is not None and num_samples < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(num_samples))

    is_text = "text_key" in meta or "formatter" in meta
    if is_text:
        if tokenizer is None:
            raise ValueError(f"Task '{task_name}' is text-based: pass a tokenizer.")
        fmt_key = meta.get("formatter")
        fmt_fn = _FORMATTERS.get(fmt_key) if fmt_key else None
        # Fall back to task_name for prompt formatters so that
        # plain-text datasets (e.g. LAMBADA) can still define them.
        pfmt_fn = _PROMPT_FORMATTERS.get(fmt_key) if fmt_key else _PROMPT_FORMATTERS.get(task_name)
        ds = TextDataset(
            hf_dataset, tokenizer,
            text_key=meta.get("text_key", "text"),
            max_length=max_length,
            formatter=fmt_fn,
            prompt_formatter=pfmt_fn,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    image_key = meta.get("image_key", "image")
    label_key = meta.get("label_key", "label")
    ds = ImageClassificationDataset(hf_dataset, image_key=image_key, label_key=label_key, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
