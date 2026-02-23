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
        ans_text = choices[ans]
    else:
        ans_text = choices[ord(ans) - ord("A")]
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Question: {q}\n{opts}\nAnswer: {ans_text}"


def _format_arc(row: Dict[str, Any]) -> str:
    q = row["question"]
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    opts = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(len(labels)))
    key = row["answerKey"]
    ans_text = texts[labels.index(key)]
    return f"Question: {q}\n{opts}\nAnswer: {ans_text}"


_FORMATTERS: Dict[str, Callable] = {
    "gsm8k": _format_gsm8k,
    "mmlu":  _format_mmlu,
    "arc":   _format_arc,
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
    # Text / LLM  — plain text
    "wikitext":      {"hf_id": "Salesforce/wikitext",  "hf_subset": "wikitext-2-raw-v1", "text_key": "text", "split_map": {"test": "test"}},
    "ptb":           {"hf_id": "ptb-text-only/ptb_text_only", "text_key": "sentence", "split_map": {"test": "test"}},
    "c4":            {"hf_id": "allenai/c4",           "hf_subset": "en", "text_key": "text",  "split_map": {"test": "validation"}, "streaming": True},
    "lambada":       {"hf_id": "EleutherAI/lambada_openai", "text_key": "text", "split_map": {"test": "test"}},
    # Text / LLM  — structured Q&A  (formatter converts row → plain text)
    "gsm8k":         {"hf_id": "openai/gsm8k",        "hf_subset": "main",          "formatter": "gsm8k", "split_map": {"test": "test"}},
    "mmlu":          {"hf_id": "cais/mmlu",            "hf_subset": "all",           "formatter": "mmlu",  "split_map": {"test": "test"}},
    "arc":           {"hf_id": "allenai/ai2_arc",      "hf_subset": "ARC-Challenge", "formatter": "arc",   "split_map": {"test": "test"}},
    "arc_easy":      {"hf_id": "allenai/ai2_arc",      "hf_subset": "ARC-Easy",      "formatter": "arc",   "split_map": {"test": "test"}},
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
      1. ``text_key`` — read a named column directly (C4, WikiText, …).
      2. ``formatter`` — apply a callable ``row → str`` for structured
         datasets (GSM8K, MMLU, ARC, …).
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        text_key: str = "text",
        max_length: int = 512,
        num_samples: Optional[int] = None,
        formatter: Optional[Callable[[Dict], str]] = None,
    ):
        if formatter is not None:
            texts = [formatter(row) for row in hf_dataset]
        else:
            texts = [row[text_key] for row in hf_dataset]
        texts = [t for t in texts if t.strip()]
        if num_samples is not None:
            texts = texts[:num_samples]
        self.encodings = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.num_samples = len(texts)

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


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
        shuffle:     Whether to shuffle.
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
        rows = list(hf_dataset.take(num_samples or 256))
        from datasets import Dataset as HFDataset
        hf_dataset = HFDataset.from_list(rows)
    elif num_samples is not None and num_samples < len(hf_dataset):
        hf_dataset = hf_dataset.select(range(num_samples))

    is_text = "text_key" in meta or "formatter" in meta
    if is_text:
        if tokenizer is None:
            raise ValueError(f"Task '{task_name}' is text-based: pass a tokenizer.")
        fmt_fn = _FORMATTERS.get(meta["formatter"]) if "formatter" in meta else None
        ds = TextDataset(
            hf_dataset, tokenizer,
            text_key=meta.get("text_key", "text"),
            max_length=max_length,
            num_samples=num_samples,
            formatter=fmt_fn,
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
