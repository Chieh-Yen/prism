"""
Feature extractors — abstract base class + CLIP and LLM implementations.

Each extractor takes a loaded model + data and returns (Z, H):
    Z: (n, d)  feature matrix from the backbone
    H: (d, C)  prediction-head weight matrix
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractor(ABC):
    """Abstract base: extract features Z and head weights H from a model."""

    @abstractmethod
    def extract_features(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Tensor:
        """Return Z with shape (n, d)."""

    @abstractmethod
    def extract_head(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Tensor:
        """Return H with shape (d, C)."""

    def extract(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Convenience: return (Z, H) in one call."""
        Z = self.extract_features(model, dataloader, device)
        H = self.extract_head(model, **kwargs)
        return Z, H


class CLIPExtractor(FeatureExtractor):
    """Feature extraction for CLIP / SigLIP vision-language models.

    Z = normalised image embeddings  (n, d)
    H = normalised zero-shot text embeddings transposed to  (d, C)
    """

    def __init__(self, processor=None):
        self.processor = processor

    def extract_features(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Tensor:
        model.eval()
        all_features: List[Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting CLIP features", leave=False):
                images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                feats = model.get_image_features(pixel_values=images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_features.append(feats.cpu())
        return torch.cat(all_features, dim=0)

    def extract_head(
        self,
        model: torch.nn.Module,
        *,
        text_weights: Optional[Tensor] = None,
        task_name: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ) -> Tensor:
        """Return (d, C) head weights.

        If ``text_weights`` is provided directly, use that.
        Otherwise, derive from the model's text encoder + class templates.
        """
        if text_weights is not None:
            H = text_weights.float()
            if H.shape[0] > H.shape[1]:
                H = H.T
            return H

        raise ValueError(
            "CLIPExtractor.extract_head requires pre-computed text_weights. "
            "Compute zero-shot weights externally and pass them in."
        )


class LLMExtractor(FeatureExtractor):
    """Feature extraction for causal language models (Llama, Mistral, …).

    Z = last hidden state after the final layer norm  (n_tokens, d)
    H = lm_head weight transposed to  (d, vocab_size)
    """

    @staticmethod
    def _get_backbone(model: torch.nn.Module):
        """Return the transformer backbone (before lm_head).

        Supports Llama/Mistral (model.model), GPT-2/GPT-Neo (model.transformer).
        """
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model
        if hasattr(model, "transformer"):
            return model.transformer
        raise AttributeError(
            f"Cannot locate backbone in {type(model).__name__}. "
            "Expected .model (Llama/Mistral) or .transformer (GPT-2)."
        )

    def extract_features(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Tensor:
        model.eval()
        backbone = self._get_backbone(model)
        all_features: List[Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting LLM features", leave=False):
                if isinstance(batch, dict):
                    batch_on_device = {k: v.to(device) for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch_on_device = {"input_ids": batch[0].to(device), "attention_mask": batch[1].to(device)}
                else:
                    batch_on_device = {"input_ids": batch.to(device)}

                bsz = batch_on_device["input_ids"].shape[0]
                for j in range(bsz):
                    single = {k: v[j : j + 1] for k, v in batch_on_device.items()}
                    out = backbone(**single)
                    hidden = out.last_hidden_state  # (1, seq, d)

                    mask = single.get("attention_mask")
                    if mask is not None:
                        length = mask.sum().long() - 1
                        feat = hidden[0, length, :]
                    else:
                        feat = hidden[0, -1, :]

                    all_features.append(feat.float().cpu().unsqueeze(0))

        return torch.cat(all_features, dim=0)

    def extract_head(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Tensor:
        """Return (d, vocab_size)."""
        head = model.lm_head.weight.data.float().cpu()  # (vocab, d)
        return head.T  # (d, vocab)


# ------------------------------------------------------------------
# Registry / factory
# ------------------------------------------------------------------
_EXTRACTOR_MAP = {
    "clip": CLIPExtractor,
    "siglip": CLIPExtractor,
    "llm": LLMExtractor,
}


def get_extractor(name: str, **kwargs) -> FeatureExtractor:
    """Look up an extractor by short name (``clip``, ``llm``, …)."""
    key = name.lower()
    if key not in _EXTRACTOR_MAP:
        raise ValueError(f"Unknown extractor '{name}'. Choose from {list(_EXTRACTOR_MAP.keys())}")
    return _EXTRACTOR_MAP[key](**kwargs)
