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

    def __init__(self, processor=None, offload_to_cpu: bool = True):
        self.processor = processor
        self.offload_to_cpu = offload_to_cpu

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
                all_features.append(feats.cpu() if self.offload_to_cpu else feats)
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
    """Feature extraction for causal language models (Llama, Mistral, Gemma, …).

    Z = last hidden state after the final layer norm  (n_tokens, d)
    H = lm_head weight transposed to  (d, vocab_size)

    Supported model structures
    --------------------------
    Standard causal LM (Llama / Mistral / Gemma-2 / Qwen / OLMo …):
        model.model.embed_tokens   — transformer backbone at model.model
        model.lm_head              — language model head

    GPT-2 / GPT-Neo style:
        model.transformer          — transformer backbone

    Multimodal conditional-generation (Gemma-3 4B/12B/27B …):
        model.language_model.model — LM backbone nested under language_model
        model.language_model.lm_head

    Multimodal conditional-generation (Mistral3 / Ministral-3 2512 …):
        model.model.language_model — text backbone inside Mistral3Model wrapper
        model.lm_head              — LM head at the outer model level
    """

    def __init__(self, offload_to_cpu: bool = True):
        self.offload_to_cpu = offload_to_cpu

    @staticmethod
    def _get_backbone(model: torch.nn.Module):
        """Return the transformer backbone (before lm_head).

        Probe order (most-specific first):
          1. model.model.embed_tokens       — standard causal LM (Llama/Mistral/Gemma-2/Qwen)
          2. model.transformer              — GPT-2 / GPT-Neo
          3. model.language_model.model     — Gemma-3 4B/12B/27B multimodal
          4. model.language_model           — multimodal with flat backbone (embed_tokens at top)
          5. model.model.language_model     — Mistral3ForConditionalGeneration
             (Mistral3Model wrapper → MistralModel backbone)
        """
        # 1. Standard causal LM
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model
        # 2. GPT-2 / GPT-Neo style
        if hasattr(model, "transformer"):
            return model.transformer
        # 3. Gemma-3 4B/12B/27B: Gemma3ForConditionalGeneration
        #    backbone at model.language_model.model
        if (hasattr(model, "language_model")
                and hasattr(model.language_model, "model")
                and hasattr(model.language_model.model, "embed_tokens")):
            return model.language_model.model
        # 4. Flat multimodal: backbone IS model.language_model (embed_tokens directly)
        if (hasattr(model, "language_model")
                and hasattr(model.language_model, "embed_tokens")):
            return model.language_model
        # 5. Mistral3ForConditionalGeneration:
        #    model.model = Mistral3Model (vision + language wrapper)
        #    model.model.language_model = MistralModel (text backbone, has embed_tokens)
        if (hasattr(model, "model")
                and hasattr(model.model, "language_model")
                and hasattr(model.model.language_model, "embed_tokens")):
            return model.model.language_model
        raise AttributeError(
            f"Cannot locate backbone in {type(model).__name__}. "
            "Tried: .model, .transformer, .language_model.model, "
            ".language_model, .model.language_model."
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

                        # Backbone only accepts model inputs, not metadata keys
                inp = {k: v for k, v in batch_on_device.items() if k != "prompt_length"}
                out = backbone(**inp)
                hidden = out.last_hidden_state  # (bsz, seq, d)

                masks = inp.get("attention_mask")  # (bsz, seq)
                bsz = hidden.shape[0]
                if masks is not None:
                    lengths = masks.sum(dim=1).long() - 1          # (bsz,)
                    feats = hidden[torch.arange(bsz, device=hidden.device), lengths]  # (bsz, d)
                else:
                    feats = hidden[:, -1, :]  # (bsz, d)

                t = feats.float()
                all_features.append(t.cpu() if self.offload_to_cpu else t)

        return torch.cat(all_features, dim=0)

    def extract_head(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Tensor:
        """Return (d, vocab_size).

        Handles both standard causal LM (.lm_head) and multimodal models
        where the head is nested (.language_model.lm_head).
        """
        if hasattr(model, "lm_head"):
            w = model.lm_head.weight.data
            head = w.cpu().float() if self.offload_to_cpu else w.float()  # (vocab, d)
        elif (hasattr(model, "language_model")
              and hasattr(model.language_model, "lm_head")):
            w = model.language_model.lm_head.weight.data
            head = w.cpu().float() if self.offload_to_cpu else w.float()
        else:
            raise AttributeError(
                f"Cannot locate lm_head in {type(model).__name__}. "
                "Expected .lm_head or .language_model.lm_head."
            )
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
