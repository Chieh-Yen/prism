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

    def __init__(self, offload_to_cpu: bool = False):
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

    @staticmethod
    def _extract_z(
        hidden: Tensor,
        masks: Optional[Tensor],
        z_mode: str,
        prompt_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract per-sample Z from hidden states based on ``z_mode``.

        Args:
            hidden: (bsz, seq, d) last hidden states.
            masks:  (bsz, seq) attention mask (1 = real token).
            z_mode: ``"last_token"`` | ``"mean_pool"`` | ``"last_context_token"``
                    | ``"concat"``.
            prompt_lens: (bsz,) token count of the prompt prefix (required
                         for ``last_context_token``).
        Returns:
            For most modes: (bsz, d) feature vectors.
            For ``"concat"``: (total_valid_tokens, d) — all valid tokens at
            positions 0 … T−2 per sample (the last token is excluded because
            it has no next-token label to pair with).
        """
        bsz = hidden.shape[0]
        device = hidden.device

        if z_mode == "concat":
            # Per-token Z: positions 0..T-2 (skip last valid — no paired loss).
            parts: List[Tensor] = []
            for i in range(bsz):
                if masks is not None:
                    length = masks[i].sum().long().item()
                    if length > 1:
                        parts.append(hidden[i, :length - 1])
                else:
                    parts.append(hidden[i, :-1])
            return torch.cat(parts, dim=0) if parts else hidden.new_empty(0, hidden.shape[-1])

        if z_mode == "mean_pool":
            if masks is not None:
                mask_f = masks.unsqueeze(-1).to(hidden.dtype)   # (bsz, seq, 1)
                summed = (hidden * mask_f).sum(dim=1)           # (bsz, d)
                counts = masks.sum(dim=1, keepdim=True).to(hidden.dtype).clamp(min=1)
                return summed / counts
            return hidden.mean(dim=1)

        if z_mode == "last_context_token":
            if prompt_lens is not None:
                positions = (prompt_lens - 1).clamp(min=0).to(device)
                return hidden[torch.arange(bsz, device=device), positions]
            # Fallback: last valid token (same as last_token)

        # Default: last_token
        if masks is not None:
            lengths = masks.sum(dim=1).long() - 1
            lengths = lengths.to(device)
            return hidden[torch.arange(bsz, device=device), lengths]
        return hidden[:, -1, :]

    def extract_features(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        z_mode: str = "last_token",
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

                # Extract metadata before passing to backbone
                prompt_lens = batch_on_device.pop("prompt_length", None)
                batch_on_device.pop("reasoning_length", None)  # not needed here
                inp = {k: v for k, v in batch_on_device.items()}
                out = backbone(**inp)
                hidden = out.last_hidden_state  # (bsz, seq, d)

                masks = inp.get("attention_mask")  # (bsz, seq)
                feats = self._extract_z(hidden, masks, z_mode, prompt_lens)

                t = feats.float()
                all_features.append(t.cpu() if self.offload_to_cpu else t)

        return torch.cat(all_features, dim=0)

    def extract_features_and_loss_per_sample(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        *,
        chunk_size: int = 2048,
        z_mode: str = "last_token",
    ) -> Tuple[Tensor, Dict]:
        """Single forward pass that returns both features AND per-sample loss stats.

        Calls ``model(..., output_hidden_states=True)`` once per batch, obtaining
        the last hidden state (for features) **and** logits (for loss) in the same
        pass — halving the number of forward passes versus calling
        ``extract_features()`` + ``compute_lm_loss_per_sample()`` separately.

        Args:
            z_mode: Feature extraction mode — ``"last_token"`` (default),
                ``"mean_pool"`` (average all non-padding tokens), or
                ``"last_context_token"`` (last prompt token, requires
                ``prompt_length`` in data).

        The returned ``loss_stats`` dict contains:
            ``losses``               — (n,) per-sample average CE loss (full text)
            ``token_losses``         — (N_tokens,) per-token CE, or None
                                       (only for ``concat`` z_mode; paired with Z)
            ``answer_losses``        — (n,) answer-only loss, or None
            ``has_answer_loss``
            ``final_answer_losses``  — (n,) final-number-only loss (GSM8K), or None
            ``has_final_answer_loss``
            ``grad_norm_p95 / _max / _mean`` — empirical K_pred proxy

        Falls back to the two-pass approach automatically if
        ``output_hidden_states`` is not supported by the loaded model.
        """
        model.eval()
        all_features: List[Tensor] = []
        sample_losses: List[float] = []
        token_losses: List[Tensor] = []       # per-token CE for concat mode
        answer_losses: List[float] = []
        final_answer_losses: List[float] = []
        has_prompt = False
        has_reasoning = False
        is_concat = (z_mode == "concat")
        all_grad_norms: List[Tensor] = []
        CHUNK = chunk_size
        ce_none = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

        # Probe once whether the model supports output_hidden_states.
        # Some quantised wrappers silently ignore the flag; we verify on the
        # first batch and fall back to two passes if hidden_states is absent.
        _checked = False
        _use_combined = True  # updated after first batch

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features + loss", leave=False):
                if isinstance(batch, dict):
                    batch_on_device = {k: v.to(device) for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch_on_device = {
                        "input_ids": batch[0].to(device),
                        "attention_mask": batch[1].to(device),
                    }
                else:
                    batch_on_device = {"input_ids": batch.to(device)}

                prompt_lens = batch_on_device.pop("prompt_length", None)
                reasoning_lens = batch_on_device.pop("reasoning_length", None)
                if prompt_lens is not None:
                    has_prompt = True
                if reasoning_lens is not None:
                    has_reasoning = True

                inp = {k: v for k, v in batch_on_device.items()}
                masks = inp.get("attention_mask")
                bsz = inp["input_ids"].shape[0]

                # ── Forward pass ──────────────────────────────────────────
                if not _checked:
                    try:
                        out = model(**inp, output_hidden_states=True)
                        _use_combined = (out.hidden_states is not None
                                         and len(out.hidden_states) > 0)
                    except TypeError:
                        _use_combined = False
                    _checked = True
                    if not _use_combined:
                        # Re-run without the flag for this batch
                        out = model(**inp)
                else:
                    if _use_combined:
                        out = model(**inp, output_hidden_states=True)
                    else:
                        out = model(**inp)

                all_logits = out.logits  # (bsz, seq, V)

                # ── Feature extraction ────────────────────────────────────
                if _use_combined:
                    hidden = out.hidden_states[-1]  # post-norm last layer (bsz, seq, d)
                else:
                    # Fallback: derive from backbone directly
                    backbone = self._get_backbone(model)
                    bb_out = backbone(**inp)
                    hidden = bb_out.last_hidden_state

                feats = self._extract_z(hidden, masks, z_mode, prompt_lens)
                all_features.append(feats.float().cpu() if self.offload_to_cpu else feats.float())

                # ── Per-sample loss ───────────────────────────────────────
                for j in range(bsz):
                    labels = inp["input_ids"][j].clone()
                    if masks is not None:
                        labels[masks[j] == 0] = -100

                    shift_logits = all_logits[j, :-1, :]
                    shift_labels = labels[1:]
                    seq_len = shift_logits.shape[0]

                    loss_sum, n_valid = 0.0, 0
                    for s in range(0, seq_len, CHUNK):
                        e = min(s + CHUNK, seq_len)
                        cl = shift_labels[s:e]
                        v = (cl != -100).sum().item()
                        if v == 0:
                            continue
                        loss_sum += ce_none(shift_logits[s:e], cl).item() * v
                        n_valid += v
                    sample_losses.append(loss_sum / max(n_valid, 1))

                    # Per-token losses (concat mode): each z_t at position t
                    # is paired with CE(h(z_t), y_{t+1}).  Only valid tokens
                    # are included, matching _extract_z("concat") which
                    # returns positions 0..T-2.
                    if is_concat:
                        ce_per_tok = torch.nn.functional.cross_entropy(
                            shift_logits, shift_labels,
                            reduction="none", ignore_index=-100,
                        )
                        valid = shift_labels != -100
                        token_losses.append(ce_per_tok[valid].cpu())

                    # Answer-only loss (tokens after prompt_length)
                    if prompt_lens is not None:
                        pl = int(prompt_lens[j].item())
                        ans_labels = labels.clone()
                        ans_labels[:pl] = -100
                        shift_ans = ans_labels[1:]
                        a_sum, a_valid = 0.0, 0
                        for s in range(0, seq_len, CHUNK):
                            e = min(s + CHUNK, seq_len)
                            cl = shift_ans[s:e]
                            v = (cl != -100).sum().item()
                            if v == 0:
                                continue
                            a_sum += ce_none(shift_logits[s:e], cl).item() * v
                            a_valid += v
                        answer_losses.append(a_sum / max(a_valid, 1))

                    # Final-answer-only loss (tokens after reasoning_length,
                    # i.e. only the final number in GSM8K)
                    if reasoning_lens is not None:
                        rl = int(reasoning_lens[j].item())
                        final_labels = labels.clone()
                        final_labels[:rl] = -100
                        shift_final = final_labels[1:]
                        f_sum, f_valid = 0.0, 0
                        for s in range(0, seq_len, CHUNK):
                            e = min(s + CHUNK, seq_len)
                            cl = shift_final[s:e]
                            v = (cl != -100).sum().item()
                            if v == 0:
                                continue
                            f_sum += ce_none(shift_logits[s:e], cl).item() * v
                            f_valid += v
                        final_answer_losses.append(f_sum / max(f_valid, 1))

                    # Per-token gradient norms (empirical K_pred)
                    gn_logits = shift_logits
                    gn_targets = shift_labels
                    if masks is not None:
                        token_mask = masks[j, 1:].bool()
                        gn_logits = gn_logits[token_mask]
                        gn_targets = gn_targets[token_mask]
                    gn_len = gn_logits.shape[0]
                    for s in range(0, gn_len, CHUNK):
                        e = min(s + CHUNK, gn_len)
                        p = torch.softmax(gn_logits[s:e].float(), dim=-1)
                        oh = torch.zeros_like(p)
                        oh.scatter_(1, gn_targets[s:e].unsqueeze(1), 1.0)
                        gnorms = (p - oh).norm(dim=-1)
                        all_grad_norms.append(gnorms.cpu())

                del all_logits

        Z = torch.cat(all_features, dim=0)
        all_gn = torch.cat(all_grad_norms) if all_grad_norms else torch.zeros(1)
        _tok_losses = torch.cat(token_losses) if token_losses else None
        loss_stats: Dict = {
            "losses": torch.tensor(sample_losses),
            "token_losses": _tok_losses,
            "answer_losses": torch.tensor(answer_losses) if has_prompt else None,
            "has_answer_loss": has_prompt,
            "final_answer_losses": torch.tensor(final_answer_losses) if has_reasoning else None,
            "has_final_answer_loss": has_reasoning,
            "grad_norm_p95": torch.quantile(all_gn, 0.95).item(),
            "grad_norm_max": all_gn.max().item(),
            "grad_norm_mean": all_gn.mean().item(),
        }
        return Z, loss_stats

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
