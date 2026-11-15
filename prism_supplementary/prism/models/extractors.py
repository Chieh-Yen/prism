"""
Feature extractor for causal language models.

Returns (Z, H):
    Z: (n, d)  feature matrix from the backbone
    H: (d, C)  prediction-head weight matrix
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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


class LLMExtractor(FeatureExtractor):
    """Feature extraction for causal language models.

    Z = last hidden state after the final layer norm  (n_tokens, d)
    H = lm_head weight transposed to  (d, vocab_size)

    Model structures the probes recognise
    -------------------------------------
    The experiments of the paper use the Llama-3.1, Qwen3,
    DeepSeek-R1-Distill-Llama and Ministral-3 families.

    Standard causal LM (Llama / Qwen / Mistral):
        model.model.embed_tokens:   transformer backbone at model.model
        model.lm_head:              language model head

    Multimodal conditional-generation (Mistral3 / Ministral-3 2512):
        model.language_model:       text backbone, when the class exposes
                                    model.model.language_model at the top
        model.model.language_model: text backbone inside Mistral3Model wrapper
        model.lm_head:              LM head at the outer model level
    """

    def __init__(self):
        pass

    @staticmethod
    def _get_backbone(model: torch.nn.Module):
        """Return the transformer backbone (before lm_head).

        Probe order (most-specific first):
          1. model.model.embed_tokens:   standard causal LM (Llama/Qwen/Mistral)
          2. model.language_model:       multimodal with flat backbone (embed_tokens at top)
          3. model.model.language_model: Mistral3ForConditionalGeneration
             (Mistral3Model wrapper → MistralModel backbone)
        """
        # 1. Standard causal LM
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model
        # 2. Flat multimodal: backbone IS model.language_model (embed_tokens directly)
        if (hasattr(model, "language_model")
                and hasattr(model.language_model, "embed_tokens")):
            return model.language_model
        # 3. Mistral3ForConditionalGeneration:
        #    model.model = Mistral3Model (vision + language wrapper)
        #    model.model.language_model = MistralModel (text backbone, has embed_tokens)
        if (hasattr(model, "model")
                and hasattr(model.model, "language_model")
                and hasattr(model.model.language_model, "embed_tokens")):
            return model.model.language_model
        raise AttributeError(
            f"Cannot locate backbone in {type(model).__name__}. "
            "Tried: .model, .language_model, .model.language_model."
        )

    @staticmethod
    def _extract_z(
        hidden: Tensor,
        masks: Optional[Tensor],
        z_mode: str,
        prompt_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract Z from hidden states (``z_mode`` must be ``"concat"``).

        Args:
            hidden: (bsz, seq, d) last hidden states.
            masks:  (bsz, seq) attention mask (1 = real token).
            z_mode: ``"concat"``, the only implemented mode.
            prompt_lens: (bsz,) token count of the prompt prefix, or None.
        Returns:
            (total_valid_tokens, d): per-token hidden states whose logits
            predict the next token.  Corpus datasets (no ``prompt_lens``):
            positions 0 … T−2.  Q&A datasets (with ``prompt_lens``):
            answer-region positions prompt_len−1 … T−2 only, so each z_t is
            paired with an answer-token loss.
        """
        if z_mode != "concat":
            raise ValueError(f"unsupported z_mode {z_mode!r}; only 'concat' is implemented")
        bsz = hidden.shape[0]
        parts: List[Tensor] = []
        for i in range(bsz):
            length = masks[i].sum().long().item() if masks is not None else hidden.shape[1]
            # Answer-region start: first z that predicts an answer token.
            # Corpus (no prompt_lens): position 0.
            # Q&A: position prompt_len - 1 (its logits predict y_{prompt_len}).
            start = max(int(prompt_lens[i].item()) - 1, 0) if prompt_lens is not None else 0
            end = length - 1  # exclude last valid token (no next-token label)
            if end > start:
                parts.append(hidden[i, start:end])
        return torch.cat(parts, dim=0) if parts else hidden.new_empty(0, hidden.shape[-1])

    def extract_features(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        z_mode: str = "concat",
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

                prompt_lens = batch_on_device.pop("prompt_length", None)
                out = backbone(**batch_on_device)
                hidden = out.last_hidden_state  # (bsz, seq, d)

                masks = batch_on_device.get("attention_mask")
                feats = self._extract_z(hidden, masks, z_mode, prompt_lens)

                all_features.append(feats.float().cpu())

        return torch.cat(all_features, dim=0)

    def extract_features_and_loss_per_sample(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        *,
        z_mode: str = "concat",
        z_modes: Optional[List[str]] = None,
    ):
        """Single forward pass that returns both features AND per-sample loss stats.

        Calls ``model(..., output_hidden_states=True)`` once per batch, obtaining
        the last hidden state (for features) **and** logits (for loss) in the same
        pass; halving the number of forward passes versus extracting features and
        loss separately.

        Args:
            z_mode:  Feature extraction mode (``"concat"``).
            z_modes: List form of ``z_mode`` (``["concat"]``).  When provided,
                     returns ``(Dict[str, Tensor], Dict)`` instead of
                     ``(Tensor, Dict)``.

        The returned ``loss_stats`` dict contains:
            ``losses``              ; (n,) per-sample average CE loss (full text)
            ``token_losses``        ; (N_tokens,) per-token CE, or None
                                       (present when ``"concat"`` is among z_modes)
            ``answer_losses``       ; (n,) answer-only loss, or None
            ``has_answer_loss``
        """
        _return_dict = z_modes is not None
        if z_modes is None:
            z_modes = [z_mode]

        model.eval()
        all_features: Dict[str, List[Tensor]] = {zm: [] for zm in z_modes}
        sample_losses: List[float] = []
        token_losses: List[Tensor] = []       # per-token CE for concat mode
        answer_losses: List[float] = []
        has_prompt = False
        is_concat = ("concat" in z_modes)

        _checked = False
        _use_combined = True

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
                if prompt_lens is not None:
                    has_prompt = True

                masks = batch_on_device.get("attention_mask")
                bsz = batch_on_device["input_ids"].shape[0]

                # ── Forward pass ──────────────────────────────────────────
                if not _checked:
                    try:
                        out = model(**batch_on_device, output_hidden_states=True)
                        _use_combined = (out.hidden_states is not None
                                         and len(out.hidden_states) > 0)
                    except TypeError:
                        _use_combined = False
                    _checked = True
                    if not _use_combined:
                        out = model(**batch_on_device)
                else:
                    if _use_combined:
                        out = model(**batch_on_device, output_hidden_states=True)
                    else:
                        out = model(**batch_on_device)

                all_logits = out.logits  # (bsz, seq, V)

                # ── Feature extraction ────────────────────────────────────
                if _use_combined:
                    hidden = out.hidden_states[-1]  # post-norm last layer
                else:
                    backbone = self._get_backbone(model)
                    bb_out = backbone(**batch_on_device)
                    hidden = bb_out.last_hidden_state

                # Free all intermediate hidden states (layers 0..N-2); saves
                # ~2 GB for 32-layer models with bsz=8, seq=1024.
                del out

                for zm in z_modes:
                    feats = self._extract_z(hidden, masks, zm, prompt_lens)
                    all_features[zm].append(feats.float().cpu())
                del hidden

                # ── Per-sample loss (vectorized) ──────────────────────────
                V = all_logits.shape[-1]
                shift_labels = batch_on_device["input_ids"][:, 1:].clone()       # (bsz, T-1)
                if masks is not None:
                    shift_labels[masks[:, 1:] == 0] = -100
                T_m1 = shift_labels.shape[1]

                # Single batched CE; reduction='none' returns 0 at ignore positions
                ce_per_token = torch.nn.functional.cross_entropy(
                    all_logits[:, :-1, :].reshape(-1, V),
                    shift_labels.reshape(-1),
                    reduction="none", ignore_index=-100,
                ).reshape(bsz, T_m1)                                 # (bsz, T-1)

                valid_mask = (shift_labels != -100).float()           # (bsz, T-1)
                valid_counts = valid_mask.sum(dim=1).clamp(min=1)     # (bsz,)

                # Full-text per-sample loss
                batch_losses = (ce_per_token * valid_mask).sum(dim=1) / valid_counts
                sample_losses.extend(batch_losses.tolist())

                # Per-token losses (concat mode)
                if is_concat:
                    positions = torch.arange(T_m1, device=ce_per_token.device)
                    for j in range(bsz):
                        if prompt_lens is not None:
                            pl = int(prompt_lens[j].item())
                            tok_valid = valid_mask[j].bool() & (positions >= (pl - 1))
                        else:
                            tok_valid = valid_mask[j].bool()
                        token_losses.append(ce_per_token[j][tok_valid].float())

                # Answer-only loss; reuse ce_per_token with tighter mask
                if prompt_lens is not None:
                    pos_2d = torch.arange(T_m1, device=shift_labels.device).unsqueeze(0)
                    ans_valid = valid_mask.clone()
                    ans_valid[pos_2d < (prompt_lens.unsqueeze(1) - 1)] = 0
                    ans_counts = ans_valid.sum(dim=1).clamp(min=1)
                    batch_ans = (ce_per_token * ans_valid).sum(dim=1) / ans_counts
                    answer_losses.extend(batch_ans.tolist())

                del all_logits

        Z_dict = {zm: torch.cat(all_features[zm], dim=0) for zm in z_modes}
        _tok_losses = torch.cat(token_losses) if token_losses else None

        loss_stats: Dict = {
            "losses": torch.tensor(sample_losses),
            "token_losses": _tok_losses,
            "answer_losses": torch.tensor(answer_losses) if has_prompt else None,
            "has_answer_loss": has_prompt,
        }
        if _return_dict:
            return Z_dict, loss_stats
        return Z_dict[z_modes[0]], loss_stats

    def extract_head(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> Tensor:
        """Return (d, vocab_size) in model dtype.

        Returns a contiguous copy so it does NOT keep the model alive
        after the caller deletes the model.  Stays in model dtype (e.g.
        BF16) to avoid a large float32 allocation on GPU; callers (metrics,
        bounds) cast it themselves.
        """
        if hasattr(model, "lm_head"):
            w = model.lm_head.weight.data  # (vocab, d)
        else:
            raise AttributeError(
                f"Cannot locate lm_head in {type(model).__name__}. "
                "Expected .lm_head."
            )
        return w.T.contiguous()  # (d, vocab); detached copy, won't pin model
