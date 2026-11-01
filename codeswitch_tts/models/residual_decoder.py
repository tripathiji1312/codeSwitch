"""Hierarchical Residual Decoder for Conditional Generation of Codebooks 1-8.

To prevent speculative acceptance collapse across the full 9-codebook DAC representation,
speculative verification is performed strictly on Codebook 0.
This module synthesizes the remaining acoustic residual detail (Codebooks 1 through 8)
in parallel via a lightweight Non-Autoregressive (NAR) predictor conditioned on the verified Codebook 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from codeswitch_tts.config import ResidualModelConfig
from codeswitch_tts.models.draft_transformer import PositionalEncoding


# Exponentially growing receptive field.  The original four blocks covered
# ~61 frames; the capacity-bumped residual uses all six for ~253 frames.
_RESIDUAL_DILATIONS = (1, 2, 4, 8, 16, 32)


@dataclass
class ResidualOutput:
    """Output container for the Hierarchical Residual Decoder."""
    residual_codebooks: torch.Tensor  # (batch, 8, num_frames) - Codebooks 1-8
    full_codebooks: torch.Tensor      # (batch, 9, num_frames) - Codebooks 0-8 ready for DAC decoder
    logits: torch.Tensor              # (batch, 8, num_frames, 1024)


class ResidualBlock1D(nn.Module):
    """Dilated 1D Convolutional Residual Block with Pre-LayerNorm."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        residual = x
        h = self.conv1(self.act(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(self.act(self.norm2(h)))
        return residual + h


class ResidualDecoder(nn.Module):
    """Lightweight Non-Autoregressive Residual Codebook Predictor.
    
    Conditioned on verified Codebook 0, predicts Codebooks 1 to 8 simultaneously.
    """

    def __init__(self, config: Optional[ResidualModelConfig] = None):
        super().__init__()
        self.config = config or ResidualModelConfig()

        self.cb0_embedding = nn.Embedding(self.config.codebook_size, self.config.d_model)
        self.pos_encoder = PositionalEncoding(self.config.d_model, max_len=2048)

        # Dilated residual blocks with exponentially growing receptive field.
        # The config may request up to the six supported dilation rates.
        if self.config.num_layers > len(_RESIDUAL_DILATIONS):
            raise ValueError(
                f"ResidualModelConfig.num_layers={self.config.num_layers} exceeds "
                f"the {len(_RESIDUAL_DILATIONS)} supported dilations"
            )
        self.blocks = nn.ModuleList([
            ResidualBlock1D(
                channels=self.config.d_model,
                kernel_size=self.config.kernel_size,
                dilation=d,
                dropout=self.config.dropout,
            )
            for d in _RESIDUAL_DILATIONS[:self.config.num_layers]
        ])

        # 8 parallel heads: one for each residual codebook (1 through 8)
        self.heads = nn.ModuleList([
            nn.Conv1d(self.config.d_model, self.config.codebook_size, kernel_size=1)
            for _ in range(self.config.num_residual_codebooks)
        ])

        # Optional text/confidence conditioning.  ``None`` on legacy
        # checkpoints, so v2-v6 CB0-only state dicts still load unchanged.
        self.text_proj: Optional[nn.Module] = None
        self.confidence_embedding: Optional[nn.Embedding] = None
        self.fusion_gate: Optional[nn.Module] = None
        self.cb0_null_embedding: Optional[nn.Parameter] = None
        if bool(getattr(self.config, "text_conditioning", False)):
            self.text_proj = nn.Sequential(
                nn.Linear(int(self.config.text_d_model), self.config.d_model),
                nn.GELU(),
                nn.Linear(self.config.d_model, self.config.d_model),
            )
            self.cb0_null_embedding = nn.Parameter(torch.zeros(self.config.d_model))
        if bool(getattr(self.config, "confidence_conditioning", False)):
            self.confidence_embedding = nn.Embedding(
                int(self.config.confidence_bins), self.config.d_model
            )
        if self.text_proj is not None and self.confidence_embedding is not None:
            self.fusion_gate = nn.Sequential(
                nn.Linear(3 * self.config.d_model, self.config.d_model),
                nn.GELU(),
                nn.Linear(self.config.d_model, 1),
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cb0_embedding.weight, mean=0.0, std=0.02)
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        if self.cb0_null_embedding is not None:
            nn.init.zeros_(self.cb0_null_embedding)
        if self.confidence_embedding is not None:
            nn.init.normal_(self.confidence_embedding.weight, mean=0.0, std=0.02)

    def _confidence_bins(self, confidence: torch.Tensor) -> torch.Tensor:
        bins = max(1, int(self.config.confidence_bins))
        scaled = confidence.clamp(0.0, 1.0) * float(bins - 1)
        return scaled.round().long().clamp(0, bins - 1)

    def forward(
        self,
        cb0_tokens: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = False,
        text_features: Optional[torch.Tensor] = None,
        cb0_confidence: Optional[torch.Tensor] = None,
    ) -> ResidualOutput:
        """Forward pass predicting Codebooks 1 through 8 from verified Codebook 0.
        
        Args:
            cb0_tokens: (batch, num_frames) verified Codebook 0 tokens
            temperature: Sampling temperature
            sample: Whether to sample from categorical distribution or take argmax
            text_features: optional (batch, num_frames, text_d_model) frame-level
                text features.  Required when the config enables text
                conditioning; omitted for legacy CB0-only models.
            cb0_confidence: optional (batch, num_frames) in [0, 1].  The draft's
                argmax probability (or 1.0 for teacher CB0).  Used by the
                confidence-conditioned fusion gate.
            
        Returns:
            ResidualOutput containing residual_codebooks (B, 8, T) and full_codebooks (B, 9, T)
        """
        # Ensure 2D tensor: (batch, time)
        if cb0_tokens.dim() == 1:
            cb0_tokens = cb0_tokens.unsqueeze(0)
        elif cb0_tokens.dim() == 3 and cb0_tokens.size(1) == 1:
            cb0_tokens = cb0_tokens.squeeze(1)

        batch_size, time_steps = cb0_tokens.shape

        # 1. Embed Codebook 0 (and optionally text/confidence) and add positions
        cb0_is_null = cb0_tokens < 0
        safe_cb0 = cb0_tokens.clamp(min=0)
        cb0_embed = self.cb0_embedding(safe_cb0)  # (batch, time, d_model)
        if self.cb0_null_embedding is not None and bool(cb0_is_null.any()):
            null = self.cb0_null_embedding.to(cb0_embed.dtype)
            cb0_embed = torch.where(cb0_is_null.unsqueeze(-1), null, cb0_embed)

        text_embed = None
        if self.text_proj is not None and text_features is not None:
            if text_features.dim() != 3 or text_features.shape[:2] != (batch_size, time_steps):
                raise ValueError(
                    "text_features must be (batch, time, text_d_model); got "
                    f"{tuple(text_features.shape)} for cb0 {(batch_size, time_steps)}"
                )
            text_embed = self.text_proj(text_features.to(cb0_embed.dtype))
        elif self.text_proj is not None:
            # Explicit text ablation: run CB0-only without inventing features.
            text_embed = torch.zeros_like(cb0_embed)

        conf_embed = None
        if self.confidence_embedding is not None:
            if cb0_confidence is None:
                cb0_confidence = torch.ones(
                    (batch_size, time_steps), device=cb0_tokens.device, dtype=cb0_embed.dtype
                )
            else:
                cb0_confidence = cb0_confidence.to(device=cb0_tokens.device, dtype=cb0_embed.dtype)
                if cb0_confidence.shape != (batch_size, time_steps):
                    raise ValueError(
                        "cb0_confidence must match cb0_tokens; got "
                        f"{tuple(cb0_confidence.shape)} vs {(batch_size, time_steps)}"
                    )
            conf_embed = self.confidence_embedding(self._confidence_bins(cb0_confidence))

        if self.fusion_gate is not None and text_embed is not None and conf_embed is not None:
            gate = torch.sigmoid(
                self.fusion_gate(torch.cat([cb0_embed, text_embed, conf_embed], dim=-1))
            )
            x = gate * cb0_embed + (1.0 - gate) * text_embed + conf_embed
        else:
            x = cb0_embed
            if text_embed is not None:
                x = x + text_embed
            if conf_embed is not None:
                x = x + conf_embed

        x = self.pos_encoder(x)

        # 2. Run through Dilated Residual Backbone
        # Transpose to (batch, d_model, time) for Conv1D
        h = x.transpose(1, 2)
        for block in self.blocks:
            h = block(h)

        # 3. Parallel prediction across all 8 residual codebook heads
        head_logits: list[torch.Tensor] = []
        pred_tokens: list[torch.Tensor] = []

        for head in self.heads:
            # head(h): (batch, codebook_size, time)
            logits_i = head(h).transpose(1, 2)  # (batch, time, codebook_size)
            head_logits.append(logits_i.unsqueeze(1))  # (batch, 1, time, codebook_size)

            if sample:
                scaled_logits = logits_i / max(temperature, 1e-5)
                probs = F.softmax(scaled_logits, dim=-1)
                tokens_flat = torch.multinomial(probs.view(-1, self.config.codebook_size), num_samples=1)
                tokens_i = tokens_flat.view(batch_size, time_steps)
            else:
                tokens_i = torch.argmax(logits_i, dim=-1)  # (batch, time)

            pred_tokens.append(tokens_i.unsqueeze(1))  # (batch, 1, time)

        # 4. Concatenate residual heads: (batch, 8, time)
        residual_codebooks = torch.cat(pred_tokens, dim=1)
        stacked_logits = torch.cat(head_logits, dim=1)  # (batch, 8, time, 1024)

        # 5. Assemble full 9-codebook representation: (batch, 9, time)
        cb0_expanded = cb0_tokens.unsqueeze(1)  # (batch, 1, time)
        full_codebooks = torch.cat([cb0_expanded, residual_codebooks], dim=1)

        return ResidualOutput(
            residual_codebooks=residual_codebooks,
            full_codebooks=full_codebooks,
            logits=stacked_logits,
        )
