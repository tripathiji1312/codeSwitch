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

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cb0_embedding.weight, mean=0.0, std=0.02)
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(
        self,
        cb0_tokens: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> ResidualOutput:
        """Forward pass predicting Codebooks 1 through 8 from verified Codebook 0.
        
        Args:
            cb0_tokens: (batch, num_frames) verified Codebook 0 tokens
            temperature: Sampling temperature
            sample: Whether to sample from categorical distribution or take argmax
            
        Returns:
            ResidualOutput containing residual_codebooks (B, 8, T) and full_codebooks (B, 9, T)
        """
        # Ensure 2D tensor: (batch, time)
        if cb0_tokens.dim() == 1:
            cb0_tokens = cb0_tokens.unsqueeze(0)
        elif cb0_tokens.dim() == 3 and cb0_tokens.size(1) == 1:
            cb0_tokens = cb0_tokens.squeeze(1)

        batch_size, time_steps = cb0_tokens.shape

        # 1. Embed Codebook 0 and add positional encodings
        x = self.cb0_embedding(cb0_tokens)  # (batch, time, d_model)
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
