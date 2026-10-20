"""Discrete Codebook-0 Non-Autoregressive (NAR) Draft Transformer.

Architecture:
  - Text Tokenization & Character/Subword Embedding (vocab_size=4096, d_model=512)
  - Positional Encodings
  - 4-layer Transformer Encoder (d_model=512, 8 heads, d_ff=2048, ~65M parameters)
  - Temporal Length Projector (aligns text to 86 DAC frames per 1-second chunk)
  - Categorical Classification Head predicting DAC Codebook-0 distributions P̂(y_t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from codeswitch_tts.config import DraftModelConfig


@dataclass
class DraftOutput:
    """Output container for the Draft Transformer."""
    logits: torch.Tensor          # (batch, num_frames, codebook_size)
    probabilities: torch.Tensor   # (batch, num_frames, codebook_size)
    tokens: torch.Tensor          # (batch, num_frames) - sampled or argmax draft tokens


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device, x.dtype)


class CharacterTokenizer:
    """Robust UTF-8 and Tamil script tokenizer mapping characters to discrete IDs."""

    def __init__(self, vocab_size: int = 4096):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        ids = [self.bos_id]
        for ch in text:
            code = ord(ch)
            # Map ASCII (32..126) and Tamil Unicode (0x0B80..0x0BFF -> 2944..3071)
            if 32 <= code <= 126:
                idx = 10 + (code - 32)
            elif 0x0B80 <= code <= 0x0BFF:
                idx = 200 + (code - 0x0B80)
            else:
                idx = 400 + (code % (self.vocab_size - 400))
            ids.append(idx % self.vocab_size)
        ids.append(self.eos_id)
        return ids

    def batch_encode(self, texts: List[str], max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t) for t in texts]
        if max_len is None:
            max_len = max(len(seq) for seq in encoded)
        
        batch_size = len(texts)
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, seq in enumerate(encoded):
            truncated = seq[:max_len]
            input_ids[i, :len(truncated)] = torch.tensor(truncated, dtype=torch.long)
            attention_mask[i, :len(truncated)] = True

        return input_ids, attention_mask


class TemporalLengthProjector(nn.Module):
    """Projects text representation of arbitrary length to acoustic DAC frames (~86/s)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, target_frames: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        # x: (batch, text_len, d_model)
        # Transpose for 1D convolution: (batch, d_model, text_len)
        x_trans = x.transpose(1, 2)
        mask: Optional[torch.Tensor] = None
        if isinstance(target_frames, torch.Tensor):
            lengths = target_frames.to(device=x.device, dtype=torch.long).flatten()
            if lengths.numel() != x.shape[0]:
                raise ValueError(
                    f"target_frames has {lengths.numel()} entries for batch size "
                    f"{x.shape[0]}"
                )
            max_t = int(lengths.max().item())
            if max_t <= 0:
                raise ValueError("target_frames must be positive")
            # Each sample is interpolated to its own duration.  Padding after the
            # valid region is masked after every convolution (not just zeroed in
            # the input) so a batch's valid tail is identical to a one-sample run.
            interpolated = x_trans.new_zeros((x_trans.shape[0], x_trans.shape[1], max_t))
            for i, n in enumerate(lengths.tolist()):
                if n <= 0:
                    continue
                interpolated[i:i + 1, :, :n] = F.interpolate(
                    x_trans[i:i + 1], size=int(n), mode="linear", align_corners=False
                )
            mask = (
                torch.arange(max_t, device=x.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).unsqueeze(1)  # (B, 1, T)
        else:
            target_frames = int(target_frames)
            if target_frames <= 0:
                raise ValueError("target_frames must be positive")
            interpolated = F.interpolate(
                x_trans, size=target_frames, mode="linear", align_corners=False
            )
        # Refinement convolutions.  A conv's bias would otherwise leak into the
        # padded tail and change the last valid frame of the next conv.
        h = self.conv1(interpolated)
        if mask is not None:
            h = h * mask
        h = h.transpose(1, 2)
        h = self.act(self.norm(h))
        if mask is not None:
            h = h * mask.transpose(1, 2)
        h = self.conv2(h.transpose(1, 2))
        if mask is not None:
            h = h * mask
        return h.transpose(1, 2) + interpolated.transpose(1, 2)


class DraftTransformer(nn.Module):
    """Discrete Codebook-0 NAR Draft Model (~65M parameters).
    
    Predicts Codebook 0 DAC categorical distributions P̂(y_t) directly from text.
    """

    def __init__(self, config: Optional[DraftModelConfig] = None):
        super().__init__()
        self.config = config or DraftModelConfig()
        
        self.tokenizer = CharacterTokenizer(vocab_size=self.config.text_vocab_size)
        self.text_embedding = nn.Embedding(self.config.text_vocab_size, self.config.d_model)
        self.pos_encoder = PositionalEncoding(self.config.d_model, max_len=self.config.max_text_len)

        # 4-Layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.d_model),
            enable_nested_tensor=False,
        )

        # Temporal expansion module (Text -> Audio Frame Domain)
        self.temporal_projector = TemporalLengthProjector(self.config.d_model)

        # Codebook 0 Categorical Head (vocab = 1024)
        self.cb0_head = nn.Linear(self.config.d_model, self.config.codebook_size)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cb0_head.weight, mean=0.0, std=0.02)
        if self.cb0_head.bias is not None:
            nn.init.zeros_(self.cb0_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        temperature: float = 1.0,
        sample: bool = True,
        frame_lengths: Optional[torch.Tensor] = None,
    ) -> DraftOutput:
        """Forward pass generating Codebook 0 categorical distributions.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) boolean mask
            num_frames: Target number of audio frames for every sample
                (default: config.frames_per_chunk = 86)
            temperature: Sampling temperature
            sample: Whether to sample from categorical distribution or take argmax
            frame_lengths: Optional (batch,) per-sample target frame counts.
                When supplied, each sample is projected to its own duration and
                the output is padded to the batch maximum.  Training must use
                this instead of ``num_frames=max_t`` to avoid text stretching.
            
        Returns:
            DraftOutput with logits, probabilities, and discrete draft tokens
        """
        target_frames: Union[int, torch.Tensor]
        if frame_lengths is not None:
            target_frames = torch.as_tensor(frame_lengths, device=input_ids.device)
        else:
            target_frames = num_frames or self.config.frames_per_chunk
        
        # 1. Text embedding & positional encoding
        x = self.text_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        # 2. Transformer Encoder
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        h = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 3. Temporal Projector to 1-second audio frame rate (~86 frames)
        frame_features = self.temporal_projector(h, target_frames=target_frames)

        # 4. Predict Codebook 0 categorical distributions
        logits = self.cb0_head(frame_features)  # (batch, target_frames, 1024)

        # 5. Compute probabilities P̂(y_t)
        scaled_logits = logits / max(temperature, 1e-5)
        probs = F.softmax(scaled_logits, dim=-1)

        # 6. Sample or argmax draft tokens
        if sample:
            b, t, v = probs.shape
            probs_flat = probs.view(-1, v)
            tokens_flat = torch.multinomial(probs_flat, num_samples=1)
            tokens = tokens_flat.view(b, t)
        else:
            tokens = torch.argmax(probs, dim=-1)

        return DraftOutput(
            logits=logits,
            probabilities=probs,
            tokens=tokens,
        )

    def generate_chunk(
        self,
        text: Union[str, List[str]],
        num_frames: Optional[int] = None,
        temperature: float = 1.0,
        sample: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> DraftOutput:
        """Helper method to generate draft Codebook 0 tokens directly from raw text string(s)."""
        if isinstance(text, str):
            text = [text]
        
        input_ids, mask = self.tokenizer.batch_encode(text)
        if device is not None:
            input_ids = input_ids.to(device)
            mask = mask.to(device)

        return self.forward(
            input_ids=input_ids,
            attention_mask=mask,
            num_frames=num_frames,
            temperature=temperature,
            sample=sample,
        )
