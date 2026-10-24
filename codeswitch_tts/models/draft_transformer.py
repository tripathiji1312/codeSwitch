"""Discrete Codebook-0 Non-Autoregressive (NAR) Draft Transformer.

Architecture:
  - Text Tokenization & Character/Subword Embedding (vocab_size=4096, d_model=512)
  - Positional Encodings
  - 4-layer Transformer Encoder (d_model=512, 8 heads, d_ff=2048, ~65M parameters)
  - Temporal alignment: either the legacy nearest-neighbour length projector or
    an opt-in frame-query decoder that self-attends over acoustic frames and
    cross-attends to the text encoder memory
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
    duration_logits: Optional[torch.Tensor] = None  # (batch, text_len) log1p(frames)
    frame_durations: Optional[torch.Tensor] = None  # (batch, text_len) int frame counts


# ===========================================================================
# Duration utilities
# ===========================================================================
def normalize_token_durations(
    durations: torch.Tensor,
    lengths: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Turn positive per-token durations into integer frame counts.

    Every row sums to exactly ``lengths[b]``.  Largest-remainder rounding is
    used per sample so the length regulator agrees with the acoustic target.
    ``valid_mask`` should exclude padding and BOS/EOS; those tokens receive 0.
    """
    if durations.ndim != 2:
        raise ValueError(f"durations must be 2-D, got {tuple(durations.shape)}")
    batch, seq_len = durations.shape
    lengths = torch.as_tensor(lengths, device=durations.device, dtype=torch.long).flatten()
    if lengths.numel() != batch:
        raise ValueError(f"lengths has {lengths.numel()} entries for batch {batch}")
    if valid_mask is None:
        valid_mask = torch.ones_like(durations, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=durations.device, dtype=torch.bool)
        if valid_mask.shape != durations.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} != durations "
                f"{tuple(durations.shape)}"
            )

    positive = torch.clamp(durations.float(), min=0.0) * valid_mask.float()
    out = torch.zeros((batch, seq_len), device=durations.device, dtype=torch.long)
    for row in range(batch):
        target = int(lengths[row].item())
        if target <= 0:
            continue
        valid = valid_mask[row]
        if not bool(valid.any()):
            out[row, 0] = target
            continue
        values = positive[row]
        total = float(values.sum().item())
        if total <= 0.0:
            values = valid.float() * (float(target) / float(valid.sum().item()))
        else:
            values = values * (float(target) / total)
        floors = torch.floor(values).long()
        remainder = target - int(floors.sum().item())
        if remainder > 0:
            fractional = values - floors.float()
            fractional = torch.where(
                valid, fractional, torch.full_like(fractional, -1.0)
            )
            order = torch.argsort(fractional, descending=True, stable=True)
            for index in order.tolist():
                if remainder <= 0:
                    break
                if bool(valid[index]):
                    floors[index] += 1
                    remainder -= 1
        # Defensive if every fractional part was zero (integer-valued seconds).
        cursor = 0
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        while remainder > 0 and valid_indices:
            floors[valid_indices[cursor % len(valid_indices)]] += 1
            remainder -= 1
            cursor += 1
        out[row] = floors
    return out


def expand_text_by_durations(
    memory: torch.Tensor,
    token_durations: torch.Tensor,
    max_frames: int,
) -> torch.Tensor:
    """Repeat each text position by its duration to build per-frame features."""
    if memory.shape[:2] != token_durations.shape:
        raise ValueError(
            f"memory {tuple(memory.shape[:2])} and durations "
            f"{tuple(token_durations.shape)} disagree"
        )
    batch, _, d_model = memory.shape
    indices = []
    for row in range(batch):
        counts = token_durations[row].clamp(min=0).to(dtype=torch.long)
        index = torch.repeat_interleave(
            torch.arange(counts.numel(), device=memory.device), counts
        )
        if index.numel() >= max_frames:
            index = index[:max_frames]
        elif index.numel() > 0:
            pad = index[-1].expand(max_frames - index.numel())
            index = torch.cat((index, pad), dim=0)
        else:
            index = torch.zeros(max_frames, device=memory.device, dtype=torch.long)
        indices.append(index)
    index_tensor = torch.stack(indices, dim=0)  # (B, max_frames)
    return torch.gather(
        memory,
        1,
        index_tensor.unsqueeze(-1).expand(-1, -1, d_model),
    )


class DurationPredictor(nn.Module):
    """Predict log1p(per-character DAC frames) from text encoder states."""

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.net(memory).squeeze(-1)


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


class FrameDecoder(nn.Module):
    """Text-conditioned non-autoregressive acoustic-frame decoder.

    Each acoustic frame starts as a learned query plus a positional code and a
    whole-utterance length embedding.  Decoder self-attention lets frames see
    one another, while cross-attention reads the text encoder memory.  This is
    deliberately opt-in: the legacy length projector is kept byte-for-byte so
    older checkpoints continue to load without new keys.
    """

    def __init__(self, config: DraftModelConfig):
        super().__init__()
        if config.frame_decoder_layers < 1:
            raise ValueError("FrameDecoder requires frame_decoder_layers >= 1")
        if config.max_frames < 1:
            raise ValueError("max_frames must be positive")

        self.d_model = config.d_model
        self.max_frames = config.max_frames
        self.frame_queries = nn.Parameter(
            torch.empty(1, self.max_frames, self.d_model)
        )
        nn.init.normal_(self.frame_queries, mean=0.0, std=0.02)

        self.pos_encoder = PositionalEncoding(
            self.d_model, max_len=self.max_frames
        )
        self.length_embedding = nn.Embedding(self.max_frames + 1, self.d_model)
        nn.init.normal_(self.length_embedding.weight, mean=0.0, std=0.02)

        # Only created when duration conditioning is explicitly enabled.  This
        # keeps v5 state dicts byte-compatible.
        self.duration_proj: Optional[nn.Module] = None
        if bool(getattr(config, "duration_conditioning", False)):
            self.duration_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.frame_decoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        lengths: torch.Tensor,
        token_durations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode frames for a batch.

        Args:
            memory: (batch, text_len, d_model) encoded text.
            memory_key_padding_mask: (batch, text_len) True for text padding.
            lengths: (batch,) positive, per-sample frame counts.
            token_durations: optional (batch, text_len) integer frame counts.
                When supplied (and duration conditioning is enabled), each
                frame query receives the text state for its aligned character.
        """
        if lengths.ndim != 1 or lengths.numel() != memory.shape[0]:
            raise ValueError(
                f"lengths must have shape ({memory.shape[0]},), got "
                f"{tuple(lengths.shape)}"
            )
        if int(lengths.min().item()) <= 0:
            raise ValueError("frame lengths must be positive")
        max_t = int(lengths.max().item())
        if max_t > self.max_frames:
            raise ValueError(
                f"frame length {max_t} exceeds max_frames={self.max_frames}"
            )

        batch = memory.shape[0]
        queries = self.frame_queries[:, :max_t, :].expand(batch, -1, -1)
        queries = self.pos_encoder(queries)
        queries = queries + self.length_embedding(lengths).unsqueeze(1)

        if token_durations is not None and self.duration_proj is not None:
            aligned = expand_text_by_durations(memory, token_durations, max_t)
            queries = queries + self.duration_proj(aligned)

        positions = torch.arange(max_t, device=memory.device).unsqueeze(0)
        tgt_key_padding_mask = positions >= lengths.unsqueeze(1)
        return self.decoder(
            queries,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


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

        # Optional text-conditioned frame decoder.  None on legacy checkpoints.
        self.frame_decoder: Optional[FrameDecoder] = None
        if self.config.frame_decoder_layers < 0:
            raise ValueError("frame_decoder_layers must be >= 0")
        if self.config.frame_decoder_layers > 0:
            self.frame_decoder = FrameDecoder(self.config)
        if self.config.duration_conditioning and self.frame_decoder is None:
            raise ValueError(
                "duration_conditioning requires frame_decoder_layers >= 1"
            )

        # Optional FastSpeech-2-style duration head.  Trained with CTC-derived
        # durations; used at inference to build the frame layout without an
        # external aligner.
        self.duration_predictor: Optional[DurationPredictor] = None
        if self.config.duration_conditioning:
            self.duration_predictor = DurationPredictor(
                self.config.d_model,
                hidden=self.config.duration_hidden,
                dropout=self.config.dropout,
            )

        # Codebook 0 Categorical Head (vocab = 1024)
        self.cb0_head = nn.Linear(self.config.d_model, self.config.codebook_size)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cb0_head.weight, mean=0.0, std=0.02)
        if self.cb0_head.bias is not None:
            nn.init.zeros_(self.cb0_head.bias)

    @staticmethod
    def _duration_valid_mask(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            valid = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        # PAD=0, BOS=1, EOS=2.  Character tokens, including <unk>=3, may carry
        # duration; padding/sentinels must not.
        return valid & (input_ids > 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        temperature: float = 1.0,
        sample: bool = True,
        frame_lengths: Optional[torch.Tensor] = None,
        token_durations: Optional[torch.Tensor] = None,
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
                When supplied, each sample is decoded/projected to its own
                duration and the output is padded to the batch maximum.
                Training must use this instead of ``num_frames=max_t`` to avoid
                text stretching.
            token_durations: Optional (batch, seq_len) ground-truth frame
                counts.  Used to lay out the frame queries during training
                (oracle alignment); when omitted, the duration head predicts
                its own layout.
            
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

        # 3. Text -> acoustic-frame alignment.
        # The legacy projector interpolates text features.  The opt-in decoder
        # instead gives every frame its own query and cross-attends to text.
        if self.frame_decoder is not None:
            if isinstance(target_frames, torch.Tensor):
                lengths = target_frames.to(device=input_ids.device, dtype=torch.long).flatten()
            else:
                lengths = torch.full(
                    (input_ids.shape[0],), int(target_frames),
                    device=input_ids.device, dtype=torch.long,
                )

            duration_logits: Optional[torch.Tensor] = None
            frame_durations: Optional[torch.Tensor] = None
            if self.duration_predictor is not None:
                duration_logits = self.duration_predictor(h)
                if token_durations is not None:
                    token_durations = torch.as_tensor(
                        token_durations, device=input_ids.device, dtype=torch.float32
                    )
                    if token_durations.shape != input_ids.shape:
                        raise ValueError(
                            "token_durations shape must match input_ids: "
                            f"{tuple(token_durations.shape)} vs "
                            f"{tuple(input_ids.shape)}"
                        )
                    frame_durations = normalize_token_durations(
                        token_durations,
                        lengths,
                        valid_mask=self._duration_valid_mask(
                            input_ids, attention_mask
                        ),
                    )
                else:
                    # Training targets log1p(frame counts), so invert that
                    # transform here.  softplus(log1p(x)) is not the inverse
                    # and would badly distort the character layout.
                    predicted = torch.expm1(duration_logits).clamp(min=1e-4)
                    frame_durations = normalize_token_durations(
                        predicted,
                        lengths,
                        valid_mask=self._duration_valid_mask(
                            input_ids, attention_mask
                        ),
                    )
            frame_features = self.frame_decoder(
                h,
                memory_key_padding_mask=src_key_padding_mask,
                lengths=lengths,
                token_durations=frame_durations,
            )
        else:
            if token_durations is not None:
                raise ValueError(
                    "token_durations requires the frame decoder; "
                    "set frame_decoder_layers >= 1"
                )
            if not isinstance(target_frames, torch.Tensor) and int(target_frames) <= 0:
                raise ValueError("target_frames must be positive")
            duration_logits = None
            frame_durations = None
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
            duration_logits=duration_logits,
            frame_durations=frame_durations,
        )

    def generate_chunk(
        self,
        text: Union[str, List[str]],
        num_frames: Optional[int] = None,
        temperature: float = 1.0,
        sample: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        token_durations: Optional[torch.Tensor] = None,
    ) -> DraftOutput:
        """Helper method to generate draft Codebook 0 tokens directly from raw text string(s)."""
        if isinstance(text, str):
            text = [text]
        
        input_ids, mask = self.tokenizer.batch_encode(text)
        if device is not None:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            if token_durations is not None:
                token_durations = token_durations.to(device)

        return self.forward(
            input_ids=input_ids,
            attention_mask=mask,
            num_frames=num_frames,
            temperature=temperature,
            sample=sample,
            token_durations=token_durations,
        )
