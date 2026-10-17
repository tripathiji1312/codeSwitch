"""Configuration module for CodeSwitch-TTS.

Defines sample rates, DAC neural codec parameters, model dimensions,
memory envelopes, and streaming orchestrator constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class AudioConfig:
    """Acoustic and DAC codec configuration."""
    sample_rate: int = 44100
    dac_hop_length: int = 512
    frame_rate: float = 44100.0 / 512.0  # ~86.1328 frames/sec
    chunk_duration_sec: float = 1.0
    frames_per_chunk: int = 86  # 1 second of DAC frames
    num_codebooks: int = 9
    codebook_size: int = 1024
    highpass_cutoff_hz: float = 60.0  # 4th-order Butterworth filter to prevent DC/sub-bass rumble
    crossfade_samples: int = 256  # ~5.8ms crossfade between chunks
    audio_channels: int = 1


@dataclass
class DraftModelConfig:
    """Discrete Codebook-0 NAR Draft Transformer architecture (~65M parameters)."""
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1
    text_vocab_size: int = 4096
    codebook_size: int = 1024  # Codebook 0 vocabulary
    max_text_len: int = 512
    frames_per_chunk: int = 86


@dataclass
class ResidualModelConfig:
    """Hierarchical Residual Decoder architecture for Codebooks 1-8."""
    d_model: int = 384
    num_layers: int = 4
    num_residual_codebooks: int = 8  # Codebooks 1 through 8
    codebook_size: int = 1024
    kernel_size: int = 5
    dropout: float = 0.1


@dataclass
class SpeculativeEngineConfig:
    """Speculative decoding hyperparameters."""
    target_model_id: str = "ai4bharat/indic-parler-tts"
    voice_description: str = (
        "Jaya speaks with a slightly low-pitched voice at a moderate pace. "
        "The recording is very high quality, clear and close-sounding, with no background noise."
    )
    temperature: float = 1.0
    max_chunk_frames: int = 86
    speculative_window: int = 16  # Verification batch slice length
    chars_per_sec: float = 13.0
    generation_seed: int = 42


@dataclass
class SystemConfig:
    """Hardware, precision, and VRAM configuration."""
    device: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    torch_dtype: torch.dtype = field(init=False)
    max_vram_gb: float = 12.0  # Fit comfortably inside 16GB T4 GPU
    output_dir: str = "output_audio"
    # Safety switch: the standalone fallback modules have random weights, so the
    # waveform they produce is noise, not speech. Synthesis refuses to run in
    # that state unless this is explicitly enabled (telemetry-only runs).
    allow_untrained_fallback: bool = False

    def __post_init__(self):
        if self.device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32


@dataclass
class CodeSwitchTTSConfig:
    """Root configuration aggregator."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    draft: DraftModelConfig = field(default_factory=DraftModelConfig)
    residual: ResidualModelConfig = field(default_factory=ResidualModelConfig)
    speculative: SpeculativeEngineConfig = field(default_factory=SpeculativeEngineConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


# Global default instance
DEFAULT_CONFIG = CodeSwitchTTSConfig()
