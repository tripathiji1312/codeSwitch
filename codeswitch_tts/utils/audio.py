"""Audio processing, DAC neural codec decoding, and acoustic post-processing.

Handles:
  - 44.1kHz Descript Audio Codec (DAC) decoding from 9 codebooks
  - 4th-order Butterworth high-pass filtering at 60Hz (eliminates DC rumble)
  - Inter-chunk cross-fading for seam-free streaming
  - Peak normalization and 16-bit PCM WAV exporting
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn

from codeswitch_tts.config import AudioConfig, DEFAULT_CONFIG


DAC_PRETRAINED_ID = "ylacombe/dac_44khz"


def apply_highpass_filter(
    audio: np.ndarray,
    cutoff_hz: float = 60.0,
    sample_rate: int = 44100,
    order: int = 4,
) -> np.ndarray:
    """Applies a 4th-order Butterworth high-pass filter.
    
    Removes low-frequency DAC neural codec rumble (<60Hz) while preserving
    fundamental speech frequencies (Tamil speech range: ~85-300Hz).
    """
    if len(audio) < order * 3:
        return audio
    sos = scipy.signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, audio)
    return filtered.astype(np.float32)


def crossfade_chunks(
    chunk1: np.ndarray,
    chunk2: np.ndarray,
    crossfade_samples: int = 256,
) -> np.ndarray:
    """Applies a linear crossfade at the boundary of chunk1 and chunk2."""
    if len(chunk1) < crossfade_samples or len(chunk2) < crossfade_samples:
        return np.concatenate([chunk1, chunk2])

    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

    tail = chunk1[-crossfade_samples:] * fade_out + chunk2[:crossfade_samples] * fade_in
    chunk1_trimmed = chunk1[:-crossfade_samples]
    chunk2_trimmed = chunk2[crossfade_samples:]
    blended = np.concatenate([chunk1_trimmed, tail, chunk2_trimmed])
    return blended



def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Soft peak normalization to prevent digital clipping."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return audio * (target_peak / peak)
    return audio


def save_wav(
    filepath: Union[str, Path],
    audio_data: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 44100,
    normalize: bool = True,
):
    """Saves audio data to a 16-bit PCM WAV file."""
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().float().numpy()

    audio_data = audio_data.flatten()
    if normalize:
        audio_data = normalize_peak(audio_data)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(filepath), audio_data, sample_rate, subtype="PCM_16")


class DACWrapper:
    """Neural codec decoder wrapping DAC 44.1kHz.

    Decodes full 9-codebook discrete tokens (B, 9, T) into 44.1kHz audio waveforms.
    Pretrained weights are mandatory unless the caller explicitly opts into the
    synthetic telemetry-only fallback.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        model: Optional[Any] = None,
        allow_fallback: bool = False,
    ):
        self.config = config or DEFAULT_CONFIG.audio
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.allow_fallback = bool(allow_fallback)
        self.load_error: Optional[str] = None
        if model is not None:
            self.model = model
            self.backend = "passed_model"
        else:
            self.model = None
            self.backend = "none"
            self._load_dac()

    def _load_dac(self):
        """Load a real DAC decoder, or refuse/opt into the synthetic fallback.

        Never silently installs random weights: the previous ``hf_local`` branch
        produced white noise for every diagnostic wav.
        """
        errors: List[str] = []

        # Attempt 1: transformers-native DAC (223-tensor layout; the same weights
        # Parler's audio_encoder carries, but loadable without Parler).
        try:
            from transformers import DacModel

            self.model = DacModel.from_pretrained(DAC_PRETRAINED_ID).to(self.device)
            self.model.eval()
            self.backend = "hf_pretrained"
            return
        except Exception as exc:
            errors.append(
                f"transformers DacModel({DAC_PRETRAINED_ID}): "
                f"{type(exc).__name__}: {exc}"
            )

        # Attempt 2: official descript-audio-codec package (DAC package layout).
        try:
            import dac
            from dac.model import DAC

            model_path = dac.utils.download(model_type="44khz")
            self.model = DAC.load(model_path).to(self.device)
            self.model.eval()
            self.backend = "dac_official"
            return
        except Exception as exc:
            errors.append(f"descript-audio-codec: {type(exc).__name__}: {exc}")

        self.model = None
        self.load_error = "; ".join(errors)
        if not self.allow_fallback:
            self.backend = "unavailable"
            raise RuntimeError(
                "No pretrained DAC weights available; refusing to install random "
                f"weights. Tried: {self.load_error}. Pass allow_fallback=True only "
                "for telemetry-only runs."
            )
        self.backend = "fallback_neural"

    @torch.inference_mode()
    def decode(self, codebooks: torch.Tensor) -> np.ndarray:
        """Decodes 9-codebook discrete tokens into 44.1kHz PCM audio.
        
        Args:
            codebooks: (batch, 9, num_frames) or (9, num_frames) discrete indices in [0, 1023]
            
        Returns:
            np.ndarray: Audio waveform of shape (audio_samples,) at 44.1kHz
        """
        if codebooks.dim() == 2:
            codebooks = codebooks.unsqueeze(0)  # Add batch dim -> (1, 9, T)
        
        batch_size, num_cbs, num_frames = codebooks.shape
        codebooks = codebooks.to(self.device)

        if self.backend in ("passed_model", "hf_pretrained") and self.model is not None:
            try:
                out = self.model.decode(audio_codes=codebooks)
                if hasattr(out, "audio_values"):
                    audio = out.audio_values
                    while audio.dim() > 1:
                        audio = audio[0]
                    return audio.cpu().float().numpy()
            except Exception:
                pass
            try:
                out = self.model.decode(codebooks.unsqueeze(0).unsqueeze(0), audio_scales=[None])
                if hasattr(out, "audio_values"):
                    audio = out.audio_values
                    while audio.dim() > 1:
                        audio = audio[0]
                    return audio.cpu().float().numpy()
            except Exception:
                pass

        if self.backend == "dac_official" and self.model is not None:
            try:
                z, _, _ = self.model.quantizer.from_codes(codebooks)
                audio_tensor = self.model.decode(z)
                return audio_tensor[0, 0].cpu().float().numpy()
            except Exception:
                pass

        if self.backend != "fallback_neural":
            raise RuntimeError(
                "DACWrapper has no usable DAC backend: "
                f"{self.backend} ({self.load_error or 'no loader details'})"
            )

        # Graceful synthetic acoustic decoder fallback:
        # Generates smooth continuous harmonic audio driven by Codebook 0 (pitch/harmonics)
        # and Codebooks 1-8 (formant textures)
        hop = self.config.dac_hop_length
        total_samples = num_frames * hop
        t = np.linspace(0, num_frames / self.config.frame_rate, total_samples, endpoint=False)
        
        # Fundamental frequency modulated by Codebook 0 tokens
        cb0 = codebooks[0, 0].cpu().numpy().astype(np.float32)
        f0_envelope = 130.0 + (cb0 % 120.0)  # Human voice range ~130-250 Hz
        # Interpolate f0 envelope to full audio samples
        f0_curve = np.interp(np.linspace(0, num_frames, total_samples), np.arange(num_frames), f0_envelope)
        
        phase = 2 * np.pi * np.cumsum(f0_curve) / self.config.sample_rate
        # Harmonics
        waveform = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
        
        # Modulate amplitude with residual codebook energy
        cb_res = codebooks[0, 1:].float().mean(dim=0).cpu().numpy()
        amp_envelope = 0.3 + 0.7 * (cb_res % 100.0) / 100.0
        amp_curve = np.interp(np.linspace(0, num_frames, total_samples), np.arange(num_frames), amp_envelope)
        waveform = waveform * amp_curve

        return waveform.astype(np.float32)


class AudioProcessor:
    """End-to-end audio post-processor."""

    def __init__(self, config: Optional[AudioConfig] = None,
                 dac_model: Optional[Any] = None, allow_fallback: bool = False):
        self.config = config or DEFAULT_CONFIG.audio
        self.dac = DACWrapper(
            config=self.config, model=dac_model, allow_fallback=allow_fallback
        )

    def process_and_filter(self, raw_audio: np.ndarray) -> np.ndarray:
        """Applies 60Hz high-pass filtering and peak normalization."""
        filtered = apply_highpass_filter(
            raw_audio,
            cutoff_hz=self.config.highpass_cutoff_hz,
            sample_rate=self.config.sample_rate,
        )
        normalized = normalize_peak(filtered, target_peak=0.92)
        return normalized

    def decode_and_postprocess(self, codebooks: torch.Tensor) -> np.ndarray:
        """Decodes 9-codebook tensor and applies acoustic post-processing."""
        raw_audio = self.dac.decode(codebooks)
        return self.process_and_filter(raw_audio)
