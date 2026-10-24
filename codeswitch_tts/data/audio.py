"""Small audio helpers shared by the real-speech pipeline and aligners.

The training/evaluation code already depends on ``soundfile``, ``scipy`` and
``torchaudio``.  This module keeps the data-prep path free of librosa and of
any assumption about the source dataset's sampling rate.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def resample_audio(
    audio: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    """Resample mono float audio with a polyphase filter.

    ``scipy`` is already a runtime dependency.  Returning float32 matches the
    DAC and Wav2Vec2 feature extractors.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sample_rate = int(sample_rate)
    target_sample_rate = int(target_sample_rate)
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if sample_rate == target_sample_rate or audio.size == 0:
        return audio

    from scipy.signal import resample_poly

    divisor = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = sample_rate // divisor
    return np.asarray(resample_poly(audio, up, down), dtype=np.float32)


def _as_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # soundfile/torchaudio return (frames, channels); average channels.
        audio = audio.mean(axis=1)
    return audio.reshape(-1)


def _decode_with_torchaudio(payload: Any) -> Tuple[np.ndarray, int]:
    import torchaudio

    if isinstance(payload, (bytes, bytearray)):
        waveform, sample_rate = torchaudio.load(io.BytesIO(payload))
    else:
        waveform, sample_rate = torchaudio.load(str(payload))
    array = waveform.detach().cpu().numpy()
    if array.ndim == 2:
        array = array.mean(axis=0)
    return _as_mono(array), int(sample_rate)


def decode_audio(value: Any, base_dir: Optional[Path] = None) -> Tuple[np.ndarray, int]:
    """Decode a dataset audio cell into ``(mono float32 array, sample_rate)``.

    Handles the shapes produced by Hugging Face ``datasets`` and by raw
    parquet rows:

    * ``{"array": ..., "sampling_rate": ...}``
    * ``{"bytes": ...}`` or ``{"path": ...}``
    * ``torchcodec.AudioDecoder`` (``datasets`` >= 4 with torchcodec)
    * a filesystem path
    * ``(array, sample_rate)``

    ``soundfile`` is tried first (WAV/FLAC/OGG).  ``torchaudio`` is the
    fallback for containers such as MP3 that libsndfile cannot open.
    """
    if value is None:
        raise ValueError("audio cell is empty")

    # Newer ``datasets`` releases return a torchcodec decoder rather than the
    # historical ``{"array": ..., "sampling_rate": ...}`` dict.  The decoder
    # exposes the waveform through ``get_all_samples()``.
    if hasattr(value, "get_all_samples") and callable(value.get_all_samples):
        samples = value.get_all_samples()
        array = getattr(samples, "data", None)
        sample_rate = getattr(samples, "sample_rate", None)
        if sample_rate is None:
            metadata = getattr(value, "metadata", None)
            sample_rate = getattr(metadata, "sample_rate", None)
        if array is None or sample_rate is None:
            raise ValueError("torchcodec audio cell has no samples or sample rate")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"torchcodec audio cell has invalid sample rate {sample_rate}")
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = np.asarray(array)
        if array.ndim == 2:
            # torchcodec returns (channels, frames).
            array = array.mean(axis=0)
        return _as_mono(array), sample_rate

    if isinstance(value, tuple) and len(value) == 2:
        array, sample_rate = value
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"audio tuple has invalid sample rate {sample_rate}")
        return _as_mono(array), sample_rate

    if isinstance(value, dict):
        if value.get("array") is not None:
            sample_rate = int(
                value.get("sampling_rate") or value.get("sample_rate") or 0
            )
            if sample_rate <= 0:
                raise ValueError(
                    "audio array cell has no sampling_rate; cannot resample safely"
                )
            return _as_mono(value["array"]), sample_rate
        if value.get("bytes") is not None:
            payload: Any = value["bytes"]
            try:
                import soundfile as sf

                array, sample_rate = sf.read(io.BytesIO(payload), always_2d=False)
                return _as_mono(array), int(sample_rate)
            except Exception:
                return _decode_with_torchaudio(payload)
        if value.get("path") is not None:
            path = Path(str(value["path"]))
            if not path.is_absolute() and base_dir is not None:
                path = Path(base_dir) / path
            value = path
        else:
            raise ValueError(f"unsupported audio cell keys: {sorted(value)}")

    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        if not path.is_file():
            raise FileNotFoundError(f"audio path not found: {path}")
        try:
            import soundfile as sf

            array, sample_rate = sf.read(str(path), always_2d=False)
            return _as_mono(array), int(sample_rate)
        except Exception:
            return _decode_with_torchaudio(path)

    if hasattr(value, "__array__"):
        raise ValueError(
            "audio array without a sampling rate; expected an (array, rate) tuple"
        )
    raise ValueError(f"unsupported audio cell type: {type(value)!r}")
