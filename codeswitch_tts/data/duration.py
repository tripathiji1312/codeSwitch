"""CTC forced alignment -> per-character durations in seconds.

Why this module exists
----------------------
The v4/v5 draft learned the marginal CB0 distribution instead of text
conditioning on ~1,800 utterances.  Duration supervision gives the frame
decoder an explicit text-to-frame layout, the same lever FastSpeech 2 uses to
become sample-efficient.

The aligner is deliberately model-agnostic: any Hugging Face
``AutoModelForCTC`` with a character tokenizer works (Tamil Wav2Vec2, English
Wav2Vec2, IndicWav2Vec, ...).  The Viterbi core is pure NumPy so it can be
tested without downloading weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from codeswitch_tts.data.audio import resample_audio


@dataclass(frozen=True)
class DurationAlignment:
    """Result of aligning one utterance.

    ``char_durations_sec`` has exactly ``len(text)`` entries.  Characters the
    CTC vocabulary cannot represent (punctuation, foreign script) get 0.0 and
    lower ``coverage``.  The caller converts seconds to DAC frames once it knows
    the codec's frame count.
    """

    char_durations_sec: np.ndarray
    score: float
    coverage: float
    model_id: str
    n_ctc_frames: int


class DurationExtractor(Protocol):
    """The seam the real-speech pipeline depends on."""

    model_id: str

    def align(
        self, audio: np.ndarray, sample_rate: int, text: str
    ) -> Optional[DurationAlignment]:
        ...


# ===========================================================================
# Text -> CTC label sequence
# ===========================================================================
def map_text_to_ctc_labels(
    text: str, vocab: Dict[str, int]
) -> Tuple[List[int], List[int], int]:
    """Map characters to CTC token ids.

    Returns ``(label_ids, text_positions, n_content_chars)``.  Only exact
    vocabulary hits are accepted: ``convert_tokens_to_ids`` would silently map
    unknown characters to ``<unk>`` and destroy the alignment.  Spaces map to
    the Wav2Vec2 word delimiter ``"|"``.
    """
    labels: List[int] = []
    positions: List[int] = []
    n_content = 0
    for index, char in enumerate(str(text)):
        if char.isspace():
            token = "|"
        else:
            n_content += 1
            token = char
        token_id = vocab.get(token)
        if token_id is None and not char.isspace():
            # English CTC vocabularies are usually uppercase.
            token_id = vocab.get(char.upper())
        if token_id is None and not char.isspace():
            token_id = vocab.get(char.lower())
        if token_id is None:
            continue
        labels.append(int(token_id))
        positions.append(index)
    return labels, positions, n_content


# ===========================================================================
# CTC Viterbi forced alignment
# ===========================================================================
def ctc_forced_align(
    log_probs: np.ndarray,
    labels: Sequence[int],
    blank_id: int = 0,
) -> Optional[Tuple[np.ndarray, float]]:
    """Return ``(frames_per_label, log_prob_per_frame)`` or ``None``.

    This is the standard CTC forward/Viterbi recurrence over the blank-
    interleaved label sequence.  It is exact for a fixed transcript and is
    intentionally non-batched: callers process one utterance at a time and the
    Viterbi loop is vectorised across states.
    """
    log_probs = np.asarray(log_probs, dtype=np.float32)
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must be 2-D, got {log_probs.shape}")
    labels = np.asarray(list(labels), dtype=np.int64)
    n_frames, vocab_size = log_probs.shape
    n_labels = int(labels.size)
    if n_frames <= 0:
        return None
    if n_labels == 0:
        return np.zeros(0, dtype=np.int64), 0.0
    if n_labels > n_frames:
        # CTC cannot emit more labels than frames.
        return None
    if int(labels.min()) < 0 or int(labels.max()) >= vocab_size:
        raise ValueError("label id outside the logits vocabulary")
    if not (0 <= int(blank_id) < vocab_size):
        raise ValueError("blank_id outside the logits vocabulary")

    n_states = 2 * n_labels + 1
    extended = np.full(n_states, int(blank_id), dtype=np.int64)
    extended[1::2] = labels

    neg_inf = np.float32(-1e30)
    alpha = np.full((n_frames, n_states), neg_inf, dtype=np.float32)
    back = np.zeros((n_frames, n_states), dtype=np.uint8)

    alpha[0, 0] = log_probs[0, int(blank_id)]
    if n_states > 1:
        alpha[0, 1] = log_probs[0, int(labels[0])]

    # Skip transitions (s-2) are legal only into a label state that is
    # different from the previous label; repeated labels need a blank between.
    skip_allowed = np.zeros(n_states, dtype=bool)
    if n_states > 2:
        skip_allowed[2:] = (extended[2:] != int(blank_id)) & (
            extended[2:] != extended[:-2]
        )

    state_ids = np.arange(n_states)
    for t in range(1, n_frames):
        previous = alpha[t - 1]
        stay = previous
        step = np.full(n_states, neg_inf, dtype=np.float32)
        if n_states > 1:
            step[1:] = previous[:-1]
        skip = np.full(n_states, neg_inf, dtype=np.float32)
        if n_states > 2:
            skip[2:] = np.where(skip_allowed[2:], previous[:-2], neg_inf)
        candidates = np.stack((stay, step, skip), axis=0)
        best_move = np.argmax(candidates, axis=0).astype(np.uint8)
        best_score = candidates[best_move, state_ids]
        alpha[t] = best_score + log_probs[t, extended]
        back[t] = best_move

    final_state = n_states - 1
    if n_states > 1 and alpha[-1, n_states - 2] > alpha[-1, final_state]:
        final_state = n_states - 2
    final_score = float(alpha[-1, final_state])
    if final_score <= -1e29:
        return None

    states = np.empty(n_frames, dtype=np.int64)
    state = final_state
    for t in range(n_frames - 1, -1, -1):
        states[t] = state
        if t > 0:
            state -= int(back[t, state])

    frames_per_label = np.zeros(n_labels, dtype=np.int64)
    for state in states:
        if state % 2 == 1:
            frames_per_label[state // 2] += 1
    return frames_per_label, final_score / float(n_frames)


# ===========================================================================
# Seconds -> DAC frames
# ===========================================================================
def durations_to_dac_frames(
    char_durations_sec: np.ndarray, total_frames: int
) -> np.ndarray:
    """Convert positive per-character seconds to integer DAC frames.

    The result sums to exactly ``total_frames``.  Largest-remainder rounding
    avoids the drift that would otherwise make the length regulator and the
    acoustic target disagree by one or more frames.
    """
    seconds = np.maximum(
        np.asarray(char_durations_sec, dtype=np.float64).reshape(-1), 0.0
    )
    total_frames = int(total_frames)
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    if total_frames == 0 or seconds.size == 0 or float(seconds.sum()) <= 0.0:
        return np.zeros(seconds.size, dtype=np.int32)

    scaled = seconds / float(seconds.sum()) * total_frames
    frames = np.floor(scaled).astype(np.int64)
    remainder = total_frames - int(frames.sum())
    eligible = seconds > 0.0
    if remainder > 0 and eligible.any():
        fractional = scaled - frames
        order = np.argsort(-fractional, kind="stable")
        for index in order:
            if remainder <= 0:
                break
            if eligible[index]:
                frames[index] += 1
                remainder -= 1
    # Defensive: floating-point floors can leave a remainder even after the
    # first pass (for example when all fractional parts are zero).  Spread it
    # over alignable characters without ever assigning frames to padding.
    if remainder > 0:
        eligible_indices = np.flatnonzero(eligible)
        cursor = 0
        while remainder > 0 and eligible_indices.size:
            frames[eligible_indices[cursor % eligible_indices.size]] += 1
            remainder -= 1
            cursor += 1
    return frames.astype(np.int32)


# ===========================================================================
# Hugging Face CTC adapter
# ===========================================================================
class CTCDurationExtractor:
    """Align transcripts with a Hugging Face ``AutoModelForCTC``.

    Parameters are injectable (``model``/``processor``) so tests and future
    NeMo adapters can exercise the seam without downloading weights.
    """

    def __init__(
        self,
        model_id: str = "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
        device: str = "cpu",
        revision: Optional[str] = None,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        blank_id: Optional[int] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        import torch

        self.model_id = str(model_id)
        self.device = torch.device(device)
        self.revision = revision
        self.dtype = dtype or torch.float32
        self._model = model
        self._processor = processor
        if self._model is None or self._processor is None:
            self._load()
        assert self._processor is not None and self._model is not None
        self.tokenizer = self._processor.tokenizer
        self.vocab = dict(self.tokenizer.get_vocab())
        self.blank_id = (
            int(blank_id)
            if blank_id is not None
            else int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        )
        feature_extractor = getattr(self._processor, "feature_extractor", None)
        self.sample_rate = int(
            sample_rate
            or getattr(feature_extractor, "sampling_rate", 0)
            or 16000
        )
        self.frame_rate = self._infer_frame_rate(self.sample_rate)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        kwargs = {"revision": self.revision} if self.revision else {}
        self._processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self._model = (
            AutoModelForCTC.from_pretrained(
                self.model_id, torch_dtype=self.dtype, **kwargs
            )
            .to(self.device)
            .eval()
        )

    def _infer_frame_rate(self, sample_rate: int) -> float:
        config = getattr(self._model, "config", None)
        stride = getattr(config, "conv_stride", None)
        if stride:
            values = [stride] if isinstance(stride, int) else list(stride)
            product = int(np.prod([int(value) for value in values]))
            if product > 0:
                return float(sample_rate) / float(product)
        # Wav2Vec2's default stack.  Recorded in output metadata; a wrong
        # frame rate only rescales durations, it cannot corrupt labels.
        return float(sample_rate) / 320.0

    @property
    def model(self) -> Any:
        return self._model

    def align(
        self, audio: np.ndarray, sample_rate: int, text: str
    ) -> Optional[DurationAlignment]:
        import torch
        import torch.nn.functional as F

        text = str(text)
        if not text.strip():
            return None
        audio = resample_audio(audio, sample_rate, self.sample_rate)
        if audio.size < int(0.05 * self.sample_rate):
            return None
        labels, positions, n_content = map_text_to_ctc_labels(text, self.vocab)
        if not labels or n_content == 0:
            return None

        inputs = self._processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_values = inputs["input_values"].to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            logits = self._model(input_values=input_values).logits
        log_probs = (
            F.log_softmax(logits[0].float(), dim=-1).detach().cpu().numpy()
        )
        aligned = ctc_forced_align(log_probs, labels, blank_id=self.blank_id)
        if aligned is None:
            return None
        frames_per_label, score = aligned
        char_seconds = np.zeros(len(text), dtype=np.float32)
        for position, frames in zip(positions, frames_per_label):
            char_seconds[position] = float(frames) / max(1e-9, self.frame_rate)
        # Coverage is over all characters, including spaces and punctuation,
        # so it never exceeds 1.0.  Unsupported characters simply contribute 0.
        coverage = float(len(positions) / max(1, len(text)))
        return DurationAlignment(
            char_durations_sec=char_seconds,
            score=float(score),
            coverage=coverage,
            model_id=self.model_id,
            n_ctc_frames=int(log_probs.shape[0]),
        )


class LazyCTCDurationExtractor:
    """Load a CTC model only when the first matching row arrives.

    A two-corpus run (Tamil + English) otherwise pays for both model loads
    before it has read a single row.  The wrapper is intentionally tiny: it
    delegates everything after construction.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        revision: Optional[str] = None,
    ) -> None:
        self.model_id = str(model_id)
        self.device = device
        self.revision = revision
        self._extractor: Optional[CTCDurationExtractor] = None
        self.load_count = 0

    def align(
        self, audio: np.ndarray, sample_rate: int, text: str
    ) -> Optional[DurationAlignment]:
        if self._extractor is None:
            self._extractor = CTCDurationExtractor(
                model_id=self.model_id,
                device=self.device,
                revision=self.revision,
            )
            self.load_count += 1
        return self._extractor.align(audio, sample_rate, text)


def align_text_to_seconds(
    text: str,
    labels: Sequence[int],
    positions: Sequence[int],
    frames_per_label: Sequence[int],
    frame_rate: float,
) -> np.ndarray:
    """Small pure helper used by tests and custom adapters."""
    out = np.zeros(len(text), dtype=np.float32)
    for position, frames in zip(positions, frames_per_label):
        out[int(position)] = float(frames) / max(1e-9, float(frame_rate))
    return out
