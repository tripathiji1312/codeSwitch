"""Content-addressed speech shards shared by the teacher and real-data pipelines.

A shard is one utterance: its text, its nine DAC codebooks, and (optionally)
per-character durations derived from a forced aligner.  The on-disk format is a
superset of the original ``teacher_cache`` format.  Old shards without the new
keys load unchanged; new shards carry alignment metadata so the draft trainer
can supervise its FrameDecoder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_SHARD_VERSION = 2


def stable_shard_index(key: str) -> int:
    """Return a stable, collision-resistant integer index for a shard key.

    The old teacher cache used corpus positions (small integers).  Real-speech
    caches are built incrementally from multiple sources, so position is not
    stable.  A 60-bit content hash gives deterministic indices across runs and
    leaves collisions effectively impossible at 10^5-10^6 shards.
    """
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


@dataclass
class SpeechShard:
    """One utterance: normalized text, 9xT DAC codes, optional durations.

    ``char_durations`` is aligned to ``norm_text`` (one entry per Python
    character), in **DAC frames**, and must sum to ``frames`` when present.
    The BOS/EOS tokens added by ``CharacterTokenizer`` are not represented
    here; the trainer inserts zero-duration sentinels around this array.
    """

    index: int
    raw_text: str
    norm_text: str
    n_chars: int
    frames: int
    codes: np.ndarray  # (9, T) integer codebooks 0-8
    audio_sec: float
    target_source: str
    gen_ms: float
    # Optional alignment metadata.  ``None`` means "this shard has no usable
    # per-character durations"; it is not the same as an all-zero alignment.
    char_durations: Optional[np.ndarray] = None
    duration_source: str = "none"
    alignment_score: float = 0.0
    alignment_coverage: float = 0.0
    source: str = ""
    speaker: str = ""
    language: str = ""
    sample_rate: int = 0

    def __post_init__(self) -> None:
        self.codes = np.asarray(self.codes)
        if self.codes.ndim != 2 or self.codes.shape[0] != 9:
            raise ValueError(f"codes must be (9, T), got {self.codes.shape}")
        if int(self.frames) != int(self.codes.shape[1]):
            raise ValueError(
                f"frames={self.frames} but codes has {self.codes.shape[1]} timesteps"
            )
        if self.char_durations is not None:
            durations = np.asarray(self.char_durations).reshape(-1)
            if durations.size and durations.size != int(self.n_chars):
                raise ValueError(
                    f"char_durations has {durations.size} entries for "
                    f"n_chars={self.n_chars}"
                )
            if durations.size and int(durations.min()) < 0:
                raise ValueError("char_durations must be non-negative")
            if durations.size and not np.allclose(durations, np.rint(durations)):
                raise ValueError("char_durations must contain integer frame counts")
            if durations.size and int(durations.sum()) != int(self.frames):
                raise ValueError(
                    f"char_durations sum {int(durations.sum())} != frames {self.frames}"
                )
            self.char_durations = durations.astype(np.int32, copy=False)

    @property
    def has_durations(self) -> bool:
        return self.char_durations is not None and self.char_durations.size > 0

    @property
    def duration_coverage(self) -> float:
        """Fraction of characters that received a positive duration."""
        if not self.has_durations:
            return 0.0
        return float((self.char_durations > 0).mean())

    def save(self, path: Path) -> None:
        """Write the shard atomically (``np.savez_compressed`` + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        durations = (
            self.char_durations
            if self.char_durations is not None
            else np.zeros(0, dtype=np.int32)
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                shard_version=np.asarray(_SHARD_VERSION, dtype=np.int32),
                index=np.asarray(self.index, dtype=np.int64),
                raw_text=np.asarray(self.raw_text),
                norm_text=np.asarray(self.norm_text),
                n_chars=np.asarray(self.n_chars, dtype=np.int32),
                frames=np.asarray(self.frames, dtype=np.int32),
                codes=self.codes.astype(np.int16, copy=False),
                audio_sec=np.asarray(self.audio_sec, dtype=np.float32),
                target_source=np.asarray(self.target_source),
                gen_ms=np.asarray(self.gen_ms, dtype=np.float32),
                char_durations=durations.astype(np.int32, copy=False),
                duration_source=np.asarray(self.duration_source),
                alignment_score=np.asarray(self.alignment_score, dtype=np.float32),
                alignment_coverage=np.asarray(self.alignment_coverage, dtype=np.float32),
                source=np.asarray(self.source),
                speaker=np.asarray(self.speaker),
                language=np.asarray(self.language),
                sample_rate=np.asarray(self.sample_rate, dtype=np.int32),
            )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SpeechShard":
        data = np.load(path, allow_pickle=False)
        durations: Optional[np.ndarray] = None
        if "char_durations" in data.files:
            raw = data["char_durations"].astype(np.int64, copy=False).reshape(-1)
            if raw.size:
                durations = raw.astype(np.int32, copy=False)
        return cls(
            index=int(data["index"]),
            raw_text=str(data["raw_text"]),
            norm_text=str(data["norm_text"]),
            n_chars=int(data["n_chars"]),
            frames=int(data["frames"]),
            codes=data["codes"].astype(np.int64, copy=False),
            audio_sec=float(data["audio_sec"]),
            target_source=str(data["target_source"]),
            gen_ms=float(data["gen_ms"]),
            char_durations=durations,
            duration_source=(
                str(data["duration_source"]) if "duration_source" in data.files else "none"
            ),
            alignment_score=(
                float(data["alignment_score"]) if "alignment_score" in data.files else 0.0
            ),
            alignment_coverage=(
                float(data["alignment_coverage"])
                if "alignment_coverage" in data.files
                else 0.0
            ),
            source=str(data["source"]) if "source" in data.files else "",
            speaker=str(data["speaker"]) if "speaker" in data.files else "",
            language=str(data["language"]) if "language" in data.files else "",
            sample_rate=(
                int(data["sample_rate"]) if "sample_rate" in data.files else 0
            ),
        )
