"""Real-speech -> DAC codebook shards, with optional CTC duration metadata.

This is Move 1 of the data-scale plan: instead of waiting for a large
teacher-generated corpus, encode transcribed Rasa / IndicVoices-R / English
speech directly with the DAC encoder.  The resulting shards are the same
format the draft trainer already consumes, plus per-character durations from
Move 2.

The module is split at three seams:

``iter_*`` adapters
    Turn Hugging Face streaming rows, local parquet files, or a JSONL/CSV
    manifest into :class:`SpeechRow` objects.
``DACEncoder``
    The only component that touches the codec.
``RealSpeechPipeline``
    Filters, normalizes, encodes, aligns, deduplicates, and writes shards.

Tests inject fake encoders and duration extractors, so none of this requires a
GPU, a gated dataset, or a model download.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from codeswitch_tts.data.audio import decode_audio, resample_audio
from codeswitch_tts.data.duration import (
    DurationExtractor,
    durations_to_dac_frames,
)
from codeswitch_tts.data.shards import SpeechShard, stable_shard_index

_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_AUDIO_COLUMNS = (
    "audio",
    "audio_path",
    "audio_filepath",
    "audio_file",
    "wav",
    "path",
    "file",
)
_TEXT_COLUMNS = (
    "text",
    "transcript",
    "transcription",
    "sentence",
    "normalized_text",
    "label",
)
_SPEAKER_COLUMNS = ("speaker", "speaker_id", "speaker_name", "spk_id")
_LANGUAGE_COLUMNS = ("language", "lang", "locale")
_ID_COLUMNS = ("id", "audio_id", "utt_id", "utterance_id", "clip_id")
_STYLE_COLUMNS = ("style", "emotion", "register")


def detect_language(text: str) -> str:
    """Cheap script-based language tag for model routing.

    Returns ``ta``, ``en``, ``code-mixed``, or ``unknown``.  This is not a
    language detector; it only decides which CTC vocabulary is plausible.
    """
    text = str(text)
    tamil = len(_TAMIL_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if tamil and latin:
        return "code-mixed"
    if tamil:
        return "ta"
    if latin:
        return "en"
    return "unknown"


def normalize_language_tag(value: str, text: str = "") -> str:
    """Turn dataset labels such as ``Tamil`` / ``English`` into ``ta`` / ``en``.

    Datasets are inconsistent: FLEURS stores ``Tamil``, Rasa may store ``ta``
    or omit the field entirely.  Routing CTC by the raw display name silently
    produces duration-less shards, so normalize at the adapter boundary.
    """
    raw = str(value or "").strip().lower().replace("_", "-")
    if not raw:
        return detect_language(text)
    if "tamil" in raw and "english" in raw:
        return "code-mixed"
    if raw in {"code-mixed", "code mixed", "mixed", "tanglish", "codemixed"}:
        return "code-mixed"
    if raw in {"ta", "tam", "tamil", "ta-in", "ta-lk", "ta-sg"} or "tamil" in raw:
        return "ta"
    if raw in {"en", "eng", "english", "en-in", "en-us", "en-gb"} or "english" in raw:
        return "en"
    return raw


@dataclass
class SpeechRow:
    """One decoded utterance, before DAC encoding."""

    audio: np.ndarray
    sample_rate: int
    text: str
    language: str = ""
    source: str = ""
    speaker: str = ""
    row_id: str = ""
    style: str = ""


@dataclass
class PipelineConfig:
    """Filtering and output policy for :class:`RealSpeechPipeline`."""

    out_dir: Path
    min_duration_sec: float = 0.6
    max_duration_sec: float = 30.0
    min_chars: int = 4
    max_frames: int = 2048
    limit: int = 0
    max_hours: float = 0.0
    overwrite: bool = False
    require_durations: bool = False
    source_name: str = "real_speech"
    dac_model_id: str = "ylacombe/dac_44khz"
    duration_models: Dict[str, str] = field(default_factory=dict)
    seed: int = 42


class DACEncoder:
    """Hugging Face DAC encoder adapter.

    ``ylacombe/dac_44khz`` was verified locally: a 1.0 s waveform produces
    ``audio_codes`` of shape ``(1, 9, 86)``.  The config's rounded
    ``frame_rate=87`` is deliberately ignored; ``sampling_rate / hop_length``
    is the real rate (86.1328 Hz).
    """

    def __init__(
        self,
        model_id: str = "ylacombe/dac_44khz",
        device: str = "cpu",
        model: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        frame_rate: Optional[float] = None,
    ) -> None:
        import torch

        self.model_id = str(model_id)
        self.device = torch.device(device)
        self._model = model
        if self._model is None:
            from transformers import DacModel

            self._model = (
                DacModel.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )
        config = getattr(self._model, "config", None)
        self.sample_rate = int(
            sample_rate
            or getattr(config, "sampling_rate", 0)
            or 44100
        )
        hop_length = int(getattr(config, "hop_length", 0) or 512)
        self.frame_rate = float(
            frame_rate or (float(self.sample_rate) / float(hop_length))
        )

    @property
    def model(self) -> Any:
        return self._model

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return ``(9, T)`` int64 DAC codes for mono float audio."""
        import torch

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("cannot encode empty audio")
        audio = resample_audio(audio, sample_rate, self.sample_rate)
        tensor = torch.from_numpy(audio).float().reshape(1, 1, -1)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            output = self._model.encode(input_values=tensor)
        codes = getattr(output, "audio_codes", output)
        if codes is None:
            raise RuntimeError(
                "DAC encoder returned no audio_codes; expected DacModel.encode"
            )
        codes = codes[0].detach().cpu().numpy().astype(np.int64, copy=False)
        if codes.ndim != 2 or codes.shape[0] != 9:
            raise RuntimeError(f"expected (9, T) DAC codes, got {codes.shape}")
        return codes


def _first_key(row: Mapping[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            return key
    return None


def row_to_speech(
    row: Mapping[str, Any],
    index: int,
    base_dir: Optional[Path] = None,
    source: str = "",
    default_language: str = "",
    text_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    speaker_column: Optional[str] = None,
    style_filter: Sequence[str] = (),
) -> Optional[SpeechRow]:
    """Convert a dataset/parquet row into a decoded :class:`SpeechRow`."""
    audio_key = audio_column or _first_key(row, _AUDIO_COLUMNS)
    text_key = text_column or _first_key(row, _TEXT_COLUMNS)
    if audio_key is None:
        raise KeyError(f"no audio column in row keys {sorted(row)}")
    if text_key is None:
        raise KeyError(f"no text column in row keys {sorted(row)}")
    text = str(row.get(text_key) or "").strip()
    if not text:
        return None
    style = ""
    style_key = _first_key(row, _STYLE_COLUMNS)
    if style_key is not None:
        style = str(row.get(style_key) or "")
    if style_filter and style not in style_filter:
        return None
    language_key = _first_key(row, _LANGUAGE_COLUMNS)
    language = (
        str(row.get(language_key) or "")
        if language_key is not None
        else default_language
    )
    language = normalize_language_tag(language, text)
    speaker_key = speaker_column or _first_key(row, _SPEAKER_COLUMNS)
    speaker = (
        str(row.get(speaker_key) or "")
        if speaker_key is not None
        else ""
    )
    id_key = _first_key(row, _ID_COLUMNS)
    row_id = str(row.get(id_key)) if id_key is not None else str(index)
    audio, sample_rate = decode_audio(row.get(audio_key), base_dir=base_dir)
    return SpeechRow(
        audio=audio,
        sample_rate=int(sample_rate),
        text=text,
        language=language,
        source=source,
        speaker=speaker,
        row_id=row_id,
        style=style,
    )


# ===========================================================================
# Input adapters
# ===========================================================================
def iter_manifest(
    path: Path,
    default_language: str = "",
    text_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    speaker_column: Optional[str] = None,
    style_filter: Sequence[str] = (),
) -> Iterator[SpeechRow]:
    """Yield rows from a JSONL or CSV manifest.

    Required fields: an audio path/column and a text column.  Paths are
    resolved relative to the manifest directory.
    """
    path = Path(path)
    base_dir = path.parent
    source = path.stem
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                speech = row_to_speech(
                    row,
                    index=index,
                    base_dir=base_dir,
                    source=source,
                    default_language=default_language,
                    text_column=text_column,
                    audio_column=audio_column,
                    speaker_column=speaker_column,
                    style_filter=style_filter,
                )
                if speech is not None:
                    yield speech
        return

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            speech = row_to_speech(
                row,
                index=index,
                base_dir=base_dir,
                source=source,
                default_language=default_language,
                text_column=text_column,
                audio_column=audio_column,
                speaker_column=speaker_column,
                style_filter=style_filter,
            )
            if speech is not None:
                yield speech


def _iter_parquet_file(
    parquet_path: Path,
    root: Path,
    batch_size: int,
    default_language: str,
    text_column: Optional[str],
    audio_column: Optional[str],
    speaker_column: Optional[str],
    style_filter: Sequence[str],
) -> Iterator[SpeechRow]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    relative = parquet_path.relative_to(root) if root in parquet_path.parents else parquet_path.name
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            speech = row_to_speech(
                row,
                index=row_index,
                base_dir=parquet_path.parent,
                source=f"parquet:{root.name}",
                default_language=default_language,
                text_column=text_column,
                audio_column=audio_column,
                speaker_column=speaker_column,
                style_filter=style_filter,
            )
            row_index += 1
            if speech is not None:
                # Include the parquet file and row so row ids are unique across
                # shards even when the dataset has no explicit id column.
                speech.row_id = f"{relative}:{row_index - 1}:{speech.row_id}"
                yield speech


def iter_parquet_dir(
    root: Path,
    default_language: str = "",
    batch_size: int = 64,
    text_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    speaker_column: Optional[str] = None,
    style_filter: Sequence[str] = (),
) -> Iterator[SpeechRow]:
    """Yield rows from every ``*.parquet`` under ``root`` (recursive)."""
    root = Path(root)
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    for parquet_path in files:
        yield from _iter_parquet_file(
            parquet_path,
            root=root,
            batch_size=batch_size,
            default_language=default_language,
            text_column=text_column,
            audio_column=audio_column,
            speaker_column=speaker_column,
            style_filter=style_filter,
        )


def iter_hf_dataset(
    dataset: str,
    config: Optional[str] = None,
    split: str = "train",
    default_language: str = "",
    text_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    speaker_column: Optional[str] = None,
    style_filter: Sequence[str] = (),
    token: Optional[str] = None,
    streaming: bool = True,
) -> Iterator[SpeechRow]:
    """Stream a Hugging Face dataset.

    Gated corpora (Rasa, IndicVoices-R) need ``HF_TOKEN`` in the environment;
    the caller must have accepted the dataset terms on the Hub.  Streaming
    avoids materializing 60+ parquet files before the first shard is written.
    """
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        "token": token,
    }
    if config:
        dataset_obj = load_dataset(dataset, config, **kwargs)
    else:
        dataset_obj = load_dataset(dataset, **kwargs)

    # ``datasets>=4`` decodes an Audio column to a torchcodec.AudioDecoder.
    # That pins the data path to a torchcodec wheel matching Kaggle's Torch ABI.
    # Ask for raw bytes/path instead, then decode with soundfile/torchaudio.
    features = getattr(dataset_obj, "features", None)
    audio_key = audio_column
    if audio_key is None and features is not None and hasattr(features, "__contains__"):
        audio_key = _first_key(features, _AUDIO_COLUMNS)
    if audio_key:
        try:
            from datasets import Audio

            dataset_obj = dataset_obj.cast_column(audio_key, Audio(decode=False))
        except Exception:
            # Some datasets expose a non-standard audio feature.  The caller's
            # decode_audio adapter still handles dict/bytes/decoder cells.
            pass

    source = f"hf:{dataset}" + (f"/{config}" if config else "") + f"/{split}"
    for index, row in enumerate(dataset_obj):
        speech = row_to_speech(
            row,
            index=index,
            base_dir=None,
            source=source,
            default_language=default_language,
            text_column=text_column,
            audio_column=audio_column,
            speaker_column=speaker_column,
            style_filter=style_filter,
        )
        if speech is not None:
            yield speech


# ===========================================================================
# Pipeline
# ===========================================================================
def _load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            shard = record.get("shard")
            if shard:
                # The pipeline keys records by the shard filename.  Older
                # manifests may contain an absolute path; normalize it so a
                # resumed run does not count the same shard twice.
                records[Path(str(shard)).name] = record
    return records


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_manifest_atomic(path: Path, records: Mapping[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for shard in sorted(records):
            handle.write(json.dumps(records[shard], ensure_ascii=False) + "\n")
    tmp.replace(path)


class RealSpeechPipeline:
    """Encode decoded speech rows into :class:`SpeechShard` files.

    The pipeline is deliberately deterministic: the shard key is a hash of
    source, row id, normalized text, and audio bytes.  Re-running the same
    input is a no-op unless ``overwrite=True``.
    """

    def __init__(
        self,
        config: PipelineConfig,
        dac_encoder: Any,
        duration_extractors: Optional[Mapping[str, DurationExtractor]] = None,
        normalizer: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dac_encoder = dac_encoder
        self.duration_extractors = dict(duration_extractors or {})
        self.normalizer = normalizer
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self.summary_path = self.out_dir / "meta.json"
        self._records = _load_manifest(self.manifest_path)

    def _normalize(self, text: str) -> str:
        if self.normalizer is None:
            return str(text).strip()
        try:
            result = self.normalizer.normalize(text)
        except Exception:
            return str(text).strip()
        return str(getattr(result, "normalized_text", result)).strip()

    def _duration_extractor(self, language: str) -> Optional[DurationExtractor]:
        language = normalize_language_tag(language)
        if language in self.duration_extractors:
            return self.duration_extractors[language]
        return self.duration_extractors.get("default")

    def _shard_key(self, row: SpeechRow, norm_text: str) -> str:
        audio_hash = hashlib.sha1(
            np.ascontiguousarray(row.audio, dtype=np.float32).tobytes()
        ).hexdigest()
        return "|".join(
            (
                row.source or self.config.source_name,
                row.row_id or "",
                norm_text,
                audio_hash,
            )
        )

    def _shard_record(self, shard: SpeechShard, shard_path: Path, key: str) -> Dict[str, Any]:
        return {
            "shard": str(shard_path),
            "key": key,
            "index": int(shard.index),
            "text_preview": shard.norm_text[:96],
            "raw_chars": len(shard.raw_text),
            "norm_chars": shard.n_chars,
            "frames": int(shard.frames),
            "audio_sec": float(shard.audio_sec),
            "duration_source": shard.duration_source,
            "duration_coverage": float(shard.duration_coverage),
            "alignment_score": float(shard.alignment_score),
            "source": shard.source,
            "speaker": shard.speaker,
            "language": shard.language,
            "sample_rate": int(shard.sample_rate),
        }

    def _summary(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        records = list(self._records.values())
        duration_records = [
            record for record in records if record.get("duration_source", "none") != "none"
        ]
        sources: Dict[str, int] = {}
        languages: Dict[str, int] = {}
        for record in records:
            sources[str(record.get("source", ""))] = sources.get(str(record.get("source", "")), 0) + 1
            languages[str(record.get("language", ""))] = languages.get(str(record.get("language", "")), 0) + 1
        return {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "out_dir": str(self.out_dir),
            "dac_model": getattr(self.dac_encoder, "model_id", "injected"),
            "dac_frame_rate": float(getattr(self.dac_encoder, "frame_rate", 0.0)),
            "duration_models": dict(self.config.duration_models),
            "filters": {
                "min_duration_sec": self.config.min_duration_sec,
                "max_duration_sec": self.config.max_duration_sec,
                "min_chars": self.config.min_chars,
                "max_frames": self.config.max_frames,
                "limit": self.config.limit,
                "max_hours": self.config.max_hours,
            },
            "stats": stats,
            "shards": len(records),
            "hours": float(sum(float(r.get("audio_sec", 0.0)) for r in records) / 3600.0),
            "duration_shards": len(duration_records),
            "mean_duration_coverage": (
                float(
                    np.mean([float(r.get("duration_coverage", 0.0)) for r in duration_records])
                )
                if duration_records
                else 0.0
            ),
            "sources": sources,
            "languages": languages,
        }

    def run(self, rows: Iterable[SpeechRow]) -> Dict[str, Any]:
        cfg = self.config
        stats: Dict[str, Any] = {
            "rows_seen": 0,
            "written": 0,
            "skipped_existing": 0,
            "filtered_short": 0,
            "filtered_long": 0,
            "filtered_chars": 0,
            "filtered_frames": 0,
            "duration_ok": 0,
            "duration_missing": 0,
            "errors": 0,
            "audio_seconds": 0.0,
        }
        indices: Dict[int, str] = {
            int(record["index"]): str(record["shard"])
            for record in self._records.values()
            if "index" in record
        }
        error_messages: List[str] = []
        started = time.perf_counter()

        for row in rows:
            stats["rows_seen"] += 1
            audio_seconds = float(row.audio.size) / max(1, int(row.sample_rate))
            if audio_seconds < cfg.min_duration_sec:
                stats["filtered_short"] += 1
                continue
            if audio_seconds > cfg.max_duration_sec:
                stats["filtered_long"] += 1
                continue
            norm_text = self._normalize(row.text)
            if len(norm_text) < cfg.min_chars:
                stats["filtered_chars"] += 1
                continue
            language = normalize_language_tag(row.language or "", norm_text)
            key = self._shard_key(row, norm_text)
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
            shard_path = self.out_dir / f"shard_{digest[:16]}.npz"
            if shard_path.is_file() and not cfg.overwrite:
                stats["skipped_existing"] += 1
                if shard_path.name not in self._records:
                    try:
                        existing = SpeechShard.load(shard_path)
                    except Exception:
                        existing = None
                    if existing is not None:
                        self._records[shard_path.name] = self._shard_record(
                            existing, shard_path, key
                        )
                continue

            try:
                row_started = time.perf_counter()
                codes = np.asarray(self.dac_encoder.encode(row.audio, row.sample_rate))
                frames = int(codes.shape[1]) if codes.ndim == 2 else 0
                if frames <= 0:
                    raise RuntimeError(f"encoder returned empty codes {codes.shape}")
                if frames > cfg.max_frames:
                    stats["filtered_frames"] += 1
                    continue

                char_durations: Optional[np.ndarray] = None
                duration_source = "none"
                alignment_score = 0.0
                alignment_coverage = 0.0
                extractor = self._duration_extractor(language)
                if extractor is not None:
                    alignment = extractor.align(row.audio, row.sample_rate, norm_text)
                    if alignment is not None and float(
                        np.asarray(alignment.char_durations_sec).sum()
                    ) > 0.0:
                        char_durations = durations_to_dac_frames(
                            alignment.char_durations_sec, frames
                        )
                        duration_source = f"ctc:{alignment.model_id}"
                        alignment_score = float(alignment.score)
                        alignment_coverage = float(alignment.coverage)
                        stats["duration_ok"] += 1
                    else:
                        stats["duration_missing"] += 1
                elif cfg.require_durations:
                    stats["duration_missing"] += 1
                if cfg.require_durations and char_durations is None:
                    stats["errors"] += 1
                    if len(error_messages) < 20:
                        error_messages.append(
                            f"missing duration for {key[:80]} (language={language})"
                        )
                    continue

                index = stable_shard_index(key)
                if index in indices and indices[index] != str(shard_path):
                    raise RuntimeError(
                        f"shard index collision {index}: {indices[index]} vs {shard_path}"
                    )
                shard = SpeechShard(
                    index=index,
                    raw_text=row.text,
                    norm_text=norm_text,
                    n_chars=len(norm_text),
                    frames=frames,
                    codes=codes.astype(np.int64, copy=False),
                    audio_sec=audio_seconds,
                    target_source="real_dac",
                    gen_ms=(time.perf_counter() - row_started) * 1000.0,
                    char_durations=char_durations,
                    duration_source=duration_source,
                    alignment_score=alignment_score,
                    alignment_coverage=alignment_coverage,
                    source=row.source or cfg.source_name,
                    speaker=row.speaker,
                    language=language,
                    sample_rate=int(row.sample_rate),
                )
                if char_durations is not None and int(char_durations.sum()) != frames:
                    raise RuntimeError(
                        f"duration sum {int(char_durations.sum())} != frames {frames}"
                    )
                shard.save(shard_path)
                indices[index] = str(shard_path)
                self._records[shard_path.name] = self._shard_record(
                    shard, shard_path, key
                )
                stats["written"] += 1
                stats["audio_seconds"] += audio_seconds
                if cfg.limit and stats["written"] >= cfg.limit:
                    break
                if cfg.max_hours and stats["audio_seconds"] >= cfg.max_hours * 3600.0:
                    break
            except Exception as exc:  # keep a long batch alive; report at the end
                stats["errors"] += 1
                if len(error_messages) < 20:
                    error_messages.append(f"{key[:80]}: {type(exc).__name__}: {exc}")

        stats["elapsed_sec"] = time.perf_counter() - started
        stats["errors_detail"] = error_messages
        _write_manifest_atomic(self.manifest_path, self._records)
        _write_json_atomic(self.summary_path, self._summary(stats))
        if cfg.require_durations and stats["duration_missing"]:
            raise RuntimeError(
                f"{stats['duration_missing']} rows had no usable CTC alignment; "
                "inspect meta.json or rerun with --durations none"
            )
        return stats
