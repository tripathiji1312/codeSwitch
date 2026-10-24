"""Tests for the real-speech data pipeline and CTC duration alignment.

These tests are CPU-only and do not download datasets or model weights.  The
DAC encoder and CTC extractor are injected as fakes; the Viterbi aligner is
exercised with synthetic log-probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from codeswitch_tts.data.audio import decode_audio
from codeswitch_tts.data.shards import SpeechShard, stable_shard_index


# ===========================================================================
# Shard contract
# ===========================================================================
def _shard(**overrides):
    text = overrides.pop("norm_text", "நாளை வருகிறேன்.")
    frames = overrides.pop("frames", 8)
    codes = overrides.pop(
        "codes", np.stack([np.arange(frames) % 1024 for _ in range(9)])
    )
    durations = np.zeros(len(text), dtype=np.int32)
    for offset in range(frames):
        durations[offset % len(text)] += 1
    defaults = dict(
        index=7,
        raw_text=text,
        norm_text=text,
        n_chars=len(text),
        frames=frames,
        codes=codes,
        audio_sec=0.5,
        target_source="real_dac",
        gen_ms=12.0,
        char_durations=durations,
        duration_source="ctc:fake",
        alignment_score=-0.2,
        alignment_coverage=1.0,
        source="unit-test",
        speaker="speaker-1",
        language="ta",
        sample_rate=16000,
    )
    defaults.update(overrides)
    return SpeechShard(**defaults)


def test_speech_shard_roundtrips_duration_metadata(tmp_path: Path):
    shard = _shard()
    path = tmp_path / "shard.npz"
    shard.save(path)
    loaded = SpeechShard.load(path)
    assert loaded.index == shard.index
    assert loaded.norm_text == shard.norm_text
    assert loaded.frames == shard.frames
    assert loaded.has_durations
    assert np.array_equal(loaded.char_durations, shard.char_durations)
    assert loaded.duration_source == "ctc:fake"
    assert loaded.alignment_coverage == pytest.approx(1.0)
    assert loaded.source == "unit-test"
    assert loaded.language == "ta"


def test_speech_shard_loads_legacy_cache_without_duration_keys(tmp_path: Path):
    """The v1 teacher cache wrote npz files without duration metadata."""
    text = "வணக்கம் சார்."
    frames = 5
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        index=np.asarray(0),
        raw_text=np.asarray(text),
        norm_text=np.asarray(text),
        n_chars=np.asarray(len(text)),
        frames=np.asarray(frames),
        codes=np.zeros((9, frames), dtype=np.int16),
        audio_sec=np.asarray(0.25),
        target_source=np.asarray("lm"),
        gen_ms=np.asarray(3.0),
    )
    loaded = SpeechShard.load(path)
    assert not loaded.has_durations
    assert loaded.char_durations is None
    assert loaded.duration_source == "none"


def test_speech_shard_rejects_misaligned_durations():
    with pytest.raises(ValueError, match="char_durations"):
        _shard(char_durations=np.ones(2, dtype=np.int32))


def test_speech_shard_rejects_duration_sum_mismatch():
    text = "abc"
    with pytest.raises(ValueError, match="char_durations sum"):
        _shard(
            raw_text=text,
            norm_text=text,
            n_chars=3,
            char_durations=np.ones(3, dtype=np.int32),
        )


def test_stable_shard_index_is_deterministic_and_large():
    a = stable_shard_index("rasa:Tamil:row-1")
    b = stable_shard_index("rasa:Tamil:row-1")
    c = stable_shard_index("rasa:Tamil:row-2")
    assert a == b
    assert a != c
    assert a >= 2**40


# ===========================================================================
# CTC duration alignment (pure NumPy; no model download)
# ===========================================================================
from codeswitch_tts.data.duration import (  # noqa: E402
    align_text_to_seconds,
    ctc_forced_align,
    durations_to_dac_frames,
    map_text_to_ctc_labels,
)


def _peaked_log_probs(
    frames: list[int], blank_id: int = 0, vocab_size: int = 8
) -> np.ndarray:
    """Build log-probs whose argmax path visits ``frames`` in order."""
    logits = np.full((len(frames), vocab_size), -10.0, dtype=np.float32)
    for t, label in enumerate(frames):
        logits[t, blank_id] = -10.0
        logits[t, label] = 10.0
    logits -= logits.max(axis=1, keepdims=True)
    logits -= np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return logits.astype(np.float32)


def test_ctc_forced_align_recovers_clean_label_timing():
    # Three labels, one frame each; the Viterbi path skips the blanks.
    log_probs = _peaked_log_probs([1, 2, 3])
    result = ctc_forced_align(log_probs, [1, 2, 3], blank_id=0)
    assert result is not None
    durations, score = result
    assert durations.tolist() == [1, 1, 1]
    assert score > -0.1


def test_ctc_forced_align_inserts_blank_between_repeated_labels():
    log_probs = _peaked_log_probs([1, 0, 1])
    result = ctc_forced_align(log_probs, [1, 1], blank_id=0)
    assert result is not None
    durations, _ = result
    assert durations.tolist() == [1, 1]


def test_ctc_forced_align_rejects_more_labels_than_frames():
    log_probs = _peaked_log_probs([1, 2])
    assert ctc_forced_align(log_probs, [1, 2, 3], blank_id=0) is None


def test_map_text_to_ctc_labels_ignores_unknown_characters_and_maps_space():
    vocab = {"க": 1, "ா": 2, "|": 3}
    labels, positions, n_content = map_text_to_ctc_labels("க ாA", vocab)
    assert labels == [1, 3, 2]
    assert positions == [0, 1, 2]
    assert n_content == 3  # A is content but not representable


def test_durations_to_dac_frames_sums_exactly_and_preserves_zero_chars():
    frames = durations_to_dac_frames(np.array([1.0, 0.0, 0.0, 3.0]), 9)
    assert frames.dtype == np.int32
    assert int(frames.sum()) == 9
    assert frames[1] == 0 and frames[2] == 0
    assert frames[0] > 0 and frames[3] > 0


def test_durations_to_dac_frames_largest_remainder_is_stable():
    frames = durations_to_dac_frames(np.ones(4), 10)
    assert frames.tolist() == [3, 3, 2, 2]
    assert int(frames.sum()) == 10


def test_durations_to_dac_frames_handles_all_zero_alignment():
    frames = durations_to_dac_frames(np.zeros(5), 12)
    assert frames.tolist() == [0, 0, 0, 0, 0]


def test_align_text_to_seconds_maps_positions():
    seconds = align_text_to_seconds(
        "ab", [1, 2], [0, 1], [50, 100], frame_rate=50.0
    )
    assert seconds.tolist() == [1.0, 2.0]


# ===========================================================================
# Real-speech pipeline (injected encoder/aligner; no GPU or downloads)
# ===========================================================================
from types import SimpleNamespace  # noqa: E402

from codeswitch_tts.data.duration import DurationAlignment  # noqa: E402
from codeswitch_tts.data.real_speech import (  # noqa: E402
    PipelineConfig,
    RealSpeechPipeline,
    SpeechRow,
    detect_language,
    iter_manifest,
    iter_parquet_dir,
    normalize_language_tag,
)


class _FakeDAC:
    model_id = "fake-dac"
    frame_rate = 20.0

    def encode(self, audio, sample_rate):
        assert sample_rate == 16000
        assert len(audio) > 0
        return np.zeros((9, 20), dtype=np.int64)


class _FakeDurations:
    model_id = "fake-ctc"

    def align(self, audio, sample_rate, text):
        return DurationAlignment(
            char_durations_sec=np.full(len(text), 1.0 / len(text), dtype=np.float32),
            score=-0.01,
            coverage=1.0,
            model_id=self.model_id,
            n_ctc_frames=50,
        )


class _FakeNormalizer:
    def normalize(self, text):
        return SimpleNamespace(normalized_text=text)


def _row(text="நாளை வருகிறேன்", seconds=0.5):
    return SpeechRow(
        audio=np.zeros(int(16000 * seconds), dtype=np.float32),
        sample_rate=16000,
        text=text,
        language="ta",
        source="unit",
        speaker="spk",
        row_id="row-1",
    )


def test_detect_language_script_routing():
    assert detect_language("நாளை வருகிறேன்") == "ta"
    assert detect_language("The bus arrives") == "en"
    assert detect_language("Unga cab ippo vandhuchu") == "en"
    assert detect_language("உங்க cab இப்போ") == "code-mixed"
    assert normalize_language_tag("Tamil") == "ta"
    assert normalize_language_tag("English") == "en"
    assert normalize_language_tag("Tamil, English") == "code-mixed"
    assert normalize_language_tag("", "நாளை வருகிறேன்") == "ta"


def test_real_speech_pipeline_writes_duration_shard_and_resumes(tmp_path: Path):
    cfg = PipelineConfig(
        out_dir=tmp_path,
        min_duration_sec=0.1,
        min_chars=2,
        limit=10,
        require_durations=True,
    )
    pipeline = RealSpeechPipeline(
        cfg,
        dac_encoder=_FakeDAC(),
        duration_extractors={"ta": _FakeDurations()},
        normalizer=_FakeNormalizer(),
    )
    stats = pipeline.run([_row(), _row(text="மீண்டும்", seconds=0.05)])
    assert stats["written"] == 1
    assert stats["filtered_short"] == 1
    assert stats["duration_ok"] == 1

    shard_files = sorted(tmp_path.glob("shard_*.npz"))
    assert len(shard_files) == 1
    shard = SpeechShard.load(shard_files[0])
    assert shard.target_source == "real_dac"
    assert shard.has_durations
    assert int(shard.char_durations.sum()) == shard.frames == 20
    assert shard.duration_source == "ctc:fake-ctc"
    assert shard.language == "ta"

    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8")
    assert "\"duration_source\": \"ctc:fake-ctc\"" in manifest
    summary = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert summary["shards"] == 1
    assert summary["duration_shards"] == 1
    assert summary["hours"] == pytest.approx(0.5 / 3600.0, rel=1e-6)

    # A re-run with the same key must not duplicate the shard.
    again = RealSpeechPipeline(
        cfg,
        dac_encoder=_FakeDAC(),
        duration_extractors={"ta": _FakeDurations()},
        normalizer=_FakeNormalizer(),
    ).run([_row()])
    assert again["written"] == 0
    assert again["skipped_existing"] == 1
    summary_again = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert summary_again["shards"] == 1
    assert summary_again["duration_shards"] == 1


def test_real_speech_pipeline_requires_durations_when_asked(tmp_path: Path):
    cfg = PipelineConfig(
        out_dir=tmp_path,
        min_duration_sec=0.1,
        min_chars=2,
        require_durations=True,
    )
    pipeline = RealSpeechPipeline(cfg, dac_encoder=_FakeDAC(), normalizer=_FakeNormalizer())
    with pytest.raises(RuntimeError, match="no usable CTC alignment"):
        pipeline.run([_row()])


def test_decode_audio_accepts_torchcodec_style_decoder():
    """datasets >=4 returns an AudioDecoder, not a dict; decode it explicitly."""
    from types import SimpleNamespace

    class _Array:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.arange(8, dtype=np.float32).reshape(2, 4)

    class _Samples:
        data = _Array()
        sample_rate = 16000

    class _Decoder:
        metadata = SimpleNamespace(sample_rate=16000)

        def get_all_samples(self):
            return _Samples()

    audio, sample_rate = decode_audio(_Decoder())
    assert sample_rate == 16000
    assert audio.shape == (4,)
    assert audio.dtype == np.float32
    # torchcodec returns (channels, frames); mono is the channel mean.
    assert np.allclose(audio, np.arange(8, dtype=np.float32).reshape(2, 4).mean(axis=0))


def test_decode_audio_rejects_torchcodec_cell_without_sample_rate():
    from types import SimpleNamespace

    class _Decoder:
        metadata = None

        def get_all_samples(self):
            return SimpleNamespace(data=np.zeros(4, dtype=np.float32))

    with pytest.raises(ValueError, match="sample rate"):
        decode_audio(_Decoder())


def test_prepare_cli_rejects_require_durations_without_ctc(tmp_path: Path):
    from prepare_real_speech import main

    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    assert main([
        "--out", str(tmp_path / "out"),
        "--manifest", str(manifest),
        "--durations", "none",
        "--require-durations",
    ]) == 2


def test_iter_manifest_decodes_relative_audio_path(tmp_path: Path):
    audio_path = tmp_path / "utt.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio": "utt.wav",
                "text": "நாளை வருகிறேன்",
                "language": "ta",
                "speaker": "spk",
                "id": "utt-1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = list(iter_manifest(manifest))
    assert len(rows) == 1
    assert rows[0].sample_rate == 16000
    assert rows[0].audio.shape == (1600,)
    assert rows[0].language == "ta"
    assert rows[0].row_id == "utt-1"


class _FakeCTCTokenizer:
    pad_token_id = 0

    def get_vocab(self):
        return {"<pad>": 0, "a": 1, "b": 2, "|": 3}


class _FakeCTCProcessor:
    def __init__(self):
        self.tokenizer = _FakeCTCTokenizer()
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)

    def __call__(self, audio, **kwargs):
        import torch

        return {"input_values": torch.zeros(1, 1600, dtype=torch.float32)}


class _FakeCTCModel:
    config = SimpleNamespace(conv_stride=[320])

    def __call__(self, input_values):
        import torch

        log_probs = _peaked_log_probs([1, 0, 2], vocab_size=4)
        return SimpleNamespace(logits=torch.from_numpy(log_probs).unsqueeze(0))


def test_ctc_duration_extractor_adapter_returns_character_seconds():
    from codeswitch_tts.data.duration import CTCDurationExtractor

    extractor = CTCDurationExtractor(
        model_id="fake",
        model=_FakeCTCModel(),
        processor=_FakeCTCProcessor(),
        device="cpu",
        sample_rate=16000,
        blank_id=0,
    )
    alignment = extractor.align(np.zeros(1600, dtype=np.float32), 16000, "ab")
    assert alignment is not None
    assert alignment.model_id == "fake"
    assert alignment.n_ctc_frames == 3
    assert alignment.coverage == pytest.approx(1.0)
    assert alignment.char_durations_sec.tolist() == pytest.approx([0.02, 0.02])


def test_iter_parquet_dir_reads_hf_style_audio_bytes(tmp_path: Path):
    import io

    import pandas as pd

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    frame = pd.DataFrame(
        {
            "audio": [{"bytes": buffer.getvalue(), "path": "utt.wav"}],
            "text": ["நாளை வருகிறேன்"],
            "speaker_id": ["spk-1"],
            "language": ["ta"],
        }
    )
    frame.to_parquet(tmp_path / "Tamil.parquet")
    rows = list(iter_parquet_dir(tmp_path, default_language="ta"))
    assert len(rows) == 1
    assert rows[0].sample_rate == 16000
    assert rows[0].speaker == "spk-1"
    assert rows[0].language == "ta"
    # The parquet adapter prefixes the file/row so row ids stay unique across
    # the 67-file Rasa Tamil split.
    assert rows[0].row_id.startswith("Tamil.parquet:0:")
