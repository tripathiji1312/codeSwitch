"""Comprehensive Unit & Regression Tests for CodeSwitch-TTS."""

import pytest
import torch
import numpy as np
from pathlib import Path

from codeswitch_tts.config import CodeSwitchTTSConfig
from codeswitch_tts.frontend.normalizer import TanglishNormalizer
from codeswitch_tts.models.draft_transformer import DraftTransformer, DraftOutput
from codeswitch_tts.models.residual_decoder import ResidualDecoder, ResidualOutput
from codeswitch_tts.engine.speculative_engine import SpeculativeEngine
from codeswitch_tts.engine.streaming_pipeline import ChunkedStreamingPipeline
from codeswitch_tts.utils.audio import AudioProcessor, crossfade_chunks, apply_highpass_filter, save_wav


def test_normalizer_native_tamil():
    norm = TanglishNormalizer()
    res = norm.normalize("உங்கள் பயணம் வெற்றிகரமாக முடிந்தது.")
    assert "உங்கள் பயணம்" in res.normalized_text
    assert res.detected_language == "ta"


def test_normalizer_tanglish_loanwords_preservation():
    norm = TanglishNormalizer()
    res = norm.normalize("Unga cab ippo pickup location-ku vandhuchu.")
    # English transit loanwords must be preserved
    assert "cab" in res.normalized_text
    assert "pickup" in res.normalized_text
    assert "location-க்கு" in res.normalized_text
    # Tanglish words must be transliterated
    assert "உங்க" in res.normalized_text
    assert "இப்போ" in res.normalized_text
    assert "வந்துச்சு" in res.normalized_text
    assert res.detected_language == "code-mixed"


def test_normalizer_transit_entities():
    norm = TanglishNormalizer()
    res = norm.normalize("Driver-ku OTP 5291 sollunga, total fare Rs. 245.")
    # OTP digits spelled out
    assert "OTP five two nine one" in res.normalized_text
    # Fare expanded
    assert "two hundred forty-five rupees" in res.normalized_text
    # English stem + Tamil suffix
    assert "Driver-க்கு" in res.normalized_text


def test_draft_transformer_forward():
    model = DraftTransformer()
    model.eval()
    draft_out = model.generate_chunk("Hello", num_frames=86)
    assert isinstance(draft_out, DraftOutput)
    assert draft_out.logits.shape == (1, 86, 1024)
    assert draft_out.probabilities.shape == (1, 86, 1024)
    assert draft_out.tokens.shape == (1, 86)
    # Probabilities must sum to 1.0 along vocabulary dimension
    probs_sum = draft_out.probabilities.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)


def test_residual_decoder_forward():
    decoder = ResidualDecoder()
    decoder.eval()
    cb0_tokens = torch.randint(0, 1024, (1, 86))
    res_out = decoder(cb0_tokens)
    assert isinstance(res_out, ResidualOutput)
    # 8 residual codebooks
    assert res_out.residual_codebooks.shape == (1, 8, 86)
    # 9 full codebooks (CB0 + CB1-8)
    assert res_out.full_codebooks.shape == (1, 9, 86)
    # Verify first codebook matches cb0_tokens
    assert torch.equal(res_out.full_codebooks[:, 0, :], cb0_tokens)


def test_speculative_verification():
    engine = SpeculativeEngine()
    draft_tokens = torch.randint(0, 1024, (1, 16))
    draft_probs = torch.softmax(torch.randn(1, 16, 1024), dim=-1)
    
    verif_res = engine.verify_chunk(draft_tokens, draft_probs)
    assert verif_res.draft_tokens_evaluated > 0
    assert 0.0 <= verif_res.acceptance_rate <= 100.0
    assert verif_res.verified_tokens.dim() == 2


def test_audio_crossfade_and_filter():
    chunk1 = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    chunk2 = np.cos(np.linspace(0, 10, 1000)).astype(np.float32)
    blended = crossfade_chunks(chunk1, chunk2, crossfade_samples=256)
    assert isinstance(blended, np.ndarray)
    assert len(blended) == len(chunk1) + len(chunk2) - 256

    filtered = apply_highpass_filter(blended, cutoff_hz=60.0, sample_rate=44100)
    assert len(filtered) == len(blended)


def test_end_to_end_streaming_pipeline(tmp_path):
    pipeline = ChunkedStreamingPipeline()
    benchmark_res = pipeline.synthesize("Test streaming audio.")
    assert benchmark_res.duration_sec > 0.0
    assert benchmark_res.ttfa_ms > 0.0
    assert len(benchmark_res.audio_waveform) > 0
    
    test_out = tmp_path / "test.wav"
    save_wav(test_out, benchmark_res.audio_waveform, sample_rate=benchmark_res.sample_rate)
    assert test_out.exists()
