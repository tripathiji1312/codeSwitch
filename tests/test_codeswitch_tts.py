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
from codeswitch_tts.utils.audio import (
    AudioProcessor, DACWrapper, crossfade_chunks, apply_highpass_filter, save_wav
)


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


def test_generate_chunk_argmax_is_deterministic():
    model = DraftTransformer()
    model.eval()
    out_a = model.generate_chunk("Hello", num_frames=8, sample=False)
    out_b = model.generate_chunk("Hello", num_frames=8, sample=False)
    assert torch.equal(out_a.tokens, out_b.tokens)
    assert torch.equal(out_a.tokens, out_a.probabilities.argmax(dim=-1))


def test_draft_forward_uses_per_sample_frame_lengths():
    """Short samples must not be stretched over a longer batch-mate's frames."""
    from codeswitch_tts import DraftModelConfig

    config = DraftModelConfig(d_model=16, n_heads=2, num_layers=1, d_ff=32)
    model = DraftTransformer(config).eval()
    input_ids, mask = model.tokenizer.batch_encode(["Hello", "Hello there"])
    with torch.no_grad():
        batched = model(
            input_ids=input_ids,
            attention_mask=mask,
            frame_lengths=torch.tensor([4, 7]),
            sample=False,
        )
        one = model(
            input_ids=input_ids[:1],
            attention_mask=mask[:1],
            num_frames=4,
            sample=False,
        )
    assert batched.logits.shape == (2, 7, 1024)
    assert batched.tokens.shape == (2, 7)
    # The first sample's four valid frames must match a dedicated four-frame
    # forward pass; only the padded tail may differ.
    assert torch.allclose(batched.logits[0, :4], one.logits[0], atol=1e-5)


def test_frame_decoder_forward_and_per_sample_masking():
    """The opt-in frame decoder must condition on text and mask padded frames."""
    from codeswitch_tts import DraftModelConfig

    config = DraftModelConfig(
        d_model=16, n_heads=2, num_layers=1, d_ff=32,
        frame_decoder_layers=1, max_frames=32,
    )
    model = DraftTransformer(config).eval()
    input_ids, mask = model.tokenizer.batch_encode(["Hello", "Hello there"])
    with torch.no_grad():
        batched = model(
            input_ids=input_ids,
            attention_mask=mask,
            frame_lengths=torch.tensor([4, 7]),
            sample=False,
        )
        one = model(
            input_ids=input_ids[:1],
            attention_mask=mask[:1],
            num_frames=4,
            sample=False,
        )
        again = model(
            input_ids=input_ids,
            attention_mask=mask,
            frame_lengths=torch.tensor([4, 7]),
            sample=False,
        )
    assert batched.logits.shape == (2, 7, 1024)
    assert batched.tokens.shape == (2, 7)
    assert torch.allclose(batched.logits[0, :4], one.logits[0], atol=1e-5)
    assert torch.equal(batched.tokens, again.tokens)
    assert torch.equal(batched.tokens, batched.probabilities.argmax(dim=-1))


def test_legacy_draft_checkpoint_has_no_frame_decoder_keys():
    """Default configs keep the old state dict so v2/v3/v4 checkpoints load."""
    from codeswitch_tts import DraftModelConfig

    config = DraftModelConfig(d_model=16, n_heads=2, num_layers=1, d_ff=32)
    model = DraftTransformer(config)
    assert config.frame_decoder_layers == 0
    assert model.frame_decoder is None
    assert not any(key.startswith("frame_decoder.") for key in model.state_dict())
    restored = DraftTransformer(config)
    restored.load_state_dict(model.state_dict())


def test_arg_parser_exposes_sample_mode():
    import run_full_evaluation as R

    assert R.build_arg_parser().parse_args([]).sample_mode == "sample"
    assert R.build_arg_parser().parse_args(["--sample-mode", "argmax"]).sample_mode == "argmax"


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


def test_dacwrapper_refuses_random_hf_local_weights(monkeypatch):
    """A missing pretrained DAC must fail, not silently install random weights."""
    from transformers import DacModel

    monkeypatch.setattr(
        DacModel, "from_pretrained",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))),
    )
    try:  # make the test hermetic when descript-audio-codec is installed
        import dac

        monkeypatch.setattr(
            dac.utils, "download",
            lambda **k: (_ for _ in ()).throw(RuntimeError("offline")),
        )
    except ImportError:
        pass

    with pytest.raises(RuntimeError):
        DACWrapper()
    wrapper = DACWrapper(allow_fallback=True)
    assert wrapper.backend == "fallback_neural"
    assert wrapper.model is None
    assert "offline" in (wrapper.load_error or "")


def test_audioprocessor_passed_model_skips_loading():
    ap = AudioProcessor(dac_model=object())
    assert ap.dac.backend == "passed_model"


def test_end_to_end_streaming_pipeline(tmp_path, monkeypatch):
    # Local/CPU runs have no Indic Parler-TTS weights, so explicitly allow the
    # telemetry-only fallback (its audio is synthetic, not speech). Stub the DAC
    # loader so this test never downloads the 306 MB pretrained checkpoint.
    monkeypatch.setattr(
        DACWrapper, "_load_dac",
        lambda self: (setattr(self, "backend", "fallback_neural"),
                      setattr(self, "model", None)),
    )
    config = CodeSwitchTTSConfig()
    config.system.allow_untrained_fallback = True
    pipeline = ChunkedStreamingPipeline(config=config)
    benchmark_res = pipeline.synthesize("Test streaming audio.")
    assert benchmark_res.duration_sec > 0.0
    assert benchmark_res.ttfa_ms > 0.0
    assert len(benchmark_res.audio_waveform) > 0
    
    test_out = tmp_path / "test.wav"
    save_wav(test_out, benchmark_res.audio_waveform, sample_rate=benchmark_res.sample_rate)
    assert test_out.exists()


def test_assets_loading():
    """Verify voice profiles, gazetteers, and character vocabularies load from assets."""
    from codeswitch_tts import (
        load_voice_profiles,
        load_proper_nouns,
        load_character_vocab,
        TanglishNormalizer,
    )
    import os

    # 1. Voice profiles
    voices = load_voice_profiles()
    assert "ta_female_neutral" in voices
    assert "ta_male_neutral" in voices
    assert os.path.exists(voices["ta_female_neutral"].ref_audio_path)
    assert len(voices["ta_female_neutral"].ref_text) > 0

    # 2. Gazetteer proper nouns
    proper_nouns = load_proper_nouns()
    assert "chennai central" in proper_nouns or "omr" in proper_nouns
    assert "rapido" in proper_nouns

    # 3. Character vocab dictionary
    vocab = load_character_vocab()
    assert len(vocab) > 1000

    # 4. Normalizer integration
    norm = TanglishNormalizer()
    res = norm.normalize("Rapido pickup at OMR, OTP 4321.")
    assert "OTP four three two one" in res.normalized_text
    assert "OMR" in res.normalized_text or "omr" in res.normalized_text.lower()


def test_protobuf_runtime_version_fallback_api():
    """Protobuf gencode calls ``runtime_version.Domain.PUBLIC`` at import time.

    The fallback shim must therefore expose ``Domain`` as a class with PUBLIC,
    never as a function (that regressed into "'function' object has no
    attribute 'PUBLIC'" on Kaggle and broke the Parler-TTS import).
    """
    from codeswitch_tts._compat import install_protobuf_runtime_version_fallback

    install_protobuf_runtime_version_fallback()

    from google.protobuf import runtime_version

    assert hasattr(runtime_version, "Domain")
    assert runtime_version.Domain.PUBLIC is not None
    assert runtime_version.Domain.GOOGLE_INTERNAL is not None
    # Must be callable with the exact signature used by modern *_pb2.py gencode.
    assert (
        runtime_version.ValidateProtobufRuntimeVersion(
            runtime_version.Domain.PUBLIC, 5, 29, 5, "", "regression_pb2.py"
        )
        is None
    )

