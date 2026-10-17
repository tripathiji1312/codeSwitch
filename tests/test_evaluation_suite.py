"""Tests for the CodeSwitch-TTS evaluation suite.

Covers the pieces that can run without GPU/Parler:
  * benchmark data integrity (60-utterance set, disjoint training corpus)
  * linguistic zone instrumentation (char -> frame alignment, boundaries)
  * speculative rejection sampling math
  * CER cleaning protocol
  * length model fitting
  * result aggregation + LaTeX tables + figure generation (fabricated rows)
  * checkpoint/JSONL round-trips for the shared data structures
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import codeswitch_benchmark_data as data
import run_full_evaluation as R
from train_draft_distill import LengthModel, _CodeCaptureStreamer


# ===========================================================================
# Benchmark data
# ===========================================================================
def test_eval_set_balance_and_uniqueness():
    assert len(data.EVAL_UTTERANCES) == 60
    counts = data.eval_register_counts()
    assert counts == {
        "native_tamil": 20,
        "romanized_tanglish": 20,
        "transit_codemixed": 20,
    }
    ids = [u["id"] for u in data.EVAL_UTTERANCES]
    texts = [u["text"] for u in data.EVAL_UTTERANCES]
    assert len(set(ids)) == 60
    assert len(set(texts)) == 60
    for utt in data.EVAL_UTTERANCES:
        assert utt["register"] in data.REGISTER_LABELS
        assert utt["text"].strip()


def test_training_corpus_disjoint_and_sized():
    corpus = data.build_training_corpus(n=120, seed=3)
    assert len(corpus) == 120
    assert len(set(corpus)) == 120
    data.assert_corpus_disjoint_from_eval(corpus)
    # A deliberate leak must be caught
    with pytest.raises(ValueError):
        data.assert_corpus_disjoint_from_eval([data.EVAL_UTTERANCES[0]["text"]])


# ===========================================================================
# Zone instrumentation
# ===========================================================================
def _zones_for(text: str, n_frames: int = 120):
    from codeswitch_tts import TanglishNormalizer

    norm = TanglishNormalizer().normalize(text)
    spans = R.find_transit_spans(norm.normalized_text, norm.entity_replacements)
    char_zones = R.label_char_zones(norm.normalized_text, spans)
    return norm, char_zones, *R.assign_frame_zones(char_zones, n_frames)


def test_zones_pure_tamil_has_no_boundaries():
    norm, char_zones, frame_zones, transitions = _zones_for(
        "வணக்கம், உங்கள் ஆர்டர் தயாராகிவிட்டது."
    )
    assert (char_zones == R.ZONE_TAMIL).any()
    assert transitions == 0
    assert (frame_zones == R.ZONE_BOUNDARY).sum() == 0
    assert (frame_zones == R.ZONE_TAMIL).sum() > 0
    assert frame_zones.shape[0] == 120


def test_zones_transit_codemix_detects_all_four_zones():
    norm, char_zones, frame_zones, transitions = _zones_for(
        "Driver-ku OTP 8421 sollunga, fare Rs. 350."
    )
    norm_text = norm.normalized_text
    # verbalizer expanded the OTP digit-by-digit
    assert "eight four two one" in norm_text.lower()
    present = set(np.unique(frame_zones).tolist())
    assert R.ZONE_TAMIL in present            # சொல்லுங்க etc.
    assert R.ZONE_ENGLISH in present          # "Driver", "fare"
    assert R.ZONE_TRANSIT in present          # verbalised entities
    assert R.ZONE_BOUNDARY in present         # cross-lingual switches
    assert transitions >= 2
    assert (frame_zones == R.ZONE_BOUNDARY).sum() <= 3 * transitions


def test_frame_zone_alignment_is_bounded_and_monotonic():
    text = "Unga order vandhuduchu, OTP 8421 confirm aagiduchu."
    for n_frames in (30, 86, 300):
        _, _, frame_zones, transitions = _zones_for(text, n_frames=n_frames)
        assert frame_zones.shape == (n_frames,)
        assert set(np.unique(frame_zones).tolist()) <= {
            R.ZONE_NONE, R.ZONE_TAMIL, R.ZONE_ENGLISH, R.ZONE_TRANSIT, R.ZONE_BOUNDARY
        }
        assert (frame_zones == R.ZONE_BOUNDARY).sum() <= 3 * max(transitions, 1)


# ===========================================================================
# Speculative rejection sampling
# ===========================================================================
def _one_hot_probs(n_frames: int, vocab: int, token: int, floor: float = 0.0) -> np.ndarray:
    probs = np.full((n_frames, vocab), floor, dtype=np.float32)
    probs[:, token] = 1.0 - floor * (vocab - 1)
    return probs


def test_rejection_sampling_accepts_when_distributions_match():
    rng = np.random.RandomState(0)
    vocals = 16
    p = _one_hot_probs(10, vocals, token=5)
    q = p.copy()
    tokens = np.full(10, 5, dtype=np.int64)
    kept, accepted, n_resampled = R.rejection_sample(q, p, tokens, rng)
    assert accepted.all()
    assert n_resampled == 0
    assert (kept == tokens).all()
    assert R.acceptance_rate(accepted) == pytest.approx(100.0)


def test_rejection_sampling_rejects_impossible_draft_token():
    rng = np.random.RandomState(1)
    vocab = 8
    p = _one_hot_probs(5, vocab, token=1)
    q = _one_hot_probs(5, vocab, token=2)
    tokens = np.full(5, 1, dtype=np.int64)
    kept, accepted, n_resampled = R.rejection_sample(q, p, tokens, rng)
    assert not accepted.any()
    assert n_resampled == 5
    # resampling must come from the residual / target support: token 2
    assert (kept == 2).all()


def test_rejection_sampling_determinism_and_rate_range():
    vocab = 32
    rng_probs = np.random.RandomState(7)
    p = rng_probs.dirichlet(np.ones(vocab), size=20).astype(np.float32)
    q = rng_probs.dirichlet(np.ones(vocab), size=20).astype(np.float32)
    tokens = np.array([int(np.random.RandomState(100 + i).choice(vocab)) for i in range(20)])
    kept_a, acc_a, _ = R.rejection_sample(q, p, tokens, np.random.RandomState(11))
    kept_b, acc_b, _ = R.rejection_sample(q, p, tokens, np.random.RandomState(11))
    assert (kept_a == kept_b).all()
    assert (acc_a == acc_b).all()
    rate = R.acceptance_rate(acc_a)
    assert 0.0 <= rate <= 100.0


def test_acceptance_rate_handles_empty():
    assert R.acceptance_rate(None) is None
    assert R.acceptance_rate(np.asarray([], dtype=bool)) is None


def test_synthetic_probs_are_normalised():
    probs = R.TeacherForcedScorer.synthetic_probs(6, 64, np.random.RandomState(2))
    assert probs.shape == (6, 64)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)


# ===========================================================================
# CER cleaning protocol
# ===========================================================================
def test_clean_for_cer_protocol():
    cleaned = R.clean_for_cer("Driver-ku OTP 8421 sollunga, fare Rs. 350!")
    assert "driver" in cleaned
    assert "eight four two one" in cleaned
    assert "three five zero" in cleaned
    assert "-" not in cleaned and "," not in cleaned and "!" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_for_cer_identical_texts_agree():
    a = R.clean_for_cer("OTP 3312 one minute-la expire aagum.")
    b = R.clean_for_cer("otp 3312 one minute la expire aagum")
    assert a == b


def test_asr_score_zero_for_identical():
    cer, wer = R.ASREvaluator.score("OTP 8421 sollunga", "OTP 8421 sollunga.")
    assert cer == pytest.approx(0.0)
    assert wer == pytest.approx(0.0)
    cer2, wer2 = R.ASREvaluator.score("OTP 8421 sollunga", None)
    assert cer2 is None and wer2 is None


# ===========================================================================
# Length model
# ===========================================================================
def test_length_model_fit_predict_roundtrip():
    chars = [40, 60, 80, 100, 120]
    frames = [int(6.6 * c + 2) for c in chars]
    model = LengthModel.fit(chars, frames)
    assert model.a == pytest.approx(6.6, abs=0.5)
    assert model.mae_frames < 3.0
    pred = model.predict("அ" * 80)
    assert model.min_frames <= pred <= model.max_frames
    # clamps for extreme inputs
    assert model.predict("x") == model.min_frames
    assert model.predict("x" * 10_000) == model.max_frames
    roundtrip = LengthModel.from_dict(model.to_dict())
    assert roundtrip.predict("அ" * 80) == pred
    assert roundtrip.to_dict() == model.to_dict()
    assert roundtrip.n_samples == model.n_samples


def test_length_model_degenerate_fit():
    model = LengthModel.fit([], [])
    assert model.min_frames >= 20
    assert model.max_frames >= model.min_frames
    assert model.predict("abc") >= model.min_frames


# ===========================================================================
# Code capture streamer tensor handling (no model needed)
# ===========================================================================
class _DummyDecoder:
    num_codebooks = 9


def test_code_capture_streamer_shapes():
    import torch

    streamer = _CodeCaptureStreamer(_DummyDecoder(), generation_config=None)
    streamer.put(torch.zeros(9, 1, dtype=torch.long))          # initial delayed prompt
    streamer.put(torch.arange(9, dtype=torch.long))            # per-step (9,)
    streamer.put(torch.arange(9, dtype=torch.long)[:, None])   # per-step (9, 1)
    streamer.put(torch.zeros(1, 9, 1, dtype=torch.long))       # tolerated (1, 9, 1)
    assert streamer.token_cache is not None
    assert streamer.token_cache.shape[0] == 9
    assert streamer.token_cache.shape[1] == 4
    # No generation config -> extraction reports unavailable rather than crashing
    assert streamer.extract_codes() is None or streamer.extract_codes().shape[0] == 9


# ===========================================================================
# Aggregation, tables, figures
# ===========================================================================
def _fabricated_rows() -> list[dict]:
    rows: list[dict] = []
    zone_stats_c = {
        "1": {"n_frames": 100, "accept_true": 70.0, "accept_synth": 78.0},
        "2": {"n_frames": 40, "accept_true": 66.0, "accept_synth": 75.0},
        "3": {"n_frames": 20, "accept_true": 61.0, "accept_synth": 39.5},
        "4": {"n_frames": 12, "accept_true": 48.0, "accept_synth": 33.0},
    }
    for i in range(3):
        rows.append({
            "schema": R.SCHEMA_VERSION, "config": "A", "config_label": R.CONFIG_LABELS["A"],
            "utt_id": f"ta-0{i + 1}", "register": "native_tamil", "utt_index": i,
            "text": "x", "normalized_text": "x", "sample_rate": 44100,
            "ttfa_compute_ms": None, "ttfa_e2e_ms": 1200.0 + 50 * i,
            "total_ms": 1500.0 + 40 * i, "rtf": 0.5, "audio_sec": 3.0,
            "accept_true": None, "accept_synth": None, "cer": 0.20 + 0.01 * i,
            "wer": 0.4, "zone_stats": {}, "chunks": [], "verify_mode": "none",
            "error": None,
        })
        rows.append({
            "schema": R.SCHEMA_VERSION, "config": "C", "config_label": R.CONFIG_LABELS["C"],
            "utt_id": f"ta-0{i + 1}", "register": "native_tamil", "utt_index": i,
            "text": "x", "normalized_text": "x", "sample_rate": 44100,
            "ttfa_compute_ms": 260.0 + 5 * i, "ttfa_e2e_ms": 300.0 + 5 * i,
            "total_ms": 3200.0 + 100 * i, "rtf": 1.05, "audio_sec": 3.1,
            "draft_ms": 20.0 + i, "verify_ms_total": 90.0, "residual_ms_total": 130.0,
            "dac_ms_total": 200.0, "accept_true": 68.0 + i, "accept_synth": 74.0 + i,
            "cer": 0.15 - 0.005 * i, "wer": 0.3, "n_frames": 172, "n_transitions": 4,
            "zone_stats": zone_stats_c, "verify_mode": "true", "error": None,
            "chunks": [
                {"idx": 0, "n_frames": 86, "verify_ms": 90.0, "residual_ms": 130.0,
                 "dac_ms": 200.0, "ready_ms": 300.0 + 5 * i, "accept_true": 70.0,
                 "accept_synth": 75.0},
                {"idx": 1, "n_frames": 86, "verify_ms": 85.0, "residual_ms": 120.0,
                 "dac_ms": 210.0, "ready_ms": 900.0 + 5 * i, "accept_true": 66.0,
                 "accept_synth": 72.0},
            ],
        })
    return rows


def test_aggregation_and_zone_weighting():
    rows = _fabricated_rows()
    agg = R.aggregate_results(rows, ["A", "B", "C"])
    assert agg["A"]["n"] == 3 and agg["C"]["n"] == 3
    assert agg["B"]["n"] == 0
    assert agg["C"]["ttfa_compute"]["p50"] == pytest.approx(265.0)
    assert agg["C"]["ttfa_compute_cold"] == pytest.approx(260.0)
    # warm excludes the first utterance
    assert agg["C"]["ttfa_compute_warm"]["n"] == 2
    assert agg["C"]["cer"]["mean"] == pytest.approx((0.15 + 0.145 + 0.14) / 3)
    zones = agg["C"]["zones"]
    assert zones["1"]["n_frames"] == 300
    assert zones["1"]["accept_true"] == pytest.approx(70.0)
    assert zones["4"]["accept_true"] == pytest.approx(48.0)
    assert zones["3"]["accept_synth"] == pytest.approx(39.5)
    # A has no zones
    assert agg["A"]["zones"]["1"]["n_frames"] == 0


def test_latex_tables_render():
    agg = R.aggregate_results(_fabricated_rows(), ["A", "C"])
    table1 = R.build_table_I(agg)
    table2 = R.build_table_II(agg)
    for table in (table1, table2):
        assert "\\begin{tabular}" in table
        assert "\\bottomrule" in table
        assert "\\end{tabular}" in table
    assert "Pure Tamil" in table2
    assert "$\\pm$" in table1  # std formatting present
    assert "49.5" in table2 or "48.0" in table2  # boundary value rendered
    assert "--" in table1      # missing values rendered


def test_tables_and_figures_from_fabricated_rows(tmp_path: Path):
    rows = _fabricated_rows()
    agg = R.aggregate_results(rows, ["A", "C"])
    results = tmp_path / "raw.jsonl"
    R.save_results(results, rows)
    reloaded = R.load_results(results)
    assert len(reloaded) == len(rows)
    assert reloaded[0]["config"] == "A"

    fig1 = R.figure1_latency_quality(agg, tmp_path)
    fig2 = R.figure2_zone_acceptance(agg, tmp_path)
    fig3 = R.figure3_stream_timeline(rows, tmp_path)
    for result, stem in ((fig1, "fig1_latency_quality_tradeoff"),
                         (fig2, "fig2_boundary_acceptance_breakdown"),
                         (fig3, "fig3_streaming_chunk_timeline")):
        assert result, f"{stem} was not produced"
        for path in result:
            assert Path(path).exists()
            assert Path(path).stat().st_size > 1000
    assert (tmp_path / "fig1_latency_quality_tradeoff.png").exists()
    assert (tmp_path / "fig2_boundary_acceptance_breakdown.pdf").exists()


def test_figure2_annotation_drop_direction():
    """Boundary acceptance must be plotted from real zone stats (dropping)."""
    rows = _fabricated_rows()
    agg = R.aggregate_results(rows, ["C"])
    zones = agg["C"]["zones"]
    base = zones["1"]["accept_true"]
    assert zones["4"]["accept_true"] < base  # boundary harder than pure Tamil
    boundary_delta = zones["4"]["accept_true"] - base
    assert boundary_delta < 0


def test_results_roundtrip_preserves_chunk_telemetry(tmp_path: Path):
    rows = _fabricated_rows()
    path = tmp_path / "res.jsonl"
    R.save_results(path, rows)
    loaded = R.load_results(path)
    c_row = [r for r in loaded if r["config"] == "C"][0]
    assert len(c_row["chunks"]) == 2
    assert c_row["chunks"][1]["ready_ms"] > c_row["chunks"][0]["ready_ms"]
    # JSON is UTF-8 and non-ASCII safe
    assert json.dumps(c_row, ensure_ascii=False)


def test_zone_stats_attachment_helper():
    synth = R.EvalSynthesizer.__new__(R.EvalSynthesizer)  # bypass __init__
    frame_zones = np.array([1, 1, 2, 3, 3, 4], dtype=np.int64)
    accept_true = np.array([True, True, False, True, False, False])
    accept_synth = np.array([True, True, True, True, False, False])
    row: dict = {}
    synth._attach_zone_stats(row, frame_zones, accept_true, accept_synth)
    zones = row["zone_stats"]
    assert zones["1"]["n_frames"] == 2
    assert zones["1"]["accept_true"] == pytest.approx(100.0)
    assert zones["3"]["accept_true"] == pytest.approx(50.0)
    assert zones["4"]["accept_true"] == pytest.approx(0.0)
    assert zones["2"]["accept_synth"] == pytest.approx(100.0)


def test_module_arg_parsers_exist():
    eval_args = R.build_arg_parser().parse_args([])
    assert eval_args.ckpt == "codeswitch_ckpt"
    assert eval_args.configs == "A,B,C"
    assert math.isfinite(eval_args.seed)

    import train_draft_distill as T

    train_args = T.build_arg_parser().parse_args([])
    assert train_args.num_sentences == 400
    assert train_args.target_source == "auto"
    assert train_args.dry_run is False
