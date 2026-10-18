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
    frame_zones, transitions, switch_frames = R.assign_frame_zones(char_zones, n_frames)
    return norm, char_zones, frame_zones, transitions, switch_frames


def test_zones_pure_tamil_has_no_boundaries():
    norm, char_zones, frame_zones, transitions, switch_frames = _zones_for(
        "வணக்கம், உங்கள் ஆர்டர் தயாராகிவிட்டது."
    )
    assert (char_zones == R.ZONE_TAMIL).any()
    assert transitions == 0
    assert switch_frames == []
    assert (frame_zones == R.ZONE_BOUNDARY).sum() == 0
    assert (frame_zones == R.ZONE_TAMIL).sum() > 0
    assert frame_zones.shape[0] == 120


def test_zones_transit_codemix_detects_all_four_zones():
    norm, char_zones, frame_zones, transitions, switch_frames = _zones_for(
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
    assert len(switch_frames) == transitions
    assert switch_frames == sorted(switch_frames)
    assert (frame_zones == R.ZONE_BOUNDARY).sum() <= 3 * transitions


def test_frame_zone_alignment_is_bounded_and_monotonic():
    text = "Unga order vandhuduchu, OTP 8421 confirm aagiduchu."
    for n_frames in (30, 86, 300):
        _, _, frame_zones, transitions, _sw = _zones_for(text, n_frames=n_frames)
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


def test_stats_bootstrap_ci_brackets_median():
    vals = [10.0, 11.0, 12.0, 13.0, 30.0, 9.0, 10.5, 11.5]
    stat = R._stats(vals, ci=True)
    assert stat["p50"] is not None
    assert stat["ci_lo"] <= stat["p50"] <= stat["ci_hi"]
    # deterministic for a fixed seed
    again = R._stats(vals, ci=True)
    assert (stat["ci_lo"], stat["ci_hi"]) == (again["ci_lo"], again["ci_hi"])
    # tiny samples do not pretend to have an interval
    tiny = R._stats([1.0, 2.0], ci=True)
    assert tiny["ci_lo"] is None and tiny["ci_hi"] is None
    # no CI requested -> no CI keys
    assert "ci_lo" not in R._stats(vals)


def test_tex_ci_formats_scaled_brackets():
    entry = R._stats([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ci=True)
    cell = R._tex_ci(entry, 1, scale=100.0)
    assert cell.startswith("35.0 [") and cell.endswith("]")
    assert R._tex_ci({}, 1) == ""
    assert R._tex_ci({"p50": 1.0}, 1) == ""


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
            "gate_threshold": 0.0, "gate_stat": "utt_min", "fallback": False,
            "draft_confidence": 0.42 + 0.02 * i,
            "switch_frames": [40, 120],
            "frame_zones_b64": R.pack_u8(np.array(
                [1] * 40 + [4] * 3 + [2] * 77 + [4] * 3 + [1] * 49, dtype=np.uint8)),
            "frame_accept_true_b64": R.pack_bits(
                np.array([(j % 3 != 0) for j in range(172)], dtype=bool)),
            "frame_accept_synth_b64": R.pack_bits(
                np.array([(j % 4 != 0) for j in range(172)], dtype=bool)),
            "frame_valid_b64": R.pack_bits(np.ones(172, dtype=bool)),
            "chunks": [
                {"idx": 0, "n_frames": 86, "verify_ms": 90.0, "residual_ms": 130.0,
                 "dac_ms": 200.0, "ready_ms": 300.0 + 5 * i, "accept_true": 70.0,
                 "accept_synth": 75.0, "confidence": 0.44},
                {"idx": 1, "n_frames": 86, "verify_ms": 85.0, "residual_ms": 120.0,
                 "dac_ms": 210.0, "ready_ms": 900.0 + 5 * i, "accept_true": 66.0,
                 "accept_synth": 72.0, "confidence": 0.38},
            ],
        })
        # gated variant at tau=0.5: utterance 0 falls back to AR, 1 and 2 stay speculative
        if i == 0:
            rows.append({
                "schema": R.SCHEMA_VERSION, "config": "C", "config_label": R.CONFIG_LABELS["C"],
                "utt_id": f"ta-0{i + 1}", "register": "native_tamil", "utt_index": i,
                "text": "x", "normalized_text": "x", "sample_rate": 44100,
                "ttfa_compute_ms": 900.0, "ttfa_e2e_ms": 1100.0, "total_ms": 7000.0,
                "rtf": 2.2, "audio_sec": 3.1, "accept_true": None, "accept_synth": None,
                "cer": 0.14, "wer": 0.3, "n_frames": None, "zone_stats": {}, "chunks": [],
                "verify_mode": "ar_fallback", "error": None, "fallback": True,
                "gate_threshold": 0.5, "gate_stat": "utt_min", "draft_confidence": 0.21,
            })
        else:
            rows.append({
                "schema": R.SCHEMA_VERSION, "config": "C", "config_label": R.CONFIG_LABELS["C"],
                "utt_id": f"ta-0{i + 1}", "register": "native_tamil", "utt_index": i,
                "text": "x", "normalized_text": "x", "sample_rate": 44100,
                "ttfa_compute_ms": 270.0 + 5 * i, "ttfa_e2e_ms": 310.0 + 5 * i,
                "total_ms": 3300.0 + 100 * i, "rtf": 1.06, "audio_sec": 3.1,
                "accept_true": 50.0 + i, "accept_synth": 55.0 + i,
                "cer": 0.12 - 0.005 * i, "wer": 0.3, "n_frames": 172, "n_transitions": 4,
                "zone_stats": zone_stats_c, "verify_mode": "true", "error": None,
                "fallback": False, "gate_threshold": 0.5, "gate_stat": "utt_min",
                "draft_confidence": 0.51 + 0.02 * i, "chunks": [],
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


def _rows_multi_register() -> list[dict]:
    """Config-C rows spanning the three registers, one poisoned by an ASR loop."""
    rows: list[dict] = []
    cer_by_reg = {"native_tamil": 0.10, "romanized_tanglish": 0.50, "transit_codemixed": 9.0}
    for i, reg in enumerate(cer_by_reg):
        rows.append({
            "schema": R.SCHEMA_VERSION, "config": "C", "config_label": R.CONFIG_LABELS["C"],
            "utt_id": f"u-0{i + 1}", "register": reg, "utt_index": i,
            "text": "x", "normalized_text": "x", "sample_rate": 44100,
            "ttfa_compute_ms": 100.0 + i, "ttfa_e2e_ms": 110.0 + i,
            "total_ms": 200.0, "rtf": 0.07, "audio_sec": 3.0,
            "accept_true": 40.0 + i, "accept_synth": 25.0 + i,
            "logq_draft": -3.20 + 0.01 * i, "logq_kept": -3.00 + 0.01 * i,
            "cer": cer_by_reg[reg], "wer": 0.3, "zone_stats": {}, "chunks": [],
            "verify_mode": "true", "error": None,
        })
    return rows


def test_register_aggregation_and_asr_failures():
    agg = R.aggregate_results(_rows_multi_register(), ["C"])["C"]
    regs = agg["registers"]
    assert set(regs) == {"native_tamil", "romanized_tanglish", "transit_codemixed"}
    assert regs["native_tamil"]["cer"]["p50"] == pytest.approx(0.10)
    assert regs["transit_codemixed"]["asr_fail"] == 1
    assert agg["asr_fail"] == 1
    assert agg["asr_fail_utts"] == ["u-03"]
    # the median survives the hallucination row, the mean does not
    assert agg["cer"]["p50"] == pytest.approx(0.50)
    assert agg["cer"]["mean"] > 3.0
    # target-correction diagnostics aggregate over the rows that carry them
    assert agg["logq_draft"]["mean"] == pytest.approx(-3.19)
    assert agg["logq_kept"]["mean"] > agg["logq_draft"]["mean"]


def test_table_three_and_median_cer_reporting():
    agg = R.aggregate_results(_rows_multi_register(), ["C"])
    table3 = R.build_table_III(agg)
    assert "\\begin{tabular}" in table3 and "\\bottomrule" in table3
    assert "Native Tamil" in table3 and "Transit Code-Mixed" in table3
    assert "logq" not in table3  # diagnostics must not leak into the LaTeX
    table1 = R.build_table_I(agg)
    assert "CER p50 (\\%)" in table1
    assert "1/3" in table1                       # ASR-failure census
    assert "log-likelihood" in table1            # target-correction footnote
    assert f"{agg['C']['logq_draft']['mean']:.2f}" in table1
    assert f"{agg['C']['logq_kept']['mean']:.2f}" in table1


def test_table_three_empty_aggregate_placeholder():
    table3 = R.build_table_III({"A": {"n": 0}})
    assert "\\begin{tabular}" in table3 and "\\end{tabular}" in table3
    assert "-- & --" in table3


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
    fig4 = R.figure4_scaling_and_registers(rows, agg, tmp_path)
    for result, stem in ((fig1, "fig1_latency_quality_tradeoff"),
                         (fig2, "fig2_boundary_acceptance_breakdown"),
                         (fig3, "fig3_streaming_chunk_timeline"),
                         (fig4, "fig4_scaling_and_registers")):
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
    assert eval_args.gate_thresholds == "0.0"
    assert eval_args.gate_stat in ("utt_min", "utt_mean")

    import train_draft_distill as T

    train_args = T.build_arg_parser().parse_args([])
    assert train_args.num_sentences == 300
    assert train_args.target_source == "auto"
    assert train_args.dry_run is False


# ===========================================================================
# Confidence gate, frame telemetry, distance-to-switch analysis
# ===========================================================================
def test_switch_distances_and_buckets():
    dist = R.switch_distances(20, [5, 12])
    # frames before the first switch are sentinel (n_frames) -> "far" bucket
    assert dist[0] == 20 and dist[4] == 20
    assert dist[5] == 0 and dist[6] == 1 and dist[11] == 6
    assert dist[12] == 0 and dist[19] == 7
    assert dist.dtype.kind == "i"
    # no switches -> everything far
    assert (R.switch_distances(10, []) == 10).all()
    # no frames -> empty
    assert R.switch_distances(0, [1]).size == 0


def test_pack_roundtrip_frame_telemetry():
    flags = np.array([True, False, True, True, False, False, True, False, True])
    packed = R.pack_bits(flags)
    assert isinstance(packed, str) and packed
    assert (R.unpack_bits(packed, flags.size) == flags).all()
    zones = R.pack_u8([0, 1, 2, 3, 4, 4])
    assert (R.unpack_u8(zones) == np.array([0, 1, 2, 3, 4, 4], dtype=np.uint8)).all()


def test_frame_accept_arrays_requires_valid_bits():
    row = _fabricated_rows()[1]  # first config-C row
    arrs = R.frame_accept_arrays(row)
    assert arrs is not None
    accept, valid, dist = arrs
    assert accept.size == 172 and valid.all() and dist.shape == (172,)
    # rows without packed telemetry (old JSONLs / fallbacks) are skipped
    assert R.frame_accept_arrays({"n_frames": 10}) is None
    assert R.frame_accept_arrays({"n_frames": 0, "frame_accept_true_b64": "AA==",
                                  "frame_valid_b64": "AA=="}) is None


def test_switch_distance_table_aggregates_and_pairs():
    rows = _fabricated_rows()
    table = R.switch_distance_table(rows)
    assert [r["bucket"] for r in table] == [b[0] for b in R.SWITCH_DIST_BUCKETS]
    by_bucket = {r["bucket"]: r for r in table}
    # frames 0-1 after a switch exist for both switches of every utterance: 3 x 2 x 2
    assert by_bucket["0-1"]["n_frames"] == 12
    assert by_bucket["0-1"]["n_utts"] == 3
    assert by_bucket["16+"]["n_frames"] > 0        # frames before the first switch
    assert by_bucket["0-1"]["mean"] is not None
    # paired stats against the far bucket are reported for the near bucket
    assert by_bucket["0-1"]["n_paired"] >= 2
    assert by_bucket["0-1"]["p_vs_far"] is not None
    # no config-C rows with frame data -> graceful empty analysis
    empty = R.switch_distance_table([{"config": "A", "utt_id": "x"}])
    assert all(r["mean"] is None for r in empty)


def test_gate_sweep_aggregation_and_fallback_exclusion():
    rows = _fabricated_rows()
    sweep = R.aggregate_gate_sweep(rows)
    assert [round(e["threshold"], 2) for e in sweep] == [0.0, 0.5]
    open_pt, gated = sweep
    assert open_pt["fallback_utts"] == 0
    assert gated["fallback_utts"] == 1
    assert gated["fallback_rate"] == pytest.approx(1 / 3)
    # acceptance excludes the fallback row, latency/CER include it
    assert gated["accept_true"]["n"] == 2
    assert gated["cer"]["n"] == 3
    assert gated["cer"]["p50"] is not None
    # here the gated mixture is slower (fallback cost) on the latency side
    assert gated["total"]["p50"] > open_pt["total"]["p50"]


def test_gate_threshold_filtering_in_aggregate():
    rows = _fabricated_rows()
    default = R.aggregate_results(rows, ["C"], gate_threshold=0.0)
    gated = R.aggregate_results(rows, ["C"], gate_threshold=0.5)
    assert default["C"]["n"] == 3          # un-gated rows only
    assert gated["C"]["n"] == 3            # 2 speculative + 1 fallback row
    assert gated["C"]["fallback_utts"] == 1
    # a threshold with no rows at all never empties the comparison (all C rows used)
    fallback = R.aggregate_results(rows, ["C"], gate_threshold=0.9)
    assert fallback["C"]["n"] == 6


def test_synth_C_falls_back_below_threshold(monkeypatch):
    """The gate hands low-confidence utterances to the AR path and labels it."""
    synth = R.EvalSynthesizer.__new__(R.EvalSynthesizer)
    synth.args = type("A", (), {"boundary_frames": 3})()
    synth.scorer = object()
    calls = {"chunked": 0, "ar": 0}

    def fake_conf(utt, seed, gate_stat="utt_min"):
        return 0.10

    def fake_chunked(utt, utt_index, seed, verify):
        calls["chunked"] += 1
        return {"verify_mode": "true", "accept_true": 40.0, "n_transitions": 2}, np.zeros(4, dtype=np.float32)

    def fake_ar(utt, utt_index, seed):
        calls["ar"] += 1
        return {"config": "A", "accept_true": None, "n_frames": None,
                "ttfa_compute_ms": 900.0, "zone_stats": {}}, np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(synth, "_draft_confidence", fake_conf)
    monkeypatch.setattr(synth, "_chunked_synthesis", fake_chunked)
    monkeypatch.setattr(synth, "synth_A", fake_ar)

    row, _ = synth.synth_C({"id": "ta-01", "register": "native_tamil", "text": "x"}, 0, 7,
                           gate_threshold=0.5)
    assert calls == {"chunked": 0, "ar": 1}
    assert row["config"] == "C" and row["fallback"] is True
    assert row["verify_mode"] == "ar_fallback"
    assert row["gate_threshold"] == 0.5
    assert row["accept_true"] is None and row["zone_stats"] == {}

    # high confidence -> normal speculative path, no fallback
    monkeypatch.setattr(synth, "_draft_confidence", lambda utt, seed, gate_stat="utt_min": 0.90)
    row2, _ = synth.synth_C({"id": "ta-01", "register": "native_tamil", "text": "x"}, 0, 7,
                            gate_threshold=0.5)
    assert calls == {"chunked": 1, "ar": 1}
    assert row2["fallback"] is False and row2["gate_threshold"] == 0.5
    assert row2["draft_confidence"] == pytest.approx(0.90)

    # threshold 0 disables the gate entirely (back-compatible behaviour)
    row3, _ = synth.synth_C({"id": "ta-01", "register": "native_tamil", "text": "x"}, 0, 7,
                            gate_threshold=0.0)
    assert row3["fallback"] is False and row3["gate_threshold"] == 0.0


def test_tables_IV_and_V_render():
    rows = _fabricated_rows()
    switch_rows = R.switch_distance_table(rows)
    gate_rows = R.aggregate_gate_sweep(rows)
    table4 = R.build_table_IV(switch_rows)
    table5 = R.build_table_V(gate_rows, R.aggregate_results(rows, ["A", "C"]).get("A"))
    for table in (table4, table5):
        assert "\\begin{tabular}" in table and "\\end{tabular}" in table
        assert "\\bottomrule" in table
    assert "$\\geq$16" in table4
    assert "AR fallback" in table5 or "tau" in table5 or "$\\tau$" in table5
    assert "AR (A)" in table5    # the pure-AR endpoint is listed


def test_figure5_gate_pareto(tmp_path: Path):
    rows = _fabricated_rows()
    agg = R.aggregate_results(rows, ["A", "C"])
    paths = R.figure5_gate_pareto(rows, agg, tmp_path)
    assert paths
    for p in paths:
        assert Path(p).exists() and Path(p).stat().st_size > 1000
    # no gated rows at all -> no figure (nothing to plot)
    assert R.figure5_gate_pareto([r for r in rows if r["config"] == "A"], agg, tmp_path) is None


# ===========================================================================
# Audio-encoder decode robustness
#
# Regression cover for two Kaggle failures on indic-parler-tts:
#   * the teacher's audio encoder is ``transformers.models.dac.modeling_dac
#     .DacModel``, whose ``decode`` takes 3-D ``(batch, K, T)`` codes; the old
#     code only tried the parler-wrapper 4-D layout, so every decode returned
#     ``None`` (capture validation "MISMATCH" + fallback everywhere);
#   * ``DacModel.encode`` runs in fp16 while the teacher waveform is float32,
#     raising "Input type (float) and bias type (c10::Half) should be the same".
# ===========================================================================
class _FakeDecodeOut:
    def __init__(self, audio_values):
        self.audio_values = audio_values


class _TransformersStyleDecoder:
    """3-D ``(batch, K, T)`` codes, ``audio_scales`` keyword rejected."""

    def __init__(self):
        self.shapes: list[tuple] = []

    def decode(self, audio_codes, audio_scales=None):
        import torch

        if audio_scales is not None:
            raise TypeError("unexpected keyword argument 'audio_scales'")
        if audio_codes.dim() != 3:
            raise ValueError(f"expected 3-D audio_codes, got {tuple(audio_codes.shape)}")
        self.shapes.append(tuple(audio_codes.shape))
        t = audio_codes.shape[2]
        return _FakeDecodeOut(torch.full((1, 1, t * 4), 0.25))


class _ParlerWrapperStyleDecoder:
    """4-D ``(chunk, batch, K, T)`` codes, requires ``audio_scales``."""

    def __init__(self):
        self.shapes: list[tuple] = []

    def decode(self, audio_codes, audio_scales):
        import torch

        if not audio_scales:
            raise ValueError("audio_scales required")
        if audio_codes.dim() != 4:
            raise ValueError(f"expected 4-D audio_codes, got {tuple(audio_codes.shape)}")
        self.shapes.append(tuple(audio_codes.shape))
        t = audio_codes.shape[3]
        return _FakeDecodeOut(torch.full((1, 1, t * 4), 0.5))


def test_decode_codes_handles_transformers_dac_layout():
    import torch

    dec = _TransformersStyleDecoder()
    wave = R._decode_codes(dec, torch.zeros(9, 12, dtype=torch.long))
    assert wave is not None and wave.size == 12 * 4
    assert dec.shapes == [(1, 9, 12)]


def test_decode_codes_handles_parler_wrapper_layout():
    import torch

    dec = _ParlerWrapperStyleDecoder()
    wave = R._decode_codes(dec, torch.zeros(9, 7, dtype=torch.long))
    assert wave is not None and wave.size == 7 * 4
    assert dec.shapes == [(1, 1, 9, 7)]


def test_capture_validation_decode_returns_error_on_total_failure():
    import torch

    class _AlwaysFails:
        def decode(self, *args, **kwargs):
            raise RuntimeError("boom")

    import train_draft_distill as T

    wave, err = T._decode_codes_for_validation(_AlwaysFails(), torch.zeros(9, 5, dtype=torch.long))
    assert wave is None
    assert "boom" in err  # error text surfaces in the teacher validation log


def test_zero_lag_corr_detects_identical_and_uncorrelated_signals():
    import train_draft_distill as T

    rng = np.random.RandomState(0)
    a = rng.randn(2048)
    assert T._zero_lag_corr(a, a.copy()) == pytest.approx(1.0, abs=1e-9)
    assert abs(T._zero_lag_corr(a, rng.randn(2048))) < 0.1
    # constant input -> undefined correlation rather than a crash
    assert T._zero_lag_corr(np.ones(64), a[:64]) is None


def test_trim_valid_frames_keeps_interior_clean_columns():
    """Mirrors generate(): columns with any EOS/pad token are dropped, not the
    whole tail after the first bad column."""
    import train_draft_distill as T

    collector = T.TeacherCollector.__new__(T.TeacherCollector)
    collector.max_frames = 100
    codes = np.array([
        [1, 2, 1024, 4, 5],
        [1, 2, 3, 4, 5],
    ], dtype=np.int64)
    trimmed = collector._trim_valid_frames(codes, codebook_size=1024)
    assert trimmed.tolist() == [[1, 2, 4, 5], [1, 2, 4, 5]]


def test_scorer_uses_codebook0_logits_and_codebook_vocab():
    """Regression for two Kaggle failures: the LM wrapper returns
    ``logits`` shaped (K, seq, model_vocab) — the scorer used to read
    ``last_hidden_state`` (AttributeError) and treat ``decoder.config.vocab_size``
    (1088) as the code vocabulary, which then broke Config C's rejection
    sampling with "operands could not be broadcast together ... (1088,) (1024,)"."""
    import torch

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _Decoder:
        num_codebooks = 9

        def __init__(self):
            self.config = type("Cfg", (), {"vocab_size": 1088})()
            self.seen_ids = None

        def __call__(self, input_ids=None, **kwargs):
            self.seen_ids = input_ids
            seq = input_ids.shape[1]
            logits = torch.zeros(self.num_codebooks, seq, 1088)
            for p in range(seq):
                logits[0, p, p % 1024] = 5.0  # position-dependent cb0 preference
            return _Out(logits)

    scorer = R.TeacherForcedScorer.__new__(R.TeacherForcedScorer)
    scorer.engine = type("Eng", (), {"device": torch.device("cpu")})()
    scorer.decoder = _Decoder()
    scorer.num_codebooks = 9
    scorer.vocab = 1024
    scorer.bos_id = 1025
    scorer.prompt_cross_attention = True
    scorer.verbose = False
    ctx = {
        "offset": 0,
        "encoder_hidden_states": torch.zeros(1, 4, 8),
        "encoder_attention_mask": None,
    }
    prefix = np.asarray([11, 12], dtype=np.int64)
    chunk = np.asarray([13, 14, 15], dtype=np.int64)
    probs = scorer._forward_probs(ctx, prefix, chunk)

    assert probs is not None
    assert probs.shape == (3, 1024)  # codebook vocab, not the padded head width
    assert probs.sum(axis=1) == pytest.approx(np.ones(3), abs=1e-5)
    # logits[p] scores frame p - 1; the first scored frame is index len(prefix)
    assert list(probs.argmax(axis=1)) == [2, 3, 4]
    # ids: (K, L) with row 0 = [bos] + prefix + chunk, other rows all bos
    ids = scorer.decoder.seen_ids
    assert tuple(ids.shape) == (9, 6)
    assert int(ids[0, 0]) == 1025
    assert ids[0, 1:].tolist() == [11, 12, 13, 14, 15]
    assert (ids[1:] == 1025).all()


def test_scorer_offset_accounts_for_prepended_prompt():
    import torch

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _Decoder:
        num_codebooks = 9

        def __init__(self):
            self.config = type("Cfg", (), {"vocab_size": 1088})()

        def __call__(self, input_ids=None, prompt_hidden_states=None, **kwargs):
            seq_tokens = input_ids.shape[1]
            prompt_len = 0 if prompt_hidden_states is None else prompt_hidden_states.shape[1]
            # the decoder concatenates prompt embeddings in front of the token
            # embeddings, so logits cover P + L positions
            seq = prompt_len + seq_tokens
            logits = torch.zeros(self.num_codebooks, seq, 1088)
            for p in range(seq):
                logits[0, p, p % 1024] = 5.0
            return _Out(logits)

    scorer = R.TeacherForcedScorer.__new__(R.TeacherForcedScorer)
    scorer.engine = type("Eng", (), {"device": torch.device("cpu")})()
    scorer.decoder = _Decoder()
    scorer.num_codebooks = 9
    scorer.vocab = 1024
    scorer.bos_id = 1025
    scorer.prompt_cross_attention = False  # prompt prepended in the decoder
    scorer.verbose = False
    ctx = {
        "offset": 3,
        "encoder_hidden_states": torch.zeros(1, 4, 8),
        "encoder_attention_mask": None,
        "prompt_hidden_states": torch.zeros(1, 3, 8),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    probs = scorer._forward_probs(ctx, np.asarray([1], dtype=np.int64), np.asarray([2, 3], dtype=np.int64))
    assert probs is not None and probs.shape == (2, 1024)
    # token k sits after the prompt, so frame k is scored by logits[prompt_len + k]
    assert list(probs.argmax(axis=1)) == [4, 5]


def test_codec_reencode_casts_waveform_to_encoder_dtype():
    """Regression: fp16 teacher + float32 waveform raised
    'Input type (float) and bias type (c10::Half) should be the same'."""
    import torch

    import train_draft_distill as T

    class _EncodeOut:
        audio_codes = torch.zeros(1, 9, 5, dtype=torch.long)

    class _Encoder:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))
            self.seen_shape: tuple | None = None

        def parameters(self):
            return iter([self.weight])

        def encode(self, wave):
            if wave.dtype != torch.float16:
                raise RuntimeError(
                    "Input type (float) and bias type (c10::Half) should be the same"
                )
            self.seen_shape = tuple(wave.shape)
            return _EncodeOut()

    enc = _Encoder()
    collector = T.TeacherCollector.__new__(T.TeacherCollector)
    collector.engine = type("Eng", (), {"device": torch.device("cpu")})()
    collector.engine.ar_model = type("Ar", (), {"audio_encoder": enc})()
    collector.verbose = True

    codes = collector._codec_reencode(np.zeros(4096, dtype=np.float32))
    assert codes is not None and codes.shape == (9, 5)
    assert enc.seen_shape == (1, 1, 4096)

