"""Tests for the corpus builder and the shared Tamil->Latin transliterator.

Covers the pieces that decide the *training text distribution* and the
script-robust CER metric:
  * transliteration rules (consonants, vowel signs, virama, mixed tokens)
  * Latin-normalised CER scoring
  * entity injection / augmentation registers
  * register-balanced corpus assembly + eval-set disjointness
  * the writer's file format
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

import build_codeswitch_corpus as B
import codeswitch_benchmark_data as data
import run_full_evaluation as R
from codeswitch_tts.frontend.tamil_latin import latinize_text, transliterate_word


# ===========================================================================
# Transliteration
# ===========================================================================
def test_transliterate_inherent_vowel_and_signs():
    assert transliterate_word("கடை") == "katai"
    assert transliterate_word("வணக்கம்") == "vanakkam"
    assert transliterate_word("நன்றி") == "nanri"
    # virama kills the inherent vowel
    assert transliterate_word("சொல்") == "sol"


def test_transliterate_keeps_latin_digits_and_punctuation():
    assert transliterate_word("order") == "order"
    assert transliterate_word("OTP1234") == "OTP1234"
    assert transliterate_word("8402.") == "8402."
    assert transliterate_word("டாக்சி-ல") == "taaksi-la"


def test_latinize_text_mixed_script():
    text = "உங்க order நாளைக்கு வரும்."
    out = latinize_text(text)
    assert "order" in out
    assert "ungka" in out or "ungal" in out
    assert "ந" not in out
    # pure-Latin input is returned untouched (fast path)
    assert latinize_text("order ready") == "order ready"


def test_score_latin_removes_script_mismatch_penalty():
    ref = "ஆர்டர் தயார்"
    # a "perfect" transcription written in Latin script
    latin_hyp = latinize_text(ref)
    scripted_cer, _ = R.ASREvaluator.score(ref, latin_hyp)
    latin_cer, _ = R.ASREvaluator.score_latin(ref, latin_hyp)
    assert scripted_cer is not None and latin_cer is not None
    assert latin_cer < scripted_cer
    assert latin_cer == pytest.approx(0.0, abs=1e-9)
    # identical Tamil-script hypothesis is 0 either way
    cer_same, _ = R.ASREvaluator.score(ref, ref)
    assert cer_same == pytest.approx(0.0, abs=1e-9)
    assert R.ASREvaluator.score_latin(ref, None) == (None, None)


# ===========================================================================
# Corpus augmentation
# ===========================================================================
def test_fill_entity_slots_are_deterministic():
    rng = random.Random(3)
    filled = B.fill_entity("Order ID {otp4} ready, fare Rs. {fare}.", rng)
    assert "{otp4}" not in filled and "{fare}" not in filled
    otp = [w for w in filled.split() if w.isdigit() and len(w) == 4]
    assert otp, filled
    assert 1000 <= int(otp[0]) <= 9999


def test_inject_entity_preserves_tamil_sentence():
    rng = random.Random(5)
    base = "நாளை காலையில் வருகிறேன்."
    for _ in range(20):
        out = B.inject_entity(base, "OTP", rng)
        assert "OTP" in out
        # the base sentence survives in at least one of the two clauses
        assert any(tok in out for tok in ("நாளை", "காலையில்", "வருகிறேன்"))


def test_augment_assigns_registers():
    rng = random.Random(11)
    regs = set()
    for _ in range(200):
        text, reg = B.augment("நாளை காலையில் வருகிறேன்.", rng, 0.5, 0.4, 0.15)
        regs.add(reg)
        assert B.classify_register(text) == reg
    assert regs == {"native_tamil", "romanized_tanglish", "transit_codemixed"}


def test_classify_register_rules():
    assert B.classify_register("நாளை வருகிறேன்.") == "native_tamil"
    assert B.classify_register("naalai varen.") == "romanized_tanglish"
    assert B.classify_register("நாளை OTP வரும்.") == "transit_codemixed"


# ===========================================================================
# Corpus assembly
# ===========================================================================
def test_corpus_is_register_balanced_and_disjoint(tmp_path: Path):
    corpus, counts = B.build_corpus(
        n_templates=700, n_wikipedia=0, code_mix_rate=0.35, romanize_rate=0.30,
        numeric_rate=0.12, seed=13, passes=3,
    )
    assert len(corpus) == sum(counts.values()) > 100
    # every register is represented and no quota exceeds 50% of the corpus
    for reg in B.REGISTERS:
        assert counts.get(reg, 0) > 0
        assert counts[reg] / len(corpus) <= 0.5
    assert len(set(corpus)) == len(corpus)
    # disjoint from the benchmark (the assert raises on leakage)
    data.assert_corpus_disjoint_from_eval(corpus)
    # the corpus file is plain text, one sentence per line, UTF-8
    out = tmp_path / "corpus.txt"
    out.write_text("\n".join(corpus) + "\n", encoding="utf-8")
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(corpus)
    # and it round-trips through the loader used by train_draft_distill
    assert data.load_corpus_json(out) == corpus


def test_corpus_extra_file_accepts_txt_and_json(tmp_path: Path):
    txt = tmp_path / "extra.txt"
    txt.write_text("# comment\nநாளை காலையில் வருகிறேன்.\nஇன்று மழை பெய்யும்.\n", encoding="utf-8")
    js = tmp_path / "extra.json"
    js.write_text(json.dumps([{"text": "எங்கள் கடை திறந்திருக்கும்."}], ensure_ascii=False),
                  encoding="utf-8")
    loaded = B.load_extra_files([str(txt), str(js)])
    assert len(loaded) == 3
    assert B.load_extra_files([str(tmp_path / "missing.txt")]) == []


def test_corpus_clean_sentence_filters():
    assert B.clean_sentence("இவர் ஒரு இந்திய அரசியல்வாதி") == "இவர் ஒரு இந்திய அரசியல்வாதி."
    assert B.clean_sentence("短") is None                      # no Tamil at all
    assert B.clean_sentence("https://example.com இங்கே பாருங்கள்") is None
    assert B.clean_sentence("(இது) ஒரு சோதனை வாக்கியம் ஆகும்") is None
    assert B.clean_sentence("12345678 என்ற எண்") is None        # long digit run
    assert B.clean_sentence("one two three four five") is None  # Latin-dominant
    assert B.clean_sentence("மிகச் சிறிய") is None               # too few words


def test_build_corpus_arg_parser_defaults():
    args = B.build_arg_parser().parse_args([])
    assert args.out == "codeswitch_corpus.txt"
    assert args.templates == 2000 and args.wikipedia == 0
    assert args.passes == 3
    assert args.balance == "0.34,0.33,0.33"


# ===========================================================================
# Perceptual / spectral metrics
# ===========================================================================
def test_mcd_between_identical_and_different_signals():
    sr = 16000
    rng = np.random.RandomState(0)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.4 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = (0.2 * rng.randn(sr)).astype(np.float32)
    same = R.mcd_between(tone, tone, sample_rate=sr)
    assert same is not None and same < 1.0
    diff = R.mcd_between(tone, noise, sample_rate=sr)
    assert diff is not None and diff > same


def test_attach_mcd_from_audio_reads_wavs(tmp_path: Path):
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    a = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    b = (0.3 * np.sin(2 * np.pi * 260 * t)).astype(np.float32)
    for name, wav in (("a.wav", a), ("b.wav", b), ("c.wav", a.copy())):
        sf.write(tmp_path / name, wav, sr)
    rows = [
        {"config": "A", "utt_id": "ta-01", "audio_path": str(tmp_path / "a.wav")},
        {"config": "B", "utt_id": "ta-01", "audio_path": str(tmp_path / "b.wav")},
        {"config": "C", "utt_id": "ta-01", "audio_path": str(tmp_path / "c.wav")},
        {"config": "B", "utt_id": "ta-02", "audio_path": str(tmp_path / "missing.wav")},
    ]
    computed = R.attach_mcd_from_audio(rows, verbose=False)
    assert computed == 2
    assert rows[0]["mcd_vs_a"] == pytest.approx(0.0)
    assert 0.0 <= rows[2]["mcd_vs_a"] < 1.0     # identical signal
    assert rows[1]["mcd_vs_a"] > rows[2]["mcd_vs_a"]
    assert "mcd_vs_a" not in rows[3]            # missing wav is skipped, not zero


def test_utmos_scorer_disabled_path():
    scorer = R.UTMOSScorer("cpu", enabled=False)
    assert scorer.available is False
    assert scorer.score(np.zeros(1600, dtype=np.float32), 16000) is None
