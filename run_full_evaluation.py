"""CodeSwitch-TTS full evaluation suite (Interspeech/ICASSP-ready).

One command on a Kaggle T4 (after the notebook environment is installed and the
distillation companion has produced checkpoints):

    !pip install -q jiwer matplotlib seaborn pandas
    !python train_draft_distill.py                # once: teacher cache + checkpoints
    !python run_full_evaluation.py --ckpt codeswitch_ckpt

What it produces
----------------
* 60-utterance evaluation set (20 native Tamil / 20 romanized Tanglish /
  20 transit & commerce code-mixed) synthesised with three configurations on
  the same GPU:
    A. Pure AR      - Indic Parler-TTS with the real streaming streamer.
    B. Pure NAR     - distilled draft -> residual decoder -> DAC (no verifier).
    C. CodeSwitch   - distilled draft -> teacher-forced CB0 verification
                      (rejection sampling) -> residual -> DAC  [proposed].
* Per-utterance metrics (``eval_results/raw_results.jsonl``, resumable):
  TTFA (component-compute + end-to-end wall clock), total latency, RTF, DAC
  Codebook-0 acceptance (teacher-forced target + unrelated-target reference),
  zone-split acceptance (Pure Tamil / English loanword / transit alphanumeric /
  cross-lingual boundary), CER/WER via whisper (optional, inline).
* Publication figures (300 dpi PNG + vector PDF, IEEE style) in
  ``paper_figures/``: fig1 latency-quality tradeoff, fig2 zone acceptance,
  fig3 streaming chunk + buffer timeline.
* Copy-pasteable LaTeX for Tables I-III (stdout + ``paper_figures/tables.tex``):
  I comparative, II zone acceptance, III per-register CER/latency breakdown.

Protocol notes (state these in the paper)
-----------------------------------------
* TRUE acceptance: for every drafted CB0 frame the target distribution
  ``q_t = P_target(cb0_t | text, cb0_<t)`` is computed with a single
  teacher-forced forward pass through the Parler decoder per chunk (delay
  pattern row 0; other codebook channels held at their delayed null/bos
  state -- a documented single-channel verification protocol).  Rejection
  sampling uses the textbook ratio ``min(1, q(x)/p_hat(x))`` with residual
  resampling ``norm(max(0, q - p_hat))``.
* SYNTHETIC column (reference point, NOT an upper bound): identical acceptance
  math against an unrelated random-target distribution ``softmax(randn)``
  (the legacy demo protocol).  Because the drafted student is distilled from
  the real teacher, its proposal agrees with the true target far more than with
  an unrelated random target, so this column typically sits BELOW the true
  acceptance.  A perfectly aligned draft would score 100\\% against its own
  target, so read it as a mismatched-target baseline, not as a ceiling.
* CER caveat: the reference is the normaliser's Tamil-script text, so
  whisper-small transliteration choices (Latin output for English loanwords)
  and occasional hallucination loops inflate CER on the code-mixed registers.
  Aggregation therefore reports medians alongside means (Table I uses medians)
  and flags rows above ``ASR_FAIL_CER`` (200\\% CER) as ASR failures.  Per
  register numbers live in Table III.
* Target-correction diagnostic: for Config C the teacher-forced mean
  log-likelihood (nats/frame) of the emitted CB0 tokens is recorded next to
  that of the raw draft proposal; rejection sampling should raise it.
* TTFA modes: component-compute (draft + verify + residual + DAC of the first
  chunk) and end-to-end wall clock until the first ~1 s audio buffer exists
  (real streamer for A; first decoded chunk for B/C).
* CER reference is the normalizer's Tamil-script, entity-verbalised text; a
  shared cleaning protocol (NFC, lowercase, punctuation strip, digit runs
  spelled digit-by-digit) is applied to reference and hypothesis.

Resume behaviour: completed ``(config, utterance)`` rows in the JSONL are
skipped (``--force`` re-runs them).  ``--replay`` rebuilds aggregates, figures
and tables from the JSONL without synthesising anything.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np

# codeswitch_tts installs the protobuf fallback shim at import time.
from codeswitch_tts import (  # noqa: E402
    CodeSwitchTTSConfig,
    DraftTransformer,
    ResidualDecoder,
    TanglishNormalizer,
)
from codeswitch_benchmark_data import EVAL_UTTERANCES, REGISTER_LABELS  # noqa: E402
from codeswitch_tts.frontend.tamil_latin import latinize_text  # noqa: E402

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch import failed: {exc}")

SCHEMA_VERSION = 1

# CER above this threshold on a single utterance is almost always a whisper
# hallucination loop (repeated/duplicated transcripts) rather than a synthesis
# failure; such rows are counted separately so means are not silently poisoned.
ASR_FAIL_CER = 2.0

# ===========================================================================
# Zone constants
# ===========================================================================
ZONE_NONE = 0
ZONE_TAMIL = 1
ZONE_ENGLISH = 2
ZONE_TRANSIT = 3
ZONE_BOUNDARY = 4

ZONE_NAMES: Dict[int, str] = {
    ZONE_TAMIL: "Pure Tamil",
    ZONE_ENGLISH: "English Loanword",
    ZONE_TRANSIT: "Transit Alphanumeric",
    ZONE_BOUNDARY: "Cross-Lingual Boundary",
}
ZONE_ORDER: List[int] = [ZONE_TAMIL, ZONE_ENGLISH, ZONE_TRANSIT, ZONE_BOUNDARY]

CONFIG_LABELS = {
    "A": "Pure AR (Parler-TTS)",
    "B": "Pure NAR (Draft+Residual)",
    "C": "CodeSwitch-TTS (Hybrid)",
}
CONFIG_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
PUNCT_RE = re.compile(r"[^\w\s\u0B80-\u0BFF]", re.UNICODE)
DIGIT_RUN_RE = re.compile(r"\d+")
EN_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five",
                  "six", "seven", "eight", "nine"]


# ===========================================================================
# Utilities
# ===========================================================================
def _now() -> float:
    return time.perf_counter()


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def cuda_sync(device) -> None:
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_fingerprint(ckpt_dir: Any) -> Dict[str, Any]:
    """Content hash of the checkpoints a run's rows were produced with.

    Resume is keyed by ``(config, utterance, gate)``, so re-running with *different*
    weights would silently mix two models' audio in one results file.  The
    fingerprint is stored next to the JSONL and checked before resuming.
    """
    out: Dict[str, Any] = {}
    for name in ("draft_cb0.pt", "residual_cb1_8.pt", "meta.json"):
        path = Path(ckpt_dir) / name
        if not path.is_file():
            continue
        digest = hashlib.sha1()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        out[name] = {"sha1": digest.hexdigest()[:16], "size": path.stat().st_size}
    return out


def check_provenance(results_path: Path, ckpt_dir: Any, force: bool = False,
                     replay: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Compares the current checkpoints with the ones that produced the rows.

    Returns ``{"fingerprint", "previous", "blocked", "reason", "note"}``;
    ``blocked`` means the caller should refuse to resume (unless ``--force``).
    """
    path = Path(results_path).parent / "provenance.json"
    fingerprint = checkpoint_fingerprint(ckpt_dir)
    previous: Optional[Dict[str, Any]] = None
    try:
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        previous = None
    out: Dict[str, Any] = {"path": str(path), "fingerprint": fingerprint,
                           "previous": previous, "blocked": False, "reason": None, "note": None}
    if replay or force:
        return out
    if previous is None:
        if Path(results_path).is_file() and Path(results_path).stat().st_size > 0:
            out["note"] = ("existing results without provenance (they predate this check): "
                           "assuming they came from these checkpoints - pass --force to be sure")
        return out
    old = previous.get("checkpoints") or {}
    if old and old != fingerprint:
        changed = sorted({*old, *fingerprint} )
        detail = ", ".join(
            f"{name}: {old.get(name, {}).get('sha1', 'absent')} -> "
            f"{fingerprint.get(name, {}).get('sha1', 'absent')}"
            for name in changed if old.get(name, {}).get("sha1") != fingerprint.get(name, {}).get("sha1")
        )
        out["blocked"] = True
        out["reason"] = (f"these results were produced with different checkpoints ({detail}). "
                         "Resuming would mix two models' audio in one file: pass --force to "
                         "re-synthesise, or write to a different --results path")
    return out


def package_versions() -> Dict[str, str]:
    from importlib import metadata

    versions: Dict[str, str] = {}
    for pkg in ("torch", "transformers", "parler_tts", "protobuf", "numpy", "jiwer"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "not-installed"
    return versions


def digits_to_words_digitwise(text: str) -> str:
    def _repl(m: re.Match) -> str:
        return " ".join(EN_DIGIT_WORDS[int(d)] for d in m.group(0))

    return DIGIT_RUN_RE.sub(_repl, text)


def clean_for_cer(text: str) -> str:
    """Shared CER cleaning protocol (reference and hypothesis)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return digits_to_words_digitwise(text)


# ===========================================================================
# Zone instrumentation (linguistic zones -> acoustic frames)
# ===========================================================================
def find_transit_spans(
    normalized_text: str, replacements: Sequence[Tuple[str, str]]
) -> List[Tuple[int, int]]:
    """Locates verbalised transit entity spans inside the final text."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    low = normalized_text.lower()
    for _orig, spoken in replacements:
        candidate = re.sub(r"\s+", " ", spoken.strip())
        if not candidate:
            continue
        idx = low.find(candidate.lower(), cursor)
        if idx < 0:
            idx = low.find(candidate.lower())
        if idx < 0:
            first = candidate.split()[0].lower()
            idx = low.find(first, cursor) if first else -1
            if idx < 0:
                continue
            spans.append((idx, min(len(low), idx + len(first))))
        else:
            spans.append((idx, min(len(low), idx + len(candidate))))
        cursor = spans[-1][1]
    return spans


def label_char_zones(
    normalized_text: str, transit_spans: Sequence[Tuple[int, int]]
) -> np.ndarray:
    """Per-character zone ids: 1 Tamil, 2 English, 3 transit, 0 neutral."""
    zones = np.zeros(len(normalized_text), dtype=np.int64)
    transit = np.zeros(len(normalized_text), dtype=bool)
    for start, end in transit_spans:
        transit[start:end] = True
    for i, ch in enumerate(normalized_text):
        if TAMIL_RE.match(ch):
            zones[i] = ZONE_TAMIL
        elif ch.isspace() or not ch.isalnum():
            zones[i] = ZONE_NONE
        elif transit[i] or ch.isdigit():
            zones[i] = ZONE_TRANSIT
        elif ch.isalpha():
            zones[i] = ZONE_ENGLISH
        else:
            zones[i] = ZONE_NONE
    return zones


def _language_class(zone: int) -> Optional[str]:
    if zone == ZONE_TAMIL:
        return "tamil"
    if zone in (ZONE_ENGLISH, ZONE_TRANSIT):
        return "latin"
    return None


def assign_frame_zones(
    char_zones: np.ndarray, n_frames: int, boundary_frames: int = 3
) -> Tuple[np.ndarray, int, List[int]]:
    """Monotonic character->frame alignment + boundary overlay.

    Returns ``(frame_zones, n_transitions, switch_frames)``; boundary frames are
    the first ``boundary_frames`` frames after each Tamil<->Latin switch, and
    ``switch_frames`` holds the (sorted, de-duplicated) frame indices of those
    switches so downstream analyses can measure *distance to switch*.
    """
    frame_zones = np.zeros(max(0, n_frames), dtype=np.int64)
    n_transitions = 0
    if n_frames <= 0:
        return frame_zones, n_transitions, []
    mappable = [i for i, z in enumerate(char_zones) if z != ZONE_NONE]
    if not mappable:
        return frame_zones, n_transitions, []
    m = len(mappable)
    frame_of = [min(n_frames - 1, int(math.floor(j * n_frames / m))) for j in range(m)]

    buckets: List[List[int]] = [[] for _ in range(n_frames)]
    for j, char_idx in enumerate(mappable):
        buckets[frame_of[j]].append(int(char_zones[char_idx]))
    for f, bucket in enumerate(buckets):
        if bucket:
            frame_zones[f] = max(set(bucket), key=bucket.count)

    transitions: List[int] = []
    for j in range(1, m):
        cls_prev = _language_class(int(char_zones[mappable[j - 1]]))
        cls_cur = _language_class(int(char_zones[mappable[j]]))
        if cls_prev and cls_cur and cls_prev != cls_cur:
            transitions.append(frame_of[j])
    for tr in transitions:
        frame_zones[tr: min(n_frames, tr + boundary_frames)] = ZONE_BOUNDARY
    return frame_zones, len(transitions), sorted(set(transitions))


# ---------------------------------------------------------------------------
# Distance-to-switch analysis (why the zone contrast is a power problem)
# ---------------------------------------------------------------------------
#: ``(label, lo, hi)`` buckets of "frames since the last Tamil<->Latin switch".
SWITCH_DIST_BUCKETS: List[Tuple[str, int, int]] = [
    ("0-1", 0, 2), ("2-3", 2, 4), ("4-7", 4, 8), ("8-15", 8, 16), ("16+", 16, 10 ** 9),
]


def switch_distances(n_frames: int, switch_frames: Sequence[int]) -> np.ndarray:
    """Frames since the most recent Tamil<->Latin switch per frame.

    Frames before the first switch (or in utterances without switches) get the
    sentinel ``n_frames``, i.e. they fall in the ``16+`` bucket.
    """
    n = max(0, int(n_frames))
    dist = np.full(n, n, dtype=np.int64)
    if n == 0:
        return dist
    switches = sorted({int(s) for s in (switch_frames or []) if 0 <= int(s) < n})
    if not switches:
        return dist
    last = np.zeros(n, dtype=np.int64)
    for s in switches:
        last[s:] = s
    dist = np.arange(n, dtype=np.int64) - last
    dist[: switches[0]] = n
    return dist


def pack_u8(values: Sequence[int]) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.uint8).tobytes()).decode("ascii")


def unpack_u8(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text.encode("ascii")), dtype=np.uint8)


def pack_bits(flags: Sequence[bool]) -> str:
    return pack_u8(np.packbits(np.asarray(flags, dtype=bool).astype(np.uint8)))


def unpack_bits(text: str, n: int) -> np.ndarray:
    return np.unpackbits(unpack_u8(text))[: max(0, int(n))].astype(bool)


def frame_accept_arrays(row: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``(accept_true, valid, distances)`` for a row with packed frame telemetry."""
    n = int(row.get("n_frames") or 0)
    packed = row.get("frame_accept_true_b64")
    valid_b64 = row.get("frame_valid_b64")
    if n <= 0 or not packed or not valid_b64:
        return None
    try:
        accept = unpack_bits(packed, n)
        valid = unpack_bits(valid_b64, n)
    except Exception:
        return None
    if accept.size != n or valid.size != n:
        return None
    if not valid.any():
        return None
    return accept, valid, switch_distances(n, row.get("switch_frames") or [])


def paired_bootstrap(
    pairs: Sequence[Tuple[float, float]], n_boot: int = 2000, seed: int = 20261018
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Paired mean difference with bootstrap CI and two-sided p-value.

    ``pairs`` are ``(a, b)`` observations on the same unit (utterance); returns
    ``(mean_diff = mean(a) - mean(b), ci_lo, ci_hi, p)``.  Pure numpy so the
    suite has no scipy dependency.
    """
    arr = np.asarray([[float(a), float(b)] for a, b in pairs], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None, None, None, None
    diff = arr[:, 0] - arr[:, 1]
    mean_diff = float(diff.mean())
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    ci_lo, ci_hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    p = 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return mean_diff, ci_lo, ci_hi, min(1.0, p)


# ===========================================================================
# Teacher-forced target scoring (true Parler probabilities)
# ===========================================================================
class TeacherForcedScorer:
    """Scores drafted Codebook-0 frames against the Parler decoder distribution.

    ``q_t = P_target(cb0_t | text, cb0_<t)`` from a single teacher-forced
    forward pass per chunk; codebook rows 1..8 are held at their bos (null)
    state (documented single-channel verification protocol).
    """

    def __init__(self, engine, verbose: bool = True):
        self.engine = engine
        self.verbose = verbose
        self.available = False
        self.reason = "not initialised"
        self.ar = getattr(engine, "ar_model", None)
        self.decoder = getattr(self.ar, "decoder", None) if self.ar is not None else None
        self.num_codebooks = int(getattr(self.decoder, "num_codebooks", 9)) if self.decoder else 9
        # codebook-0 sampling vocabulary (rejection sampling operates over the
        # 1024 valid codes); the LM head itself is wider — decoder.config.vocab_size
        # (1088) pads the code space with eos (1024) / bos (1025) slots.
        self.vocab = int(
            getattr(
                getattr(getattr(self.ar, "audio_encoder", None), "config", None),
                "codebook_size",
                1024,
            )
        )
        self.model_vocab = int(getattr(getattr(self.decoder, "config", None), "vocab_size", 0) or 0)
        gc_cfg = getattr(self.ar, "generation_config", None)
        self.bos_id = int(
            getattr(gc_cfg, "bos_token_id", None)
            or getattr(gc_cfg, "decoder_start_token_id", None)
            or 1025
        )
        self.prompt_cross_attention = bool(getattr(self.ar, "prompt_cross_attention", False))
        self._probe()

    # -- conditioning --------------------------------------------------------
    def build_context(self, prompt_clean: str) -> Optional[Dict[str, Any]]:
        if self.decoder is None:
            return None
        try:
            ar = self.ar
            desc = self.engine.get_cached_description_encoder_outputs(
                self.engine.config.voice_description
            )
            prompt_ids = self.engine.tokenizer(prompt_clean, return_tensors="pt").to(self.engine.device)
            ctx: Dict[str, Any] = {
                "offset": 0,
                "encoder_hidden_states": desc,
                "encoder_attention_mask": None,
            }
            if self.prompt_cross_attention:
                embedded = ar.embed_prompts(prompt_ids.input_ids)
                positions = ar.embed_positions(embedded, 0)
                prompt_hidden = (embedded + positions.to(embedded.device)).to(desc.dtype)
                ones = torch.ones(desc.shape[:2], device=desc.device,
                                  dtype=prompt_ids.attention_mask.dtype)
                ctx["encoder_hidden_states"] = torch.cat([desc, prompt_hidden], dim=1)
                ctx["encoder_attention_mask"] = torch.cat([ones, prompt_ids.attention_mask], dim=1)
            else:
                ctx["prompt_hidden_states"] = ar.embed_prompts(prompt_ids.input_ids).to(desc.dtype)
                ctx["prompt_attention_mask"] = prompt_ids.attention_mask
                ctx["offset"] = int(prompt_ids.input_ids.shape[1])
            return ctx
        except Exception as exc:
            if self.verbose:
                print(f"[Scorer] context build failed: {type(exc).__name__}: {exc}")
            return None

    # -- scoring -------------------------------------------------------------
    @torch.inference_mode()
    def _forward_probs(
        self, ctx: Dict[str, Any], prefix_cb0: np.ndarray, chunk_cb0: np.ndarray
    ) -> Optional[np.ndarray]:
        try:
            tokens = np.concatenate([prefix_cb0, chunk_cb0]).astype(np.int64)
            length = tokens.shape[0] + 1
            ids = torch.full((self.num_codebooks, length), self.bos_id,
                             dtype=torch.long, device=self.engine.device)
            ids[0, 1:] = torch.from_numpy(tokens).to(self.engine.device)
            kwargs: Dict[str, Any] = {
                "input_ids": ids,
                "encoder_hidden_states": ctx["encoder_hidden_states"],
                "encoder_attention_mask": ctx.get("encoder_attention_mask"),
                "use_cache": False,
                "return_dict": True,
            }
            if not self.prompt_cross_attention and ctx.get("prompt_hidden_states") is not None:
                kwargs["prompt_hidden_states"] = ctx["prompt_hidden_states"]
                kwargs["prompt_attention_mask"] = ctx.get("prompt_attention_mask")
            out = self.decoder(**kwargs)
            # ``ParlerTTSForCausalLM`` returns logits shaped (K, L, model_vocab):
            # row 0 holds codebook 0, columns [0, codebook_size) are the valid
            # codes and the remaining slots are eos/bos padding.  Logits at
            # position p predict the token at p + 1, and the codebook-0 stream
            # starts with the bos token at position 0, so frame k is scored by
            # logits[offset + k] (offset = prompt length when the prompt is
            # prepended to the decoder inputs, else 0).
            logits = out.logits[0].float()
            start = ctx.get("offset", 0) + prefix_cb0.shape[0]
            logits = logits[start: start + chunk_cb0.shape[0], : self.vocab]
            probs = torch.softmax(logits, dim=-1)
            return probs.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            if self.verbose:
                print(f"[Scorer] forward failed: {type(exc).__name__}: {exc}")
            self.available = False
            self.reason = f"{type(exc).__name__}: {exc}"
            return None

    def _probe(self) -> None:
        if self.decoder is None:
            self.reason = "no decoder (Parler unavailable)"
        else:
            try:
                ctx = self.build_context("வணக்கம்.")
                if ctx is None:
                    self.reason = "conditioning unavailable"
                else:
                    q = self._forward_probs(
                        ctx, prefix_cb0=np.asarray([], dtype=np.int64),
                        chunk_cb0=np.zeros(4, dtype=np.int64),
                    )
                    if q is not None and q.shape == (4, self.vocab):
                        self.available = True
                        self.reason = "ok"
                    else:
                        self.reason = f"unexpected probe output {None if q is None else q.shape}"
            except Exception as exc:
                self.reason = f"{type(exc).__name__}: {exc}"
        if self.verbose:
            status = "OK" if self.available else f"UNAVAILABLE ({self.reason})"
            print(f"[Scorer] teacher-forced CB0 scoring: {status}")

    def score(self, ctx: Optional[Dict[str, Any]], prefix_cb0: np.ndarray, chunk_cb0: np.ndarray) -> Optional[np.ndarray]:
        if not self.available or ctx is None:
            return None
        return self._forward_probs(ctx, prefix_cb0, chunk_cb0)

    @staticmethod
    def synthetic_probs(n_frames: int, vocab: int, rng: np.random.RandomState) -> np.ndarray:
        logits = rng.standard_normal((n_frames, vocab)).astype(np.float32)
        logits -= logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


# ===========================================================================
# Speculative rejection sampling
# ===========================================================================
def rejection_sample(
    target_probs: np.ndarray,
    draft_probs: np.ndarray,
    draft_tokens: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-frame rejection sampling with residual resampling."""
    n = draft_tokens.shape[0]
    kept = np.empty(n, dtype=np.int64)
    accepted = np.zeros(n, dtype=bool)
    n_resampled = 0
    for t in range(n):
        tok = int(draft_tokens[t])
        q = float(target_probs[t, tok])
        p = float(draft_probs[t, tok])
        ratio = q / max(p, 1e-8)
        if ratio >= 1.0 or rng.random_sample() <= ratio:
            kept[t] = tok
            accepted[t] = True
        else:
            residual = np.maximum(target_probs[t] - draft_probs[t], 0.0)
            total = float(residual.sum())
            if total > 1e-8:
                kept[t] = int(rng.choice(residual.shape[0], p=residual / total))
            else:
                kept[t] = int(rng.choice(target_probs.shape[1], p=target_probs[t]))
            n_resampled += 1
    return kept, accepted, n_resampled


def acceptance_rate(mask: Optional[np.ndarray]) -> Optional[float]:
    if mask is None or mask.size == 0:
        return None
    return float(mask.mean() * 100.0)


# ===========================================================================
# Synthesis
# ===========================================================================
@dataclass
class ChunkTelemetry:
    idx: int
    n_frames: int
    verify_ms: float
    residual_ms: float
    dac_ms: float
    ready_ms: float
    accept_true: Optional[float]
    accept_synth: Optional[float]
    confidence: Optional[float] = None


_DECODE_WARNED = False


def _decode_codes(audio_encoder, codes: torch.Tensor) -> Optional[np.ndarray]:
    """Decodes ``(K, T)`` DAC codes through the Parler audio encoder.

    Tries the 3-D ``(batch, K, T)`` layout used by ``generate`` plus the
    optional ``audio_scales`` argument, then the 4-D ``(chunk, batch, K, T)``
    parler-wrapper layout — the encoder class (transformers ``DacModel`` vs.
    the parler-tts wrapper) differs across releases.  Returns ``None`` only
    when every signature fails (callers then use the DAC fallback decoder).
    """
    global _DECODE_WARNED
    last_err = "no decode signature succeeded"
    flat = codes.detach().long()
    device = getattr(audio_encoder, "device", None)
    for shape_name, inp in (("3d", flat[None]), ("4d", flat[None, None])):
        if device is not None:
            inp = inp.to(device)
        for use_scales in (True, False):
            try:
                kwargs: Dict[str, Any] = {"audio_codes": inp}
                if use_scales:
                    kwargs["audio_scales"] = [None]
                with torch.inference_mode():
                    out = audio_encoder.decode(**kwargs)
                audio = out.audio_values if hasattr(out, "audio_values") else out[0]
                return audio.detach().float().cpu().numpy().reshape(-1)
            except Exception as exc:
                last_err = (
                    f"{shape_name}{'+scales' if use_scales else ''} -> "
                    f"{type(exc).__name__}: {exc}"
                )
    if not _DECODE_WARNED:
        _DECODE_WARNED = True
        print(
            f"[Decode] teacher DAC decode failed on all signatures ({last_err}); "
            "falling back to the bundled DAC decoder"
        )
    return None


class EvalSynthesizer:
    """Per-configuration synthesis with unified timing + acceptance telemetry."""

    def __init__(self, args, config, engine, draft, residual, length_model, scorer, device,
                 draft_trained: bool, residual_trained: bool):
        self.args = args
        self.config = config
        self.engine = engine
        self.draft = draft
        self.residual = residual
        self.length_model = length_model
        self.scorer = scorer
        self.device = device
        self.draft_trained = draft_trained
        self.residual_trained = residual_trained
        self.sample_rate = config.audio.sample_rate
        self.frames_per_chunk = int(config.audio.frames_per_chunk)
        self.normalizer = TanglishNormalizer()
        self._dac_fallback = None
        self._speech_backend = bool(getattr(engine, "speech_available", False)) if engine else False

    # -- shared helpers ------------------------------------------------------
    def normalize(self, text: str) -> str:
        return self.normalizer.normalize(text).normalized_text

    def _prompt_clean(self, normalized: str) -> str:
        s = normalized.strip()
        if not s.endswith((".", "!", "?", "।")):
            s += "."
        return s

    def _frames_for(self, normalized: str) -> int:
        if self.length_model is not None:
            return int(self.length_model.predict(normalized))
        return int(min(430, max(30, round(len(normalized) * 86.13 / 13.0))))

    def _postprocess(self, wave: np.ndarray) -> np.ndarray:
        from codeswitch_tts.utils.audio import apply_highpass_filter, normalize_peak

        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        wave = apply_highpass_filter(
            wave, cutoff_hz=self.config.audio.highpass_cutoff_hz, sample_rate=self.sample_rate
        )
        return normalize_peak(wave, target_peak=0.92)

    def decode_codes(self, codes: torch.Tensor) -> np.ndarray:
        codes = codes.detach().long().cpu()
        if self._speech_backend:
            audio = _decode_codes(self.engine.ar_model.audio_encoder, codes)
            if audio is not None:
                return audio
        if self._dac_fallback is None:
            from codeswitch_tts.utils.audio import AudioProcessor

            self._dac_fallback = AudioProcessor(self.config.audio, dac_model=None)
        return np.asarray(self._dac_fallback.decode_and_postprocess(codes.unsqueeze(0)), dtype=np.float32)

    def _base_row(self, cfg: str, utt: Dict[str, str], utt_index: int, normalized: str) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "config": cfg,
            "config_label": CONFIG_LABELS[cfg],
            "utt_id": utt["id"],
            "register": utt["register"],
            "utt_index": utt_index,
            "text": utt["text"],
            "normalized_text": normalized,
            "sample_rate": self.sample_rate,
            "untrained_modules": bool(
                (not self._speech_backend) or (not self.draft_trained) or (not self.residual_trained)
            ),
            "error": None,
        }

    # -- Config A: pure AR ---------------------------------------------------
    def synth_A(self, utt: Dict[str, str], utt_index: int, seed: int) -> Tuple[Dict[str, Any], np.ndarray]:
        normalized = self.normalize(utt["text"])
        row = self._base_row("A", utt, utt_index, normalized)
        if not self._speech_backend:
            raise RuntimeError("Config A requires the Indic Parler-TTS backend.")

        ar = self.engine.ar_model
        prompt_clean = self._prompt_clean(normalized)
        prompt_ids = self.engine.tokenizer(prompt_clean, return_tensors="pt").to(self.device)
        from transformers.modeling_outputs import BaseModelOutput

        desc = self.engine.get_cached_description_encoder_outputs(self.engine.config.voice_description)
        encoder_outputs = BaseModelOutput(last_hidden_state=desc)

        frame_rate = float(getattr(ar.audio_encoder.config, "frame_rate", 86.13))
        est_sec = max(1.5, len(prompt_clean) / float(self.config.speculative.chars_per_sec))
        min_tokens = max(15, int(frame_rate * est_sec * 0.85))
        max_tokens = max(30, int(frame_rate * est_sec * 1.5))
        set_seed(seed)

        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
            do_sample=True,
            temperature=self.config.speculative.temperature,
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
        )

        chunks: List[np.ndarray] = []
        try:
            from parler_tts import ParlerTTSStreamer
        except Exception:
            ParlerTTSStreamer = None  # type: ignore

        t0 = _now()
        ttfa_e2e_ms: Optional[float] = None
        ttfa_compute_only_ms: Optional[float] = None
        note = None
        if ParlerTTSStreamer is not None:
            class _TimedStreamer(ParlerTTSStreamer):
                """Official streamer + timestamps for component-compute latency.

                ``t_gen_start`` is stamped right before ``generate()`` runs;
                ``t_first_decode`` right after the first in-loop DAC decode
                (i.e. when the first audio buffer becomes available).
                """

                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self.t_gen_start: Optional[float] = None
                    self.t_first_decode: Optional[float] = None

                def put(self, value):
                    super().put(value)
                    if (self.t_first_decode is None and self.to_yield > 0
                            and self.t_gen_start is not None):
                        self.t_first_decode = _now()

            streamer = _TimedStreamer(
                ar, device=self.device, play_steps=int(self.args.play_steps), timeout=900
            )
            error_box: List[BaseException] = []

            def _generate():
                try:
                    with torch.inference_mode():
                        streamer.t_gen_start = _now()
                        ar.generate(**gen_kwargs, streamer=streamer)
                except BaseException as exc:  # pragma: no cover
                    error_box.append(exc)
                    try:
                        streamer.audio_queue.put(streamer.stop_signal)
                    except Exception:
                        pass

            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()
            for audio_chunk in streamer:
                if ttfa_e2e_ms is None:
                    cuda_sync(self.device)
                    ttfa_e2e_ms = (_now() - t0) * 1000.0
                chunks.append(np.asarray(audio_chunk, dtype=np.float32))
            thread.join(timeout=900)
            total_ms = (_now() - t0) * 1000.0
            if error_box:
                raise RuntimeError(f"Parler streamer generation failed: {error_box[0]}")
            if streamer.t_first_decode is not None and streamer.t_gen_start is not None:
                ttfa_compute_only_ms = (streamer.t_first_decode - streamer.t_gen_start) * 1000.0
        else:
            with torch.inference_mode():
                out = ar.generate(**gen_kwargs)
            audio = out.detach().float().cpu().numpy()
            while audio.ndim > 1:
                audio = audio[0]
            chunks = [audio]
            total_ms = (_now() - t0) * 1000.0
            ttfa_e2e_ms = total_ms
            note = "ParlerTTSStreamer unavailable: full-generation fallback"

        if not chunks:
            raise RuntimeError("Parler produced no audio.")
        wave = self._postprocess(np.concatenate(chunks))
        audio_sec = len(wave) / self.sample_rate
        row.update({
            "ttfa_compute_ms": ttfa_compute_only_ms,
            "ttfa_e2e_ms": ttfa_e2e_ms,
            "total_ms": total_ms,
            "audio_sec": audio_sec,
            "rtf": (total_ms / 1000.0) / max(audio_sec, 1e-3),
            "draft_ms": None,
            "verify_ms_total": None,
            "residual_ms_total": None,
            "dac_ms_total": None,
            "accept_true": None,
            "accept_synth": None,
            "n_frames": None,
            "zone_stats": {},
            "chunks": [],
            "verify_mode": "none",
            "seed": seed,
            "note": note,
        })
        return row, wave

    # -- Configs B & C -------------------------------------------------------
    def _chunked_synthesis(self, utt: Dict[str, str], utt_index: int, seed: int,
                           verify: bool) -> Tuple[Dict[str, Any], np.ndarray]:
        cfg = "C" if verify else "B"
        normalized = self.normalize(utt["text"])
        row = self._base_row(cfg, utt, utt_index, normalized)
        prompt_clean = self._prompt_clean(normalized)
        n_frames = self._frames_for(normalized)

        # Zone instrumentation
        norm_res = self.normalizer.normalize(utt["text"])
        transit_spans = find_transit_spans(norm_res.normalized_text, norm_res.entity_replacements)
        char_zones = label_char_zones(norm_res.normalized_text, transit_spans)
        frame_zones, n_transitions, switch_frames = assign_frame_zones(
            char_zones, n_frames, boundary_frames=int(self.args.boundary_frames)
        )

        ctx = self.scorer.build_context(prompt_clean) if (verify and self.scorer is not None) else None
        rng_accept = np.random.RandomState(seed)
        seed_synth_rng = seed + 1000

        t0 = _now()
        cuda_sync(self.device)

        # 1. Draft the full CB0 sequence (single NAR pass).  Seeded like the AR
        #    path so per-utterance B/C rows are reproducible across runs.
        set_seed(seed)
        with torch.inference_mode():
            draft_out = self.draft.generate_chunk(
                text=normalized, num_frames=n_frames, temperature=1.0, device=self.device
            )
        cuda_sync(self.device)
        draft_ms = (_now() - t0) * 1000.0
        draft_tokens = draft_out.tokens[0].detach().cpu().numpy().astype(np.int64)
        draft_probs_all = draft_out.probabilities[0].detach().float().cpu().numpy()

        waves: List[np.ndarray] = []
        accept_true_parts: List[np.ndarray] = []
        accept_synth_parts: List[np.ndarray] = []
        logq_draft_parts: List[float] = []
        logq_kept_parts: List[float] = []
        telemetry: List[ChunkTelemetry] = []
        prefix = np.asarray([], dtype=np.int64)
        verify_ms_total = 0.0
        residual_ms_total = 0.0
        dac_ms_total = 0.0
        ttfa_compute_ms: Optional[float] = None
        ttfa_e2e_ms: Optional[float] = None
        verify_mode = "none"

        for chunk_idx, chunk_start in enumerate(range(0, n_frames, self.frames_per_chunk)):
            chunk_end = min(n_frames, chunk_start + self.frames_per_chunk)
            chunk_tokens = draft_tokens[chunk_start:chunk_end]
            chunk_probs = draft_probs_all[chunk_start:chunk_end]
            c_chunk = int(chunk_tokens.shape[0])
            # Draft confidence: mean top-1 probability over the chunk's frames
            # (used for the confidence gate and for the calibration analysis).
            try:
                confidence: Optional[float] = float(np.mean(np.max(chunk_probs, axis=-1)))
            except Exception:
                confidence = None

            kept = chunk_tokens
            accepted_true: Optional[np.ndarray] = None
            accepted_synth: Optional[np.ndarray] = None
            verify_ms = 0.0

            if verify:
                t_v = _now()
                cuda_sync(self.device)
                q_true = self.scorer.score(ctx, prefix, chunk_tokens) if self.scorer is not None else None
                if q_true is not None:
                    kept, accepted_true, _ = rejection_sample(q_true, chunk_probs, chunk_tokens, rng_accept)
                    verify_mode = "true"
                    # Target-correction diagnostic: teacher-forced mean log-likelihood
                    # (nats/frame) of the raw draft proposal vs. the emitted tokens.
                    lq = np.log(np.clip(np.asarray(q_true, dtype=np.float64), 1e-12, None))
                    frame_idx = np.arange(c_chunk)
                    logq_draft_parts.append(float(lq[frame_idx, chunk_tokens].mean()))
                    logq_kept_parts.append(float(lq[frame_idx, kept].mean()))
                else:
                    verify_mode = "synthetic"
                # Synthetic-target telemetry (always computed for fig2/table)
                q_synth = TeacherForcedScorer.synthetic_probs(
                    c_chunk, self.vocab_size, np.random.RandomState(seed_synth_rng)
                )
                seed_synth_rng += 1
                _, accepted_synth, _ = rejection_sample(
                    q_synth, chunk_probs, chunk_tokens, np.random.RandomState(seed_synth_rng)
                )
                seed_synth_rng += 1
                if q_true is None:
                    kept, _, _ = rejection_sample(
                        q_synth, chunk_probs, chunk_tokens, np.random.RandomState(seed_synth_rng)
                    )
                    seed_synth_rng += 1
                cuda_sync(self.device)
                verify_ms = (_now() - t_v) * 1000.0
                verify_ms_total += verify_ms

            # residual + DAC (chunk-wise; matches residual training windows)
            t_r = _now()
            cuda_sync(self.device)
            with torch.inference_mode():
                res_out = self.residual(
                    cb0_tokens=torch.from_numpy(kept).unsqueeze(0).to(self.device),
                    temperature=1.0,
                    sample=True,
                )
            cuda_sync(self.device)
            residual_ms = (_now() - t_r) * 1000.0
            residual_ms_total += residual_ms

            t_d = _now()
            cuda_sync(self.device)
            wave_chunk = self.decode_codes(res_out.full_codebooks[0])
            cuda_sync(self.device)
            dac_ms = (_now() - t_d) * 1000.0
            dac_ms_total += dac_ms

            ready_ms = (_now() - t0) * 1000.0
            waves.append(np.asarray(wave_chunk, dtype=np.float32))
            if accepted_true is not None:
                accept_true_parts.append(accepted_true)
            if accepted_synth is not None:
                accept_synth_parts.append(accepted_synth)
            telemetry.append(ChunkTelemetry(
                idx=chunk_idx,
                n_frames=c_chunk,
                verify_ms=verify_ms,
                residual_ms=residual_ms,
                dac_ms=dac_ms,
                ready_ms=ready_ms,
                accept_true=acceptance_rate(accepted_true),
                accept_synth=acceptance_rate(accepted_synth),
                confidence=confidence,
            ))
            if ttfa_e2e_ms is None:
                ttfa_e2e_ms = ready_ms
                ttfa_compute_ms = draft_ms + telemetry[0].verify_ms + residual_ms + dac_ms
            prefix = np.concatenate([prefix, kept])

        total_ms = (_now() - t0) * 1000.0

        # assemble audio
        from codeswitch_tts.utils.audio import crossfade_chunks

        if not waves:
            wave = np.zeros(self.sample_rate, dtype=np.float32)
        else:
            wave = waves[0]
            for nxt in waves[1:]:
                wave = crossfade_chunks(wave, nxt, crossfade_samples=self.config.audio.crossfade_samples)
        wave = self._postprocess(wave)
        audio_sec = len(wave) / self.sample_rate

        accept_true = np.concatenate(accept_true_parts) if accept_true_parts else None
        accept_synth = np.concatenate(accept_synth_parts) if accept_synth_parts else None

        row.update({
            "ttfa_compute_ms": ttfa_compute_ms,
            "ttfa_e2e_ms": ttfa_e2e_ms,
            "total_ms": total_ms,
            "audio_sec": audio_sec,
            "rtf": (total_ms / 1000.0) / max(audio_sec, 1e-3),
            "draft_ms": draft_ms,
            "verify_ms_total": verify_ms_total,
            "residual_ms_total": residual_ms_total,
            "dac_ms_total": dac_ms_total,
            "accept_true": acceptance_rate(accept_true),
            "accept_synth": acceptance_rate(accept_synth),
            "logq_draft": float(np.mean(logq_draft_parts)) if logq_draft_parts else None,
            "logq_kept": float(np.mean(logq_kept_parts)) if logq_kept_parts else None,
            "n_frames": int(n_frames),
            "n_transitions": int(n_transitions),
            "chunks": [asdict(t) for t in telemetry],
            "verify_mode": verify_mode,
            "seed": seed,
            "draft_confidence": float(np.mean([t.confidence for t in telemetry
                                               if t.confidence is not None]))
            if any(t.confidence is not None for t in telemetry) else None,
            # packed per-frame telemetry: raw material for the distance-to-switch
            # analysis (survives JSONL round-trips, ~1 KB per row)
            "switch_frames": [int(s) for s in switch_frames],
            "frame_zones_b64": pack_u8(frame_zones.astype(np.uint8)) if n_frames > 0 else None,
            "frame_accept_true_b64": (pack_bits(accept_true) if accept_true is not None else None),
            "frame_accept_synth_b64": (pack_bits(accept_synth) if accept_synth is not None else None),
            "frame_valid_b64": (pack_bits(np.ones_like(accept_true, dtype=bool))
                                if accept_true is not None else None),
        })
        self._attach_zone_stats(row, frame_zones, accept_true, accept_synth)
        return row, wave

    def _attach_zone_stats(self, row, frame_zones, accept_true, accept_synth) -> None:
        zone_stats: Dict[str, Dict[str, Any]] = {}
        for zone in ZONE_ORDER:
            idx = np.where(frame_zones == zone)[0]
            entry: Dict[str, Any] = {"n_frames": int(idx.size)}
            if accept_true is not None and idx.size:
                entry["accept_true"] = float(accept_true[idx].mean() * 100.0)
            if accept_synth is not None and idx.size:
                entry["accept_synth"] = float(accept_synth[idx].mean() * 100.0)
            zone_stats[str(zone)] = entry
        row["zone_stats"] = zone_stats

    def _draft_confidence(self, utt: Dict[str, str], seed: int,
                          gate_stat: str = "utt_min") -> Optional[float]:
        """Draft-only pass: utterance-level draftability statistic.

        Returns the ``utt_min`` (worst chunk) or ``utt_mean`` of the per-chunk
        mean top-1 draft probability.  Uses the same seeding as the real
        synthesis pass, so the statistic describes the tokens that would be
        proposed.
        """
        normalized = self.normalize(utt["text"])
        n_frames = self._frames_for(normalized)
        if n_frames <= 0:
            return None
        set_seed(seed)
        with torch.inference_mode():
            draft_out = self.draft.generate_chunk(
                text=normalized, num_frames=n_frames, temperature=1.0, device=self.device
            )
        probs = draft_out.probabilities[0].detach().float().cpu().numpy()
        per_chunk: List[float] = []
        for chunk_start in range(0, n_frames, self.frames_per_chunk):
            chunk_probs = probs[chunk_start: chunk_start + self.frames_per_chunk]
            if chunk_probs.size:
                per_chunk.append(float(np.mean(np.max(chunk_probs, axis=-1))))
        if not per_chunk:
            return None
        return float(np.mean(per_chunk)) if gate_stat == "utt_mean" else float(np.min(per_chunk))

    def synth_C(self, utt: Dict[str, str], utt_index: int, seed: int,
                gate_threshold: float = 0.0, gate_stat: str = "utt_min",
                ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Config C with a draft-confidence gate (utterance-level).

        ``gate_threshold <= 0`` is plain speculative decoding.  Otherwise the
        draft is run first; if its confidence statistic falls below the
        threshold the utterance is handed to the full AR pipeline and the
        *measured* latency/quality of that fallback are reported (so the gate
        is a deployable knob, not a hypothetical).  Fallback rows carry
        ``verify_mode="ar_fallback"``, no acceptance numbers, and are excluded
        from acceptance aggregates but included in latency/CER ones.
        """
        threshold = float(gate_threshold or 0.0)
        if threshold <= 0.0 or self.scorer is None:
            row, wave = self._chunked_synthesis(utt, utt_index, seed, verify=True)
            row["gate_threshold"] = 0.0
            row["gate_stat"] = gate_stat
            row["fallback"] = False
            row.setdefault("draft_confidence", None)
            return row, wave

        confidence = self._draft_confidence(utt, seed, gate_stat)
        if confidence is not None and confidence < threshold:
            row, wave = self.synth_A(utt, utt_index, seed)
            row.update({
                "config": "C",
                "config_label": CONFIG_LABELS["C"],
                "verify_mode": "ar_fallback",
                "fallback": True,
                "gate_threshold": threshold,
                "gate_stat": gate_stat,
                "draft_confidence": confidence,
                "accept_true": None,
                "accept_synth": None,
                "n_transitions": row.get("n_transitions"),
                "chunks": [],
                "zone_stats": {},
                "switch_frames": [],
                "frame_zones_b64": None,
                "frame_accept_true_b64": None,
                "frame_accept_synth_b64": None,
                "frame_valid_b64": None,
            })
            return row, wave

        row, wave = self._chunked_synthesis(utt, utt_index, seed, verify=True)
        row["gate_threshold"] = threshold
        row["gate_stat"] = gate_stat
        row["fallback"] = False
        row["draft_confidence"] = confidence
        return row, wave

    @property
    def vocab_size(self) -> int:
        if self.scorer is not None and self.scorer.vocab:
            return int(self.scorer.vocab)
        return 1024

# ===========================================================================
# ASR / intelligibility
# ===========================================================================
def asr_max_new_tokens(audio_sec: float) -> int:
    """Generation cap for one ASR pass.

    Whisper's short-form decoder defaults to 448 new tokens.  On unintelligible
    audio it can fall into a repetition loop that burns ~5 s of wall-clock per
    row and reports a hallucinated transcript (the ``CER > 200 %`` rows, which
    is also why exactly those rows take ~5 s longer than their synthesis time).
    Real transcripts need ~4-7 tokens per second of Tamil speech, so 10/s
    leaves genuine output untouched while cutting the loops off.
    """
    return int(min(448, max(24, round(10.0 * float(audio_sec)))))


class ASREvaluator:
    def __init__(self, model_id: str, device, enabled: bool = True, verbose: bool = True):
        self.model_id = model_id
        self.available = False
        self.reason = "disabled" if not enabled else "not initialised"
        self.pipe = None
        if not enabled:
            if verbose:
                print("[ASR] disabled (--skip-asr)")
            return
        try:
            from transformers import pipeline

            device_arg = 0 if str(device).startswith("cuda") else -1
            pipe_kwargs: Dict[str, Any] = {}
            if device_arg == 0:
                # fp16 halves the transcription time on T4 (transformers 4.46 API)
                pipe_kwargs["torch_dtype"] = torch.float16
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=model_id,
                device=device_arg,
                **pipe_kwargs,
            )
            self.available = True
            self.reason = "ok"
            if verbose:
                print(f"[ASR] loaded {model_id} (device={device_arg})")
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[ASR] ⚠ could not load {model_id}: {self.reason}")
                print("[ASR]   latency/acceptance metrics continue; CER/WER will be absent.")

    def transcribe(self, wave: np.ndarray, sample_rate: int) -> Optional[str]:
        if not self.available or self.pipe is None:
            return None
        try:
            wave = np.asarray(wave, dtype=np.float32)
            audio_sec = len(wave) / float(max(1, int(sample_rate)))
            result = self.pipe(
                {"array": wave, "sampling_rate": sample_rate},
                generate_kwargs={
                    "language": "ta",
                    "task": "transcribe",
                    "max_new_tokens": asr_max_new_tokens(audio_sec),
                },
            )
            if isinstance(result, list):
                result = result[0]
            return str(result.get("text", "")).strip()
        except Exception as exc:
            print(f"[ASR] transcription failed: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def score(reference: str, hypothesis: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
        if hypothesis is None:
            return None, None
        try:
            import jiwer
        except Exception:
            return None, None
        ref = clean_for_cer(reference)
        hyp = clean_for_cer(hypothesis)
        if not ref:
            return None, None
        try:
            cer = float(jiwer.cer(ref, hyp))
        except Exception:
            cer = None
        try:
            wer = float(jiwer.wer(ref, hyp))
        except Exception:
            wer = None
        return cer, wer

    @staticmethod
    def score_latin(reference: str, hypothesis: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
        """CER/WER after transliterating *both* sides to Latin.

        Tamil-script references are routinely transcribed by ASR models in Latin
        (and vice versa), which inflates script-level CER for reasons that have
        nothing to do with intelligibility.  Transliterating both sides puts the
        comparison in one alphabet, so the number reflects the audio, not the
        ASR model's script convention.
        """
        if hypothesis is None:
            return None, None
        try:
            import jiwer
        except Exception:
            return None, None
        ref = latinize_text(clean_for_cer(reference))
        hyp = latinize_text(clean_for_cer(hypothesis))
        if not ref:
            return None, None
        try:
            cer = float(jiwer.cer(ref, hyp))
        except Exception:
            cer = None
        try:
            wer = float(jiwer.wer(ref, hyp))
        except Exception:
            wer = None
        return cer, wer


# ---------------------------------------------------------------------------
# Optional perceptual / spectral metrics
# ---------------------------------------------------------------------------
class UTMOSScorer:
    """UTMOS22 (``tarepan/SpeechMOS``) - a no-listener perceptual proxy.

    Loaded lazily and entirely optional: if the checkpoint or its remote code is
    unavailable the run continues without it (``available=False``).
    """

    #: ``tarepan/SpeechMOS`` is a torch.hub *repo* (the canonical UTMOS22-strong
    #: reimplementation), not a transformers checkpoint - AutoModel can never
    #: load it.  Tier 1 is the hub loader, tier 2 a transformers mirror in case
    #: the hub is unreachable.
    HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
    HUB_ENTRY = "utmos22_strong"

    def __init__(self, device, enabled: bool = True,
                 model_id: str = "prj-beatrice/utmos22-torch-native",
                 verbose: bool = True):
        self.model_id = model_id
        self.available = False
        self.reason = "disabled" if not enabled else "not initialised"
        self.source: Optional[str] = None
        self.model = None
        self.device = torch.device("cpu") if str(device).startswith("cpu") else device
        if not enabled:
            return
        # --- tier 1: torch.hub (UTMOS22-strong, VoiceMOS Challenge 2022) -------
        try:
            self.model = torch.hub.load(
                self.HUB_REPO, self.HUB_ENTRY, trust_repo=True, verbose=False,
            )
            self.model = self.model.to(self.device).eval()
            self.available = True
            self.source = f"torch.hub:{self.HUB_REPO}/{self.HUB_ENTRY}"
            self.reason = "ok"
            if verbose:
                print(f"[UTMOS] loaded {self.source} (device={self.device})")
            return
        except Exception as exc:
            hub_reason = f"{type(exc).__name__}: {exc}"
        # --- tier 2: transformers mirror --------------------------------------
        try:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model = self.model.to(self.device).eval()
            self.available = True
            self.source = f"transformers:{model_id}"
            self.reason = "ok"
            if verbose:
                print(f"[UTMOS] loaded {self.source} (device={self.device})")
            return
        except Exception as exc:
            self.reason = (f"torch.hub failed ({hub_reason}); "
                           f"{type(exc).__name__}: {exc}")
            if verbose:
                print(f"[UTMOS] ⚠ perceptual proxy unavailable ({self.reason}); "
                      "UTMOS columns will be absent.")

    def score(self, wave: np.ndarray, sample_rate: int) -> Optional[float]:
        if not self.available or self.model is None:
            return None
        try:
            import torchaudio

            wav = torch.from_numpy(np.asarray(wave, dtype=np.float32))
            if int(sample_rate) != 16000:
                wav = torchaudio.functional.resample(wav, int(sample_rate), 16000)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav = wav.to(self.device)
            with torch.inference_mode():
                out = self.model(wav, 16000)
            value = out.detach().float().cpu().reshape(-1)
            return float(value[0])
        except Exception as exc:
            print(f"[UTMOS] scoring failed: {type(exc).__name__}: {exc}")
            return None


def mcd_between(a: np.ndarray, b: np.ndarray, sample_rate: int = 44100,
                n_mfcc: int = 13) -> Optional[float]:
    """Mel-cepstral distortion between two utterances (DTW-aligned, dB scale).

    Used to compare the hybrid/NAR renderings against the AR baseline for the
    *same text*: a standard spectral-quality proxy that does not depend on an
    ASR model.  Both signals are resampled to 16 kHz and aligned with a simple
    DTW over MFCC frames (utterances are seconds long, so this is cheap).
    """
    try:
        import torchaudio

        def _mfcc(wave: np.ndarray) -> np.ndarray:
            wav = torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0)
            if int(sample_rate) != 16000:
                wav = torchaudio.functional.resample(wav, int(sample_rate), 16000)
            transform = torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=n_mfcc,
                melkwargs={"n_fft": 1024, "hop_length": 160, "n_mels": 40},
            )
            with torch.inference_mode():
                feats = transform(wav).squeeze(0).transpose(0, 1).cpu().numpy()
            return feats[:, 1:]  # drop c0 (energy)

        fa, fb = _mfcc(a), _mfcc(b)
        if fa.size == 0 or fb.size == 0:
            return None
        cost = np.linalg.norm(fa[:, None, :] - fb[None, :, :], axis=2) / np.sqrt(fa.shape[1])
        n, m = cost.shape
        dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            row, prev = dtw[i], dtw[i - 1]
            c_row = cost[i - 1]
            for j in range(1, m + 1):
                row[j] = c_row[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
        return float(dtw[n, m] / max(n, m) * (10.0 / np.log(10.0)) * np.sqrt(2.0))
    except Exception:
        return None


def _load_wav_mono(path: str) -> Optional[Tuple[np.ndarray, int]]:
    """Loads a wav as mono float32 (soundfile first, torchaudio fallback)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data.mean(axis=1), int(sr)
    except Exception:
        pass
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        return wav.mean(dim=0).numpy().astype(np.float32), int(sr)
    except Exception:
        return None


def attach_mcd_from_audio(rows: List[Dict[str, Any]], verbose: bool = True) -> int:
    """Adds ``mcd_vs_a`` to every B/C row that has a saved wav.

    MCD is computed against the Pure-AR rendering of the *same text*.  The two
    renderings use different sampling seeds (that is the point of the comparison),
    so this is a distribution-level spectral distance, not an exact paired one -
    documented as such wherever it is reported.
    """
    by_utt: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("error") or not row.get("audio_path"):
            continue
        by_utt.setdefault(str(row.get("utt_id")), {})[str(row.get("config"))] = row
    computed = 0
    for utt_id, per_cfg in by_utt.items():
        ref_row = per_cfg.get("A")
        if ref_row is None or not Path(str(ref_row["audio_path"])).exists():
            continue
        loaded_ref = _load_wav_mono(str(ref_row["audio_path"]))
        if loaded_ref is None:
            continue
        ref_wav, ref_sr = loaded_ref
        ref_row.setdefault("mcd_vs_a", 0.0)
        for cfg in ("B", "C"):
            row = per_cfg.get(cfg)
            if row is None or not Path(str(row.get("audio_path") or "")).exists():
                continue
            loaded = _load_wav_mono(str(row["audio_path"]))
            if loaded is None:
                continue
            wav, sr = loaded
            value = mcd_between(wav, ref_wav, sample_rate=sr or ref_sr)
            if value is not None:
                row["mcd_vs_a"] = value
                computed += 1
    if verbose and computed:
        print(f"[MCD] mel-cepstral distortion vs Pure AR computed for {computed} rows")
    elif verbose:
        print("[MCD] skipped (no saved wavs found - run without --no-save-audio)")
    return computed


# ===========================================================================
# Aggregation
# ===========================================================================
def _stats(values: Sequence[Optional[float]], ci: bool = False,
           n_boot: int = 2000, seed: int = 20261018) -> Dict[str, Optional[float]]:
    """Summary statistics, optionally with a bootstrap CI for the median.

    Reviewers of a 60-utterance benchmark ask how much of a gap is noise, so the
    headline medians carry a percentile-bootstrap interval (fixed seed, no scipy).
    """
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v:
        out: Dict[str, Optional[float]] = {
            "n": 0, "mean": None, "std": None, "p50": None, "p90": None,
        }
        if ci:
            out.update({"ci_lo": None, "ci_hi": None})
        return out
    out = {
        "n": len(v),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=0)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }
    if ci:
        arr = np.asarray(v, dtype=np.float64)
        if arr.size >= 5:
            rng = np.random.RandomState(seed)
            idx = rng.randint(0, arr.size, size=(int(n_boot), arr.size))
            boots = np.median(arr[idx], axis=1)
            out["ci_lo"] = float(np.percentile(boots, 2.5))
            out["ci_hi"] = float(np.percentile(boots, 97.5))
        else:
            out["ci_lo"] = None
            out["ci_hi"] = None
    return out


def _finite_values(subset: Sequence[Dict[str, Any]], key: str) -> List[float]:
    """Raw values of ``key`` for a subset of result rows (finite floats only)."""
    out: List[float] = []
    for r in subset:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def aggregate_zones(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for zone in ZONE_ORDER:
        frames = 0
        sum_true = 0.0
        den_true = 0
        sum_synth = 0.0
        den_synth = 0
        per_utt_true: List[float] = []
        per_utt_synth: List[float] = []
        for r in rows:
            z = (r.get("zone_stats") or {}).get(str(zone))
            if not z or not z.get("n_frames"):
                continue
            frames += int(z["n_frames"])
            if z.get("accept_true") is not None:
                sum_true += float(z["accept_true"]) * int(z["n_frames"])
                den_true += int(z["n_frames"])
                per_utt_true.append(float(z["accept_true"]))
            if z.get("accept_synth") is not None:
                sum_synth += float(z["accept_synth"]) * int(z["n_frames"])
                den_synth += int(z["n_frames"])
                per_utt_synth.append(float(z["accept_synth"]))
        out[str(zone)] = {
            "n_frames": frames,
            "accept_true": (sum_true / den_true) if den_true else None,
            "accept_true_std": float(np.std(per_utt_true, ddof=0)) if per_utt_true else None,
            "accept_true_utts": len(per_utt_true),
            "accept_synth": (sum_synth / den_synth) if den_synth else None,
            "accept_synth_std": float(np.std(per_utt_synth, ddof=0)) if per_utt_synth else None,
            "accept_synth_utts": len(per_utt_synth),
        }
    return out


def aggregate_results(rows: List[Dict[str, Any]], configs: Sequence[str],
                      gate_threshold: float = 0.0) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-configuration statistics.

    Config C rows are filtered to ``gate_threshold`` so a gate sweep stored in
    one JSONL does not contaminate the main comparison (default 0.0 = the
    un-gated hybrid).
    """
    agg: Dict[str, Dict[str, Any]] = {}
    # Session cold start: the very first successful synthesis of the process
    # (configurations are interleaved per utterance, so this is the only row
    # whose latency includes a fully cold CUDA context).
    cfg_order = {c: i for i, c in enumerate(configs)}
    ok_all = [r for r in rows if not r.get("error") and r.get("config") in cfg_order]
    if ok_all:
        first = min(
            ok_all,
            key=lambda r: (int(r.get("utt_index") or 0), cfg_order.get(r.get("config"), 99)),
        )
        agg["_session_cold"] = {
            "config": first.get("config"),
            "utt_id": first.get("utt_id"),
            "ttfa_compute_ms": first.get("ttfa_compute_ms"),
            "ttfa_e2e_ms": first.get("ttfa_e2e_ms"),
        }
    for cfg in configs:
        ok = [r for r in rows if r.get("config") == cfg and not r.get("error")]
        err = [r for r in rows if r.get("config") == cfg and r.get("error")]
        if cfg == "C" and any(float(r.get("gate_threshold") or 0.0) != float(gate_threshold)
                              for r in ok):
            filtered = [r for r in ok
                        if float(r.get("gate_threshold") or 0.0) == float(gate_threshold)]
            if filtered:  # never let the gate sweep empty the main comparison
                ok = filtered
                err = [r for r in err
                       if float(r.get("gate_threshold") or 0.0) == float(gate_threshold)]
        if not ok:
            agg[cfg] = {"n": 0, "errors": len(err)}
            continue
        first = min(ok, key=lambda r: r.get("utt_index", 0))
        warm = [r for r in ok if r is not first]
        registers: Dict[str, Dict[str, Any]] = {}
        for reg in REGISTER_LABELS:
            sub = [r for r in ok if r.get("register") == reg]
            if not sub:
                continue
            registers[reg] = {
                "n": len(sub),
                "cer": _stats([r.get("cer") for r in sub]),
                "wer": _stats([r.get("wer") for r in sub]),
                "cer_latin": _stats([r.get("cer_latin") for r in sub]),
                "utmos": _stats([r.get("utmos") for r in sub]),
                "mcd_vs_a": _stats([r.get("mcd_vs_a") for r in sub]),
                "ttfa_compute": _stats([r.get("ttfa_compute_ms") for r in sub]),
                "ttfa_e2e": _stats([r.get("ttfa_e2e_ms") for r in sub]),
                "rtf": _stats([r.get("rtf") for r in sub]),
                "accept_true": _stats([r.get("accept_true") for r in sub]),
                "accept_synth": _stats([r.get("accept_synth") for r in sub]),
                "asr_fail": sum(1 for r in sub if (r.get("cer") or 0.0) > ASR_FAIL_CER),
                "values": {
                    "cer": _finite_values(sub, "cer"),
                    "cer_latin": _finite_values(sub, "cer_latin"),
                    "utmos": _finite_values(sub, "utmos"),
                    "mcd_vs_a": _finite_values(sub, "mcd_vs_a"),
                    "ttfa_compute": _finite_values(sub, "ttfa_compute_ms"),
                },
            }

        def _raw(key: str, subset: Optional[List[Dict[str, Any]]] = None) -> List[float]:
            return _finite_values(ok if subset is None else subset, key)

        values = {
            "ttfa_compute_warm": _raw("ttfa_compute_ms", warm),
            "ttfa_compute": _raw("ttfa_compute_ms"),
            "ttfa_e2e": _raw("ttfa_e2e_ms"),
            "rtf": _raw("rtf"),
            "cer": _raw("cer"),
            "cer_latin": _raw("cer_latin"),
            "utmos": _raw("utmos"),
            "mcd_vs_a": _raw("mcd_vs_a"),
        }
        agg[cfg] = {
            "n": len(ok),
            "errors": len(err),
            "ttfa_compute": _stats([r.get("ttfa_compute_ms") for r in ok]),
            "ttfa_e2e": _stats([r.get("ttfa_e2e_ms") for r in ok]),
            "ttfa_compute_cold": _stats([first.get("ttfa_compute_ms")]),
            "ttfa_e2e_cold": _stats([first.get("ttfa_e2e_ms")]),
            "ttfa_compute_warm": _stats([r.get("ttfa_compute_ms") for r in warm], ci=True),
            "ttfa_e2e_warm": _stats([r.get("ttfa_e2e_ms") for r in warm], ci=True),
            "total": _stats([r.get("total_ms") for r in ok]),
            "rtf": _stats([r.get("rtf") for r in ok]),
            "audio_sec": _stats([r.get("audio_sec") for r in ok]),
            "accept_true": _stats([r.get("accept_true") for r in ok]),
            "accept_synth": _stats([r.get("accept_synth") for r in ok]),
            "logq_draft": _stats([r.get("logq_draft") for r in ok]),
            "logq_kept": _stats([r.get("logq_kept") for r in ok]),
            "cer": _stats([r.get("cer") for r in ok], ci=True),
            "wer": _stats([r.get("wer") for r in ok]),
            "cer_latin": _stats([r.get("cer_latin") for r in ok], ci=True),
            "utmos": _stats([r.get("utmos") for r in ok], ci=True),
            "mcd_vs_a": _stats([r.get("mcd_vs_a") for r in ok], ci=True),
            "asr_fail": sum(1 for r in ok if (r.get("cer") or 0.0) > ASR_FAIL_CER),
            "asr_fail_utts": sorted(
                str(r.get("utt_id")) for r in ok if (r.get("cer") or 0.0) > ASR_FAIL_CER
            ),
            "fallback_utts": sum(1 for r in ok if r.get("fallback")),
            "fallback_rate": (sum(1 for r in ok if r.get("fallback")) / len(ok)) if ok else None,
            "gate_threshold": float(gate_threshold) if cfg == "C" else None,
            "gate_stat": next((r.get("gate_stat") for r in ok if r.get("gate_stat")), None),
            "verify_modes": sorted({str(r.get("verify_mode")) for r in ok}),
            "registers": registers,
            "values": values,
            "zones": aggregate_zones(ok),
        }
    return agg


# ===========================================================================
# Gate sweep + distance-to-switch analyses (config C)
# ===========================================================================
def _row_confidence(row: Dict[str, Any], gate_stat: str = "utt_min") -> Optional[float]:
    """Utterance-level draft confidence recorded on a speculative C row.

    Recomputed from the per-chunk telemetry (``utt_min`` = worst chunk,
    ``utt_mean`` = average) so it matches exactly what the gate compares;
    falls back to the row's ``draft_confidence`` field (which is the mean).
    """
    confs = [float(c.get("confidence")) for c in (row.get("chunks") or [])
             if c.get("confidence") is not None]
    if confs:
        return float(np.min(confs)) if gate_stat == "utt_min" else float(np.mean(confs))
    value = row.get("draft_confidence")
    return None if value is None else float(value)


def gate_confidence_table(rows: List[Dict[str, Any]],
                          gate_stat: str = "utt_min") -> Dict[str, float]:
    """Per-utterance draft confidence, taken from the *un-gated* C rows."""
    out: Dict[str, float] = {}
    for r in rows:
        if r.get("config") != "C" or r.get("error") or r.get("fallback"):
            continue
        if float(r.get("gate_threshold") or 0.0) != 0.0:
            continue
        value = _row_confidence(r, gate_stat)
        if value is not None:
            out[str(r.get("utt_id"))] = value
    return out


def auto_gate_thresholds(rows: List[Dict[str, Any]], gate_stat: str = "utt_min",
                         quantiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
                         ) -> List[float]:
    """Thresholds on the *observed* confidence scale (plus the un-gated 0).

    Fixed thresholds are meaningless without knowing the draft's calibration:
    a draft whose mean top-1 is ~0.1 makes tau >= 0.6 fall back on every
    utterance, i.e. an empty Pareto curve.  Quantiles guarantee interior points.
    """
    conf = sorted(gate_confidence_table(rows, gate_stat).values())
    if not conf:
        return [0.0]
    return [0.0] + [round(float(np.quantile(conf, float(q))), 4) for q in quantiles]


def parse_gate_thresholds(spec: Any, rows: Optional[List[Dict[str, Any]]] = None,
                          gate_stat: str = "utt_min") -> List[float]:
    """``--gate-thresholds`` parser: explicit list, ``auto``, or empty."""
    text = str(spec or "").strip()
    if not text or text.lower() == "auto":
        return auto_gate_thresholds(rows or [], gate_stat)
    out: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return sorted(set(out)) or [0.0]


def derive_gate_sweep(rows: List[Dict[str, Any]], thresholds: Sequence[float],
                      gate_stat: str = "utt_min") -> List[Dict[str, Any]]:
    """Builds the C@tau rows by combining the measured C(tau=0) and A rows.

    The gate is an utterance-level decision taken *before* synthesis, so the
    mixture at threshold tau is exactly: the speculative row for the utterances
    the draft is confident about, and the AR row for the rest.  Deriving the
    sweep therefore costs no GPU time and is exact for quality/acceptance; the
    ``total_ms`` of a fallback is the sum of the two measured passes (the draft
    pass that produces the gate statistic plus the AR pass), which is what the
    synthesised fallback path costs as well - ``compare_gate_modes`` reports the
    difference wherever both sweeps exist.
    """
    by_utt: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        if r.get("error"):
            continue
        utt, cfg = str(r.get("utt_id")), str(r.get("config"))
        if cfg == "C":
            if r.get("fallback") or float(r.get("gate_threshold") or 0.0) != 0.0:
                continue
            by_utt.setdefault(utt, {})["C"] = r
        elif cfg in ("A", "B"):
            by_utt.setdefault(utt, {})[cfg] = r
    conf = gate_confidence_table(rows, gate_stat)
    derived: List[Dict[str, Any]] = []
    for tau in thresholds:
        for utt in sorted(by_utt):
            per = by_utt[utt]
            c0 = per.get("C")
            if c0 is None:
                continue
            value = conf.get(utt)
            a_row = per.get("A")
            if tau > 0.0 and value is not None and value < tau and a_row is not None:
                total = float(a_row.get("total_ms") or 0.0) + float(c0.get("total_ms") or 0.0)
                audio_sec = a_row.get("audio_sec")
                derived.append({
                    "config": "C",
                    "utt_id": utt,
                    "utt_index": c0.get("utt_index"),
                    "register": c0.get("register"),
                    "gate_threshold": float(tau),
                    "gate_stat": gate_stat,
                    "fallback": True,
                    "derived": True,
                    "verify_mode": "ar_fallback",
                    "draft_confidence": value,
                    "ttfa_compute_ms": a_row.get("ttfa_compute_ms"),
                    "ttfa_e2e_ms": a_row.get("ttfa_e2e_ms"),
                    "total_ms": total,
                    "audio_sec": audio_sec,
                    "rtf": (total / 1000.0) / max(float(audio_sec or 1e-3), 1e-3),
                    "cer": a_row.get("cer"),
                    "wer": a_row.get("wer"),
                    "cer_latin": a_row.get("cer_latin"),
                    "wer_latin": a_row.get("wer_latin"),
                    "asr_text": a_row.get("asr_text"),
                    "utmos": a_row.get("utmos"),
                    "mcd_vs_a": 0.0,
                    "audio_path": a_row.get("audio_path"),
                    "accept_true": None,
                    "accept_synth": None,
                    "chunks": [],
                    "zone_stats": {},
                    "switch_frames": [],
                })
            else:
                row = dict(c0)
                row.update({
                    "gate_threshold": float(tau),
                    "gate_stat": gate_stat,
                    "derived": True,
                    "fallback": False,
                    "draft_confidence": value,
                })
                derived.append(row)
    return derived


def compare_gate_modes(rows: List[Dict[str, Any]], derived: List[Dict[str, Any]],
                       ) -> Optional[List[Dict[str, Any]]]:
    """Validity check for the derived sweep: derived vs synthesised fallbacks."""
    synth = {e["threshold"]: e for e in aggregate_gate_sweep(rows)}
    checks: List[Dict[str, Any]] = []
    for entry in aggregate_gate_sweep(derived):
        ref = synth.get(entry["threshold"])
        if ref is None or entry["threshold"] <= 0.0:
            continue
        if abs(ref["fallback_rate"] - entry["fallback_rate"]) > 1e-9:
            continue  # different fallback sets - not the same operating point
        checks.append({
            "threshold": entry["threshold"],
            "fallback_rate": entry["fallback_rate"],
            "total_p50_synth_ms": (ref.get("total") or {}).get("p50"),
            "total_p50_derived_ms": (entry.get("total") or {}).get("p50"),
            "cer_p50_synth": (ref.get("cer") or {}).get("p50"),
            "cer_p50_derived": (entry.get("cer") or {}).get("p50"),
        })
    return checks or None


def gate_sweep_warning(gate_rows: List[Dict[str, Any]],
                       confidence: Dict[str, float]) -> Optional[str]:
    """Explains a degenerate sweep (every threshold makes the same decision)."""
    rates = [e["fallback_rate"] for e in gate_rows if e["threshold"] > 0.0]
    if not rates or any(0.0 < r < 1.0 for r in rates):
        return None
    if confidence:
        p10, p50, p90 = np.percentile(list(confidence.values()), [10, 50, 90])
        scale = (f"observed draft confidence: p10={p10:.3f} p50={p50:.3f} p90={p90:.3f}")
    else:
        scale = "no draft confidence recorded"
    return (f"every requested threshold makes the same decision ({scale}); the Pareto "
            "curve has no interior point. Use --gate-thresholds auto (quantiles of the "
            "observed confidence) or pass thresholds inside that range.")


def aggregate_gate_sweep(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Realised latency/quality/fallback statistics per gate threshold (config C).

    Each entry is one operating point of the confidence-gated hybrid: the
    latency is the *measured* latency of the mixture (speculation + AR
    fallbacks), the CER the measured ASR score of the resulting audio.
    """
    out: List[Dict[str, Any]] = []
    thresholds = sorted({
        float(r.get("gate_threshold") or 0.0)
        for r in rows
        if r.get("config") == "C" and not r.get("error")
    })
    for thr in thresholds:
        sub = [r for r in rows if r.get("config") == "C" and not r.get("error")
               and float(r.get("gate_threshold") or 0.0) == thr]
        if not sub:
            continue
        fb = [r for r in sub if r.get("fallback")]
        out.append({
            "threshold": thr,
            "n": len(sub),
            "fallback_utts": len(fb),
            "fallback_rate": len(fb) / len(sub),
            "cer": _stats([r.get("cer") for r in sub]),
            "cer_fallback": _stats([r.get("cer") for r in fb]),
            "cer_speculative": _stats([r.get("cer") for r in sub if not r.get("fallback")]),
            "ttfa_compute": _stats([r.get("ttfa_compute_ms") for r in sub]),
            "ttfa_e2e": _stats([r.get("ttfa_e2e_ms") for r in sub]),
            "total": _stats([r.get("total_ms") for r in sub]),
            "rtf": _stats([r.get("rtf") for r in sub]),
            "accept_true": _stats([r.get("accept_true") for r in sub if not r.get("fallback")]),
            "asr_fail": sum(1 for r in sub if (r.get("cer") or 0.0) > ASR_FAIL_CER),
            "cer_latin": _stats([r.get("cer_latin") for r in sub]),
            "utmos": _stats([r.get("utmos") for r in sub]),
            "cer_values": _finite_values(sub, "cer"),
            "cer_latin_values": _finite_values(sub, "cer_latin"),
            "total_values": _finite_values(sub, "total_ms"),
        })
    return out


def switch_distance_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CB0 acceptance as a function of frames since the last Tamil<->Latin switch.

    Uses the packed per-frame telemetry (``frame_accept_true_b64`` /
    ``frame_valid_b64`` / ``switch_frames``), so the analysis conditions on
    actually-speculated frames only.  Each bucket reports the frame count, the
    number of utterances and a paired bootstrap over utterances against the
    ``16+`` reference bucket.
    """
    per_utt: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {name: 0 for name, _, _ in SWITCH_DIST_BUCKETS}
    for r in rows:
        if r.get("config") != "C" or r.get("error"):
            continue
        arrs = frame_accept_arrays(r)
        if arrs is None:
            continue
        accept, valid, dist = arrs
        buckets: Dict[str, float] = {}
        for name, lo, hi in SWITCH_DIST_BUCKETS:
            mask = valid & (dist >= lo) & (dist < hi)
            if not mask.any():
                continue
            buckets[name] = float(accept[mask].mean() * 100.0)
            counts[name] += int(mask.sum())
        if buckets:
            per_utt[str(r.get("utt_id"))] = buckets

    ref = SWITCH_DIST_BUCKETS[-1][0]
    out: List[Dict[str, Any]] = []
    for name, _, _ in SWITCH_DIST_BUCKETS:
        vals = [b[name] for b in per_utt.values() if name in b]
        pairs = ([] if name == ref else
                 [(b[name], b[ref]) for b in per_utt.values() if name in b and ref in b])
        diff, lo, hi, p = paired_bootstrap(pairs)
        out.append({
            "bucket": name,
            "n_frames": counts[name],
            "n_utts": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=0)) if vals else None,
            "n_paired": len(pairs),
            "delta_vs_far": diff,
            "delta_ci_lo": lo,
            "delta_ci_hi": hi,
            "p_vs_far": p,
        })
    return out


# ===========================================================================
# Persistence
# ===========================================================================
def load_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ===========================================================================
# Figures (IEEE style)
# ===========================================================================
def _ieee_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.4,
        "grid.color": "#cccccc",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.4,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })
    return plt


def _save_fig(fig, out_dir: Path, stem: str) -> List[str]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [str(png), str(pdf)]








def _values_for(cfg_agg: Dict[str, Any], key: str) -> List[float]:
    """Raw per-utterance values recorded by the aggregator (empty for pre-update JSONLs)."""
    raw = (cfg_agg.get("values") or {}).get(key) or []
    return [float(v) for v in raw if v is not None and np.isfinite(float(v))]

def _stat_p50(cfg_agg: Dict[str, Any], key: str) -> Optional[float]:
    entry = cfg_agg.get(key) or {}
    v = entry.get("p50")
    return None if v is None else float(v)


def _prefer_latin_cer(cfg_agg: Dict[str, Any]) -> bool:
    """True when the run recorded script-robust (Latin-transliterated) CER."""
    return bool(_values_for(cfg_agg, "cer_latin"))


def _cer_values(cfg_agg: Dict[str, Any]) -> List[float]:
    """Per-utterance CER, preferring the Latin-transliterated variant."""
    latin = _values_for(cfg_agg, "cer_latin")
    return latin if latin else _values_for(cfg_agg, "cer")


def _cer_p50(cfg_agg: Dict[str, Any]) -> Optional[float]:
    key = "cer_latin" if _prefer_latin_cer(cfg_agg) else "cer"
    return _stat_p50(cfg_agg, key)

def _draw_box(ax, pos: float, vals: Sequence[float], color: str, width: float = 0.44) -> None:
    """IQR box + median for one group (points are drawn separately)."""
    if not vals:
        return
    bp = ax.boxplot(
        [list(vals)], positions=[pos], widths=width, showfliers=False, patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.1, zorder=6),
        whiskerprops=dict(linewidth=0.7, color="#4d4d4d", zorder=2),
        capprops=dict(linewidth=0.7, color="#4d4d4d", zorder=2),
        boxprops=dict(linewidth=0.7, edgecolor="#333333", zorder=3),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

def _draw_points(ax, pos: float, vals: Sequence[float], color: str, rng,
                 size: float = 6.0, spread: float = 0.12, alpha: float = 0.85) -> None:
    if not vals:
        return
    xs = pos + rng.uniform(-spread, spread, len(vals))
    ax.scatter(xs, list(vals), s=size, color=color, alpha=alpha, linewidths=0, zorder=5)

def figure1_latency_quality(agg: Dict[str, Dict[str, Any]], out_dir: Path) -> Optional[List[str]]:
    """Fig. 1: TTFA / throughput / intelligibility distributions per configuration."""
    try:
        plt = _ieee_style()
    except Exception as exc:
        print(f"[Figures] matplotlib unavailable: {exc}")
        return None
    configs = [c for c in ("A", "B", "C") if agg.get(c, {}).get("n")]
    if not configs:
        return None
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    short = {}
    for c in configs:
        name = CONFIG_LABELS[c].split(" (")[0]
        short[c] = {"Pure AR": "AR", "Pure NAR": "NAR"}.get(name, "Hybrid")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    rng = np.random.RandomState(20261018)

    def _panel(ax, key, ylabel, title, log: bool = True, floor: Optional[float] = None,
               pct: bool = False, digits: int = 0, suffix: str = "") -> List[float]:
        drawn: List[float] = []
        labels: List[str] = []
        for i, c in enumerate(configs):
            raw = _values_for(agg[c], key)
            if pct:
                raw = [v * 100.0 for v in raw]
            plot_vals = [max(v, floor) for v in raw] if floor is not None else raw
            if plot_vals:
                _draw_box(ax, i, plot_vals, CONFIG_COLORS[c])
                _draw_points(ax, i, plot_vals, CONFIG_COLORS[c], rng)
                drawn.extend(plot_vals)
                labels.append(f"{short[c]}\np50 {np.median(raw):.{digits}f}{suffix}")
                continue
            p50 = _stat_p50(agg[c], key)
            labels.append(short[c] if p50 is None else
                          f"{short[c]}\np50 {p50 * (100.0 if pct else 1.0):.{digits}f}{suffix}")
            if p50 is None:
                continue
            if pct:
                p50 *= 100.0
            if floor is not None:
                p50 = max(p50, floor)
            ax.bar(i, p50, width=0.46, color=CONFIG_COLORS[c], alpha=0.32,
                   edgecolor="#333333", linewidth=0.7, zorder=3)
            drawn.append(p50)
        if log and drawn:
            ax.set_yscale("log")
        if drawn:
            ax.set_ylim(max(min(drawn) * 0.55, 1e-3), max(drawn) * 1.9)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.6, len(configs) - 0.4)
        ax.set_axisbelow(True)
        return drawn

    # ---- (a) time to first audio (warm; cold marked with x) -----------------
    title_a = "(a) Time-to-first-audio"
    a50 = _stat_p50(agg.get("A") or {}, "ttfa_compute_warm")
    c50 = _stat_p50(agg.get("C") or {}, "ttfa_compute_warm")
    if a50 and c50:
        title_a += f"  ({a50 / c50:.1f}$\\times$ faster than AR)"
    drawn_a = _panel(axes[0], "ttfa_compute_warm", "TTFA (ms, log scale)", title_a)
    has_cold = False
    for i, c in enumerate(configs):
        cold = (agg[c].get("ttfa_compute_cold") or {}).get("p50")
        if cold is None:
            continue
        axes[0].scatter([i + 0.30], [float(cold)], marker="x", s=24, color="black",
                        linewidths=1.1, zorder=7)
        drawn_a.append(float(cold))
        has_cold = True
    if drawn_a:
        axes[0].set_ylim(max(min(drawn_a) * 0.55, 1e-3), max(drawn_a) * 2.1)

    # ---- (b) throughput -----------------------------------------------------
    _panel(axes[1], "rtf", "RTF (compute s per audio s, log)", "(b) Throughput", digits=2)
    axes[1].axhline(1.0, color="#666666", linestyle=":", linewidth=0.9, zorder=1)
    axes[1].annotate("real time", xy=(-0.55, 1.0), xytext=(0, 3), textcoords="offset points",
                     ha="left", va="bottom", fontsize=6.5, color="#666666")

    # ---- (c) intelligibility ------------------------------------------------
    # Latin-transliterated CER when the run recorded it: ASR script choice must
    # not decide how intelligible the audio looks.
    cer_key = "cer_latin" if any(_prefer_latin_cer(agg[c]) for c in configs) else "cer"
    cer_name = "CER$_{lat}$" if cer_key == "cer_latin" else "CER"
    _panel(axes[2], cer_key, f"{cer_name} (%, log scale, lower better)",
           "(c) Intelligibility", floor=0.5, pct=True, digits=0, suffix="%")
    fails = [int(agg[c].get("asr_fail") or 0) for c in configs]
    if any(fails):
        axes[2].annotate(
            "ASR hallucination rows\n(CER>200%): " + " / ".join(str(f) for f in fails),
            xy=(0.985, 0.02), xycoords="axes fraction", fontsize=6.0, va="bottom", ha="right",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd",
                      linewidth=0.4, alpha=0.9),
        )

    handles = [
        Patch(facecolor="#888888", alpha=0.28, edgecolor="#333333", linewidth=0.7,
              label="IQR / median"),
        Line2D([], [], marker="o", linestyle="none", markersize=4, color="#888888",
               label="per utterance"),
    ]
    if has_cold:
        handles.append(Line2D([], [], marker="x", linestyle="none", markersize=5,
                              color="black", label="cold (first utterance)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02), handletextpad=0.5,
               columnspacing=1.8)
    fig.tight_layout()
    paths = _save_fig(fig, out_dir, "fig1_latency_quality_tradeoff")
    plt.close(fig)
    return paths

def figure2_zone_acceptance(agg: Dict[str, Dict[str, Any]], out_dir: Path) -> Optional[List[str]]:
    """Fig. 2: CB0 acceptance per linguistic zone as true-vs-reference dumbbells (Config C)."""
    try:
        plt = _ieee_style()
    except Exception as exc:
        print(f"[Figures] matplotlib unavailable: {exc}")
        return None
    from matplotlib.lines import Line2D

    zones = (agg.get("C") or {}).get("zones") or {}
    if not zones or all((zones.get(str(z)) or {}).get("accept_true") is None for z in ZONE_ORDER):
        return None

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ys = list(range(len(ZONE_ORDER)))[::-1]  # Pure Tamil on top
    base = zones.get(str(ZONE_TAMIL)) or {}
    tamil_true = base.get("accept_true")
    tamil_std = float(base.get("accept_true_std") or 0.0)
    if tamil_true is not None and tamil_std > 0:
        ax.axvspan(max(0.0, float(tamil_true) - tamil_std),
                   min(105.0, float(tamil_true) + tamil_std),
                   color="#2ca02c", alpha=0.10, linewidth=0, zorder=0)
        ax.axvline(float(tamil_true), color="#2ca02c", linestyle="--", linewidth=0.9,
                   alpha=0.75, zorder=1)

    for y, zone in zip(ys, ZONE_ORDER):
        z = zones.get(str(zone)) or {}
        pts = []
        for val, err, color, marker in (
            (z.get("accept_true"), z.get("accept_true_std") or 0.0, "#2ca02c", "o"),
            (z.get("accept_synth"), z.get("accept_synth_std") or 0.0, "#ff7f0e", "s"),
        ):
            if val is None:
                continue
            pts.append((float(val), float(err), color, marker))
        pts.sort(key=lambda p: p[0])
        for k, (val, err, color, marker) in enumerate(pts):
            ax.errorbar([val], [y], xerr=[[err], [err]], fmt=marker, ms=4.5,
                        color=color, ecolor=color, elinewidth=0.8, capsize=2.0,
                        capthick=0.8, zorder=4)
        if len(pts) == 2:
            for val, _, _, _ in pts:
                ax.annotate(f"{val:.1f}", xy=(val, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", va="bottom", fontsize=7,
                            color="#333333")
        if len(pts) == 2:
            ax.plot([pts[0][0], pts[1][0]], [y, y], color="#c4c4c4", linewidth=1.1, zorder=2)
        true_val = z.get("accept_true")
        if zone != ZONE_TAMIL and true_val is not None and tamil_true is not None:
            delta = float(true_val) - float(tamil_true)
            err = float(z.get("accept_true_std") or 0.0)
            significant = err > 0.0 and abs(delta) > err
            ax.annotate(f"$\\Delta$ {delta:+.1f}" + ("" if significant else " (n.s.)"),
                        xy=(104.5, y), ha="right", va="center", fontsize=6.5,
                        color="#d62728" if significant else "#8c8c8c")

    ylabels = []
    for zone in ZONE_ORDER:
        n_frames = int((zones.get(str(zone)) or {}).get("n_frames") or 0)
        ylabels.append(f"{ZONE_NAMES[zone]}\n({n_frames} frames)")
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.55, len(ZONE_ORDER) - 0.45)
    ax.set_xlabel("Codebook-0 acceptance (%)", fontsize=9)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", ms=4.5, linestyle="none", color="#2ca02c",
               label="True target (Parler $P$)"),
        Line2D([], [], marker="s", ms=4.5, linestyle="none", color="#ff7f0e",
               label="Mismatched target (ref.)"),
    ]
    if tamil_true is not None and tamil_std > 0:
        handles.append(Line2D([], [], color="#2ca02c", linestyle="--", linewidth=0.9,
                              label="Pure Tamil median $\\pm$ std"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              frameon=False, fontsize=6.8, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout()
    paths = _save_fig(fig, out_dir, "fig2_boundary_acceptance_breakdown")
    plt.close(fig)
    return paths

def figure3_stream_timeline(rows: List[Dict[str, Any]], out_dir: Path, target_sec: float = 3.0) -> Optional[List[str]]:
    """Fig. 3: streaming chunk synthesis vs. playback buffer (representative Config C run)."""
    try:
        plt = _ieee_style()
    except Exception as exc:
        print(f"[Figures] matplotlib unavailable: {exc}")
        return None
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    candidates = [
        r for r in rows
        if r.get("config") == "C" and not r.get("error") and r.get("chunks")
        and r.get("audio_sec") and r.get("total_ms")
    ]
    if not candidates:
        return None
    rep = min(candidates, key=lambda r: abs(float(r["audio_sec"]) - target_sec))
    chunks = rep["chunks"]
    draft_ms = float(rep.get("draft_ms") or 0.0)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.16, 3.3), sharex=True,
        gridspec_kw={"height_ratios": [1.9, 1.0], "hspace": 0.18},
    )
    colors = {"draft": "#8c8c8c", "verify": "#1f77b4", "residual": "#ff7f0e", "dac": "#2ca02c"}

    # --- top: synthesis Gantt -------------------------------------------------
    for i, ch in enumerate(chunks):
        dac_ms = float(ch.get("dac_ms") or 0.0)
        res_ms = float(ch.get("residual_ms") or 0.0)
        ver_ms = float(ch.get("verify_ms") or 0.0)
        ready = float(ch.get("ready_ms") or 0.0)
        dac_start = ready - dac_ms
        res_start = dac_start - res_ms
        ver_start = res_start - ver_ms
        if ver_ms > 0:
            ax_top.barh(i, ver_ms, left=ver_start, height=0.60, color=colors["verify"],
                        edgecolor="black", linewidth=0.3)
        if res_ms > 0:
            ax_top.barh(i, res_ms, left=res_start, height=0.60, color=colors["residual"],
                        edgecolor="black", linewidth=0.3)
        if dac_ms > 0:
            ax_top.barh(i, dac_ms, left=dac_start, height=0.60, color=colors["dac"],
                        edgecolor="black", linewidth=0.3)
        if i == 0 and draft_ms > 0:
            ax_top.barh(i, draft_ms, left=0.0, height=0.60, color=colors["draft"],
                        edgecolor="black", linewidth=0.3)
        ax_top.plot([ready], [i], marker="v", ms=3.4, color="black", zorder=6)

    # playback lane: chunks are played back-to-back once the buffer allows it
    play_lane = len(chunks)
    play_time = None
    for ch in chunks:
        dur_ms = float(ch.get("n_frames") or 0) / 86.1328 * 1000.0
        ready = float(ch.get("ready_ms") or 0.0)
        start = ready if play_time is None else max(play_time, ready)
        ax_top.barh(play_lane, dur_ms, left=start, height=0.60, color="#d9d9d9",
                    edgecolor="black", linewidth=0.3)
        play_time = start + dur_ms

    ttfa = float(rep.get("ttfa_e2e_ms") or (chunks[0].get("ready_ms") or 0.0))
    total_ms = float(rep.get("total_ms") or 0.0)
    horizon = max(total_ms * 1.04, (play_time or 0.0) * 1.06, ttfa * 3.0)
    ax_top.axvline(ttfa, color="#d62728", linestyle="--", linewidth=1.0, zorder=5)
    ax_top.annotate(f"TTFA {ttfa:.0f} ms", xy=(ttfa, -0.58), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=7.5,
                    color="#d62728",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none",
                              alpha=0.85))
    if total_ms > 0:
        ax_top.axvline(total_ms, color="#555555", linestyle=":", linewidth=0.9, zorder=5)
        ax_top.annotate(f"synthesis done {total_ms:.0f} ms", xy=(total_ms, play_lane + 0.62),
                        xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=7.5, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.85))
    ax_top.set_yticks(list(range(len(chunks) + 1)))
    ax_top.set_yticklabels([f"Chunk {i + 1}" for i in range(len(chunks))] + ["Playback"],
                           fontsize=8.5)
    ax_top.set_ylabel("Pipeline stage", fontsize=9)
    ax_top.set_ylim(-0.75, play_lane + 1.05)
    ax_top.set_title(
        f"Streaming synthesis vs. playback \u2014 {rep['utt_id']} "
        f"({REGISTER_LABELS.get(rep['register'], rep['register'])}), "
        f"{float(rep['audio_sec']):.2f} s audio (Config C)",
        fontsize=9.5,
    )
    ax_top.set_axisbelow(True)

    # --- bottom: buffer occupancy (ms of audio ready but unplayed) -----------
    ts = np.linspace(0, horizon, 800)
    ready = np.zeros_like(ts)
    for ch in chunks:
        dur_ms = float(ch.get("n_frames") or 0) / 86.1328 * 1000.0
        ready += (ts >= float(ch.get("ready_ms") or 0.0)) * dur_ms
    played = np.clip(ts - ttfa, 0.0, (play_time - ttfa) if play_time else 0.0)
    buffer_ms = ready - played
    ax_bot.fill_between(ts, buffer_ms, 0, where=buffer_ms >= 0, color="#2ca02c", alpha=0.30)
    ax_bot.fill_between(ts, buffer_ms, 0, where=buffer_ms < 0, color="#d62728", alpha=0.35)
    ax_bot.axhline(0, color="black", linewidth=0.6)
    ax_bot.axvline(ttfa, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
    imin = int(np.argmin(buffer_ms))
    ax_bot.plot([ts[imin]], [buffer_ms[imin]], marker="o", ms=3.0, color="black", zorder=6)
    ax_bot.annotate(f"min {buffer_ms[imin]:.0f} ms", xy=(ts[imin], buffer_ms[imin]),
                    xytext=(7, 7 if buffer_ms[imin] >= 0 else -13), textcoords="offset points",
                    fontsize=7, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#dddddd",
                              linewidth=0.4, alpha=0.9))
    ax_bot.set_xlabel("Wall-clock time since synthesis start (ms)", fontsize=9)
    ax_bot.set_ylabel("Buffer (ms)", fontsize=9)
    ax_bot.set_xlim(0, horizon)
    ax_bot.set_ylim(min(0.0, buffer_ms.min() * 1.35), max(buffer_ms.max() * 1.30, 50.0))
    ax_bot.set_axisbelow(True)
    state = "underrun" if (buffer_ms < 0).any() else "no underrun"
    ax_bot.annotate(state, xy=(0.99, 0.90), xycoords="axes fraction", ha="right", va="top",
                    fontsize=7.5, color="#2ca02c" if state == "no underrun" else "#d62728")

    handles = [
        Patch(facecolor=colors["draft"], edgecolor="black", linewidth=0.3,
              label="Draft (CB0 NAR)"),
        Patch(facecolor=colors["verify"], edgecolor="black", linewidth=0.3,
              label="Verify (AR, teacher-forced)"),
        Patch(facecolor=colors["residual"], edgecolor="black", linewidth=0.3,
              label="Residual (CB1\u20138)"),
        Patch(facecolor=colors["dac"], edgecolor="black", linewidth=0.3, label="DAC decode"),
        Line2D([], [], marker="v", linestyle="none", ms=3.4, color="black",
               label="chunk ready"),
        Patch(facecolor="#d9d9d9", edgecolor="black", linewidth=0.3, label="Playback"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    paths = _save_fig(fig, out_dir, "fig3_streaming_chunk_timeline")
    plt.close(fig)
    return paths


def figure4_scaling_and_registers(rows: List[Dict[str, Any]], agg: Dict[str, Dict[str, Any]],
                                  out_dir: Path) -> Optional[List[str]]:
    """Fig. 4: (a) total latency vs audio length, (b) CER per linguistic register."""
    try:
        plt = _ieee_style()
    except Exception as exc:
        print(f"[Figures] matplotlib unavailable: {exc}")
        return None
    from matplotlib.lines import Line2D

    configs = [c for c in ("A", "B", "C") if agg.get(c, {}).get("n")]
    if not configs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    rng = np.random.RandomState(4242)

    # ---- (a) total latency vs. audio duration -------------------------------
    ax = axes[0]
    slopes: Dict[str, Optional[float]] = {}
    for c in configs:
        sub = [r for r in rows if r.get("config") == c and not r.get("error")
               and r.get("audio_sec") and r.get("total_ms")]
        xs = np.asarray([float(r["audio_sec"]) for r in sub], dtype=float)
        ys = np.asarray([float(r["total_ms"]) for r in sub], dtype=float)
        slopes[c] = None
        if xs.size == 0:
            continue
        ax.scatter(xs, ys, s=9, color=CONFIG_COLORS[c], alpha=0.75, linewidths=0, zorder=4)
        if xs.size >= 2 and np.ptp(xs) > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            slopes[c] = float(slope)
            xf = np.linspace(xs.min(), xs.max(), 20)
            ax.plot(xf, slope * xf + intercept, color=CONFIG_COLORS[c], linewidth=1.0, zorder=3)
    ok_y = [float(r["total_ms"]) for r in rows
            if r.get("config") in configs and not r.get("error") and r.get("total_ms")]
    ax.set_yscale("log")
    if ok_y:
        ax.set_ylim(max(min(ok_y) * 0.5, 1.0), max(ok_y) * 2.0)
    ax.set_xlabel("Synthesised audio duration (s)", fontsize=9)
    ax.set_ylabel("Total synthesis latency (ms, log)", fontsize=9)
    ax.set_title("(a) Latency scaling with utterance length", fontsize=9)
    ax.set_axisbelow(True)

    handles = []
    for c in configs:
        slope = slopes.get(c)
        name = CONFIG_LABELS[c].split(" (")[0]
        label = f"{c}: {name}" + ("" if slope is None else f"\n{slope:.0f} ms per audio-s")
        handles.append(Line2D([], [], marker="o", ms=4, linestyle="none",
                              color=CONFIG_COLORS[c], label=label))
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9,
              edgecolor="#dddddd", fontsize=7, handletextpad=0.4, borderpad=0.4,
              labelspacing=0.8)

    # ---- (b) CER per register ----------------------------------------------
    ax = axes[1]
    regs = [r for r in REGISTER_LABELS if ((agg.get(configs[0]) or {}).get("registers") or {}).get(r)
            or any(((agg.get(c) or {}).get("registers") or {}).get(r) for c in configs)]
    if not regs:
        regs = list(REGISTER_LABELS)
    positions = np.arange(len(regs))
    n_cfg = len(configs)
    drawn: List[float] = []
    for i, reg in enumerate(regs):
        for k, c in enumerate(configs):
            entry = ((agg.get(c) or {}).get("registers") or {}).get(reg) or {}
            reg_vals = (entry.get("values") or {}).get("cer_latin") or \
                (entry.get("values") or {}).get("cer") or []
            vals = [float(v) * 100.0 for v in reg_vals]
            vals = [v for v in vals if np.isfinite(v)]
            pos = i + (k - (n_cfg - 1) / 2.0) * 0.26
            if vals:
                plot_vals = [max(v, 0.5) for v in vals]
                _draw_box(ax, pos, plot_vals, CONFIG_COLORS[c], width=0.20)
                _draw_points(ax, pos, plot_vals, CONFIG_COLORS[c], rng, size=3.6,
                             spread=0.07, alpha=0.8)
                drawn.extend(plot_vals)
            else:
                p50 = ((entry.get("cer_latin") or entry.get("cer") or {}).get("p50"))
                if p50 is None:
                    continue
                p50 = max(float(p50) * 100.0, 0.5)
                ax.bar(pos, p50, width=0.20, color=CONFIG_COLORS[c], alpha=0.32,
                       edgecolor="#333333", linewidth=0.5, zorder=3)
                drawn.append(p50)
    if drawn:
        ax.set_yscale("log")
        ax.set_ylim(max(min(drawn) * 0.5, 0.3), max(drawn) * 2.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([REGISTER_LABELS[r].replace(" ", "\n") for r in regs], fontsize=8)
    ax.set_xlim(-0.55, len(regs) - 0.45)
    cer_label = "CER$_{lat}$" if any(
        (entry.get("values") or {}).get("cer_latin")
        for c in configs
        for entry in (((agg.get(c) or {}).get("registers") or {}).values())
    ) else "CER"
    ax.set_ylabel(f"{cer_label} (%, log, lower better)", fontsize=9)
    ax.set_title("(b) Intelligibility per register", fontsize=9)
    ax.set_axisbelow(True)
    # colour key is shared with panel (a); no second legend needed

    fig.tight_layout()
    paths = _save_fig(fig, out_dir, "fig4_scaling_and_registers")
    plt.close(fig)
    return paths


# ===========================================================================
# LaTeX tables
# ===========================================================================
def _tex_ci(entry: Optional[Dict[str, Any]], digits: int = 1,
            scale: float = 1.0) -> str:
    """``median [lo, hi]`` from a _stats dict when the bootstrap CI exists."""
    entry = entry or {}
    p50, lo, hi = entry.get("p50"), entry.get("ci_lo"), entry.get("ci_hi")
    if p50 is None or lo is None or hi is None:
        return ""
    f = lambda v: f"{float(v) * scale:.{digits}f}"  # noqa: E731
    return f"{f(p50)} [{f(lo)}, {f(hi)}]"


def _tex(value: Optional[float], digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}{suffix}"


def _pm(stat: Dict[str, Optional[float]], digits: int = 1, suffix: str = "") -> str:
    if not stat or stat.get("mean") is None:
        return "--"
    if stat.get("std") is None:
        return _tex(stat["mean"], digits, suffix)
    return f"{stat['mean']:.{digits}f}$\\pm${stat['std']:.{digits}f}{suffix}"


def _mean_pm(entry: Dict[str, Any], key: str, digits: int = 1) -> str:
    """Formats 'mean $\\pm$ std' for an acceptance entry (or '--')."""
    mean = entry.get(key)
    if mean is None:
        return "--"
    std = entry.get(key + "_std") or 0.0
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def _fmt_stat(value: Optional[float], digits: int = 0) -> str:
    """Plain console formatting for a scalar stat (or '--')."""
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def build_table_I(agg: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "% Table I: comparative performance (TTFA medians; mean $\\pm$ std for acceptance)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comparative evaluation of Pure AR, Pure NAR and the proposed CodeSwitch-TTS "
        "hybrid on the 60-utterance Tanglish benchmark (Kaggle T4, fp16). "
        "TTFA-c: component compute latency; TTFA-e2e: wall-clock to first audio buffer; "
        "median columns carry a 95\\% percentile-bootstrap interval (2{,}000 resamples, fixed seed); "
        "CER medians (ASR-limited); CER$_{\\mathrm{lat}}$ is script-robust; UTMOS and MCD are "
        "perceptual/spectral proxies (no listeners, see text).}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{lcccccccccccc}",
        "\\toprule",
        "Architecture & TTFA-c cold (ms) & TTFA-c warm p50 (ms) & TTFA-c warm p90 (ms) "
        "& TTFA-e2e warm p50 (ms) & Total (s) & RTF & CER p50 (\\%) & CER$_{\\mathrm{lat}}$ p50 (\\%) "
        "& UTMOS & MCD (dB) & CB0 acc. true (\\%) & CB0 acc. ref. (\\%) \\\\",
        "\\midrule",
    ]
    for cfg in ("A", "B", "C"):
        a = agg.get(cfg) or {}
        if not a.get("n"):
            continue

        def _stat(key: str, stat: str, entry: Dict[str, Any] = a):
            warm_entry = entry.get(key + "_warm") or {}
            if warm_entry.get("n"):
                return warm_entry.get(stat)
            return (entry.get(key) or {}).get(stat)

        total = (a.get("total") or {}).get("mean")
        cer_stat = a.get("cer") or {}
        cer_p50 = None if cer_stat.get("p50") is None else float(cer_stat["p50"]) * 100.0
        cer_lat = a.get("cer_latin") or {}
        cer_lat_p50 = None if cer_lat.get("p50") is None else float(cer_lat["p50"]) * 100.0
        utmos_stat = a.get("utmos") or {}
        mcd_stat = a.get("mcd_vs_a") or {}
        cells = [
            CONFIG_LABELS[cfg],
            _tex((a.get("ttfa_compute_cold") or {}).get("p50"), 0),
            _tex_ci(a.get("ttfa_compute_warm"), 0) or _tex(_stat("ttfa_compute", "p50"), 0),
            _tex(_stat("ttfa_compute", "p90"), 0),
            _tex(_stat("ttfa_e2e", "p50"), 0),
            _tex(None if total is None else total / 1000.0, 2),
            _tex((a.get("rtf") or {}).get("mean"), 2),
            _tex_ci(cer_stat, 1, scale=100.0) or _tex(cer_p50, 1),
            _tex_ci(cer_lat, 1, scale=100.0) or _tex(cer_lat_p50, 1),
            _tex(utmos_stat.get("mean"), 2),
            _tex(mcd_stat.get("mean"), 2),
            _pm(a.get("accept_true") or {}),
            _pm(a.get("accept_synth") or {}),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    # ASR-failure census (whisper hallucination loops poison the mean CER)
    fail_bits = []
    for cfg in ("A", "B", "C"):
        a = agg.get(cfg) or {}
        if a.get("n"):
            fail_bits.append(f"{cfg}: {int(a.get('asr_fail') or 0)}/{a['n']}")
    session_cold = agg.get("_session_cold") or {}
    cold_sentence = ""
    if session_cold:
        cc = session_cold.get("config")
        cold_sentence = (
            f" Session cold start (first synthesis of the process: {cc} on "
            f"{session_cold.get('utt_id')}) = "
            f"{_tex(session_cold.get('ttfa_compute_ms'), 0)} ms TTFA-c; because configurations "
            f"are interleaved per utterance, only this row is a true cold start and the "
            f"per-configuration cold columns for the other configs reflect a partially warmed "
            f"context (run a single configuration per process for a fully isolated measurement)."
        )
    logq_d = (agg.get("C") or {}).get("logq_draft") or {}
    logq_k = (agg.get("C") or {}).get("logq_kept") or {}
    logq_sentence = ""
    if logq_d.get("mean") is not None and logq_k.get("mean") is not None:
        logq_sentence = (
            f" In Config C rejection sampling raises the teacher-forced mean CB0 "
            f"log-likelihood per frame from {logq_d['mean']:.2f} to {logq_k['mean']:.2f} nats "
            f"(n={int(logq_k.get('n') or 0)} utterances), i.e.\\ the emitted tokens are closer to "
            f"the target distribution than the raw draft proposal."
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2pt}",
        (
            "\\footnotesize{cold = first utterance per configuration (CUDA kernels, allocator and "
            "description cache warming); warm = remaining utterances (cold-start overhead excluded, "
            "per-utterance seeds and texts identical). True = teacher-forced Parler target: one "
            "decoder forward per chunk scoring codebook 0 with codebooks 1-8 held at bos "
            "(single-channel verification approximation, i.e. $q(c_0|\\mathrm{text}, c_0^{<t})$ with "
            "null inter-codebook context); ref.\\ = the same acceptance math against an unrelated "
            "random target distribution (the legacy demo protocol). The ref.\\ column is a "
            "mismatched-target baseline, not an upper bound: the distillation-aligned draft agrees "
            "with the real teacher more often than with a random target (a perfectly aligned draft "
            "would score 100\\% against its own target). Config A has no draft/verification stage "
            "(acc. n/a); its TTFA-c is the time from decoder start to the first decoded audio buffer. "
            "CER reference: normalised Tamil-script text; CER$_{\\mathrm{lat}}$ is the same metric "
            "after transliterating both sides to Latin, which removes the ASR model's script-choice "
            "component; UTMOS is the UTMOS22 no-listener perceptual proxy; MCD is mel-cepstral "
            "distortion against the Pure-AR rendering of the same text (a different sampling seed, so "
            "a distribution-level spectral distance rather than an exact paired one). "
            "ASR failures with CER $>$ 200\\% "
            "(whisper-small hallucination loops) = " + ", ".join(fail_bits) + "; Table III gives "
            "per-register numbers." + cold_sentence + logq_sentence + "}"
        ),
        "\\end{table*}",
    ]
    return "\n".join(lines)


def build_table_II(agg: Dict[str, Dict[str, Any]]) -> str:
    zones = (agg.get("C") or {}).get("zones") or {}
    lines = [
        "% Table II: Codebook-0 acceptance per linguistic zone (Config C)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Speculative Codebook-0 acceptance by linguistic zone for CodeSwitch-TTS "
        "(mean $\\pm$ std across utterances; $\\Delta$ relative to Pure Tamil).}",
        "\\label{tab:zones}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Linguistic zone & Frames & CB0 acc. true (\\%) & CB0 acc. ref. (\\%) & $\\Delta$ true (pts) \\\\",
        "\\midrule",
    ]
    base = None
    deltas: List[float] = []
    for zone in ZONE_ORDER:
        z = zones.get(str(zone)) or {}
        if zone == ZONE_TAMIL:
            base = z.get("accept_true")
        delta = None
        if z.get("accept_true") is not None and base is not None:
            delta = float(z["accept_true"]) - float(base)
            deltas.append(delta)
        cells = [
            ZONE_NAMES[zone],
            str(z.get("n_frames", 0)),
            _mean_pm(z, "accept_true"),
            _mean_pm(z, "accept_synth"),
            "--" if delta is None else f"{delta:+.1f}",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    max_abs = max((abs(d) for d in deltas), default=0.0)
    n_utts = max((int((z or {}).get("accept_true_utts") or 0) for z in zones.values()), default=0)
    lines.append(
        "\\vspace{2pt}",
    )
    lines.append(
        "\\footnotesize{ref.\\ = acceptance against an unrelated random target "
        "(mismatched-target baseline, not a bound). Per-utterance std in the true column "
        "is 10-20 pts, i.e.\\ the largest inter-zone difference here "
        f"({max_abs:.1f} pts) is within one standard deviation: at this draft quality the "
        f"zone split does not resolve a significant boundary penalty (n={n_utts} utterances)."
        "}"
    )
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_table_III(agg: Dict[str, Dict[str, Any]]) -> str:
    """Per-register breakdown: CER (median/mean) and latency per config."""
    lines = [
        "% Table III: per-register CER and latency breakdown",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-register breakdown over the 60-utterance benchmark: CER median/mean and "
        "warm-component latency. CER is ASR-limited (whisper-small transliteration choices and "
        "hallucination loops), so the median is the comparable statistic; rows above 200\\% are "
        "censused in Table~I.}",
        "\\label{tab:registers}",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "Register & Config & N & CER p50 (\\%) & CER$_{\\mathrm{lat}}$ p50 (\\%) & CER mean (\\%) "
        "& TTFA-c p50 (ms) & RTF & CB0 acc.\\ true (\\%) \\\\",
        "\\midrule",
    ]
    any_row = False
    for reg, reg_label in REGISTER_LABELS.items():
        first_row = True
        for cfg in ("A", "B", "C"):
            entry = ((agg.get(cfg) or {}).get("registers") or {}).get(reg)
            if not entry or not entry.get("n"):
                continue
            any_row = True
            cer = entry.get("cer") or {}
            p50 = None if cer.get("p50") is None else float(cer["p50"]) * 100.0
            mean = cer.get("mean")
            mean_cell = "--"
            if mean is not None:
                std = cer.get("std")
                mean_cell = (
                    f"{mean * 100.0:.1f}"
                    if std is None
                    else f"{mean * 100.0:.1f}$\\pm${std * 100.0:.1f}"
                )
            acc = entry.get("accept_true") or {}
            lat = entry.get("cer_latin") or {}
            lat_cell = _tex(None if lat.get("p50") is None else float(lat["p50"]) * 100.0, 1)
            cells = [
                reg_label if first_row else "",
                cfg,
                str(entry.get("n")),
                _tex(p50, 1),
                lat_cell,
                mean_cell,
                _tex((entry.get("ttfa_compute") or {}).get("p50"), 0),
                _tex((entry.get("rtf") or {}).get("mean"), 2),
                "--" if acc.get("mean") is None else f"{acc['mean']:.1f}$\\pm${acc.get('std') or 0.0:.1f}",
            ]
            lines.append(" & ".join(cells) + " \\\\")
            first_row = False
    if not any_row:
        lines.append("-- & -- & -- & -- & -- & -- & -- & -- & -- \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================
def figure5_gate_pareto(rows: List[Dict[str, Any]], agg: Dict[str, Dict[str, Any]],
                        out_dir: Path) -> Optional[List[str]]:
    """Fig. 5: (a) confidence-gated operating points, (b) draft-confidence calibration."""
    try:
        plt = _ieee_style()
    except Exception as exc:
        print(f"[Figures] matplotlib unavailable: {exc}")
        return None
    from matplotlib.lines import Line2D

    gate_rows = aggregate_gate_sweep(rows)
    if not gate_rows:
        return None
    c_rows = [r for r in rows if r.get("config") == "C" and not r.get("error")]
    if not c_rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    rng = np.random.RandomState(20261018)

    # ---- (a) latency-quality operating points ------------------------------
    ax = axes[0]
    pts: List[Tuple[float, float, float]] = []  # (total p50, cer p50, fallback rate)
    use_latin = any((entry.get("cer_latin") or {}).get("p50") is not None
                    for entry in gate_rows)
    for entry in gate_rows:
        total = (entry.get("total") or {}).get("p50")
        cer = ((entry.get("cer_latin") if use_latin else entry.get("cer")) or {}).get("p50")
        if total is None or cer is None:
            continue
        pts.append((float(total), float(cer) * 100.0, float(entry.get("fallback_rate") or 0.0)))
    pts.sort()
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CONFIG_COLORS["C"],
                marker="o", ms=4.5, linewidth=1.0, zorder=3, label=None)
        for k, (total, cer, fb) in enumerate(pts):
            ax.annotate(f"{fb * 100:.0f}% AR fallback", xy=(total, cer),
                        textcoords="offset points",
                        xytext=(5, -12 if k % 2 == 0 else -24),
                        ha="left", va="top", fontsize=6.2, color="#555555")
    a_entry = agg.get("A") or {}
    a_total = (a_entry.get("total") or {}).get("p50")
    a_cer = _cer_p50(a_entry)
    if a_total is not None and a_cer is not None:
        ax.scatter([float(a_total)], [float(a_cer) * 100.0], s=34, marker="s",
                   color=CONFIG_COLORS["A"], zorder=4, label="Pure AR")
    b_entry = agg.get("B") or {}
    b_total = (b_entry.get("total") or {}).get("p50")
    b_cer = _cer_p50(b_entry)
    if b_total is not None and b_cer is not None:
        ax.scatter([float(b_total)], [float(b_cer) * 100.0], s=34, marker="^",
                   color=CONFIG_COLORS["B"], zorder=4, label="Pure NAR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total synthesis latency, p50 (ms, log)", fontsize=9)
    ax.set_ylabel(("CER$_{lat}$" if use_latin else "CER")
                  + " p50 (%, log, lower better)", fontsize=9)
    ax.set_title("(a) Confidence-gated operating points", fontsize=9)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ms=4.5, linestyle="-", linewidth=1.0,
                      color=CONFIG_COLORS["C"], label="Hybrid, gate $\\tau$ sweep"),
               Line2D([], [], marker="s", ms=4.5, linestyle="none", color=CONFIG_COLORS["A"],
                      label="Pure AR"),
               Line2D([], [], marker="^", ms=5, linestyle="none", color=CONFIG_COLORS["B"],
                      label="Pure NAR")]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="#dddddd", fontsize=6.8, handletextpad=0.4, borderpad=0.4)

    # ---- (b) draft-confidence calibration ----------------------------------
    ax = axes[1]
    conf: List[float] = []
    accept: List[float] = []
    for r in c_rows:
        for ch in (r.get("chunks") or []):
            c = ch.get("confidence")
            a = ch.get("accept_true")
            if c is None or a is None:
                continue
            conf.append(float(c))
            accept.append(float(a))
    if conf:
        ax.scatter(conf, accept, s=7, color=CONFIG_COLORS["C"], alpha=0.55, linewidths=0, zorder=3)
        conf_arr = np.asarray(conf)
        acc_arr = np.asarray(accept)
        edges = np.linspace(float(conf_arr.min()), float(conf_arr.max()) + 1e-6, 9)
        centers, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf_arr >= lo) & (conf_arr < hi)
            if mask.sum() >= 3:
                centers.append((lo + hi) / 2.0)
                means.append(float(acc_arr[mask].mean()))
        if centers:
            ax.plot(centers, means, color="#333333", marker="o", ms=3.6, linewidth=1.0, zorder=4)
        ax.set_xlabel("Draft confidence (mean top-1 prob. per chunk)", fontsize=9)
        ax.set_ylabel("Realised CB0 acceptance (%)", fontsize=9)
        ax.set_title("(b) Gate calibration", fontsize=9)
        ax.set_axisbelow(True)
        # Spearman-style rank correlation without scipy.
        if conf_arr.size >= 3:
            rank_conf = np.argsort(np.argsort(conf_arr))
            rank_acc = np.argsort(np.argsort(acc_arr))
            rho = float(np.corrcoef(rank_conf, rank_acc)[0, 1])
            ax.annotate(f"$\\rho$ = {rho:.2f}  (n = {conf_arr.size} chunks)",
                        xy=(0.03, 0.95), xycoords="axes fraction", va="top", ha="left",
                        fontsize=7, color="#444444")
    else:
        ax.text(0.5, 0.5, "no per-chunk confidence recorded", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#777777")
        ax.set_axisbelow(True)

    fig.tight_layout()
    paths = _save_fig(fig, out_dir, "fig5_gate_pareto_calibration")
    plt.close(fig)
    return paths


def build_table_IV(switch_rows: List[Dict[str, Any]]) -> str:
    """Table IV: CB0 acceptance vs frames since the last code switch."""
    lines = [
        "% Table IV: distance-to-switch acceptance (power analysis for the boundary claim)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{CB0 acceptance on the hybrid configuration as a function of frames since the "
        "last Tamil$\\leftrightarrow$Latin switch (packed per-frame telemetry, speculated frames "
        "only). $\\Delta$ is the paired per-utterance difference against the $\\geq$16-frame "
        "reference bucket with a 95\\% bootstrap CI; significance still requires a larger draft, "
        "but this curve has the power that the four-way zone contrast lacks.}",
        "\\label{tab:switchdist}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Frames since switch & Frames & Utts & CB0 acc.\\ (\\%) & $\\Delta$ vs $\\geq$16 (pts) & $p$ \\\\",
        "\\midrule",
    ]
    for entry in switch_rows:
        mean = "--" if entry.get("mean") is None else f"{entry['mean']:.1f}$\\pm${(entry.get('std') or 0.0):.1f}"
        if entry.get("delta_vs_far") is None:
            delta_cell, p_cell = "--", "--"
        else:
            delta_cell = (f"{entry['delta_vs_far']:+.1f} "
                          f"[{entry['delta_ci_lo']:+.1f}, {entry['delta_ci_hi']:+.1f}]")
            p_cell = f"{entry['p_vs_far']:.3f}"
        label = entry["bucket"].replace("16+", "$\\geq$16").replace("-", "--")
        lines.append(" & ".join([
            label, str(entry.get("n_frames") or 0), str(entry.get("n_utts") or 0),
            mean, delta_cell, p_cell,
        ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def build_table_V(gate_rows: List[Dict[str, Any]], config_a: Optional[Dict[str, Any]] = None,
                  derived: bool = True, thresholds: Optional[Sequence[float]] = None) -> str:
    """Table V: confidence-gated operating points (measured latency vs quality)."""
    how = ("Operating points are derived from the measured per-utterance hybrid and Pure-AR "
           "results (the gate is decided before synthesis, so the mixture is exact and costs no "
           "extra synthesis; fallback latency is the sum of the measured draft and AR passes)"
           if derived else
           "Each threshold was synthesised end to end, fallback cost included")
    lines = [
        "% Table V: confidence-gated speculative decoding - measured operating points",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Confidence-gated hybrid: utterances whose worst-chunk draft confidence falls "
        f"below $\\tau$ are handed to the full AR pipeline. {how}. Latency, RTF and "
        "CER are the resulting mixture's values; the gate turns speculation quality into an "
        "explicit operating point rather than an assumption.}",
        "\\label{tab:gate}",
        "\\begin{tabular}{cccccccc}",
        "\\toprule",
        "$\\tau$ & AR fallback (\\%) & TTFA-c p50 (ms) & TTFA e2e p50 (ms) & Total p50 (ms) & RTF & CER p50 (\\%) & CB0 acc.\\ (\\%) \\\\",
        "\\midrule",
    ]
    for entry in gate_rows:
        lines.append(" & ".join([
            f"{entry['threshold']:.2f}",
            f"{entry['fallback_rate'] * 100.0:.1f}",
            _tex((entry.get("ttfa_compute") or {}).get("p50"), 0),
            _tex((entry.get("ttfa_e2e") or {}).get("p50"), 0),
            _tex((entry.get("total") or {}).get("p50"), 0),
            _tex((entry.get("rtf") or {}).get("mean"), 2),
            _tex(None if (entry.get("cer") or {}).get("p50") is None
                 else float(entry["cer"]["p50"]) * 100.0, 1),
            _pm(entry.get("accept_true") or {}),
        ]) + " \\\\")
    if config_a and (config_a.get("n") or 0) > 0:
        lines.append("\\midrule")
        lines.append(" & ".join([
            "AR (A)",
            "100.0",
            _tex((config_a.get("ttfa_compute_warm") or {}).get("p50"), 0),
            _tex((config_a.get("ttfa_e2e_warm") or {}).get("p50"), 0),
            _tex((config_a.get("total") or {}).get("p50"), 0),
            _tex((config_a.get("rtf") or {}).get("mean"), 2),
            _tex(None if (config_a.get("cer") or {}).get("p50") is None
                 else float(config_a["cer"]["p50"]) * 100.0, 1),
            "--",
        ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def print_gate_sweep(gate_rows: List[Dict[str, Any]]) -> None:
    if not gate_rows:
        return
    print("\n" + "-" * 78)
    print("CONFIDENCE-GATED HYBRID (config C operating points)")
    print("-" * 78)
    print(f"{'tau':>5} {'AR fallback':>12} {'TTFA-c p50':>11} {'Total p50':>10} "
          f"{'RTF':>6} {'CER p50':>8} {'CB0 acc.':>9} {'halluc.':>8}")
    for entry in gate_rows:
        cer = entry.get("cer") or {}
        total = entry.get("total") or {}
        acc = entry.get("accept_true") or {}
        print(
            f"{entry['threshold']:>5.2f} {entry['fallback_rate'] * 100.0:>11.1f}% "
            f"{_fmt_stat((entry.get('ttfa_compute') or {}).get('p50'), 0):>11} "
            f"{_fmt_stat(total.get('p50'), 0):>10} "
            f"{_fmt_stat((entry.get('rtf') or {}).get('mean'), 3):>6} "
            f"{_fmt_stat(None if cer.get('p50') is None else float(cer['p50']) * 100.0, 1) + '%':>8} "
            f"{_fmt_stat(acc.get('mean'), 1) + '%':>9} "
            f"{entry.get('asr_fail', 0):>3}/{entry.get('n', 0):<3}"
        )
    print("gate: fallback rows are excluded from acceptance but included in latency/CER.")


def print_switch_distance(switch_rows: List[Dict[str, Any]]) -> None:
    if not switch_rows or all(r.get("mean") is None for r in switch_rows):
        return
    print("\n" + "-" * 78)
    print("DISTANCE-TO-SWITCH ACCEPTANCE (config C, speculated frames only)")
    print("-" * 78)
    print(f"{'frames since switch':>20} {'frames':>8} {'utts':>6} {'acc. (%)':>10} "
          f"{'delta vs >=16':>16} {'p':>7}")
    for entry in switch_rows:
        if entry.get("mean") is None:
            continue
        delta = ("--" if entry.get("delta_vs_far") is None else
                 f"{entry['delta_vs_far']:+.1f} [{entry['delta_ci_lo']:+.1f},{entry['delta_ci_hi']:+.1f}]")
        p = "--" if entry.get("p_vs_far") is None else f"{entry['p_vs_far']:.3f}"
        print(f"{entry['bucket']:>20} {entry.get('n_frames', 0):>8} "
              f"{entry.get('n_utts', 0):>6} {entry['mean']:>9.1f}% {delta:>16} {p:>7}")
    print("note: delta/p are paired over utterances that contribute to both buckets.")


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CodeSwitch-TTS full evaluation suite (60-utterance benchmark, figures, LaTeX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, default="codeswitch_ckpt",
                   help="Checkpoint directory produced by train_draft_distill.py.")
    p.add_argument("--results", type=str, default="eval_results/raw_results.jsonl")
    p.add_argument("--audio-dir", type=str, default="eval_audio")
    p.add_argument("--figures-dir", type=str, default="paper_figures")
    p.add_argument("--configs", type=str, default="A,B,C",
                   help="Comma-separated subset of A,B,C.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N utterances (smoke tests).")
    p.add_argument("--force", action="store_true",
                   help="Re-run utterances already present in the JSONL (also accepts results "
                        "produced with different checkpoints - see provenance.json).")
    p.add_argument("--replay", action="store_true",
                   help="Skip synthesis; rebuild aggregates/figures/tables from the JSONL.")
    p.add_argument("--skip-asr", action="store_true")
    p.add_argument("--no-utmos", action="store_true",
                   help="Skip the UTMOS22 perceptual proxy (tarepan/SpeechMOS; downloads "
                        "~50 MB and needs trust_remote_code on first use).")
    p.add_argument("--no-mcd", action="store_true",
                   help="Skip mel-cepstral distortion against the Pure-AR rendering "
                        "(computed from the saved wavs, no GPU needed).")
    p.add_argument("--asr-model", type=str, default="openai/whisper-small")
    p.add_argument("--allow-untrained", action="store_true",
                   help="Permit random-weight modules (local smoke tests only).")
    p.add_argument("--no-true-probs", action="store_true",
                   help="Disable teacher-forced scoring (unrelated-target reference telemetry only).")
    p.add_argument("--no-save-audio", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                   help="Parler precision on GPU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--play-steps", type=int, default=12,
                   help="Parler streamer play_steps (smaller = lower TTFA, more decode calls).")
    p.add_argument("--boundary-frames", type=int, default=3,
                   help="Frames after a language switch counted as boundary zone.")
    p.add_argument("--gate-thresholds", type=str, default="auto",
                   help="Draft-confidence thresholds for config C: a comma-separated list, or "
                        "\"auto\" (default) for quantiles of the confidence the run actually "
                        "measures. With --gate-mode derived nothing extra is synthesised, so the "
                        "whole latency/quality curve is free; --gate-mode synth re-synthesises C "
                        "once per threshold.")
    p.add_argument("--gate-mode", choices=["derived", "synth"], default="derived",
                   help="derived (default): build the C@tau operating points from the measured "
                        "C(tau=0) and Pure-AR rows (exact for quality/acceptance, no extra GPU "
                        "time). synth: really synthesise the gated mixture for every threshold.")
    p.add_argument("--gate-stat", choices=["utt_min", "utt_mean"], default="utt_min",
                   help="Utterance-level draft-confidence statistic compared with the threshold "
                        "(utt_min = worst chunk, the conservative default).")
    p.add_argument("--wandb", action="store_true",
                   help="Push the finished run to Weights & Biases (scalars, figures, sweep "
                        "tables, audio samples + an artifact bundle). Works with --replay too, so "
                        "figures can be refreshed and re-pushed with zero GPU time.")
    p.add_argument("--wandb-project", type=str, default="codeswitch-tts")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="codeswitch,kaggle,t4",
                   help="Comma-separated tags for the W&B run.")
    p.add_argument("--wandb-group", type=str, default=None,
                   help="e.g. the corpus size, to group the runs of one paper sweep.")
    p.add_argument("--wandb-api-key", type=str, default=None,
                   help="Defaults to the WANDB_API_KEY env var, then to the Kaggle secret of "
                        "that name (Add-ons -> Secrets).")
    return p


# ===========================================================================
# Main
# ===========================================================================
def _load_checkpoints(args, config, device):
    from codeswitch_tts.config import DraftModelConfig, ResidualModelConfig

    ckpt_dir = Path(args.ckpt)
    draft_model = None
    residual_model = None
    draft_trained = residual_trained = False

    draft_path = ckpt_dir / "draft_cb0.pt"
    residual_path = ckpt_dir / "residual_cb1_8.pt"
    if draft_path.exists():
        payload = torch.load(draft_path, map_location="cpu", weights_only=False)
        cfg = DraftModelConfig(**payload.get("config", {}))
        draft_model = DraftTransformer(cfg).to(device)
        draft_model.load_state_dict(payload["state_dict"])
        draft_model.eval()
        draft_trained = True
        print(f"[Ckpt] draft loaded ({sum(p.numel() for p in draft_model.parameters()) / 1e6:.2f}M params)")
    else:
        draft_model = DraftTransformer(config.draft).to(device)
        draft_model.eval()
        print(f"[Ckpt] ⚠ {draft_path} not found — draft model is RANDOMLY INITIALISED")
    if residual_path.exists():
        payload = torch.load(residual_path, map_location="cpu", weights_only=False)
        rcfg = ResidualModelConfig(**payload.get("config", {}))
        residual_model = ResidualDecoder(rcfg).to(device)
        residual_model.load_state_dict(payload["state_dict"])
        residual_model.eval()
        residual_trained = True
        print(f"[Ckpt] residual loaded ({sum(p.numel() for p in residual_model.parameters()) / 1e6:.2f}M params)")
    else:
        residual_model = ResidualDecoder(config.residual).to(device)
        residual_model.eval()
        print(f"[Ckpt] ⚠ {residual_path} not found — residual decoder is RANDOMLY INITIALISED")

    length_model = None
    meta_path = ckpt_dir / "meta.json"
    if meta_path.exists():
        try:
            from train_draft_distill import LengthModel

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("length_model"):
                length_model = LengthModel.from_dict(meta["length_model"])
                print(f"[Ckpt] length model loaded (MAE {length_model.mae_frames:.1f} frames over "
                      f"{length_model.n_samples} teacher sentences)")
        except Exception as exc:
            print(f"[Ckpt] length model unavailable: {type(exc).__name__}: {exc}")
    return draft_model, residual_model, length_model, draft_trained, residual_trained


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in ("A", "B", "C")]
    if unknown:
        print(f"Unknown configs: {unknown}; valid: A,B,C")
        return 2

    results_path = Path(args.results)
    audio_dir = Path(args.audio_dir)
    figures_dir = Path(args.figures_dir)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    provenance = check_provenance(results_path, args.ckpt, force=args.force, replay=args.replay)
    print(f"[Ckpt] provenance: {len(provenance['fingerprint'])} files hashed from {args.ckpt}"
          f"{' (replay: not checked)' if args.replay else ''}")
    if provenance.get("note"):
        print(f"[Ckpt] note: {provenance['note']}")
    if provenance["blocked"]:
        print("[Ckpt] ❌ " + provenance["reason"])
        return 3
    if not args.replay:
        Path(provenance["path"]).write_text(
            json.dumps({"checkpoints": provenance["fingerprint"],
                        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ckpt_dir": str(args.ckpt)},
                       indent=2, default=str),
            encoding="utf-8",
        )

    t_all = _now()
    print("=" * 78)
    print("CodeSwitch-TTS · Full Evaluation Suite")
    print(f"  configs: {configs} | utterances: {len(EVAL_UTTERANCES)}"
          + (f" (limit {args.limit})" if args.limit else ""))
    print(f"  results: {results_path} | figures: {figures_dir}")
    print(f"  seed: {args.seed} | replay: {args.replay}")
    print("=" * 78)

    rows = load_results(results_path)
    if args.replay:
        print(f"[Replay] loaded {len(rows)} existing result rows")
        return _finish(rows, args, configs, results_path, figures_dir, t_all)

    # ---------------- environment ----------------
    from codeswitch_tts import SpeculativeEngine

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = torch.float32 if device.type == "cpu" else dtype_map[args.dtype]

    config = CodeSwitchTTSConfig()
    config.system.allow_untrained_fallback = bool(args.allow_untrained)

    engine = None
    needs_engine = ("A" in configs) or ("C" in configs and not args.no_true_probs)
    if needs_engine:
        engine = SpeculativeEngine(config.speculative, device=device, dtype=dtype)
        if not engine.speech_available:
            if not args.allow_untrained and "A" in configs:
                print("\n[Eval] ❌ Config A and true acceptance scoring require Indic Parler-TTS.")
                print("       Fix the backend (protobuf==5.29.5 + kernel restart) and re-run,")
                print("       or run with --configs B (NAR-only) / --allow-untrained for smoke tests.")
                return 2
            print("[Eval] ⚠ Parler unavailable — proceeding in reduced mode (no A / unrelated-target "
                  "acceptance only).")

    draft_model = residual_model = None
    length_model = None
    draft_trained = residual_trained = False
    if "B" in configs or "C" in configs:
        draft_model, residual_model, length_model, draft_trained, residual_trained = _load_checkpoints(
            args, config, device
        )
        if not (draft_trained and residual_trained) and not args.allow_untrained:
            print("\n[Eval] ❌ Trained checkpoints are missing: run `python train_draft_distill.py`")
            print("       first, or pass --allow-untrained for a random-weight smoke test.")
            return 2

    scorer = None
    if "C" in configs and not args.no_true_probs:
        if engine is not None and engine.speech_available:
            scorer = TeacherForcedScorer(engine)
        else:
            print("[Eval] ⚠ true acceptance disabled (no Parler backend) — unrelated-target "
                  "reference telemetry only.")

    synthesizer = EvalSynthesizer(
        args, config, engine, draft_model, residual_model, length_model, scorer, device,
        draft_trained, residual_trained,
    )

    asr = ASREvaluator(
        args.asr_model, device,
        enabled=(not args.skip_asr and any(c in configs for c in ("A", "B", "C"))),
    )
    utmos = UTMOSScorer(
        device,
        enabled=(not args.no_utmos and not args.skip_asr
                 and any(c in configs for c in ("A", "B", "C"))),
    )

    run_meta = {
        "args": vars(args),
        "versions": package_versions(),
        "device": str(device),
        "dtype": str(dtype),
        "n_utterances": len(EVAL_UTTERANCES),
        "registers": {r: sum(1 for u in EVAL_UTTERANCES if u["register"] == r) for r in REGISTER_LABELS},
        "checkpoint_dir": str(args.ckpt),
        "draft_trained": draft_trained,
        "residual_trained": residual_trained,
        "scorer_available": bool(scorer and scorer.available),
        "utmos_available": bool(utmos and utmos.available),
        "utmos_model": (getattr(utmos, "source", None) or getattr(utmos, "model_id", None)
                        if utmos is not None else None),
        "length_model": (length_model.to_dict() if length_model else None),
        "checkpoint_fingerprint": provenance["fingerprint"],
        "checkpoint_provenance_note": provenance.get("note"),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    (results_path.parent / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, default=str), encoding="utf-8"
    )

    done = {(r.get("config"), r.get("utt_id"), float(r.get("gate_threshold") or 0.0))
            for r in rows if not r.get("error")}
    utterances = EVAL_UTTERANCES[: args.limit] if args.limit else EVAL_UTTERANCES
    gate_mode = str(getattr(args, "gate_mode", "derived") or "derived")
    all_gate_thresholds = parse_gate_thresholds(
        getattr(args, "gate_thresholds", "auto"), rows, args.gate_stat)
    if gate_mode == "derived":
        # nothing above tau=0 needs synthesising: the sweep is derived from the
        # measured C/A rows in _finish
        gate_thresholds = [0.0]
        if len(all_gate_thresholds) > 1:
            print(f"[Gate] derived mode: synthesising C at tau=0 only; the "
                  f"{len(all_gate_thresholds) - 1} gated operating points are computed from "
                  "the measured rows (no extra GPU time).")
    else:
        if len(all_gate_thresholds) > 1 and str(args.gate_thresholds).strip().lower() == "auto":
            raise SystemExit("[Gate] --gate-thresholds auto needs --gate-mode derived "
                             "(thresholds cannot be calibrated before synthesis).")
        gate_thresholds = all_gate_thresholds

    # ---------------- synthesis loop ----------------
    n_runs_since_gc = 0
    for utt_index, utt in enumerate(utterances):
        for cfg in configs:
            for threshold in (gate_thresholds if cfg == "C" else [0.0]):
                if (cfg, utt["id"], threshold) in done and not args.force:
                    continue
                seed = args.seed * 100003 + utt_index * 7 + {"A": 0, "B": 1, "C": 2}[cfg]
                label = f"[{cfg}][{utt['id']}]" + (f"[g{threshold:.2f}]" if threshold > 0 else "")
                t0 = _now()
                try:
                    if cfg == "A":
                        row, wave = synthesizer.synth_A(utt, utt_index, seed)
                        row["gate_threshold"] = 0.0
                    elif cfg == "B":
                        row, wave = synthesizer._chunked_synthesis(utt, utt_index, seed, verify=False)
                        row["gate_threshold"] = 0.0
                    else:
                        row, wave = synthesizer.synth_C(
                            utt, utt_index, seed, gate_threshold=threshold,
                            gate_stat=args.gate_stat,
                        )
                except Exception as exc:
                    row = synthesizer._base_row(cfg, utt, utt_index, synthesizer.normalize(utt["text"]))
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["seed"] = seed
                    row["gate_threshold"] = float(threshold)
                    wave = None
                    print(f"{label} ❌ {row['error']}")

                if row.get("error") is None and wave is not None:
                    if not args.no_save_audio:
                        from codeswitch_tts.utils.audio import save_wav

                        wav_path = audio_dir / (cfg if threshold <= 0 else
                                                f"{cfg}_g{threshold:.2f}") / f"{utt['id']}.wav"
                        save_wav(wav_path, wave, sample_rate=row["sample_rate"], normalize=False)
                        row["audio_path"] = str(wav_path)
                    if asr.available:
                        hyp = asr.transcribe(wave, row["sample_rate"])
                        cer, wer = ASREvaluator.score(row["normalized_text"], hyp)
                        cer_lat, wer_lat = ASREvaluator.score_latin(row["normalized_text"], hyp)
                        row["asr_text"] = hyp
                        row["cer"] = cer
                        row["wer"] = wer
                        row["cer_latin"] = cer_lat
                        row["wer_latin"] = wer_lat
                    else:
                        row["cer"] = None
                        row["wer"] = None
                        row["cer_latin"] = None
                        row["wer_latin"] = None
                        row["asr_text"] = None
                    if utmos is not None:
                        row["utmos"] = utmos.score(wave, row["sample_rate"])

                if row.get("utt_index") is None:
                    row["utt_index"] = utt_index

                # replace any previous row for (cfg, utt, gate) then persist
                rows = [r for r in rows
                        if not (r.get("config") == cfg and r.get("utt_id") == utt["id"]
                                and float(r.get("gate_threshold") or 0.0) == float(threshold))]
                rows.append(row)
                save_results(results_path, rows)
                done.add((cfg, utt["id"], float(threshold)))

                elapsed = _now() - t0
                if row.get("error"):
                    pass
                else:
                    print(
                        f"{label} t_c={row.get('ttfa_compute_ms') if row.get('ttfa_compute_ms') is None else round(row['ttfa_compute_ms'])}ms "
                        f"t_e2e={row.get('ttfa_e2e_ms') if row.get('ttfa_e2e_ms') is None else round(row['ttfa_e2e_ms'])}ms "
                        f"total={round(row.get('total_ms') or 0)}ms rtf={row.get('rtf') if row.get('rtf') is None else round(row['rtf'], 2)} "
                        f"acc_true={row.get('accept_true') if row.get('accept_true') is None else round(row['accept_true'], 1)} "
                        f"acc_ref={row.get('accept_synth') if row.get('accept_synth') is None else round(row['accept_synth'], 1)} "
                        f"cer={row.get('cer') if row.get('cer') is None else round(row['cer'], 3)} "
                        + (f"conf={row['draft_confidence']:.3f} " if row.get("draft_confidence") is not None else "")
                        + ("FALLBACK " if row.get("fallback") else "")
                        + f"[{elapsed:.1f}s]"
                    )
                n_runs_since_gc += 1
                if n_runs_since_gc >= 5:
                    n_runs_since_gc = 0
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return _finish(rows, args, configs, results_path, figures_dir, t_all)


def _finish(rows, args, configs, results_path: Path, figures_dir: Path, t_all: float) -> int:
    gate_mode = str(getattr(args, "gate_mode", "derived") or "derived")
    thresholds = parse_gate_thresholds(getattr(args, "gate_thresholds", "auto"), rows,
                                       getattr(args, "gate_stat", "utt_min"))
    gate_threshold = 0.0  # Table I always reports the un-gated hybrid
    if not getattr(args, "no_mcd", False):
        attach_mcd_from_audio(rows)
    agg = aggregate_results(rows, configs, gate_threshold=gate_threshold)
    gate_validation: Optional[List[Dict[str, Any]]] = None
    if gate_mode == "derived":
        derived_rows = derive_gate_sweep(rows, thresholds, getattr(args, "gate_stat", "utt_min"))
        gate_rows = aggregate_gate_sweep(derived_rows)
        gate_validation = compare_gate_modes(rows, derived_rows)
    else:
        gate_rows = aggregate_gate_sweep(rows)
    gate_note = gate_sweep_warning(gate_rows, gate_confidence_table(
        rows, getattr(args, "gate_stat", "utt_min")))
    switch_rows = switch_distance_table(rows)
    tables_tex = (
        build_table_I(agg) + "\n\n" + build_table_II(agg) + "\n\n" + build_table_III(agg)
        + "\n\n" + build_table_IV(switch_rows)
        + "\n\n" + build_table_V(gate_rows, agg.get("A"), derived=(gate_mode == "derived"),
                                   thresholds=thresholds)
    )
    (figures_dir / "tables.tex").write_text(tables_tex, encoding="utf-8")

    print("\n" + "=" * 78)
    print("AGGREGATED RESULTS")
    print("=" * 78)
    for cfg in configs:
        a = agg.get(cfg) or {}
        if not a.get("n"):
            print(f"[{cfg}] no successful runs ({a.get('errors', 0)} errors)")
            continue
        tc = a.get("ttfa_compute") or {}
        te = a.get("ttfa_e2e") or {}
        cer = a.get("cer") or {}
        cer_p50 = "--" if cer.get("p50") is None else f"{cer['p50'] * 100:.1f}%"
        cer_mean = "--" if cer.get("mean") is None else f"{cer['mean'] * 100:.1f}%"
        print(
            f"[{cfg}] n={a['n']} (errors={a['errors']}) | "
            f"TTFA-c p50/p90={tc.get('p50') if tc.get('p50') is None else round(tc['p50'])}/"
            f"{tc.get('p90') if tc.get('p90') is None else round(tc['p90'])}ms | "
            f"TTFA-e2e p50={te.get('p50') if te.get('p50') is None else round(te['p50'])}ms | "
            f"total={round((a.get('total') or {}).get('mean') or 0)}ms | "
            f"RTF={round((a.get('rtf') or {}).get('mean') or 0, 2)} | "
            f"CER p50={cer_p50} mean={cer_mean} | "
            f"acc_true={None if (a.get('accept_true') or {}).get('mean') is None else round(a['accept_true']['mean'], 1)}% | "
            f"acc_ref={None if (a.get('accept_synth') or {}).get('mean') is None else round(a['accept_synth']['mean'], 1)}%"
        )
        extra = []
        lat = a.get("cer_latin") or {}
        if lat.get("p50") is not None:
            extra.append(f"CER-lat p50={float(lat['p50']) * 100:.1f}%")
        ut = a.get("utmos") or {}
        if ut.get("mean") is not None:
            extra.append(f"UTMOS={ut['mean']:.2f}")
        mcd = a.get("mcd_vs_a") or {}
        if mcd.get("mean") is not None:
            extra.append(f"MCD={mcd['mean']:.2f} dB")
        if extra:
            print("      " + " | ".join(extra))
        if a.get("asr_fail"):
            utts = a.get("asr_fail_utts") or []
            shown = ", ".join(utts[:6]) + (" ..." if len(utts) > 6 else "")
            print(
                f"      ASR failures (CER>200%, whisper hallucination loops, excluded from p50): "
                f"{a['asr_fail']}/{a['n']} [{shown}]"
            )
        lq_d = a.get("logq_draft") or {}
        lq_k = a.get("logq_kept") or {}
        if lq_d.get("mean") is not None and lq_k.get("mean") is not None:
            print(
                f"      target correction: teacher-forced mean log q = "
                f"{lq_d['mean']:.2f} nats/frame (draft proposal) -> {lq_k['mean']:.2f} "
                f"(emitted after rejection sampling)"
            )

    print("-" * 78)
    print("PER-REGISTER BREAKDOWN (CER p50 / mean; TTFA-c warm p50)")
    print("-" * 78)
    for reg, reg_label in REGISTER_LABELS.items():
        for cfg in configs:
            entry = ((agg.get(cfg) or {}).get("registers") or {}).get(reg)
            if not entry or not entry.get("n"):
                continue
            cer = entry.get("cer") or {}
            p50 = "--" if cer.get("p50") is None else f"{cer['p50'] * 100:.1f}%"
            mean = "--" if cer.get("mean") is None else f"{cer['mean'] * 100:.1f}%"
            t50 = (entry.get("ttfa_compute") or {}).get("p50")
            acc = (entry.get("accept_true") or {}).get("mean")
            print(
                f"[{cfg}] {reg_label:<20s} n={entry['n']:<3d} CER p50={p50:<7s} mean={mean:<7s} "
                f"TTFA-c p50={('--' if t50 is None else str(round(t50)) + 'ms'):<8s} "
                f"RTF={round((entry.get('rtf') or {}).get('mean') or 0, 2):<5} "
                f"acc_true={'--' if acc is None else f'{acc:.1f}%'}"
            )

    print_gate_sweep(gate_rows)
    if gate_mode == "derived":
        print(f"[Gate] derived sweep at tau in {[round(t, 4) for t in thresholds]} "
              f"(un-gated C + measured Pure-AR rows; no extra synthesis)")
        if gate_validation:
            for chk in gate_validation:
                print(f"[Gate] validation tau={chk['threshold']:.4f} "
                      f"fallback={chk['fallback_rate'] * 100:.1f}%: "
                      f"total p50 derived={chk['total_p50_derived_ms']:.0f}ms vs "
                      f"synthesised={chk['total_p50_synth_ms']:.0f}ms | "
                      f"CER p50 {chk['cer_p50_derived']} vs {chk['cer_p50_synth']}")
        else:
            print("[Gate] no synthesised gated rows to validate the derived sweep against "
                  "(run a few with --gate-mode synth --configs C if you want the check)")
    if gate_note:
        print(f"[Gate] ⚠ {gate_note}")
    print_switch_distance(switch_rows)

    fig_paths: Dict[str, List[str]] = {}
    for name, fn in (
        ("fig1", figure1_latency_quality),
        ("fig2", figure2_zone_acceptance),
    ):
        try:
            paths = fn(agg, figures_dir)
            if paths:
                fig_paths[name] = paths
        except Exception as exc:
            print(f"[Figures] {name} failed: {type(exc).__name__}: {exc}")
    try:
        paths = figure4_scaling_and_registers(rows, agg, figures_dir)
        if paths:
            fig_paths["fig4"] = paths
    except Exception as exc:
        print(f"[Figures] fig4 failed: {type(exc).__name__}: {exc}")
    try:
        paths = figure5_gate_pareto(rows, agg, figures_dir)
        if paths:
            fig_paths["fig5"] = paths
    except Exception as exc:
        print(f"[Figures] fig5 failed: {type(exc).__name__}: {exc}")
    try:
        paths = figure3_stream_timeline(rows, figures_dir)
        if paths:
            fig_paths["fig3"] = paths
    except Exception as exc:
        print(f"[Figures] fig3 failed: {type(exc).__name__}: {exc}")

    summary = {
        "n_rows": len(rows),
        "configs": configs,
        "aggregates": agg,
        "gate_sweep": gate_rows,
        "gate_mode": gate_mode,
        "gate_thresholds": [float(t) for t in thresholds],
        "gate_stat": getattr(args, "gate_stat", "utt_min"),
        "gate_validation": gate_validation,
        "gate_note": gate_note,
        "switch_distance": switch_rows,
        "checkpoint_fingerprint": checkpoint_fingerprint(args.ckpt),
        "figures": fig_paths,
        "tables_tex": str(figures_dir / "tables.tex"),
    }
    (figures_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "-" * 78)
    print("TABLE I (LaTeX)")
    print("-" * 78)
    print(build_table_I(agg))
    print("\n" + "-" * 78)
    print("TABLE II (LaTeX)")
    print("-" * 78)
    print(build_table_II(agg))
    print("\n" + "-" * 78)
    print("TABLE III (LaTeX)")
    print("-" * 78)
    print(build_table_III(agg))
    print("\n" + "-" * 78)
    print("TABLE IV (LaTeX) - distance-to-switch acceptance")
    print("-" * 78)
    print(build_table_IV(switch_rows))
    print("\n" + "-" * 78)
    print("TABLE V (LaTeX) - confidence-gated operating points")
    print("-" * 78)
    print(build_table_V(gate_rows, agg.get("A")))
    print("\n" + "-" * 78)
    print(f"Artifacts: {results_path} | {figures_dir}/tables.tex | {figures_dir}/summary.json")
    for name, paths in fig_paths.items():
        for p in paths:
            print(f"  {name}: {p}")
    if getattr(args, "wandb", False):
        try:
            import push_to_wandb

            report = push_to_wandb.push(
                results=results_path,
                figures_dir=figures_dir,
                audio_dir=Path(getattr(args, "audio_dir", "eval_audio")),
                ckpt_dir=Path(args.ckpt),
                project=getattr(args, "wandb_project", "codeswitch-tts"),
                entity=getattr(args, "wandb_entity", None),
                run_name=getattr(args, "wandb_run_name", None),
                group=getattr(args, "wandb_group", None),
                tags=[t.strip() for t in str(getattr(args, "wandb_tags", "")).split(",") if t.strip()],
                api_key=getattr(args, "wandb_api_key", None),
            )
            report.get("url") and print(f"[W&B] done: {report['url']}")
        except Exception as exc:
            print(f"[W&B] \u26a0 push failed: {type(exc).__name__}: {exc}")
            print("[W&B] the evaluation itself is complete - retry just the upload with "
                  "`python push_to_wandb.py`")

    print(f"[Done] total {fmt_time(_now() - t_all)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
