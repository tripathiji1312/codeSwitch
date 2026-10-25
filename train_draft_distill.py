"""Self-distillation trainer for the CodeSwitch-TTS draft & residual models.

One-command Kaggle workflow (after the notebook environment is set up):

    !python train_draft_distill.py

What it does
------------
1. TEACHER PASS — Indic Parler-TTS synthesizes a bootstrap Tanglish/Tamil
   corpus (``codeswitch_benchmark_data.build_training_corpus``, 300 sentences
   by default, disjoint from the 60-utterance evaluation set).  For every
   sentence the 9-codebook DAC token stream is captured and cached in
   ``teacher_cache/`` (re-runs reuse the cache).
2. DRAFT TRAINING — the non-autoregressive Codebook-0 ``DraftTransformer`` is
   distilled from (normalized text -> teacher CB0 frames).
3. RESIDUAL TRAINING — the ``ResidualDecoder`` is trained to predict
   Codebooks 1-8 from teacher Codebook 0 (teacher-forced, 86-frame windows).
4. A data-driven text-length model (chars -> DAC frames) is fitted from the
   teacher set and stored, so the evaluation suite can size utterances the
   same way the teacher did.
5. Checkpoints + ``meta.json`` are written to ``codeswitch_ckpt/`` and a
   sanity WAV is rendered through draft -> residual -> DAC.

The whole pipeline is budgeted to finish within ~30-45 minutes on a Kaggle T4
(``--max-minutes`` / ``--max-minutes-residual`` guard each training stage).

Two teacher targets are supported (``--target-source``):

* ``codec``  — re-encode the teacher waveform with the DAC encoder
  (version-robust default).
* ``lm``     — capture the decoder LM's own sampled token stream during
  generation (validated against the decoded waveform; automatic fallback to
  ``codec`` if the capture is not consistent with the audio).
* ``auto``   — try ``lm`` once; keep it only if it validates.

Local smoke test (no GPU, no Parler required):

    python train_draft_distill.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np

# Importing codeswitch_tts installs the protobuf runtime_version fallback shim
# first (Kaggle stale-protobuf safety net) before parler_tts/transformers load.
from codeswitch_tts import (  # noqa: E402
    CodeSwitchTTSConfig,
    DraftModelConfig,
    DraftTransformer,
    ResidualDecoder,
    TanglishNormalizer,
)
from codeswitch_benchmark_data import (  # noqa: E402
    assert_corpus_disjoint_from_eval,
    build_training_corpus,
    load_corpus_json,
)
from codeswitch_tts.data.shards import SpeechShard  # noqa: E402

try:  # torch is imported through codeswitch_tts already, but keep it explicit
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch import failed: {exc}")


# ===========================================================================
# Small utilities
# ===========================================================================

def _now() -> float:
    return time.perf_counter()


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _grad_scaler(device) -> Optional["torch.cuda.amp.GradScaler"]:
    """Version-tolerant AMP GradScaler factory (``None`` on CPU)."""
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:  # older torch
        return torch.cuda.amp.GradScaler()


def package_versions() -> Dict[str, str]:
    from importlib import metadata

    versions: Dict[str, str] = {}
    for pkg in ("torch", "transformers", "parler_tts", "protobuf", "numpy"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


# ===========================================================================
# Length model: characters -> DAC frames
# ===========================================================================

@dataclass
class LengthModel:
    """Least-squares text-length -> acoustic-frame-count model."""

    a: float = 6.6
    b: float = 0.0
    min_frames: int = 30
    max_frames: int = 430
    n_samples: int = 0
    mae_frames: float = 0.0

    @classmethod
    def fit(cls, n_chars: Sequence[float], frames: Sequence[float]) -> "LengthModel":
        x = np.asarray(n_chars, dtype=np.float64)
        y = np.asarray(frames, dtype=np.float64)
        if len(x) >= 2 and float(np.var(x)) > 1e-6:
            a, b = np.polyfit(x, y, 1)
            pred = a * x + b
            mae = float(np.mean(np.abs(pred - y)))
        else:  # degenerate fallback ~ 13 chars/sec at 86.13 frames/sec
            a, b, mae = 86.13 / 13.0, 0.0, 0.0
            pred = a * x + b if len(x) else np.asarray([])
        lo = int(max(20, np.floor(np.min(y) * 0.6))) if len(y) else 30
        hi = int(max(60, np.ceil(np.max(y) * 1.25))) if len(y) else 430
        return cls(a=float(a), b=float(b), min_frames=lo, max_frames=hi,
                   n_samples=len(x), mae_frames=mae)

    def predict(self, text: str) -> int:
        frames = self.a * len(text) + self.b
        return int(min(self.max_frames, max(self.min_frames, round(frames))))

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "LengthModel":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ===========================================================================
# Teacher data collection
# ===========================================================================

def _norm_text(text: Any) -> str:
    """Whitespace/case-insensitive sentence identity (teacher-cache key)."""
    return " ".join(str(text or "").split()).casefold()


def _same_sentence(a: str, b: str) -> bool:
    """Whitespace/case-insensitive sentence identity for teacher-cache validation."""
    return _norm_text(a) == _norm_text(b)


def teacher_shard_name(text: str) -> str:
    """Content-addressed cache filename: ``shard_<sha1(normalized text)[:16]>.npz``.

    Addressing shards by *content* rather than position means a corpus change only
    costs the sentences that are actually new (growing the corpus, reordering it or
    swapping sources all keep the untouched sentences' caches), which is what makes
    a multi-session teacher pass cheap.
    """
    digest = hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()[:16]
    return f"shard_{digest}.npz"


# Backwards-compatible name: the old teacher cache and the new real-speech
# pipeline share one on-disk contract (``SpeechShard``).
TeacherShard = SpeechShard


class _CodeCaptureStreamer:
    """Passive HF streamer that records the raw decoder token cache.

    Mirrors what ``parler_tts.streamer.ParlerTTSStreamer`` receives during
    generation (``put`` of the initial delayed prompt ``(K, L0)``, then one
    token per codebook per step ``(K,)``) but extracts DAC codes instead of
    decoding audio.  Tensor-shape handling is deliberately defensive because
    call sites differ across transformers releases.
    """

    def __init__(self, decoder, generation_config):
        self.decoder = decoder
        self.generation_config = generation_config
        self.num_codebooks = int(getattr(decoder, "num_codebooks", 9))
        self.token_cache: Optional[torch.Tensor] = None

    # -- HF BaseStreamer interface -----------------------------------------
    def put(self, value: torch.Tensor) -> None:
        value = value.detach()
        if value.dim() == 3:  # tolerate (1, K, L) style payloads
            value = value.reshape(value.shape[0] * value.shape[1], value.shape[2])
        if value.dim() == 1:
            value = value[:, None]
        if value.dim() != 2:
            value = value.reshape(-1, 1)
        if value.shape[0] != self.num_codebooks and value.shape[1] == self.num_codebooks:
            value = value.transpose(0, 1)  # tolerate (L, K)
        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.cat([self.token_cache, value], dim=-1)

    def end(self) -> None:  # pragma: no cover - nothing to flush
        pass

    # -- extraction ---------------------------------------------------------
    def extract_codes(self) -> Optional[np.ndarray]:
        """Reconstruct (num_codebooks, T) DAC codes from the token cache."""
        if self.token_cache is None:
            return None
        ids = self.token_cache.cpu().long()
        gc_cfg = self.generation_config
        bos_id = getattr(gc_cfg, "bos_token_id", None) or getattr(
            gc_cfg, "decoder_start_token_id", None
        )
        pad_id = getattr(gc_cfg, "pad_token_id", None) or bos_id
        if bos_id is None or pad_id is None:
            return None
        try:
            _, pattern = self.decoder.build_delay_pattern_mask(
                ids[:, :1],
                bos_token_id=int(bos_id),
                pad_token_id=int(pad_id),
                max_length=ids.shape[-1],
            )
            filled = self.decoder.apply_delay_pattern_mask(ids, pattern)
            mask = (pattern != int(bos_id)) & (pattern != int(pad_id))
            codes = filled[mask].reshape(1, self.num_codebooks, -1)[0]
            return codes.numpy().astype(np.int64)
        except Exception:
            return None


def _decode_codes_for_validation(
    audio_encoder, codes: torch.Tensor
) -> Tuple[Optional[np.ndarray], str]:
    """Decodes ``(K, T)`` codes through the audio encoder for sanity checks.

    Returns ``(waveform, error)``; the waveform is ``None`` when every decode
    signature failed and ``error`` then holds the last exception.  Both the
    3-D ``(batch, K, T)`` layout used by ``ParlerTTSForCausalLM.generate`` and
    the 4-D ``(chunk, batch, K, T)`` layout are attempted, with and without
    the optional ``audio_scales`` argument, because the audio encoder class
    (transformers ``DacModel`` vs. the parler-tts wrapper) differs per release.
    """
    last_err = "no decode signature succeeded"
    flat = codes.long()
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
                return audio.detach().float().cpu().numpy().reshape(-1), ""
            except Exception as exc:
                last_err = (
                    f"{shape_name}{'+scales' if use_scales else ''} -> "
                    f"{type(exc).__name__}: {exc}"
                )
    return None, last_err


def _zero_lag_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Normalised zero-lag correlation; ``None`` when either side is constant."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(min(a.size, b.size))
    if n < 2:
        return None
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom <= 1e-12:
        return None
    return float((a * b).sum() / denom)


class TeacherCollector:
    """Runs the Parler teacher over the corpus and caches DAC code sequences."""

    def __init__(
        self,
        engine,
        out_dir: Path,
        normalizer: TanglishNormalizer,
        seed: int = 42,
        target_source: str = "auto",
        max_frames: int = 430,
        verbose: bool = True,
    ):
        self.engine = engine
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = normalizer
        self.seed = seed
        self.target_source = target_source
        self.max_frames = max_frames
        self.verbose = verbose
        self._lm_capture_ok: Optional[bool] = None  # decided on first sentence
        self.stats = {"lm": 0, "codec": 0, "fallback": 0}

    # -- helpers -------------------------------------------------------------
    def _prepare_prompt(self, text: str) -> Tuple[str, str]:
        norm = self.normalizer.normalize(text).normalized_text
        prompt_clean = norm.strip()
        if not prompt_clean.endswith((".", "!", "?", "।")):
            prompt_clean += "."
        return text, prompt_clean

    def _trim_valid_frames(self, codes: np.ndarray, codebook_size: int) -> np.ndarray:
        """Drops frames holding EOS/BOS/pad artefacts and applies the frame cap.

        Mirrors ``ParlerTTSForCausalLM.generate``: the returned audio is decoded
        from the columns in which *every* codebook holds a valid token
        (``sample_mask``), so the same columns are kept here (a trailing
        diagonal of forced-EOS tokens would otherwise inflate the length).
        """
        if codes.size == 0:
            return codes
        valid = (codes < codebook_size).all(axis=0)
        codes = codes[:, valid]
        return codes[:, : self.max_frames]

    def _generate_and_capture(self, prompt_clean: str, index: int):
        """Generates teacher audio (+ LM codes when capture is enabled)."""
        ar = self.engine.ar_model
        from transformers.modeling_outputs import BaseModelOutput

        desc = self.engine.get_cached_description_encoder_outputs(
            self.engine.config.voice_description
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=desc)
        prompt_ids = self.engine.tokenizer(prompt_clean, return_tensors="pt").to(self.engine.device)

        frame_rate = float(getattr(ar.audio_encoder.config, "frame_rate", 86.13))
        est_sec = max(1.2, len(prompt_clean) / float(self.engine.config.chars_per_sec))
        min_tokens = max(12, int(frame_rate * est_sec * 0.7))
        max_tokens = max(40, int(frame_rate * est_sec * 1.8) + 30)

        set_seed(self.seed + index)

        want_lm = self.target_source in ("auto", "lm") and self._lm_capture_ok is not False
        streamer = (
            _CodeCaptureStreamer(ar.decoder, ar.generation_config) if want_lm else None
        )

        kwargs = dict(
            encoder_outputs=encoder_outputs,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
            do_sample=True,
            temperature=self.engine.config.temperature,
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
        )
        if streamer is not None:
            kwargs["streamer"] = streamer

        with torch.inference_mode():
            out_values = ar.generate(**kwargs)

        audio = out_values.detach().cpu().float().numpy()
        while audio.ndim > 1:
            audio = audio[0]

        lm_codes: Optional[np.ndarray] = None
        if streamer is not None:
            lm_codes = streamer.extract_codes()
        if lm_codes is not None and self._lm_capture_ok is None:
            # Validate once: the captured codes must decode back to the same
            # waveform the teacher returned.  Both sides run the identical
            # canonical post-processing, so lengths must agree within 10%
            # and the overlapping audio must correlate (>0.9).  Otherwise
            # the capture layout differs from this parler-tts build -> fall
            # back to DAC re-encoding for the whole corpus.
            hop = int(np.prod(getattr(ar.audio_encoder.config, "upsampling_ratios", [8, 8, 4, 2])))
            codebook_size = int(ar.audio_encoder.config.codebook_size)
            trimmed = self._trim_valid_frames(lm_codes[:9], codebook_size)
            decoded, decode_err = _decode_codes_for_validation(
                ar.audio_encoder, torch.from_numpy(trimmed)
            )
            n_teacher = int(len(audio))
            if decoded is None:
                consistent = False
                detail = f"decode failed ({decode_err})"
            else:
                n = int(min(len(decoded), n_teacher))
                len_ratio = n / max(len(decoded), n_teacher, 1)
                corr = _zero_lag_corr(decoded[:n], audio[:n]) if n else None
                consistent = trimmed.shape[1] > 0 and len_ratio >= 0.90 and (
                    corr is None or corr >= 0.90
                )
                detail = (
                    f"frames={trimmed.shape[1]} frame_samples={trimmed.shape[1] * hop} "
                    f"decoded={len(decoded)} teacher={n_teacher} "
                    f"len_ratio={len_ratio:.3f} "
                    f"corr={'n/a' if corr is None else f'{corr:.3f}'}"
                )
            self._lm_capture_ok = bool(consistent)
            if self.verbose:
                print(
                    f"[Teacher] LM token capture validation: "
                    f"{'OK' if consistent else 'MISMATCH -> DAC re-encode fallback'} "
                    f"({detail})"
                )

        return audio, lm_codes

    def _codec_reencode(self, audio: np.ndarray) -> Optional[np.ndarray]:
        ar = self.engine.ar_model
        try:
            wave = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, 1, -1)
            # The teacher may run in fp16 while the waveform arrives as float32;
            # the DAC encoder rejects mixed dtypes ("Input type (float) and bias
            # type (c10::Half) should be the same"), so follow the module dtype.
            try:
                param = next(ar.audio_encoder.parameters())
                wave = wave.to(device=param.device, dtype=param.dtype)
            except StopIteration:
                wave = wave.to(self.engine.device)
            with torch.inference_mode():
                try:
                    enc = ar.audio_encoder.encode(wave)
                except AttributeError:  # encoder exposes only forward()
                    enc = ar.audio_encoder(wave)
            codes = getattr(enc, "audio_codes", None)
            if codes is None and isinstance(enc, (list, tuple)):
                codes = enc[0]
            if codes is None:
                raise RuntimeError(
                    f"encoder output has no audio_codes ({type(enc).__name__})"
                )
            codes = codes.detach().cpu().long()
            if codes.dim() == 4:  # (chunks=1, batch=1, K, T)
                codes = codes[0, 0]
            elif codes.dim() == 3:  # (batch=1, K, T)
                codes = codes[0]
            return codes.numpy().astype(np.int64)
        except Exception as exc:
            if self.verbose:
                print(f"[Teacher] DAC re-encode failed: {type(exc).__name__}: {exc}")
            return None

    def _shard_path(self, text: str) -> Path:
        return self.out_dir / teacher_shard_name(text)

    def _adopt_legacy_shards(self) -> Dict[str, Path]:
        """Maps normalized text -> path for position-named shards (older caches).

        Before content addressing, shards were ``shard_00000.npz`` and a corpus
        change silently invalidated them.  Those files are still perfectly good
        *content*, so they are re-indexed by their sentence text and renamed on
        first use - a previous session's teacher pass is not wasted.
        """
        legacy: Dict[str, Path] = {}
        for path in sorted(self.out_dir.glob("shard_*.npz")):
            if re.fullmatch(r"shard_[0-9a-f]{16}", path.stem):
                continue
            try:
                cached = TeacherShard.load(path)
            except Exception:
                continue
            legacy.setdefault(_norm_text(cached.raw_text), path)
            if len(legacy) >= 5000:
                break
        if legacy and self.verbose:
            print(f"[Teacher] found {len(legacy)} position-named shards from an older cache; "
                  "adopting the ones that match the current corpus.")
        return legacy

    # -- main loop -----------------------------------------------------------
    def collect(self, corpus: Sequence[str]) -> Tuple[List[TeacherShard], LengthModel]:
        codebook_size = int(self.engine.ar_model.audio_encoder.config.codebook_size)
        total = len(corpus)
        shards: List[TeacherShard] = []
        t_start = _now()
        legacy = self._adopt_legacy_shards()
        self.stats["reused"] = sum(1 for text in corpus if self._shard_path(text).exists())
        if self.verbose:
            print(f"[Teacher] cache: {self.stats['reused']}/{total} sentences already collected "
                  f"({total - self.stats['reused']} to go, ~"
                  f"{_fmt_time(7.9 * (total - self.stats['reused']))} at 7.9 s/sentence)")
        for i, text in enumerate(corpus):
            shard_path = self._shard_path(text)
            if not shard_path.exists() and legacy:
                donor = legacy.pop(_norm_text(text), None)
                if donor is not None:
                    try:
                        donor.replace(shard_path)          # adopt: rename in place
                        self.stats["adopted"] = self.stats.get("adopted", 0) + 1
                    except Exception:
                        pass
            if shard_path.exists():
                cached = TeacherShard.load(shard_path)
                if _same_sentence(cached.raw_text, text):
                    cached.index = i            # position in *this* corpus
                    shards.append(cached)
                    continue
                # A hash collision (or a hand-edited cache) would otherwise train
                # the student on another sentence's codes; recollect it.
                if self.stats.get("stale", 0) == 0:
                    print(f"[Teacher] ⚠ cached shard text differs from the current corpus "
                          f"({shard_path.name}: cached={cached.raw_text[:48]!r} vs "
                          f"now={text[:48]!r}); recollecting the changed sentences.")
                self.stats["stale"] = self.stats.get("stale", 0) + 1

            t0 = _now()
            raw_text, prompt_clean = self._prepare_prompt(text)
            try:
                audio, lm_codes = self._generate_and_capture(prompt_clean, i)
            except Exception as exc:
                print(f"[Teacher] ⚠ generation failed for sentence {i}: {type(exc).__name__}: {exc}")
                continue

            use_lm = (
                lm_codes is not None
                and lm_codes.shape[0] >= 9
                and (self.target_source == "lm" or (self.target_source == "auto" and self._lm_capture_ok))
            )
            if use_lm:
                codes = lm_codes[:9]
                source = "lm"
            else:
                codes = self._codec_reencode(audio)
                source = "codec"
            if self._lm_capture_ok is False and lm_codes is not None:
                self.stats["fallback"] += 1

            if codes is None:
                print(f"[Teacher] ⚠ sentence {i}: no code stream available; skipping")
                continue

            codes = self._trim_valid_frames(codes, codebook_size)
            if codes.shape[1] < 8:
                print(f"[Teacher] ⚠ sentence {i}: teacher audio too short ({codes.shape[1]} frames); skipping")
                continue
            self.stats[source] = self.stats.get(source, 0) + 1

            shard = TeacherShard(
                index=i,
                raw_text=raw_text,
                norm_text=prompt_clean,
                n_chars=len(prompt_clean),
                frames=int(codes.shape[1]),
                codes=codes,
                audio_sec=float(len(audio) / self.engine.ar_model.config.sampling_rate),
                target_source=source,
                gen_ms=(_now() - t0) * 1000.0,
            )
            shard.save(shard_path)
            shards.append(shard)

            if self.verbose and (i % 10 == 0 or i == total - 1):
                elapsed = _now() - t_start
                rate = elapsed / max(1, len(shards))
                eta = rate * max(0, total - i - 1)
                print(
                    f"[Teacher] {len(shards)}/{total} sentences | "
                    f"last: {shard.frames}f ({shard.audio_sec:.2f}s, {source}) | "
                    f"elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)}"
                )
            if len(shards) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not shards:
            raise RuntimeError("Teacher collection produced no usable shards.")

        length_model = LengthModel.fit(
            [s.n_chars for s in shards], [s.frames for s in shards]
        )
        if self.verbose:
            print(
                f"[Teacher] collected {len(shards)} shards"
                + (f" ({self.stats['stale']} recollected: cache text mismatch)"
                   if self.stats.get("stale") else "")
                + (f" (adopted {self.stats['adopted']} position-named shards)"
                   if self.stats.get("adopted") else "") + " | "
                f"frames: min={min(s.frames for s in shards)} "
                f"med={int(np.median([s.frames for s in shards]))} "
                f"max={max(s.frames for s in shards)} | "
                f"length model: frames ≈ {length_model.a:.2f}·chars {length_model.b:+.1f} "
                f"(MAE {length_model.mae_frames:.1f})"
            )
        return shards, length_model


# ===========================================================================
# Draft CB0 training
# ===========================================================================

def length_bucketed_order(lengths: Sequence[int], batch_size: int,
                          shuffle: bool = False, rng: Optional[random.Random] = None
                          ) -> List[int]:
    """Group indices with similar frame counts so ``_collate``'s max_t is close
    to every sample's own length.

    Shuffle-only batching padded short sentences up to the longest sentence in
    the batch, training them against a longer frame axis than their own target.
    Per-sample ``frame_lengths`` is the actual fix; bucketing still keeps the
    padded tail small and makes batches cheaper.
    """
    order = sorted(range(len(lengths)), key=lambda i: int(lengths[i]))
    blocks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        (rng or random).shuffle(blocks)
    return [i for block in blocks for i in block]


def sequence_diversity_stats(tokens: np.ndarray) -> Dict[str, float]:
    """Measure whether an argmax sequence is a sequence or one repeated token."""
    flat = np.asarray(tokens, dtype=np.int64).reshape(-1)
    n = int(flat.size)
    if n == 0:
        return {"unique": 0.0, "top_frac": 1.0, "longest_run": 0.0,
                "entropy_nats": 0.0}
    counts = np.bincount(flat, minlength=1024).astype(np.float64)
    probs = counts[counts > 0] / n
    entropy = float(-(probs * np.log(probs)).sum())
    longest = run = 1
    for i in range(1, n):
        run = run + 1 if flat[i] == flat[i - 1] else 1
        longest = max(longest, run)
    return {
        "unique": float(np.count_nonzero(counts)),
        "top_frac": float(counts.max() / n),
        "longest_run": float(longest),
        "entropy_nats": entropy,
    }


class DraftTrainer:
    def __init__(
        self,
        draft_model: DraftTransformer,
        device: torch.device,
        lr: float = 3e-4,
        batch_size: int = 24,
        seed: int = 42,
        duration_loss_weight: float = 0.1,
    ):
        self.model = draft_model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = draft_model.tokenizer
        self.duration_loss_weight = float(max(0.0, duration_loss_weight))
        self.last_duration_loss: Optional[float] = None

    @property
    def uses_durations(self) -> bool:
        return (
            self.model.frame_decoder is not None
            and getattr(self.model.frame_decoder, "duration_proj", None) is not None
        )

    def _collate(self, batch: List[TeacherShard]):
        texts = [s.norm_text for s in batch]
        input_ids, attn = self.tokenizer.batch_encode(texts)
        max_t = max(s.frames for s in batch)
        targets = torch.full((len(batch), max_t), -100, dtype=torch.long)
        frame_lengths = torch.tensor([int(s.frames) for s in batch], dtype=torch.long)
        # Per-token durations are in the *tokenizer* coordinate system: one
        # zero-duration BOS, then one entry per text character, then EOS.
        token_durations = torch.zeros_like(input_ids, dtype=torch.float32)
        duration_supervised = torch.zeros(len(batch), dtype=torch.bool)
        for i, s in enumerate(batch):
            t = s.codes[0]
            if int(t.shape[0]) != int(s.frames):
                raise ValueError(
                    f"shard {s.index}: frames={s.frames} but CB0 has {t.shape[0]}"
                )
            targets[i, : t.shape[0]] = torch.from_numpy(t).long()
            n_tokens = int(attn[i].sum().item())
            if (
                s.has_durations
                and s.char_durations is not None
                and len(s.char_durations) == len(s.norm_text)
            ):
                with_eos = np.concatenate(
                    ([0.0], s.char_durations.astype(np.float32), [0.0])
                )
                take = min(int(with_eos.size), n_tokens)
                token_durations[i, :take] = torch.from_numpy(
                    with_eos[:take].copy()
                )
                duration_supervised[i] = True
            else:
                # Uniform layout for shards without an aligner (legacy teacher
                # cache).  The duration loss is masked out for these rows.
                content = max(1, n_tokens - 2)
                token_durations[i, 1:1 + content] = float(s.frames) / content
        return (
            input_ids.to(self.device),
            attn.to(self.device),
            targets.to(self.device),
            max_t,
            frame_lengths.to(self.device),
            token_durations.to(self.device),
            duration_supervised.to(self.device),
        )

    def _run_epoch(self, shards, optimizer, scaler, train: bool):
        self.model.train(train)
        order = length_bucketed_order(
            [s.frames for s in shards], self.batch_size,
            shuffle=train, rng=random,
        )
        total_loss, total_frames = 0.0, 0
        total_duration_loss, duration_batches = 0.0, 0
        use_amp = self.device.type == "cuda"
        for start in range(0, len(order), self.batch_size):
            batch = [shards[j] for j in order[start:start + self.batch_size]]
            (
                input_ids,
                attn,
                targets,
                max_t,
                frame_lengths,
                token_durations,
                duration_supervised,
            ) = self._collate(batch)
            duration_kwargs = (
                {"token_durations": token_durations} if self.uses_durations else {}
            )
            with torch.set_grad_enabled(train):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        frame_lengths=frame_lengths,
                        sample=False,
                        **duration_kwargs,
                    )
                    loss = torch.nn.functional.cross_entropy(
                        out.logits.reshape(-1, out.logits.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=-100,
                    )
                    duration_loss = None
                    if (
                        out.duration_logits is not None
                        and self.duration_loss_weight > 0.0
                        and duration_supervised.any()
                    ):
                        duration_mask = (
                            duration_supervised.unsqueeze(1)
                            & (token_durations > 0)
                            & attn
                        )
                        if duration_mask.any():
                            duration_loss = torch.nn.functional.smooth_l1_loss(
                                out.duration_logits[duration_mask],
                                torch.log1p(token_durations[duration_mask]),
                            )
                            loss = loss + self.duration_loss_weight * duration_loss
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
            n_frames = int((targets != -100).sum().item())
            total_loss += float(loss.detach().cpu()) * n_frames
            total_frames += n_frames
            if duration_loss is not None:
                total_duration_loss += float(duration_loss.detach().cpu())
                duration_batches += 1
        self.last_duration_loss = (
            total_duration_loss / duration_batches if duration_batches else None
        )
        return total_loss / max(1, total_frames)

    @torch.inference_mode()
    def evaluate_topk(
        self, shards, k: int = 5, use_gt_durations: bool = False
    ) -> Tuple[float, float]:
        self.model.eval()
        top1 = topk = total = 0
        diversities: List[Dict[str, float]] = []
        order = length_bucketed_order(
            [s.frames for s in shards], self.batch_size, shuffle=False
        )
        for start in range(0, len(order), self.batch_size):
            batch = [shards[j] for j in order[start:start + self.batch_size]]
            (
                input_ids,
                attn,
                targets,
                max_t,
                frame_lengths,
                token_durations,
                _duration_supervised,
            ) = self._collate(batch)
            duration_kwargs = (
                {"token_durations": token_durations}
                if use_gt_durations and self.uses_durations
                else {}
            )
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                frame_lengths=frame_lengths,
                sample=False,
                **duration_kwargs,
            )
            logits = out.logits
            for i, s in enumerate(batch):
                t = s.codes[0]
                lg = logits[i, : t.shape[0]]
                pred1 = lg.argmax(dim=-1)
                target = torch.from_numpy(t).long().to(pred1.device)
                top1 += int((pred1 == target).sum().item())
                topk_pred = lg.topk(min(k, lg.shape[-1]), dim=-1).indices
                topk += int((topk_pred == target.unsqueeze(-1)).any(dim=-1).sum().item())
                diversities.append(
                    sequence_diversity_stats(pred1.detach().cpu().numpy())
                )
                total += t.shape[0]
        if diversities:
            self.last_argmax_stats = {
                key: float(np.mean([d[key] for d in diversities]))
                for key in ("unique", "top_frac", "longest_run", "entropy_nats")
            }
        else:
            self.last_argmax_stats = {}
        if total == 0:
            return 0.0, 0.0
        return top1 / total, topk / total

    def train(
        self,
        shards: List[TeacherShard],
        epochs: int = 80,
        patience: int = 12,
        max_minutes: float = 12.0,
        log_every: int = 1,
        min_epochs: int = 0,
    ) -> Dict[str, float]:
        # deterministic split: every 10th shard is validation
        train_shards = [s for i, s in enumerate(shards) if i % 10 != 9]
        val_shards = [s for i, s in enumerate(shards) if i % 10 == 9] or train_shards[-1:]
        print(
            f"[Draft] training on {len(train_shards)} sentences, "
            f"validating on {len(val_shards)}"
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        scaler = _grad_scaler(self.device)

        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = -1
        best_duration_loss: Optional[float] = None
        stale = 0
        t_start = _now()
        stats: Dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            t_epoch = _now()
            train_loss = self._run_epoch(train_shards, optimizer, scaler, train=True)
            val_loss = self._run_epoch(val_shards, optimizer, scaler, train=False)
            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            improved = val_loss < best_val - 1e-4
            if improved:
                best_val = val_loss
                best_epoch = epoch
                best_duration_loss = self.last_duration_loss
                stale = 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                stale += 1
            if epoch % log_every == 0 or improved or epoch == 1:
                elapsed = _now() - t_start
                print(
                    f"[Draft] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}"
                    f"{'  *best*' if improved else ''} | lr {lr_now:.2e} | "
                    f"{_fmt_time(_now() - t_epoch)}/ep | total {_fmt_time(elapsed)}"
                )
            if stale >= patience and epoch >= int(min_epochs):
                print(f"[Draft] early stopping at epoch {epoch} (patience {patience})")
                break
            if (_now() - t_start) > max_minutes * 60.0:
                print(f"[Draft] time budget of {max_minutes:.1f} min reached at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.last_duration_loss = best_duration_loss
        # Primary metric is inference-honest: no external durations.  The GT
        # pass is an oracle-alignment ablation for diagnosing the duration head.
        gt_top1 = gt_top5 = None
        val_duration_shards = sum(1 for s in val_shards if s.has_durations)
        if self.model.duration_predictor is not None and val_duration_shards:
            gt_top1, gt_top5 = self.evaluate_topk(val_shards, use_gt_durations=True)
        top1, top5 = self.evaluate_topk(val_shards)
        diversity = getattr(self, "last_argmax_stats", {})
        duration_shards = sum(1 for s in shards if s.has_durations)
        stats.update(
            best_val_loss=float(best_val),
            best_epoch=int(best_epoch),
            epochs_run=int(epoch),
            val_top1=float(top1),
            val_top5=float(top5),
            duration_conditioning=bool(self.model.duration_predictor is not None),
            duration_loss_weight=float(self.duration_loss_weight),
            duration_shards=int(duration_shards),
            duration_shard_fraction=float(duration_shards / max(1, len(shards))),
            val_duration_shards=int(val_duration_shards),
            val_duration_loss=(
                float(self.last_duration_loss)
                if self.last_duration_loss is not None
                else None
            ),
            val_argmax_unique=float(diversity.get("unique", 0.0)),
            val_argmax_top_frac=float(diversity.get("top_frac", 1.0)),
            val_argmax_longest_run=float(diversity.get("longest_run", 0.0)),
            val_argmax_entropy_nats=float(diversity.get("entropy_nats", 0.0)),
            train_seconds=float(_now() - t_start),
        )
        if gt_top1 is not None:
            stats["val_top1_gt_durations"] = float(gt_top1)
            stats["val_top5_gt_durations"] = float(gt_top5)
        print(
            f"[Draft] done | best val {best_val:.4f} @ epoch {best_epoch} | "
            f"CB0 top-1 {top1 * 100:.2f}% | top-5 {top5 * 100:.2f}% | "
            f"argmax unique {stats['val_argmax_unique']:.1f}, "
            f"top {stats['val_argmax_top_frac']:.1%}, "
            f"run {stats['val_argmax_longest_run']:.0f}, "
            f"H {stats['val_argmax_entropy_nats']:.2f}"
        )
        if gt_top1 is not None:
            print(
                f"[Draft] oracle-duration ablation | CB0 top-1 {gt_top1 * 100:.2f}% | "
                f"top-5 {gt_top5 * 100:.2f}% | duration shards "
                f"{duration_shards}/{len(shards)}"
            )
        return stats


# ===========================================================================
# Residual CB1-8 training
# ===========================================================================

class ResidualTrainer:
    WINDOW = 86  # ~1 s of DAC frames; matches the streaming chunk size

    def __init__(
        self,
        residual_model: ResidualDecoder,
        device: torch.device,
        lr: float = 3e-4,
        batch_size: int = 32,
        seed: int = 42,
        draft_model: Optional[torch.nn.Module] = None,
        sched_target: float = 0.0,
    ):
        self.model = residual_model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        # Scheduled sampling: the residual is trained on teacher CB0 but runs on
        # *draft* CB0 at inference.  Feeding it draft-produced CB0 for a growing
        # fraction of windows closes that exposure bias (target 0 disables it).
        self.draft = draft_model
        self.sched_target = float(max(0.0, min(1.0, sched_target)))

    def precompute_draft_cb0(self, shards: List[TeacherShard]) -> Dict[int, np.ndarray]:
        """One NAR draft pass per shard: the CB0 the residual will actually see.

        Cheap relative to the teacher pass (a single forward per sentence) and
        only computed when scheduled sampling is enabled.
        """
        out: Dict[int, np.ndarray] = {}
        if self.draft is None or self.sched_target <= 0.0:
            return out
        self.draft.eval()
        for shard in shards:
            try:
                with torch.inference_mode():
                    draft_out = self.draft.generate_chunk(
                        text=shard.norm_text, num_frames=int(shard.frames),
                        temperature=1.0, sample=False, device=self.device,
                    )
                tokens = draft_out.tokens[0].detach().cpu().numpy().astype(np.int64)
                if tokens.shape[0] >= int(shard.frames):
                    out[int(shard.index)] = tokens[: int(shard.frames)]
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[Residual] draft pass failed for shard {shard.index}: "
                      f"{type(exc).__name__}: {exc}")
        return out

    def build_windows(self, shards: List[TeacherShard],
                      draft_cb0: Optional[Dict[int, np.ndarray]] = None):
        windows = []
        total_frames = 0
        windowed_frames = 0
        short_shards = 0
        for s in shards:
            t = int(s.frames)
            total_frames += t
            if t < self.WINDOW:
                # Cannot form even one full window; excluded from residual training.
                short_shards += 1
                continue
            windowed_frames += t  # windows tile [0, t) exactly, so coverage is full
            draft_tokens = (draft_cb0 or {}).get(int(s.index))
            for start in range(0, t, self.WINDOW):
                end = min(start + self.WINDOW, t)
                if end - start < self.WINDOW:
                    # Right-align the trailing window: every window must be exactly
                    # WINDOW frames or batch stacking (_batches) breaks on ragged tails.
                    start = t - self.WINDOW
                    end = t
                draft_slice = (draft_tokens[start:end]
                               if draft_tokens is not None else None)
                windows.append((s.codes[0, start:end], s.codes[1:9, start:end], draft_slice))
        dropped = total_frames - windowed_frames
        if dropped:
            print(f"[Residual] windows cover {windowed_frames}/{total_frames} teacher "
                  f"frames; {dropped} frames dropped ({short_shards} shard(s) shorter "
                  f"than {self.WINDOW} frames)")
        return windows

    def _batches(self, windows, shuffle: bool, sched_p: float = 0.0, rng=None):
        order = list(range(len(windows)))
        if shuffle:
            random.shuffle(order)
        rng = rng or random.Random(self.seed)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            inputs = []
            for j in idx:
                teacher_cb0, _, draft_cb0 = windows[j]
                if sched_p > 0.0 and draft_cb0 is not None and rng.random() < sched_p:
                    inputs.append(draft_cb0)
                else:
                    inputs.append(teacher_cb0)
            lengths = {int(x.shape[0]) for x in inputs}
            if len(lengths) > 1:
                raise ValueError(
                    f"residual batch mixes {sorted(lengths)}-frame windows; "
                    "build_windows must emit fixed-length windows"
                )
            cb0 = torch.stack([torch.from_numpy(x).long() for x in inputs]).to(self.device)
            tgt = torch.stack([torch.from_numpy(windows[j][1]).long() for j in idx]).to(self.device)
            yield cb0, tgt

    def _epoch(self, windows, optimizer, scaler, train: bool, sched_p: float = 0.0,
               rng=None):
        self.model.train(train)
        use_amp = self.device.type == "cuda"
        loss_sum, n_batches = 0.0, 0
        per_cb_loss = np.zeros(8, dtype=np.float64)
        for cb0, tgt in self._batches(windows, shuffle=train, sched_p=sched_p, rng=rng):
            with torch.set_grad_enabled(train):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    out = self.model(cb0_tokens=cb0, sample=False)
                    logits = out.logits  # (B, 8, T, V)
                    b, c, t, v = logits.shape
                    loss = sum(
                        torch.nn.functional.cross_entropy(
                            logits[:, i].reshape(-1, v), tgt[:, i].reshape(-1)
                        )
                        for i in range(c)
                    ) / c
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
            loss_sum += float(loss.detach().cpu())
            n_batches += 1
        return loss_sum / max(1, n_batches), per_cb_loss

    @torch.inference_mode()
    def evaluate(self, windows, sched_p: float = 0.0, rng=None) -> Tuple[float, List[float]]:
        """CB1-8 top-1 on either teacher (sched_p=0) or draft CB0 (sched_p=1)."""
        self.model.eval()
        top1 = np.zeros(8, dtype=np.float64)
        total = 0
        for cb0, tgt in self._batches(windows, shuffle=False, sched_p=sched_p, rng=rng):
            out = self.model(cb0_tokens=cb0, sample=False)
            pred = out.residual_codebooks  # (B, 8, T)
            top1 += (pred == tgt).float().sum(dim=(0, 2)).cpu().numpy()
            total += tgt.shape[0] * tgt.shape[2]
        return float(total), (top1 / max(1, total)).tolist()

    def train(
        self,
        shards: List[TeacherShard],
        epochs: int = 60,
        patience: int = 10,
        max_minutes: float = 6.0,
        sched_target: float = 0.0,
        min_epochs: int = 0,
        ramp_epochs: int = 0,
    ) -> Dict[str, Any]:
        target = float(max(0.0, min(1.0, sched_target if sched_target > 0 else self.sched_target)))
        # precompute_draft_cb0 gates on self.sched_target; keep the trainer and
        # the explicit train() argument in sync.
        self.sched_target = target
        ramp_epochs = max(1, min(int(ramp_epochs or epochs), epochs))
        draft_cb0 = self.precompute_draft_cb0(shards) if target > 0 else {}
        train_shards = [s for i, s in enumerate(shards) if i % 10 != 9]
        val_shards = [s for i, s in enumerate(shards) if i % 10 == 9]
        train_w = self.build_windows(train_shards, draft_cb0)
        val_w = self.build_windows(val_shards, draft_cb0)
        if not val_w and train_w:
            # Degenerate corpus: keep at least one validation window rather than
            # dividing by zero. This is the only allowed train/val overlap.
            val_w = train_w[-1:]
        if not train_w:
            raise RuntimeError("No residual training windows could be built.")
        sched_rng = random.Random(self.seed + 17)
        val_rng = random.Random(self.seed + 23)
        # Checkpoint selection must track the inference distribution.  Before
        # this fix validation always used teacher CB0 (sched_p=0), so the best
        # state was an early, teacher-only model even when training had ramped
        # to draft CB0 later.
        val_sched_p = 1.0 if (target > 0 and draft_cb0) else 0.0
        val_label = "draft CB0" if val_sched_p > 0 else "teacher CB0"
        print(
            f"[Residual] {len(train_w) + len(val_w)} windows of {self.WINDOW} frames "
            f"({len(train_w)} train / {len(val_w)} val, shard-disjoint)"
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        scaler = _grad_scaler(self.device)
        # Track the best draft-CB0 state over all epochs for diagnostics, and
        # separately the best state during the full-draft hold.  The latter is
        # what we deploy: selecting a pre-ramp state is exactly the v2 failure
        # mode (best_epoch=11 at 3.3 % draft exposure).
        best_any_val = float("inf")
        best_any_state = None
        best_any_epoch = 0
        best_any_sched_p = 0.0
        best_full_val = float("inf")
        best_full_state = None
        best_full_epoch = 0
        epochs_completed = 0
        stale = 0
        t_start = _now()
        stats: Dict[str, Any] = {}
        if target > 0:
            if not draft_cb0:
                raise RuntimeError(
                    "Scheduled sampling was requested but no draft CB0 is available "
                    "(the draft pass failed or --skip-draft was used without a usable "
                    "--draft-ckpt). Refusing to silently train on teacher CB0 only."
                )
            print(f"[Residual] scheduled sampling: {len(draft_cb0)}/{len(shards)} shards have "
                  f"draft CB0, ramp 0 -> {target:.2f} over {ramp_epochs} epochs")
        print(f"[Residual] checkpoint selection: validation on {val_label}")
        for epoch in range(1, epochs + 1):
            epochs_completed = epoch
            ramp_frac = (1.0 if ramp_epochs <= 1 else
                         min(1.0, max(0.0, (epoch - 1) / (ramp_epochs - 1))))
            sched_p = target * ramp_frac if target > 0 and draft_cb0 else 0.0
            train_loss, _ = self._epoch(train_w, optimizer, scaler, train=True,
                                        sched_p=sched_p, rng=sched_rng)
            val_loss, _ = self._epoch(val_w, optimizer, scaler, train=False,
                                      sched_p=val_sched_p, rng=val_rng)
            scheduler.step(val_loss)
            improved = val_loss < best_any_val - 1e-4
            if improved:
                best_any_val = val_loss
                best_any_epoch = epoch
                best_any_sched_p = sched_p
                stale = 0
                best_any_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                stale += 1
            full_now = bool(target > 0 and draft_cb0 and sched_p >= target - 1e-9)
            if full_now and val_loss < best_full_val - 1e-4:
                best_full_val = val_loss
                best_full_epoch = epoch
                best_full_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Residual] epoch {epoch:3d} | train {train_loss:.4f} (sched {sched_p:.2f}) "
                f"| val {val_loss:.4f} [{val_label}]"
                f"{'  *best*' if improved else ''} | lr {lr_now:.2e} "
                f"| total {_fmt_time(_now() - t_start)}"
            )
            if stale >= patience and epoch >= int(min_epochs):
                print(f"[Residual] early stopping at epoch {epoch}")
                break
            if (_now() - t_start) > max_minutes * 60.0:
                print(f"[Residual] time budget of {max_minutes:.1f} min reached")
                break
        if best_full_state is not None:
            selected_state = best_full_state
            selected_val = best_full_val
            selected_epoch = best_full_epoch
            selected_sched_p = float(target)
            selected_full_draft = True
        elif best_any_state is not None:
            selected_state = best_any_state
            selected_val = best_any_val
            selected_epoch = best_any_epoch
            selected_sched_p = best_any_sched_p
            selected_full_draft = False
        else:
            selected_state = None
            selected_val = float("inf")
            selected_epoch = 0
            selected_sched_p = 0.0
            selected_full_draft = False
        if selected_state is not None:
            self.model.load_state_dict(selected_state)
        _, top1_teacher = self.evaluate(val_w, sched_p=0.0)
        _, top1_draft = self.evaluate(val_w, sched_p=1.0) if draft_cb0 else (0.0, top1_teacher)
        val_loss_teacher, _ = self._epoch(val_w, optimizer, scaler, train=False,
                                          sched_p=0.0, rng=val_rng)
        final_ramp_frac = (1.0 if ramp_epochs <= 1 else
                           min(1.0, max(0.0, (epochs_completed - 1) / (ramp_epochs - 1))))
        final_sched_p = target * final_ramp_frac if target > 0 and draft_cb0 else 0.0
        selection_metric = "teacher_cb0_val_loss"
        if val_sched_p > 0:
            selection_metric = ("draft_cb0_val_loss_full_ramp" if selected_full_draft
                                else "draft_cb0_val_loss_pre_full_ramp")
        stats.update(
            best_val_loss=float(selected_val),
            best_val_loss_teacher=float(val_loss_teacher),
            best_epoch=int(selected_epoch),
            best_sched_p=float(selected_sched_p),
            selected_full_draft=bool(selected_full_draft),
            best_val_sched_p=float(val_sched_p),
            selection_metric=selection_metric,
            final_sched_p=float(final_sched_p),
            ramp_epochs=int(ramp_epochs),
            epochs_run=int(epochs_completed),
            best_val_loss_any=float(best_any_val),
            best_epoch_any=int(best_any_epoch),
            best_sched_p_any=float(best_any_sched_p),
            best_full_epoch=int(best_full_epoch) if best_full_state is not None else 0,
            best_full_val_loss=(float(best_full_val) if best_full_state is not None else None),
            cb_top1=top1_teacher,
            cb_top1_mean=float(np.mean(top1_teacher)),
            cb_top1_teacher=top1_teacher,
            cb_top1_teacher_mean=float(np.mean(top1_teacher)),
            cb_top1_draft=top1_draft,
            cb_top1_draft_mean=float(np.mean(top1_draft)),
            train_seconds=float(_now() - t_start),
        )
        print(
            "[Residual] done | selected val {:.4f} @ epoch {} (sched {:.2f}, {}) | "
            "mean CB1-8 top-1 teacher {:.2f}% draft {:.2f}%".format(
                selected_val, selected_epoch, selected_sched_p,
                "full-draft hold" if selected_full_draft else "pre-ramp fallback",
                100 * float(np.mean(top1_teacher)),
                100 * float(np.mean(top1_draft)),
            )
        )
        return stats


# ===========================================================================
# Checkpoint IO
# ===========================================================================

def save_checkpoint(path: Path, model: torch.nn.Module, config_obj, meta: Dict) -> None:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dataclasses.asdict(config_obj),
        "meta": meta,
    }
    torch.save(payload, path)
    print(f"[Save] {path} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")


def resolve_checkpoint_file(path: str, filename: str) -> Path:
    """Accept either a checkpoint directory or the checkpoint file itself."""
    p = Path(path)
    if p.is_dir():
        p = p / filename
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def load_draft_checkpoint(path: str, device: torch.device) -> Tuple[DraftTransformer, Path, Dict[str, Any]]:
    """Load a trained draft for a residual-only run.

    ``--skip-draft`` used to set ``draft_model=None``, silently disabling
    scheduled sampling and training the residual on teacher CB0 only.  This
    loader makes the existing v2 draft an explicit, checked input.
    """
    file = resolve_checkpoint_file(path, "draft_cb0.pt")
    payload = torch.load(file, map_location="cpu", weights_only=False)
    cfg = DraftModelConfig(**payload.get("config", {}))
    model = DraftTransformer(cfg).to(device)
    model.load_state_dict({
        k.removeprefix("module."): v for k, v in payload["state_dict"].items()
    })
    model.eval()
    print(f"[Draft] loaded {file} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")
    return model, file, payload


def load_draft_source_meta(file: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Carry the source draft's stats into the residual-only output meta."""
    meta: Dict[str, Any] = {}
    parent_meta = file.parent / "meta.json"
    if parent_meta.is_file():
        try:
            doc = json.loads(parent_meta.read_text(encoding="utf-8"))
            if isinstance(doc.get("draft"), dict):
                meta.update(doc["draft"])
        except Exception as exc:  # pragma: no cover - malformed source meta
            print(f"[Draft] warning: could not read {parent_meta}: {exc}")
    if not meta and isinstance(payload.get("meta"), dict):
        meta.update(payload["meta"])
    meta["loaded_from"] = str(file)
    meta["reused"] = True
    return meta


def copy_draft_checkpoint(source: Path, ckpt_dir: Path) -> Path:
    """Keep the residual-only output a complete, evaluable checkpoint dir."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dst = ckpt_dir / "draft_cb0.pt"
    if source.resolve() == dst.resolve():
        return dst
    shutil.copy2(source, dst)
    print(f"[Save] {dst} (copied from {source})")
    return dst


# ===========================================================================
# Dry-run mode (no Parler needed) — exercises the full training path
# ===========================================================================

def run_dry_run(args) -> None:
    print("=" * 78)
    print("[Dry-run] Fabricating teacher shards (no Parler / GPU required).")
    print("=" * 78)
    set_seed(args.seed)
    corpus = build_training_corpus(n=args.num_sentences, seed=args.seed)
    assert_corpus_disjoint_from_eval(corpus)
    normalizer = TanglishNormalizer()
    out_dir = Path(args.teacher_cache) / "dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    shards: List[TeacherShard] = []
    for i, text in enumerate(corpus):
        norm = normalizer.normalize(text).normalized_text
        frames = max(30, int(len(norm) * 6.6) + rng.randint(-8, 8))
        t = np.arange(frames)
        codes = np.stack(
            [
                ((t * 3 + len(norm) * 7) % 1024).astype(np.int64),
                *[((t * (c + 2) + i * 5) % 1024).astype(np.int64) for c in range(8)],
            ]
        )
        shards.append(
            TeacherShard(
                index=i,
                raw_text=text,
                norm_text=norm,
                n_chars=len(norm),
                frames=frames,
                codes=codes,
                audio_sec=frames / 86.13,
                target_source="dryrun",
                gen_ms=0.0,
            )
        )
    length_model = LengthModel.fit([s.n_chars for s in shards], [s.frames for s in shards])
    print(f"[Dry-run] {len(shards)} fabricated shards | length MAE {length_model.mae_frames:.1f} frames")

    device = torch.device("cpu")
    config = CodeSwitchTTSConfig()
    draft = DraftTransformer(config.draft).to(device)
    residual = ResidualDecoder(config.residual).to(device)

    draft_stats = DraftTrainer(draft, device, batch_size=8, seed=args.seed).train(
        shards, epochs=3, patience=3, max_minutes=2.0
    )
    residual_stats = ResidualTrainer(
        residual, device, batch_size=8, seed=args.seed,
        draft_model=draft, sched_target=args.residual_sched,
    ).train(shards, epochs=3, patience=3, max_minutes=2.0, sched_target=args.residual_sched)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": "dry-run",
        "length_model": length_model.to_dict(),
        "draft": draft_stats,
        "residual": residual_stats,
        "versions": package_versions(),
    }
    save_checkpoint(ckpt_dir / "draft_cb0.pt", draft, config.draft,
                    {"dry_run": True, **draft_stats, "param_count": sum(p.numel() for p in draft.parameters())})
    save_checkpoint(ckpt_dir / "residual_cb1_8.pt", residual, config.residual,
                    {"dry_run": True, **residual_stats, "param_count": sum(p.numel() for p in residual.parameters())})
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Dry-run] PASS — checkpoints written to {ckpt_dir}")


# ===========================================================================
# Main
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Self-distillation trainer for CodeSwitch-TTS draft/residual models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Run the full training path on fabricated data (no Parler/GPU needed).")
    p.add_argument("--corpus-json", type=str, default=None,
                   help="Optional external training corpus (one sentence per line or JSON list).")
    p.add_argument("--num-sentences", type=int, default=300,
                   help="Number of training sentences (bootstrap corpus size).")
    p.add_argument("--target-source", choices=["auto", "lm", "codec"], default="auto",
                   help="Teacher token target: LM-sampled captures, DAC re-encode, or auto.")
    p.add_argument("--teacher-cache", type=str, default="teacher_cache",
                   help="Directory for cached teacher code shards.")
    p.add_argument("--ckpt-dir", type=str, default="codeswitch_ckpt",
                   help="Output directory for trained checkpoints.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (default: cuda:0 when available).")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                   help="Parler teacher precision on GPU (fp16 is T4-native).")
    p.add_argument("--draft-epochs", type=int, default=80)
    p.add_argument("--draft-batch", type=int, default=24)
    p.add_argument("--draft-lr", type=float, default=3e-4)
    p.add_argument("--draft-frame-decoder-layers", type=int, default=0,
                   help="Add a text-conditioned frame decoder with this many layers. "
                        "0 keeps the legacy temporal projector (default).")
    p.add_argument("--draft-duration-conditioning", action="store_true",
                   help="Add a duration head and feed CTC-derived per-character durations "
                        "into the frame decoder. Requires --draft-frame-decoder-layers >= 1.")
    p.add_argument("--draft-duration-loss-weight", type=float, default=0.1,
                   help="Weight for the duration-prediction smooth-L1 loss.")
    p.add_argument("--require-durations", action="store_true",
                   help="Refuse to train on any shard without per-character durations.")
    p.add_argument("--max-minutes", type=float, default=12.0,
                   help="Wall-clock budget for draft training.")
    p.add_argument("--min-epochs", type=int, default=0,
                   help="Epochs that must elapse before early stopping can fire (both stages). "
                        "The 300-sentence pilot early-stopped at epoch 10 of ~9 batches/epoch "
                        "(~200 optimizer steps) and its residual never left the uniform-noise "
                        "regime; a floor of 150-400 turns the same wall-clock budget into a "
                        "trained student. --max-minutes* still caps the run.")
    p.add_argument("--residual-epochs", type=int, default=60)
    p.add_argument("--residual-batch", type=int, default=32)
    p.add_argument("--residual-lr", type=float, default=3e-4)
    p.add_argument("--residual-sched", type=float, default=0.0,
                   help="Scheduled-sampling target fraction (0-1.0). The residual is trained on "
                        "teacher CB0 but runs on draft CB0, so a ramp to 1.0 closes that exposure "
                        "bias; requires an in-run draft or --draft-ckpt (default: off).")
    p.add_argument("--residual-ramp-epochs", type=int, default=0,
                   help="Reach the scheduled-sampling target by this epoch and hold it for the "
                        "remaining epochs. 0 ramps over all --residual-epochs.")
    p.add_argument("--residual-d-model", type=int, default=0,
                   help="Override ResidualModelConfig.d_model (0 keeps the 384 default).")
    p.add_argument("--residual-layers", type=int, default=0,
                   help="Override ResidualModelConfig.num_layers (0 keeps the 4 default; "
                        "the decoder supports up to 6 dilation rates).")
    p.add_argument("--max-minutes-residual", type=float, default=6.0,
                   help="Wall-clock budget for residual training.")
    p.add_argument("--skip-teacher", action="store_true", help="Reuse cached shards only.")
    p.add_argument("--skip-draft", action="store_true",
                   help="Skip draft training; requires --draft-ckpt for a residual run so the "
                        "output checkpoint is complete and scheduled sampling is real.")
    p.add_argument("--draft-ckpt", type=str, default=None,
                   help="Existing draft checkpoint directory or draft_cb0.pt file to load when "
                        "--skip-draft. It is copied into --ckpt-dir for evaluation.")
    p.add_argument("--skip-residual", action="store_true")
    p.add_argument("--num-workers-note", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t_all = _now()

    if args.dry_run:
        run_dry_run(args)
        print(f"[Done] dry-run total {_fmt_time(_now() - t_all)}")
        return 0

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = dtype_map[args.dtype]

    config = CodeSwitchTTSConfig()
    config.system.allow_untrained_fallback = False  # teacher MUST be real speech
    if args.draft_frame_decoder_layers < 0 or args.draft_frame_decoder_layers > 6:
        print("[Draft] --draft-frame-decoder-layers must be between 0 and 6")
        return 2
    config.draft.frame_decoder_layers = int(args.draft_frame_decoder_layers)
    if args.draft_duration_conditioning:
        if config.draft.frame_decoder_layers < 1:
            print(
                "[Draft] --draft-duration-conditioning requires "
                "--draft-frame-decoder-layers >= 1"
            )
            return 2
        config.draft.duration_conditioning = True
    if args.draft_duration_loss_weight < 0.0:
        print("[Draft] --draft-duration-loss-weight must be >= 0")
        return 2
    if args.residual_d_model:
        if args.residual_d_model < 64:
            print("[Residual] --residual-d-model must be >= 64")
            return 2
        config.residual.d_model = int(args.residual_d_model)
    if args.residual_layers:
        if args.residual_layers < 1 or args.residual_layers > 6:
            print("[Residual] --residual-layers must be between 1 and 6")
            return 2
        config.residual.num_layers = int(args.residual_layers)

    print("=" * 78)
    print("CodeSwitch-TTS · Draft/Residual Self-Distillation")
    print(f"  device: {device} | teacher dtype: {dtype} | seed: {args.seed}")
    print(f"  corpus: {args.num_sentences} sentences | target: {args.target_source}")
    print("=" * 78)

    # ---------------- corpus ----------------
    if args.corpus_json:
        corpus = load_corpus_json(args.corpus_json)
        if len(corpus) > args.num_sentences:
            random.Random(args.seed).shuffle(corpus)
            corpus = corpus[: args.num_sentences]
        print(f"[Corpus] loaded {len(corpus)} sentences from {args.corpus_json}")
    else:
        corpus = build_training_corpus(n=args.num_sentences, seed=args.seed)
        print(f"[Corpus] generated {len(corpus)} bootstrap sentences (seed {args.seed})")
    assert_corpus_disjoint_from_eval(corpus)

    normalizer = TanglishNormalizer()
    teacher_dir = Path(args.teacher_cache)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- teacher pass ----------------
    engine = None
    shards: List[TeacherShard] = []
    if args.skip_teacher:
        shard_files = sorted(teacher_dir.glob("shard_*.npz"))
        if not shard_files:
            print(f"[Teacher] --skip-teacher set but no shards in {teacher_dir}")
            return 2
        seen: Dict[int, str] = {}
        for f in shard_files:
            s = TeacherShard.load(f)
            if s.codes.ndim != 2 or s.codes.shape[0] != 9:
                raise SystemExit(
                    f"[Teacher] {f.name}: codes shape {s.codes.shape}; expected (9, T)"
                )
            if int(s.frames) != int(s.codes.shape[1]):
                raise SystemExit(
                    f"[Teacher] {f.name}: frames={s.frames} but codes has "
                    f"{s.codes.shape[1]} timesteps; target would be truncated"
                )
            if int(s.codes.min()) < 0 or int(s.codes.max()) >= 1024:
                raise SystemExit(
                    f"[Teacher] {f.name}: codes outside [0, 1024): "
                    f"min={int(s.codes.min())} max={int(s.codes.max())}"
                )
            if int(s.index) in seen:
                raise SystemExit(
                    f"[Teacher] duplicate shard index {s.index}: "
                    f"{f.name} and {seen[int(s.index)]}"
                )
            seen[int(s.index)] = f.name
            shards.append(s)
        if len({s.target_source for s in shards}) > 1:
            print("[Teacher] ⚠ mixed target_source in cache; targets were captured "
                  "with different methods")
        print(f"[Teacher] loaded {len(shards)} cached shards")
    else:
        from codeswitch_tts import SpeculativeEngine

        engine = SpeculativeEngine(config.speculative, device=device, dtype=dtype)
        if not engine.speech_available:
            print(
                "\n[Teacher] ❌ Indic Parler-TTS is not available, so no teacher signal can be\n"
                "          collected. Apply the Kaggle fix (install protobuf==5.29.5 last,\n"
                "          RESTART the kernel), then re-run. For a local smoke test use:\n"
                "            python train_draft_distill.py --dry-run\n"
            )
            return 2
        collector = TeacherCollector(
            engine=engine,
            out_dir=teacher_dir,
            normalizer=normalizer,
            seed=args.seed,
            target_source=args.target_source,
        )
        try:
            shards, length_model = collector.collect(corpus)
        except KeyboardInterrupt:
            print("[Teacher] interrupted — partial shards are cached; re-run to resume.")
            return 130

    duration_shards = sum(1 for s in shards if s.has_durations)
    if args.require_durations and duration_shards < len(shards):
        missing = [s for s in shards if not s.has_durations]
        print(
            f"[Draft] ❌ --require-durations: {len(missing)}/{len(shards)} shards "
            f"have no CTC durations (first: {missing[0].norm_text[:60]!r}). "
            "Run prepare_real_speech.py with --durations ctc, or drop the flag."
        )
        return 2
    if args.draft_duration_conditioning:
        print(
            f"[Draft] duration supervision: {duration_shards}/{len(shards)} shards "
            f"({duration_shards / max(1, len(shards)):.1%})"
        )

    # Recompute the length model over whatever shards exist (resume-safe).
    length_model = LengthModel.fit([s.n_chars for s in shards], [s.frames for s in shards])
    corpus_hours = float(sum(float(s.audio_sec) for s in shards) / 3600.0)
    print(f"[Data] corpus {len(shards)} shards, {corpus_hours:.3f} h")
    if corpus_hours < 1.0:
        print(
            "[Data] WARNING: corpus is under 1 h; a failed 5% residual gate is a "
            "data-scale result first, not a model-architecture verdict. Use a larger "
            "real slice (the notebook default is 5000 rows capped at 6 h)."
        )

    # ---------------- trainers ----------------
    draft: Optional[DraftTransformer] = None
    draft_source_path: Optional[Path] = None
    draft_stats: Dict[str, Any] = {}
    residual_stats: Dict[str, Any] = {}

    if args.skip_draft and not args.skip_residual and not args.draft_ckpt:
        print(
            "[Draft] ❌ --skip-draft requires --draft-ckpt for a residual run: the output\n"
            "          checkpoint must contain a draft, and scheduled sampling must use the\n"
            "          real draft rather than silently falling back to teacher CB0."
        )
        return 2

    if not args.skip_draft:
        draft = DraftTransformer(config.draft).to(device)
        draft_stats = DraftTrainer(
            draft,
            device,
            lr=args.draft_lr,
            batch_size=args.draft_batch,
            seed=args.seed,
            duration_loss_weight=args.draft_duration_loss_weight,
        ).train(shards, epochs=args.draft_epochs, max_minutes=args.max_minutes,
                min_epochs=args.min_epochs)
    elif args.draft_ckpt:
        draft, draft_source_path, draft_source_payload = load_draft_checkpoint(
            args.draft_ckpt, device
        )
        draft_stats = load_draft_source_meta(draft_source_path, draft_source_payload)
        if Path(args.ckpt_dir).resolve() == draft_source_path.parent.resolve():
            print(
                "[Save] ❌ --ckpt-dir resolves to the draft source directory; refusing to "
                "overwrite the previous checkpoint. Use a new directory (e.g. "
                "codeswitch_ckpt_v3)."
            )
            return 2

    if not args.skip_residual:
        residual = ResidualDecoder(config.residual).to(device)
        residual_stats = ResidualTrainer(
            residual, device, lr=args.residual_lr, batch_size=args.residual_batch, seed=args.seed,
            draft_model=draft,
            sched_target=args.residual_sched,
        ).train(shards, epochs=args.residual_epochs, max_minutes=args.max_minutes_residual,
                sched_target=args.residual_sched, min_epochs=args.min_epochs,
                ramp_epochs=args.residual_ramp_epochs)

    # ---------------- checkpoints ----------------
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_draft and draft is not None:
        save_checkpoint(
            ckpt_dir / "draft_cb0.pt", draft, config.draft,
            {"param_count": sum(p.numel() for p in draft.parameters()), **draft_stats},
        )
    elif draft_source_path is not None:
        copy_draft_checkpoint(draft_source_path, ckpt_dir)
    if not args.skip_residual:
        save_checkpoint(
            ckpt_dir / "residual_cb1_8.pt", residual, config.residual,
            {"param_count": sum(p.numel() for p in residual.parameters()), **residual_stats},
        )

    meta = {
        "corpus_size": len(shards),
        "corpus_hours": float(corpus_hours),
        "corpus_source": args.corpus_json or f"bootstrap(seed={args.seed})",
        "target_source": args.target_source,
        "target_stats": getattr(engine, "backend", "n/a"),
        "duration_conditioning": bool(
            draft is not None and draft.config.duration_conditioning
        ),
        "duration_shards": int(duration_shards),
        "duration_shard_fraction": float(duration_shards / max(1, len(shards))),
        "residual_scheduled_sampling": args.residual_sched,
        "residual_ramp_epochs": int(args.residual_ramp_epochs or args.residual_epochs),
        "residual_only": bool(args.skip_draft),
        "draft_source": str(draft_source_path) if draft_source_path is not None else None,
        "residual_capacity": {
            "d_model": int(config.residual.d_model),
            "num_layers": int(config.residual.num_layers),
        },
        "length_model": length_model.to_dict(),
        "draft": draft_stats,
        "residual": residual_stats,
        "versions": package_versions(),
        "device": str(device),
        "dtype": str(dtype),
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------- sanity render ----------------
    if engine is not None and shards and not args.skip_draft and not args.skip_residual:
        try:
            sanity_text = shards[-1].norm_text
            t_frames = length_model.predict(sanity_text)
            with torch.inference_mode():
                draft_out = draft.generate_chunk(
                    text=sanity_text, num_frames=t_frames, temperature=1.0, device=device
                )
                res_out = residual(cb0_tokens=draft_out.tokens, sample=True)
                audio_enc = engine.ar_model.audio_encoder
                wave, decode_err = _decode_codes_for_validation(
                    audio_enc, res_out.full_codebooks[0]
                )
                if wave is None:
                    print(f"[Sanity] decode failed: {decode_err}")
            if wave is not None:
                import soundfile as sf

                from codeswitch_tts.utils.audio import apply_highpass_filter, normalize_peak

                wave = normalize_peak(
                    apply_highpass_filter(wave, sample_rate=config.audio.sample_rate),
                    target_peak=0.92,
                )
                wav_path = ckpt_dir / "sanity_codeswitch_sample.wav"
                sf.write(str(wav_path), wave, config.audio.sample_rate, subtype="PCM_16")
                ref_codes = shards[-1].codes[0]
                n_cmp = int(min(t_frames, ref_codes.shape[0]))
                draft_top1 = (
                    float((draft_out.tokens[0, :n_cmp].cpu().numpy() == ref_codes[:n_cmp]).mean())
                    if n_cmp
                    else 0.0
                )
                print(
                    f"[Sanity] rendered {wav_path} | draft CB0 top-1 vs teacher "
                    f"{draft_top1 * 100:.1f}% | text: {sanity_text[:60]}"
                )
        except Exception as exc:
            print(f"[Sanity] skipped: {type(exc).__name__}: {exc}")

    print("=" * 78)
    print(f"[Done] total {_fmt_time(_now() - t_all)}")
    saved_checkpoints = [
        path for path in (ckpt_dir / "draft_cb0.pt", ckpt_dir / "residual_cb1_8.pt")
        if path.is_file()
    ]
    print(
        "  checkpoints :",
        ", ".join(str(path) for path in saved_checkpoints)
        if saved_checkpoints else "none (metadata only)",
    )
    print(f"  meta        : {ckpt_dir}/meta.json  (length model + training stats)")
    print("  next        : python run_full_evaluation.py --ckpt codeswitch_ckpt")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
