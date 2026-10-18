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
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np

# Importing codeswitch_tts installs the protobuf runtime_version fallback shim
# first (Kaggle stale-protobuf safety net) before parler_tts/transformers load.
from codeswitch_tts import (  # noqa: E402
    CodeSwitchTTSConfig,
    DraftTransformer,
    ResidualDecoder,
    TanglishNormalizer,
)
from codeswitch_benchmark_data import (  # noqa: E402
    assert_corpus_disjoint_from_eval,
    build_training_corpus,
    load_corpus_json,
)

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

def _same_sentence(a: str, b: str) -> bool:
    """Whitespace/case-insensitive sentence identity for teacher-cache validation."""
    norm = lambda t: " ".join(str(t or "").split()).casefold()  # noqa: E731
    return norm(a) == norm(b)


@dataclass
class TeacherShard:
    """One teacher utterance: normalized text + 9xT DAC codes."""

    index: int
    raw_text: str
    norm_text: str
    n_chars: int
    frames: int
    codes: np.ndarray  # (9, T) int16
    audio_sec: float
    target_source: str
    gen_ms: float

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            index=self.index,
            raw_text=self.raw_text,
            norm_text=self.norm_text,
            n_chars=self.n_chars,
            frames=self.frames,
            codes=self.codes.astype(np.int16),
            audio_sec=self.audio_sec,
            target_source=self.target_source,
            gen_ms=self.gen_ms,
        )

    @classmethod
    def load(cls, path: Path) -> "TeacherShard":
        data = np.load(path, allow_pickle=False)
        return cls(
            index=int(data["index"]),
            raw_text=str(data["raw_text"]),
            norm_text=str(data["norm_text"]),
            n_chars=int(data["n_chars"]),
            frames=int(data["frames"]),
            codes=data["codes"].astype(np.int64),
            audio_sec=float(data["audio_sec"]),
            target_source=str(data["target_source"]),
            gen_ms=float(data["gen_ms"]),
        )


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

    # -- main loop -----------------------------------------------------------
    def collect(self, corpus: Sequence[str]) -> Tuple[List[TeacherShard], LengthModel]:
        codebook_size = int(self.engine.ar_model.audio_encoder.config.codebook_size)
        total = len(corpus)
        shards: List[TeacherShard] = []
        t_start = _now()
        for i, text in enumerate(corpus):
            shard_path = self.out_dir / f"shard_{i:05d}.npz"
            if shard_path.exists():
                cached = TeacherShard.load(shard_path)
                if _same_sentence(cached.raw_text, text):
                    shards.append(cached)
                    continue
                # The cache is indexed by position, so a changed corpus would
                # otherwise train the student on the *previous* sentence's
                # teacher codes.  Recollect the mismatching positions.
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
                + (f" ({self.stats['stale']} recollected after a corpus change)"
                   if self.stats.get("stale") else "") + " | "
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

class DraftTrainer:
    def __init__(
        self,
        draft_model: DraftTransformer,
        device: torch.device,
        lr: float = 3e-4,
        batch_size: int = 24,
        seed: int = 42,
    ):
        self.model = draft_model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = draft_model.tokenizer

    def _collate(self, batch: List[TeacherShard]):
        texts = [s.norm_text for s in batch]
        input_ids, attn = self.tokenizer.batch_encode(texts)
        max_t = max(s.frames for s in batch)
        targets = torch.full((len(batch), max_t), -100, dtype=torch.long)
        for i, s in enumerate(batch):
            t = s.codes[0]
            targets[i, : t.shape[0]] = torch.from_numpy(t).long()
        return (
            input_ids.to(self.device),
            attn.to(self.device),
            targets.to(self.device),
            max_t,
        )

    def _run_epoch(self, shards, optimizer, scaler, train: bool):
        self.model.train(train)
        order = list(range(len(shards)))
        if train:
            random.shuffle(order)
        total_loss, total_frames = 0.0, 0
        use_amp = self.device.type == "cuda"
        for start in range(0, len(order), self.batch_size):
            batch = [shards[j] for j in order[start:start + self.batch_size]]
            input_ids, attn, targets, max_t = self._collate(batch)
            with torch.set_grad_enabled(train):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        num_frames=max_t,
                        sample=False,
                    )
                    loss = torch.nn.functional.cross_entropy(
                        out.logits.reshape(-1, out.logits.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=-100,
                    )
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
        return total_loss / max(1, total_frames)

    @torch.inference_mode()
    def evaluate_topk(self, shards, k: int = 5) -> Tuple[float, float]:
        self.model.eval()
        top1 = topk = total = 0
        for start in range(0, len(shards), self.batch_size):
            batch = shards[start:start + self.batch_size]
            input_ids, attn, targets, max_t = self._collate(batch)
            out = self.model(
                input_ids=input_ids, attention_mask=attn, num_frames=max_t, sample=False
            )
            logits = out.logits
            for i, s in enumerate(batch):
                t = s.codes[0]
                lg = logits[i, : t.shape[0]]
                pred1 = lg.argmax(dim=-1)
                top1 += int((pred1 == torch.from_numpy(t).long().to(pred1.device)).sum().item())
                topk_pred = lg.topk(min(k, lg.shape[-1]), dim=-1).indices
                topk += int(
                    (topk_pred == torch.from_numpy(t).long().to(pred1.device).unsqueeze(-1))
                    .any(dim=-1)
                    .sum()
                    .item()
                )
                total += t.shape[0]
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
            if stale >= patience:
                print(f"[Draft] early stopping at epoch {epoch} (patience {patience})")
                break
            if (_now() - t_start) > max_minutes * 60.0:
                print(f"[Draft] time budget of {max_minutes:.1f} min reached at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        top1, top5 = self.evaluate_topk(val_shards)
        stats.update(
            best_val_loss=float(best_val),
            best_epoch=int(best_epoch),
            epochs_run=int(epoch),
            val_top1=float(top1),
            val_top5=float(top5),
            train_seconds=float(_now() - t_start),
        )
        print(
            f"[Draft] done | best val {best_val:.4f} @ epoch {best_epoch} | "
            f"CB0 top-1 {top1 * 100:.2f}% | top-5 {top5 * 100:.2f}%"
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
        self.sched_target = float(max(0.0, min(0.9, sched_target)))

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
                        temperature=1.0, device=self.device,
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
        for s in shards:
            t = s.frames
            draft_tokens = (draft_cb0 or {}).get(int(s.index))
            for start in range(0, max(1, t - self.WINDOW + 1), self.WINDOW):
                end = min(start + self.WINDOW, t)
                if end - start < 24:
                    continue
                draft_slice = (draft_tokens[start:end]
                               if draft_tokens is not None else None)
                windows.append((s.codes[0, start:end], s.codes[1:9, start:end], draft_slice))
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
    def evaluate(self, windows) -> Tuple[float, List[float]]:
        self.model.eval()
        top1 = np.zeros(8, dtype=np.float64)
        total = 0
        for cb0, tgt in self._batches(windows, shuffle=False):
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
    ) -> Dict[str, float]:
        target = float(max(0.0, min(0.9, sched_target if sched_target > 0 else self.sched_target)))
        draft_cb0 = self.precompute_draft_cb0(shards) if target > 0 else {}
        windows = self.build_windows(shards, draft_cb0)
        if not windows:
            raise RuntimeError("No residual training windows could be built.")
        sched_rng = random.Random(self.seed + 17)
        train_w = [w for i, w in enumerate(windows) if i % 10 != 9]
        val_w = [w for i, w in enumerate(windows) if i % 10 == 9] or windows[-1:]
        print(
            f"[Residual] {len(windows)} windows of {self.WINDOW} frames "
            f"({len(train_w)} train / {len(val_w)} val)"
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        scaler = _grad_scaler(self.device)
        best_val = float("inf")
        best_state = None
        stale = 0
        t_start = _now()
        stats: Dict[str, float] = {}
        if target > 0:
            if draft_cb0:
                print(f"[Residual] scheduled sampling: {len(draft_cb0)}/{len(shards)} shards have "
                      f"draft CB0, ramp 0 -> {target:.2f} over {epochs} epochs")
            else:
                print("[Residual] ⚠ scheduled sampling requested but no draft CB0 is available "
                      "(draft not trained in this run or the draft pass failed) — the residual "
                      "will train on teacher CB0 only.")
        for epoch in range(1, epochs + 1):
            sched_p = target * (epoch - 1) / max(1, epochs - 1)
            train_loss, _ = self._epoch(train_w, optimizer, scaler, train=True,
                                        sched_p=sched_p, rng=sched_rng)
            val_loss, _ = self._epoch(val_w, optimizer, scaler, train=False)
            scheduler.step(val_loss)
            improved = val_loss < best_val - 1e-4
            if improved:
                best_val = val_loss
                stale = 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                stale += 1
            print(
                f"[Residual] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}"
                f"{'  *best*' if improved else ''} | total {_fmt_time(_now() - t_start)}"
            )
            if stale >= patience:
                print(f"[Residual] early stopping at epoch {epoch}")
                break
            if (_now() - t_start) > max_minutes * 60.0:
                print(f"[Residual] time budget of {max_minutes:.1f} min reached")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        _, top1 = self.evaluate(val_w)
        stats.update(
            best_val_loss=float(best_val),
            epochs_run=int(epoch),
            cb_top1=top1,
            cb_top1_mean=float(np.mean(top1)),
            train_seconds=float(_now() - t_start),
        )
        print(
            "[Residual] done | best val {:.4f} | mean CB1-8 top-1 {:.2f}%".format(
                best_val, 100 * float(np.mean(top1))
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
    p.add_argument("--max-minutes", type=float, default=12.0,
                   help="Wall-clock budget for draft training.")
    p.add_argument("--residual-epochs", type=int, default=60)
    p.add_argument("--residual-batch", type=int, default=32)
    p.add_argument("--residual-lr", type=float, default=3e-4)
    p.add_argument("--residual-sched", type=float, default=0.0,
                   help="Scheduled-sampling target fraction (0-0.9). The residual is trained on "
                        "teacher CB0 but runs on draft CB0, so a ramp to ~0.5 closes that exposure "
                        "bias; requires the draft to be trained in the same run (default: off).")
    p.add_argument("--max-minutes-residual", type=float, default=6.0,
                   help="Wall-clock budget for residual training.")
    p.add_argument("--skip-teacher", action="store_true", help="Reuse cached shards only.")
    p.add_argument("--skip-draft", action="store_true")
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
        shards = [TeacherShard.load(f) for f in shard_files]
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

    # Recompute the length model over whatever shards exist (resume-safe).
    length_model = LengthModel.fit([s.n_chars for s in shards], [s.frames for s in shards])

    # ---------------- trainers ----------------
    draft = DraftTransformer(config.draft).to(device)
    residual = ResidualDecoder(config.residual).to(device)
    draft_stats: Dict[str, float] = {}
    residual_stats: Dict[str, float] = {}

    if not args.skip_draft:
        draft_stats = DraftTrainer(
            draft, device, lr=args.draft_lr, batch_size=args.draft_batch, seed=args.seed
        ).train(shards, epochs=args.draft_epochs, max_minutes=args.max_minutes)
    if not args.skip_residual:
        residual_stats = ResidualTrainer(
            residual, device, lr=args.residual_lr, batch_size=args.residual_batch, seed=args.seed,
            draft_model=(draft if not args.skip_draft else None),
            sched_target=args.residual_sched,
        ).train(shards, epochs=args.residual_epochs, max_minutes=args.max_minutes_residual,
                sched_target=args.residual_sched)

    # ---------------- checkpoints ----------------
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_draft:
        save_checkpoint(
            ckpt_dir / "draft_cb0.pt", draft, config.draft,
            {"param_count": sum(p.numel() for p in draft.parameters()), **draft_stats},
        )
    if not args.skip_residual:
        save_checkpoint(
            ckpt_dir / "residual_cb1_8.pt", residual, config.residual,
            {"param_count": sum(p.numel() for p in residual.parameters()), **residual_stats},
        )

    meta = {
        "corpus_size": len(shards),
        "corpus_source": args.corpus_json or f"bootstrap(seed={args.seed})",
        "target_source": args.target_source,
        "target_stats": getattr(engine, "backend", "n/a"),
        "residual_scheduled_sampling": args.residual_sched,
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
    print(f"  checkpoints : {ckpt_dir}/draft_cb0.pt, {ckpt_dir}/residual_cb1_8.pt")
    print(f"  meta        : {ckpt_dir}/meta.json  (length model + training stats)")
    print("  next        : python run_full_evaluation.py --ckpt codeswitch_ckpt")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
