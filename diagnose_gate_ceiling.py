#!/usr/bin/env python3
"""Measure the information ceiling behind the unchanged 5% gate.

The gate decomposes exactly as

    rd_a = p * c + (1 - p) * w

where ``p`` is draft CB0 top-1, ``c`` is residual CB1-8 top-1 when CB0 is
correct, and ``w`` is residual CB1-8 top-1 when CB0 is wrong.  A residual
retrain can only move ``c`` and ``w``; the draft can only move ``p``.  This
script measures the *data* ceilings that bound those terms, without training a
new model:

1. exact p/c/w decomposition and per-codebook table on the gate sample;
2. draft-CB0 top-k coverage (the reranking headroom);
3. a character-aligned text-only n-gram residual ceiling (the ``w`` ceiling
   for a residual that ignores CB0 and uses only text);
4. a CB0-only n-gram residual ceiling (a floor for ``c``);
5. a text+CB0 n-gram residual estimate;
6. a CB0 AR n-gram estimate (a floor for a causal draft);
7. confidence-bin diagnostics for the draft's argmax.

Nothing here changes the gate.  The numbers are *estimates of what the data
supports*, not model results.  A low ceiling is evidence that the unchanged
gate cannot be reached by that path; a high ceiling is a reason to train.

Example:
    python diagnose_gate_ceiling.py \
        --ckpt codeswitch_ckpt_v6_6h \
        --teacher-cache real_cache/rasa_ta \
        --shards 4 --out eval_results_v6_ceiling
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from codeswitch_tts import CodeSwitchTTSConfig
from codeswitch_tts.config import DraftModelConfig, ResidualModelConfig
from codeswitch_tts.models.draft_transformer import DraftTransformer
from codeswitch_tts.models.residual_decoder import ResidualDecoder
from codeswitch_tts.utils.audio import AudioProcessor
from train_draft_distill import TeacherShard

SR = 44100
CHANCE = 100.0 / 1024.0
CB0_SIZE = 1024


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------
def load_models(ckpt: Path, device: torch.device):
    """Load draft and residual.  Mirrors ``diagnose_audio_bug.load_models``."""
    meta = json.loads((ckpt / "meta.json").read_text()) if (ckpt / "meta.json").is_file() else {}
    dp = torch.load(ckpt / "draft_cb0.pt", map_location="cpu", weights_only=False)
    draft = DraftTransformer(DraftModelConfig(**dp.get("config", {}))).to(device)
    draft.load_state_dict({k.removeprefix("module."): v for k, v in dp["state_dict"].items()})
    draft.eval()

    rp = torch.load(ckpt / "residual_cb1_8.pt", map_location="cpu", weights_only=False)
    residual = ResidualDecoder(ResidualModelConfig(**rp.get("config", {}))).to(device)
    residual.load_state_dict({k.removeprefix("module."): v for k, v in rp["state_dict"].items()})
    residual.eval()
    print(
        f"[ckpt] draft {sum(p.numel() for p in draft.parameters()) / 1e6:.2f}M | "
        f"residual {sum(p.numel() for p in residual.parameters()) / 1e6:.2f}M | "
        f"text_cond={getattr(residual.config, 'text_conditioning', False)} "
        f"conf_cond={getattr(residual.config, 'confidence_conditioning', False)}"
    )
    return draft, residual, meta


def load_shards(cache: Path, n: int, min_frames: int = 20) -> List[TeacherShard]:
    paths = sorted(cache.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.npz under {cache}")
    if 0 < n < len(paths):
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, n).astype(int)]
    shards: List[TeacherShard] = []
    for path in paths:
        try:
            shard = TeacherShard.load(path)
        except Exception as exc:
            print(f"[shard] skip {path.name}: {exc}")
            continue
        if shard.frames >= min_frames:
            shards.append(shard)
    if not shards:
        raise RuntimeError("no usable shards")
    print(f"[shard] {len(shards)} shards")
    return shards


def _draft_pass(draft, text: str, n: int, device, sample: bool, seed: int = 42):
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = draft.generate_sequence(
            text=text, num_frames=n, temperature=1.0, sample=sample, device=device
        )
    tokens = out.tokens[0].cpu().numpy().astype(np.int64)[:n]
    probs = out.probabilities[0].float().cpu().numpy()[:n]
    features = None
    if getattr(out, "frame_features", None) is not None:
        features = out.frame_features[0].float().cpu().numpy()[:n]
    return tokens, probs, features


def _residual_pass(residual, cb0: np.ndarray, device, text_features=None, confidence=None):
    kwargs = {"cb0_tokens": torch.from_numpy(cb0).unsqueeze(0).to(device), "sample": False}
    if getattr(residual.config, "text_conditioning", False):
        if text_features is None:
            raise ValueError("residual expects text_features")
        kwargs["text_features"] = torch.from_numpy(text_features).unsqueeze(0).to(device)
    if getattr(residual.config, "confidence_conditioning", False):
        if confidence is None:
            confidence = np.ones_like(cb0, dtype=np.float32)
        kwargs["cb0_confidence"] = torch.from_numpy(confidence).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = residual(**kwargs)
    return out.residual_codebooks[0].cpu().numpy().astype(np.int64)


def _windows(total: int, width: int = 86) -> List[Tuple[int, int]]:
    if total <= width:
        return [(0, total)]
    out = []
    for start in range(0, total, width):
        end = min(start + width, total)
        if end - start < width:
            start, end = total - width, total
        out.append((start, end))
    return out


# ---------------------------------------------------------------------------
# Exact gate decomposition
# ---------------------------------------------------------------------------
def gate_decomposition(
    draft,
    residual,
    shards: Sequence[TeacherShard],
    device,
    frames: int = 86,
) -> Dict:
    """Exact conditional decomposition.

    ``c`` is CB1-8 accuracy on frames whose draft CB0 is **correct**; ``w`` is
    accuracy on frames whose draft CB0 is **wrong**.  The identity

        rd_a = p * c + (1 - p) * w

    only holds for those conditional terms.  A plain overall draft-CB0
    accuracy is recorded separately as ``rd_a_overall`` for cross-checking.
    """
    rows = []
    per_cb_teacher_ok = np.zeros(8, dtype=np.float64)
    per_cb_teacher_all = np.zeros(8, dtype=np.float64)
    per_cb_draft_wrong = np.zeros(8, dtype=np.float64)
    per_cb_draft_all = np.zeros(8, dtype=np.float64)
    per_cb_text_wrong = np.zeros(8, dtype=np.float64)
    window_frames = 0
    correct_frames = 0
    wrong_frames = 0
    correct_cb0 = 0
    topk_hits = {k: 0 for k in (1, 3, 5, 10, 20, 50)}
    confidence_rows = []
    # The official gate averages per-shard window means (see
    # ``diagnose_audio_bug.py``), which is not the same estimator as pooling
    # all frames.  Keep both so the notebook never conflates them.
    shard_p_means: List[float] = []
    shard_rt_means: List[float] = []
    shard_rd_means: List[float] = []
    shard_c_means: List[float] = []
    shard_w_means: List[float] = []

    for shard in shards:
        codes = np.asarray(shard.codes, dtype=np.int64)
        T = int(min(shard.frames, codes.shape[1]))
        codes = np.clip(codes[:, :T], 0, 1023)
        draft_tokens, draft_probs, features = _draft_pass(
            draft, shard.norm_text, T, device, sample=False
        )
        correct = draft_tokens == codes[0]
        order = np.argsort(draft_probs, axis=-1)[:, ::-1]
        shard_p_windows: List[float] = []
        shard_rt_windows: List[float] = []
        shard_rd_windows: List[float] = []
        shard_c_windows: List[float] = []
        shard_w_windows: List[float] = []

        for start, end in _windows(T, frames):
            tcb0 = codes[0, start:end]
            tgt = codes[1:9, start:end]
            r_teacher = _residual_pass(
                residual,
                tcb0,
                device,
                text_features=features[start:end] if features is not None else None,
                confidence=np.ones(end - start, dtype=np.float32),
            )
            r_draft = _residual_pass(
                residual,
                draft_tokens[start:end],
                device,
                text_features=features[start:end] if features is not None else None,
                confidence=draft_probs[start:end].max(-1),
            )
            ok = correct[start:end]
            bad = ~ok
            n_ok, n_bad = int(ok.sum()), int(bad.sum())
            # ``p`` and the top-k coverage must use the same overlapping
            # evaluation windows as ``c``/``w``; raw shard frames would make
            # the identity drift whenever a trailing window overlaps.
            window_frames += int(end - start)
            correct_cb0 += n_ok
            win_order = np.argsort(draft_probs[start:end], axis=-1)[:, ::-1]
            for k in topk_hits:
                topk_hits[k] += int(
                    (win_order[:, :k] == tcb0[:, None]).any(-1).sum()
                )
            teacher_hit = r_teacher == tgt
            draft_hit = r_draft == tgt
            shard_p_windows.append(float(ok.mean()))
            shard_rt_windows.append(float(teacher_hit.mean()))
            shard_rd_windows.append(float(draft_hit.mean()))
            if n_ok:
                shard_c_windows.append(float(teacher_hit[:, ok].mean()))
            if n_bad:
                shard_w_windows.append(float(draft_hit[:, bad].mean()))
            per_cb_teacher_all += teacher_hit.sum(axis=1)
            per_cb_draft_all += draft_hit.sum(axis=1)
            if n_ok:
                per_cb_teacher_ok += teacher_hit[:, ok].sum(axis=1)
                correct_frames += n_ok
            if n_bad:
                per_cb_draft_wrong += draft_hit[:, bad].sum(axis=1)
                wrong_frames += n_bad
                if getattr(residual.config, "text_conditioning", False):
                    r_text = _residual_pass(
                        residual,
                        np.full(end - start, -1, dtype=np.int64),
                        device,
                        text_features=features[start:end] if features is not None else None,
                        confidence=np.zeros(end - start, dtype=np.float32),
                    )
                    per_cb_text_wrong += (r_text == tgt)[:, bad].sum(axis=1)
            # confidence bins; store (confidence, cb0_correct, residual_draft_correct_mean)
            conf = draft_probs[start:end].max(-1)
            for c, is_ok, acc in zip(
                conf.tolist(),
                ok.tolist(),
                draft_hit.mean(axis=0).tolist(),
            ):
                confidence_rows.append((float(c), bool(is_ok), float(acc)))

        shard_p_means.append(
            float(np.mean(shard_p_windows)) if shard_p_windows else 0.0
        )
        shard_rt_means.append(
            float(np.mean(shard_rt_windows)) if shard_rt_windows else 0.0
        )
        shard_rd_means.append(
            float(np.mean(shard_rd_windows)) if shard_rd_windows else 0.0
        )
        if shard_c_windows:
            shard_c_means.append(float(np.mean(shard_c_windows)))
        if shard_w_windows:
            shard_w_means.append(float(np.mean(shard_w_windows)))

        rows.append(
            {
                "index": int(shard.index),
                "frames": T,
                "cb0_top1": (
                    float(np.mean(shard_p_windows) * 100)
                    if shard_p_windows else 0.0
                ),
                "cb0_top5": float(
                    (order[:, :5] == codes[0, :, None]).any(-1).mean() * 100
                ),
                "rt_a": (
                    float(np.mean(shard_rt_windows) * 100)
                    if shard_rt_windows else 0.0
                ),
                "rd_a": (
                    float(np.mean(shard_rd_windows) * 100)
                    if shard_rd_windows else 0.0
                ),
            }
        )

    p = 100.0 * correct_cb0 / max(1, window_frames)
    per_cb_c = per_cb_teacher_ok / max(1, correct_frames)
    per_cb_w = per_cb_draft_wrong / max(1, wrong_frames)
    per_cb_text = per_cb_text_wrong / max(1, wrong_frames)
    c = float(per_cb_c.mean() * 100)
    w = float(per_cb_w.mean() * 100)
    w_text = float(per_cb_text.mean() * 100)
    rd_a = p / 100.0 * c + (1 - p / 100.0) * w
    rd_text = p / 100.0 * c + (1 - p / 100.0) * w_text
    rd_a_overall = float(
        (per_cb_draft_all / max(1, window_frames)).mean() * 100
    )
    rt_a_overall = float(
        (per_cb_teacher_all / max(1, window_frames)).mean() * 100
    )
    p_gate_style = float(np.mean(shard_p_means) * 100) if shard_p_means else 0.0
    rt_a_gate_style = (
        float(np.mean(shard_rt_means) * 100) if shard_rt_means else 0.0
    )
    rd_a_gate_style = (
        float(np.mean(shard_rd_means) * 100) if shard_rd_means else 0.0
    )
    c_gate_style = (
        float(np.mean(shard_c_means) * 100) if shard_c_means else 0.0
    )
    w_gate_style = (
        float(np.mean(shard_w_means) * 100) if shard_w_means else 0.0
    )
    gate_style_identity = (
        p_gate_style / 100.0 * c_gate_style
        + (1.0 - p_gate_style / 100.0) * w_gate_style
    )

    bins = []
    confidence_rows.sort(key=lambda x: x[0])
    n = len(confidence_rows)
    for b in range(10):
        lo, hi = b * n // 10, (b + 1) * n // 10
        chunk = confidence_rows[lo:hi]
        if not chunk:
            continue
        bins.append(
            {
                "decile": b,
                "n": len(chunk),
                "conf_mean": float(np.mean([x[0] for x in chunk])),
                "cb0_top1": float(np.mean([x[1] for x in chunk]) * 100),
                "rd_a": float(np.mean([x[2] for x in chunk]) * 100),
            }
        )

    return {
        "shards": rows,
        "p": p,
        "c": c,
        "w": w,
        "rd_a_from_terms": rd_a,
        "rd_a_overall": rd_a_overall,
        "rt_a": rt_a_overall,
        "decomposition_error": abs(rd_a - rd_a_overall),
        # The official gate averages per-shard window means.  These fields make
        # that estimator explicit; the p/c/w identity is exact only in the
        # frame-weighted view above.
        "p_gate_style": p_gate_style,
        "c_gate_style": c_gate_style,
        "w_gate_style": w_gate_style,
        "rd_a_gate_style": rd_a_gate_style,
        "rt_a_gate_style": rt_a_gate_style,
        "gate_style_identity": gate_style_identity,
        "gate_style_identity_error": abs(gate_style_identity - rd_a_gate_style),
        "correct_frames": correct_frames,
        "wrong_frames": wrong_frames,
        "per_cb_teacher": [float(x * 100) for x in per_cb_c],
        "per_cb_draft": [float(x * 100) for x in per_cb_w],
        "per_cb_draft_overall": [
            float(x * 100) for x in per_cb_draft_all / max(1, window_frames)
        ],
        "per_cb_text": (
            [float(x * 100) for x in per_cb_text]
            if getattr(residual.config, "text_conditioning", False) else None
        ),
        "w_text_model": w_text if getattr(residual.config, "text_conditioning", False) else None,
        "rd_a_if_text_only": (
            rd_text if getattr(residual.config, "text_conditioning", False) else None
        ),
        "topk_coverage": {
            str(k): float(v * 100 / max(1, window_frames)) for k, v in topk_hits.items()
        },
        "confidence_deciles": bins,
        "required_p_at_current_c_w": (
            float((5.0 - w) / (c - w) * 100) if c > w else None
        ),
        "required_w_at_current_p_c": (
            float((5.0 - p / 100.0 * c) / (1.0 - p / 100.0))
            if p < 100.0 else None
        ),
        "oracle_topk_rd_a_bound": {
            str(k): float(
                (v / max(1, window_frames)) * c
                + (1 - v / max(1, window_frames)) * w
            )
            for k, v in topk_hits.items()
        },
        "chance": CHANCE,
    }


# ---------------------------------------------------------------------------
# Text alignment and n-gram ceilings
# ---------------------------------------------------------------------------
@dataclass
class FrameView:
    """Per-frame character context for one shard."""

    char: np.ndarray      # (T,) Unicode codepoint of the aligned character
    prev_char: np.ndarray
    next_char: np.ndarray
    pos_bin: np.ndarray   # 0..3 position within the character run
    codes: np.ndarray     # (9, T)


def frame_view(shard: TeacherShard) -> Optional[FrameView]:
    if not shard.has_durations:
        return None
    text = shard.norm_text
    durations = np.asarray(shard.char_durations, dtype=np.int64)
    if durations.size != len(text) or int(durations.sum()) != int(shard.frames):
        return None
    # repeat_interleave maps each frame to its character index
    char_idx = np.repeat(np.arange(len(text), dtype=np.int64), durations)
    T = char_idx.size
    if T <= 0:
        return None
    char = np.array([ord(text[i]) for i in char_idx], dtype=np.int64)
    prev_char = np.array(
        [ord(text[i - 1]) if i > 0 else 0 for i in char_idx], dtype=np.int64
    )
    next_char = np.array(
        [ord(text[i + 1]) if i + 1 < len(text) else 0 for i in char_idx],
        dtype=np.int64,
    )
    # position bin within each character run
    pos = np.zeros(T, dtype=np.int64)
    cursor = 0
    for d in durations:
        if d > 0:
            pos[cursor:cursor + d] = np.minimum(
                3, (np.arange(int(d)) * 4) // int(d)
            )
        cursor += int(d)
    codes = np.asarray(shard.codes, dtype=np.int64)[:, :T]
    return FrameView(char, prev_char, next_char, pos, codes)


def _iter_views(shards: Iterable[TeacherShard]) -> Iterable[FrameView]:
    for shard in shards:
        view = frame_view(shard)
        if view is not None:
            yield view


def _concat_views(views: Iterable[FrameView]) -> Dict[str, np.ndarray]:
    chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
    for view in views:
        chunks["char"].append(view.char)
        chunks["prev_char"].append(view.prev_char)
        chunks["next_char"].append(view.next_char)
        chunks["pos_bin"].append(view.pos_bin)
        chunks["codes"].append(view.codes)
    if not chunks["char"]:
        raise RuntimeError("no duration-aligned shards")
    return {k: np.concatenate(v, axis=-1) for k, v in chunks.items()}


def _contexts(data: Dict[str, np.ndarray], kind: str) -> List[np.ndarray]:
    """Return coarse-to-fine context arrays for a backoff chain."""
    if kind == "text":
        return [
            data["char"] * 4 + data["pos_bin"],
            data["char"],
            np.zeros_like(data["char"]),
        ]
    if kind == "text3":
        return [
            data["prev_char"] * 4096 + data["char"] * 4 + data["pos_bin"],
            data["char"] * 4 + data["pos_bin"],
            data["char"],
            np.zeros_like(data["char"]),
        ]
    if kind == "cb0":
        return [data["codes"][0], np.zeros_like(data["char"])]
    if kind == "text_cb0":
        return [
            data["codes"][0] * 4096 + data["char"] * 4 + data["pos_bin"],
            data["codes"][0] * 64 + data["char"],
            data["codes"][0],
            data["char"] * 4 + data["pos_bin"],
            data["char"],
            np.zeros_like(data["char"]),
        ]
    if kind == "cb0_ar":
        # previous CB0, current char, previous char; pads at sequence start are
        # represented by the same value as the first frame, which is why this
        # is a *lower bound*, not an upper bound.
        prev = np.concatenate([[data["codes"][0][0]], data["codes"][0][:-1]])
        prevprev = np.concatenate([[prev[0]], prev[:-1]])
        return [
            prev * 4096 + data["char"],
            prev,
            data["char"],
            np.zeros_like(data["char"]),
        ]
    raise ValueError(kind)


def _mode_accuracy(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    kind: str,
    target_cb: int,
    min_count: int = 5,
    max_train_frames: int = 0,
) -> Dict[str, float]:
    """Backoff n-gram mode accuracy for one codebook.

    The target is ``codes[target_cb]``; contexts are built by ``_contexts``.
    The finest context is tried first and backed off when its count is below
    ``min_count``.  This is a memorization/ceiling estimate, not a neural
    generalization estimate.
    """
    y = train["codes"][target_cb]
    if max_train_frames and y.size > max_train_frames:
        step = max(1, y.size // max_train_frames)
        y = y[::step]
        context_arrays = [c[::step] for c in _contexts(train, kind)]
    else:
        context_arrays = _contexts(train, kind)
    counts: List[Dict[int, Counter]] = []
    for ctx in context_arrays:
        table: Dict[int, Counter] = defaultdict(Counter)
        for c, t in zip(ctx.tolist(), y.tolist()):
            table[int(c)][int(t)] += 1
        counts.append(table)

    context_arrays_val = _contexts(val, kind)
    y_val = val["codes"][target_cb]
    correct = 0
    covered = 0
    for i in range(y_val.size):
        target = int(y_val[i])
        pred = None
        for level, table in enumerate(counts):
            key = int(context_arrays_val[level][i])
            counter = table.get(key)
            if counter and sum(counter.values()) >= min_count:
                pred = counter.most_common(1)[0][0]
                covered += 1
                break
        if pred is None:
            # global mode
            global_counter = counts[-1].get(0)
            pred = global_counter.most_common(1)[0][0] if global_counter else 0
        if pred == target:
            correct += 1
    return {
        "accuracy": 100.0 * correct / max(1, y_val.size),
        "coverage": 100.0 * covered / max(1, y_val.size),
        "frames": int(y_val.size),
    }


def ngram_ceilings(shards: Sequence[TeacherShard], seed: int = 42) -> Dict:
    views = list(_iter_views(shards))
    if not views:
        return {"error": "no duration-aligned shards"}
    rng = random.Random(seed)
    rng.shuffle(views)
    split = max(1, int(len(views) * 0.9))
    train = _concat_views(views[:split])
    val = _concat_views(views[split:]) if len(views) > split else _concat_views(views[:1])
    out: Dict[str, object] = {
        "train_frames": int(train["char"].size),
        "val_frames": int(val["char"].size),
        "train_shards": split,
        "val_shards": len(views) - split,
    }
    for kind in ("text", "text3", "cb0", "text_cb0"):
        per_cb = []
        for cb in range(1, 9):
            per_cb.append(_mode_accuracy(train, val, kind, cb, min_count=5))
        out[kind] = {
            "per_cb": per_cb,
            "mean_accuracy": float(np.mean([x["accuracy"] for x in per_cb])),
            "mean_coverage": float(np.mean([x["coverage"] for x in per_cb])),
        }
    ar = _mode_accuracy(train, val, "cb0_ar", 0, min_count=5)
    out["cb0_ar"] = ar
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("eval_results_ceiling"))
    ap.add_argument("--shards", type=int, default=4,
                    help="Gate sample size. 4 is the official gate sample.")
    ap.add_argument("--ngram-shards", type=int, default=0,
                    help="Shards for n-gram ceilings (0 = all).")
    ap.add_argument("--skip-ngram", action="store_true",
                    help="Only compute the p/c/w decomposition; skip n-gram baselines.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] ckpt={args.ckpt} cache={args.teacher_cache} device={device}")

    draft, residual, meta = load_models(Path(args.ckpt), device)
    gate_shards = load_shards(Path(args.teacher_cache), args.shards)
    gate = gate_decomposition(draft, residual, gate_shards, device)

    ceiling_shards = (
        []
        if args.skip_ngram
        else load_shards(Path(args.teacher_cache), args.ngram_shards)
        if args.ngram_shards
        else gate_shards
        if len(gate_shards) >= 20
        else load_shards(Path(args.teacher_cache), 0)
    )
    print(f"[ngram] {len(ceiling_shards)} shards")
    ngram = ngram_ceilings(ceiling_shards, seed=args.seed) if ceiling_shards else {}

    # Projections that keep the gate threshold fixed.
    p = gate["p"] / 100.0
    c = gate["c"]
    w = gate["w"]
    projections = {}
    for name in ("text", "text3", "cb0", "text_cb0"):
        if name in ngram and "mean_accuracy" in ngram[name]:
            w_hat = float(ngram[name]["mean_accuracy"])
            projections[name] = {
                "w_hat": w_hat,
                "rd_a_if_w_equals_hat": p * c + (1 - p) * w_hat,
                "pass": bool((p * c + (1 - p) * w_hat) >= 5.0),
            }
    if "cb0_ar" in ngram:
        p_hat = float(ngram["cb0_ar"]["accuracy"])
        projections["ar_draft"] = {
            "p_hat": p_hat,
            "rd_a_if_p_equals_hat_at_current_c_w": p_hat / 100.0 * c
            + (1 - p_hat / 100.0) * w,
            "pass": bool(
                (p_hat / 100.0 * c + (1 - p_hat / 100.0) * w) >= 5.0
            ),
        }

    report = {
        "git_commit": _git_commit(),
        "checkpoint": str(args.ckpt),
        "cache": str(args.teacher_cache),
        "gate_shards": args.shards,
        "gate": gate,
        "ngram": ngram,
        "projections": projections,
        "truth": _truth_statement(gate, ngram),
    }
    (out / "ceiling_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== GATE DECOMPOSITION (unchanged thresholds) ===")
    print(f"p  (draft CB0 top-1)          = {gate['p']:.4f}%")
    print(f"c  (residual CB1-8 | CB0 ok)  = {gate['c']:.4f}%")
    print(f"w  (residual CB1-8 | CB0 bad) = {gate['w']:.4f}%")
    print(f"rd_a = p*c + (1-p)*w          = {gate['rd_a_from_terms']:.4f}%  "
          f"(frame-weighted identity)")
    print(f"official gate-style rd_a      = {gate['rd_a_gate_style']:.4f}%  "
          f"{'PASS' if gate['rd_a_gate_style'] >= 5 else 'FAIL'}  "
          f"(equal-weight per shard, matches diagnose_audio_bug)")
    print(f"gate-style identity check     = {gate['gate_style_identity']:.4f}%  "
          f"(abs error {gate['gate_style_identity_error']:.4f} points; "
          "p/c/w means are nonlinear, so this is not exact)")
    print(f"required p at current c,w     = "
          f"{gate['required_p_at_current_c_w']:.2f}%" if gate['required_p_at_current_c_w'] is not None
          else "required p at current c,w     = n/a (c <= w)")
    if gate.get("required_w_at_current_p_c") is not None:
        print(f"required w at current p,c     = "
              f"{gate['required_w_at_current_p_c']:.2f}%")
    if gate.get("w_text_model") is not None:
        print(f"w with text-only model        = {gate['w_text_model']:.4f}%  "
              f"rd_a={gate['rd_a_if_text_only']:.4f}%  "
              f"{'PASS' if gate['rd_a_if_text_only'] >= 5 else 'FAIL'}")
    print("top-k draft CB0 coverage      = " + ", ".join(
        f"top{k}={v:.2f}%" for k, v in gate["topk_coverage"].items()
    ))
    print("oracle top-k rd_a upper bound = " + ", ".join(
        f"top{k}={v:.2f}%" for k, v in gate["oracle_topk_rd_a_bound"].items()
    ))
    print("\n=== N-GRAM CEILINGS (mean CB1-8, held-out shards) ===")
    for name in ("text", "text3", "cb0", "text_cb0"):
        if name in ngram and "mean_accuracy" in ngram[name]:
            print(f"{name:9s} accuracy={ngram[name]['mean_accuracy']:.2f}% "
                  f"coverage={ngram[name]['mean_coverage']:.1f}%")
    if "cb0_ar" in ngram:
        print(f"cb0_ar    accuracy={ngram['cb0_ar']['accuracy']:.2f}% "
              f"coverage={ngram['cb0_ar']['coverage']:.1f}%")
    print("\n=== PROJECTIONS (gate unchanged) ===")
    for name, proj in projections.items():
        print(name, json.dumps(proj, ensure_ascii=False))
    print("\n=== TRUTH ===")
    print(report["truth"])
    print(f"\n[out] {out / 'ceiling_report.json'}")
    return 0


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _truth_statement(gate: Dict, ngram: Dict) -> str:
    p, c, w = gate["p"], gate["c"], gate["w"]
    official = gate.get("rd_a_gate_style", gate["rd_a_from_terms"])
    lines = []
    if official >= 5.0:
        lines.append(
            "The unchanged gate passes on this sample "
            f"(official gate-style rd_a={official:.3f}%). Verify with the "
            "60-utterance suite."
        )
    else:
        lines.append(
            f"The unchanged gate fails: official gate-style rd_a={official:.3f}% < 5%. "
            f"The frame-weighted identity is rd_a={gate['rd_a_from_terms']:.3f}% "
            f"with p={p:.3f}%, c={c:.3f}%, w={w:.3f}%."
        )
    if c > w:
        lines.append(
            "At fixed c/w, the draft would need p="
            f"{gate['required_p_at_current_c_w']:.1f}% to pass."
        )
    if gate.get("required_w_at_current_p_c") is not None:
        lines.append(
            "At fixed p/c, the wrong-CB0 residual accuracy would need w="
            f"{gate['required_w_at_current_p_c']:.2f}% to pass."
        )
    if gate.get("w_text_model") is not None:
        lines.append(
            f"The actual text-only residual fallback has w={gate['w_text_model']:.2f}% "
            f"and rd_a={gate['rd_a_if_text_only']:.2f}% "
            f"({'PASS' if gate['rd_a_if_text_only'] >= 5 else 'FAIL'})."
        )
    for name in ("text", "text3"):
        if name in ngram and "mean_accuracy" in ngram[name]:
            w_hat = ngram[name]["mean_accuracy"]
            rd = p / 100.0 * c + (1 - p / 100.0) * w_hat
            lines.append(
                f"A {name}-only n-gram residual reaches w≈{w_hat:.2f}% and "
                f"rd_a≈{rd:.2f}% ({'PASS' if rd >= 5 else 'FAIL'}); this is an "
                "empirical floor, not a neural ceiling."
            )
    if "cb0_ar" in ngram:
        p_hat = ngram["cb0_ar"]["accuracy"]
        rd = p_hat / 100.0 * c + (1 - p_hat / 100.0) * w
        lines.append(
            f"A CB0 AR n-gram draft floor is p≈{p_hat:.2f}% and rd_a≈{rd:.2f}% "
            f"({'PASS' if rd >= 5 else 'FAIL'}); a stronger causal model can exceed it."
        )
    lines.append(
        "These are empirical baselines, not upper bounds; a pass still requires "
        "the actual residual and the unchanged gate."
    )
    return " ".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
