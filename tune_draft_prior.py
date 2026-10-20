#!/usr/bin/env python3
"""Sweep teacher-prior logit adjustment for the NAR draft's argmax.

The v3 draft's argmax collapses to a single CB0 token on every frame
(``top_frac ~= 0.999``, ``longest_run ~= T``), while its sampled sequence is
diverse.  A NAR model's per-frame argmax is the conditional mode; a strong
global token prior can make that mode constant even when the text-conditioned
distribution is broad.  This script subtracts ``alpha * log p_teacher(token)``
from the draft logits and reports:

* CB0 top-1 and sequence-diversity stats for each alpha;
* residual CB1-8 top-1 on the adjusted draft CB0;
* teacher-residual and full-student WAVs for selected alphas.

It does not modify or retrain the checkpoint.  Run it on the existing v3
directory and teacher cache:

    python tune_draft_prior.py \
      --ckpt codeswitch_ckpt_v3 \
      --teacher-cache teacher_cache \
      --out audio_diag_v3_prior \
      --shards 4 \
      --alphas 0,0.25,0.5,0.75,1.0 \
      --audio-alphas 0,0.5,1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from codeswitch_tts import CodeSwitchTTSConfig
from codeswitch_tts.utils.audio import AudioProcessor, save_wav
from diagnose_audio_bug import (
    CHANCE,
    code_stats,
    decode,
    load_models,
    load_shards,
    mean,
    per_cb,
    residual_tokens,
    stats,
    top1,
    windows,
)


def teacher_prior(shards: Sequence, smoothing: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log prior, counts) for teacher CB0 over the supplied shards."""
    counts = np.zeros(1024, dtype=np.int64)
    for shard in shards:
        cb0 = np.asarray(shard.codes[0], dtype=np.int64).reshape(-1)
        counts += np.bincount(cb0, minlength=1024)
    probs = (counts.astype(np.float64) + smoothing) / (counts.sum() + smoothing * 1024)
    return np.log(probs).astype(np.float32), counts


def apply_prior_adjustment(
    logits: np.ndarray, log_prior: np.ndarray, alpha: float
) -> np.ndarray:
    """Subtract ``alpha * log p_teacher(token)`` from (T, V) logits."""
    logits = np.asarray(logits, dtype=np.float32)
    log_prior = np.asarray(log_prior, dtype=np.float32).reshape(-1)
    if logits.ndim != 2 or logits.shape[1] != log_prior.shape[0]:
        raise ValueError(
            f"logits shape {logits.shape} does not match prior length {log_prior.shape}"
        )
    return logits - float(alpha) * log_prior[None, :]


def draft_logits(draft, text: str, n: int, device: torch.device) -> np.ndarray:
    """Greedy-forward logits for one shard, without selecting tokens."""
    ids, mask = draft.tokenizer.batch_encode([text])
    with torch.inference_mode():
        out = draft.forward(
            ids.to(device), mask.to(device), num_frames=int(n),
            temperature=1.0, sample=False,
        )
    return out.logits[0, : int(n)].detach().float().cpu().numpy()


def residual_score(
    residual, cb0: np.ndarray, target: np.ndarray, device: torch.device, width: int = 86
) -> float:
    """Mean per-codebook CB1-8 top-1 over fixed-length windows."""
    total = int(min(len(cb0), target.shape[-1]))
    if total <= 0:
        return float("nan")
    scores: List[float] = []
    for start, end in windows(total, width):
        _, pred = residual_tokens(residual, cb0[start:end], False, device)
        scores.append(mean(per_cb(pred, target[:, start:end])))
    return mean(scores)


def _safe_alpha(value: str) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def parse_alphas(raw: str) -> List[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one alpha is required")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sweep teacher-prior logit adjustment for draft argmax.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", default="codeswitch_ckpt_v3")
    ap.add_argument("--teacher-cache", default="teacher_cache")
    ap.add_argument("--out", default="audio_diag_v3_prior")
    ap.add_argument("--shards", type=int, default=4,
                    help="Evaluation shards; 0 means all cached shards.")
    ap.add_argument("--prior-shards", type=int, default=0,
                    help="Shards used to estimate the teacher prior; 0 means all.")
    ap.add_argument("--frames", type=int, default=86,
                    help="Residual scoring window.")
    ap.add_argument("--audio-frames", type=int, default=172)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--audio-alphas", default="0,0.5,1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args(argv)

    alphas = parse_alphas(args.alphas)
    audio_alphas = set(parse_alphas(args.audio_alphas))
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] ckpt={args.ckpt} cache={args.teacher_cache} device={device}")
    print(f"[setup] alphas={alphas} audio_alphas={sorted(audio_alphas)}")

    draft, residual, meta = load_models(Path(args.ckpt), device)
    eval_shards = load_shards(Path(args.teacher_cache), args.shards)
    prior_shards = load_shards(Path(args.teacher_cache), args.prior_shards)
    log_prior, counts = teacher_prior(prior_shards)
    np.save(out / "draft_prior_log_probs.npy", log_prior)
    print(
        f"[prior] {len(prior_shards)} shards, {int(counts.sum())} CB0 frames; "
        f"top token={int(np.argmax(counts))} ({counts.max() / counts.sum():.2%})"
    )
    (out / "prior_meta.json").write_text(json.dumps({
        "shards": len(prior_shards),
        "frames": int(counts.sum()),
        "top_token": int(np.argmax(counts)),
        "top_frac": float(counts.max() / counts.sum()),
        "log_prior_file": "draft_prior_log_probs.npy",
    }, indent=2))

    processor = None
    if not args.no_audio:
        try:
            processor = AudioProcessor(CodeSwitchTTSConfig().audio)
            print(f"[audio] DAC backend: {processor.dac.backend}")
            if processor.dac.backend not in ("hf_pretrained", "dac_official"):
                print("[audio] non-pretrained DAC; skipping wavs")
                processor = None
        except Exception as exc:
            print(f"[audio] DAC unavailable ({type(exc).__name__}: {exc}); skipping wavs")
            processor = None

    rows: Dict[float, List[Dict[str, float]]] = {float(a): [] for a in alphas}
    wav_rows: List[Dict[str, float]] = []
    for shard in eval_shards:
        codes = np.asarray(shard.codes, dtype=np.int64)
        if codes.shape[0] != 9 and codes.shape[1] == 9:
            codes = codes.T
        T = int(min(shard.frames, codes.shape[1]))
        codes = np.clip(codes[:, :T], 0, 1023)
        logits = draft_logits(draft, shard.norm_text, T, device)
        if logits.shape[0] != T:
            raise RuntimeError(f"draft logits shape {logits.shape} for T={T}")

        A = min(args.audio_frames, T)
        for alpha in alphas:
            tokens = apply_prior_adjustment(logits, log_prior, alpha).argmax(axis=-1)
            tokens = tokens.astype(np.int64)
            cs = code_stats(tokens)
            rd = residual_score(residual, tokens, codes[1:9, :T], device, args.frames)
            cb = top1(tokens, codes[0, :T])
            row = {"alpha": float(alpha), "cb0": float(cb), "rd_a": float(rd), **cs}
            rows[float(alpha)].append(row)
            print(
                f"[shard {shard.index:4d}] alpha={alpha:.2f} cb0={cb:.2f}% "
                f"rd_a={rd:.3f}% unique={cs['unique']:4d} "
                f"top={cs['top_frac']:.1%} run={cs['longest_run']:3d} "
                f"H={cs['entropy_nats']:.2f}"
            )

            if processor is None or float(alpha) not in audio_alphas:
                continue
            # Teacher residual isolates the adjusted CB0; student residual shows
            # the deployed v3 path.  The teacher version is the one to listen to
            # before retraining the residual.
            teacher_codes = np.vstack([tokens[:A], codes[1:9, :A]])
            _, student_codes = residual_tokens(residual, tokens[:A], False, device)
            for tag, cc in (("teacher_res", teacher_codes), ("student_full", student_codes)):
                w = decode(processor, cc, f"shard_{shard.index:04d}/adj{_safe_alpha(alpha)}/{tag}")
                if w is None:
                    continue
                p = out / f"shard_{shard.index:04d}_adj{_safe_alpha(alpha)}_{tag}.wav"
                save_wav(p, w, sample_rate=44100)
                st = stats(w)
                st.update({"alpha": float(alpha), "tag": tag, "file": str(p),
                           "shard": int(shard.index)})
                wav_rows.append(st)
                print(f"  [wav] {p.name:56s} rms={st['rms']:.4f} "
                      f"flat={st['flatness']:.3f} zcr={st['zcr']:.3f}")

    summary: Dict[str, Dict[str, float]] = {}
    for alpha in alphas:
        rs = rows[float(alpha)]
        summary[f"{alpha:.2f}"] = {
            "cb0_top1": mean([r["cb0"] for r in rs]),
            "rd_a": mean([r["rd_a"] for r in rs]),
            "unique": mean([r["unique"] for r in rs]),
            "top_frac": mean([r["top_frac"] for r in rs]),
            "longest_run": mean([r["longest_run"] for r in rs]),
            "entropy_nats": mean([r["entropy_nats"] for r in rs]),
        }
    print(f"\n=== PRIOR SWEEP ({len(eval_shards)} shards; chance={CHANCE:.4f}%) ===")
    print("alpha  cb0_top1  rd_a    unique  top_frac  run    H")
    for alpha in alphas:
        s = summary[f"{alpha:.2f}"]
        print(f"{alpha:5.2f}  {s['cb0_top1']:7.3f}%  {s['rd_a']:6.3f}%  "
              f"{s['unique']:6.0f}  {s['top_frac']:8.3f}  {s['longest_run']:5.0f}  "
              f"{s['entropy_nats']:5.2f}")

    report = {
        "chance": CHANCE,
        "alphas": [float(a) for a in alphas],
        "audio_alphas": sorted(float(a) for a in audio_alphas),
        "eval_shards": len(eval_shards),
        "prior": json.loads((out / "prior_meta.json").read_text()),
        "summary": summary,
        "per_shard": {f"{a:.2f}": rows[float(a)] for a in alphas},
        "wavs": wav_rows,
        "meta": meta,
    }
    (out / "prior_sweep.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[done] {out / 'prior_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
