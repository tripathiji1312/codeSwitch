#!/usr/bin/env python3
"""One-sentence argmax audio smoke test for Plan 013.

This is deliberately *not* the 60-utterance benchmark.  It answers three
questions before the full run is allowed to start:

1. Does the retrained student synthesize a real WAV, decoded by Parler's real
   in-memory DAC (or another explicitly allowed pretrained DAC)?
2. What is the code-level top-1 accuracy of the draft CB0 on this sentence,
   measured against a freshly generated Parler teacher target?
3. Does the residual decoder produce usable codebooks 1-8 on the teacher CB0
   and on the draft CB0 it will actually see?

The teacher target is generated in a temporary directory: no shard from the
training cache is used, and nothing from this smoke test leaks into training.

Typical use on Kaggle (after the retrain has produced ``codeswitch_ckpt_v2``):

    python smoke_test_argmax.py \
        --ckpt codeswitch_ckpt_v2 \
        --utt-id ta-01 \
        --config C \
        --out output_audio/test_fixed_argmax.wav

The script refuses to write a WAV through a synthetic/random DAC.  It prints a
warning, but does not fail, when the one-sentence top-1 number is weak: a
single sentence is a smoke signal, while the post-retrain diagnostic and the
60-utterance gate remain the authoritative checks.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from codeswitch_tts import CodeSwitchTTSConfig, SpeculativeEngine, TanglishNormalizer
from codeswitch_tts.utils.audio import save_wav
from codeswitch_benchmark_data import EVAL_UTTERANCES
from diagnose_audio_bug import CHANCE, mean, per_cb, top1, windows
from run_full_evaluation import EvalSynthesizer, TeacherForcedScorer, _load_checkpoints
from train_draft_distill import TeacherCollector


REAL_BACKENDS = {
    "parler_audio_encoder",  # Parler's in-memory audio encoder (preferred)
    "passed_model",          # AudioProcessor handed the same Parler DAC object
    "hf_pretrained",         # standalone ylacombe/dac_44khz weights
    "dac_official",          # official descript-audio-codec weights
    "parler_streamer",       # Config A only; never expected for B/C
}
PARLER_BACKENDS = {"parler_audio_encoder", "passed_model"}


def _audio_stats(wave: np.ndarray, sample_rate: int = 44100) -> Dict[str, float]:
    """Cheap descriptive stats; high flatness/ZCR is a symptom of static/noise."""
    w = np.asarray(wave, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return {"sec": 0.0, "rms": 0.0, "peak": 0.0, "flatness": 0.0, "zcr": 0.0}
    spec = np.abs(np.fft.rfft(w)) + 1e-12
    return {
        "sec": float(len(w) / float(sample_rate)),
        "rms": float(np.sqrt(np.mean(w * w))),
        "peak": float(np.max(np.abs(w))),
        "flatness": float(np.exp(np.mean(np.log(spec))) / np.mean(spec)),
        "zcr": float(np.mean(np.abs(np.diff(np.signbit(w).astype(np.int8))))),
    }


def _residual_top1(
    residual,
    cb0: np.ndarray,
    target_cb1_8: np.ndarray,
    device: torch.device,
    seed: int,
    width: int = 86,
) -> float:
    """Windowed residual CB1-8 top-1, matching diagnose_audio_bug.py."""
    scores = []
    total = int(min(len(cb0), target_cb1_8.shape[-1]))
    if total <= 0:
        return float("nan")
    for start, end in windows(total, width):
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = residual(
                cb0_tokens=torch.from_numpy(cb0[start:end]).unsqueeze(0).to(device),
                temperature=1.0,
                sample=False,
            )
        pred = out.residual_codebooks[0].detach().cpu().numpy().astype(np.int64)
        scores.append(mean(per_cb(pred, target_cb1_8[:, start:end])))
    return mean(scores)


def _generate_teacher(
    engine: SpeculativeEngine,
    text: str,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate the exact sentence's teacher codes in a throwaway cache."""
    normalizer = TanglishNormalizer()
    with tempfile.TemporaryDirectory(prefix="codeswitch_smoke_teacher_") as tmp:
        collector = TeacherCollector(
            engine=engine,
            out_dir=Path(tmp),
            normalizer=normalizer,
            seed=seed,
            target_source="auto",
            verbose=True,
        )
        shards, _ = collector.collect([text])
        if not shards:
            raise RuntimeError("teacher generation produced no shard")
        shard = shards[0]
        codes = np.asarray(shard.codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[0] != 9:
            raise RuntimeError(f"unexpected teacher code shape: {codes.shape}")
        codes = np.clip(codes, 0, 1023)
        meta = {
            "teacher_frames": int(codes.shape[1]),
            "teacher_source": shard.target_source,
            "teacher_audio_sec": float(shard.audio_sec),
            "teacher_gen_ms": float(shard.gen_ms),
        }
        return codes, meta


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-sentence Plan-013 argmax audio smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default="codeswitch_ckpt",
                   help="Checkpoint directory; use codeswitch_ckpt_v2 after retraining.")
    p.add_argument("--utt-id", default="ta-01",
                   help="Utterance id from codeswitch_benchmark_data.EVAL_UTTERANCES.")
    p.add_argument("--config", choices=["B", "C"], default="C",
                   help="B = NAR only; C = speculative verification (the proposed path).")
    p.add_argument("--out", default="output_audio/test_fixed_argmax.wav",
                   help="WAV path; a .json report is written beside it.")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                   help="Parler precision on GPU.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the teacher sample and code-level top-1 probe.")
    p.add_argument("--skip-top1", action="store_true",
                   help="Skip the extra teacher generation/top-1 block (audio only).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    utt = next((u for u in EVAL_UTTERANCES if u["id"] == args.utt_id), None)
    if utt is None:
        print(f"[smoke] unknown utterance id {args.utt_id!r}; "
              f"valid: {', '.join(u['id'] for u in EVAL_UTTERANCES[:6])}, ...")
        return 2
    utt_index = EVAL_UTTERANCES.index(utt)
    out_path = Path(args.out)
    report_path = out_path.with_suffix(".json")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = torch.float32 if device.type == "cpu" else dtype_map[args.dtype]

    print("=" * 78)
    print("CodeSwitch-TTS · one-sentence argmax smoke test")
    print(f"  utt: {utt['id']} | register: {utt['register']} | config: {args.config}")
    print(f"  ckpt: {args.ckpt} | device: {device} | dtype: {dtype}")
    print(f"  text: {utt['text']}")
    print("=" * 78)

    config = CodeSwitchTTSConfig()
    config.system.allow_untrained_fallback = False

    engine = SpeculativeEngine(config.speculative, device=device, dtype=dtype)
    if not engine.speech_available:
        print("\n[smoke] ❌ Parler-TTS is unavailable. This smoke test refuses to "
              "write noise; fix the Parler/protobuf stack and rerun.")
        return 2

    draft_model, residual_model, length_model, draft_trained, residual_trained = _load_checkpoints(
        args, config, device
    )
    if not (draft_trained and residual_trained):
        print("\n[smoke] ❌ trained draft/residual checkpoints are missing; refusing to "
              "smoke-test random weights.")
        return 2

    # The eval loop's seed formula, so this WAV is the same operating point the
    # 60-utterance run will use for this utterance and config.
    eval_seed = args.seed * 100003 + utt_index * 7 + (2 if args.config == "C" else 1)
    synth_args = argparse.Namespace(
        sample_mode="argmax",
        allow_untrained=False,
        boundary_frames=3,
        play_steps=12,
        seed=args.seed,
    )
    scorer = TeacherForcedScorer(engine) if args.config == "C" else None
    if args.config == "C" and (scorer is None or not scorer.available):
        print("\n[smoke] ❌ Config C needs true teacher-forced CB0 scoring, but the "
              f"scorer is unavailable ({getattr(scorer, 'reason', 'not created')}). "
              "Refusing to call a synthetic-target run a Config-C smoke test.")
        return 2
    synth = EvalSynthesizer(
        synth_args, config, engine, draft_model, residual_model, length_model,
        scorer, device, draft_trained, residual_trained,
    )

    # ---- code-level top-1 on this exact sentence -------------------------
    teacher_meta: Dict[str, Any] = {}
    cb0_acc: Optional[float] = None
    cb0_conf: Optional[float] = None
    rt_acc: Optional[float] = None
    rd_acc: Optional[float] = None
    if not args.skip_top1:
        print("\n[smoke] generating a fresh Parler teacher target for top-1 ...")
        try:
            teacher_codes, teacher_meta = _generate_teacher(engine, utt["text"], args.seed)
            T = int(teacher_codes.shape[1])
            normalized = synth.normalize(utt["text"])
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                draft_out = draft_model.generate_chunk(
                    text=normalized,
                    num_frames=T,
                    temperature=1.0,
                    sample=False,
                    device=device,
                )
            draft_tokens = draft_out.tokens[0].detach().cpu().numpy().astype(np.int64)
            draft_probs = draft_out.probabilities[0].detach().float().cpu().numpy()
            cb0_acc = top1(draft_tokens, teacher_codes[0, :T])
            cb0_conf = mean(np.max(draft_probs, axis=-1))
            rt_acc = _residual_top1(
                residual_model, teacher_codes[0, :T], teacher_codes[1:9, :T],
                device, args.seed,
            )
            rd_acc = _residual_top1(
                residual_model, draft_tokens, teacher_codes[1:9, :T],
                device, args.seed,
            )
            print(f"[smoke] top-1 CB0      : {cb0_acc:.3f}%  (chance {CHANCE:.4f}%)")
            print(f"[smoke] mean top-1 conf: {cb0_conf * 100.0:.1f}%")
            print(f"[smoke] residual CB1-8 on teacher CB0: {rt_acc:.3f}%")
            print(f"[smoke] residual CB1-8 on draft CB0  : {rd_acc:.3f}%")
            if cb0_acc < 1.0:
                print("[smoke] ⚠ CB0 top-1 is below 1%; listen carefully before "
                      "launching the 60-utterance run.")
            if (rt_acc is not None and rt_acc < 5.0) or (rd_acc is not None and rd_acc < 5.0):
                print("[smoke] ⚠ residual top-1 is below the 5% post-retrain gate; "
                      "this sentence may still sound noisy.")
        except Exception as exc:
            print(f"[smoke] ⚠ top-1 block failed ({type(exc).__name__}: {exc}); "
                  "the WAV will still be generated so it can be listened to.")

    # ---- actual argmax synthesis through the real pipeline ---------------
    print(f"\n[smoke] synthesizing {utt['id']} with config {args.config} / argmax ...")
    try:
        if args.config == "C":
            row, wave = synth.synth_C(
                utt, utt_index, eval_seed, gate_threshold=0.0, gate_stat="utt_min",
            )
        else:
            row, wave = synth._chunked_synthesis(utt, utt_index, eval_seed, verify=False)
            row["gate_threshold"] = 0.0
    except Exception as exc:
        print(f"[smoke] ❌ synthesis failed: {type(exc).__name__}: {exc}")
        return 1

    if row.get("error"):
        print(f"[smoke] ❌ pipeline reported an error: {row['error']}")
        return 1
    if wave is None or np.asarray(wave).size == 0:
        print("[smoke] ❌ pipeline returned no audio.")
        return 1
    if str(row.get("sample_mode")) != "argmax":
        print(f"[smoke] ❌ pipeline wrote sample_mode={row.get('sample_mode')!r}, "
              "not 'argmax'.")
        return 1
    if args.config == "C" and row.get("verify_mode") != "true":
        print(f"[smoke] ❌ config C did not use true teacher-forced CB0 verification "
              f"(verify_mode={row.get('verify_mode')!r}); refusing to certify it.")
        return 1

    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    sample_rate = int(row.get("sample_rate", 44100))
    save_wav(out_path, wave, sample_rate=sample_rate, normalize=False)
    stats = _audio_stats(wave, sample_rate=sample_rate)
    backend = str(row.get("dac_backend") or "unknown")

    print("\n" + "-" * 78)
    print("RESULT")
    print("-" * 78)
    print(f"  utterance            : {utt['id']} ({utt['register']})")
    print(f"  sample mode          : {row.get('sample_mode')}")
    print(f"  verify mode          : {row.get('verify_mode')}")
    print(f"  draft acceptance     : {row.get('accept_true')}")
    print(f"  pipeline frames      : {row.get('n_frames')}")
    if teacher_meta:
        print(f"  teacher frames       : {teacher_meta.get('teacher_frames')} "
              f"({teacher_meta.get('teacher_source')} target)")
    if cb0_acc is not None:
        print(f"  draft CB0 top-1      : {cb0_acc:.3f}%  (chance {CHANCE:.4f}%)")
        print(f"  mean top-1 confidence: {cb0_conf * 100.0:.1f}%")
    if rt_acc is not None:
        print(f"  residual teacher top1: {rt_acc:.3f}%")
    if rd_acc is not None:
        print(f"  residual draft top1  : {rd_acc:.3f}%")
    print(f"  DAC backend          : {backend}")
    if backend in PARLER_BACKENDS:
        print("  DAC provenance       : real Parler in-memory DAC ✅")
    elif backend in REAL_BACKENDS:
        print("  DAC provenance       : real pretrained DAC (not Parler's encoder) ✅")
    else:
        print("  DAC provenance       : ❌ not a real DAC backend")
    print(f"  audio                : {stats['sec']:.2f}s | rms={stats['rms']:.4f} | "
          f"peak={stats['peak']:.3f} | flatness={stats['flatness']:.3f} | "
          f"zcr={stats['zcr']:.3f}")
    print(f"  WAV saved            : {out_path.resolve()}")
    print(f"  report               : {report_path.resolve()}")
    print(f"\nLISTEN: {out_path.resolve()}")
    if backend not in REAL_BACKENDS:
        print("\n[smoke] ❌ refusing to certify a non-real DAC backend.")
    elif stats["flatness"] > 0.55 or stats["zcr"] > 0.35:
        print("\n[smoke] ⚠ audio statistics look noise-like; listen before the full run.")

    report: Dict[str, Any] = {
        "utt_id": utt["id"],
        "register": utt["register"],
        "text": utt["text"],
        "config": args.config,
        "sample_mode": row.get("sample_mode"),
        "verify_mode": row.get("verify_mode"),
        "accept_true_pct": row.get("accept_true"),
        "checkpoint": str(args.ckpt),
        "eval_seed": eval_seed,
        "pipeline_frames": row.get("n_frames"),
        "ttfa_compute_ms": row.get("ttfa_compute_ms"),
        "ttfa_e2e_ms": row.get("ttfa_e2e_ms"),
        "total_ms": row.get("total_ms"),
        "rtf": row.get("rtf"),
        "dac_backend": backend,
        "audio_path": str(out_path.resolve()),
        "sample_rate": sample_rate,
        "audio_stats": stats,
        "top1_cb0_pct": cb0_acc,
        "draft_confidence_pct": None if cb0_conf is None else cb0_conf * 100.0,
        "residual_teacher_cb1_8_top1_pct": rt_acc,
        "residual_draft_cb1_8_top1_pct": rd_acc,
        "chance_pct": CHANCE,
        **teacher_meta,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if backend in REAL_BACKENDS else 1


if __name__ == "__main__":
    raise SystemExit(main())
