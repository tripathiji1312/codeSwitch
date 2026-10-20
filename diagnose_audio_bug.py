#!/usr/bin/env python3
"""Diagnose CodeSwitch-TTS B/C noise using the restored teacher cache.

No Parler load is required: the 2000 teacher shards already contain the
(text, 9 x T DAC codes) pairs the residual was trained on.  If the residual
cannot reproduce CB1-8 on those shards, it never learned the residual.  DAC is
only needed for the comparison wavs.

    python diagnose_audio_bug.py
    python diagnose_audio_bug.py --no-audio
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from codeswitch_tts import CodeSwitchTTSConfig, DraftTransformer, ResidualDecoder
from codeswitch_tts.config import DraftModelConfig, ResidualModelConfig
from codeswitch_tts.utils.audio import AudioProcessor, save_wav
from train_draft_distill import LengthModel, TeacherShard

SR = 44100
CHANCE = 100.0 / 1024.0


def load_models(ckpt: Path, device: torch.device):
    meta = json.loads((ckpt / "meta.json").read_text()) if (ckpt / "meta.json").is_file() else {}
    dp = torch.load(ckpt / "draft_cb0.pt", map_location="cpu", weights_only=False)
    draft = DraftTransformer(DraftModelConfig(**dp.get("config", {}))).to(device)
    draft.load_state_dict({k.removeprefix("module."): v for k, v in dp["state_dict"].items()})
    draft.eval()
    rp = torch.load(ckpt / "residual_cb1_8.pt", map_location="cpu", weights_only=False)
    residual = ResidualDecoder(ResidualModelConfig(**rp.get("config", {}))).to(device)
    residual.load_state_dict({k.removeprefix("module."): v for k, v in rp["state_dict"].items()})
    residual.eval()
    print(f"[ckpt] draft {sum(p.numel() for p in draft.parameters()) / 1e6:.2f}M | "
          f"residual {sum(p.numel() for p in residual.parameters()) / 1e6:.2f}M")
    if meta.get("length_model"):
        lm = LengthModel.from_dict(meta["length_model"])
        print(f"[ckpt] length model MAE {lm.mae_frames:.1f} frames over {lm.n_samples}")
    return draft, residual, meta


def load_shards(cache: Path, n: int) -> List[TeacherShard]:
    paths = sorted(cache.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.npz under {cache}")
    if 0 < n < len(paths):
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, n).astype(int)]
    shards = []
    for p in paths:
        try:
            s = TeacherShard.load(p)
        except Exception as exc:
            print(f"[shard] skip {p.name}: {exc}")
            continue
        if s.frames >= 20:
            shards.append(s)
    print(f"[shard] {len(shards)} shards; sources "
          f"{ {s.target_source: sum(t.target_source == s.target_source for t in shards) for s in shards} }")
    if not shards:
        raise RuntimeError("no usable teacher shards")
    return shards


def draft_tokens(draft, text: str, n: int, sample: bool, device, seed: int = 42):
    torch.manual_seed(seed)
    ids, mask = draft.tokenizer.batch_encode([text])
    with torch.inference_mode():
        out = draft.forward(ids.to(device), mask.to(device), num_frames=n,
                            temperature=1.0, sample=sample)
    tok = out.tokens[0].cpu().numpy().astype(np.int64)
    return tok[:n]


def residual_tokens(residual, cb0: np.ndarray, sample: bool, device, seed: int = 42):
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = residual(cb0_tokens=torch.from_numpy(cb0).unsqueeze(0).to(device),
                       temperature=1.0, sample=sample)
    return (out.full_codebooks[0].cpu().numpy().astype(np.int64),
            out.residual_codebooks[0].cpu().numpy().astype(np.int64))


def top1(pred: np.ndarray, target: np.ndarray) -> float:
    n = min(pred.shape[-1], target.shape[-1])
    return float((pred[..., :n] == target[..., :n]).mean() * 100.0) if n else float("nan")


def per_cb(pred: np.ndarray, target: np.ndarray) -> List[float]:
    return [top1(pred[k], target[k]) for k in range(min(pred.shape[0], target.shape[0]))]


def windows(total: int, width: int) -> List[Tuple[int, int]]:
    if total <= width:
        return [(0, total)]
    out = []
    for s in range(0, total, width):
        e = min(s + width, total)
        if e - s < width:
            s, e = total - width, total
        out.append((s, e))
    return out


def mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def decode(processor, codes: np.ndarray, tag: str) -> Optional[np.ndarray]:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim == 1:
        codes = codes[None, :]
    try:
        w = processor.decode_and_postprocess(torch.from_numpy(codes).unsqueeze(0))
    except Exception as exc:
        print(f"[decode] {tag}: {type(exc).__name__}: {exc}")
        return None
    return np.asarray(w, dtype=np.float32).reshape(-1)


def stats(w: np.ndarray) -> Dict[str, float]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return {"sec": 0.0, "rms": 0.0, "peak": 0.0, "flatness": 0.0, "zcr": 0.0}
    spec = np.abs(np.fft.rfft(w)) + 1e-12
    return {"sec": round(len(w) / SR, 3), "rms": round(float(np.sqrt(np.mean(w * w))), 5),
            "peak": round(float(np.max(np.abs(w))), 5),
            "flatness": round(float(np.exp(np.mean(np.log(spec))) / np.mean(spec)), 4),
            "zcr": round(float(np.mean(np.abs(np.diff(np.signbit(w).astype(np.int8))))), 4)}


def pad9(cb0: np.ndarray) -> np.ndarray:
    out = np.zeros((9, cb0.shape[-1]), dtype=np.int64)
    out[0] = cb0
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="codeswitch_ckpt")
    ap.add_argument("--teacher-cache", default="teacher_cache")
    ap.add_argument("--out", default="/kaggle/working/audio_diag")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--frames", type=int, default=86)
    ap.add_argument("--audio-frames", type=int, default=172)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] ckpt={args.ckpt} cache={args.teacher_cache} device={device}")

    draft, residual, meta = load_models(Path(args.ckpt), device)
    print("\n=== meta ===")
    print(json.dumps({k: meta.get(k) for k in ("draft", "residual", "length_model")},
                     indent=2, ensure_ascii=False)[:4000])

    shards = load_shards(Path(args.teacher_cache), args.shards)
    processor = None
    if not args.no_audio:
        processor = AudioProcessor(CodeSwitchTTSConfig().audio)
        print(f"[audio] DAC backend: {processor.dac.backend}")
        if processor.dac.backend == "fallback_neural":
            print("[audio] ⚠ DAC weights unavailable; wavs are synthetic tones (code metrics still valid)")
    rows, wav_rows = [], []
    for shard in shards:
        codes = np.asarray(shard.codes, dtype=np.int64)
        if codes.shape[0] != 9 and codes.shape[1] == 9:
            codes = codes.T
        T = int(min(shard.frames, codes.shape[1]))
        codes = np.clip(codes[:, :T], 0, 1023)
        ds, da = draft_tokens(draft, shard.norm_text, T, True, device), \
                 draft_tokens(draft, shard.norm_text, T, False, device)
        win = []
        for s, e in windows(T, args.frames):
            tcb0, tres = codes[0, s:e], codes[1:9, s:e]
            _, rts = residual_tokens(residual, tcb0, True, device)
            _, rta = residual_tokens(residual, tcb0, False, device)
            _, rds = residual_tokens(residual, ds[s:e], True, device)
            _, rda = residual_tokens(residual, da[s:e], False, device)
            win.append({"cb0_s": top1(ds[s:e], tcb0), "cb0_a": top1(da[s:e], tcb0),
                        "rt_s": mean(per_cb(rts, tres)), "rt_a": mean(per_cb(rta, tres)),
                        "rd_s": mean(per_cb(rds, tres)), "rd_a": mean(per_cb(rda, tres))})
        row = {"shard": shard.norm_text[:60], "frames": T, "source": shard.target_source,
               "cb0_s": mean([w["cb0_s"] for w in win]), "cb0_a": mean([w["cb0_a"] for w in win]),
               "rt_s": mean([w["rt_s"] for w in win]), "rt_a": mean([w["rt_a"] for w in win]),
               "rd_s": mean([w["rd_s"] for w in win]), "rd_a": mean([w["rd_a"] for w in win])}
        rows.append(row)
        print(f"[shard {shard.index:4d}] T={T:3d} draft CB0 s/a={row['cb0_s']:.2f}/{row['cb0_a']:.2f}% "
              f"res teacher s/a={row['rt_s']:.2f}/{row['rt_a']:.2f}% "
              f"res draft s/a={row['rd_s']:.2f}/{row['rd_a']:.2f}%")

        if processor is None:
            continue
        A = min(args.audio_frames, T)
        dsA, daA = ds[:A], da[:A]
        fts, _ = residual_tokens(residual, codes[0, :A], True, device)
        fta, _ = residual_tokens(residual, codes[0, :A], False, device)
        fds, _ = residual_tokens(residual, dsA, True, device)
        fda, _ = residual_tokens(residual, daA, False, device)
        variants = {
            "teacher_full": codes[:, :A],
            "student_full_sample": fds,
            "student_full_argmax": fda,
            "teacher_cb0_student_res_sample": fts,
            "teacher_cb0_student_res_argmax": fta,
            "student_cb0_teacher_res_sample": np.vstack([dsA, codes[1:9, :A]]),
            "student_cb0_only_sample": pad9(dsA),
            "teacher_cb0_only": pad9(codes[0, :A]),
        }
        prefix = f"shard_{shard.index:04d}"
        for tag, cc in variants.items():
            w = decode(processor, cc, f"{prefix}/{tag}")
            if w is None:
                continue
            p = out / f"{prefix}_{tag}.wav"
            save_wav(p, w, sample_rate=SR)
            st = stats(w)
            st.update({"tag": tag, "file": str(p)})
            wav_rows.append(st)
            print(f"  [wav] {p.name:52s} {st['sec']:.2f}s rms={st['rms']:.4f} "
                  f"flat={st['flatness']:.3f} zcr={st['zcr']:.3f}")

    summary = {k: mean([r[k] for r in rows]) for k in ("cb0_s", "cb0_a", "rt_s", "rt_a", "rd_s", "rd_a")}
    print(f"\n=== SUMMARY ({len(rows)} shards) === chance={CHANCE:.4f}%")
    print(f"draft CB0        sample={summary['cb0_s']:.3f}% argmax={summary['cb0_a']:.3f}%")
    print(f"residual teacher sample={summary['rt_s']:.3f}% argmax={summary['rt_a']:.3f}%")
    print(f"residual draft   sample={summary['rd_s']:.3f}% argmax={summary['rd_a']:.3f}%")
    print("\n=== VERDICT ===")
    if summary["rt_a"] < 0.5:
        print("- Residual is at chance on its OWN training shards -> it never learned CB1-8. "
              "Retrain the residual; inspect its loss/cb_top1 curve.")
    elif summary["rt_a"] >= 5 and summary["rd_a"] < 1:
        print("- Residual works on teacher CB0 but collapses on draft CB0 -> the draft CB0 is the poison.")
    elif summary["rt_a"] >= 5 and summary["rd_a"] >= 5:
        print("- Residual works on both teacher and draft CB0 at the code level; if audio is still "
              "noise, the DAC code layout/order is wrong (compare teacher_full.wav).")
    else:
        print(f"- Weak residual (teacher {summary['rt_a']:.3f}% / draft {summary['rd_a']:.3f}%): "
              "inspect per-codebook numbers and the wavs.")
    if abs(summary["cb0_s"] - summary["cb0_a"]) > 2:
        print(f"- Draft sampling changes CB0 top-1 by {abs(summary['cb0_s'] - summary['cb0_a']):.2f} points.")
    if abs(summary["rt_s"] - summary["rt_a"]) > 2:
        print("- Residual sampling changes CB1-8 top-1 by more than 2 points.")

    report = {"summary": summary, "per_shard": rows, "wavs": wav_rows, "meta": meta}
    (out / "diagnostic_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[done] {out / 'diagnostic_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
