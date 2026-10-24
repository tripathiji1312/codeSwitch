#!/usr/bin/env python3
"""Prepare real-speech DAC shards for draft training.

Examples
--------
Stream the gated Rasa Tamil split (HF_TOKEN must be in the environment):

    export HF_TOKEN=...
    python prepare_real_speech.py \
        --out real_cache/rasa_ta \
        --hf-dataset ai4bharat/Rasa --hf-config Tamil --hf-split train \
        --language ta --limit 200 --max-hours 2

Encode a local parquet tree (the same layout 01_download_data.py produces):

    python prepare_real_speech.py \
        --out real_cache/indicvoices_ta \
        --input-dir raw_audio/indicvoices_r \
        --language ta --durations ctc

Augment an existing shard cache without durations:

    python prepare_real_speech.py --out real_cache/smoke \
        --manifest smoke.jsonl --durations none --limit 1

The output is the same ``shard_*.npz`` contract as ``teacher_cache``.  Point
``train_draft_distill.py --teacher-cache real_cache/...`` at it; the trainer
will consume the optional ``char_durations`` field automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from codeswitch_tts.data.duration import LazyCTCDurationExtractor
from codeswitch_tts.data.real_speech import (
    DACEncoder,
    PipelineConfig,
    RealSpeechPipeline,
    SpeechRow,
    iter_hf_dataset,
    iter_manifest,
    iter_parquet_dir,
)

DEFAULT_TAMIL_CTC = "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250"
DEFAULT_ENGLISH_CTC = "facebook/wav2vec2-large-960h-lv60-self"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode transcribed speech into CodeSwitch-TTS DAC shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True, help="Output shard directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSONL/CSV with audio + text fields.")
    source.add_argument("--input-dir", type=Path, help="Recursively read *.parquet under this directory.")
    source.add_argument("--hf-dataset", type=str, help="Hugging Face dataset id, e.g. ai4bharat/Rasa.")

    parser.add_argument("--hf-config", type=str, default=None, help="HF dataset config name (e.g. Tamil).")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--language", type=str, default="",
                        help="Default language tag for rows without one (ta/en/code-mixed).")
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--audio-column", type=str, default=None)
    parser.add_argument("--speaker-column", type=str, default=None)
    parser.add_argument("--style-filter", type=str, default="",
                        help="Comma-separated style values to keep (empty keeps all).")
    parser.add_argument("--source-name", type=str, default="real_speech")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Parquet row batch size (HF streaming is one row at a time).")

    parser.add_argument("--min-duration", type=float, default=0.6)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many new shards (0 = all).")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="Stop after this many encoded audio hours (0 = all).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode rows even when a matching shard already exists.")

    parser.add_argument("--normalize", dest="normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--durations", choices=["ctc", "none"], default="ctc",
                        help="CTC-align normalized text; use 'none' for codec-only shards.")
    parser.add_argument("--ctc-model-ta", type=str, default=DEFAULT_TAMIL_CTC)
    parser.add_argument("--ctc-model-en", type=str, default=DEFAULT_ENGLISH_CTC)
    parser.add_argument("--ctc-model-code-mixed", type=str, default=None,
                        help="Optional CTC model for code-mixed text; usually unavailable.")
    parser.add_argument("--ctc-revision", type=str, default=None,
                        help="Optional HF revision for every CTC model.")
    parser.add_argument("--require-durations", action="store_true",
                        help="Fail if any row cannot be aligned.")
    parser.add_argument("--dac-model", type=str, default="ylacombe/dac_44khz")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device for DAC and CTC (default: cuda when available).")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print row progress every N input rows (0 disables).")
    return parser


def _parse_filter(value: str) -> Sequence[str]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _log_rows(rows: Iterable[SpeechRow], every: int) -> Iterator[SpeechRow]:
    count = 0
    for row in rows:
        count += 1
        if every and count % every == 0:
            print(
                f"[Data] read {count} rows | latest language={row.language or '?'} "
                f"| chars={len(row.text)}"
            )
        yield row


def _build_rows(args: argparse.Namespace) -> Iterable[SpeechRow]:
    style_filter = _parse_filter(args.style_filter)
    if args.manifest is not None:
        return iter_manifest(
            args.manifest,
            default_language=args.language,
            text_column=args.text_column,
            audio_column=args.audio_column,
            speaker_column=args.speaker_column,
            style_filter=style_filter,
        )
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise NotADirectoryError(f"--input-dir is not a directory: {args.input_dir}")
        return iter_parquet_dir(
            args.input_dir,
            default_language=args.language,
            batch_size=args.batch_size,
            text_column=args.text_column,
            audio_column=args.audio_column,
            speaker_column=args.speaker_column,
            style_filter=style_filter,
        )
    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        print(
            "[Data] HF_TOKEN is not set. Rasa and IndicVoices-R are gated; "
            "accept their terms on the Hub and export HF_TOKEN.",
            file=sys.stderr,
        )
    return iter_hf_dataset(
        args.hf_dataset,
        config=args.hf_config,
        split=args.hf_split,
        default_language=args.language,
        text_column=args.text_column,
        audio_column=args.audio_column,
        speaker_column=args.speaker_column,
        style_filter=style_filter,
        token=token,
    )


def _build_extractors(
    args: argparse.Namespace, device: str
) -> dict:
    if args.durations == "none":
        return {}
    extractors = {
        "ta": LazyCTCDurationExtractor(
            args.ctc_model_ta, device=device, revision=args.ctc_revision
        ),
        "en": LazyCTCDurationExtractor(
            args.ctc_model_en, device=device, revision=args.ctc_revision
        ),
    }
    if args.ctc_model_code_mixed:
        extractors["code-mixed"] = LazyCTCDurationExtractor(
            args.ctc_model_code_mixed, device=device, revision=args.ctc_revision
        )
    return extractors


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("CodeSwitch-TTS · real-speech shard preparation")
    print(f"  out: {args.out} | device: {device} | durations: {args.durations}")
    print("=" * 78)

    if args.limit < 0 or args.max_hours < 0:
        print("[Data] --limit and --max-hours must be >= 0", file=sys.stderr)
        return 2
    if args.min_duration < 0 or args.max_duration <= args.min_duration:
        print("[Data] require 0 <= --min-duration < --max-duration", file=sys.stderr)
        return 2
    if args.require_durations and args.durations == "none":
        print(
            "[Data] --require-durations cannot be combined with --durations none",
            file=sys.stderr,
        )
        return 2

    normalizer = None
    if args.normalize:
        from codeswitch_tts import TanglishNormalizer

        normalizer = TanglishNormalizer()

    config = PipelineConfig(
        out_dir=args.out,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        min_chars=args.min_chars,
        max_frames=args.max_frames,
        limit=args.limit,
        max_hours=args.max_hours,
        overwrite=args.overwrite,
        require_durations=args.require_durations,
        source_name=args.source_name,
        dac_model_id=args.dac_model,
        duration_models=(
            {
                "ta": args.ctc_model_ta,
                "en": args.ctc_model_en,
                **(
                    {"code-mixed": args.ctc_model_code_mixed}
                    if args.ctc_model_code_mixed
                    else {}
                ),
            }
            if args.durations == "ctc"
            else {}
        ),
    )

    try:
        dac = DACEncoder(args.dac_model, device=device)
    except Exception as exc:
        print(f"[Data] failed to load DAC encoder: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    extractors = _build_extractors(args, device)
    pipeline = RealSpeechPipeline(
        config,
        dac_encoder=dac,
        duration_extractors=extractors,
        normalizer=normalizer,
    )
    try:
        stats = pipeline.run(_log_rows(_build_rows(args), args.log_every))
    except Exception as exc:
        print(f"[Data] pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if (args.out / "meta.json").is_file():
            print(f"[Data] partial summary: {args.out / 'meta.json'}", file=sys.stderr)
        return 2

    print("-" * 78)
    print(
        f"[Data] rows={stats['rows_seen']} written={stats['written']} "
        f"skipped={stats['skipped_existing']} errors={stats['errors']} | "
        f"audio={stats['audio_seconds'] / 3600.0:.2f} h"
    )
    print(
        f"[Data] filters: short={stats['filtered_short']} long={stats['filtered_long']} "
        f"chars={stats['filtered_chars']} frames={stats['filtered_frames']} | "
        f"durations ok={stats['duration_ok']} missing={stats['duration_missing']}"
    )
    if stats.get("errors_detail"):
        print("[Data] first errors:")
        for message in stats["errors_detail"][:10]:
            print(f"  - {message}")
    print(f"[Data] shards: {args.out}")
    print(f"[Data] summary: {args.out / 'meta.json'}")
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
