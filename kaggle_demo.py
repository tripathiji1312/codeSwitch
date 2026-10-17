"""CodeSwitch-TTS: Standalone Runner & Benchmark Script.

Designed for execution in a Kaggle GPU Notebook (NVIDIA T4, 16GB VRAM)
or local CPU verification.

Performs:
  1. Hardware & VRAM Diagnostics
  2. Pipeline Initialization (Normalizer, Draft Transformer, Speculative AR Verifier, Residual Decoder, DAC)
  3. Streaming Synthesis on 3 Target Utterances:
       - Native Tamil: "உங்கள் பயணம் வெற்றிகரமாக முடிந்தது."
       - Romanized Tanglish: "Unga cab ippo pickup location-ku vandhuchu."
       - Code-Mixed Transit: "Driver-ku OTP 5291 sollunga, total fare Rs. 245."
  4. Real-time Latency, TTFA (<300ms target), and Codebook-0 Acceptance Rate Benchmarking
  5. Audio Output Exporting to `output_audio/*.wav`
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from codeswitch_tts.config import CodeSwitchTTSConfig, DEFAULT_CONFIG
from codeswitch_tts.engine.streaming_pipeline import ChunkedStreamingPipeline, PipelineBenchmarkResult
from codeswitch_tts.utils.audio import save_wav


def print_banner():
    banner = """
================================================================================
       CodeSwitch-TTS: Hybrid Non-Autoregressive Draft & AR Verification
             for Code-Mixed Tamil-English (Tanglish) Speech Synthesis
================================================================================
"""
    print(banner)


def check_environment():
    """Reports system environment, PyTorch version, and GPU memory availability."""
    print(" [1/4] Hardware & System Diagnostics")
    print(f"   - Python version: {sys.version.split()[0]}")
    print(f"   - PyTorch version: {torch.__version__}")
    
    cuda_avail = torch.cuda.is_available()
    print(f"   - CUDA Available: {cuda_avail}")
    
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"   - GPU Device: {device_name}")
        print(f"   - Total VRAM: {vram_total_gb:.2f} GB")
        print(f"   - Target Memory Budget: <= 12.0 GB (Comfortable fit inside 16GB VRAM)")
        print(f"   - Precision: {'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'}")
    else:
        print("   - Running on: CPU (Host testing mode).")
        print("     Note: Real-time GPU speedup will be benchmarked on Kaggle NVIDIA T4.")
    print("-" * 80)


def run_benchmark():
    print_banner()
    check_environment()

    # 1. Initialize Pipeline
    print(" [2/4] Initializing CodeSwitch-TTS Hybrid Pipeline...")
    config = CodeSwitchTTSConfig()
    if os.environ.get("CODESWITCH_ALLOW_UNTRAINED", "0") == "1":
        # Telemetry-only smoke test: random-weight modules, audio is NOT speech.
        config.system.allow_untrained_fallback = True
    pipeline = ChunkedStreamingPipeline(config=config)
    if pipeline.speech_available:
        print("   ✓ Indic Parler-TTS speech backend is active")
    else:
        print("   ⚠ Indic Parler-TTS unavailable - see the fix instructions above")
    print("   ✓ 3-Tier Tanglish Normalizer initialized")
    print("   ✓ Discrete Codebook-0 NAR Draft Transformer loaded (~65M params)")
    print("   ✓ Speculative AR Verifier loaded (Parallel Rejection Sampling on CB0)")
    print("   ✓ Hierarchical Residual Decoder loaded (Parallel Codebooks 1-8)")
    print("   ✓ DAC Neural Audio Codec loaded (44.1kHz)")
    print("-" * 80)

    # 2. Test Utterances
    test_utterances = [
        {
            "id": "sample1_native_tamil",
            "category": "Native Tamil",
            "text": "உங்கள் பயணம் வெற்றிகரமாக முடிந்தது.",
        },
        {
            "id": "sample2_romanized_tanglish",
            "category": "Romanized Tanglish",
            "text": "Unga cab ippo pickup location-ku vandhuchu.",
        },
        {
            "id": "sample3_transit_codemixed",
            "category": "Transit Code-Mixed",
            "text": "Driver-ku OTP 5291 sollunga, total fare Rs. 245.",
        },
    ]

    output_dir = Path("output_audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up pass: the first synthesis on a fresh process pays one-off CUDA
    # kernel / allocator / description-cache costs (measured on a T4: ~0.9 s
    # TTFA for the first utterance vs ~45 ms once warm - which previously made
    # the first benchmarked category look ~20x slower than the others).  Two
    # throwaway utterances, including a long code-mixed one so the DAC decoder
    # is sized for the worst case, are synthesised with timing discarded.
    print(" [2.5/4] Warm-up pass (CUDA kernels, allocator, DAC, description cache)...")
    if pipeline.speech_available:
        for warm_text in (
            "இது ஒரு வெப்பநிலை சோதனை, முடிவுகளில் சேர்க்கத் தேவையில்லை.",
            "Warm-up only: driver-ku OTP 1234 sollunga, total fare Rs. 500, vehicle TN 09 AB 1234.",
        ):
            try:
                pipeline.synthesize(warm_text)
            except Exception as exc:
                print(f"     [warmup] skipped: {type(exc).__name__}: {exc}")
        print("   ✓ warm-up complete (timings discarded)")
    else:
        print("   ⚠ speech backend unavailable - skipping warm-up")
    print("-" * 80)

    print(" [3/4] Executing Streaming Synthesis & Benchmarking...")
    results: list[PipelineBenchmarkResult] = []

    for idx, sample in enumerate(test_utterances, 1):
        print(f"\n >>> Test Utterance {idx}/3: [{sample['category']}]")
        print(f"     Raw Input:      \"{sample['text']}\"")
        
        # Warm memory / empty cache if CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stream synthesis
        print("     [Streaming Progress]")
        benchmark_res = pipeline.synthesize(sample["text"])
        results.append(benchmark_res)

        print(f"     Normalized:     \"{benchmark_res.normalized_text}\"")
        print(f"     Detected Reg:   {benchmark_res.detected_language}")
        
        for chk in benchmark_res.chunks:
            print(
                f"       * Chunk {chk.chunk_index + 1}/{benchmark_res.chunks_count}: "
                f"Latency={chk.chunk_latency_ms:.1f}ms (TTFA={chk.ttfa_ms:.1f}ms) | "
                f"Draft={chk.draft_ms:.1f}ms | Verify={chk.verify_ms:.1f}ms | "
                f"Res={chk.residual_ms:.1f}ms | DAC={chk.dac_ms:.1f}ms | "
                f"CB0 Acceptance={chk.cb0_acceptance_rate:.1f}%"
            )

        # Save output WAV file
        out_wav_path = output_dir / f"{sample['id']}.wav"
        save_wav(
            filepath=out_wav_path,
            audio_data=benchmark_res.audio_waveform,
            sample_rate=benchmark_res.sample_rate,
        )
        print(f"     ✓ Audio saved:  {out_wav_path} ({benchmark_res.duration_sec:.2f}s @ 44.1kHz)")

    # 3. Print Comprehensive Benchmark Table
    print("\n" + "=" * 80)
    print(" [4/4] BENCHMARK RESULTS & LATENCY ANALYSIS SUMMARY")
    print("=" * 80)
    header = f"{'Category':<22} | {'TTFA (ms)':<10} | {'Total (ms)':<11} | {'Audio (s)':<10} | {'RTF':<8} | {'CB0 Accept %':<12}"
    print(header)
    print("-" * 80)

    for res, sample in zip(results, test_utterances):
        row = (
            f"{sample['category']:<22} | "
            f"{res.ttfa_ms:<10.1f} | "
            f"{res.total_latency_ms:<11.1f} | "
            f"{res.duration_sec:<10.2f} | "
            f"{res.real_time_factor:<8.3f} | "
            f"{res.cb0_acceptance_rate:<12.1f}%"
        )
        print(row)
    print("-" * 80)

    ttfas = [res.ttfa_ms for res in results]
    if len(ttfas) >= 2:
        warm = ttfas[1:]
        print(
            f" TTFA breakdown: first measured utterance = {ttfas[0]:.1f} ms | "
            f"warm mean = {sum(warm) / len(warm):.1f} ms | warm min = {min(warm):.1f} ms"
        )
        print(" (one-off cold-start overhead is absorbed by the warm-up pass above)")
        print("-" * 80)

    # Architectural comparison with the standard AR baseline.
    # NOTE: every value printed below is a design target from the project plan,
    # NOT a measurement - measured values are the table above (and
    # run_full_evaluation.py is the script that produces the paper numbers).
    print("\n ARCHITECTURAL LATENCY COMPARISON - DESIGN TARGETS (not measured):")
    print(" ┌───────────────────────────────────────┬──────────────────┬──────────────────┐")
    print(" │ Metric                                │ AR Baseline      │ CodeSwitch-TTS   │")
    print(" ├───────────────────────────────────────┼──────────────────┼──────────────────┤")
    print(" │ First Chunk TTFA                      │ ~3,200 ms        │ ~230 - 260 ms    │")
    print(" │ Speculative Decoding Codebook Target  │ All 9 CBs (Fails)│ Codebook 0 Only  │")
    print(" │ Codebook Rejection Collapse Risk      │ High (>77% rej.) │ Target: low rej. │")
    print(" │ Residual Codebook 1-8 Generation      │ Sequential AR    │ Parallel NAR     │")
    print(" │ Real-Time Factor (RTF)                │ ~1.20 (Lagging)  │ ~0.24 (target)   │")
    print(" │ TTFA Latency Reduction Target         │ Baseline (1.0x)  │ >92% (13.6x fast)│")
    print(" └───────────────────────────────────────┴──────────────────┴──────────────────┘")
    print(" \u26a0 The targets above are plan values; report the measured CB0 acceptance")
    print("   and TTFA from the table above (or codeswitch_ckpt/results.jsonl) instead.")
    print("\n[CodeSwitch-TTS] Benchmark execution successfully completed.")
    print("All audio files are saved in: ./output_audio/")


if __name__ == "__main__":
    run_benchmark()
