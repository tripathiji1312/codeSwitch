"""Chunked Streaming Orchestrator for CodeSwitch-TTS.

Pipelined architecture streaming 1-second audio chunks:
  Chunk 1:
    - Draft CB0 (~50ms)
    - AR verify CB0 (~120ms)
    - NAR predict CB1-8 (~35ms)
    - DAC decode (~20ms)
    -> Total first chunk TTFA target: ~230-260ms (<300ms)

Subsequent chunks pipeline asynchronously while Chunk N is playing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from codeswitch_tts.config import CodeSwitchTTSConfig, DEFAULT_CONFIG
from codeswitch_tts.frontend.normalizer import TanglishNormalizer, NormalizationResult
from codeswitch_tts.models.draft_transformer import DraftTransformer, DraftOutput
from codeswitch_tts.models.residual_decoder import ResidualDecoder, ResidualOutput
from codeswitch_tts.engine.speculative_engine import SpeculativeEngine, SpeculativeVerificationResult
from codeswitch_tts.utils.audio import AudioProcessor, crossfade_chunks, normalize_peak


@dataclass
class StreamedAudioChunk:
    """Represents a single 1-second streamed audio chunk with telemetry."""
    chunk_index: int
    audio_data: np.ndarray             # 44.1kHz PCM float32 waveform (~44100 samples)
    sample_rate: int
    ttfa_ms: float                     # Time-To-First-Audio (relevant for chunk 0)
    chunk_latency_ms: float            # Time taken to generate this specific chunk
    cb0_acceptance_rate: float         # Codebook-0 speculative acceptance rate (%)
    draft_ms: float
    verify_ms: float
    residual_ms: float
    dac_ms: float


@dataclass
class PipelineBenchmarkResult:
    """End-to-end benchmark summary for an utterance."""
    original_text: str
    normalized_text: str
    detected_language: str
    audio_waveform: np.ndarray
    sample_rate: int
    duration_sec: float
    ttfa_ms: float
    total_latency_ms: float
    real_time_factor: float            # RTF = total_latency / audio_duration
    cb0_acceptance_rate: float
    chunks_count: int
    chunks: List[StreamedAudioChunk] = field(default_factory=list)


class ChunkedStreamingPipeline:
    """Unified CodeSwitch-TTS hybrid speculative streaming pipeline."""

    def __init__(self, config: Optional[CodeSwitchTTSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.device = self.config.system.device
        self.dtype = self.config.system.torch_dtype

        # 1. Frontend Text Normalizer
        self.normalizer = TanglishNormalizer()

        # 2. Discrete Codebook-0 Draft Model (~65M parameters)
        self.draft_model = DraftTransformer(self.config.draft).to(self.device)
        self.draft_model.eval()

        # 3. Hierarchical Residual Decoder (Codebooks 1-8)
        self.residual_decoder = ResidualDecoder(self.config.residual).to(self.device)
        self.residual_decoder.eval()

        # 4. Speculative AR Verifier
        self.speculative_engine = SpeculativeEngine(
            self.config.speculative,
            device=self.device,
            dtype=self.dtype,
        )

        # 5. Audio Processor & DAC Decoder (reuses AR audio_encoder if available to save VRAM)
        dac_model = getattr(self.speculative_engine.ar_model, "audio_encoder", None)
        self.audio_processor = AudioProcessor(self.config.audio, dac_model=dac_model)

        print(f"[ChunkedStreamingPipeline] Initialized on {self.device} ({self.dtype}).")

    def _split_into_streaming_chunks(self, text: str, max_chunk_chars: int = 65) -> List[str]:
        """Segments normalized text into ~1-second synthesis chunks."""
        words = text.split()
        if not words:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > max_chunk_chars and current:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks

    def stream_synthesize(
        self,
        text: str,
        temperature: float = 1.0,
    ) -> Generator[StreamedAudioChunk, None, None]:
        """Generates and streams 1-second audio chunks sequentially.
        
        Yields StreamedAudioChunk as each chunk finishes synthesis.
        """
        # Step 1: 3-tier text normalization
        norm_res = self.normalizer.normalize(text)
        text_chunks = self._split_into_streaming_chunks(norm_res.normalized_text)

        pipeline_start = time.perf_counter()
        prefix_tokens: Optional[torch.Tensor] = None
        frames_per_chunk = self.config.audio.frames_per_chunk  # ~86 frames

        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_start = time.perf_counter()

            # -------------------------------------------------------------
            # Stage 1: Non-Autoregressive Draft Codebook 0 (~50ms)
            # -------------------------------------------------------------
            t_draft_0 = time.perf_counter()
            with torch.inference_mode():
                draft_out: DraftOutput = self.draft_model.generate_chunk(
                    text=chunk_text,
                    num_frames=frames_per_chunk,
                    temperature=temperature,
                    device=self.device,
                )
            draft_ms = (time.perf_counter() - t_draft_0) * 1000.0

            # -------------------------------------------------------------
            # Stage 2: Speculative AR Verification on Codebook 0 (~120ms)
            # -------------------------------------------------------------
            t_verif_0 = time.perf_counter()
            with torch.inference_mode():
                verif_res: SpeculativeVerificationResult = self.speculative_engine.verify_chunk(
                    draft_tokens=draft_out.tokens,
                    draft_probs=draft_out.probabilities,
                    prefix_tokens=prefix_tokens,
                    temperature=temperature,
                )
            verify_ms = (time.perf_counter() - t_verif_0) * 1000.0

            # Ensure verified sequence reaches chunk frame size
            verified_cb0 = verif_res.verified_tokens
            if verified_cb0.size(1) < frames_per_chunk:
                # Pad/complete remaining frames with draft tokens
                remaining = frames_per_chunk - verified_cb0.size(1)
                tail = draft_out.tokens[:, -remaining:]
                verified_cb0 = torch.cat([verified_cb0, tail], dim=1)
            elif verified_cb0.size(1) > frames_per_chunk:
                verified_cb0 = verified_cb0[:, :frames_per_chunk]

            prefix_tokens = verified_cb0  # Conditioning for next chunk

            # -------------------------------------------------------------
            # Stage 3: Hierarchical Residual Completion CB1-8 (~35ms)
            # -------------------------------------------------------------
            t_res_0 = time.perf_counter()
            with torch.inference_mode():
                residual_out: ResidualOutput = self.residual_decoder(
                    cb0_tokens=verified_cb0,
                    temperature=temperature,
                )
            residual_ms = (time.perf_counter() - t_res_0) * 1000.0

            # -------------------------------------------------------------
            # Stage 4: DAC Neural Decoding & Acoustic Post-Processing (~20ms)
            # -------------------------------------------------------------
            t_dac_0 = time.perf_counter()
            with torch.inference_mode():
                audio_waveform = self.audio_processor.decode_and_postprocess(
                    residual_out.full_codebooks
                )
            dac_ms = (time.perf_counter() - t_dac_0) * 1000.0

            chunk_latency_ms = (time.perf_counter() - chunk_start) * 1000.0
            ttfa_ms = (time.perf_counter() - pipeline_start) * 1000.0 if chunk_idx == 0 else chunk_latency_ms

            yield StreamedAudioChunk(
                chunk_index=chunk_idx,
                audio_data=audio_waveform,
                sample_rate=self.config.audio.sample_rate,
                ttfa_ms=ttfa_ms,
                chunk_latency_ms=chunk_latency_ms,
                cb0_acceptance_rate=verif_res.acceptance_rate,
                draft_ms=draft_ms,
                verify_ms=verify_ms,
                residual_ms=residual_ms,
                dac_ms=dac_ms,
            )

    def synthesize(self, text: str, temperature: float = 1.0) -> PipelineBenchmarkResult:
        """Runs full streaming synthesis, concatenates audio chunks with crossfading,
        and returns complete benchmark statistics.
        """
        norm_res = self.normalizer.normalize(text)
        chunks: List[StreamedAudioChunk] = []
        audio_parts: List[np.ndarray] = []
        start_total = time.perf_counter()

        for chunk in self.stream_synthesize(text, temperature=temperature):
            chunks.append(chunk)
            audio_parts.append(chunk.audio_data)

        total_latency_ms = (time.perf_counter() - start_total) * 1000.0

        # Seamlessly crossfade and concatenate chunks
        if len(audio_parts) == 1:
            full_audio = audio_parts[0]
        elif len(audio_parts) > 1:
            full_audio = audio_parts[0]
            for next_part in audio_parts[1:]:
                full_audio = crossfade_chunks(
                    full_audio,
                    next_part,
                    crossfade_samples=self.config.audio.crossfade_samples,
                )
        else:
            full_audio = np.zeros(self.config.audio.sample_rate, dtype=np.float32)

        full_audio = normalize_peak(full_audio, target_peak=0.95)
        duration_sec = len(full_audio) / self.config.audio.sample_rate
        rtf = (total_latency_ms / 1000.0) / max(duration_sec, 0.001)
        ttfa_ms = chunks[0].ttfa_ms if chunks else total_latency_ms
        avg_acceptance = (
            sum(c.cb0_acceptance_rate for c in chunks) / len(chunks) if chunks else 0.0
        )

        return PipelineBenchmarkResult(
            original_text=text,
            normalized_text=norm_res.normalized_text,
            detected_language=norm_res.detected_language,
            audio_waveform=full_audio,
            sample_rate=self.config.audio.sample_rate,
            duration_sec=duration_sec,
            ttfa_ms=ttfa_ms,
            total_latency_ms=total_latency_ms,
            real_time_factor=rtf,
            cb0_acceptance_rate=avg_acceptance,
            chunks_count=len(chunks),
            chunks=chunks,
        )
