"""CodeSwitch-TTS: A Hybrid Non-Autoregressive Draft and Autoregressive
Verification Framework for Code-Mixed Tamil-English Speech Synthesis.
"""

from codeswitch_tts.config import (
    CodeSwitchTTSConfig,
    DEFAULT_CONFIG,
    AudioConfig,
    DraftModelConfig,
    ResidualModelConfig,
    SpeculativeEngineConfig,
    SystemConfig,
)
from codeswitch_tts.frontend.normalizer import TanglishNormalizer
from codeswitch_tts.models.draft_transformer import DraftTransformer
from codeswitch_tts.models.residual_decoder import ResidualDecoder
from codeswitch_tts.engine.speculative_engine import SpeculativeEngine
from codeswitch_tts.engine.streaming_pipeline import ChunkedStreamingPipeline
from codeswitch_tts.utils.audio import AudioProcessor, DACWrapper

__all__ = [
    "CodeSwitchTTSConfig",
    "DEFAULT_CONFIG",
    "AudioConfig",
    "DraftModelConfig",
    "ResidualModelConfig",
    "SpeculativeEngineConfig",
    "SystemConfig",
    "TanglishNormalizer",
    "DraftTransformer",
    "ResidualDecoder",
    "SpeculativeEngine",
    "ChunkedStreamingPipeline",
    "AudioProcessor",
    "DACWrapper",
]
