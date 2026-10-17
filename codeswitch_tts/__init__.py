"""CodeSwitch-TTS: A Hybrid Non-Autoregressive Draft and Autoregressive
Verification Framework for Code-Mixed Tamil-English Speech Synthesis.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

# Ensure google.protobuf.runtime_version is available & API-correct even on
# stale protobuf 4.x runtimes (Kaggle safety net - see codeswitch_tts/_compat.py).
from codeswitch_tts._compat import install_protobuf_runtime_version_fallback

install_protobuf_runtime_version_fallback()

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
from codeswitch_tts.utils.assets import (
    VoiceProfile,
    load_voice_profiles,
    load_proper_nouns,
    load_character_vocab,
    get_assets_dir,
)

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
    "VoiceProfile",
    "load_voice_profiles",
    "load_proper_nouns",
    "load_character_vocab",
    "get_assets_dir",
]
