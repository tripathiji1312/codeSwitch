"""CodeSwitch-TTS: A Hybrid Non-Autoregressive Draft and Autoregressive
Verification Framework for Code-Mixed Tamil-English Speech Synthesis.
"""

import os
import sys
import types

# Ensure Protocol Buffers uses pure Python backend to avoid Kaggle C++ runtime_version mismatches
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

try:
    import google.protobuf
    try:
        from google.protobuf import runtime_version
    except ImportError:
        _m = types.ModuleType("google.protobuf.runtime_version")
        _m.Domain = lambda *args, **kwargs: None
        _m.Version = lambda *args, **kwargs: None
        _m.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
        _m.DOMAIN = None
        _m.OSS_DOMAIN = None
        _m.VersionError = Exception
        google.protobuf.runtime_version = _m
        sys.modules["google.protobuf.runtime_version"] = _m
except Exception:
    pass

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
