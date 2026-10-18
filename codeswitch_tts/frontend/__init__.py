"""Frontend text processing and normalization package for CodeSwitch-TTS."""

from codeswitch_tts.frontend.normalizer import TanglishNormalizer, NormalizationResult
from codeswitch_tts.frontend.tamil_latin import (
    latinize_text,
    transliterate_word,
)

__all__ = [
    "TanglishNormalizer",
    "NormalizationResult",
    "transliterate_word",
    "latinize_text",
]
