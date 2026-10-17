"""Neural network architectures for CodeSwitch-TTS."""

from codeswitch_tts.models.draft_transformer import DraftTransformer, DraftOutput
from codeswitch_tts.models.residual_decoder import ResidualDecoder, ResidualOutput

__all__ = [
    "DraftTransformer",
    "DraftOutput",
    "ResidualDecoder",
    "ResidualOutput",
]
