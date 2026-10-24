"""Data preparation and alignment utilities for CodeSwitch-TTS.

This package is deliberately independent of the training entrypoint so the
offline data pipeline can be imported and tested without loading Parler, DAC,
or a CTC model.
"""

from codeswitch_tts.data.shards import SpeechShard, stable_shard_index

__all__ = ["SpeechShard", "stable_shard_index"]
