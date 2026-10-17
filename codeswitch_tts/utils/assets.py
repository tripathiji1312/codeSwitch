"""Asset manager for CodeSwitch-TTS: Voice Profiles, Gazetteers, and Vocabularies.

Provides access to:
  1. Zero-shot voice profiles (ref.wav + voice.yaml) from `movio/config/voices/`
  2. Proper noun gazetteer (transit locations, ride-hailing brands) from `movio/config/gazetteer/`
  3. Character/letter dictionary from `movio/models/indicf5_tanglish/vocab.txt`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@dataclass
class VoiceProfile:
    """Voice profile containing reference audio and transcript for zero-shot cloning."""
    name: str
    ref_audio_path: str
    ref_text: str
    gender: str = "neutral"
    style: str = "neutral"
    speed_factor: float = 1.0
    pitch_factor: float = 1.0


def get_assets_dir() -> Path:
    """Returns absolute path to the codeswitch_tts/assets directory."""
    return ASSETS_DIR


def load_voice_profiles(voices_dir: Optional[Union[str, Path]] = None) -> Dict[str, VoiceProfile]:
    """Loads all zero-shot voice profiles (ref.wav + voice.yaml) from assets or custom directory."""
    dir_path = Path(voices_dir) if voices_dir else ASSETS_DIR / "voices"
    profiles: Dict[str, VoiceProfile] = {}
    if not dir_path.exists():
        return profiles

    for voice_folder in sorted(dir_path.iterdir()):
        if not voice_folder.is_dir():
            continue
        meta_yaml = voice_folder / "voice.yaml"
        ref_wav = voice_folder / "ref.wav"
        if not meta_yaml.exists():
            continue

        meta: Dict[str, Union[str, float]] = {}
        try:
            import yaml
            meta = yaml.safe_load(meta_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            for line in meta_yaml.read_text(encoding="utf-8").splitlines():
                if ":" in line and not line.strip().startswith("#"):
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")

        audio_rel = meta.get("ref_audio", "ref.wav")
        audio_path = (
            str((voice_folder / str(audio_rel)).resolve())
            if (voice_folder / str(audio_rel)).exists()
            else str(ref_wav.resolve())
        )
        ref_text = str(meta.get("ref_text", "")).strip()

        prof = VoiceProfile(
            name=str(meta.get("name", voice_folder.name)),
            ref_audio_path=audio_path,
            ref_text=ref_text,
            gender=str(meta.get("gender", "neutral")),
            style=str(meta.get("style", "neutral")),
            speed_factor=float(meta.get("speed_factor", 1.0)),
            pitch_factor=float(meta.get("pitch_factor", 1.0)),
        )
        profiles[prof.name] = prof

    return profiles


def load_proper_nouns(gazetteer_path: Optional[Union[str, Path]] = None) -> Set[str]:
    """Loads Latin-script proper nouns, location names, and brand terms to preserve from transliteration."""
    path = Path(gazetteer_path) if gazetteer_path else ASSETS_DIR / "gazetteer" / "proper_nouns.txt"
    if not path.exists():
        return set()
    nouns: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            nouns.add(line.lower())
    return nouns


def load_character_vocab(vocab_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Loads the character/letter dictionary used by character-level DiT models."""
    path = Path(vocab_path) if vocab_path else ASSETS_DIR / "vocab" / "vocab.txt"
    if not path.exists():
        return []
    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
