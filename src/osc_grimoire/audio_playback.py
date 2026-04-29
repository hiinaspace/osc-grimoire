from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from .voice_features import FloatArray


class AudioPlayer(Protocol):
    def play_file(self, path: Path) -> None: ...


class SoundDeviceAudioPlayer:
    def play_file(self, path: Path) -> None:
        import sounddevice as sd

        audio, sample_rate = load_audio_for_playback(path)
        sd.stop()
        sd.play(audio, sample_rate, blocking=False)


def load_audio_for_playback(path: Path) -> tuple[FloatArray, int]:
    if not path.exists():
        raise ValueError(f"Audio sample not found: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio sample is empty: {path}")
    return audio, int(sample_rate)
