from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .config import VoiceRecognitionConfig

LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float32]


def load_audio_mono(path: Path, sample_rate: int = 16000) -> FloatArray:
    audio, source_sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio in {path}")
    mono = audio.mean(axis=1).astype(np.float32)
    if int(source_sample_rate) != sample_rate:
        mono = resample_audio(mono, int(source_sample_rate), sample_rate)
    return mono.astype(np.float32)


def resample_audio(audio: FloatArray, source_rate: int, target_rate: int) -> FloatArray:
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return array
    if source_rate == target_rate:
        return array.astype(np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    duration = array.shape[0] / float(source_rate)
    target_count = max(1, int(round(duration * target_rate)))
    source_positions = np.arange(array.shape[0], dtype=np.float64) / float(source_rate)
    target_positions = np.arange(target_count, dtype=np.float64) / float(target_rate)
    resampled = np.interp(target_positions, source_positions, array).astype(np.float32)
    return resampled


def trim_voice_audio(audio: FloatArray, config: VoiceRecognitionConfig) -> FloatArray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1).astype(np.float32)
    if array.size == 0:
        return array

    frame_length = min(2048, max(1, array.size))
    hop_length = max(1, frame_length // 4)
    padded = np.pad(array, (frame_length // 2, frame_length // 2), mode="constant")
    frame_count = 1 + max(0, (padded.size - frame_length) // hop_length)
    rms = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_length
        frame = padded[start : start + frame_length]
        rms[index] = np.sqrt(np.mean(frame * frame, dtype=np.float64))

    ref = float(np.max(rms))
    if ref <= 1e-9:
        return array.astype(np.float32)

    threshold = ref * (10.0 ** (-config.trim_top_db / 20.0))
    non_silent = np.flatnonzero(rms > threshold)
    if non_silent.size == 0:
        LOGGER.debug(
            "Audio was entirely trimmed at top_db=%s; using untrimmed signal.",
            config.trim_top_db,
        )
        return array.astype(np.float32)

    start = int(non_silent[0] * hop_length)
    end = min(array.size, int((non_silent[-1] + 1) * hop_length))
    return array[start:end].astype(np.float32)
