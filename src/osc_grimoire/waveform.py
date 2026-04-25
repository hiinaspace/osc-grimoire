from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .voice_features import FloatArray


def load_waveform_preview(path: Path, points: int = 160) -> FloatArray:
    audio, _sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    return downsample_waveform(mono, points)


def downsample_waveform(audio: FloatArray, points: int = 160) -> FloatArray:
    if points <= 0:
        raise ValueError("points must be positive")
    if audio.size == 0:
        return np.zeros(points, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size <= points:
        padded = np.zeros(points, dtype=np.float32)
        padded[: audio.size] = audio
        return _normalize_preview(padded)

    edges = np.linspace(0, audio.size, points + 1, dtype=np.int64)
    preview = np.empty(points, dtype=np.float32)
    for i in range(points):
        chunk = audio[edges[i] : edges[i + 1]]
        if chunk.size == 0:
            preview[i] = 0.0
        else:
            index = int(np.argmax(np.abs(chunk)))
            preview[i] = chunk[index]
    return _normalize_preview(preview)


def _normalize_preview(preview: FloatArray) -> FloatArray:
    peak = float(np.max(np.abs(preview))) if preview.size else 0.0
    if peak <= 1e-9:
        return preview.astype(np.float32)
    return (preview / peak).astype(np.float32)
