from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

from .config import VoiceRecognitionConfig

LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float32]


def extract_mfcc(
    wav_path: Path, config: VoiceRecognitionConfig, sample_rate: int = 16000
) -> FloatArray:
    audio, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio in {wav_path}")
    trimmed = trim_voice_audio(audio, config)
    mfcc = librosa.feature.mfcc(y=trimmed, sr=sample_rate, n_mfcc=config.n_mfcc)
    if config.drop_mfcc_c0:
        mfcc = mfcc[1:]
    # librosa returns (n_mfcc, T); DTW wants (T, n_mfcc).
    features = mfcc.T.astype(np.float32)
    return _normalize_cepstra(features) if config.cepstral_normalize else features


def extract_mfcc_from_array(
    audio: FloatArray, config: VoiceRecognitionConfig, sample_rate: int = 16000
) -> FloatArray:
    if audio.size == 0:
        raise ValueError("Empty audio array")
    trimmed = trim_voice_audio(audio, config)
    mfcc = librosa.feature.mfcc(y=trimmed, sr=sample_rate, n_mfcc=config.n_mfcc)
    if config.drop_mfcc_c0:
        mfcc = mfcc[1:]
    features = mfcc.T.astype(np.float32)
    return _normalize_cepstra(features) if config.cepstral_normalize else features


def trim_voice_audio(audio: FloatArray, config: VoiceRecognitionConfig) -> FloatArray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1).astype(np.float32)
    trimmed, _ = librosa.effects.trim(array, top_db=config.trim_top_db)
    if trimmed.size == 0:
        LOGGER.debug(
            "Audio was entirely trimmed at top_db=%s; using untrimmed signal.",
            config.trim_top_db,
        )
        return array.astype(np.float32)
    return trimmed.astype(np.float32)


def _normalize_cepstra(features: FloatArray) -> FloatArray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)
