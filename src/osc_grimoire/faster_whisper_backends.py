from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .config import VoiceRecognitionConfig
from .voice_embedding_backends import (
    _dtw_distance,
    _huggingface_token,
    _l2_normalize,
    _whisper_frame_count,
)
from .voice_features import FloatArray, trim_voice_audio
from .voice_recognizer import VoiceTemplateBackend

LOGGER = logging.getLogger(__name__)

DEFAULT_FASTER_WHISPER_MODEL = "tiny"
DEFAULT_FASTER_WHISPER_COMPUTE_TYPE = "int8"


class MissingFasterWhisperDependenciesError(RuntimeError):
    pass


@dataclass(frozen=True)
class FasterWhisperBundle:
    model_name: str
    model: Any


def missing_faster_whisper_dependencies_message() -> str:
    return (
        "faster-whisper backends require `faster-whisper`. Run `uv sync`, then retry."
    )


def faster_whisper_dtw_backend(
    model_name: str = DEFAULT_FASTER_WHISPER_MODEL,
) -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name=_backend_name("faster-whisper-dtw", model_name),
        extract_path=lambda path, config: _extract_faster_whisper_frames_from_path(
            path, config, model_name
        ),
        extract_array=lambda audio, config, sample_rate: _extract_faster_whisper_frames(
            audio, config, sample_rate, model_name
        ),
        distance=_dtw_distance,
        aggregate=lambda distances: float(np.median(distances)),
    )


def _backend_name(kind: str, model_name: str) -> str:
    short_model = model_name.rsplit("/", 1)[-1]
    return f"{kind}:{short_model}"


def _extract_faster_whisper_frames_from_path(
    path: Path, config: VoiceRecognitionConfig, model_name: str
) -> FloatArray:
    audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    return _extract_faster_whisper_frames(
        audio.astype(np.float32), config, int(sample_rate), model_name
    )


def _extract_faster_whisper_frames(
    audio: FloatArray,
    config: VoiceRecognitionConfig,
    sample_rate: int,
    model_name: str,
) -> FloatArray:
    audio = _prepare_audio(audio, config, sample_rate)
    bundle = _load_faster_whisper_model(model_name)
    features = _extract_whisper_features(
        audio,
        bundle.model.feature_extractor,
    )
    encoded = bundle.model.encode(features)
    hidden = np.asarray(encoded, dtype=np.float32).squeeze(0)
    frame_count = _whisper_frame_count(audio, hidden.shape[0])
    return _l2_normalize(hidden[:frame_count])


def _prepare_audio(
    audio: FloatArray, config: VoiceRecognitionConfig, sample_rate: int
) -> FloatArray:
    if audio.size == 0:
        raise ValueError("Empty audio array")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return trim_voice_audio(audio, config)


def _extract_whisper_features(audio: FloatArray, feature_extractor: Any) -> FloatArray:
    target_samples = int(feature_extractor.n_samples)
    if audio.shape[0] < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.shape[0]))
    else:
        audio = audio[:target_samples]
    features = feature_extractor(audio.astype(np.float32), padding=0)
    return features[..., : feature_extractor.nb_max_frames].astype(np.float32)


@functools.lru_cache(maxsize=4)
def _load_faster_whisper_model(model_name: str) -> FasterWhisperBundle:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise MissingFasterWhisperDependenciesError(
            missing_faster_whisper_dependencies_message()
        ) from exc

    LOGGER.info("Loading faster-whisper model %s", model_name)
    model = WhisperModel(
        _normalize_model_name(model_name),
        device="cpu",
        compute_type=DEFAULT_FASTER_WHISPER_COMPUTE_TYPE,
        use_auth_token=_huggingface_token(),
    )
    return FasterWhisperBundle(model_name=model_name, model=model)


def _normalize_model_name(model_name: str) -> str:
    if model_name.startswith("openai/whisper-"):
        return model_name.removeprefix("openai/whisper-")
    if model_name.startswith("Systran/faster-whisper-"):
        return model_name.removeprefix("Systran/faster-whisper-")
    return model_name
