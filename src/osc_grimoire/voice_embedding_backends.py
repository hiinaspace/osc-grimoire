from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .config import VoiceRecognitionConfig
from .voice_features import FloatArray
from .voice_recognizer import VoiceTemplateBackend

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "microsoft/wavlm-base-plus"


class MissingEmbeddingDependenciesError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingModelBundle:
    model_name: str
    feature_extractor: Any
    model: Any
    torch: Any


def missing_embedding_dependencies_message() -> str:
    return (
        "Embedding backends require optional ML dependencies. "
        "Run `uv sync --group ml`, then retry."
    )


def wavlm_dtw_backend(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name=_backend_name("wavlm-dtw", model_name),
        extract_path=lambda path, config: _extract_frame_embeddings_from_path(
            path, config, model_name
        ),
        extract_array=lambda audio, config, sample_rate: _extract_frame_embeddings(
            audio, config, sample_rate, model_name
        ),
        distance=_dtw_distance,
        aggregate=lambda distances: float(np.median(distances)),
    )


def wavlm_mean_backend(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name=_backend_name("wavlm-mean", model_name),
        extract_path=lambda path, config: _extract_mean_embedding_from_path(
            path, config, model_name
        ),
        extract_array=lambda audio, config, sample_rate: _extract_mean_embedding(
            audio, config, sample_rate, model_name
        ),
        distance=_cosine_distance,
        aggregate=lambda distances: float(np.median(distances)),
    )


def _backend_name(kind: str, model_name: str) -> str:
    short_model = model_name.rsplit("/", 1)[-1]
    return f"{kind}:{short_model}"


def _extract_frame_embeddings_from_path(
    path: Path, config: VoiceRecognitionConfig, model_name: str
) -> FloatArray:
    audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    return _extract_frame_embeddings(
        audio.astype(np.float32), config, int(sample_rate), model_name
    )


def _extract_mean_embedding_from_path(
    path: Path, config: VoiceRecognitionConfig, model_name: str
) -> FloatArray:
    audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    return _extract_mean_embedding(
        audio.astype(np.float32), config, int(sample_rate), model_name
    )


def _extract_frame_embeddings(
    audio: FloatArray,
    config: VoiceRecognitionConfig,
    sample_rate: int,
    model_name: str,
) -> FloatArray:
    audio = _prepare_audio(audio, config, sample_rate)
    bundle = _load_model(model_name)
    inputs = bundle.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
    )
    with bundle.torch.inference_mode():
        outputs = bundle.model(**inputs)
    hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
    hidden = hidden.astype(np.float32)
    return _l2_normalize(hidden)


def _extract_mean_embedding(
    audio: FloatArray,
    config: VoiceRecognitionConfig,
    sample_rate: int,
    model_name: str,
) -> FloatArray:
    frames = _extract_frame_embeddings(audio, config, sample_rate, model_name)
    pooled = frames.mean(axis=0, keepdims=True)
    return _l2_normalize(pooled)


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
    trimmed, _ = librosa.effects.trim(audio, top_db=config.trim_top_db)
    if trimmed.size == 0:
        trimmed = audio
    return trimmed.astype(np.float32)


@functools.lru_cache(maxsize=2)
def _load_model(model_name: str) -> EmbeddingModelBundle:
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise MissingEmbeddingDependenciesError(
            missing_embedding_dependencies_message()
        ) from exc

    LOGGER.info("Loading embedding model %s", model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return EmbeddingModelBundle(
        model_name=model_name,
        feature_extractor=feature_extractor,
        model=model,
        torch=torch,
    )


def _l2_normalize(features: FloatArray) -> FloatArray:
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return (features / np.maximum(norm, 1e-9)).astype(np.float32)


def _dtw_distance(a: FloatArray, b: FloatArray) -> float:
    from dtaidistance import dtw_ndim

    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    return float(dtw_ndim.distance(a64, b64))


def _cosine_distance(a: FloatArray, b: FloatArray) -> float:
    a_vec = np.asarray(a, dtype=np.float32).reshape(-1)
    b_vec = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-9:
        return 1.0
    similarity = float(np.dot(a_vec, b_vec) / denom)
    return 1.0 - max(min(similarity, 1.0), -1.0)
