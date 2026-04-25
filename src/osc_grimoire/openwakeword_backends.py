from __future__ import annotations

import functools
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .config import VoiceRecognitionConfig
from .voice_features import FloatArray
from .voice_recognizer import VoiceTemplateBackend

LOGGER = logging.getLogger(__name__)

MIN_OPENWAKEWORD_SAMPLES = 16000


class MissingOpenWakeWordDependenciesError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenWakeWordBundle:
    audio_features: Any


def missing_openwakeword_dependencies_message() -> str:
    return (
        "OpenWakeWord backends require optional ONNX dependencies and the "
        "OpenWakeWord source checkout. Run `uv sync --group oww`, then clone "
        "`https://github.com/dscripka/openWakeWord` to `S:\\lib\\openWakeWord` "
        "or set `OSC_GRIMOIRE_OPENWAKEWORD_REPO` to that checkout."
    )


def openwakeword_dtw_backend() -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name="oww-dtw:speech_embedding",
        extract_path=_extract_frame_embeddings_from_path,
        extract_array=_extract_frame_embeddings,
        distance=_dtw_distance,
        aggregate=lambda distances: float(np.median(distances)),
    )


def openwakeword_mean_backend() -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name="oww-mean:speech_embedding",
        extract_path=_extract_mean_embedding_from_path,
        extract_array=_extract_mean_embedding,
        distance=_cosine_distance,
        aggregate=lambda distances: float(np.median(distances)),
    )


def _extract_frame_embeddings_from_path(
    path: Path, config: VoiceRecognitionConfig
) -> FloatArray:
    audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    return _extract_frame_embeddings(audio.astype(np.float32), config, int(sample_rate))


def _extract_mean_embedding_from_path(
    path: Path, config: VoiceRecognitionConfig
) -> FloatArray:
    audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    return _extract_mean_embedding(audio.astype(np.float32), config, int(sample_rate))


def _extract_frame_embeddings(
    audio: FloatArray, config: VoiceRecognitionConfig, sample_rate: int
) -> FloatArray:
    prepared = _prepare_audio_pcm16(audio, config, sample_rate)
    bundle = _load_openwakeword()
    embeddings = bundle.audio_features.embed_clips(
        prepared[None, :], batch_size=1, ncpu=1
    )[0]
    return _l2_normalize(np.asarray(embeddings, dtype=np.float32))


def _extract_mean_embedding(
    audio: FloatArray, config: VoiceRecognitionConfig, sample_rate: int
) -> FloatArray:
    frames = _extract_frame_embeddings(audio, config, sample_rate)
    pooled = frames.mean(axis=0, keepdims=True)
    return _l2_normalize(pooled)


def _prepare_audio_pcm16(
    audio: FloatArray, config: VoiceRecognitionConfig, sample_rate: int
) -> np.ndarray:
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
    if trimmed.size < MIN_OPENWAKEWORD_SAMPLES:
        trimmed = np.pad(trimmed, (0, MIN_OPENWAKEWORD_SAMPLES - trimmed.size))
    clipped = np.clip(trimmed, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


@functools.lru_cache(maxsize=1)
def _load_openwakeword() -> OpenWakeWordBundle:
    try:
        _ensure_openwakeword_import_path()
        openwakeword = importlib.import_module("openwakeword")
        utils = importlib.import_module("openwakeword.utils")
    except (ImportError, OSError) as exc:
        raise MissingOpenWakeWordDependenciesError(
            missing_openwakeword_dependencies_message()
        ) from exc

    try:
        _download_feature_models(openwakeword, utils)
        audio_features = utils.AudioFeatures(inference_framework="onnx", ncpu=1)
    except Exception as exc:
        raise MissingOpenWakeWordDependenciesError(
            missing_openwakeword_dependencies_message()
        ) from exc

    LOGGER.info("Loaded OpenWakeWord speech_embedding ONNX feature extractor")
    return OpenWakeWordBundle(audio_features=audio_features)


def _download_feature_models(openwakeword: Any, utils: Any) -> None:
    models_dir = Path(openwakeword.FEATURE_MODELS["embedding"]["model_path"]).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    for feature_model in openwakeword.FEATURE_MODELS.values():
        onnx_url = feature_model["download_url"].replace(".tflite", ".onnx")
        onnx_path = models_dir / Path(onnx_url).name
        if not onnx_path.exists():
            utils.download_file(onnx_url, str(models_dir))


def _ensure_openwakeword_import_path() -> None:
    try:
        importlib.import_module("openwakeword")
        return
    except ImportError:
        pass

    for candidate in _openwakeword_repo_candidates():
        if (candidate / "openwakeword" / "utils.py").exists():
            sys.path.insert(0, str(candidate))
            return


def _openwakeword_repo_candidates() -> tuple[Path, ...]:
    env_path = os.environ.get("OSC_GRIMOIRE_OPENWAKEWORD_REPO")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("S:/lib/openWakeWord"))
    return tuple(candidates)


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
