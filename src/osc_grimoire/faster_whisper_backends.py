from __future__ import annotations

import functools
import importlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dtaidistance import dtw_ndim

from .config import VoiceRecognitionConfig
from .voice_features import (
    FloatArray,
    load_audio_mono,
    resample_audio,
    trim_voice_audio,
)
from .voice_recognizer import VoiceTemplateBackend

LOGGER = logging.getLogger(__name__)

DEFAULT_FASTER_WHISPER_MODEL = "tiny"
DEFAULT_FASTER_WHISPER_COMPUTE_TYPE = "int8"
NBEST_BEAM_SIZE = 10
NBEST_HYPOTHESES = 10


class MissingFasterWhisperDependenciesError(RuntimeError):
    pass


@dataclass(frozen=True)
class FasterWhisperBundle:
    model_name: str
    model: Any


@dataclass(frozen=True)
class NBestHypothesis:
    text: str
    normalized_text: str
    tokens: tuple[int, ...]
    score: float
    weight: float


@dataclass(frozen=True)
class NBestFeature:
    hypotheses: tuple[NBestHypothesis, ...]


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


def faster_whisper_nbest_backend(
    model_name: str = DEFAULT_FASTER_WHISPER_MODEL,
) -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name=_backend_name("faster-whisper-nbest", model_name),
        extract_path=lambda path, config: _extract_faster_whisper_nbest_from_path(
            path, config, model_name
        ),
        extract_array=lambda audio, config, sample_rate: _extract_faster_whisper_nbest(
            audio, config, sample_rate, model_name
        ),
        distance=_nbest_distance,
        aggregate=lambda distances: float(np.median(distances)),
    )


def _backend_name(kind: str, model_name: str) -> str:
    short_model = model_name.rsplit("/", 1)[-1]
    return f"{kind}:{short_model}"


def _extract_faster_whisper_frames_from_path(
    path: Path, config: VoiceRecognitionConfig, model_name: str
) -> FloatArray:
    audio = load_audio_mono(path, 16000)
    return _extract_faster_whisper_frames(audio, config, 16000, model_name)


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


def _extract_faster_whisper_nbest_from_path(
    path: Path, config: VoiceRecognitionConfig, model_name: str
) -> NBestFeature:
    audio = load_audio_mono(path, 16000)
    return _extract_faster_whisper_nbest(audio, config, 16000, model_name)


def _extract_faster_whisper_nbest(
    audio: FloatArray,
    config: VoiceRecognitionConfig,
    sample_rate: int,
    model_name: str,
) -> NBestFeature:
    audio = _prepare_audio(audio, config, sample_rate)
    bundle = _load_faster_whisper_model(model_name)
    features = _extract_whisper_features(audio, bundle.model.feature_extractor)
    encoded = bundle.model.encode(features)
    return _generate_nbest_hypotheses(bundle.model, encoded)


def _generate_nbest_hypotheses(model: Any, encoded_features: Any) -> NBestFeature:
    try:
        tokenizer_module = importlib.import_module("faster_whisper.tokenizer")
    except ImportError as exc:
        raise MissingFasterWhisperDependenciesError(
            missing_faster_whisper_dependencies_message()
        ) from exc
    tokenizer_class = getattr(tokenizer_module, "Tokenizer")

    tokenizer = tokenizer_class(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task="transcribe",
        language="en",
    )
    prompt = model.get_prompt(
        tokenizer,
        previous_tokens=[],
        without_timestamps=True,
    )
    result = model.model.generate(
        encoded_features,
        [prompt],
        beam_size=NBEST_BEAM_SIZE,
        patience=1,
        num_hypotheses=NBEST_HYPOTHESES,
        return_scores=True,
        return_no_speech_prob=True,
    )[0]
    sequences = tuple(tuple(int(t) for t in seq) for seq in result.sequences_ids)
    scores = tuple(float(score) for score in result.scores)
    weights = _softmax(scores)
    hypotheses: list[NBestHypothesis] = []
    for tokens, score, weight in zip(sequences, scores, weights, strict=False):
        text = tokenizer.decode(list(tokens))
        hypotheses.append(
            NBestHypothesis(
                text=text.strip(),
                normalized_text=_normalize_hypothesis_text(text),
                tokens=_content_tokens(tokens),
                score=score,
                weight=weight,
            )
        )
    return NBestFeature(tuple(h for h in hypotheses if h.normalized_text or h.tokens))


def _prepare_audio(
    audio: FloatArray, config: VoiceRecognitionConfig, sample_rate: int
) -> FloatArray:
    if audio.size == 0:
        raise ValueError("Empty audio array")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = resample_audio(audio, sample_rate, 16000)
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
        faster_whisper = importlib.import_module("faster_whisper")
    except ImportError as exc:
        raise MissingFasterWhisperDependenciesError(
            missing_faster_whisper_dependencies_message()
        ) from exc
    whisper_model = getattr(faster_whisper, "WhisperModel")

    LOGGER.info("Loading faster-whisper model %s", model_name)
    resolved = _resolve_model_path(model_name)
    model = whisper_model(
        str(resolved.model_path),
        device="cpu",
        compute_type=DEFAULT_FASTER_WHISPER_COMPUTE_TYPE,
        use_auth_token=_huggingface_token(),
        local_files_only=resolved.local_files_only,
    )
    return FasterWhisperBundle(model_name=model_name, model=model)


@dataclass(frozen=True)
class ResolvedModelPath:
    model_path: str | Path
    local_files_only: bool


def _resolve_model_path(model_name: str) -> ResolvedModelPath:
    override = os.environ.get("OSC_GRIMOIRE_MODEL_DIR")
    if override:
        return ResolvedModelPath(Path(override), True)

    for candidate in _bundled_model_candidates():
        if candidate.exists():
            return ResolvedModelPath(candidate, True)

    return ResolvedModelPath(_normalize_model_name(model_name), False)


def _bundled_model_candidates() -> tuple[Path, ...]:
    relative = Path("models") / "faster-whisper-tiny"
    candidates: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / relative)
    executable_dir = Path(sys.executable).resolve().parent
    candidates.append(executable_dir / relative)
    candidates.append(Path.cwd() / "vendor" / relative)
    candidates.append(Path(__file__).resolve().parents[2] / "vendor" / relative)
    return tuple(candidates)


def _normalize_model_name(model_name: str) -> str:
    if model_name.startswith("openai/whisper-"):
        return model_name.removeprefix("openai/whisper-")
    if model_name.startswith("Systran/faster-whisper-"):
        return model_name.removeprefix("Systran/faster-whisper-")
    return model_name


def _huggingface_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    return None


def _whisper_frame_count(audio: FloatArray, total_frames: int) -> int:
    seconds = float(audio.shape[0]) / 16000.0
    frames = int(np.ceil(seconds * total_frames / 30.0))
    return max(1, min(total_frames, frames))


def _l2_normalize(features: FloatArray) -> FloatArray:
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return (features / np.maximum(norm, 1e-9)).astype(np.float32)


def _dtw_distance(a: FloatArray, b: FloatArray) -> float:
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    return float(dtw_ndim.distance(a64, b64))


def _nbest_distance(a: NBestFeature, b: NBestFeature) -> float:
    return 1.0 - _nbest_similarity(a, b)


def _nbest_similarity(a: NBestFeature, b: NBestFeature) -> float:
    if not a.hypotheses or not b.hypotheses:
        return 0.0
    expected = 0.0
    best_weighted = 0.0
    for left in a.hypotheses:
        for right in b.hypotheses:
            similarity = _hypothesis_similarity(left, right)
            expected += left.weight * right.weight * similarity
            best_weighted = max(
                best_weighted,
                similarity * float(np.sqrt(left.weight * right.weight)),
            )
    return max(0.0, min(1.0, 0.35 * expected + 0.65 * best_weighted))


def _hypothesis_similarity(a: NBestHypothesis, b: NBestHypothesis) -> float:
    text_similarity = _text_similarity(a.normalized_text, b.normalized_text)
    token_similarity = _token_similarity(a.tokens, b.tokens)
    return max(
        text_similarity,
        0.75 * text_similarity + 0.25 * token_similarity,
        0.90 * token_similarity,
    )


def _normalize_hypothesis_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _content_tokens(tokens: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(token for token in tokens if token < 50257)


def _softmax(scores: tuple[float, ...]) -> tuple[float, ...]:
    if not scores:
        return ()
    values = np.asarray(scores, dtype=np.float64)
    values = np.exp(values - float(np.max(values)))
    total = float(np.sum(values))
    if total <= 0.0:
        weight = 1.0 / len(scores)
        return tuple(weight for _ in scores)
    return tuple(float(v / total) for v in values)


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    containment = 0.0
    if shorter in longer:
        containment = 0.92 * (len(shorter) / max(len(longer), 1))
    edit = 1.0 - (_levenshtein_distance(a, b) / max(len(a), len(b), 1))
    ngram = _ngram_jaccard(a, b)
    return max(containment, edit, ngram)


def _token_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    lcs = _lcs_length(a, b) / max(len(a), len(b), 1)
    set_union = len(set(a) | set(b))
    jaccard = len(set(a) & set(b)) / set_union if set_union else 0.0
    return max(lcs, jaccard)


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    left = _ngrams(a, n)
    right = _ngrams(b, n)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _ngrams(text: str, n: int) -> set[str]:
    if len(text) <= n:
        return set(text)
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _lcs_length(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]
