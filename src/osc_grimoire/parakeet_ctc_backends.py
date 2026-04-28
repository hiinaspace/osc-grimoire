from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import VoiceRecognitionConfig
from .voice_features import (
    FloatArray,
    load_audio_mono,
    resample_audio,
    trim_voice_audio,
)
from .voice_recognizer import VoiceTemplateBackend

LOGGER = logging.getLogger(__name__)

DEFAULT_PARAKEET_CTC_REPO = "entropora/parakeet-ctc-110m-int8"
DEFAULT_PARAKEET_CTC_MODEL_DIR = "parakeet-ctc-110m-int8"


class MissingParakeetCtcDependenciesError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParakeetCtcBundle:
    repo_id: str
    model_dir: Path
    asr: Any
    blank_id: int


@dataclass(frozen=True)
class ResolvedParakeetModel:
    model_dir: Path | None
    repo_id: str


@dataclass(frozen=True)
class CtcFeature:
    log_probs: FloatArray
    token_ids: tuple[int, ...]


def missing_parakeet_ctc_dependencies_message() -> str:
    return (
        "Parakeet CTC diagnostic backends require `onnx-asr` and "
        "`huggingface-hub`. Run `uv sync --group research`, then retry."
    )


def parakeet_ctc_forced_backend(
    repo_id: str = DEFAULT_PARAKEET_CTC_REPO,
) -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name=f"parakeet-ctc-forced:{repo_id.rsplit('/', 1)[-1]}",
        extract_path=lambda path, config: _extract_parakeet_ctc_from_path(
            path, config, repo_id
        ),
        extract_array=lambda audio, config, sample_rate: _extract_parakeet_ctc(
            audio, config, sample_rate, repo_id
        ),
        distance=_ctc_forced_distance,
        aggregate=lambda distances: float(np.median(distances)),
    )


def parakeet_ctc_token_labels(
    repo_id: str = DEFAULT_PARAKEET_CTC_REPO,
) -> dict[int, str]:
    bundle = _load_parakeet_ctc_model(repo_id)
    labels: dict[int, str] = {}
    for line in (
        (bundle.model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    ):
        parts = line.strip().split()
        if len(parts) >= 2:
            labels[int(parts[-1])] = parts[0].replace("\u2581", " ")
    return labels


def format_ctc_feature_distribution(
    feature: CtcFeature,
    token_labels: dict[int, str],
    *,
    title: str,
    top_k: int = 5,
    max_frames: int = 24,
) -> list[str]:
    lines = [title]
    collapsed = " ".join(
        _format_token(token_id, token_labels) for token_id in feature.token_ids
    )
    lines.append(f"  greedy tokens: {collapsed or '(empty)'}")
    frame_count = min(feature.log_probs.shape[0], max_frames)
    for frame in range(frame_count):
        probs = np.exp(feature.log_probs[frame])
        top_ids = np.argsort(probs)[-top_k:][::-1]
        parts = [
            f"{_format_token(int(token_id), token_labels)}={float(probs[token_id]):.2f}"
            for token_id in top_ids
        ]
        lines.append(f"  frame {frame:02d}: " + ", ".join(parts))
    if feature.log_probs.shape[0] > max_frames:
        lines.append(f"  ... {feature.log_probs.shape[0] - max_frames} more frames")
    return lines


def _extract_parakeet_ctc_from_path(
    path: Path, config: VoiceRecognitionConfig, repo_id: str
) -> CtcFeature:
    audio = load_audio_mono(path, 16000)
    return _extract_parakeet_ctc(audio, config, 16000, repo_id)


def _extract_parakeet_ctc(
    audio: FloatArray,
    config: VoiceRecognitionConfig,
    sample_rate: int,
    repo_id: str,
) -> CtcFeature:
    audio = _prepare_audio(audio, config, sample_rate)
    bundle = _load_parakeet_ctc_model(repo_id)
    waveforms = audio[None, :].astype(np.float32)
    lengths = np.asarray([audio.shape[0]], dtype=np.int64)
    features, feature_lengths = bundle.asr._preprocessor(waveforms, lengths)
    log_probs, output_lengths = bundle.asr._encode(features, feature_lengths)
    valid_log_probs = np.asarray(
        log_probs[0, : int(output_lengths[0])], dtype=np.float32
    )
    token_ids = ctc_greedy_token_ids(valid_log_probs, bundle.blank_id)
    return CtcFeature(valid_log_probs, token_ids)


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


@functools.lru_cache(maxsize=2)
def _load_parakeet_ctc_model(repo_id: str) -> ParakeetCtcBundle:
    try:
        import onnx_asr
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MissingParakeetCtcDependenciesError(
            missing_parakeet_ctc_dependencies_message()
        ) from exc

    LOGGER.info("Loading Parakeet CTC model %s", repo_id)
    resolved = _resolve_parakeet_model(repo_id)
    if resolved.model_dir is not None:
        model_dir = resolved.model_dir
    else:
        model_path = Path(hf_hub_download(repo_id, "encoder.int8.onnx")).resolve()
        tokens_path = Path(hf_hub_download(repo_id, "tokens.txt")).resolve()
        model_dir = _materialize_onnx_asr_model_dir(model_path, tokens_path)
    adapter = onnx_asr.load_model(
        "nemo-conformer-ctc",
        path=model_dir,
        providers=["CPUExecutionProvider"],
        preprocessor_config={
            "use_numpy_preprocessors": True,
            "max_concurrent_workers": 1,
        },
    )
    blank_id = _read_blank_id(model_dir / "vocab.txt")
    return ParakeetCtcBundle(
        repo_id=repo_id,
        model_dir=model_dir,
        asr=adapter.asr,
        blank_id=blank_id,
    )


def _resolve_parakeet_model(repo_id: str) -> ResolvedParakeetModel:
    override = os.environ.get("OSC_GRIMOIRE_PARAKEET_CTC_MODEL_DIR")
    if override:
        return ResolvedParakeetModel(Path(override), repo_id)
    for candidate in _bundled_parakeet_model_candidates():
        if (candidate / "model.onnx").exists() and (candidate / "vocab.txt").exists():
            return ResolvedParakeetModel(candidate, repo_id)
    return ResolvedParakeetModel(None, repo_id)


def _bundled_parakeet_model_candidates() -> tuple[Path, ...]:
    relative = Path("models") / DEFAULT_PARAKEET_CTC_MODEL_DIR
    candidates: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / relative)
    executable_dir = Path(sys.executable).resolve().parent
    candidates.append(executable_dir / relative)
    candidates.append(Path.cwd() / "vendor" / relative)
    candidates.append(Path(__file__).resolve().parents[2] / "vendor" / relative)
    return tuple(candidates)


def _materialize_onnx_asr_model_dir(model_path: Path, tokens_path: Path) -> Path:
    model_dir = Path(tempfile.mkdtemp(prefix="osc_grimoire_parakeet_ctc_"))
    shutil.copyfile(model_path, model_dir / "model.onnx")
    shutil.copyfile(tokens_path, model_dir / "vocab.txt")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemo-conformer-ctc",
                "subsampling_factor": 8,
                "features_size": 80,
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def _read_blank_id(tokens_path: Path) -> int:
    blank_id: int | None = None
    max_id = -1
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        token_id = int(parts[-1])
        max_id = max(max_id, token_id)
        if parts[0] in {"<blk>", "<blank>", "<eps>"}:
            blank_id = token_id
    if blank_id is not None:
        return blank_id
    return max_id


def _format_token(token_id: int, token_labels: dict[int, str]) -> str:
    token = token_labels.get(token_id, f"<{token_id}>")
    if token == "":
        token = "∅"
    return f"{token_id}:{token!r}"


def ctc_greedy_token_ids(log_probs: FloatArray, blank_id: int) -> tuple[int, ...]:
    tokens: list[int] = []
    previous: int | None = None
    for token_id in np.argmax(log_probs, axis=-1).astype(np.int64).tolist():
        if token_id != blank_id and token_id != previous:
            tokens.append(int(token_id))
        previous = int(token_id)
    return tuple(tokens)


def _ctc_forced_distance(query: CtcFeature, template: CtcFeature) -> float:
    if not template.token_ids:
        return 1.0
    score = ctc_sequence_log_probability(query.log_probs, template.token_ids)
    normalized = score / max(len(template.token_ids), 1)
    return float(-normalized)


def ctc_sequence_log_probability(
    log_probs: FloatArray,
    token_ids: tuple[int, ...],
    *,
    blank_id: int | None = None,
) -> float:
    if log_probs.ndim != 2:
        raise ValueError("CTC log probabilities must be a 2-D array")
    if not token_ids:
        return 0.0
    if blank_id is None:
        blank_id = log_probs.shape[1] - 1
    extended = _ctc_extended_labels(token_ids, blank_id)
    previous = np.full(len(extended), -np.inf, dtype=np.float64)
    previous[0] = float(log_probs[0, blank_id])
    if len(extended) > 1:
        previous[1] = float(log_probs[0, extended[1]])

    for frame in range(1, log_probs.shape[0]):
        current = np.full(len(extended), -np.inf, dtype=np.float64)
        for state, label in enumerate(extended):
            candidates = [previous[state]]
            if state > 0:
                candidates.append(previous[state - 1])
            if state > 1 and label != blank_id and label != extended[state - 2]:
                candidates.append(previous[state - 2])
            current[state] = _logsumexp(candidates) + float(log_probs[frame, label])
        previous = current

    return _logsumexp([previous[-1], previous[-2]])


def _ctc_extended_labels(token_ids: tuple[int, ...], blank_id: int) -> tuple[int, ...]:
    labels: list[int] = [blank_id]
    for token_id in token_ids:
        labels.append(token_id)
        labels.append(blank_id)
    return tuple(labels)


def _logsumexp(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return -np.inf
    maximum = max(finite)
    return float(maximum + np.log(sum(np.exp(v - maximum) for v in finite)))
