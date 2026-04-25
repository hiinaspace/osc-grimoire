from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .config import VoiceRecognitionConfig
from .spellbook import Spellbook
from .voice_features import FloatArray
from .voice_recognizer import (
    MFCC_DTW_BACKEND,
    BackendStats,
    VoiceTemplateBackend,
    compute_backend_stats,
    decide,
    rank_spells,
)

METADATA_FILENAME = "calibration.json"


@dataclass(frozen=True)
class CalibrationExample:
    path: Path
    kind: str
    expected_spell_id: str | None = None
    expected_spell_name: str | None = None


@dataclass(frozen=True)
class ExampleDiagnosis:
    example: CalibrationExample
    best_spell_id: str | None
    best_spell_name: str | None
    accepted: bool
    correct: bool | None
    intra_ratio: float | None
    margin_ratio: float | None
    reason: str
    extraction_seconds: float = 0.0
    peak_rss_mb: float | None = None


@dataclass(frozen=True)
class ThresholdSweepResult:
    margin_min: float
    positive_total: int
    positive_accepted: int
    positive_correct: int
    positive_wrong: int
    negative_total: int
    negative_accepted: int

    @property
    def false_rejects(self) -> int:
        return self.positive_total - self.positive_accepted


@dataclass(frozen=True)
class CalibrationReport:
    session_dir: Path
    backend_name: str
    examples: tuple[ExampleDiagnosis, ...]
    sweep: tuple[ThresholdSweepResult, ...]
    recommended_margin_min: float | None
    extraction_seconds: float = 0.0
    peak_rss_mb: float | None = None


def write_calibration_metadata(
    session_dir: Path, examples: list[CalibrationExample]
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "examples": [
            {
                "path": e.path.relative_to(session_dir).as_posix(),
                "kind": e.kind,
                "expected_spell_id": e.expected_spell_id,
                "expected_spell_name": e.expected_spell_name,
            }
            for e in examples
        ],
    }
    (session_dir / METADATA_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_calibration_examples(session_dir: Path) -> list[CalibrationExample]:
    path = session_dir / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No {METADATA_FILENAME} found in {session_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for entry in payload.get("examples", ()):
        examples.append(
            CalibrationExample(
                path=session_dir / entry["path"],
                kind=entry["kind"],
                expected_spell_id=entry.get("expected_spell_id"),
                expected_spell_name=entry.get("expected_spell_name"),
            )
        )
    return examples


def latest_calibration_session(data_dir: Path) -> Path | None:
    root = data_dir / "calibration"
    if not root.exists():
        return None
    candidates = [
        p for p in root.iterdir() if p.is_dir() and (p / METADATA_FILENAME).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def diagnose_calibration_session(
    session_dir: Path,
    spellbook: Spellbook,
    config: VoiceRecognitionConfig,
    backend: VoiceTemplateBackend = MFCC_DTW_BACKEND,
    margin_values: tuple[float, ...] = (
        0.00,
        0.03,
        0.05,
        0.07,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
    ),
) -> CalibrationReport:
    examples = load_calibration_examples(session_dir)
    peak_rss_mb = _current_rss_mb()
    backend_stats, feature_cache = compute_backend_stats(spellbook, config, backend)
    peak_rss_mb = _max_optional(peak_rss_mb, _current_rss_mb())
    diagnoses = tuple(
        _diagnose_example(
            example,
            spellbook,
            config,
            backend,
            backend_stats,
            feature_cache,
        )
        for example in examples
    )
    peak_rss_mb = _max_optional(
        peak_rss_mb,
        max(
            (d.peak_rss_mb for d in diagnoses if d.peak_rss_mb is not None),
            default=None,
        ),
    )
    sweep = tuple(
        _sweep_threshold(margin_min, diagnoses, config) for margin_min in margin_values
    )
    recommended = _recommend_margin(sweep)
    return CalibrationReport(
        session_dir=session_dir,
        backend_name=backend.name,
        examples=diagnoses,
        sweep=sweep,
        recommended_margin_min=recommended,
        extraction_seconds=backend_stats.extraction_seconds
        + sum(d.extraction_seconds for d in diagnoses),
        peak_rss_mb=peak_rss_mb,
    )


def _diagnose_example(
    example: CalibrationExample,
    spellbook: Spellbook,
    config: VoiceRecognitionConfig,
    backend: VoiceTemplateBackend,
    backend_stats: BackendStats,
    feature_cache: dict[Path, FloatArray],
) -> ExampleDiagnosis:
    start = time.perf_counter()
    query = backend.extract_path(example.path, config)
    extraction_seconds = time.perf_counter() - start
    ranking = rank_spells(
        query,
        spellbook,
        config,
        feature_cache,
        backend=backend,
        backend_stats=backend_stats,
    )
    decision = decide(ranking, config)
    best = ranking[0] if ranking else None
    correct: bool | None
    if example.kind == "positive":
        correct = best is not None and best.spell_id == example.expected_spell_id
    else:
        correct = None
    return ExampleDiagnosis(
        example=example,
        best_spell_id=best.spell_id if best is not None else None,
        best_spell_name=best.name if best is not None else None,
        accepted=decision.accepted,
        correct=correct,
        intra_ratio=decision.intra_ratio,
        margin_ratio=decision.margin_ratio,
        reason=decision.reason,
        extraction_seconds=extraction_seconds,
        peak_rss_mb=_current_rss_mb(),
    )


def _sweep_threshold(
    margin_min: float,
    diagnoses: tuple[ExampleDiagnosis, ...],
    config: VoiceRecognitionConfig,
) -> ThresholdSweepResult:
    tuned = replace(config, relative_margin_min=margin_min)
    positive_total = 0
    positive_accepted = 0
    positive_correct = 0
    positive_wrong = 0
    negative_total = 0
    negative_accepted = 0
    for diagnosis in diagnoses:
        accepted = _accepted_under(diagnosis, tuned)
        if diagnosis.example.kind == "positive":
            positive_total += 1
            if accepted:
                positive_accepted += 1
                if diagnosis.correct:
                    positive_correct += 1
                else:
                    positive_wrong += 1
        else:
            negative_total += 1
            if accepted:
                negative_accepted += 1
    return ThresholdSweepResult(
        margin_min=margin_min,
        positive_total=positive_total,
        positive_accepted=positive_accepted,
        positive_correct=positive_correct,
        positive_wrong=positive_wrong,
        negative_total=negative_total,
        negative_accepted=negative_accepted,
    )


def _accepted_under(
    diagnosis: ExampleDiagnosis, config: VoiceRecognitionConfig
) -> bool:
    if diagnosis.intra_ratio is None:
        return False
    if diagnosis.intra_ratio > config.intra_class_ratio_max:
        return False
    if (
        diagnosis.margin_ratio is not None
        and diagnosis.margin_ratio < config.relative_margin_min
    ):
        return False
    return True


def _recommend_margin(sweep: tuple[ThresholdSweepResult, ...]) -> float | None:
    viable = [r for r in sweep if r.positive_wrong == 0 and r.negative_accepted == 0]
    if not viable:
        return None
    viable.sort(key=lambda r: (-r.positive_correct, r.false_rejects, -r.margin_min))
    return viable[0].margin_min


def _current_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)
