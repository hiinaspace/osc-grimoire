from __future__ import annotations

from pathlib import Path

from osc_grimoire.calibration import CalibrationReport, ThresholdSweepResult
from osc_grimoire.cli import (
    _calibration_prompt_plan,
    _diagnose_config_for_backend,
    _print_calibration_comparison,
    _resolve_diagnose_backends,
)
from osc_grimoire.config import VoiceRecognitionConfig


def _report(name: str, hits: int, false_accepts: int) -> CalibrationReport:
    return CalibrationReport(
        session_dir=Path("session"),
        backend_name=name,
        examples=(),
        sweep=(
            ThresholdSweepResult(
                margin_min=0.2,
                positive_total=15,
                positive_accepted=hits,
                positive_correct=hits,
                positive_wrong=0,
                negative_total=15,
                negative_accepted=false_accepts,
            ),
        ),
        recommended_margin_min=0.2 if false_accepts == 0 else None,
        extraction_seconds=1.0,
        peak_rss_mb=123.0,
    )


def test_print_calibration_comparison_includes_each_backend(capsys) -> None:
    _print_calibration_comparison(
        [
            _report("faster-whisper-dtw:tiny", hits=8, false_accepts=0),
            _report("faster-whisper-dtw:base", hits=13, false_accepts=0),
        ]
    )

    output = capsys.readouterr().out
    assert "faster-whisper-dtw:tiny" in output
    assert "faster-whisper-dtw:base" in output
    assert "8/15 hits" in output
    assert "13/15 hits" in output
    assert "123 MiB" in output


def test_resolve_faster_whisper_backend_name_without_loading_model() -> None:
    backend = _resolve_diagnose_backends("faster-whisper-dtw", None)[0]
    assert backend.name == "faster-whisper-dtw:tiny"


def test_resolve_faster_whisper_nbest_backend_name_without_loading_model() -> None:
    backend = _resolve_diagnose_backends("faster-whisper-nbest", None)[0]
    assert backend.name == "faster-whisper-nbest:tiny"


def test_resolve_all_faster_whisper_backends_without_loading_model() -> None:
    backends = _resolve_diagnose_backends("all", None)
    assert [backend.name for backend in backends] == [
        "faster-whisper-dtw:tiny",
        "faster-whisper-nbest:tiny",
        "parakeet-ctc-forced:parakeet-ctc-110m-int8",
    ]


def test_resolve_parakeet_ctc_backend_name_without_loading_model() -> None:
    backend = _resolve_diagnose_backends("parakeet-ctc-forced", None)[0]
    assert backend.name == "parakeet-ctc-forced:parakeet-ctc-110m-int8"


def test_standard_calibration_prompt_plan() -> None:
    plan = _calibration_prompt_plan("standard", samples_per_spell=3)
    assert [(p.id, p.count) for p in plan] == [
        ("clean", 5),
        ("quiet", 5),
        ("slow", 5),
        ("fast", 5),
    ]


def test_custom_calibration_prompt_plan() -> None:
    plan = _calibration_prompt_plan("clean=2, loud voice=3", samples_per_spell=5)
    assert [(p.id, p.name, p.count) for p in plan] == [
        ("clean", "clean", 2),
        ("loud_voice", "loud voice", 3),
    ]


def test_faster_whisper_diagnosis_uses_backend_specific_margin() -> None:
    backend = _resolve_diagnose_backends("faster-whisper-dtw", None)[0]
    config = _diagnose_config_for_backend(VoiceRecognitionConfig(), backend)
    assert config.relative_margin_min == 0.15


def test_parakeet_ctc_diagnosis_uses_backend_specific_margin() -> None:
    backend = _resolve_diagnose_backends("parakeet-ctc-forced", None)[0]
    config = _diagnose_config_for_backend(VoiceRecognitionConfig(), backend)
    assert config.relative_margin_min == 0.20
