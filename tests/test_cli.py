from __future__ import annotations

from pathlib import Path

from osc_grimoire.calibration import CalibrationReport, ThresholdSweepResult
from osc_grimoire.cli import _print_calibration_comparison, _resolve_diagnose_backends


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
            _report("mfcc_dtw", hits=8, false_accepts=0),
            _report("wavlm-dtw:wavlm-base-plus", hits=13, false_accepts=0),
        ]
    )

    output = capsys.readouterr().out
    assert "mfcc_dtw" in output
    assert "wavlm-dtw:wavlm-base-plus" in output
    assert "8/15 hits" in output
    assert "13/15 hits" in output
    assert "123 MiB" in output


def test_resolve_conformer_backend_name_without_loading_model() -> None:
    backend = _resolve_diagnose_backends("conformer-dtw", None)[0]
    assert backend.name.startswith("conformer-dtw:")


def test_resolve_w2vbert_backend_name_without_loading_model() -> None:
    backend = _resolve_diagnose_backends("w2vbert-mean", None)[0]
    assert backend.name.startswith("w2vbert-mean:")


def test_resolve_openwakeword_backend_name_without_loading_model() -> None:
    backend = _resolve_diagnose_backends("oww-dtw", None)[0]
    assert backend.name == "oww-dtw:speech_embedding"
