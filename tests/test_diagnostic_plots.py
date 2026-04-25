from __future__ import annotations

from pathlib import Path

from osc_grimoire.calibration import CalibrationReport, ThresholdSweepResult
from osc_grimoire.diagnostic_plots import roc_points


def test_roc_points_uses_false_accept_and_true_accept_rates() -> None:
    report = CalibrationReport(
        session_dir=Path("session"),
        backend_name="backend",
        examples=(),
        sweep=(
            ThresholdSweepResult(
                margin_min=0.0,
                positive_total=10,
                positive_accepted=10,
                positive_correct=8,
                positive_wrong=2,
                negative_total=5,
                negative_accepted=4,
            ),
            ThresholdSweepResult(
                margin_min=0.5,
                positive_total=10,
                positive_accepted=6,
                positive_correct=6,
                positive_wrong=0,
                negative_total=5,
                negative_accepted=1,
            ),
        ),
        recommended_margin_min=0.5,
    )

    fpr, tpr = roc_points(report)

    assert fpr == [0.8, 0.2]
    assert tpr == [0.8, 0.6]
