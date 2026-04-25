from __future__ import annotations

from pathlib import Path

from .calibration import CalibrationReport


def plot_dependency_message() -> str:
    return (
        "Diagnostic plots require optional ML plotting dependencies. "
        "Run `uv sync --group ml`, then retry."
    )


def roc_points(report: CalibrationReport) -> tuple[list[float], list[float]]:
    fpr: list[float] = []
    tpr: list[float] = []
    for row in report.sweep:
        fpr.append(_rate(row.negative_accepted, row.negative_total))
        tpr.append(_rate(row.positive_correct, row.positive_total))
    return fpr, tpr


def write_diagnostic_plots(
    reports: list[CalibrationReport], output_dir: Path
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(plot_dependency_message()) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _write_roc_plot(plt, reports, output_dir / "roc.png"),
        _write_performance_plot(plt, reports, output_dir / "performance.png"),
    ]
    return paths


def _write_roc_plot(plt, reports: list[CalibrationReport], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for report in reports:
        fpr, tpr = roc_points(report)
        ax.plot(fpr, tpr, marker="o", label=report.backend_name)
        for row, x, y in zip(report.sweep, fpr, tpr, strict=True):
            if report.recommended_margin_min == row.margin_min:
                ax.scatter([x], [y], s=80, marker="*", zorder=3)
                ax.annotate(
                    f"{row.margin_min:.2f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 6),
                )
    ax.set_title("Calibration ROC by Backend")
    ax.set_xlabel("False accept rate")
    ax.set_ylabel("True accept rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_performance_plot(plt, reports: list[CalibrationReport], path: Path) -> Path:
    names = [r.backend_name for r in reports]
    seconds = [r.extraction_seconds for r in reports]
    rss = [r.peak_rss_mb or 0.0 for r in reports]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, seconds)
    axes[0].set_title("Feature Extraction Time")
    axes[0].set_ylabel("seconds")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[1].bar(names, rss)
    axes[1].set_title("Peak Process RSS")
    axes[1].set_ylabel("MiB")
    axes[1].tick_params(axis="x", labelrotation=30)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
