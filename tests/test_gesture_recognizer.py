from __future__ import annotations

import numpy as np

from osc_grimoire.config import GestureRecognitionConfig
from osc_grimoire.gesture_recognizer import (
    GestureTemplate,
    load_gesture_points,
    normalize_points,
    recognize_gesture,
    save_gesture_points,
)


def test_normalize_points_resamples_and_centers() -> None:
    config = GestureRecognitionConfig(point_count=16)
    points = np.asarray([[0, 0], [2, 0], [2, 2]], dtype=np.float32)

    normalized = normalize_points(points, config)

    assert normalized.shape == (16, 2)
    np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=1e-6)


def test_recognizer_matches_correct_template() -> None:
    config = GestureRecognitionConfig(score_min=0.5, margin_min=0.01)
    line = _line()
    zigzag = _zigzag()
    templates = (
        GestureTemplate("line", "Line", normalize_points(line, config)),
        GestureTemplate("zigzag", "Zigzag", normalize_points(zigzag, config)),
    )

    result = recognize_gesture(
        line + np.asarray([0.03, -0.02], dtype=np.float32), templates, config
    )

    assert result.decision.accepted
    assert result.ranking[0].spell_id == "line"


def test_recognizer_matches_closed_shape_with_different_start_point() -> None:
    config = GestureRecognitionConfig(score_min=0.5, margin_min=0.0)
    square = _square()
    shifted_square = np.roll(square, 8, axis=0)
    templates = (GestureTemplate("square", "Square", normalize_points(square, config)),)

    result = recognize_gesture(shifted_square, templates, config)

    assert result.decision.accepted
    assert result.ranking[0].spell_id == "square"


def test_recognizer_rejects_low_score() -> None:
    config = GestureRecognitionConfig(score_min=0.99, margin_min=0.0)
    templates = (GestureTemplate("line", "Line", normalize_points(_line(), config)),)

    result = recognize_gesture(_zigzag(), templates, config)

    assert not result.decision.accepted
    assert "score" in result.decision.reason


def test_recognizer_rejects_low_margin() -> None:
    config = GestureRecognitionConfig(score_min=0.0, margin_min=0.5)
    line = _line()
    templates = (
        GestureTemplate("a", "A", normalize_points(line, config)),
        GestureTemplate("b", "B", normalize_points(line * 1.1, config)),
    )

    result = recognize_gesture(line, templates, config)

    assert not result.decision.accepted
    assert "margin" in result.decision.reason


def test_gesture_points_round_trip(tmp_path) -> None:
    path = tmp_path / "gesture.json"
    points = _zigzag()

    save_gesture_points(path, points)
    loaded = load_gesture_points(path)

    np.testing.assert_allclose(loaded, points)


def _line() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    return np.column_stack([x, np.zeros_like(x)]).astype(np.float32)


def _zigzag() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    y = np.where(np.arange(24) % 2 == 0, 0.0, 0.5).astype(np.float32)
    return np.column_stack([x, y]).astype(np.float32)


def _square() -> np.ndarray:
    side = 8
    top = np.column_stack(
        [np.linspace(0.0, 1.0, side, dtype=np.float32), np.zeros(side)]
    )
    right = np.column_stack(
        [np.ones(side), np.linspace(0.0, 1.0, side, dtype=np.float32)]
    )
    bottom = np.column_stack(
        [np.linspace(1.0, 0.0, side, dtype=np.float32), np.ones(side)]
    )
    left = np.column_stack(
        [np.zeros(side), np.linspace(1.0, 0.0, side, dtype=np.float32)]
    )
    return np.vstack([top, right, bottom, left]).astype(np.float32)
