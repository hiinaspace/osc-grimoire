from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osc_grimoire.config import GestureRecognitionConfig
from osc_grimoire.gesture_capture import GestureStrokeSampler, project_position


def test_project_position_uses_hmd_right_and_up_axes() -> None:
    origin = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    position = np.asarray([1.25, 1.50, 4.0], dtype=np.float32)

    projected = project_position(position, origin, right, up)

    np.testing.assert_allclose(projected, [0.25, -0.50])


def test_stroke_sampler_filters_by_spacing_and_finishes() -> None:
    sampler = GestureStrokeSampler(GestureRecognitionConfig(sample_spacing_m=0.1))
    sampler.begin(_matrix((0, 0, 0)))

    sampler.add_controller_pose(_matrix((0.0, 0.0, 0.0)))
    sampler.add_controller_pose(_matrix((0.05, 0.0, 0.0)))
    sampler.add_controller_pose(_matrix((0.20, 0.0, 0.0)))
    points = sampler.finish()

    assert points.shape == (2, 2)
    np.testing.assert_allclose(points, [[0.0, 0.0], [0.20, 0.0]])
    assert not sampler.active


def _matrix(translation: tuple[float, float, float]):
    return SimpleNamespace(
        m=[
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
        ]
    )
