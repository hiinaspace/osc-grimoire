from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from osc_grimoire.stance_capture import (
    StanceFrame,
    StanceSample,
    StanceSampler,
    interpolate_stance_frame,
    load_stance_sample,
    save_stance_sample,
)
from osc_grimoire.stance_geometry import Pose


def test_stance_sampler_records_head_yaw_relative_frames_and_finishes() -> None:
    sampler = StanceSampler()
    hmd = _matrix((1.0, 0.0, 0.0))

    sampler.begin(
        now=10.0,
        hmd_matrix=hmd,
        left_matrix=_matrix((1.1, 0.0, 0.0)),
        right_matrix=_matrix((1.2, 0.0, 0.0)),
    )
    sampler.add_frame(
        now=10.25,
        hmd_matrix=hmd,
        left_matrix=_matrix((1.2, 0.0, 0.0)),
        right_matrix=_matrix((1.3, 0.0, 0.0)),
    )
    sample = sampler.finish(
        now=10.5,
        hmd_matrix=hmd,
        left_matrix=_matrix((1.4, 0.0, 0.0)),
        right_matrix=_matrix((1.5, 0.0, 0.0)),
    )

    assert not sampler.active
    assert [frame.t for frame in sample.frames] == [0.0, 0.25, 0.5]
    np.testing.assert_allclose(sample.frames[0].left.p, (0.1, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(sample.frames[-1].right.p, (0.5, 0.0, 0.0), atol=1e-6)


def test_stance_sampler_locks_yaw_from_start_pose() -> None:
    sampler = StanceSampler()
    start_hmd = _matrix((1.0, 0.0, 0.0))
    turned_hmd = _matrix((1.0, 0.0, 0.0), _rotation_y(math.pi / 2.0))

    sampler.begin(
        now=10.0,
        hmd_matrix=start_hmd,
        left_matrix=_matrix((1.1, 0.0, 0.0)),
        right_matrix=_matrix((1.2, 0.0, 0.0)),
    )
    sample = sampler.finish(
        now=10.5,
        hmd_matrix=turned_hmd,
        left_matrix=_matrix((1.4, 0.0, 0.0)),
        right_matrix=_matrix((1.5, 0.0, 0.0)),
    )

    np.testing.assert_allclose(sample.frames[-1].left.p, (0.4, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(sample.frames[-1].right.p, (0.5, 0.0, 0.0), atol=1e-6)


def test_stance_sample_json_round_trip(tmp_path) -> None:
    sample = StanceSample(
        frames=(
            StanceFrame(
                0.0,
                Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                Pose((0.2, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ),
            StanceFrame(
                0.5,
                Pose((0.1, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                Pose((0.3, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ),
        )
    )
    path = tmp_path / "stance.json"

    save_stance_sample(path, sample)
    loaded = load_stance_sample(path)

    assert loaded == sample


def test_interpolate_stance_frame_blends_between_samples() -> None:
    sample = StanceSample(
        frames=(
            StanceFrame(
                0.0,
                Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                Pose((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ),
            StanceFrame(
                1.0,
                Pose((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                Pose((2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ),
        )
    )

    frame = interpolate_stance_frame(sample, 0.5)

    assert frame is not None
    np.testing.assert_allclose(frame.left.p, (0.5, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(frame.right.p, (1.5, 0.0, 0.0), atol=1e-6)


def _matrix(translation: tuple[float, float, float], rotation: np.ndarray | None = None):
    rotation = np.eye(3, dtype=np.float32) if rotation is None else rotation
    return SimpleNamespace(
        m=[
            [float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2]), translation[0]],
            [float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2]), translation[1]],
            [float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2]), translation[2]],
        ]
    )


def _rotation_y(radians: float) -> np.ndarray:
    c = math.cos(radians)
    s = math.sin(radians)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
