from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from osc_grimoire.stance_geometry import (
    Pose,
    dual_pose_within_tolerance,
    hmd_relative_pose,
    hmd_yaw_relative_pose,
    matrix_from_quaternion,
    pose_from_matrix34,
    pose_lerp,
    quaternion_distance_rad,
    yaw_reference_from_matrix34,
)


def test_quaternion_distance_uses_shortest_rotation() -> None:
    assert quaternion_distance_rad((1, 0, 0, 0), (-1, 0, 0, 0)) == 0.0
    assert quaternion_distance_rad((1, 0, 0, 0), (0, 0, 0, 1)) == math.pi


def test_pose_from_matrix_extracts_position_and_orientation() -> None:
    matrix = _matrix((1.0, 2.0, 3.0), _rotation_z(math.pi / 2.0))

    pose = pose_from_matrix34(matrix)

    np.testing.assert_allclose(pose.p, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        matrix_from_quaternion(pose.q), _rotation_z(math.pi / 2.0), atol=1e-6
    )


def test_hmd_relative_pose_subtracts_head_frame() -> None:
    hmd = _matrix((1.0, 0.0, 0.0), np.eye(3, dtype=np.float32))
    controller = _matrix((1.25, -0.5, 0.75), np.eye(3, dtype=np.float32))

    pose = hmd_relative_pose(controller, hmd)

    np.testing.assert_allclose(pose.p, (0.25, -0.5, 0.75), atol=1e-6)
    np.testing.assert_allclose(pose.q, (1.0, 0.0, 0.0, 0.0), atol=1e-6)


def test_hmd_yaw_relative_pose_ignores_head_pitch_for_orientation() -> None:
    hmd = _matrix((0.0, 0.0, 0.0), _rotation_x(math.pi / 4.0))
    controller = _matrix((0.0, 0.0, 0.0), np.eye(3, dtype=np.float32))

    pose = hmd_yaw_relative_pose(controller, hmd)

    np.testing.assert_allclose(pose.q, (1.0, 0.0, 0.0, 0.0), atol=1e-6)


def test_hmd_yaw_relative_pose_can_use_locked_yaw() -> None:
    start_hmd = _matrix((0.0, 0.0, 0.0), np.eye(3, dtype=np.float32))
    turned_hmd = _matrix((0.0, 0.0, 0.0), _rotation_y(math.pi / 2.0))
    controller = _matrix((1.0, 0.0, 0.0), np.eye(3, dtype=np.float32))
    locked = yaw_reference_from_matrix34(start_hmd)

    pose = hmd_yaw_relative_pose(controller, turned_hmd, reference_yaw=locked)

    np.testing.assert_allclose(pose.p, (1.0, 0.0, 0.0), atol=1e-6)


def test_pose_lerp_interpolates_position_and_orientation() -> None:
    a = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    b = Pose((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    midpoint = pose_lerp(a, b, 0.5)

    np.testing.assert_allclose(midpoint.p, (1.0, 0.0, 0.0), atol=1e-6)
    assert quaternion_distance_rad(a.q, midpoint.q) == pytest_approx(math.pi / 2.0)


def test_dual_pose_tolerance_checks_both_hands() -> None:
    left = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    right = Pose((0.1, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    target_left = Pose((0.01, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    target_right = Pose((0.11, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    assert dual_pose_within_tolerance(
        left=left,
        right=right,
        target_left=target_left,
        target_right=target_right,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.01,
    )

    assert not dual_pose_within_tolerance(
        left=left,
        right=right,
        target_left=target_left,
        target_right=Pose((0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.01,
    )


def _matrix(translation: tuple[float, float, float], rotation: np.ndarray):
    return SimpleNamespace(
        m=[
            [float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2]), translation[0]],
            [float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2]), translation[1]],
            [float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2]), translation[2]],
        ]
    )


def _rotation_z(radians: float) -> np.ndarray:
    c = math.cos(radians)
    s = math.sin(radians)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rotation_x(radians: float) -> np.ndarray:
    c = math.cos(radians)
    s = math.sin(radians)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rotation_y(radians: float) -> np.ndarray:
    c = math.cos(radians)
    s = math.sin(radians)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
