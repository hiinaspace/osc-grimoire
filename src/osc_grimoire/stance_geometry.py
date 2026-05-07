from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Pose:
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]


@dataclass(frozen=True)
class PoseDistance:
    position_m: float
    orientation_rad: float


@dataclass(frozen=True)
class DualPoseDistance:
    left: PoseDistance
    right: PoseDistance

    @property
    def score(self) -> float:
        return (
            self.left.position_m
            + self.right.position_m
            + self.left.orientation_rad
            + self.right.orientation_rad
        )


def pose_from_matrix34(matrix: Any) -> Pose:
    rotation = rotation_from_matrix34(matrix)
    return Pose(
        p=tuple(float(v) for v in position_from_matrix34(matrix)),
        q=quaternion_from_matrix(rotation),
    )


def hmd_relative_pose(controller_matrix: Any, hmd_matrix: Any) -> Pose:
    controller = pose_from_matrix34(controller_matrix)
    hmd = pose_from_matrix34(hmd_matrix)
    return relative_pose(controller, hmd)


def yaw_reference_from_matrix34(
    hmd_matrix: Any,
) -> tuple[float, float, float, float]:
    return quaternion_from_matrix(yaw_rotation_from_matrix34(hmd_matrix))


def hmd_yaw_relative_pose(
    controller_matrix: Any,
    hmd_matrix: Any,
    *,
    reference_yaw: tuple[float, float, float, float] | None = None,
) -> Pose:
    controller = pose_from_matrix34(controller_matrix)
    hmd_position = position_from_matrix34(hmd_matrix)
    reference_rotation = (
        matrix_from_quaternion(reference_yaw)
        if reference_yaw is not None
        else yaw_rotation_from_matrix34(hmd_matrix)
    )
    return relative_pose_to_frame(controller, hmd_position, reference_rotation)


def relative_pose(pose: Pose, origin: Pose) -> Pose:
    origin_rotation = matrix_from_quaternion(origin.q)
    return relative_pose_to_frame(
        pose, np.asarray(origin.p, dtype=np.float32), origin_rotation
    )


def relative_pose_to_frame(
    pose: Pose,
    origin_position: FloatArray,
    origin_rotation: FloatArray,
) -> Pose:
    pose_rotation = matrix_from_quaternion(pose.q)
    delta = np.asarray(pose.p, dtype=np.float32) - np.asarray(
        origin_position, dtype=np.float32
    )
    relative_position = origin_rotation.T @ delta
    relative_rotation = origin_rotation.T @ pose_rotation
    return Pose(
        p=tuple(float(v) for v in relative_position),
        q=quaternion_from_matrix(relative_rotation),
    )


def absolute_pose_from_hmd_relative(relative: Pose, hmd_matrix: Any) -> Pose:
    hmd = pose_from_matrix34(hmd_matrix)
    hmd_rotation = matrix_from_quaternion(hmd.q)
    return absolute_pose_from_frame(
        relative, np.asarray(hmd.p, dtype=np.float32), hmd_rotation
    )


def absolute_pose_from_hmd_yaw_relative(
    relative: Pose,
    hmd_matrix: Any,
    *,
    reference_yaw: tuple[float, float, float, float] | None = None,
) -> Pose:
    hmd_position = position_from_matrix34(hmd_matrix)
    reference_rotation = (
        matrix_from_quaternion(reference_yaw)
        if reference_yaw is not None
        else yaw_rotation_from_matrix34(hmd_matrix)
    )
    return absolute_pose_from_frame(relative, hmd_position, reference_rotation)


def absolute_pose_from_frame(
    relative: Pose,
    origin_position: FloatArray,
    origin_rotation: FloatArray,
) -> Pose:
    relative_rotation = matrix_from_quaternion(relative.q)
    absolute_position = origin_rotation @ np.asarray(relative.p, dtype=np.float32)
    absolute_position = absolute_position + np.asarray(origin_position, dtype=np.float32)
    absolute_rotation = origin_rotation @ relative_rotation
    return Pose(
        p=tuple(float(v) for v in absolute_position),
        q=quaternion_from_matrix(absolute_rotation),
    )


def matrix34_from_pose(openvr: Any, pose: Pose) -> Any:
    rotation = matrix_from_quaternion(pose.q)
    matrix = openvr.HmdMatrix34_t()
    for row in range(3):
        for col in range(3):
            matrix.m[row][col] = float(rotation[row, col])
        matrix.m[row][3] = float(pose.p[row])
    return matrix


def position_from_matrix34(matrix: Any) -> FloatArray:
    return np.asarray([matrix.m[row][3] for row in range(3)], dtype=np.float32)


def rotation_from_matrix34(matrix: Any) -> FloatArray:
    return np.asarray(
        [[matrix.m[row][col] for col in range(3)] for row in range(3)],
        dtype=np.float32,
    )


def yaw_rotation_from_matrix34(matrix: Any) -> FloatArray:
    rotation = rotation_from_matrix34(matrix)
    forward = -rotation[:, 2]
    forward[1] = 0.0
    if float(np.linalg.norm(forward)) <= 1e-6:
        right = rotation[:, 0].copy()
        right[1] = 0.0
        if float(np.linalg.norm(right)) <= 1e-6:
            return np.eye(3, dtype=np.float32)
        right = right / np.linalg.norm(right)
        back = np.cross(right, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        back = back / np.linalg.norm(back)
        return np.column_stack(
            [right, np.asarray([0.0, 1.0, 0.0], dtype=np.float32), back]
        ).astype(np.float32)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    back = -forward
    return np.column_stack([right, up, back]).astype(np.float32)


def quaternion_normalize(
    q: tuple[float, float, float, float] | FloatArray,
) -> tuple[float, float, float, float]:
    array = np.asarray(q, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    array = array / norm
    return tuple(float(v) for v in array)


def quaternion_conjugate(
    q: tuple[float, float, float, float] | FloatArray,
) -> tuple[float, float, float, float]:
    w, x, y, z = quaternion_normalize(q)
    return (w, -x, -y, -z)


def quaternion_multiply(
    a: tuple[float, float, float, float] | FloatArray,
    b: tuple[float, float, float, float] | FloatArray,
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = quaternion_normalize(a)
    bw, bx, by, bz = quaternion_normalize(b)
    return quaternion_normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quaternion_distance_rad(
    a: tuple[float, float, float, float] | FloatArray,
    b: tuple[float, float, float, float] | FloatArray,
) -> float:
    qa = np.asarray(quaternion_normalize(a), dtype=np.float32)
    qb = np.asarray(quaternion_normalize(b), dtype=np.float32)
    dot = min(1.0, max(-1.0, abs(float(qa @ qb))))
    return float(2.0 * math.acos(dot))


def quaternion_slerp(
    a: tuple[float, float, float, float] | FloatArray,
    b: tuple[float, float, float, float] | FloatArray,
    t: float,
) -> tuple[float, float, float, float]:
    qa = np.asarray(quaternion_normalize(a), dtype=np.float32)
    qb = np.asarray(quaternion_normalize(b), dtype=np.float32)
    dot = float(qa @ qb)
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        return quaternion_normalize(qa + float(t) * (qb - qa))
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    theta = theta_0 * float(t)
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quaternion_normalize((s0 * qa) + (s1 * qb))


def quaternion_from_matrix(matrix: FloatArray) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (float(m[2, 1]) - float(m[1, 2])) / s
        y = (float(m[0, 2]) - float(m[2, 0])) / s
        z = (float(m[1, 0]) - float(m[0, 1])) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2])) * 2.0
        w = (float(m[2, 1]) - float(m[1, 2])) / s
        x = 0.25 * s
        y = (float(m[0, 1]) + float(m[1, 0])) / s
        z = (float(m[0, 2]) + float(m[2, 0])) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2])) * 2.0
        w = (float(m[0, 2]) - float(m[2, 0])) / s
        x = (float(m[0, 1]) + float(m[1, 0])) / s
        y = 0.25 * s
        z = (float(m[1, 2]) + float(m[2, 1])) / s
    else:
        s = math.sqrt(1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1])) * 2.0
        w = (float(m[1, 0]) - float(m[0, 1])) / s
        x = (float(m[0, 2]) + float(m[2, 0])) / s
        y = (float(m[1, 2]) + float(m[2, 1])) / s
        z = 0.25 * s
    return quaternion_normalize((w, x, y, z))


def matrix_from_quaternion(
    q: tuple[float, float, float, float] | FloatArray,
) -> FloatArray:
    w, x, y, z = quaternion_normalize(q)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


def pose_lerp(a: Pose, b: Pose, t: float) -> Pose:
    pa = np.asarray(a.p, dtype=np.float32)
    pb = np.asarray(b.p, dtype=np.float32)
    p = pa + float(t) * (pb - pa)
    return Pose(p=tuple(float(v) for v in p), q=quaternion_slerp(a.q, b.q, t))


def pose_distance(a: Pose, b: Pose) -> PoseDistance:
    position = float(
        np.linalg.norm(np.asarray(a.p, dtype=np.float32) - np.asarray(b.p, dtype=np.float32))
    )
    orientation = quaternion_distance_rad(a.q, b.q)
    return PoseDistance(position_m=position, orientation_rad=orientation)


def dual_pose_distance(
    *,
    left: Pose,
    right: Pose,
    target_left: Pose,
    target_right: Pose,
) -> DualPoseDistance:
    return DualPoseDistance(
        left=pose_distance(left, target_left),
        right=pose_distance(right, target_right),
    )


def dual_pose_within_tolerance(
    *,
    left: Pose,
    right: Pose,
    target_left: Pose,
    target_right: Pose,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> bool:
    distances = dual_pose_distance(
        left=left,
        right=right,
        target_left=target_left,
        target_right=target_right,
    )
    return (
        distances.left.position_m <= position_tolerance_m
        and distances.right.position_m <= position_tolerance_m
        and distances.left.orientation_rad <= orientation_tolerance_rad
        and distances.right.orientation_rad <= orientation_tolerance_rad
    )
