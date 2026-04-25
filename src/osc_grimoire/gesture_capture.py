from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GestureRecognitionConfig
from .gesture_recognizer import PointArray


@dataclass(frozen=True)
class ProjectionPlane:
    right: PointArray
    up: PointArray
    origin: PointArray


class GestureStrokeSampler:
    def __init__(self, config: GestureRecognitionConfig) -> None:
        self.config = config
        self._right: PointArray | None = None
        self._up: PointArray | None = None
        self._origin: PointArray | None = None
        self._points: list[tuple[float, float]] = []

    @property
    def active(self) -> bool:
        return self._right is not None and self._up is not None

    def begin(self, hmd_matrix: Any) -> None:
        rotation = _rotation_from_matrix34(hmd_matrix)
        self._right = _normalize(rotation[:, 0])
        self._up = _normalize(rotation[:, 1])
        self._origin = None
        self._points = []

    def add_controller_pose(self, controller_matrix: Any) -> None:
        if self._right is None or self._up is None:
            return
        position = position_from_matrix34(controller_matrix)
        if self._origin is None:
            self._origin = position
        assert self._origin is not None
        projected = project_position(position, self._origin, self._right, self._up)
        if self._points:
            previous = np.asarray(self._points[-1], dtype=np.float32)
            if np.linalg.norm(projected - previous) < self.config.sample_spacing_m:
                return
        self._points.append((float(projected[0]), float(projected[1])))

    def finish(self) -> PointArray:
        points = np.asarray(self._points, dtype=np.float32).reshape(-1, 2)
        self.cancel()
        return points

    def cancel(self) -> None:
        self._right = None
        self._up = None
        self._origin = None
        self._points = []


def project_position(
    position: PointArray, origin: PointArray, right: PointArray, up: PointArray
) -> PointArray:
    delta = np.asarray(position, dtype=np.float32) - np.asarray(
        origin, dtype=np.float32
    )
    return np.asarray([float(delta @ right), float(delta @ up)], dtype=np.float32)


def position_from_matrix34(matrix: Any) -> PointArray:
    return np.asarray([matrix.m[row][3] for row in range(3)], dtype=np.float32)


def _rotation_from_matrix34(matrix: Any) -> PointArray:
    return np.asarray(
        [[matrix.m[row][col] for col in range(3)] for row in range(3)],
        dtype=np.float32,
    )


def _normalize(vector: PointArray) -> PointArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
