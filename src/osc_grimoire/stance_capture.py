from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stance_geometry import (
    Pose,
    hmd_yaw_relative_pose,
    pose_lerp,
    yaw_reference_from_matrix34,
)

STANCE_SAMPLE_VERSION = 1
STANCE_REFERENCE_FRAME = "head_yaw_relative"


@dataclass(frozen=True)
class StanceFrame:
    t: float
    left: Pose
    right: Pose


@dataclass(frozen=True)
class StanceSample:
    frames: tuple[StanceFrame, ...]
    reference_frame: str = STANCE_REFERENCE_FRAME

    @property
    def duration_s(self) -> float:
        if not self.frames:
            return 0.0
        return max(0.0, float(self.frames[-1].t - self.frames[0].t))

    @property
    def start(self) -> StanceFrame | None:
        return self.frames[0] if self.frames else None

    @property
    def end(self) -> StanceFrame | None:
        return self.frames[-1] if self.frames else None


class StanceSampler:
    def __init__(self) -> None:
        self._started_at: float | None = None
        self._reference_yaw: tuple[float, float, float, float] | None = None
        self._frames: list[StanceFrame] = []

    @property
    def active(self) -> bool:
        return self._started_at is not None

    @property
    def sample(self) -> StanceSample:
        return StanceSample(frames=tuple(self._frames))

    def begin(
        self,
        *,
        now: float,
        hmd_matrix: Any,
        left_matrix: Any,
        right_matrix: Any,
    ) -> None:
        self._started_at = float(now)
        self._reference_yaw = yaw_reference_from_matrix34(hmd_matrix)
        self._frames = []
        self.add_frame(
            now=now,
            hmd_matrix=hmd_matrix,
            left_matrix=left_matrix,
            right_matrix=right_matrix,
        )

    def add_frame(
        self,
        *,
        now: float,
        hmd_matrix: Any,
        left_matrix: Any,
        right_matrix: Any,
    ) -> None:
        if self._started_at is None:
            return
        t = max(0.0, float(now) - self._started_at)
        self._frames.append(
            StanceFrame(
                t=t,
                left=hmd_yaw_relative_pose(
                    left_matrix,
                    hmd_matrix,
                    reference_yaw=self._reference_yaw,
                ),
                right=hmd_yaw_relative_pose(
                    right_matrix,
                    hmd_matrix,
                    reference_yaw=self._reference_yaw,
                ),
            )
        )

    def finish(
        self,
        *,
        now: float,
        hmd_matrix: Any,
        left_matrix: Any,
        right_matrix: Any,
    ) -> StanceSample:
        self.add_frame(
            now=now,
            hmd_matrix=hmd_matrix,
            left_matrix=left_matrix,
            right_matrix=right_matrix,
        )
        sample = self.sample
        self.cancel()
        return sample

    def cancel(self) -> None:
        self._started_at = None
        self._reference_yaw = None
        self._frames = []


def interpolate_stance_frame(sample: StanceSample, t: float) -> StanceFrame | None:
    if not sample.frames:
        return None
    if len(sample.frames) == 1:
        return sample.frames[0]
    if t <= sample.frames[0].t:
        return sample.frames[0]
    if t >= sample.frames[-1].t:
        return sample.frames[-1]
    for previous, current in zip(sample.frames[:-1], sample.frames[1:], strict=False):
        if previous.t <= t <= current.t:
            span = max(1e-6, current.t - previous.t)
            ratio = (t - previous.t) / span
            return StanceFrame(
                t=t,
                left=pose_lerp(previous.left, current.left, ratio),
                right=pose_lerp(previous.right, current.right, ratio),
            )
    return sample.frames[-1]


def looping_stance_frame(sample: StanceSample, now: float) -> StanceFrame | None:
    if sample.duration_s <= 0.0:
        return sample.start
    return interpolate_stance_frame(sample, float(now) % sample.duration_s)


def save_stance_sample(path: Path, sample: StanceSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STANCE_SAMPLE_VERSION,
        "reference_frame": sample.reference_frame,
        "frames": [_frame_to_json(frame) for frame in sample.frames],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stance_sample(path: Path) -> StanceSample:
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version")
    if version != STANCE_SAMPLE_VERSION:
        raise ValueError(
            f"Unsupported stance sample version {version!r} "
            f"(expected {STANCE_SAMPLE_VERSION})"
        )
    reference_frame = str(raw.get("reference_frame") or STANCE_REFERENCE_FRAME)
    frames = tuple(_frame_from_json(entry) for entry in raw.get("frames", ()))
    return StanceSample(frames=frames, reference_frame=reference_frame)


def _frame_to_json(frame: StanceFrame) -> dict:
    return {
        "t": float(frame.t),
        "left": _pose_to_json(frame.left),
        "right": _pose_to_json(frame.right),
    }


def _frame_from_json(entry: dict) -> StanceFrame:
    return StanceFrame(
        t=float(entry["t"]),
        left=_pose_from_json(entry["left"]),
        right=_pose_from_json(entry["right"]),
    )


def _pose_to_json(pose: Pose) -> dict:
    return {
        "p": [float(v) for v in pose.p],
        "q": [float(v) for v in pose.q],
    }


def _pose_from_json(entry: dict) -> Pose:
    p = tuple(float(v) for v in entry["p"])
    q = tuple(float(v) for v in entry["q"])
    if len(p) != 3 or len(q) != 4:
        raise ValueError("Stance pose must contain p[3] and q[4]")
    return Pose(p=p, q=q)
