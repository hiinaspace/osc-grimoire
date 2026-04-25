from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import GestureRecognitionConfig
from .spellbook import Spell, Spellbook, gesture_sample_abs_paths

PointArray = NDArray[np.float32]
_DISTANCE_EPSILON = 1e-6


@dataclass(frozen=True)
class GestureTemplate:
    spell_id: str
    name: str
    points: PointArray


@dataclass(frozen=True)
class GestureRanking:
    spell_id: str
    name: str
    distance: float
    score: float


@dataclass(frozen=True)
class GestureDecision:
    accepted: bool
    reason: str
    best_spell_id: str | None = None


@dataclass(frozen=True)
class GestureRecognitionResult:
    ranking: tuple[GestureRanking, ...]
    decision: GestureDecision


def recognize_gesture(
    points: PointArray,
    templates: tuple[GestureTemplate, ...],
    config: GestureRecognitionConfig,
) -> GestureRecognitionResult:
    if not templates:
        return GestureRecognitionResult(
            ranking=(),
            decision=GestureDecision(False, "no trained gesture templates"),
        )
    query = normalize_points(points, config)
    rankings = tuple(
        sorted(
            (
                _rank_template(query, template)
                for template in templates
                if template.points.shape[0] > 0
            ),
            key=lambda ranking: ranking.distance,
        )
    )
    decision = decide_gesture(rankings, config)
    return GestureRecognitionResult(ranking=rankings, decision=decision)


def decide_gesture(
    ranking: tuple[GestureRanking, ...], config: GestureRecognitionConfig
) -> GestureDecision:
    if not ranking:
        return GestureDecision(False, "no trained gesture templates")
    best = ranking[0]
    if best.score < config.score_min:
        return GestureDecision(
            False,
            f"score {best.score:.2f} below {config.score_min:.2f}",
            best.spell_id,
        )
    if len(ranking) > 1:
        second = ranking[1]
        if second.distance <= _DISTANCE_EPSILON:
            margin = 0.0
        else:
            margin = (second.distance - best.distance) / second.distance
        if margin < config.margin_min:
            return GestureDecision(
                False,
                f"margin {margin:.2f} below {config.margin_min:.2f}",
                best.spell_id,
            )
    return GestureDecision(True, "accepted", best.spell_id)


def load_gesture_templates(
    spellbook: Spellbook, config: GestureRecognitionConfig
) -> tuple[GestureTemplate, ...]:
    templates: list[GestureTemplate] = []
    for spell in spellbook.spells:
        if not spell.has_gesture:
            continue
        for path in gesture_sample_abs_paths(spellbook, spell):
            if path.exists():
                templates.append(
                    GestureTemplate(
                        spell_id=spell.id,
                        name=spell.name,
                        points=normalize_points(load_gesture_points(path), config),
                    )
                )
    return tuple(templates)


def gesture_preview_points(
    spellbook: Spellbook, spell: Spell, config: GestureRecognitionConfig
) -> PointArray | None:
    paths = gesture_sample_abs_paths(spellbook, spell)
    if not paths or not paths[0].exists():
        return None
    return normalize_points(load_gesture_points(paths[0]), config)


def normalize_points(
    points: PointArray, config: GestureRecognitionConfig
) -> PointArray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if config.duplicate_distance > 0.0:
        array = _drop_near_duplicates(array, config.duplicate_distance)
    if array.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if array.shape[0] == 1:
        return np.repeat(array, config.point_count, axis=0).astype(np.float32)
    array = _resample(array, config.point_count)
    array = _scale_to_unit_square(array)
    return _translate_centroid_to_origin(array)


def load_gesture_points(path: Path) -> PointArray:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(raw.get("points", ()), dtype=np.float32).reshape(-1, 2)


def save_gesture_points(path: Path, points: PointArray) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    payload = {
        "version": 1,
        "points": [[float(x), float(y)] for x, y in array],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rank_template(query: PointArray, template: GestureTemplate) -> GestureRanking:
    distance = _cloud_match(query, template.points)
    score = 1.0 if distance <= 1.0 else 1.0 / distance
    return GestureRanking(
        spell_id=template.spell_id,
        name=template.name,
        distance=float(distance),
        score=float(score),
    )


def _cloud_match(points: PointArray, template: PointArray) -> float:
    point_count = points.shape[0]
    if point_count == 0 or template.shape[0] == 0:
        return float("inf")
    if point_count != template.shape[0]:
        return float("inf")
    step = max(1, int(np.floor(np.sqrt(point_count))))
    best = float("inf")
    for start in range(0, point_count, step):
        best = min(best, _cloud_distance(points, template, start, best))
        best = min(best, _cloud_distance(template, points, start, best))
    return best


def _cloud_distance(
    points: PointArray, template: PointArray, start: int, best_so_far: float
) -> float:
    point_count = points.shape[0]
    unmatched = list(range(point_count))
    total = 0.0
    index = start
    weight = point_count
    while True:
        deltas = template[unmatched] - points[index]
        distances = np.sum(deltas * deltas, axis=1)
        nearest_position = int(np.argmin(distances))
        total += weight * float(distances[nearest_position])
        if total >= best_so_far:
            return total
        unmatched.pop(nearest_position)
        weight -= 1
        index = (index + 1) % point_count
        if index == start:
            return total


def _drop_near_duplicates(points: PointArray, min_distance: float) -> PointArray:
    if points.shape[0] <= 1:
        return points.astype(np.float32)
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) >= min_distance:
            kept.append(point)
    return np.asarray(kept, dtype=np.float32)


def _resample(points: PointArray, target_count: int) -> PointArray:
    path_length = _path_length(points)
    if path_length <= 0.0:
        return np.repeat(points[:1], target_count, axis=0).astype(np.float32)
    distances = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float32),
            np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
        ]
    )
    targets = np.linspace(0.0, float(path_length), target_count, dtype=np.float32)
    resampled = np.empty((target_count, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(targets, distances, points[:, 0])
    resampled[:, 1] = np.interp(targets, distances, points[:, 1])
    return resampled


def _scale_to_unit_square(points: PointArray) -> PointArray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    size = maximum - minimum
    scale = float(max(size[0], size[1]))
    if scale <= 0.0:
        return points - minimum
    return ((points - minimum) / scale).astype(np.float32)


def _translate_centroid_to_origin(points: PointArray) -> PointArray:
    return (points - points.mean(axis=0, keepdims=True)).astype(np.float32)


def _path_length(points: PointArray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
