from __future__ import annotations

from dataclasses import dataclass

from .config import StanceRecognitionConfig
from .spellbook import Spellbook, stance_sample_abs_paths
from .stance_capture import StanceSample, load_stance_sample
from .stance_geometry import (
    Pose,
    dual_pose_distance,
    dual_pose_within_tolerance,
)


@dataclass(frozen=True)
class StanceTemplate:
    spell_id: str
    name: str
    sample: StanceSample

    @property
    def start_left(self) -> Pose:
        assert self.sample.start is not None
        return self.sample.start.left

    @property
    def start_right(self) -> Pose:
        assert self.sample.start is not None
        return self.sample.start.right

    @property
    def end_left(self) -> Pose:
        assert self.sample.end is not None
        return self.sample.end.left

    @property
    def end_right(self) -> Pose:
        assert self.sample.end is not None
        return self.sample.end.right


@dataclass(frozen=True)
class StanceRanking:
    spell_id: str
    name: str
    phase: str
    score: float
    left_position_m: float
    right_position_m: float
    left_orientation_rad: float
    right_orientation_rad: float


@dataclass(frozen=True)
class StanceDecision:
    accepted: bool
    reason: str
    best_spell_id: str | None = None


@dataclass(frozen=True)
class StanceRecognitionResult:
    ranking: tuple[StanceRanking, ...]
    decision: StanceDecision


@dataclass(frozen=True)
class StanceGateEvent:
    state: str
    ranking: tuple[StanceRanking, ...] = ()
    lock_started: bool = False
    casting_started: bool = False
    casting_ended: bool = False
    result: StanceRecognitionResult | None = None


class StanceGate:
    def __init__(self, config: StanceRecognitionConfig) -> None:
        self.config = config
        self.state = "idle"
        self._candidate_ids: tuple[str, ...] = ()
        self._candidate_started_at = 0.0
        self._active_started_at = 0.0
        self._end_candidate_id: str | None = None
        self._end_candidate_started_at = 0.0
        self._last_ranking: tuple[StanceRanking, ...] = ()

    @property
    def active(self) -> bool:
        return self.state == "active"

    def reset(self) -> bool:
        was_active = self.active
        self.state = "idle"
        self._candidate_ids = ()
        self._candidate_started_at = 0.0
        self._active_started_at = 0.0
        self._end_candidate_id = None
        self._end_candidate_started_at = 0.0
        self._last_ranking = ()
        return was_active

    def update(
        self,
        *,
        now: float,
        left: Pose,
        right: Pose,
        templates: tuple[StanceTemplate, ...],
    ) -> StanceGateEvent:
        if not templates:
            casting_ended = self.reset()
            return StanceGateEvent(state=self.state, casting_ended=casting_ended)
        if self.state == "active":
            return self._update_active(now=now, left=left, right=right, templates=templates)
        return self._update_idle_or_locking(
            now=now, left=left, right=right, templates=templates
        )

    def _update_idle_or_locking(
        self,
        *,
        now: float,
        left: Pose,
        right: Pose,
        templates: tuple[StanceTemplate, ...],
    ) -> StanceGateEvent:
        ranking = _rank_start_matches(left, right, templates, self.config)
        matched_ids = tuple(row.spell_id for row in ranking)
        self._last_ranking = ranking
        if not matched_ids:
            self.reset()
            return StanceGateEvent(state=self.state, ranking=ranking)
        if self.state != "locking" or set(matched_ids) != set(self._candidate_ids):
            self.state = "locking"
            self._candidate_ids = matched_ids
            self._candidate_started_at = float(now)
            return StanceGateEvent(
                state=self.state, ranking=ranking, lock_started=True
            )
        if float(now) - self._candidate_started_at < self.config.start_hold_s:
            return StanceGateEvent(state=self.state, ranking=ranking)
        self.state = "active"
        self._active_started_at = float(now)
        return StanceGateEvent(
            state=self.state,
            ranking=ranking,
            casting_started=True,
        )

    def _update_active(
        self,
        *,
        now: float,
        left: Pose,
        right: Pose,
        templates: tuple[StanceTemplate, ...],
    ) -> StanceGateEvent:
        active_templates = tuple(
            template for template in templates if template.spell_id in self._candidate_ids
        )
        ranking = _rank_end_matches(left, right, active_templates, self.config)
        self._last_ranking = ranking
        if ranking and _end_ranking_within_tolerance(ranking[0], self.config):
            best = ranking[0]
            if self._end_candidate_id != best.spell_id:
                self._end_candidate_id = best.spell_id
                self._end_candidate_started_at = float(now)
            elif float(now) - self._end_candidate_started_at >= self.config.end_hold_s:
                result = StanceRecognitionResult(
                    ranking=ranking,
                    decision=StanceDecision(True, "accepted", best.spell_id),
                )
                self.reset()
                return StanceGateEvent(
                    state=self.state,
                    ranking=ranking,
                    casting_ended=True,
                    result=result,
                )
        else:
            self._end_candidate_id = None
            self._end_candidate_started_at = 0.0
        if float(now) - self._active_started_at >= self.config.active_timeout_s:
            result = StanceRecognitionResult(
                ranking=ranking,
                decision=StanceDecision(False, "stance timed out"),
            )
            self.reset()
            return StanceGateEvent(
                state=self.state,
                ranking=ranking,
                casting_ended=True,
                result=result,
            )
        return StanceGateEvent(state=self.state, ranking=ranking)


def load_stance_templates(spellbook: Spellbook) -> tuple[StanceTemplate, ...]:
    templates: list[StanceTemplate] = []
    for spell in spellbook.spells:
        if not spell.enabled or not spell.has_stance:
            continue
        for path in stance_sample_abs_paths(spellbook, spell):
            if not path.exists():
                continue
            sample = load_stance_sample(path)
            if len(sample.frames) >= 2:
                templates.append(
                    StanceTemplate(spell_id=spell.id, name=spell.name, sample=sample)
                )
    return tuple(templates)


def _rank_start_matches(
    left: Pose,
    right: Pose,
    templates: tuple[StanceTemplate, ...],
    config: StanceRecognitionConfig,
) -> tuple[StanceRanking, ...]:
    rankings = [
        _rank_template(
            left=left,
            right=right,
            template=template,
            phase="start",
            target_left=template.start_left,
            target_right=template.start_right,
        )
        for template in templates
        if dual_pose_within_tolerance(
            left=left,
            right=right,
            target_left=template.start_left,
            target_right=template.start_right,
            position_tolerance_m=config.start_position_tolerance_m,
            orientation_tolerance_rad=config.start_orientation_tolerance_rad,
        )
    ]
    return tuple(sorted(rankings, key=lambda row: row.score))


def _rank_end_matches(
    left: Pose,
    right: Pose,
    templates: tuple[StanceTemplate, ...],
    config: StanceRecognitionConfig,
) -> tuple[StanceRanking, ...]:
    rankings = [
        _rank_template(
            left=left,
            right=right,
            template=template,
            phase="end",
            target_left=template.end_left,
            target_right=template.end_right,
        )
        for template in templates
    ]
    return tuple(sorted(rankings, key=lambda row: row.score))


def _rank_template(
    *,
    left: Pose,
    right: Pose,
    template: StanceTemplate,
    phase: str,
    target_left: Pose,
    target_right: Pose,
) -> StanceRanking:
    distance = dual_pose_distance(
        left=left,
        right=right,
        target_left=target_left,
        target_right=target_right,
    )
    return StanceRanking(
        spell_id=template.spell_id,
        name=template.name,
        phase=phase,
        score=distance.score,
        left_position_m=distance.left.position_m,
        right_position_m=distance.right.position_m,
        left_orientation_rad=distance.left.orientation_rad,
        right_orientation_rad=distance.right.orientation_rad,
    )


def _end_ranking_within_tolerance(
    ranking: StanceRanking, config: StanceRecognitionConfig
) -> bool:
    return (
        ranking.left_position_m <= config.end_position_tolerance_m
        and ranking.right_position_m <= config.end_position_tolerance_m
        and ranking.left_orientation_rad <= config.end_orientation_tolerance_rad
        and ranking.right_orientation_rad <= config.end_orientation_tolerance_rad
    )
