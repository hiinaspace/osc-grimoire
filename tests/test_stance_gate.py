from __future__ import annotations

from osc_grimoire.config import StanceRecognitionConfig
from osc_grimoire.stance_capture import StanceFrame, StanceSample
from osc_grimoire.stance_gate import StanceGate, StanceTemplate
from osc_grimoire.stance_geometry import Pose


def test_stance_gate_stays_idle_without_templates() -> None:
    gate = StanceGate(StanceRecognitionConfig())

    event = gate.update(now=0.0, left=_pose(0.0), right=_pose(0.0), templates=())

    assert event.state == "idle"
    assert not event.casting_started
    assert event.result is None


def test_stance_gate_locks_start_then_accepts_after_end_hold() -> None:
    config = StanceRecognitionConfig(
        start_hold_s=0.5,
        end_hold_s=0.2,
        active_timeout_s=1.0,
    )
    gate = StanceGate(config)
    template = _template("spell", "Bolt", start=0.0, end=0.4)

    first = gate.update(now=0.0, left=_pose(0.01), right=_pose(0.01), templates=(template,))
    locked = gate.update(now=0.5, left=_pose(0.01), right=_pose(0.01), templates=(template,))
    holding = gate.update(now=0.6, left=_pose(0.4), right=_pose(0.4), templates=(template,))
    accepted = gate.update(
        now=0.81, left=_pose(0.4), right=_pose(0.4), templates=(template,)
    )

    assert first.state == "locking"
    assert locked.casting_started
    assert holding.result is None
    assert accepted.casting_ended
    assert accepted.result is not None
    assert accepted.result.decision.accepted
    assert accepted.result.decision.best_spell_id == "spell"


def test_stance_gate_resets_end_hold_when_end_pose_is_lost() -> None:
    config = StanceRecognitionConfig(
        start_hold_s=0.1,
        end_hold_s=0.2,
        active_timeout_s=1.0,
    )
    gate = StanceGate(config)
    template = _template("spell", "Bolt", start=0.0, end=0.4)

    gate.update(now=0.0, left=_pose(0.0), right=_pose(0.0), templates=(template,))
    gate.update(now=0.1, left=_pose(0.0), right=_pose(0.0), templates=(template,))
    gate.update(now=0.2, left=_pose(0.4), right=_pose(0.4), templates=(template,))
    gate.update(now=0.3, left=_pose(0.2), right=_pose(0.2), templates=(template,))
    holding_again = gate.update(
        now=0.35, left=_pose(0.4), right=_pose(0.4), templates=(template,)
    )
    accepted = gate.update(
        now=0.56, left=_pose(0.4), right=_pose(0.4), templates=(template,)
    )

    assert holding_again.result is None
    assert accepted.result is not None
    assert accepted.result.decision.accepted


def test_stance_gate_fizzles_after_active_timeout() -> None:
    config = StanceRecognitionConfig(start_hold_s=0.1, active_timeout_s=0.2)
    gate = StanceGate(config)
    template = _template("spell", "Bolt", start=0.0, end=0.4)

    gate.update(now=0.0, left=_pose(0.0), right=_pose(0.0), templates=(template,))
    gate.update(now=0.1, left=_pose(0.0), right=_pose(0.0), templates=(template,))
    fizzled = gate.update(now=0.31, left=_pose(0.2), right=_pose(0.2), templates=(template,))

    assert fizzled.casting_ended
    assert fizzled.result is not None
    assert not fizzled.result.decision.accepted
    assert fizzled.result.decision.reason == "stance timed out"


def test_stance_gate_keeps_start_candidate_set_until_end_match() -> None:
    config = StanceRecognitionConfig(
        start_hold_s=0.1,
        end_hold_s=0.1,
        active_timeout_s=1.0,
    )
    gate = StanceGate(config)
    first = _template("a", "A", start=0.0, end=0.4)
    second = _template("b", "B", start=0.0, end=0.8)

    gate.update(now=0.0, left=_pose(0.0), right=_pose(0.0), templates=(first, second))
    locked = gate.update(now=0.1, left=_pose(0.0), right=_pose(0.0), templates=(first, second))
    holding = gate.update(now=0.2, left=_pose(0.8), right=_pose(0.8), templates=(first, second))
    accepted = gate.update(
        now=0.31, left=_pose(0.8), right=_pose(0.8), templates=(first, second)
    )

    assert locked.casting_started
    assert holding.result is None
    assert accepted.result is not None
    assert accepted.result.decision.best_spell_id == "b"


def _template(spell_id: str, name: str, *, start: float, end: float) -> StanceTemplate:
    sample = StanceSample(
        frames=(
            StanceFrame(0.0, _pose(start), _pose(start)),
            StanceFrame(1.0, _pose(end), _pose(end)),
        )
    )
    return StanceTemplate(spell_id=spell_id, name=name, sample=sample)


def _pose(x: float) -> Pose:
    return Pose((x, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
