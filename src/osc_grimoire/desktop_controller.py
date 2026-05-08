from __future__ import annotations

import shutil
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import AppConfig, VoiceRecognitionConfig
from .gesture_recognizer import (
    GestureDecision,
    GestureRanking,
    gesture_preview_points,
    load_gesture_templates,
    recognize_gesture,
    save_gesture_points,
)
from .osc_output import (
    fizzle_osc_parameter_name,
    spell_osc_parameter_name,
    spell_osc_signal_summary,
)
from .paths import spell_samples_dir
from .spellbook import (
    OscStep,
    Spell,
    add_voice_alias,
    create_spell,
    delete_spell,
    find_spell_by_id,
    gesture_sample_path,
    load_spellbook,
    normalize_voice_alias,
    remove_voice_alias,
    replace_spell,
    save_spellbook,
    set_gesture_sample,
    set_stance_sample,
    stance_sample_path,
)
from .stance_capture import StanceSample, load_stance_sample, save_stance_sample
from .stance_gate import (
    StanceDecision,
    StanceGate,
    StanceGateEvent,
    StanceRanking,
    load_stance_templates,
)
from .stance_geometry import Pose
from .voice_features import FloatArray
from .voice_recognizer import (
    Decision,
    SpellRanking,
    TextHypothesis,
    VoiceTemplateBackend,
    decide,
    default_voice_backend,
    rank_spells,
    text_hypotheses,
)

PARAKEET_CTC_RELATIVE_MARGIN_MIN = 0.20
DEFAULT_RECOGNITION_STRICTNESS = 0.30
LENIENT_VOICE_MARGIN_MIN = 0.0
STRICT_VOICE_MARGIN_MIN = 0.45
LENIENT_VOICE_ALIAS_DISTANCE_MAX = 9.0
DEFAULT_VOICE_ALIAS_DISTANCE_MAX = 7.0
STRICT_VOICE_ALIAS_DISTANCE_MAX = 5.0
LENIENT_GESTURE_SCORE_MIN = 0.0
STRICT_GESTURE_SCORE_MIN = 0.70
DEFAULT_GESTURE_SCORE_MIN = 0.20
LENIENT_GESTURE_MARGIN_MIN = 0.0
DEFAULT_GESTURE_MARGIN_MIN = 0.03
STRICT_GESTURE_MARGIN_MIN = 0.25


class OutputSink(Protocol):
    status_text: str

    def set_voice_recording(self, recording: bool) -> None: ...

    def set_gesture_drawing(self, drawing: bool) -> None: ...

    def set_stance_casting(self, casting: bool) -> None: ...

    def pulse_stance_start(self) -> None: ...

    def set_ui_enabled(self, enabled: bool) -> None: ...

    def set_voice_enabled(self, enabled: bool) -> None: ...

    def set_gesture_enabled(self, enabled: bool) -> None: ...

    def set_stance_enabled(self, enabled: bool) -> None: ...

    def set_enable_toggles(
        self,
        *,
        ui_enabled: bool,
        gesture_enabled: bool,
        voice_enabled: bool,
        stance_enabled: bool,
    ) -> None: ...

    def pulse_spell(self, spell: Spell) -> None: ...

    def pulse_fizzle(self) -> None: ...

    def tick(self, now: float | None = None) -> None: ...


class InputSink(Protocol):
    status_text: str
    ui_enabled: bool
    gesture_enabled: bool
    voice_enabled: bool
    stance_enabled: bool

    def recent_messages(self) -> tuple[Any, ...]: ...

    def set_enabled_state(
        self,
        *,
        ui_enabled: bool | None = None,
        gesture_enabled: bool | None = None,
        voice_enabled: bool | None = None,
        stance_enabled: bool | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class DraftSpell:
    name: str


@dataclass(frozen=True)
class RecognitionResult:
    ranking: tuple[SpellRanking, ...]
    decision: Decision
    text_hypotheses: tuple[TextHypothesis, ...]
    pending_incantations: tuple[PendingIncantation, ...]
    debug_text: str


@dataclass(frozen=True)
class PendingIncantation:
    text: str
    distance: float
    token_ids: tuple[int, ...]
    hypothesis_score: float


@dataclass(frozen=True)
class GestureResult:
    ranking: tuple[GestureRanking, ...]
    decision: GestureDecision
    debug_text: str


@dataclass(frozen=True)
class StanceResult:
    ranking: tuple[StanceRanking, ...]
    decision: StanceDecision
    debug_text: str


@dataclass(frozen=True)
class UiLogEntry:
    timestamp: datetime
    message: str

    def format(self) -> str:
        return f"[{self.timestamp:%H:%M:%S}] {self.message}"


class GrimoireController:
    def __init__(
        self,
        data_dir: Path,
        config: AppConfig | None = None,
        backend: VoiceTemplateBackend | None = None,
        voice_config: VoiceRecognitionConfig | None = None,
        output: OutputSink | None = None,
        osc_input: InputSink | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.config = config or AppConfig()
        self.voice_config = voice_config or replace(
            self.config.voice,
            relative_margin_min=PARAKEET_CTC_RELATIVE_MARGIN_MIN,
            voice_alias_distance_max=DEFAULT_VOICE_ALIAS_DISTANCE_MAX,
        )
        self.backend = backend or default_voice_backend()
        self.output = output
        self.osc_input = osc_input
        self.local_ui_enabled = True
        self.local_gesture_enabled = True
        self.local_voice_enabled = True
        self.local_stance_enabled = True
        self.voice_strictness = DEFAULT_RECOGNITION_STRICTNESS
        self.gesture_strictness = DEFAULT_RECOGNITION_STRICTNESS
        self.spellbook = load_spellbook(data_dir)
        self.stance_gate = StanceGate(self.config.stance)
        self.draft: DraftSpell | None = None
        self.status = "Ready."
        self.last_result: RecognitionResult | None = None
        self.last_name_hypotheses: tuple[TextHypothesis, ...] = ()
        self.last_gesture_result: GestureResult | None = None
        self.last_stance_result: StanceResult | None = None
        self.last_match_kind: str | None = None
        self.latest_gesture_points: FloatArray | None = None
        self.latest_stance_sample: StanceSample | None = None
        self.active_stance_preview_spell_id: str | None = None
        self.active_stance_preview_sample: StanceSample | None = None
        self._stance_preview_revision = 0
        self.armed_gesture_spell_id: str | None = None
        self.armed_stance_spell_id: str | None = None
        self.ui_log: deque[UiLogEntry] = deque(maxlen=12)

    @property
    def output_status(self) -> str | None:
        return self.output.status_text if self.output is not None else None

    @property
    def input_status(self) -> str | None:
        return self.osc_input.status_text if self.osc_input is not None else None

    def recent_osc_messages(self) -> tuple[Any, ...]:
        if self.osc_input is None:
            return ()
        return self.osc_input.recent_messages()

    def add_log(self, message: str) -> None:
        self.ui_log.append(UiLogEntry(datetime.now(), message))

    @property
    def ui_enabled(self) -> bool:
        osc_enabled = self.osc_input.ui_enabled if self.osc_input is not None else True
        return self.local_ui_enabled and osc_enabled

    @property
    def gesture_enabled(self) -> bool:
        osc_enabled = (
            self.osc_input.gesture_enabled if self.osc_input is not None else True
        )
        return self.local_gesture_enabled and osc_enabled

    @property
    def voice_enabled(self) -> bool:
        osc_enabled = (
            self.osc_input.voice_enabled if self.osc_input is not None else True
        )
        return self.local_voice_enabled and osc_enabled

    @property
    def stance_enabled(self) -> bool:
        osc_enabled = (
            self.osc_input.stance_enabled if self.osc_input is not None else True
        )
        return self.local_stance_enabled and osc_enabled

    def set_gesture_enabled(self, enabled: bool) -> None:
        self.local_gesture_enabled = enabled
        if self.osc_input is not None:
            self.osc_input.set_enabled_state(gesture_enabled=enabled)
        if self.output is not None:
            self.output.set_gesture_enabled(enabled)
        self.status = f"Gesture input {'enabled' if enabled else 'disabled'}."

    def set_voice_enabled(self, enabled: bool) -> None:
        self.local_voice_enabled = enabled
        if self.osc_input is not None:
            self.osc_input.set_enabled_state(voice_enabled=enabled)
        if self.output is not None:
            self.output.set_voice_enabled(enabled)
        self.status = f"Voice input {'enabled' if enabled else 'disabled'}."

    def set_stance_enabled(self, enabled: bool) -> None:
        self.local_stance_enabled = enabled
        if self.osc_input is not None:
            self.osc_input.set_enabled_state(stance_enabled=enabled)
        if self.output is not None:
            self.output.set_stance_enabled(enabled)
        if not enabled:
            self.reset_stance_gate()
        self.status = f"Stance input {'enabled' if enabled else 'disabled'}."

    def set_ui_enabled(self, enabled: bool) -> None:
        self.local_ui_enabled = enabled
        if self.osc_input is not None:
            self.osc_input.set_enabled_state(ui_enabled=enabled)
        if self.output is not None:
            self.output.set_ui_enabled(enabled)
        self.status = f"UI {'shown' if enabled else 'hidden'}."

    def toggle_ui_enabled(self) -> None:
        self.set_ui_enabled(not self.ui_enabled)

    def set_casting_hand(self, hand: str) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("Casting hand must be 'left' or 'right'")
        book_hand = "left" if hand == "right" else "right"
        self.config = replace(
            self.config,
            openvr=replace(
                self.config.openvr,
                pointer_hand=hand,
                overlay_hand=book_hand,
            ),
        )
        self.status = f"Casting hand set to {hand}."

    def set_voice_strictness(self, value: float) -> None:
        value = min(max(float(value), 0.0), 1.0)
        self.voice_strictness = value
        self.voice_config = replace(
            self.voice_config,
            relative_margin_min=_strictness_value(
                value,
                LENIENT_VOICE_MARGIN_MIN,
                PARAKEET_CTC_RELATIVE_MARGIN_MIN,
                STRICT_VOICE_MARGIN_MIN,
            ),
            voice_alias_distance_max=_strictness_value(
                value,
                LENIENT_VOICE_ALIAS_DISTANCE_MAX,
                DEFAULT_VOICE_ALIAS_DISTANCE_MAX,
                STRICT_VOICE_ALIAS_DISTANCE_MAX,
            ),
        )
        self.status = "Voice tuning updated."

    def set_gesture_strictness(self, value: float) -> None:
        value = min(max(float(value), 0.0), 1.0)
        self.gesture_strictness = value
        self.config = replace(
            self.config,
            gesture=replace(
                self.config.gesture,
                score_min=_strictness_value(
                    value,
                    LENIENT_GESTURE_SCORE_MIN,
                    DEFAULT_GESTURE_SCORE_MIN,
                    STRICT_GESTURE_SCORE_MIN,
                ),
                margin_min=_strictness_value(
                    value,
                    LENIENT_GESTURE_MARGIN_MIN,
                    DEFAULT_GESTURE_MARGIN_MIN,
                    STRICT_GESTURE_MARGIN_MIN,
                ),
            ),
        )
        self.status = "Gesture tuning updated."

    def set_voice_recording(self, recording: bool) -> None:
        if self.output is not None:
            self.output.set_voice_recording(recording)

    def set_gesture_drawing(self, drawing: bool) -> None:
        if self.output is not None:
            self.output.set_gesture_drawing(drawing)

    def set_stance_casting(self, casting: bool) -> None:
        if self.output is not None:
            self.output.set_stance_casting(casting)

    def pulse_stance_start(self) -> None:
        if self.output is not None:
            self.output.pulse_stance_start()

    def sync_enable_toggles_to_output(self) -> None:
        if self.output is None:
            return
        self.output.set_enable_toggles(
            ui_enabled=self.ui_enabled,
            gesture_enabled=self.gesture_enabled,
            voice_enabled=self.voice_enabled,
            stance_enabled=self.stance_enabled,
        )

    def tick_outputs(self, now: float | None = None) -> None:
        if self.output is not None:
            self.output.tick(now)

    def pulse_fizzle(self) -> None:
        if self.output is not None:
            self.output.pulse_fizzle()

    def preload_backend(self) -> None:
        silence = np.zeros(self.config.audio.sample_rate, dtype=np.float32)
        self.backend.extract_array(
            silence, self.voice_config, self.config.audio.sample_rate
        )

    def reload(self) -> None:
        self.spellbook = load_spellbook(self.data_dir)

    def start_draft(self) -> DraftSpell:
        self.draft = DraftSpell(name=self.next_default_spell_name())
        self.status = f"Drafting {self.draft.name}."
        return self.draft

    def cancel_draft(self) -> None:
        self.draft = None
        self.status = "Draft cancelled."

    def update_draft_name(self, name: str) -> None:
        if self.draft is None:
            self.start_draft()
        assert self.draft is not None
        self.draft = replace(self.draft, name=name)

    def persist_draft(self) -> Spell:
        if self.draft is None:
            self.start_draft()
        assert self.draft is not None
        name = self._unique_spell_name(self.draft.name.strip() or "New Spell")
        self._validate_voice_alias(name)
        self.spellbook, spell = create_spell(self.spellbook, name)
        save_spellbook(self.spellbook)
        self.draft = None
        self.status = f"Created {spell.name}."
        return spell

    def rename_spell(self, spell_id: str, name: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        clean_name = normalize_voice_alias(name)
        unique_name = self._unique_spell_name(clean_name, spell.id)
        self._validate_voice_alias(unique_name)
        aliases = spell.voice_aliases
        if aliases == (spell.name,) or not aliases:
            aliases = (unique_name,)
        updated = replace(spell, name=unique_name, voice_aliases=aliases)
        self.spellbook = replace_spell(self.spellbook, updated)
        save_spellbook(self.spellbook)
        self.status = f"Renamed spell to {updated.name}."
        return updated

    def add_voice_alias(self, spell_id: str, alias: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        clean_alias = normalize_voice_alias(alias)
        self._validate_voice_alias(clean_alias)
        self.spellbook = add_voice_alias(self.spellbook, spell, clean_alias)
        save_spellbook(self.spellbook)
        fresh = self._spell_or_raise(spell.id)
        self.status = f"Added incantation {clean_alias}."
        return fresh

    def remove_voice_alias(self, spell_id: str, alias: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        self.spellbook = remove_voice_alias(self.spellbook, spell, alias)
        save_spellbook(self.spellbook)
        fresh = self._spell_or_raise(spell.id)
        self.status = f"Removed incantation from {fresh.name}."
        return fresh

    def update_spell_osc_sequence(
        self, spell_id: str, sequence: tuple[OscStep, ...] | None
    ) -> Spell:
        spell = self._spell_or_raise(spell_id)
        updated = replace(
            spell,
            osc_sequence=sequence,
        )
        self.spellbook = replace_spell(self.spellbook, updated)
        save_spellbook(self.spellbook)
        self.status = f"OSC signal set to {self.spell_osc_signal_summary(updated)}."
        return updated

    def spell_osc_parameter_name(self, spell: Spell) -> str:
        return spell_osc_parameter_name(spell, self.config.osc)

    def spell_osc_signal_summary(self, spell: Spell) -> str:
        return spell_osc_signal_summary(spell, self.config.osc)

    def fizzle_osc_parameter_name(self) -> str:
        return fizzle_osc_parameter_name(self.config.osc)

    def suggest_spell_name(self, audio: FloatArray) -> str:
        if audio.size == 0:
            raise ValueError("No audio captured")
        query = self.backend.extract_array(
            audio, self.voice_config, self.config.audio.sample_rate
        )
        self.last_name_hypotheses = text_hypotheses(query, self.backend)
        if not self.last_name_hypotheses:
            raise ValueError("No spoken name detected")
        name = self.last_name_hypotheses[0].text
        self.status = f"Heard spell name: {name}."
        return name

    def save_gesture_to_draft(self, points: FloatArray) -> Spell:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < self.config.gesture.min_points:
            raise ValueError(
                f"Gesture needs at least {self.config.gesture.min_points} points"
            )
        spell = self.persist_draft()
        return self.save_gesture_sample(spell.id, points)

    def delete_spell(self, spell_id: str) -> str:
        spell = self._spell_or_raise(spell_id)
        samples_dir = spell_samples_dir(self.data_dir, spell.id)
        if samples_dir.exists():
            shutil.rmtree(samples_dir)
        self.spellbook = delete_spell(self.spellbook, spell.id)
        save_spellbook(self.spellbook)
        self.last_result = None
        self.last_gesture_result = None
        self.last_stance_result = None
        self.last_match_kind = None
        self._clear_active_stance_preview(spell.id)
        self.status = f"Deleted spell {spell.name}."
        self.add_log(f"Deleted spell: {spell.name}")
        return spell.name

    def arm_gesture_recording(self, spell_id: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        self.armed_gesture_spell_id = spell.id
        self.status = (
            f"Armed gesture recording for {spell.name}. Hold right grip and draw."
        )
        return spell

    def arm_stance_recording(self, spell_id: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        self.armed_stance_spell_id = spell.id
        self.reset_stance_gate()
        self.status = (
            f"Armed stance recording for {spell.name}. Press both triggers to start."
        )
        return spell

    def handle_gesture_stroke(self, points: FloatArray) -> GestureResult | Spell:
        if self.armed_gesture_spell_id is not None:
            spell_id = self.armed_gesture_spell_id
            self.armed_gesture_spell_id = None
            if spell_id == "__draft__":
                return self.save_gesture_to_draft(points)
            return self.save_gesture_sample(spell_id, points)
        return self.recognize_gesture(points)

    def save_gesture_sample(self, spell_id: str, points: FloatArray) -> Spell:
        spell = self._spell_or_raise(spell_id)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < self.config.gesture.min_points:
            raise ValueError(
                f"Gesture needs at least {self.config.gesture.min_points} points"
            )
        path, relative_path = gesture_sample_path(self.spellbook, spell)
        save_gesture_points(path, points)
        self.spellbook = set_gesture_sample(self.spellbook, spell, relative_path)
        save_spellbook(self.spellbook)
        fresh = self._spell_or_raise(spell.id)
        self.latest_gesture_points = points
        self.last_gesture_result = None
        self.status = f"Saved gesture for {fresh.name}."
        return fresh

    def handle_stance_sample(self, sample: StanceSample) -> Spell:
        if self.armed_stance_spell_id is None:
            raise ValueError("No stance recording is armed")
        spell_id = self.armed_stance_spell_id
        self.armed_stance_spell_id = None
        return self.save_stance_sample(spell_id, sample)

    def save_stance_sample(self, spell_id: str, sample: StanceSample) -> Spell:
        spell = self._spell_or_raise(spell_id)
        if len(sample.frames) < 2:
            raise ValueError("Stance needs a start pose and an end pose")
        path, relative_path = stance_sample_path(self.spellbook, spell)
        save_stance_sample(path, sample)
        self.spellbook = set_stance_sample(self.spellbook, spell, relative_path)
        save_spellbook(self.spellbook)
        fresh = self._spell_or_raise(spell.id)
        self.latest_stance_sample = sample
        self.last_stance_result = None
        if self.active_stance_preview_spell_id == spell.id:
            self.active_stance_preview_sample = sample
            self._stance_preview_revision += 1
        self.status = f"Saved stance for {fresh.name}."
        return fresh

    def clear_gesture_sample(self, spell_id: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        for relative_path in spell.gesture_samples:
            path = self.data_dir / relative_path
            if path.exists():
                path.unlink()
        updated = replace(spell, has_gesture=False, gesture_samples=())
        self.spellbook = replace_spell(self.spellbook, updated)
        save_spellbook(self.spellbook)
        fresh = self._spell_or_raise(spell.id)
        self.last_gesture_result = None
        self.latest_gesture_points = None
        self.status = f"Cleared gesture for {fresh.name}."
        self.add_log(f"Cleared gesture: {fresh.name}")
        return fresh

    def clear_stance_sample(self, spell_id: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        for relative_path in spell.stance_samples:
            path = self.data_dir / relative_path
            if path.exists():
                path.unlink()
        updated = replace(spell, has_stance=False, stance_samples=())
        self.spellbook = replace_spell(self.spellbook, updated)
        save_spellbook(self.spellbook)
        fresh = self._spell_or_raise(spell.id)
        self.last_stance_result = None
        self.latest_stance_sample = None
        self._clear_active_stance_preview(spell.id)
        self.reset_stance_gate()
        self.status = f"Cleared stance for {fresh.name}."
        self.add_log(f"Cleared stance: {fresh.name}")
        return fresh

    def recognize_gesture(self, points: FloatArray) -> GestureResult:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        self.latest_gesture_points = points
        if points.shape[0] < self.config.gesture.min_points:
            result = GestureResult(
                ranking=(),
                decision=GestureDecision(False, "gesture too short"),
                debug_text="gesture: rejected (gesture too short)",
            )
            self.last_gesture_result = result
            self.last_match_kind = "gesture"
            self.status = "Gesture rejected."
            self.add_log(
                f"Fizzle (osc: {self.fizzle_osc_parameter_name()}): gesture too short"
            )
            self._emit_gesture_result(result)
            return result
        templates = load_gesture_templates(self.spellbook, self.config.gesture)
        raw_result = recognize_gesture(points, templates, self.config.gesture)
        result = GestureResult(
            ranking=raw_result.ranking,
            decision=raw_result.decision,
            debug_text=format_gesture_debug(raw_result.ranking, raw_result.decision),
        )
        self.last_gesture_result = result
        self.last_match_kind = "gesture"
        self.status = (
            "Gesture accepted." if result.decision.accepted else "Gesture rejected."
        )
        if result.decision.accepted and result.decision.best_spell_id is not None:
            spell = self._spell_or_raise(result.decision.best_spell_id)
            self.add_log(
                f"Accepted: {spell.name} (osc: {self.spell_osc_signal_summary(spell)})"
            )
        else:
            self.add_log(
                f"Fizzle (osc: {self.fizzle_osc_parameter_name()}): "
                f"{result.decision.reason}"
            )
        self._emit_gesture_result(result)
        return result

    def update_stance_tracking(
        self, *, now: float, left: Pose, right: Pose
    ) -> StanceGateEvent:
        if not self.stance_enabled or self.armed_stance_spell_id is not None:
            casting_ended = self.stance_gate.reset()
            if casting_ended:
                self.set_stance_casting(False)
            return StanceGateEvent(state=self.stance_gate.state)
        templates = load_stance_templates(self.spellbook)
        event = self.stance_gate.update(
            now=now,
            left=left,
            right=right,
            templates=templates,
        )
        self._handle_stance_gate_event(event)
        return event

    def reset_stance_gate(self) -> None:
        if self.stance_gate.reset():
            self.set_stance_casting(False)

    def recognize(self, audio: FloatArray) -> RecognitionResult:
        if audio.size == 0:
            raise ValueError("No audio captured")
        query = self.backend.extract_array(
            audio, self.voice_config, self.config.audio.sample_rate
        )
        ranking = tuple(
            rank_spells(
                query,
                self.spellbook,
                self.voice_config,
                backend=self.backend,
            )
        )
        decision = decide(list(ranking), self.voice_config)
        hypotheses = text_hypotheses(query, self.backend)
        pending_incantations = _score_pending_incantations(
            query, hypotheses, self.backend
        )
        result = RecognitionResult(
            ranking=ranking,
            decision=decision,
            text_hypotheses=hypotheses,
            pending_incantations=pending_incantations,
            debug_text=format_recognition_debug(ranking, decision, hypotheses),
        )
        self.last_result = result
        self.last_match_kind = "voice"
        self.status = "Accepted." if decision.accepted else "Rejected."
        if decision.accepted and ranking:
            spell = self._spell_or_raise(ranking[0].spell_id)
            self.add_log(
                f"Accepted: {ranking[0].name} "
                f"(osc: {self.spell_osc_signal_summary(spell)})"
            )
        else:
            self.add_log(
                f"Fizzle (osc: {self.fizzle_osc_parameter_name()}): "
                f"{_voice_decision_summary(ranking, decision)}"
            )
        self._emit_recognition_result(result)
        return result

    def gesture_preview(self, spell: Spell) -> FloatArray | None:
        return gesture_preview_points(self.spellbook, spell, self.config.gesture)

    def stance_preview(self, spell: Spell) -> StanceSample | None:
        if not spell.stance_samples:
            return None
        path = self.data_dir / spell.stance_samples[0]
        if not path.exists():
            return None
        return load_stance_sample(path)

    def stance_preview_enabled(self, spell_id: str) -> bool:
        return (
            self.active_stance_preview_spell_id == spell_id
            and self.active_stance_preview_sample is not None
        )

    def set_stance_preview_enabled(
        self, spell_id: str, enabled: bool
    ) -> StanceSample | None:
        spell = self._spell_or_raise(spell_id)
        if not enabled:
            self._clear_active_stance_preview(spell.id)
            self.status = f"Stopped stance preview for {spell.name}."
            return None
        sample = self.stance_preview(spell)
        if sample is None:
            raise ValueError(f"No stance recorded for {spell.name}")
        self.active_stance_preview_spell_id = spell.id
        self.active_stance_preview_sample = sample
        self._stance_preview_revision += 1
        self.latest_stance_sample = sample
        self.status = f"Previewing stance for {spell.name}."
        return sample

    def request_stance_preview(self, spell_id: str) -> StanceSample:
        sample = self.set_stance_preview_enabled(spell_id, True)
        assert sample is not None
        return sample

    def stance_preview_state(self) -> tuple[int, StanceSample | None]:
        return self._stance_preview_revision, self.active_stance_preview_sample

    def _clear_active_stance_preview(self, spell_id: str | None = None) -> None:
        if spell_id is not None and self.active_stance_preview_spell_id != spell_id:
            return
        if (
            self.active_stance_preview_spell_id is None
            and self.active_stance_preview_sample is None
        ):
            return
        self.active_stance_preview_spell_id = None
        self.active_stance_preview_sample = None
        self._stance_preview_revision += 1

    def next_default_spell_name(self) -> str:
        index = len(self.spellbook.spells) + 1
        while True:
            name = f"New Spell {index}"
            if all(s.name.casefold() != name.casefold() for s in self.spellbook.spells):
                return name
            index += 1

    def _emit_recognition_result(self, result: RecognitionResult) -> None:
        if self.output is None:
            return
        if result.decision.accepted and result.ranking:
            self.output.pulse_spell(self._spell_or_raise(result.ranking[0].spell_id))
        else:
            self.output.pulse_fizzle()

    def _emit_gesture_result(self, result: GestureResult) -> None:
        if self.output is None:
            return
        if result.decision.accepted and result.decision.best_spell_id is not None:
            self.output.pulse_spell(self._spell_or_raise(result.decision.best_spell_id))
        else:
            self.output.pulse_fizzle()

    def _handle_stance_gate_event(self, event: StanceGateEvent) -> None:
        if event.casting_started:
            self.set_stance_casting(True)
            self.pulse_stance_start()
            self.status = "Stance started."
        if event.result is not None:
            result = StanceResult(
                ranking=event.result.ranking,
                decision=event.result.decision,
                debug_text=format_stance_debug(
                    event.result.ranking, event.result.decision
                ),
            )
            self.last_stance_result = result
            self.last_match_kind = "stance"
            self.status = (
                "Stance accepted." if result.decision.accepted else "Stance fizzled."
            )
            if result.decision.accepted and result.decision.best_spell_id is not None:
                spell = self._spell_or_raise(result.decision.best_spell_id)
                self.add_log(
                    f"Accepted: {spell.name} "
                    f"(osc: {self.spell_osc_signal_summary(spell)})"
                )
            else:
                self.add_log(
                    f"Fizzle (osc: {self.fizzle_osc_parameter_name()}): "
                    f"{result.decision.reason}"
                )
            self._emit_stance_result(result)
        if event.casting_ended:
            self.set_stance_casting(False)

    def _emit_stance_result(self, result: StanceResult) -> None:
        if self.output is None:
            return
        if result.decision.accepted and result.decision.best_spell_id is not None:
            self.output.pulse_spell(self._spell_or_raise(result.decision.best_spell_id))
        else:
            self.output.pulse_fizzle()

    def _spell_or_raise(self, spell_id: str) -> Spell:
        spell = find_spell_by_id(self.spellbook, spell_id)
        if spell is None:
            raise ValueError(f"Spell {spell_id!r} not found")
        return spell

    def _validate_voice_alias(self, alias: str) -> None:
        if self.backend.tokenize_text is None:
            return
        self.backend.tokenize_text(alias)

    def _unique_spell_name(self, name: str, current_spell_id: str | None = None) -> str:
        existing = {
            s.name.casefold() for s in self.spellbook.spells if s.id != current_spell_id
        }
        if name.casefold() not in existing:
            return name
        index = 2
        while f"{name} {index}".casefold() in existing:
            index += 1
        return f"{name} {index}"


def format_recognition_debug(
    ranking: tuple[SpellRanking, ...],
    decision: Decision,
    hypotheses: tuple[TextHypothesis, ...] = (),
) -> str:
    lines: list[str] = []
    for i, row in enumerate(ranking):
        marker = "*" if i == 0 else " "
        lines.append(
            f"{marker} {row.name:<10} d={row.aggregate_distance:7.2f} "
            f"incantation={row.alias!r}"
        )
    verdict = "ACCEPTED" if decision.accepted else "rejected"
    distance = (
        f"{decision.best_distance:.2f}/{decision.distance_max:.2f}"
        if decision.best_distance is not None
        else "n/a"
    )
    margin_ratio = (
        f"{decision.margin_ratio:.2f}/{decision.margin_ratio_min:.2f}"
        if decision.margin_ratio is not None
        else "n/a"
    )
    lines.append(
        f"decision: {verdict} distance={distance} "
        f"margin_ratio={margin_ratio} ({decision.reason})"
    )
    if hypotheses:
        phrase_list = ", ".join(f"{h.text} ({h.score:.1f})" for h in hypotheses)
        lines.append(f"heard: {phrase_list}")
    return "\n".join(lines)


def _voice_decision_summary(
    ranking: tuple[SpellRanking, ...], decision: Decision
) -> str:
    if not ranking:
        return "no incantations"
    if (
        decision.best_distance is not None
        and decision.best_distance > decision.distance_max
    ):
        return f"low confidence for {ranking[0].name}"
    if (
        decision.margin_ratio is not None
        and decision.margin_ratio < decision.margin_ratio_min
        and len(ranking) > 1
    ):
        return f"too close between {ranking[0].name} and {ranking[1].name}"
    return decision.reason


def _score_pending_incantations(
    query: Any,
    hypotheses: tuple[TextHypothesis, ...],
    backend: VoiceTemplateBackend,
) -> tuple[PendingIncantation, ...]:
    if backend.tokenize_text is None or backend.distance_to_tokens is None:
        return ()
    pending: list[PendingIncantation] = []
    seen: set[str] = set()
    for hypothesis in hypotheses:
        key = hypothesis.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            token_ids = backend.tokenize_text(hypothesis.text)
        except ValueError:
            continue
        pending.append(
            PendingIncantation(
                text=hypothesis.text,
                distance=backend.distance_to_tokens(query, token_ids),
                token_ids=token_ids,
                hypothesis_score=hypothesis.score,
            )
        )
    pending.sort(key=lambda row: row.distance)
    return tuple(pending)


def _strictness_value(
    value: float, lenient: float, default: float, strict: float
) -> float:
    if value <= DEFAULT_RECOGNITION_STRICTNESS:
        ratio = value / DEFAULT_RECOGNITION_STRICTNESS
        return lenient + (default - lenient) * ratio
    ratio = (value - DEFAULT_RECOGNITION_STRICTNESS) / (
        1.0 - DEFAULT_RECOGNITION_STRICTNESS
    )
    return default + (strict - default) * ratio


def format_gesture_debug(
    ranking: tuple[GestureRanking, ...], decision: GestureDecision
) -> str:
    if not ranking:
        return f"gesture: rejected ({decision.reason})"
    lines: list[str] = []
    for index, row in enumerate(ranking):
        marker = "*" if index == 0 else " "
        lines.append(
            f"{marker} {row.name:<10} score={row.score:5.2f} d={row.distance:5.2f}"
        )
    state = "ACCEPTED" if decision.accepted else "rejected"
    lines.append(f"gesture decision: {state} ({decision.reason})")
    return "\n".join(lines)


def format_stance_debug(
    ranking: tuple[StanceRanking, ...], decision: StanceDecision
) -> str:
    if not ranking:
        return f"stance: rejected ({decision.reason})"
    lines: list[str] = []
    for index, row in enumerate(ranking):
        marker = "*" if index == 0 else " "
        lines.append(
            f"{marker} {row.name:<10} {row.phase} d={row.score:5.2f} "
            f"lp={row.left_position_m:4.2f} rp={row.right_position_m:4.2f} "
            f"lq={row.left_orientation_rad:4.2f} rq={row.right_orientation_rad:4.2f}"
        )
    state = "ACCEPTED" if decision.accepted else "rejected"
    lines.append(f"stance decision: {state} ({decision.reason})")
    return "\n".join(lines)
