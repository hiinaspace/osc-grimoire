from __future__ import annotations

import random
import shutil
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from .audio_playback import AudioPlayer, SoundDeviceAudioPlayer
from .config import AppConfig, VoiceRecognitionConfig
from .gesture_recognizer import (
    GestureDecision,
    GestureRanking,
    gesture_preview_points,
    load_gesture_templates,
    recognize_gesture,
    save_gesture_points,
)
from .osc_output import fizzle_osc_parameter_name, spell_osc_parameter_name
from .paths import spell_samples_dir
from .spellbook import (
    Spell,
    Spellbook,
    add_voice_sample,
    create_spell,
    delete_spell,
    find_spell_by_id,
    gesture_sample_path,
    load_spellbook,
    next_voice_sample_path,
    remove_voice_sample,
    replace_spell,
    save_spellbook,
    set_gesture_sample,
    voice_sample_abs_paths,
)
from .voice_features import FloatArray, trim_voice_audio
from .voice_recognizer import (
    BackendStats,
    Decision,
    SpellRanking,
    VoiceFeature,
    VoiceTemplateBackend,
    compute_backend_stats,
    compute_intra_class_median,
    decide,
    default_voice_backend,
    rank_spells,
    recompute_spell_voice_stats,
)
from .waveform import load_waveform_preview

DEFAULT_SAMPLE_TARGET = 10
PARAKEET_CTC_RELATIVE_MARGIN_MIN = 0.20
DEFAULT_RECOGNITION_STRICTNESS = 0.30
LENIENT_VOICE_MARGIN_MIN = 0.0
STRICT_VOICE_MARGIN_MIN = 0.45
LENIENT_VOICE_INTRA_RATIO_MAX = 999.0
STRICT_VOICE_INTRA_RATIO_MAX = 1.15
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

    def pulse_spell(self, spell: Spell) -> None: ...

    def pulse_fizzle(self) -> None: ...

    def tick(self, now: float | None = None) -> None: ...


class InputSink(Protocol):
    status_text: str
    ui_enabled: bool
    gesture_enabled: bool
    voice_enabled: bool

    def recent_messages(self) -> tuple[Any, ...]: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class DraftSpell:
    name: str


@dataclass(frozen=True)
class RecognitionResult:
    ranking: tuple[SpellRanking, ...]
    decision: Decision
    debug_text: str


@dataclass(frozen=True)
class GestureResult:
    ranking: tuple[GestureRanking, ...]
    decision: GestureDecision
    debug_text: str


@dataclass(frozen=True)
class UiLogEntry:
    timestamp: datetime
    message: str

    def format(self) -> str:
        return f"[{self.timestamp:%H:%M:%S}] {self.message}"


class VoiceTrainingController:
    def __init__(
        self,
        data_dir: Path,
        config: AppConfig | None = None,
        backend: VoiceTemplateBackend | None = None,
        voice_config: VoiceRecognitionConfig | None = None,
        output: OutputSink | None = None,
        osc_input: InputSink | None = None,
        audio_player: AudioPlayer | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.config = config or AppConfig()
        self.voice_config = voice_config or replace(
            self.config.voice, relative_margin_min=PARAKEET_CTC_RELATIVE_MARGIN_MIN
        )
        self.backend = backend or default_voice_backend()
        self.output = output
        self.osc_input = osc_input
        self.audio_player = audio_player or SoundDeviceAudioPlayer()
        self.local_ui_enabled = True
        self.local_gesture_enabled = True
        self.local_voice_enabled = True
        self.voice_strictness = DEFAULT_RECOGNITION_STRICTNESS
        self.gesture_strictness = DEFAULT_RECOGNITION_STRICTNESS
        self.spellbook = load_spellbook(data_dir)
        self.draft: DraftSpell | None = None
        self.status = "Ready."
        self.last_result: RecognitionResult | None = None
        self.last_gesture_result: GestureResult | None = None
        self.last_match_kind: str | None = None
        self.latest_gesture_points: FloatArray | None = None
        self.armed_gesture_spell_id: str | None = None
        self.ui_log: deque[UiLogEntry] = deque(maxlen=12)
        self._backend_stats: BackendStats | None = None
        self._feature_cache: dict[Path, VoiceFeature] | None = None

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

    def set_gesture_enabled(self, enabled: bool) -> None:
        self.local_gesture_enabled = enabled
        self.status = f"Gesture input {'enabled' if enabled else 'disabled'}."

    def set_voice_enabled(self, enabled: bool) -> None:
        self.local_voice_enabled = enabled
        self.status = f"Voice input {'enabled' if enabled else 'disabled'}."

    def set_ui_enabled(self, enabled: bool) -> None:
        self.local_ui_enabled = enabled
        self.status = f"UI {'shown' if enabled else 'hidden'}."

    def toggle_ui_enabled(self) -> None:
        self.set_ui_enabled(not self.local_ui_enabled)

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
            intra_class_ratio_max=_strictness_value(
                value,
                LENIENT_VOICE_INTRA_RATIO_MAX,
                self.config.voice.intra_class_ratio_max,
                STRICT_VOICE_INTRA_RATIO_MAX,
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

    def tick_outputs(self, now: float | None = None) -> None:
        if self.output is not None:
            self.output.tick(now)

    def pulse_fizzle(self) -> None:
        if self.output is not None:
            self.output.pulse_fizzle()

    def preload_backend(self) -> None:
        # Force lazy model load before the UI appears so the first recording action
        # does not hitch on Hugging Face metadata/model loading. Then warm the
        # spellbook feature cache so the first test/recognize action only needs
        # to extract the new query audio.
        silence = np.zeros(self.config.audio.sample_rate, dtype=np.float32)
        self.backend.extract_array(
            silence, self.voice_config, self.config.audio.sample_rate
        )
        self._recognition_cache()

    def reload(self) -> None:
        self.spellbook = load_spellbook(self.data_dir)
        self._invalidate_recognition_cache()

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
        self.spellbook, spell = create_spell(self.spellbook, name)
        save_spellbook(self.spellbook)
        self.draft = None
        self.status = f"Created {spell.name}."
        return spell

    def rename_spell(self, spell_id: str, name: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Spell name cannot be empty")
        updated = replace(spell, name=self._unique_spell_name(clean_name, spell.id))
        self.spellbook = replace_spell(self.spellbook, updated)
        save_spellbook(self.spellbook)
        self.status = f"Renamed spell to {updated.name}."
        return updated

    def update_spell_osc_address(self, spell_id: str, value: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        updated = replace(spell, osc_address=value.strip() or None)
        self.spellbook = replace_spell(self.spellbook, updated)
        save_spellbook(self.spellbook)
        self.status = f"OSC parameter set to {self.spell_osc_parameter_name(updated)}."
        return updated

    def spell_osc_parameter_name(self, spell: Spell) -> str:
        return spell_osc_parameter_name(spell, self.config.osc)

    def fizzle_osc_parameter_name(self) -> str:
        return fizzle_osc_parameter_name(self.config.osc)

    def suggest_spell_name(self, audio: FloatArray) -> str:
        if audio.size == 0:
            raise ValueError("No audio captured")
        from .parakeet_ctc_backends import transcribe_parakeet_ctc_name

        name = transcribe_parakeet_ctc_name(
            audio, self.voice_config, self.config.audio.sample_rate
        )
        if not name:
            raise ValueError("No spoken name detected")
        self.status = f"Heard spell name: {name}."
        return name

    def add_sample_to_spell(self, spell_id: str, audio: FloatArray) -> Spell:
        spell = self._spell_or_raise(spell_id)
        if audio.size == 0:
            raise ValueError("No audio captured")
        audio = trim_voice_audio(audio, self.voice_config)
        path, relative_path = next_voice_sample_path(self.spellbook, spell)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, self.config.audio.sample_rate)
        self.spellbook = add_voice_sample(self.spellbook, spell, relative_path)
        self.spellbook = self._recompute_spell(spell.id)
        save_spellbook(self.spellbook)
        self._update_recognition_cache_after_add(path, audio)
        fresh = self._spell_or_raise(spell.id)
        self.status = f"Saved sample {len(fresh.voice_samples)}/{DEFAULT_SAMPLE_TARGET} for {fresh.name}."
        return fresh

    def add_sample_to_draft(self, audio: FloatArray) -> Spell:
        if audio.size == 0:
            raise ValueError("No audio captured")
        spell = self.persist_draft()
        return self.add_sample_to_spell(spell.id, audio)

    def save_gesture_to_draft(self, points: FloatArray) -> Spell:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < self.config.gesture.min_points:
            raise ValueError(
                f"Gesture needs at least {self.config.gesture.min_points} points"
            )
        spell = self.persist_draft()
        return self.save_gesture_sample(spell.id, points)

    def delete_sample(self, spell_id: str, relative_path: str) -> Spell:
        spell = self._spell_or_raise(spell_id)
        self.spellbook = remove_voice_sample(self.spellbook, spell, relative_path)
        path = self.data_dir / relative_path
        if path.exists():
            path.unlink()
        self.spellbook = self._recompute_spell(spell.id)
        save_spellbook(self.spellbook)
        self._update_recognition_cache_after_delete(path)
        fresh = self._spell_or_raise(spell.id)
        self.status = f"Deleted sample from {fresh.name}."
        return fresh

    def play_sample(self, relative_path: str) -> None:
        path = self.data_dir / relative_path
        self.audio_player.play_file(path)
        self.status = "Playing sample."

    def play_random_sample(self, spell_id: str) -> None:
        spell = self._spell_or_raise(spell_id)
        if not spell.voice_samples:
            raise ValueError(f"{spell.name} has no voice samples")
        relative_path = random.choice(spell.voice_samples)
        self.audio_player.play_file(self.data_dir / relative_path)
        self.status = f"Playing {spell.name} sample."

    def delete_spell(self, spell_id: str) -> str:
        spell = self._spell_or_raise(spell_id)
        samples_dir = spell_samples_dir(self.data_dir, spell.id)
        if samples_dir.exists():
            shutil.rmtree(samples_dir)
        self.spellbook = delete_spell(self.spellbook, spell.id)
        save_spellbook(self.spellbook)
        self._invalidate_recognition_cache()
        self.last_result = None
        self.last_gesture_result = None
        self.last_match_kind = None
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
                f"Accepted: {spell.name} (osc: {self.spell_osc_parameter_name(spell)})"
            )
        else:
            self.add_log(
                f"Fizzle (osc: {self.fizzle_osc_parameter_name()}): "
                f"{result.decision.reason}"
            )
        self._emit_gesture_result(result)
        return result

    def recognize(self, audio: FloatArray) -> RecognitionResult:
        if audio.size == 0:
            raise ValueError("No audio captured")
        backend_stats, feature_cache = self._recognition_cache()
        query = self.backend.extract_array(
            audio, self.voice_config, self.config.audio.sample_rate
        )
        ranking = tuple(
            rank_spells(
                query,
                self.spellbook,
                self.voice_config,
                feature_cache,
                backend=self.backend,
                backend_stats=backend_stats,
            )
        )
        decision = decide(list(ranking), self.voice_config)
        result = RecognitionResult(
            ranking=ranking,
            decision=decision,
            debug_text=format_recognition_debug(ranking, decision),
        )
        self.last_result = result
        self.last_match_kind = "voice"
        self.status = "Accepted." if decision.accepted else "Rejected."
        if decision.accepted and ranking:
            spell = self._spell_or_raise(ranking[0].spell_id)
            self.add_log(
                f"Accepted: {ranking[0].name} "
                f"(osc: {self.spell_osc_parameter_name(spell)})"
            )
        else:
            self.add_log(
                f"Fizzle (osc: {self.fizzle_osc_parameter_name()}): "
                f"{_voice_decision_summary(ranking, decision)}"
            )
        self._emit_recognition_result(result)
        return result

    def sample_previews(self, spell: Spell, points: int = 160) -> list[FloatArray]:
        previews: list[FloatArray] = []
        for path in voice_sample_abs_paths(self.spellbook, spell):
            if path.exists():
                previews.append(load_waveform_preview(path, points))
            else:
                previews.append(np.zeros(points, dtype=np.float32))
        return previews

    def gesture_preview(self, spell: Spell) -> FloatArray | None:
        return gesture_preview_points(self.spellbook, spell, self.config.gesture)

    def next_default_spell_name(self) -> str:
        index = len(self.spellbook.spells) + 1
        while True:
            name = f"New Spell {index}"
            if all(s.name.casefold() != name.casefold() for s in self.spellbook.spells):
                return name
            index += 1

    def _recompute_spell(self, spell_id: str) -> Spellbook:
        spell = self._spell_or_raise(spell_id)
        return recompute_spell_voice_stats(self.spellbook, spell, self.config.voice)

    def _recognition_cache(self) -> tuple[BackendStats, dict[Path, VoiceFeature]]:
        if self._backend_stats is None or self._feature_cache is None:
            self._backend_stats, self._feature_cache = compute_backend_stats(
                self.spellbook, self.voice_config, self.backend
            )
        return self._backend_stats, self._feature_cache

    def _update_recognition_cache_after_add(
        self, path: Path, audio: FloatArray
    ) -> None:
        if self._feature_cache is None:
            return
        self._feature_cache[path] = self.backend.extract_array(
            audio, self.voice_config, self.config.audio.sample_rate
        )
        self._refresh_backend_stats_from_cache()

    def _update_recognition_cache_after_delete(self, path: Path) -> None:
        if self._feature_cache is None:
            return
        self._feature_cache.pop(path, None)
        self._refresh_backend_stats_from_cache()

    def _refresh_backend_stats_from_cache(self) -> None:
        assert self._feature_cache is not None
        intra_class_medians: dict[str, float | None] = {}
        for spell in self.spellbook.spells:
            if not spell.has_voice or not spell.voice_samples:
                continue
            features = [
                self._feature_cache[path]
                for path in voice_sample_abs_paths(self.spellbook, spell)
                if path in self._feature_cache
            ]
            intra_class_medians[spell.id] = compute_intra_class_median(
                features, self.backend
            )
        self._backend_stats = BackendStats(intra_class_medians, 0.0)

    def _invalidate_recognition_cache(self) -> None:
        self._backend_stats = None
        self._feature_cache = None

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

    def _spell_or_raise(self, spell_id: str) -> Spell:
        spell = find_spell_by_id(self.spellbook, spell_id)
        if spell is None:
            raise ValueError(f"Spell {spell_id!r} not found")
        return spell

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
    ranking: tuple[SpellRanking, ...], decision: Decision
) -> str:
    lines: list[str] = []
    for i, row in enumerate(ranking):
        marker = "*" if i == 0 else " "
        intra = (
            f"{row.intra_class_median:7.2f}"
            if row.intra_class_median is not None
            else "    n/a"
        )
        samples = ", ".join(f"{d:.2f}" for d in row.per_sample_distances)
        lines.append(
            f"{marker} {row.name:<10} d={row.aggregate_distance:7.2f} "
            f"intra_med={intra} per_sample=[{samples}]"
        )
    verdict = "ACCEPTED" if decision.accepted else "rejected"
    intra_ratio = (
        f"{decision.intra_ratio:.2f}/{decision.intra_ratio_max:.2f}"
        if decision.intra_ratio is not None
        else "n/a"
    )
    margin_ratio = (
        f"{decision.margin_ratio:.2f}/{decision.margin_ratio_min:.2f}"
        if decision.margin_ratio is not None
        else "n/a"
    )
    lines.append(
        f"decision: {verdict} intra_ratio={intra_ratio} "
        f"margin_ratio={margin_ratio} ({decision.reason})"
    )
    return "\n".join(lines)


def _voice_decision_summary(
    ranking: tuple[SpellRanking, ...], decision: Decision
) -> str:
    if not ranking:
        return "no trained voice samples"
    if (
        decision.intra_ratio is not None
        and decision.intra_ratio > decision.intra_ratio_max
    ):
        return f"low confidence for {ranking[0].name}"
    if (
        decision.margin_ratio is not None
        and decision.margin_ratio < decision.margin_ratio_min
        and len(ranking) > 1
    ):
        return f"too close between {ranking[0].name} and {ranking[1].name}"
    return decision.reason


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
