from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from .config import AppConfig, VoiceRecognitionConfig
from .gesture_recognizer import (
    GestureDecision,
    GestureRanking,
    gesture_preview_points,
    load_gesture_templates,
    recognize_gesture,
    save_gesture_points,
)
from .spellbook import (
    Spell,
    Spellbook,
    add_voice_sample,
    create_spell,
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
class EmbeddingPoint:
    x: float
    y: float
    label: str
    group: str
    kind: str
    age: int = 0
    accepted: bool | None = None


class VoiceTrainingController:
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
            self.config.voice, relative_margin_min=PARAKEET_CTC_RELATIVE_MARGIN_MIN
        )
        self.backend = backend or default_voice_backend()
        self.output = output
        self.osc_input = osc_input
        self.spellbook = load_spellbook(data_dir)
        self.draft: DraftSpell | None = None
        self.status = "Ready."
        self.last_result: RecognitionResult | None = None
        self.last_gesture_result: GestureResult | None = None
        self.latest_gesture_points: FloatArray | None = None
        self.armed_gesture_spell_id: str | None = None
        self.recent_query_vectors: list[tuple[FloatArray, Decision, str]] = []
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

    @property
    def ui_enabled(self) -> bool:
        return self.osc_input.ui_enabled if self.osc_input is not None else True

    @property
    def gesture_enabled(self) -> bool:
        return self.osc_input.gesture_enabled if self.osc_input is not None else True

    @property
    def voice_enabled(self) -> bool:
        return self.osc_input.voice_enabled if self.osc_input is not None else True

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
            self.status = "Gesture rejected."
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
        self.status = (
            "Gesture accepted." if result.decision.accepted else "Gesture rejected."
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
        best_name = ranking[0].name if ranking else "(none)"
        if isinstance(query, np.ndarray):
            self.recent_query_vectors.append((_mean_vector(query), decision, best_name))
            self.recent_query_vectors = self.recent_query_vectors[-8:]
        self.status = "Accepted." if decision.accepted else "Rejected."
        self._emit_recognition_result(result)
        return result

    def embedding_points(self) -> tuple[EmbeddingPoint, ...]:
        _backend_stats, feature_cache = self._recognition_cache()
        vectors: list[FloatArray] = []
        metadata: list[tuple[str, str, bool | None, int]] = []
        for spell in self.spellbook.spells:
            for path in voice_sample_abs_paths(self.spellbook, spell):
                features = feature_cache.get(path)
                if features is None:
                    continue
                if isinstance(features, np.ndarray):
                    vectors.append(_mean_vector(features))
                    metadata.append((spell.name, "sample", None, 0))

        history_count = len(self.recent_query_vectors)
        for index, (vector, decision, best_name) in enumerate(
            self.recent_query_vectors
        ):
            vectors.append(vector)
            age = history_count - index - 1
            metadata.append((best_name, "query", decision.accepted, age))

        if not vectors:
            return ()

        projected = _pca_2d(np.vstack(vectors).astype(np.float32))
        return tuple(
            EmbeddingPoint(
                x=float(projected[i, 0]),
                y=float(projected[i, 1]),
                label=label,
                group=label,
                kind=kind,
                accepted=accepted,
                age=age,
            )
            for i, (label, kind, accepted, age) in enumerate(metadata)
        )

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
        self.recent_query_vectors.clear()

    def _update_recognition_cache_after_delete(self, path: Path) -> None:
        if self._feature_cache is None:
            return
        self._feature_cache.pop(path, None)
        self._refresh_backend_stats_from_cache()
        self.recent_query_vectors.clear()

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
        self.recent_query_vectors.clear()

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


def _mean_vector(features: FloatArray) -> FloatArray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1).mean(axis=0).astype(np.float32)
    return array.reshape(array.shape[0], -1).mean(axis=0).astype(np.float32)


def _pca_2d(vectors: FloatArray) -> FloatArray:
    if vectors.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    projected = centered @ components
    if projected.shape[1] == 1:
        projected = np.column_stack(
            [projected[:, 0], np.zeros(projected.shape[0], dtype=np.float32)]
        )
    return projected[:, :2].astype(np.float32)
