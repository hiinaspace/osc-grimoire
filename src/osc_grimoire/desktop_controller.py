from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import AppConfig, VoiceRecognitionConfig
from .spellbook import (
    Spell,
    Spellbook,
    add_voice_sample,
    create_spell,
    find_spell_by_id,
    load_spellbook,
    next_voice_sample_path,
    remove_voice_sample,
    replace_spell,
    save_spellbook,
    voice_sample_abs_paths,
)
from .voice_embedding_backends import whisper_dtw_backend
from .voice_features import FloatArray
from .voice_recognizer import (
    BackendStats,
    Decision,
    SpellRanking,
    VoiceTemplateBackend,
    compute_backend_stats,
    decide,
    rank_spells,
    recompute_spell_voice_stats,
)
from .waveform import load_waveform_preview

DEFAULT_SAMPLE_TARGET = 10
WHISPER_DTW_RELATIVE_MARGIN_MIN = 0.15


@dataclass(frozen=True)
class DraftSpell:
    name: str


@dataclass(frozen=True)
class RecognitionResult:
    ranking: tuple[SpellRanking, ...]
    decision: Decision
    debug_text: str


class VoiceTrainingController:
    def __init__(
        self,
        data_dir: Path,
        config: AppConfig | None = None,
        backend: VoiceTemplateBackend | None = None,
        voice_config: VoiceRecognitionConfig | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.config = config or AppConfig()
        self.voice_config = voice_config or replace(
            self.config.voice, relative_margin_min=WHISPER_DTW_RELATIVE_MARGIN_MIN
        )
        self.backend = backend or whisper_dtw_backend()
        self.spellbook = load_spellbook(data_dir)
        self.draft: DraftSpell | None = None
        self.status = "Ready."
        self.last_result: RecognitionResult | None = None
        self._backend_stats: BackendStats | None = None
        self._feature_cache: dict[Path, FloatArray] | None = None

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
        self._invalidate_recognition_cache()
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

    def add_sample_to_spell(self, spell_id: str, audio: FloatArray) -> Spell:
        spell = self._spell_or_raise(spell_id)
        if audio.size == 0:
            raise ValueError("No audio captured")
        path, relative_path = next_voice_sample_path(self.spellbook, spell)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, self.config.audio.sample_rate)
        self.spellbook = add_voice_sample(self.spellbook, spell, relative_path)
        self.spellbook = self._recompute_spell(spell.id)
        save_spellbook(self.spellbook)
        self._invalidate_recognition_cache()
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
        self._invalidate_recognition_cache()
        fresh = self._spell_or_raise(spell.id)
        self.status = f"Deleted sample from {fresh.name}."
        return fresh

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
        self.status = "Accepted." if decision.accepted else "Rejected."
        return result

    def sample_previews(self, spell: Spell, points: int = 160) -> list[FloatArray]:
        previews: list[FloatArray] = []
        for path in voice_sample_abs_paths(self.spellbook, spell):
            if path.exists():
                previews.append(load_waveform_preview(path, points))
            else:
                previews.append(np.zeros(points, dtype=np.float32))
        return previews

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

    def _recognition_cache(self) -> tuple[BackendStats, dict[Path, FloatArray]]:
        if self._backend_stats is None or self._feature_cache is None:
            self._backend_stats, self._feature_cache = compute_backend_stats(
                self.spellbook, self.voice_config, self.backend
            )
        return self._backend_stats, self._feature_cache

    def _invalidate_recognition_cache(self) -> None:
        self._backend_stats = None
        self._feature_cache = None

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
