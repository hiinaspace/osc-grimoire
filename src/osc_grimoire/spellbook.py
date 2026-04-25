from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from .paths import spell_samples_dir, spellbook_path

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Spell:
    id: str
    name: str
    enabled: bool = True
    has_gesture: bool = False
    has_voice: bool = True
    voice_samples: tuple[str, ...] = ()
    gesture_samples: tuple[str, ...] = ()
    osc_address: str | None = None
    intra_class_median: float | None = None


@dataclass(frozen=True)
class Spellbook:
    data_dir: Path
    spells: tuple[Spell, ...] = field(default_factory=tuple)


def load_spellbook(data_dir: Path) -> Spellbook:
    path = spellbook_path(data_dir)
    if not path.exists():
        LOGGER.info("No spellbook at %s; starting empty.", path)
        return Spellbook(data_dir=data_dir)

    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported spellbook version {version!r} (expected {SCHEMA_VERSION})"
        )

    spells = tuple(_spell_from_json(entry) for entry in raw.get("spells", ()))
    return Spellbook(data_dir=data_dir, spells=spells)


def save_spellbook(spellbook: Spellbook) -> None:
    path = spellbook_path(spellbook.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCHEMA_VERSION,
        "spells": [_spell_to_json(s) for s in spellbook.spells],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.debug("Saved %d spell(s) to %s", len(spellbook.spells), path)


def find_spell_by_name(spellbook: Spellbook, name: str) -> Spell | None:
    for spell in spellbook.spells:
        if spell.name.casefold() == name.casefold():
            return spell
    return None


def find_spell_by_id(spellbook: Spellbook, spell_id: str) -> Spell | None:
    for spell in spellbook.spells:
        if spell.id == spell_id:
            return spell
    return None


def create_spell(spellbook: Spellbook, name: str) -> tuple[Spellbook, Spell]:
    if find_spell_by_name(spellbook, name) is not None:
        raise ValueError(f"Spell named {name!r} already exists")
    spell = Spell(id=uuid.uuid4().hex, name=name)
    return replace(spellbook, spells=(*spellbook.spells, spell)), spell


def replace_spell(spellbook: Spellbook, updated: Spell) -> Spellbook:
    new_spells = tuple(updated if s.id == updated.id else s for s in spellbook.spells)
    return replace(spellbook, spells=new_spells)


def delete_spell(spellbook: Spellbook, spell_id: str) -> Spellbook:
    return replace(
        spellbook,
        spells=tuple(s for s in spellbook.spells if s.id != spell_id),
    )


def add_voice_sample(
    spellbook: Spellbook, spell: Spell, relative_path: str
) -> Spellbook:
    # Re-fetch by id so a stale `spell` value (with empty voice_samples)
    # appends to the latest, not overwrites it.
    current = find_spell_by_id(spellbook, spell.id)
    if current is None:
        raise ValueError(f"Spell {spell.id!r} not in spellbook")
    updated = replace(current, voice_samples=(*current.voice_samples, relative_path))
    return replace_spell(spellbook, updated)


def voice_sample_abs_paths(spellbook: Spellbook, spell: Spell) -> list[Path]:
    return [spellbook.data_dir / rel for rel in spell.voice_samples]


def next_voice_sample_path(spellbook: Spellbook, spell: Spell) -> tuple[Path, str]:
    """Return (absolute_path, relative_path) for the next voice sample slot."""
    samples_dir = spell_samples_dir(spellbook.data_dir, spell.id)
    samples_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(samples_dir.glob("voice_*.wav"))
    n = len(existing) + 1
    while True:
        candidate = samples_dir / f"voice_{n:03d}.wav"
        if not candidate.exists():
            relative = candidate.relative_to(spellbook.data_dir).as_posix()
            return candidate, relative
        n += 1


def _spell_from_json(entry: dict) -> Spell:
    modalities = entry.get("modalities", {})
    samples = entry.get("samples", {})
    recognition = entry.get("recognition") or {}
    osc = entry.get("osc") or {}
    return Spell(
        id=entry["id"],
        name=entry["name"],
        enabled=entry.get("enabled", True),
        has_gesture=bool(modalities.get("gesture", False)),
        has_voice=bool(modalities.get("voice", True)),
        voice_samples=tuple(samples.get("voices", ())),
        gesture_samples=tuple(samples.get("gestures", ())),
        osc_address=osc.get("address"),
        intra_class_median=recognition.get("intra_class_median"),
    )


def _spell_to_json(spell: Spell) -> dict:
    return {
        "id": spell.id,
        "name": spell.name,
        "enabled": spell.enabled,
        "modalities": {
            "gesture": spell.has_gesture,
            "voice": spell.has_voice,
        },
        "osc": ({"address": spell.osc_address} if spell.osc_address else None),
        "recognition": {
            "intra_class_median": spell.intra_class_median,
        },
        "samples": {
            "voices": list(spell.voice_samples),
            "gestures": list(spell.gesture_samples),
        },
    }
