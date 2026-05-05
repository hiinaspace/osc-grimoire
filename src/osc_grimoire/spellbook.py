from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from .paths import spell_samples_dir, spellbook_path

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 2
PRESET_SPELL_NAMES = ("Alohomora", "Spongify", "Rictusempra", "Flipendo")
OscValue = bool | int | float


@dataclass(frozen=True)
class OscAction:
    parameter: str
    value: OscValue


@dataclass(frozen=True)
class Spell:
    id: str
    name: str
    enabled: bool = True
    has_gesture: bool = False
    has_voice: bool = True
    voice_aliases: tuple[str, ...] = ()
    gesture_samples: tuple[str, ...] = ()
    osc_address: str | None = None
    osc_on_cast: tuple[OscAction, ...] | None = None
    osc_after_cast: tuple[OscAction, ...] | None = None


@dataclass(frozen=True)
class Spellbook:
    data_dir: Path
    spells: tuple[Spell, ...] = field(default_factory=tuple)


def load_spellbook(data_dir: Path, *, seed_presets: bool = True) -> Spellbook:
    path = spellbook_path(data_dir)
    if not path.exists():
        LOGGER.info("No spellbook at %s; starting with presets.", path)
        return (
            seeded_spellbook(data_dir) if seed_presets else Spellbook(data_dir=data_dir)
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup_path = _backup_corrupt_spellbook(path)
        LOGGER.exception(
            "Could not parse spellbook at %s; moved it to %s and starting with presets.",
            path,
            backup_path,
        )
        return (
            seeded_spellbook(data_dir) if seed_presets else Spellbook(data_dir=data_dir)
        )
    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported spellbook version {version!r} (expected {SCHEMA_VERSION}). "
            "Reset or manually port the old data directory."
        )

    spells = tuple(_spell_from_json(entry) for entry in raw.get("spells", ()))
    return Spellbook(data_dir=data_dir, spells=spells)


def seeded_spellbook(data_dir: Path) -> Spellbook:
    spells = tuple(
        Spell(
            id=f"preset_{name.lower()}",
            name=name,
            has_gesture=False,
            has_voice=True,
            voice_aliases=(name,),
        )
        for name in PRESET_SPELL_NAMES
    )
    return Spellbook(data_dir=data_dir, spells=spells)


def save_spellbook(spellbook: Spellbook) -> None:
    path = spellbook_path(spellbook.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCHEMA_VERSION,
        "spells": [_spell_to_json(s) for s in spellbook.spells],
    }
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    LOGGER.debug("Saved %d spell(s) to %s", len(spellbook.spells), path)


def _backup_corrupt_spellbook(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.corrupt-{timestamp}.bak")
    suffix = 2
    while backup_path.exists():
        backup_path = path.with_name(f"{path.name}.corrupt-{timestamp}-{suffix}.bak")
        suffix += 1
    path.replace(backup_path)
    return backup_path


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
    clean_name = normalize_voice_alias(name)
    if find_spell_by_name(spellbook, clean_name) is not None:
        raise ValueError(f"Spell named {clean_name!r} already exists")
    spell = Spell(id=uuid.uuid4().hex, name=clean_name, voice_aliases=(clean_name,))
    return replace(spellbook, spells=(*spellbook.spells, spell)), spell


def replace_spell(spellbook: Spellbook, updated: Spell) -> Spellbook:
    new_spells = tuple(updated if s.id == updated.id else s for s in spellbook.spells)
    return replace(spellbook, spells=new_spells)


def delete_spell(spellbook: Spellbook, spell_id: str) -> Spellbook:
    return replace(
        spellbook,
        spells=tuple(s for s in spellbook.spells if s.id != spell_id),
    )


def add_voice_alias(spellbook: Spellbook, spell: Spell, alias: str) -> Spellbook:
    current = find_spell_by_id(spellbook, spell.id)
    if current is None:
        raise ValueError(f"Spell {spell.id!r} not in spellbook")
    clean_alias = normalize_voice_alias(alias)
    existing = {a.casefold() for a in current.voice_aliases}
    if clean_alias.casefold() in existing:
        raise ValueError(
            f"Incantation {clean_alias!r} already exists for {current.name}"
        )
    updated = replace(current, voice_aliases=(*current.voice_aliases, clean_alias))
    return replace_spell(spellbook, updated)


def remove_voice_alias(spellbook: Spellbook, spell: Spell, alias: str) -> Spellbook:
    current = find_spell_by_id(spellbook, spell.id)
    if current is None:
        raise ValueError(f"Spell {spell.id!r} not in spellbook")
    clean_alias = normalize_voice_alias(alias)
    updated = replace(
        current,
        voice_aliases=tuple(
            a for a in current.voice_aliases if a.casefold() != clean_alias.casefold()
        ),
    )
    return replace_spell(spellbook, updated)


def replace_voice_aliases(
    spellbook: Spellbook, spell: Spell, aliases: tuple[str, ...]
) -> Spellbook:
    current = find_spell_by_id(spellbook, spell.id)
    if current is None:
        raise ValueError(f"Spell {spell.id!r} not in spellbook")
    cleaned: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        clean_alias = normalize_voice_alias(alias)
        key = clean_alias.casefold()
        if key not in seen:
            cleaned.append(clean_alias)
            seen.add(key)
    updated = replace(current, voice_aliases=tuple(cleaned))
    return replace_spell(spellbook, updated)


def set_gesture_sample(
    spellbook: Spellbook, spell: Spell, relative_path: str
) -> Spellbook:
    current = find_spell_by_id(spellbook, spell.id)
    if current is None:
        raise ValueError(f"Spell {spell.id!r} not in spellbook")
    updated = replace(
        current,
        has_gesture=True,
        gesture_samples=(relative_path,),
    )
    return replace_spell(spellbook, updated)


def gesture_sample_abs_paths(spellbook: Spellbook, spell: Spell) -> list[Path]:
    return [spellbook.data_dir / rel for rel in spell.gesture_samples]


def gesture_sample_path(spellbook: Spellbook, spell: Spell) -> tuple[Path, str]:
    samples_dir = spell_samples_dir(spellbook.data_dir, spell.id)
    samples_dir.mkdir(parents=True, exist_ok=True)
    candidate = samples_dir / "gesture_001.json"
    relative = candidate.relative_to(spellbook.data_dir).as_posix()
    return candidate, relative


def normalize_voice_alias(alias: str) -> str:
    cleaned = " ".join(alias.strip().split())
    if not cleaned:
        raise ValueError("Incantation cannot be empty")
    return cleaned


def parse_osc_actions(text: str) -> tuple[OscAction, ...]:
    actions: list[OscAction] = []
    for raw_part in text.replace("\n", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"OSC action {part!r} must be parameter=value")
        parameter, raw_value = part.split("=", 1)
        actions.append(
            OscAction(
                parameter=normalize_osc_parameter(parameter),
                value=parse_osc_value(raw_value),
            )
        )
    return tuple(actions)


def format_osc_actions(actions: tuple[OscAction, ...]) -> str:
    return ", ".join(format_osc_action(action) for action in actions)


def format_osc_action(action: OscAction) -> str:
    value = action.value
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return f"{action.parameter}={text}"


def parse_osc_value(text: str) -> OscValue:
    clean = text.strip()
    folded = clean.casefold()
    if folded == "true":
        return True
    if folded == "false":
        return False
    try:
        return int(clean)
    except ValueError:
        pass
    try:
        return float(clean)
    except ValueError as exc:
        raise ValueError(
            f"OSC value {text!r} must be true, false, int, or float"
        ) from exc


def normalize_osc_parameter(parameter: str) -> str:
    clean = parameter.strip()
    if not clean:
        raise ValueError("OSC parameter cannot be empty")
    return clean


def _spell_from_json(entry: dict) -> Spell:
    modalities = entry.get("modalities", {})
    samples = entry.get("samples", {})
    recognition = entry.get("recognition") or {}
    osc = entry.get("osc") or {}
    aliases = tuple(
        normalize_voice_alias(alias)
        for alias in recognition.get("voice_aliases", ())
        if str(alias).strip()
    )
    on_cast = _osc_actions_from_json(osc, "on_cast")
    after_cast = _osc_actions_from_json(osc, "after_cast")
    if on_cast is None and after_cast is None and osc.get("mode") == "int":
        parameter = str(osc.get("address") or entry["name"])
        on_cast = (
            OscAction(
                parameter=normalize_osc_parameter(parameter),
                value=int(osc.get("int_value", 1)),
            ),
        )
        after_cast = (
            OscAction(
                parameter=normalize_osc_parameter(parameter),
                value=int(osc.get("int_reset_value", 0)),
            ),
        )
    return Spell(
        id=entry["id"],
        name=entry["name"],
        enabled=entry.get("enabled", True),
        has_gesture=bool(modalities.get("gesture", False)),
        has_voice=bool(modalities.get("voice", True)),
        voice_aliases=aliases,
        gesture_samples=tuple(samples.get("gestures", ())),
        osc_address=osc.get("address"),
        osc_on_cast=on_cast,
        osc_after_cast=after_cast,
    )


def _osc_actions_from_json(osc: dict, key: str) -> tuple[OscAction, ...] | None:
    if key not in osc:
        return None
    return tuple(_osc_action_from_json(entry) for entry in osc.get(key, ()))


def _osc_action_from_json(entry: dict) -> OscAction:
    return OscAction(
        parameter=normalize_osc_parameter(str(entry["parameter"])),
        value=_osc_value_from_json(entry["value"]),
    )


def _osc_value_from_json(value: object) -> OscValue:
    if isinstance(value, bool | int | float):
        return value
    raise ValueError(f"Unsupported OSC action value {value!r}")


def _spell_to_json(spell: Spell) -> dict:
    return {
        "id": spell.id,
        "name": spell.name,
        "enabled": spell.enabled,
        "modalities": {
            "gesture": spell.has_gesture,
            "voice": spell.has_voice,
        },
        "osc": _osc_to_json(spell),
        "recognition": {
            "voice_aliases": list(spell.voice_aliases),
        },
        "samples": {
            "gestures": list(spell.gesture_samples),
        },
    }


def _osc_to_json(spell: Spell) -> dict | None:
    payload: dict[str, object] = {}
    if spell.osc_address:
        payload["address"] = spell.osc_address
    if spell.osc_on_cast is not None:
        payload["on_cast"] = [_osc_action_to_json(a) for a in spell.osc_on_cast]
    if spell.osc_after_cast is not None:
        payload["after_cast"] = [_osc_action_to_json(a) for a in spell.osc_after_cast]
    return payload or None


def _osc_action_to_json(action: OscAction) -> dict[str, object]:
    return {"parameter": action.parameter, "value": action.value}
