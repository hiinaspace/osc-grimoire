from __future__ import annotations

import json
from pathlib import Path

import pytest

from osc_grimoire.spellbook import (
    PRESET_SPELL_NAMES,
    OscAction,
    Spell,
    Spellbook,
    add_voice_alias,
    create_spell,
    delete_spell,
    find_spell_by_name,
    gesture_sample_path,
    load_spellbook,
    remove_voice_alias,
    save_spellbook,
    set_gesture_sample,
)


def test_load_missing_seeds_preset_spells(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)

    assert tuple(s.name for s in book.spells) == PRESET_SPELL_NAMES
    assert tuple(s.voice_aliases for s in book.spells) == tuple(
        (name,) for name in PRESET_SPELL_NAMES
    )
    assert book.data_dir == tmp_path


def test_load_missing_can_skip_presets_for_tests(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)

    assert book == Spellbook(data_dir=tmp_path)


def test_create_save_load_alias_round_trip(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, spell = create_spell(book, "Flipendo")
    book = add_voice_alias(book, spell, "Fli pendo")
    save_spellbook(book)

    reloaded = load_spellbook(tmp_path)
    assert len(reloaded.spells) == 1
    s = reloaded.spells[0]
    assert s.name == "Flipendo"
    assert s.id == spell.id
    assert s.has_voice is True
    assert s.has_gesture is False
    assert s.voice_aliases == ("Flipendo", "Fli pendo")


def test_spell_osc_actions_round_trip(tmp_path: Path) -> None:
    spell = Spell(
        id="spell-1",
        name="Flipendo",
        voice_aliases=("Flipendo",),
        osc_on_cast=(
            OscAction("Spell", 3),
            OscAction("MagicPrepared", True),
        ),
        osc_after_cast=(),
    )
    save_spellbook(Spellbook(tmp_path, (spell,)))

    reloaded = load_spellbook(tmp_path)

    assert reloaded.spells[0].osc_on_cast == (
        OscAction("Spell", 3),
        OscAction("MagicPrepared", True),
    )
    assert reloaded.spells[0].osc_after_cast == ()


def test_v1_spellbook_is_explicitly_unsupported(tmp_path: Path) -> None:
    (tmp_path / "spellbook.json").write_text(
        json.dumps({"version": 1, "spells": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported spellbook version"):
        load_spellbook(tmp_path)


def test_load_corrupt_spellbook_backs_up_and_starts_with_presets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spellbook.json"
    path.write_text("{", encoding="utf-8")

    book = load_spellbook(tmp_path)

    assert tuple(s.name for s in book.spells) == PRESET_SPELL_NAMES
    assert not path.exists()
    backups = list(tmp_path.glob("spellbook.json.corrupt-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{"


def test_find_spell_is_case_insensitive(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, _ = create_spell(book, "Lumos")
    assert find_spell_by_name(book, "lumos") is not None
    assert find_spell_by_name(book, "LUMOS") is not None
    assert find_spell_by_name(book, "Nox") is None


def test_create_spell_rejects_duplicate(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, _ = create_spell(book, "Lumos")
    with pytest.raises(ValueError):
        create_spell(book, "lumos")


def test_voice_alias_helpers_validate_and_deduplicate(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, spell = create_spell(book, "Lumos")
    book = add_voice_alias(book, spell, "Loo mos")
    fresh = book.spells[0]

    assert fresh.voice_aliases == ("Lumos", "Loo mos")
    with pytest.raises(ValueError, match="already exists"):
        add_voice_alias(book, fresh, "loo mos")
    with pytest.raises(ValueError, match="cannot be empty"):
        add_voice_alias(book, fresh, " ")

    book = remove_voice_alias(book, fresh, "Loo mos")
    assert book.spells[0].voice_aliases == ("Lumos",)


def test_empty_voice_aliases_round_trip_as_voice_inactive(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, spell = create_spell(book, "Lumos")
    book = remove_voice_alias(book, spell, "Lumos")
    save_spellbook(book)

    reloaded = load_spellbook(tmp_path)

    assert reloaded.spells[0].has_voice is True
    assert reloaded.spells[0].voice_aliases == ()


def test_delete_spell_removes_entry(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, spell = create_spell(book, "Nox")
    book = delete_spell(book, spell.id)
    assert book.spells == ()


def test_set_gesture_sample_overwrites_single_gesture(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, spell = create_spell(book, "Lumos")
    _path, relative = gesture_sample_path(book, spell)

    book = set_gesture_sample(book, spell, relative)
    book = set_gesture_sample(book, spell, relative)

    updated = book.spells[0]
    assert updated.has_gesture
    assert updated.gesture_samples == (relative,)
