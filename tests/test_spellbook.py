from __future__ import annotations

import json
from pathlib import Path

import pytest

from osc_grimoire.spellbook import (
    PRESET_SPELL_NAMES,
    OscAction,
    OscPause,
    Spell,
    Spellbook,
    add_voice_alias,
    create_spell,
    delete_spell,
    find_spell_by_name,
    format_osc_sequence,
    gesture_sample_path,
    load_spellbook,
    parse_osc_sequence,
    remove_voice_alias,
    save_spellbook,
    set_gesture_sample,
    set_stance_sample,
    stance_sample_path,
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
    assert s.has_stance is False
    assert s.voice_aliases == ("Flipendo", "Fli pendo")


def test_osc_sequence_parser_formats_sends_and_pauses() -> None:
    steps = parse_osc_sequence(
        "Spell=4, (Pause 200ms), MagicPrepared=false, (pause 1.5s), Float=0.5"
    )

    assert steps == (
        OscAction("Spell", 4),
        OscPause(0.2),
        OscAction("MagicPrepared", False),
        OscPause(1.5),
        OscAction("Float", 0.5),
    )
    assert (
        format_osc_sequence(
            (
                OscAction("Spell", 4),
                OscPause(0.15),
                OscAction("Spell", False),
            )
        )
        == "Spell=4, (Pause 150ms), Spell=false"
    )


@pytest.mark.parametrize(
    "text, pattern",
    (
        ("(Wait 200ms)", "pause step"),
        ("(Pause 0ms)", "positive"),
        ("(Pause -1s)", "positive"),
        ("(Pause 200)", "pause step"),
        ("Spell=hello", "true, false, int, or float"),
        ("NotAThing", "parameter=value"),
    ),
)
def test_osc_sequence_parser_rejects_malformed_steps(
    text: str, pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        parse_osc_sequence(text)


def test_spell_osc_sequence_round_trip(tmp_path: Path) -> None:
    spell = Spell(
        id="spell-1",
        name="Flipendo",
        voice_aliases=("Flipendo",),
        osc_sequence=(
            OscAction("Spell", 3),
            OscPause(0.2),
            OscAction("MagicPrepared", True),
        ),
    )
    save_spellbook(Spellbook(tmp_path, (spell,)))

    reloaded = load_spellbook(tmp_path)
    raw = json.loads((tmp_path / "spellbook.json").read_text(encoding="utf-8"))

    assert raw["version"] == 4
    assert raw["spells"][0]["osc"]["sequence"] == [
        {"type": "send", "parameter": "Spell", "value": 3},
        {"type": "pause", "duration_ms": 200},
        {"type": "send", "parameter": "MagicPrepared", "value": True},
    ]
    assert reloaded.spells[0].osc_sequence == (
        OscAction("Spell", 3),
        OscPause(0.2),
        OscAction("MagicPrepared", True),
    )


def test_v1_spellbook_is_explicitly_unsupported(tmp_path: Path) -> None:
    (tmp_path / "spellbook.json").write_text(
        json.dumps({"version": 1, "spells": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported spellbook version"):
        load_spellbook(tmp_path)


def test_v2_spellbook_loads_with_empty_stance_fields(tmp_path: Path) -> None:
    (tmp_path / "spellbook.json").write_text(
        json.dumps(
            {
                "version": 2,
                "spells": [
                    {
                        "id": "spell-1",
                        "name": "Lumos",
                        "modalities": {"voice": True, "gesture": False},
                        "recognition": {"voice_aliases": ["Lumos"]},
                        "samples": {"gestures": []},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    book = load_spellbook(tmp_path)

    assert book.spells[0].has_stance is False
    assert book.spells[0].stance_samples == ()
    assert book.spells[0].osc_sequence is None


def test_v3_spellbook_osc_actions_migrate_to_v4_sequence(tmp_path: Path) -> None:
    (tmp_path / "spellbook.json").write_text(
        json.dumps(
            {
                "version": 3,
                "spells": [
                    {
                        "id": "spell-1",
                        "name": "Alohomora",
                        "modalities": {
                            "voice": True,
                            "gesture": False,
                            "stance": False,
                        },
                        "recognition": {"voice_aliases": ["Alohomora"]},
                        "samples": {"gestures": [], "stances": []},
                        "osc": {
                            "on_cast": [
                                {"parameter": "Spell", "value": 4},
                                {"parameter": "MagicPrepared", "value": True},
                            ],
                            "after_cast": [
                                {"parameter": "MagicPrepared", "value": False}
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    book = load_spellbook(tmp_path)

    assert book.spells[0].osc_sequence == (
        OscAction("Spell", 4),
        OscAction("MagicPrepared", True),
        OscPause(0.15),
        OscAction("MagicPrepared", False),
    )
    save_spellbook(book)
    raw = json.loads((tmp_path / "spellbook.json").read_text(encoding="utf-8"))
    assert raw["version"] == 4
    assert "sequence" in raw["spells"][0]["osc"]
    assert "on_cast" not in raw["spells"][0]["osc"]
    assert "after_cast" not in raw["spells"][0]["osc"]


def test_legacy_int_osc_mode_migrates_to_sequence(tmp_path: Path) -> None:
    (tmp_path / "spellbook.json").write_text(
        json.dumps(
            {
                "version": 2,
                "spells": [
                    {
                        "id": "spell-1",
                        "name": "Alohomora",
                        "modalities": {"voice": True, "gesture": False},
                        "recognition": {"voice_aliases": ["Alohomora"]},
                        "samples": {"gestures": []},
                        "osc": {
                            "mode": "int",
                            "address": "Spell",
                            "int_value": 7,
                            "int_reset_value": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    book = load_spellbook(tmp_path)

    assert book.spells[0].osc_sequence == (
        OscAction("Spell", 7),
        OscPause(0.15),
        OscAction("Spell", 0),
    )


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


def test_set_stance_sample_overwrites_single_stance(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path, seed_presets=False)
    book, spell = create_spell(book, "Lumos")
    _path, relative = stance_sample_path(book, spell)

    book = set_stance_sample(book, spell, relative)
    book = set_stance_sample(book, book.spells[0], relative)

    updated = book.spells[0]
    assert updated.has_stance
    assert updated.stance_samples == (relative,)
