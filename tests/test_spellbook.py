from __future__ import annotations

from pathlib import Path

import pytest

from osc_grimoire.spellbook import (
    add_voice_sample,
    create_spell,
    delete_spell,
    find_spell_by_name,
    load_spellbook,
    next_voice_sample_path,
    save_spellbook,
)


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    assert book.spells == ()
    assert book.data_dir == tmp_path


def test_create_save_load_round_trip(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    book, spell = create_spell(book, "Flipendo")
    book = add_voice_sample(book, spell, f"samples/spell_{spell.id}/voice_001.wav")
    save_spellbook(book)

    reloaded = load_spellbook(tmp_path)
    assert len(reloaded.spells) == 1
    s = reloaded.spells[0]
    assert s.name == "Flipendo"
    assert s.id == spell.id
    assert s.has_voice is True
    assert s.has_gesture is False
    assert s.voice_samples == (f"samples/spell_{spell.id}/voice_001.wav",)


def test_find_spell_is_case_insensitive(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    book, _ = create_spell(book, "Lumos")
    assert find_spell_by_name(book, "lumos") is not None
    assert find_spell_by_name(book, "LUMOS") is not None
    assert find_spell_by_name(book, "Nox") is None


def test_create_spell_rejects_duplicate(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    book, _ = create_spell(book, "Lumos")
    with pytest.raises(ValueError):
        create_spell(book, "lumos")


def test_delete_spell_removes_entry(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    book, spell = create_spell(book, "Nox")
    book = delete_spell(book, spell.id)
    assert book.spells == ()


def test_next_voice_sample_path_increments(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    book, spell = create_spell(book, "Test")
    abs1, rel1 = next_voice_sample_path(book, spell)
    abs1.touch()
    abs2, rel2 = next_voice_sample_path(book, spell)
    assert rel1 == f"samples/spell_{spell.id}/voice_001.wav"
    assert rel2 == f"samples/spell_{spell.id}/voice_002.wav"
    assert abs1 != abs2
