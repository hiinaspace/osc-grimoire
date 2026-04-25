from __future__ import annotations

from pathlib import Path

from osc_grimoire.paths import (
    default_data_dir,
    samples_root,
    spell_samples_dir,
    spellbook_path,
)


def test_default_data_dir_is_absolute() -> None:
    assert default_data_dir().is_absolute()


def test_path_helpers_use_data_dir(tmp_path: Path) -> None:
    assert spellbook_path(tmp_path) == tmp_path / "spellbook.json"
    assert samples_root(tmp_path) == tmp_path / "samples"
    assert (
        spell_samples_dir(tmp_path, "abc123") == tmp_path / "samples" / "spell_abc123"
    )
