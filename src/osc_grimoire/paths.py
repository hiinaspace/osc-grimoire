from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "osc-grimoire"


def default_data_dir() -> Path:
    return user_data_path(APP_NAME, appauthor=False, roaming=True)


def spellbook_path(data_dir: Path) -> Path:
    return data_dir / "spellbook.json"


def samples_root(data_dir: Path) -> Path:
    return data_dir / "samples"


def spell_samples_dir(data_dir: Path, spell_id: str) -> Path:
    return samples_root(data_dir) / f"spell_{spell_id}"
