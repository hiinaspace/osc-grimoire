from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


def test_production_entry_points_are_desktop_and_overlay_only() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]

    assert scripts == {
        "osc-grimoire-ui": "osc_grimoire.desktop_ui:main",
        "osc-grimoire-overlay": "osc_grimoire.openvr_overlay:main",
    }
    for target in scripts.values():
        module_name, function_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name))
