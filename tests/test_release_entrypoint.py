from __future__ import annotations

import logging
import sys
from pathlib import Path


def test_release_entrypoint_configures_file_logging(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.osc_grimoire_overlay_entry as entrypoint

    monkeypatch.setattr(entrypoint, "user_log_path", lambda *_, **__: tmp_path)

    log_path = entrypoint._configure_release_logging()
    logging.getLogger("osc_grimoire.test").warning("hello")
    print("printed")
    print("errored", file=sys.stderr)

    text = log_path.read_text(encoding="utf-8")

    assert log_path == tmp_path / "osc-grimoire-overlay.log"
    assert "WARNING osc_grimoire.test: hello" in text
    assert "stdout: printed" in text
    assert "stderr: errored" in text


def test_release_entrypoint_logs_unhandled_exception(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.osc_grimoire_overlay_entry as entrypoint

    shown_paths: list[Path] = []

    def crash() -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(entrypoint, "user_log_path", lambda *_, **__: tmp_path)
    monkeypatch.setattr(entrypoint, "main", crash)
    monkeypatch.setattr(
        entrypoint,
        "_show_unhandled_exception_dialog",
        lambda path: shown_paths.append(path),
    )

    assert entrypoint._run() == 1

    log_path = tmp_path / "osc-grimoire-overlay.log"
    text = log_path.read_text(encoding="utf-8")
    assert "ERROR scripts.osc_grimoire_overlay_entry: Unhandled fatal error" in text
    assert "RuntimeError: boom" in text
    assert shown_paths == [log_path]
