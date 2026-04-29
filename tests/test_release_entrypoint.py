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
