from __future__ import annotations

import ctypes
import faulthandler
import logging
import sys
from pathlib import Path

from platformdirs import user_log_path

from osc_grimoire.openvr_overlay import main

_FAULT_LOG_FILE = None


def _configure_release_logging() -> Path:
    global _FAULT_LOG_FILE
    log_dir = user_log_path("OSC Grimoire", "Hiina", ensure_exists=True)
    log_path = log_dir / "osc-grimoire-overlay.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    sys.stdout = _LogStream(log_path, "stdout")  # type: ignore[assignment]
    sys.stderr = _LogStream(log_path, "stderr")  # type: ignore[assignment]
    try:
        _FAULT_LOG_FILE = log_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_LOG_FILE)
    except Exception:
        logging.getLogger(__name__).debug(
            "Could not enable faulthandler", exc_info=True
        )
    return log_path


class _LogStream:
    def __init__(self, path: Path, name: str) -> None:
        self.path = path
        self.name = name

    def write(self, text: str) -> int:
        if text.strip():
            with self.path.open("a", encoding="utf-8") as handle:
                for line in text.rstrip().splitlines():
                    handle.write(f"{self.name}: {line}\n")
        return len(text)

    def flush(self) -> None:
        return


def _show_unhandled_exception_dialog(log_path: Path) -> None:
    message = (
        f"OSC Grimoire crashed during startup.\n\nDetails were written to:\n{log_path}"
    )
    try:
        ctypes.windll.user32.MessageBoxW(None, message, "OSC Grimoire", 0x10)
    except Exception:
        logging.getLogger(__name__).debug(
            "Could not show unhandled exception dialog", exc_info=True
        )


def _run() -> int:
    log_path = _configure_release_logging()
    try:
        return main()
    except Exception:
        logging.getLogger(__name__).exception("Unhandled fatal error")
        _show_unhandled_exception_dialog(log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
