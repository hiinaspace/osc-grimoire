from __future__ import annotations

import sys

from .cli import cli_main


def main() -> None:
    sys.exit(cli_main())


__all__ = ["cli_main", "main"]
