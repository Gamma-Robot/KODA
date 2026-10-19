#!/usr/bin/env python3
"""Run the standard session exporter, then align its table before publication."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


TARGET_TABLE_HEIGHT_M = 0.75005


def option_value(arguments: list[str], option: str) -> str:
    for index, token in enumerate(arguments):
        if token == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    raise ValueError(f"missing required option {option}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    standard = project_root / "process_session_noviz.py"
    aligner = project_root / "scripts" / "align_session_tables.py"
    arguments = sys.argv[1:]
    output = Path(option_value(arguments, "--out")).expanduser().resolve()
    completed = subprocess.run([sys.executable, str(standard), *arguments], check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    completed = subprocess.run(
        [
            sys.executable,
            str(aligner),
            "--root",
            str(output),
            "--target-height-m",
            str(TARGET_TABLE_HEIGHT_M),
        ],
        check=False,
        env=os.environ.copy(),
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
