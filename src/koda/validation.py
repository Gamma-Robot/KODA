from __future__ import annotations

import re
from pathlib import Path

from .assets import verify
from .paths import assets_root, repo_root

ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_])/(?:data_[A-Za-z0-9_.-]+|home(?:/|$)|Users(?:/|$)|root(?:/|$)|tmp(?:/|$))")
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".md", ".txt", ".xml"}


def scan_absolute_paths(root: Path | None = None) -> list[str]:
    root = (root or repo_root()).resolve()
    hits: list[str] = []
    ignored = {".git", ".venv", ".cache", "__pycache__", "data", "outputs", "assets", "artifacts"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in ignored for part in relative_parts):
            continue
        try:
            content = path.read_text(errors="ignore")
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            if ABSOLUTE.search(line):
                hits.append(f"{path.relative_to(root)}:{line_number}")
    return sorted(set(hits))


def validate(profile: str = "smoke") -> dict:
    missing = verify(profile=profile)
    hits = scan_absolute_paths(repo_root())
    return {
        "repo_root": str(repo_root()),
        "assets_root": str(assets_root()),
        "missing_assets": missing,
        "absolute_paths_in_code": hits,
        "passed": not missing and not hits,
    }
