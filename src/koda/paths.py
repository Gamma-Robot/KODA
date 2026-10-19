from __future__ import annotations
import os
from pathlib import Path
def repo_root() -> Path:
    return Path(os.environ.get("KODA_HOME", Path(__file__).resolve().parents[2])).resolve()
def assets_root() -> Path:
    return Path(os.environ.get("KODA_ASSETS", repo_root() / "assets")).expanduser().resolve()
def cache_root() -> Path:
    return Path(os.environ.get("KODA_CACHE", repo_root() / ".cache")).expanduser().resolve()
