from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from .paths import assets_root, repo_root

MANIFEST = repo_root() / "assets" / "manifest.yaml"


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("assets manifest schema_version must be 1")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "koda-pipeline/0.1"})
    with urllib.request.urlopen(request) as source, partial.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    partial.replace(target)


def _download_hf_file(item: dict[str, Any], target: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = Path(
        hf_hub_download(
            repo_id=item["repo_id"],
            filename=item["path"],
            repo_type=item.get("repo_type", "model"),
            revision=item.get("revision", "main"),
            local_dir=str(target.parent),
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Hugging Face repositories often keep a checkpoint below a model-family
    # directory while KODA exposes a stable flat path under assets/models.
    # Move the downloaded file to the manifest target when the paths differ.
    if cached.resolve() != target.resolve():
        shutil.move(str(cached), str(target))


def _download_hf_snapshot(item: dict[str, Any], target: Path) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=item["repo_id"],
        repo_type=item.get("repo_type", "model"),
        revision=item.get("revision", "main"),
        local_dir=str(target),
        allow_patterns=item.get("allow_patterns"),
        ignore_patterns=item.get("ignore_patterns"),
    )
    flatten = item.get("flatten")
    if flatten:
        source = target / flatten
        destination = target / Path(flatten).name
        if source.is_file() and source != destination:
            shutil.move(str(source), str(destination))


def _matches(item: dict[str, Any], target: Path) -> tuple[bool, str | None]:
    if not target.exists():
        return False, "missing"
    expected = item.get("sha256")
    if expected and target.is_file() and sha256(target) != expected:
        return False, "sha256 mismatch"
    required = item.get("required", [])
    for relative in required:
        if not (target / relative).exists():
            return False, f"missing {relative}"
    marker = item.get("marker")
    if marker and not (target / marker).exists():
        return False, f"missing marker {marker}"
    return True, None


def download(profile: str = "full", manifest: Path = MANIFEST) -> list[Path]:
    payload = load_manifest(manifest)
    selected = payload.get("profiles", {}).get(profile)
    if selected is None:
        raise ValueError(f"unknown asset profile: {profile}")
    downloaded: list[Path] = []
    manual: list[str] = []
    for name in selected:
        item = payload["assets"][name]
        target = assets_root() / item["target"]
        valid, _ = _matches(item, target)
        if valid:
            downloaded.append(target)
            continue
        source = item.get("source")
        try:
            if source == "huggingface":
                _download_hf_file(item, target)
            elif source == "huggingface_snapshot":
                _download_hf_snapshot(item, target)
            elif source == "url":
                _download_url(item["url"], target)
            elif source == "manual":
                manual.append(f"{name}: {item.get('instructions', item.get('official_url', 'manual setup required'))}")
                continue
            else:
                raise ValueError(f"unsupported source {source!r} for {name}")
        except Exception as exc:
            raise RuntimeError(f"failed to download asset {name} into {target}: {exc}") from exc
        valid, reason = _matches(item, target)
        if not valid:
            raise RuntimeError(f"downloaded asset {name} is incomplete at {target}: {reason}")
        downloaded.append(target)
    if manual:
        details = "\n".join(f"- {entry}" for entry in manual)
        print(
            "The selected profile still needs licensed or manually released assets:\n"
            + details
            + "\nInstall them at the manifest targets, then run koda assets verify --profile "
            + profile
            + ".",
            file=sys.stderr,
        )
    return downloaded


def verify(manifest: Path = MANIFEST, profile: str | None = None) -> list[str]:
    payload = load_manifest(manifest)
    names = payload["profiles"].get(profile, payload["assets"].keys()) if profile else payload["assets"].keys()
    missing: list[str] = []
    for name in names:
        item = payload["assets"][name]
        target = assets_root() / item["target"]
        valid, reason = _matches(item, target)
        if not valid:
            missing.append(f"{name}: {reason}")
    return missing
