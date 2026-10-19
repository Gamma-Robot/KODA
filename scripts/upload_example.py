#!/usr/bin/env python3
"""Upload the single KODA example MCAP to the project Hugging Face dataset.

The MCAP is intentionally outside Git. Pass --mcap from the downloaded sample
or use --folder when a prepared release directory already exists.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

REPO_ID = os.environ.get("KODA_HF_REPO", "TeleEmbodied/koda-assets")
RELATIVE = Path("samples/session_20260926_005030_iphone14_3/session_data_20260926_005030_iphone14_3.mcap")

def upload(api: HfApi, folder: Path, repo_id: str) -> None:
    try:
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add KODA example session",
        )
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 403:
            raise SystemExit(
                "Hugging Face rejected the upload (403). Use a write-scoped token "
                "and make the dataset public if it should be downloadable without a token."
            ) from exc
        raise

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", type=Path, help="source MCAP outside the repository")
    parser.add_argument("--folder", type=Path, help="prepared folder to upload")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or run hf auth login.")
    if args.folder and args.mcap:
        raise SystemExit("Choose either --folder or --mcap.")
    root = Path(__file__).resolve().parents[1]
    default_folder = root / "release/hf/koda-assets"
    if args.folder:
        folder = args.folder.expanduser().resolve()
    elif args.mcap:
        source = args.mcap.expanduser().resolve()
        if not source.is_file():
            raise SystemExit(f"MCAP not found: {source}")
        with tempfile.TemporaryDirectory(prefix="koda-hf-") as tmp:
            folder = Path(tmp) / "koda-assets"
            destination = folder / RELATIVE
            destination.parent.mkdir(parents=True)
            shutil.copy2(source, destination)
            (folder / "metadata.json").write_text(json.dumps({"dataset": "KODA", "example": RELATIVE.as_posix()}, indent=2) + "\n")
            upload(HfApi(token=token), folder, args.repo_id)
        print(f"uploaded {args.repo_id}")
        return 0
    if not args.folder and default_folder.is_dir():
        folder = default_folder
    if not folder.is_dir():
        raise SystemExit(f"Prepared folder not found: {folder}; pass --mcap /path/to/session.mcap")
    upload(HfApi(token=token), folder, args.repo_id)
    print(f"uploaded {args.repo_id}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
