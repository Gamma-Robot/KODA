#!/usr/bin/env python3
"""EfficientSAM3 client: one loaded model, many prompt/episode jobs.

Talks to ``efficientsam3_worker.py`` over a Unix socket when
``EFFICIENTSAM3_WORKER_SOCK`` (or ``--socket``) is set.  Without a socket the
CLI still loads the checkpoint **once** and runs every prompt in-process, so
contact precompute no longer pays a fresh ``from_pretrained`` per prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


def prompt_slug(prompt: str) -> str:
    """Match ``run_contact_precompute.sh`` slug: lower, space→_, alnum+_."""
    mapped = []
    for char in prompt:
        if char == " ":
            mapped.append("_")
        elif char.isupper():
            mapped.append(char.lower())
        else:
            mapped.append(char)
    return "".join(ch for ch in mapped if ch.isalnum() or ch == "_")


def json_socket_request(
    sock_path: str | Path,
    payload: dict[str, Any],
    *,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if timeout_s is not None:
            sock.settimeout(float(timeout_s))
        sock.connect(str(sock_path))
        sock.sendall(encoded)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("EfficientSAM3 worker closed the connection")
            buf += chunk
        reply = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
        if not isinstance(reply, dict):
            raise ValueError("EfficientSAM3 worker reply must be a JSON object")
        return reply
    finally:
        sock.close()


def resolve_socket(explicit: str | Path | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(Path(explicit).expanduser())
    return str(os.environ.get("EFFICIENTSAM3_WORKER_SOCK", "")).strip()


def segment_job(
    *,
    video: Path,
    episode_dir: Path,
    prompt: str,
    start_frame: int = 0,
    max_frames: int = 0,
    conf_threshold: float = 0.20,
    close_px: int = 0,
    dilation_px: int = 0,
    temporal_radius: int = 0,
    batch_size: int = 16,
    resolution: int = 1008,
    force: bool = False,
    packed_rgb: Path | None = None,
    socket_path: str | Path | None = None,
    timeout_s: float | None = 7200.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "op": "segment",
        "video": str(Path(video).resolve()),
        "episode_dir": str(Path(episode_dir).resolve()),
        "prompt": str(prompt),
        "start_frame": int(start_frame),
        "max_frames": int(max_frames),
        "conf_threshold": float(conf_threshold),
        "close_px": int(close_px),
        "dilation_px": int(dilation_px),
        "temporal_radius": int(temporal_radius),
        "batch_size": int(batch_size),
        "resolution": int(resolution),
        "force": bool(force),
    }
    if packed_rgb is not None:
        payload["packed_rgb"] = str(Path(packed_rgb).resolve())
    sock = resolve_socket(socket_path)
    if not sock:
        raise RuntimeError("EFFICIENTSAM3_WORKER_SOCK / --socket is required for segment_job")
    reply = json_socket_request(sock, payload, timeout_s=timeout_s)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error") or "EfficientSAM3 worker segment failed")
    return reply


def segment_jobs(
    *,
    video: Path,
    jobs: list[tuple[str, Path]],
    start_frame: int = 0,
    max_frames: int = 0,
    conf_threshold: float = 0.20,
    close_px: int = 0,
    dilation_px: int = 0,
    temporal_radius: int = 0,
    batch_size: int = 16,
    resolution: int = 1008,
    force: bool = False,
    packed_rgb: Path | None = None,
    socket_path: str | Path | None = None,
    timeout_s: float | None = 7200.0,
) -> dict[str, Any]:
    sock = resolve_socket(socket_path)
    if not sock:
        raise RuntimeError("EFFICIENTSAM3_WORKER_SOCK / --socket is required for segment_jobs")
    payload: dict[str, Any] = {
        "op": "segment",
        "video": str(Path(video).resolve()),
        "prompts": [prompt for prompt, _ in jobs],
        "episode_dirs": [str(Path(directory).resolve()) for _, directory in jobs],
        "start_frame": int(start_frame),
        "max_frames": int(max_frames),
        "conf_threshold": float(conf_threshold),
        "close_px": int(close_px),
        "dilation_px": int(dilation_px),
        "temporal_radius": int(temporal_radius),
        "batch_size": int(batch_size),
        "resolution": int(resolution),
        "force": bool(force),
    }
    if packed_rgb is not None:
        payload["packed_rgb"] = str(Path(packed_rgb).resolve())
    reply = json_socket_request(sock, payload, timeout_s=timeout_s)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error") or "EfficientSAM3 worker segment failed")
    return reply


def _run_local(
    *,
    video: Path,
    jobs: list[tuple[str, Path]],
    start_frame: int,
    max_frames: int,
    conf_threshold: float,
    close_px: int,
    dilation_px: int,
    temporal_radius: int,
    batch_size: int,
    resolution: int,
    force: bool,
    device: str,
    packed_rgb: Path | None = None,
) -> list[dict[str, Any]]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from sam3_efficientsam3_hand_masks import (
        DEFAULT_EFF_CKPT,
        load_efficientsam3_image_model,
        run_episode_efficientsam3,
    )

    records = []
    pending: list[tuple[str, Path]] = []
    for prompt, episode_dir in jobs:
        meta_path = episode_dir / "mask_metadata.json"
        if meta_path.is_file() and not force:
            records.append(
                {
                    "prompt": prompt,
                    "directory": str(episode_dir),
                    "reused": True,
                    "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
                }
            )
        else:
            pending.append((prompt, episode_dir))
    if not pending:
        return records
    model = load_efficientsam3_image_model(
        device=device, conf_threshold=float(conf_threshold)
    )
    metadata = run_episode_efficientsam3(
        model,
        video=video,
        packed_rgb=packed_rgb,
        jobs=pending,
        start_frame=start_frame,
        max_frames=max_frames,
        close_px=close_px,
        dilation_px=dilation_px,
        temporal_radius=temporal_radius,
        conf_threshold=conf_threshold,
        batch_size=batch_size,
        resolution=resolution,
        eff_ckpt=DEFAULT_EFF_CKPT,
    )
    per_prompt = {item["prompts"][0]: item for item in metadata.get("records", [])}
    for prompt, episode_dir in pending:
        records.append(
            {
                "prompt": prompt,
                "directory": str(episode_dir),
                "reused": False,
                "metadata": per_prompt.get(prompt, metadata),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", default="", help="Unix socket; falls back to EFFICIENTSAM3_WORKER_SOCK")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True)
    p.add_argument("--prompts-pipe", default="", help="pipe-separated prompts, same as CUP_PROMPTS")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--confidence", type=float, default=0.20)
    p.add_argument("--close-px", type=int, default=0)
    p.add_argument("--dilation-px", type=int, default=0)
    p.add_argument("--temporal-radius", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--packed-rgb", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"missing video: {video}")
    mask_root.mkdir(parents=True, exist_ok=True)
    prompts = [item.strip() for item in args.prompts_pipe.split("|") if item.strip()]
    if not prompts:
        raise SystemExit("no prompts in --prompts-pipe")
    jobs = [(prompt, mask_root / prompt_slug(prompt)) for prompt in prompts]
    sock = resolve_socket(args.socket)
    packed_rgb = args.packed_rgb.expanduser().resolve() if args.packed_rgb is not None else None
    if sock:
        segment_jobs(
            video=video,
            jobs=jobs,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            conf_threshold=args.confidence,
            close_px=args.close_px,
            dilation_px=args.dilation_px,
            temporal_radius=args.temporal_radius,
            batch_size=args.batch_size,
            resolution=args.resolution,
            force=args.force,
            packed_rgb=packed_rgb,
            socket_path=sock,
        )
        records = []
        for prompt, episode_dir in jobs:
            meta_path = episode_dir / "mask_metadata.json"
            records.append(
                {
                    "prompt": prompt,
                    "directory": str(episode_dir),
                    "reused": meta_path.is_file(),
                    "metadata": json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta_path.is_file()
                    else None,
                }
            )
    else:
        records = _run_local(
            video=video,
            jobs=jobs,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            conf_threshold=args.confidence,
            close_px=args.close_px,
            dilation_px=args.dilation_px,
            temporal_radius=args.temporal_radius,
            batch_size=args.batch_size,
            resolution=args.resolution,
            force=args.force,
            device=args.device,
            packed_rgb=packed_rgb,
        )
    manifest = {
        "schema_version": 1,
        "video": str(video),
        "prompts": prompts,
        "socket": sock or None,
        "records": records,
    }
    (mask_root / "sam3_prompt_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "prompts": len(prompts), "socket": bool(sock)}), flush=True)


if __name__ == "__main__":
    main()
