#!/usr/bin/env python3
"""Persistent EfficientSAM3 worker. Load the TinyViT once, serve clip jobs.

One Unix-socket process per GPU. Contact precompute and human-arm masks send
``segment`` jobs instead of starting a new Python interpreter per prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sam3_efficientsam3_hand_masks import (  # noqa: E402
    DEFAULT_EFF_CKPT,
    load_efficientsam3_image_model,
    run_episode_efficientsam3,
)


def recv_json(connection: socket.socket) -> dict[str, Any]:
    buf = b""
    while b"\n" not in buf:
        chunk = connection.recv(65536)
        if not chunk:
            raise ConnectionError("client closed the connection")
        buf += chunk
    payload = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    return payload


def send_json(connection: socket.socket, payload: dict[str, Any]) -> None:
    connection.sendall((json.dumps(payload, default=str) + "\n").encode("utf-8"))


class EfficientSAM3Worker:
    def __init__(self, *, device: str, conf_threshold: float) -> None:
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.model = None

    def warmup(self) -> None:
        print(
            f"[efficientsam3-worker] loading checkpoint={DEFAULT_EFF_CKPT} device={self.device}",
            flush=True,
        )
        self.model = load_efficientsam3_image_model(
            device=self.device, conf_threshold=self.conf_threshold
        )
        print("[efficientsam3-worker] model ready", flush=True)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = str(request.get("op") or "")
        if op == "ping":
            return {"ok": True, "ready": self.model is not None}
        if op == "shutdown":
            return {"ok": True, "shutdown": True}
        if op != "segment":
            raise ValueError(f"unsupported op: {op}")
        if self.model is None:
            raise RuntimeError("worker is not ready")
        force = bool(request.get("force"))
        packed_rgb = request.get("packed_rgb")
        packed_path = Path(str(packed_rgb)) if packed_rgb else None
        if request.get("prompts"):
            prompts = [str(item) for item in request["prompts"]]
            directories = [Path(str(item)) for item in request["episode_dirs"]]
            if len(prompts) != len(directories):
                raise ValueError("prompts and episode_dirs must have equal length")
            jobs = list(zip(prompts, directories))
        else:
            jobs = [(str(request["prompt"]), Path(str(request["episode_dir"])))]
        pending: list[tuple[str, Path]] = []
        reused: list[dict[str, Any]] = []
        for prompt, episode_dir in jobs:
            meta_file = episode_dir / "mask_metadata.json"
            if meta_file.is_file() and not force:
                reused.append(
                    {
                        "prompt": prompt,
                        "directory": str(episode_dir),
                        "reused": True,
                        "metadata": json.loads(meta_file.read_text(encoding="utf-8")),
                    }
                )
            else:
                pending.append((prompt, episode_dir))
        if not pending:
            return {
                "ok": True,
                "reused": True,
                "records": reused,
                "metadata": reused[0]["metadata"],
            }
        self.model._hdmi_conf = float(request.get("conf_threshold", self.conf_threshold))
        metadata = run_episode_efficientsam3(
            self.model,
            video=Path(str(request["video"])),
            packed_rgb=packed_path,
            jobs=pending,
            start_frame=int(request.get("start_frame") or 0),
            max_frames=int(request.get("max_frames") or 0),
            close_px=int(request.get("close_px") or 0),
            dilation_px=int(request.get("dilation_px") or 0),
            temporal_radius=int(request.get("temporal_radius") or 0),
            conf_threshold=float(request.get("conf_threshold") or self.conf_threshold),
            batch_size=max(1, int(request.get("batch_size") or 16)),
            resolution=max(64, int(request.get("resolution") or 1008)),
            eff_ckpt=DEFAULT_EFF_CKPT,
        )
        return {"ok": True, "reused": False, "metadata": metadata, "records": reused}


def serve(sock_path: Path, worker: EfficientSAM3Worker) -> None:
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(8)
    print(f"[efficientsam3-worker] READY socket={sock_path}", flush=True)
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    request = recv_json(connection)
                    reply = worker.handle(request)
                except Exception as error:
                    reply = {
                        "ok": False,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"[efficientsam3-worker] job failed: {error}", flush=True)
                send_json(connection, reply)
                if reply.get("shutdown"):
                    print("[efficientsam3-worker] shutdown requested", flush=True)
                    return
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.20)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    worker = EfficientSAM3Worker(device=str(args.device), conf_threshold=float(args.confidence))
    worker.warmup()
    serve(args.socket.expanduser().resolve(), worker)


if __name__ == "__main__":
    main()
