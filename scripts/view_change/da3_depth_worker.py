#!/usr/bin/env python3
"""Persistent DA3 depth worker. Load the metric model once, infer packed volumes."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from composite_st_depth_overlay import da3_depth, load_da3  # noqa: E402
from frame_volume import as_rgb, create_volume, open_volume  # noqa: E402


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


class DA3DepthWorker:
    def __init__(self, *, model_dir: str, da3_src: Path, device: str) -> None:
        self.model_dir = str(model_dir)
        self.da3_src = Path(da3_src)
        self.device = str(device)
        self.model = None

    def warmup(self) -> None:
        print(
            f"[da3-depth-worker] loading {self.model_dir} device={self.device}",
            flush=True,
        )
        self.model = load_da3(self.model_dir, self.da3_src, self.device)
        print("[da3-depth-worker] model ready", flush=True)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = str(request.get("op") or "")
        if op == "ping":
            return {"ok": True, "ready": self.model is not None}
        if op == "shutdown":
            return {"ok": True, "shutdown": True}
        if op != "infer_volume":
            raise ValueError(f"unsupported op: {op}")
        if self.model is None:
            raise RuntimeError("worker is not ready")
        rgb_dir = Path(str(request["rgb_dir"]))
        out_dir = Path(str(request["out_dir"]))
        process_res = int(request.get("process_res") or 504)
        fx = request.get("fx")
        fy = request.get("fy")
        metric_div = request.get("metric_focal_div")
        start = int(request.get("start") or 0)
        rgb_vol = open_volume(rgb_dir, color_space="rgb")
        n = len(rgb_vol)
        end = int(request.get("end") or n)
        end = min(end, n)
        if end <= start:
            raise ValueError(f"empty infer range {start}:{end}")
        first = as_rgb(rgb_vol[start])
        height, width = first.shape[:2]
        depth_vol = create_volume(
            out_dir, n, (height, width), np.float32, fill=0
        )
        try:
            for frame_i in range(start, end):
                image = as_rgb(rgb_vol[frame_i])[:, :, :3]
                depth = da3_depth(
                    self.model,
                    image,
                    process_res,
                    fx=None if fx is None else float(fx),
                    fy=None if fy is None else float(fy),
                    metric_focal_div=(
                        None if metric_div is None else float(metric_div)
                    ),
                )
                if depth.shape[:2] != (height, width):
                    import cv2

                    depth = cv2.resize(
                        np.asarray(depth, dtype=np.float32),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                depth_vol[frame_i] = np.asarray(depth, dtype=np.float32)
        finally:
            depth_vol.flush()
        return {
            "ok": True,
            "out_dir": str(out_dir.resolve()),
            "frames": int(end - start),
            "height": int(height),
            "width": int(width),
        }


def serve(sock_path: Path, worker: DA3DepthWorker) -> None:
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(8)
    print(f"[da3-depth-worker] READY socket={sock_path}", flush=True)
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
                    print(f"[da3-depth-worker] job failed: {error}", flush=True)
                send_json(connection, reply)
                if reply.get("shutdown"):
                    print("[da3-depth-worker] shutdown requested", flush=True)
                    return
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--da3-model-dir",
        default=str(
            Path(os.environ.get("KODA_ASSETS", str(Path(__file__).resolve().parents[2] / "assets")))
            / "models/depth_anything_3"
        ),
    )
    parser.add_argument(
        "--da3-src",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "resources/Depth-Anything-3/src",
    )
    args = parser.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    worker = DA3DepthWorker(
        model_dir=str(args.da3_model_dir),
        da3_src=args.da3_src,
        device=str(args.device),
    )
    worker.warmup()
    serve(args.socket.expanduser().resolve(), worker)


if __name__ == "__main__":
    main()
