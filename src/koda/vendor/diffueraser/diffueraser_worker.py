#!/usr/bin/env python3
"""Persistent DiffuEraser + ProPainter worker. Load weights once, inpaint many clips."""

from __future__ import annotations

import argparse
import json
import os
import socket
import traceback
from pathlib import Path
from typing import Any

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device


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


class DiffuEraserWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = None
        self.pipeline = None
        self.propainter = None

    def warmup(self) -> None:
        self.device = get_device()
        print(
            f"[diffueraser-worker] loading ckpt={self.args.ckpt} device={self.device}",
            flush=True,
        )
        self.pipeline = DiffuEraser(
            self.device,
            self.args.base_model_path,
            self.args.vae_path,
            self.args.diffueraser_path,
            ckpt=self.args.ckpt,
        )
        self.propainter = Propainter(self.args.propainter_model_dir, device=self.device)
        print("[diffueraser-worker] model ready", flush=True)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = str(request.get("op") or "")
        if op == "ping":
            return {"ok": True, "ready": self.pipeline is not None}
        if op == "shutdown":
            return {"ok": True, "shutdown": True}
        if op != "inpaint":
            raise ValueError(f"unsupported op: {op}")
        if self.pipeline is None or self.propainter is None:
            raise RuntimeError("worker is not ready")
        save_path = Path(str(request["save_path"]))
        save_path.mkdir(parents=True, exist_ok=True)
        priori_path = str(save_path / "priori.mp4")
        output_path = str(save_path / "diffueraser_result.mp4")
        video_length = float(request["video_length"])
        mask_dilation = int(request.get("mask_dilation_iter") or 4)
        keep_priori = bool(request.get("keep_priori", True))
        seed = int(request.get("seed") or 42)
        prior_resize_ratio = float(request.get("prior_resize_ratio", 0.6))
        if not 0 < prior_resize_ratio <= 1:
            raise ValueError("prior_resize_ratio must be in (0, 1]")
        self.propainter.forward(
            str(request["input_video"]),
            str(request["input_mask"]),
            priori_path,
            video_length=video_length,
            ref_stride=int(request.get("ref_stride") or 10),
            neighbor_length=int(request.get("neighbor_length") or 10),
            subvideo_length=int(request.get("subvideo_length") or 50),
            mask_dilation=mask_dilation,
            resize_ratio=prior_resize_ratio,
        )
        self.pipeline.forward(
            str(request["input_video"]),
            str(request["input_mask"]),
            priori_path,
            output_path,
            max_img_size=int(request.get("max_img_size") or 768),
            video_length=video_length,
            mask_dilation_iter=mask_dilation,
            guidance_scale=None,
            seed=seed,
            keep_priori=keep_priori,
        )
        run_manifest = {
            "backend": "DiffuEraser",
            "stages": ["ProPainter", "DiffuEraser diffusion"],
            "checkpoint": self.args.ckpt,
            "num_inference_steps": int(self.pipeline.num_inference_steps),
            "guidance_scale": float(self.pipeline.guidance_scale),
            "max_img_size": int(request.get("max_img_size") or 768),
            "seed": seed,
            "prior_resize_ratio": prior_resize_ratio,
            "priori_path": os.path.abspath(priori_path),
            "priori_preserved": bool(keep_priori and os.path.isfile(priori_path)),
            "output_path": os.path.abspath(output_path),
            "persistent_worker": True,
        }
        (save_path / "diffueraser_run.json").write_text(
            json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
        )
        return {"ok": True, "output_path": os.path.abspath(output_path), "run": run_manifest}


def serve(sock_path: Path, worker: DiffuEraserWorker) -> None:
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(4)
    print(f"[diffueraser-worker] READY socket={sock_path}", flush=True)
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
                    print(f"[diffueraser-worker] job failed: {error}", flush=True)
                send_json(connection, reply)
                if reply.get("shutdown"):
                    print("[diffueraser-worker] shutdown requested", flush=True)
                    return
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ckpt", default="Normal CFG 8-Step")
    parser.add_argument("--base-model-path", default=str(here / "weights/stable-diffusion-v1-5"))
    parser.add_argument("--vae-path", default=str(here / "weights/sd-vae-ft-mse"))
    parser.add_argument("--diffueraser-path", default=str(here / "weights/diffuEraser"))
    parser.add_argument("--propainter-model-dir", default=str(here / "weights/propainter"))
    args = parser.parse_args()
    # DiffuEraser and ProPainter still contain a few cwd-relative assets.
    # Keep the worker in the package root even when the batch cwd is ego2humanoid/.
    os.chdir(here)
    worker = DiffuEraserWorker(args)
    worker.warmup()
    serve(args.socket.expanduser().resolve(), worker)


if __name__ == "__main__":
    main()
