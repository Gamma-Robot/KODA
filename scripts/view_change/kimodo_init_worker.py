#!/usr/bin/env python3
"""Persistent Kimodo init worker. Load Kimodo-G1-RP-v1 once, serve generate jobs.

One Unix-socket process per GPU. Tracking sessions send ``generate`` jobs
instead of starting a new Python interpreter per clip. IK / physics gating
stay in ``run_clean_kimodo_init_sweep.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

WORKER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKER_DIR.parents[1]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
ASSET_ROOT = Path(os.environ.get("KODA_ASSETS", str(PROJECT_ROOT / "assets"))).expanduser().resolve()
SWEEP_DIR = RESOURCE_ROOT / "ScaleBFM" / "ScaleTrack" / "scripts" / "eval_session"
KIMODO_ROOT = Path(os.environ.get("KODA_KIMODO_ROOT", str(Path(__file__).resolve().parents[2] / "resources/kimodo")))
if not KIMODO_ROOT.is_dir():
    KIMODO_ROOT = RESOURCE_ROOT / "kimodo"

for path in (SWEEP_DIR, KIMODO_ROOT):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("CHECKPOINT_DIR", str(ASSET_ROOT / "models/kimodo"))
os.environ.setdefault("LOCAL_CACHE", "True")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODERS_DIR", str(ASSET_ROOT / "models/kimodo/text_encoders"))
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")

from run_clean_kimodo_init_sweep import (  # noqa: E402
    generate_candidate,
    load_kimodo_model,
    parse_seeds,
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


class KimodoInitWorker:
    def __init__(self, *, model_name: str) -> None:
        self.model_name = str(model_name)
        self.model = None
        self.converter = None
        self.device = None

    def warmup(self) -> None:
        print(
            f"[kimodo-init-worker] loading model={self.model_name}",
            flush=True,
        )
        self.model, self.converter, self.device = load_kimodo_model(self.model_name)
        print(f"[kimodo-init-worker] model ready device={self.device}", flush=True)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = str(request.get("op") or "")
        if op == "ping":
            return {"ok": True, "ready": self.model is not None}
        if op == "shutdown":
            return {"ok": True, "shutdown": True}
        if op != "generate":
            raise ValueError(f"unsupported op: {op}")
        if self.model is None or self.converter is None:
            raise RuntimeError("worker is not ready")
        session_npz = Path(str(request["session_npz"]))
        out_dir = Path(str(request["out_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = parse_seeds(str(request.get("seeds") or "2"))
        duration_s = float(request.get("duration") or 10.0)
        diffusion_steps = int(request.get("diffusion_steps") or 100)
        cfg = request.get("cfg_weight") or [2.0, 5.0]
        if isinstance(cfg, str):
            cfg_weight = [float(part) for part in cfg.replace(",", " ").split() if part]
        else:
            cfg_weight = [float(cfg[0]), float(cfg[1])]
        if len(cfg_weight) != 2:
            raise ValueError("cfg_weight must have two values")
        torso_init = bool(request.get("torso_init_condition"))
        generated = []
        for seed in seeds:
            _, csv_path, meta = generate_candidate(
                model=self.model,
                converter=self.converter,
                device=self.device,
                session_npz=session_npz,
                seed=int(seed),
                out_dir=out_dir,
                duration_s=duration_s,
                diffusion_steps=diffusion_steps,
                cfg_weight=cfg_weight,
                model_name=self.model_name,
                torso_init_condition=torso_init,
            )
            generated.append({"seed": int(seed), "csv": str(csv_path), "meta": meta})
        return {"ok": True, "generated": generated}


def serve(sock_path: Path, worker: KimodoInitWorker) -> None:
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(8)
    print(f"[kimodo-init-worker] READY socket={sock_path}", flush=True)
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
                    print(f"[kimodo-init-worker] job failed: {error}", flush=True)
                send_json(connection, reply)
                if reply.get("shutdown"):
                    print("[kimodo-init-worker] shutdown requested", flush=True)
                    return
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--model-name", default="Kimodo-G1-RP-v1")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    worker = KimodoInitWorker(model_name=str(args.model_name))
    worker.warmup()
    serve(args.socket.expanduser().resolve(), worker)


if __name__ == "__main__":
    main()
