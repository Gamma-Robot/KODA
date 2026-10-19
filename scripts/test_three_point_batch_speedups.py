from __future__ import annotations

import json
import socket
import sys
import threading
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_three_point_batch import (
    MPC_BACKEND_JAX,
    json_socket_request,
    parse_args,
    should_run_parity,
)


def test_should_run_parity_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THREE_POINT_RUN_PARITY", raising=False)
    assert should_run_parity(Namespace(run_parity=False)) is False
    assert should_run_parity() is False


def test_should_run_parity_flag_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THREE_POINT_RUN_PARITY", raising=False)
    assert should_run_parity(Namespace(run_parity=True)) is True
    monkeypatch.setenv("THREE_POINT_RUN_PARITY", "1")
    assert should_run_parity(Namespace(run_parity=False)) is True
    monkeypatch.setenv("THREE_POINT_RUN_PARITY", "false")
    assert should_run_parity(Namespace(run_parity=False)) is False


def test_json_socket_request_roundtrip(tmp_path: Path) -> None:
    sock_path = tmp_path / "worker.sock"
    received: dict[str, object] = {}

    def serve() -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)
        server.settimeout(5)
        connection, _ = server.accept()
        with connection:
            buf = b""
            while b"\n" not in buf:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                buf += chunk
            received.update(json.loads(buf.split(b"\n", 1)[0].decode()))
            connection.sendall(b'{"ok": true, "ready": true}\n')
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    for _ in range(50):
        if sock_path.exists():
            break
        thread.join(0.05)
    reply = json_socket_request(sock_path, {"cmd": "ping"}, timeout_s=5)
    thread.join(timeout=5)
    assert received == {"cmd": "ping"}
    assert reply == {"ok": True, "ready": True}


def test_mpc_backend_default_is_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_three_point_batch.py", "--dry-run"])
    args = parse_args()
    assert args.mpc_backend == MPC_BACKEND_JAX
