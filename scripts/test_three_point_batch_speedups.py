from __future__ import annotations

import json
import socket
import sys
import threading
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_three_point_batch import (
    JAX_PIPELINE_CONFIG_ID,
    JAX_RUN_NAME,
    MPC_BACKEND_JAX,
    MPC_BACKEND_SIL_TTS,
    SIL_TTS_PIPELINE_CONFIG_ID,
    SIL_TTS_RUN_NAME,
    configure_mpc_backend,
    convert_sil_tts_output_to_jax_layout,
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


def test_mpc_backend_sil_tts_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_three_point_batch.py", "--mpc-backend", "sil-tts", "--dry-run"],
    )
    args = parse_args()
    assert args.mpc_backend == MPC_BACKEND_SIL_TTS


def test_configure_mpc_backend_switches_publish_identity() -> None:
    try:
        assert configure_mpc_backend(MPC_BACKEND_SIL_TTS) == MPC_BACKEND_SIL_TTS
        import run_three_point_batch as batch

        assert batch.RUN_NAME == SIL_TTS_RUN_NAME
        assert batch.PIPELINE_CONFIG_ID == SIL_TTS_PIPELINE_CONFIG_ID
        assert configure_mpc_backend(MPC_BACKEND_JAX) == MPC_BACKEND_JAX
        assert batch.RUN_NAME == JAX_RUN_NAME
        assert batch.PIPELINE_CONFIG_ID == JAX_PIPELINE_CONFIG_ID
    finally:
        configure_mpc_backend(MPC_BACKEND_JAX)


def test_convert_sil_tts_output_copies_mink_layout(tmp_path: Path) -> None:
    sil_out = tmp_path / "sil_tts"
    sil_out.mkdir()
    np.savez_compressed(
        sil_out / "scaletrack_kinematics.npz",
        joint_pos=np.zeros((4, 29), dtype=np.float32),
        joint_vel=np.zeros((4, 29), dtype=np.float32),
        body_pos_w=np.zeros((4, 30, 3), dtype=np.float32),
        body_quat_w=np.zeros((4, 30, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
        fps=np.int64(50),
    )
    (sil_out / "meta.json").write_text(
        json.dumps({"mpc_backend": "sil-tts", "joint_names": ["x"]}) + "\n"
    )
    run_dir = tmp_path / "jax_mpc"
    try:
        configure_mpc_backend(MPC_BACKEND_SIL_TTS)
        convert_sil_tts_output_to_jax_layout(
            sil_output_dir=sil_out,
            run_dir=run_dir,
            reference=tmp_path / "reference.npz",
            init_state=tmp_path / "init.npz",
        )
        kinematics = run_dir / "kinematics" / f"{SIL_TTS_RUN_NAME}_kinematics.npz"
        summary = json.loads(
            (run_dir / f"{SIL_TTS_RUN_NAME}_summary.json").read_text()
        )
        assert kinematics.is_file()
        assert (run_dir / "kinematics" / "meta.json").is_file()
        assert summary["mpc_backend"] == MPC_BACKEND_SIL_TTS
        assert summary["setting"]["recorded_frames"] == 4
        assert summary["setting"]["num_knots"] == 10
    finally:
        configure_mpc_backend(MPC_BACKEND_JAX)
