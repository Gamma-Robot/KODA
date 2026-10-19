#!/usr/bin/env python3
"""Persistent JAX CEM-MPC worker for production three-point tracking.

Loads the torso-UMI-3 artifact once, JIT-compiles the parameterized N=256 and
N=1 rollouts against a static padded reference, then serves clip jobs over a
Unix socket. Clip length is a traced ``t_max`` plus padded ``ref_pos`` so a
new motion does not retrace.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DIAG_DIR = SCRIPT_DIR / "diagnostics"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DIAG_DIR))

from cem_three_point import CEMConfig  # noqa: E402
from jax_closed_loop_session_3point import (  # noqa: E402
    PRODUCTION_CEM_ITERS,
    PRODUCTION_EXECUTE_STEPS,
    PRODUCTION_HORIZON_STEPS,
    PRODUCTION_NUM_ELITES,
    PRODUCTION_NUM_SAMPLES,
    PRODUCTION_SEED,
    CompiledRollouts,
    _compile_rollouts,
    default_xml_path,
    require_production_cem,
    run_three_point_session,
)
from jax_closed_loop_three_point import DEFAULT_REF_PAD_FRAMES  # noqa: E402
from load_eager_policy import resolve_policy_artifacts  # noqa: E402
from mjx_pi_env_three_point import MjxPiEnvThreePoint  # noqa: E402

DEFAULT_MANIFEST = SCRIPT_DIR / "artifacts_torso_umi3" / "artifact_manifest.json"


def recv_json(connection: socket.socket) -> dict[str, Any]:
    buf = b""
    while b"\n" not in buf:
        chunk = connection.recv(65536)
        if not chunk:
            raise ConnectionError("client closed the connection")
        buf += chunk
    line = buf.split(b"\n", 1)[0]
    payload = json.loads(line.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    return payload


def send_json(connection: socket.socket, payload: dict[str, Any]) -> None:
    connection.sendall((json.dumps(payload, default=str) + "\n").encode("utf-8"))


def write_dummy_motion(path: Path, *, frames: int, n_bodies: int, n_joints: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(frames), 2)
    identity = np.zeros((frames, n_bodies, 4), dtype=np.float64)
    identity[..., 0] = 1.0
    np.savez(
        path,
        fps=np.asarray(50.0, dtype=np.float64),
        joint_pos=np.zeros((frames, n_joints), dtype=np.float64),
        joint_vel=np.zeros((frames, n_joints), dtype=np.float64),
        body_pos_w=np.zeros((frames, n_bodies, 3), dtype=np.float64),
        body_quat_w=identity,
        body_lin_vel_w=np.zeros((frames, n_bodies, 3), dtype=np.float64),
        body_ang_vel_w=np.zeros((frames, n_bodies, 3), dtype=np.float64),
    )


class JaxMpcWorker:
    def __init__(
        self,
        *,
        artifact_manifest: Path,
        xml_path: Path,
        pad_frames: int,
        torch_device: str = "cuda",
    ) -> None:
        self.artifact_manifest = artifact_manifest.expanduser().resolve()
        self.artifacts = self.artifact_manifest.parent
        self.xml_path = xml_path.expanduser().resolve()
        self.pad_frames = int(pad_frames)
        self.torch_device = torch_device
        self.env: MjxPiEnvThreePoint | None = None
        self.compiled: CompiledRollouts | None = None
        self.ready = False
        self.cfg = CEMConfig(
            num_samples=PRODUCTION_NUM_SAMPLES,
            num_elites=PRODUCTION_NUM_ELITES,
            horizon_steps=PRODUCTION_HORIZON_STEPS,
            execute_steps=PRODUCTION_EXECUTE_STEPS,
        )

    def warmup(self, work_dir: Path) -> None:
        spec = resolve_policy_artifacts(
            self.artifacts,
            self.artifact_manifest,
            verify_checkpoint=False,
        )
        metadata = json.loads(spec.metadata_path.read_text())
        dummy = work_dir / "warmup_motion.npz"
        write_dummy_motion(
            dummy,
            frames=2,
            n_bodies=len(metadata["body_names"]),
            n_joints=len(metadata["joint_names"]),
        )
        print(
            f"[jax-mpc-worker] warmup env pad_frames={self.pad_frames} "
            f"manifest={self.artifact_manifest}",
            flush=True,
        )
        self.env = MjxPiEnvThreePoint(
            self.artifacts,
            str(self.xml_path),
            str(dummy),
            artifact_manifest=self.artifact_manifest,
            mode_index=0,
            torch_device=self.torch_device,
        )
        self.env.reset_rsi()
        self.compiled = _compile_rollouts(
            self.env, self.cfg, pad_frames=self.pad_frames
        )
        self.ready = True
        print(
            "[jax-mpc-worker] warmup compile_timing="
            + json.dumps(self.compiled.compile_timing),
            flush=True,
        )

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = str(request.get("cmd", "run"))
        if cmd == "ping":
            return {
                "ok": True,
                "ready": bool(self.ready),
                "pad_frames": self.pad_frames,
            }
        if cmd == "shutdown":
            return {"ok": True, "shutdown": True}
        if cmd != "run":
            raise ValueError(f"unsupported cmd: {cmd}")
        if not self.ready or self.env is None or self.compiled is None:
            raise RuntimeError("worker is not ready")
        seed = int(request.get("seed", PRODUCTION_SEED))
        num_samples = int(request.get("num_samples", PRODUCTION_NUM_SAMPLES))
        horizon_steps = int(request.get("horizon_steps", PRODUCTION_HORIZON_STEPS))
        execute_steps = int(request.get("execute_steps", PRODUCTION_EXECUTE_STEPS))
        cem_iters = int(request.get("cem_iters", PRODUCTION_CEM_ITERS))
        require_production_cem(
            seed=seed,
            num_samples=num_samples,
            horizon_steps=horizon_steps,
            execute_steps=execute_steps,
            cem_iters=cem_iters,
        )
        motion_file = Path(request["motion_file"])
        init_state_npz = Path(request["init_state_npz"])
        output_dir = Path(request["output_dir"])
        manifest = Path(request.get("artifact_manifest", self.artifact_manifest))
        if manifest.expanduser().resolve() != self.artifact_manifest:
            raise ValueError(
                f"artifact_manifest mismatch: {manifest} != {self.artifact_manifest}"
            )
        steps = request.get("steps")
        result, self.env, self.compiled = run_three_point_session(
            motion_file=motion_file,
            init_state_npz=init_state_npz,
            artifact_manifest=self.artifact_manifest,
            output_dir=output_dir,
            seed=seed,
            num_samples=num_samples,
            horizon_steps=horizon_steps,
            execute_steps=execute_steps,
            cem_iters=cem_iters,
            steps=None if steps is None else int(steps),
            xml_path=self.xml_path,
            torch_device=self.torch_device,
            env=self.env,
            compiled=self.compiled,
            pad_frames=self.pad_frames,
            require_production=True,
        )
        if self.compiled.pad_frames is not None:
            self.pad_frames = int(self.compiled.pad_frames)
        summary_path = output_dir / (
            f"jax_mpc_3point_n{num_samples}_h{horizon_steps}_i{cem_iters}_"
            f"x{execute_steps}_seed{seed}_summary.json"
        )
        return {
            "ok": True,
            "summary_path": str(summary_path),
            "compile_timing": result.get("compile_timing"),
            "startup_s": result.get("startup_s"),
        }


def serve(sock_path: Path, worker: JaxMpcWorker) -> None:
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(8)
    print(f"[jax-mpc-worker] READY socket={sock_path}", flush=True)
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
                    print(f"[jax-mpc-worker] job failed: {error}", flush=True)
                send_json(connection, reply)
                if reply.get("shutdown"):
                    print("[jax-mpc-worker] shutdown requested", flush=True)
                    return
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--pad-frames", type=int, default=DEFAULT_REF_PAD_FRAMES)
    parser.add_argument("--torch-device", default="cuda")
    args = parser.parse_args()
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[jax-mpc-worker] JAX_COMPILATION_CACHE_DIR={cache_dir}", flush=True)
    worker = JaxMpcWorker(
        artifact_manifest=args.artifact_manifest,
        xml_path=args.xml if args.xml is not None else default_xml_path(),
        pad_frames=int(args.pad_frames),
        torch_device=str(args.torch_device),
    )
    with tempfile.TemporaryDirectory(prefix="jax_mpc_worker.") as tmp:
        worker.warmup(Path(tmp))
        serve(args.socket.expanduser().resolve(), worker)


if __name__ == "__main__":
    main()
