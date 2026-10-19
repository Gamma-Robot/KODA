"""Encode SONIC motion tokens from source kinematics without simulation rollout.

Assumes perfect tracking: at each step the robot pose equals kinematics[t],
so ``observation.state`` later exported via ``export-groot`` matches the source
joint trajectory (hands still come from source ``hand_qpos``).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sil_tts.artifacts import default_repo_root
from sil_tts.motion_tracking.kinematics_npz import (
    iter_session_kinematics,
    write_optimized_kinematics,
)
from sil_trackers.families.sonic.observation import (
    heading_offset,
    pack_g1_input,
    pack_low_latency_encoder_obs,
)
from sil_trackers.families.sonic.variants import LOW_LATENCY

OPTIMIZED_KINEMATICS_FILE = "optimized_kinematics.npz"
QUEUE_FILE = "encode_queue.txt"
QUEUE_LOCK_FILE = "encode_queue.lock"
QUEUE_SPEC_FILE = "encode_queue_spec.json"
QUEUE_SUMMARY_FILE = "encode_queue_summary.json"
FAILURE_FILE = "failure.json"
STOP_FILE = "STOP"
DEFAULT_ENCODER_RELATIVE = (
    Path("artifacts") / "store" / "sonic" / "low_latency" / "g1" / "model_encoder.onnx"
)
QUERY_OFFSETS = LOW_LATENCY.query_offsets
_KINEMATICS_COPY_KEYS = (
    "timestamps",
    "root_pos_w",
    "root_quat_wxyz",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "joint_pos",
    "joint_vel",
    "hand_qpos",
    "hand_qvel",
)


@dataclass(frozen=True)
class EncodeAssignment:
    motion_id: str
    source_npz: str
    output_dir: str
    action: str  # "run" | "skip"


def default_encoder_path(repo: Path | None = None) -> Path:
    root = default_repo_root() if repo is None else Path(repo)
    return (root / DEFAULT_ENCODER_RELATIVE).resolve()


def optimized_kinematics_is_complete(output_dir: str | Path) -> bool:
    path = Path(output_dir) / OPTIMIZED_KINEMATICS_FILE
    if not path.is_file():
        return False
    try:
        with np.load(path) as payload:
            token = np.asarray(payload["motion_token"])
            frames = int(np.asarray(payload["timestamps"]).shape[0])
    except Exception:
        return False
    return token.ndim == 2 and token.shape == (frames, 64)


def _string_names(values: object, field: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{field} must be a non-empty 1-D name vector")
    names = tuple(str(item) for item in array.tolist())
    if any(not name for name in names):
        raise ValueError(f"{field} contains an empty name")
    return names


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_failure(
    output_dir: Path,
    *,
    motion_id: str,
    gpu: int | None,
    error: str,
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / FAILURE_FILE,
        {
            "motion_id": motion_id,
            "gpu": gpu,
            "error": error,
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def build_encoder_obs_batch(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_quat_wxyz: np.ndarray,
    step_indices: Sequence[int],
    query_offsets: tuple[int, ...] = QUERY_OFFSETS,
    frozen_heading_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack low-latency encoder obs for selected time indices.

    Robot pose at ``t`` equals kinematics[t] (no rollout). Heading offset is
    frozen from the first encoded step, matching the online SONIC tracker.
    """

    pos = np.asarray(joint_pos, dtype=np.float32)
    vel = np.asarray(joint_vel, dtype=np.float32)
    quat = np.asarray(root_quat_wxyz, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 29:
        raise ValueError(f"joint_pos must have shape (T, 29), got {pos.shape}")
    if vel.shape != pos.shape:
        raise ValueError("joint_vel must match joint_pos shape")
    if quat.shape != (pos.shape[0], 4):
        raise ValueError(f"root_quat_wxyz must have shape (T, 4), got {quat.shape}")
    if not query_offsets:
        raise ValueError("query_offsets must be non-empty")
    frames = int(pos.shape[0])
    steps = [int(index) for index in step_indices]
    if any(index < 0 or index >= frames for index in steps):
        raise ValueError("step_indices must lie in [0, T)")
    if frozen_heading_offset is None:
        first = steps[0]
        first_queries = np.clip(
            first + np.asarray(query_offsets, dtype=np.int64), 0, frames - 1
        )
        frozen_heading_offset = heading_offset(
            quat[first : first + 1], quat[first_queries][None, :, :][:, 0]
        )
    heading = np.asarray(frozen_heading_offset, dtype=np.float32)
    if heading.shape != (1, 4):
        raise ValueError(f"frozen_heading_offset must have shape (1, 4), got {heading.shape}")

    batch = len(steps)
    obs = np.zeros((batch, 1247), dtype=np.float32)
    for row, step in enumerate(steps):
        query_idx = np.clip(
            step + np.asarray(query_offsets, dtype=np.int64), 0, frames - 1
        )
        g1 = pack_g1_input(
            pos[query_idx][None],
            vel[query_idx][None],
            quat[query_idx][None],
            quat[step : step + 1],
            heading,
            query_offsets=query_offsets,
        )
        obs[row] = pack_low_latency_encoder_obs(g1)[0]
    return obs, heading


class SonicLowLatencyEncoder:
    """Thin ONNX Runtime wrapper around the official G1 encoder graph."""

    def __init__(self, encoder_path: str | Path, *, providers: Sequence[str] | None = None):
        import onnxruntime as ort

        path = Path(encoder_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SONIC encoder ONNX missing: {path}")
        available = ort.get_available_providers()
        chosen = list(providers) if providers is not None else []
        if not chosen:
            if "CUDAExecutionProvider" in available:
                chosen.append("CUDAExecutionProvider")
            chosen.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(str(path), providers=chosen)
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs_dict":
            raise ValueError("encoder must expose a single obs_dict input")
        if len(outputs) != 1 or outputs[0].name != "encoded_tokens":
            raise ValueError("encoder must expose a single encoded_tokens output")
        self._input_name = inputs[0].name
        self.path = path

    def encode(self, obs: np.ndarray) -> np.ndarray:
        value = np.asarray(obs, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 1247:
            raise ValueError(f"encoder obs must have shape (B, 1247), got {value.shape}")
        tokens = np.empty((value.shape[0], 64), dtype=np.float32)
        # Official exported graph is fixed to batch=1.
        for index in range(value.shape[0]):
            out = self._session.run(
                None, {self._input_name: value[index : index + 1]}
            )[0]
            token = np.asarray(out, dtype=np.float32)
            if token.shape != (1, 64):
                raise ValueError(f"encoded_tokens must have shape (1, 64), got {token.shape}")
            tokens[index] = token[0]
        return tokens


def kinematics_arrays_with_tokens(
    source: Mapping[str, np.ndarray],
    motion_token: np.ndarray,
) -> dict[str, np.ndarray]:
    """Copy source kinematics as state and bind ``motion_token`` (length T-1)."""

    token = np.asarray(motion_token)
    if token.ndim != 2 or token.shape[1] != 64:
        raise ValueError(f"motion_token must have shape (T, 64), got {token.shape}")
    joint_pos = np.asarray(source["joint_pos"])
    source_frames = int(joint_pos.shape[0])
    rows = int(token.shape[0])
    if source_frames != rows + 1:
        raise ValueError(
            f"source must have T+1 frames for T tokens, got source={source_frames} "
            f"tokens={rows}"
        )
    if source_frames < 2:
        raise ValueError("source kinematics need at least 2 frames")
    fps = int(np.asarray(source["fps"]))
    arrays: dict[str, np.ndarray] = {
        "fps": np.int64(fps),
        "joint_names": np.asarray(_string_names(source["joint_names"], "joint_names")),
        "motion_token": np.ascontiguousarray(token),
    }
    for key in _KINEMATICS_COPY_KEYS:
        if key not in source:
            if key.startswith("hand_"):
                continue
            raise ValueError(f"source kinematics missing {key}")
        value = np.asarray(source[key])
        if value.shape[0] != source_frames:
            raise ValueError(f"{key} length must match joint_pos")
        arrays[key] = np.ascontiguousarray(value[:rows])
    if "hand_joint_names" in source and "hand_qpos" in arrays:
        arrays["hand_joint_names"] = np.asarray(
            _string_names(source["hand_joint_names"], "hand_joint_names")
        )
    return arrays


def encode_session_kinematics(
    source_npz: str | Path,
    output_dir: str | Path,
    *,
    encoder: SonicLowLatencyEncoder | None = None,
    encoder_path: str | Path | None = None,
    query_offsets: tuple[int, ...] = QUERY_OFFSETS,
    chunk_size: int = 64,
    overwrite_incomplete: bool = True,
) -> Path:
    """Encode one ``final_50hz/kinematics.npz`` into ``optimized_kinematics.npz``."""

    source_path = Path(source_npz).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    out_file = destination / OPTIMIZED_KINEMATICS_FILE
    if optimized_kinematics_is_complete(destination):
        return out_file
    if destination.exists() and any(destination.iterdir()):
        if not overwrite_incomplete:
            raise FileExistsError(
                f"refusing to overwrite non-empty encode directory: {destination}"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with np.load(source_path, allow_pickle=False) as payload:
        source = {key: payload[key] for key in payload.files}
    frames = int(np.asarray(source["joint_pos"]).shape[0])
    if frames < 2:
        raise ValueError(f"{source_path} needs at least 2 frames, got {frames}")
    rows = frames - 1
    runtime = encoder
    if runtime is None:
        runtime = SonicLowLatencyEncoder(
            encoder_path if encoder_path is not None else default_encoder_path()
        )

    tokens = np.empty((rows, 64), dtype=np.float32)
    frozen_heading: np.ndarray | None = None
    for start in range(0, rows, int(chunk_size)):
        stop = min(rows, start + int(chunk_size))
        obs, frozen_heading = build_encoder_obs_batch(
            joint_pos=source["joint_pos"],
            joint_vel=source["joint_vel"],
            root_quat_wxyz=source["root_quat_wxyz"],
            step_indices=range(start, stop),
            query_offsets=query_offsets,
            frozen_heading_offset=frozen_heading,
        )
        tokens[start:stop] = runtime.encode(obs)

    arrays = kinematics_arrays_with_tokens(source, tokens)
    return write_optimized_kinematics(out_file, arrays)


def plan_encode_runs(
    *,
    sessions: Sequence[tuple[str, Path]],
    output_root: str | Path,
    skip_existing: bool = True,
) -> list[EncodeAssignment]:
    output = Path(output_root)
    assignments: list[EncodeAssignment] = []
    for motion_id, source_npz in sessions:
        clip_dir = output / motion_id
        if skip_existing and optimized_kinematics_is_complete(clip_dir):
            action = "skip"
        else:
            action = "run"
        assignments.append(
            EncodeAssignment(
                motion_id=motion_id,
                source_npz=str(source_npz),
                output_dir=str(clip_dir),
                action=action,
            )
        )
    return assignments


def parse_gpu_list(raw: str) -> tuple[int, ...]:
    parts = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not parts:
        raise ValueError("gpus must be a non-empty comma-separated list")
    return tuple(int(item) for item in parts)


def pop_queued_session(queue_path: str | Path, lock_path: str | Path) -> str | None:
    queue = Path(queue_path)
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if not queue.is_file():
            return None
        remaining = [
            line.strip()
            for line in queue.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not remaining:
            return None
        claimed, rest = remaining[0], remaining[1:]
        queue.write_text("".join(f"{item}\n" for item in rest), encoding="utf-8")
        return claimed


def _continue_on_error_from_env() -> bool:
    raw = os.environ.get("CONTINUE_ON_ERROR", "0").strip().lower()
    return raw not in {"", "0", "false", "no"}


def encode_kinematics_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    gpus: Sequence[int] | None = None,
    encoder_path: str | Path | None = None,
    dry_run: bool = False,
    continue_on_error: bool | None = None,
    motion_ids: Sequence[str] | None = None,
    skip_existing: bool = True,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Encode all ``session_*/final_50hz/kinematics.npz`` under ``source_root``."""

    source = Path(source_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    encoder = (
        Path(encoder_path).expanduser().resolve()
        if encoder_path is not None
        else default_encoder_path()
    )
    if not encoder.is_file():
        raise FileNotFoundError(f"SONIC encoder ONNX missing: {encoder}")

    sessions = list(iter_session_kinematics(source))
    if motion_ids is not None:
        wanted = set(motion_ids)
        sessions = [item for item in sessions if item[0] in wanted]
        missing = sorted(wanted.difference(item[0] for item in sessions))
        if missing:
            raise KeyError(f"motion ids missing under source-root: {missing}")
    if not sessions:
        raise ValueError(f"no session_*/final_50hz/kinematics.npz under {source}")

    assignments = plan_encode_runs(
        sessions=sessions, output_root=destination, skip_existing=skip_existing
    )
    continue_errors = (
        _continue_on_error_from_env() if continue_on_error is None else bool(continue_on_error)
    )
    pending = [item for item in assignments if item.action == "run"]
    summary: dict[str, Any] = {
        "source_root": str(source),
        "output_root": str(destination),
        "encoder": str(encoder),
        "query_offsets": list(QUERY_OFFSETS),
        "gpus": list(gpus) if gpus is not None else [],
        "continue_on_error": continue_errors,
        "dry_run": bool(dry_run),
        "chunk_size": int(chunk_size),
        "assignments": [asdict(item) for item in assignments],
        "num_run": len(pending),
        "num_skip": sum(1 for item in assignments if item.action == "skip"),
        "results": [],
        "mode": "kinematics_as_state_no_rollout",
    }
    if dry_run or not pending:
        summary["success"] = True
        _write_json(destination / QUEUE_SUMMARY_FILE, summary)
        return summary

    gpu_ids = tuple(int(gpu) for gpu in (gpus if gpus is not None else (0,)))
    queue_path = destination / QUEUE_FILE
    lock_path = destination / QUEUE_LOCK_FILE
    spec_path = destination / QUEUE_SPEC_FILE
    stop_path = destination / STOP_FILE
    if stop_path.exists():
        stop_path.unlink()
    queue_path.write_text(
        "".join(f"{item.motion_id}\n" for item in pending), encoding="utf-8"
    )
    by_id = {item.motion_id: item for item in pending}
    _write_json(
        spec_path,
        {
            "source_root": str(source),
            "output_root": str(destination),
            "encoder": str(encoder),
            "query_offsets": list(QUERY_OFFSETS),
            "chunk_size": int(chunk_size),
            "continue_on_error": continue_errors,
            "sessions": {
                motion_id: {
                    "source_npz": item.source_npz,
                    "output_dir": item.output_dir,
                }
                for motion_id, item in by_id.items()
            },
            "python": sys.executable,
        },
    )

    if len(gpu_ids) == 1:
        code = encode_worker(
            gpu=gpu_ids[0],
            queue_path=queue_path,
            lock_path=lock_path,
            spec_path=spec_path,
            stop_path=stop_path,
        )
        exit_codes = [code]
    else:
        workers: list[subprocess.Popen[bytes]] = []
        try:
            for gpu in gpu_ids:
                workers.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "sil_tts.encode_kinematics_tokens",
                            "--worker-gpu",
                            str(gpu),
                            "--queue",
                            str(queue_path),
                            "--lock",
                            str(lock_path),
                            "--spec",
                            str(spec_path),
                            "--stop",
                            str(stop_path),
                        ]
                    )
                )
            exit_codes = [process.wait() for process in workers]
        except Exception:
            for process in workers:
                process.terminate()
            raise

    results = []
    for item in assignments:
        output_dir = Path(item.output_dir)
        if item.action == "skip" or optimized_kinematics_is_complete(output_dir):
            status = "skipped" if item.action == "skip" else "success"
            results.append(
                {
                    "motion_id": item.motion_id,
                    "status": status,
                    "output_dir": item.output_dir,
                    "optimized_kinematics": str(output_dir / OPTIMIZED_KINEMATICS_FILE),
                }
            )
            continue
        failure = output_dir / FAILURE_FILE
        results.append(
            {
                "motion_id": item.motion_id,
                "status": "failed" if failure.is_file() else "pending",
                "output_dir": item.output_dir,
                "failure": str(failure) if failure.is_file() else None,
            }
        )
    summary["results"] = results
    summary["worker_exit_codes"] = exit_codes
    summary["success"] = all(code == 0 for code in exit_codes) and all(
        row["status"] != "failed" for row in results
    )
    _write_json(destination / QUEUE_SUMMARY_FILE, summary)
    return summary


def encode_worker(
    *,
    gpu: int,
    queue_path: str | Path,
    lock_path: str | Path,
    spec_path: str | Path,
    stop_path: str | Path,
) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    sessions = spec["sessions"]
    encoder = SonicLowLatencyEncoder(spec["encoder"])
    query_offsets = tuple(int(value) for value in spec["query_offsets"])
    chunk_size = int(spec.get("chunk_size", 64))
    continue_on_error = bool(spec.get("continue_on_error", False))
    stop = Path(stop_path)
    failed = False
    while True:
        if stop.is_file():
            return 1 if failed else 0
        motion_id = pop_queued_session(queue_path, lock_path)
        if motion_id is None:
            return 1 if failed else 0
        item = sessions[motion_id]
        output_dir = Path(item["output_dir"])
        if optimized_kinematics_is_complete(output_dir):
            continue
        started = time.time()
        try:
            encode_session_kinematics(
                item["source_npz"],
                output_dir,
                encoder=encoder,
                query_offsets=query_offsets,
                chunk_size=chunk_size,
                overwrite_incomplete=True,
            )
        except Exception as exc:
            failed = True
            _write_failure(
                output_dir,
                motion_id=motion_id,
                gpu=int(gpu),
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.time() - started,
            )
            if not continue_on_error:
                stop.write_text(f"gpu={gpu} motion_id={motion_id}\n", encoding="utf-8")
                return 1
    return 1 if failed else 0


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sil_tts.encode_kinematics_tokens")
    parser.add_argument("--worker-gpu", type=int, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _worker_parser().parse_args(argv)
    return encode_worker(
        gpu=args.worker_gpu,
        queue_path=args.queue,
        lock_path=args.lock,
        spec_path=args.spec,
        stop_path=args.stop,
    )


if __name__ == "__main__":
    raise SystemExit(main())
