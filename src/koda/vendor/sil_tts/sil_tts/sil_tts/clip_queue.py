"""Flock-backed multi-GPU queue: one ingested clip per GPU worker."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from sil_tts.artifacts import default_repo_root
from sil_tts.motion_tracking.kinematics_npz import (
    DEFAULT_ROBOT_RELATIVE,
    load_clip_durations,
)

OPTIMIZED_KINEMATICS_FILE = "optimized_kinematics.npz"
INGESTED_DATASET_DIR = "_ingested"
QUEUE_FILE = "clip_queue.txt"
QUEUE_LOCK_FILE = "clip_queue.lock"
QUEUE_SPEC_FILE = "clip_queue_spec.json"
QUEUE_SUMMARY_FILE = "clip_queue_summary.json"
FAILURE_FILE = "failure.json"
STOP_FILE = "STOP"
_REQUIRED_KINEMATICS = {
    "fps",
    "timestamps",
    "root_pos_w",
    "root_quat_wxyz",
    "joint_pos",
    "joint_vel",
    "joint_names",
    "motion_token",
}


@dataclass(frozen=True)
class ClipAssignment:
    motion_id: str
    gpu: int
    clip_duration_seconds: float
    output_dir: str
    action: str


def experiment_robot_yaml(
    experiment_path: str | Path,
    *,
    configs_root: str | Path | None = None,
) -> Path:
    """Resolve ``robot.ref`` from an experiment YAML, else the Dex3 default."""

    path = Path(experiment_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment config must be a mapping: {path}")
    robot = payload.get("robot")
    ref = robot.get("ref") if isinstance(robot, dict) else None
    if isinstance(ref, str) and ref:
        root = (
            Path(configs_root).expanduser().resolve()
            if configs_root is not None
            else default_repo_root() / "configs"
        )
        resolved = (root / ref).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"experiment robot.ref does not exist: {resolved}")
        return resolved
    return default_repo_root() / DEFAULT_ROBOT_RELATIVE


def parse_gpu_list(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("gpus must list at least one device index")
    gpus: list[int] = []
    for part in parts:
        try:
            gpu = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid GPU index {part!r}") from exc
        if gpu < 0:
            raise ValueError("GPU indices must be non-negative")
        gpus.append(gpu)
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU indices must be unique")
    return tuple(gpus)


def optimized_kinematics_is_complete(output_dir: str | Path) -> bool:
    path = Path(output_dir) / OPTIMIZED_KINEMATICS_FILE
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not _REQUIRED_KINEMATICS.issubset(payload.files):
                return False
            token = np.asarray(payload["motion_token"])
            frames = int(np.asarray(payload["timestamps"]).shape[0])
    except (OSError, ValueError, KeyError):
        return False
    return token.ndim == 2 and token.shape == (frames, 64) and frames >= 1


def plan_clip_runs(
    *,
    motion_ids: Sequence[str],
    durations: Mapping[str, float],
    gpus: Sequence[int],
    output_root: str | Path,
    skip_existing: bool = True,
) -> tuple[ClipAssignment, ...]:
    if not gpus:
        raise ValueError("gpus must list at least one device index")
    output = Path(output_root)
    assignments: list[ClipAssignment] = []
    pending_index = 0
    for motion_id in motion_ids:
        if motion_id not in durations:
            raise KeyError(f"no ingest duration for motion_id {motion_id!r}")
        clip_dir = output / motion_id
        if skip_existing and optimized_kinematics_is_complete(clip_dir):
            assignments.append(
                ClipAssignment(
                    motion_id=motion_id,
                    gpu=-1,
                    clip_duration_seconds=float(durations[motion_id]),
                    output_dir=str(clip_dir),
                    action="skip",
                )
            )
            continue
        gpu = int(gpus[pending_index % len(gpus)])
        pending_index += 1
        assignments.append(
            ClipAssignment(
                motion_id=motion_id,
                gpu=gpu,
                clip_duration_seconds=float(durations[motion_id]),
                output_dir=str(clip_dir),
                action="run",
            )
        )
    return tuple(assignments)


def pop_queued_session(queue_path: str | Path, lock_path: str | Path) -> str | None:
    """Atomically claim the next motion id from a flock-protected queue file."""

    queue = Path(queue_path)
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if not queue.is_file():
                return None
            lines = [line.strip() for line in queue.read_text(encoding="utf-8").splitlines()]
            remaining = [line for line in lines if line]
            if not remaining:
                queue.write_text("", encoding="utf-8")
                return None
            claimed, rest = remaining[0], remaining[1:]
            queue.write_text(("\n".join(rest) + "\n") if rest else "", encoding="utf-8")
            return claimed
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_clip_queue(
    experiment_path: str | Path,
    *,
    gpus: Sequence[int],
    dataset_root: str | Path | None = None,
    source_root: str | Path | None = None,
    output_root: str | Path | None = None,
    dry_run: bool = False,
    continue_on_error: bool | None = None,
    motion_ids: Sequence[str] | None = None,
    skip_existing: bool = True,
    robot_config: str | Path | None = None,
) -> dict[str, Any]:
    """Ingest source sessions if needed, then run one SIL clip per GPU."""

    from sil_tts.run_sil import ingest_kinematics_dataset, load_experiment

    if (dataset_root is None) == (source_root is None):
        raise ValueError("provide exactly one of source_root or dataset_root")

    experiment = Path(experiment_path).expanduser().resolve()
    _, config = load_experiment(experiment)
    destination = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else default_repo_root() / "outputs" / config.name / "clips"
    )
    destination.mkdir(parents=True, exist_ok=True)
    ingest_summary: dict[str, Any] | None = None
    if source_root is not None:
        dataset = destination / INGESTED_DATASET_DIR
        ingest_robot = (
            Path(robot_config).expanduser().resolve()
            if robot_config is not None
            else experiment_robot_yaml(experiment)
        )
        ingest_summary = ingest_kinematics_dataset(
            source_root, dataset, robot_config=ingest_robot
        )
        if ingest_summary["num_ingested"] == 0:
            raise ValueError(
                f"no usable session_*/final_50hz/kinematics.npz under {source_root}"
            )
    else:
        dataset = Path(dataset_root).expanduser().resolve()
    durations = load_clip_durations(dataset)
    selected = tuple(motion_ids) if motion_ids is not None else tuple(durations)
    unknown = [motion_id for motion_id in selected if motion_id not in durations]
    if unknown:
        raise KeyError(f"motion ids are missing from clip_durations.json: {unknown}")
    gpu_ids = tuple(int(gpu) for gpu in gpus)
    assignments = plan_clip_runs(
        motion_ids=selected,
        durations=durations,
        gpus=gpu_ids,
        output_root=destination,
        skip_existing=skip_existing,
    )
    continue_errors = (
        _continue_on_error_from_env() if continue_on_error is None else bool(continue_on_error)
    )
    pending = [item for item in assignments if item.action == "run"]
    summary: dict[str, Any] = {
        "experiment": str(experiment),
        "source_root": (
            str(Path(source_root).expanduser().resolve()) if source_root is not None else None
        ),
        "dataset_root": str(dataset),
        "output_root": str(destination),
        "ingest": ingest_summary,
        "gpus": list(gpu_ids),
        "continue_on_error": continue_errors,
        "dry_run": bool(dry_run),
        "assignments": [asdict(item) for item in assignments],
        "num_run": len(pending),
        "num_skip": sum(1 for item in assignments if item.action == "skip"),
        "results": [],
    }
    if dry_run or not pending:
        summary["success"] = True
        _write_json(destination / QUEUE_SUMMARY_FILE, summary)
        return summary

    queue_path = destination / QUEUE_FILE
    lock_path = destination / QUEUE_LOCK_FILE
    spec_path = destination / QUEUE_SPEC_FILE
    stop_path = destination / STOP_FILE
    if stop_path.exists():
        stop_path.unlink()
    queue_path.write_text(
        "".join(f"{item.motion_id}\n" for item in pending),
        encoding="utf-8",
    )
    spec = {
        "experiment": str(experiment),
        "dataset_root": str(dataset),
        "output_root": str(destination),
        "durations": {item.motion_id: item.clip_duration_seconds for item in pending},
        "continue_on_error": continue_errors,
        "python": sys.executable,
    }
    _write_json(spec_path, spec)

    workers: list[subprocess.Popen[bytes]] = []
    try:
        for gpu in gpu_ids:
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "sil_tts.clip_queue",
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
    results = _collect_clip_results(assignments)
    summary["results"] = results
    summary["worker_exit_codes"] = exit_codes
    summary["success"] = all(code == 0 for code in exit_codes) and all(
        row["status"] != "failed" for row in results
    )
    _write_json(destination / QUEUE_SUMMARY_FILE, summary)
    return summary


def gpu_worker(
    *,
    gpu: int,
    queue_path: str | Path,
    lock_path: str | Path,
    spec_path: str | Path,
    stop_path: str | Path,
    execute_clip: Callable[..., int] | None = None,
) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    durations = spec["durations"]
    runner = execute_clip if execute_clip is not None else _subprocess_run_clip
    continue_on_error = bool(spec.get("continue_on_error", False))
    stop = Path(stop_path)
    failed = False
    while True:
        if stop.is_file():
            return 1 if failed else 0
        motion_id = pop_queued_session(queue_path, lock_path)
        if motion_id is None:
            return 1 if failed else 0
        output_dir = Path(spec["output_root"]) / motion_id
        if optimized_kinematics_is_complete(output_dir):
            continue
        started = time.time()
        try:
            code = runner(
                gpu=int(gpu),
                experiment=Path(spec["experiment"]),
                dataset_root=Path(spec["dataset_root"]),
                motion_id=motion_id,
                clip_duration_seconds=float(durations[motion_id]),
                output_dir=output_dir,
                python_executable=str(spec.get("python", sys.executable)),
            )
        except Exception as exc:
            code = 1
            _write_failure(
                output_dir,
                motion_id=motion_id,
                gpu=int(gpu),
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.time() - started,
            )
        if code != 0:
            failed = True
            if not (output_dir / FAILURE_FILE).is_file():
                _write_failure(
                    output_dir,
                    motion_id=motion_id,
                    gpu=int(gpu),
                    error=f"sil-tts run exited with {code}",
                    elapsed_seconds=time.time() - started,
                )
            if not continue_on_error:
                stop.write_text(f"gpu={gpu} motion_id={motion_id}\n", encoding="utf-8")
                return 1
    return 1 if failed else 0


def _subprocess_run_clip(
    *,
    gpu: int,
    experiment: Path,
    dataset_root: Path,
    motion_id: str,
    clip_duration_seconds: float,
    output_dir: Path,
    python_executable: str,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    completed = subprocess.run(
        [
            python_executable,
            "-m",
            "sil_tts.run_sil",
            "run",
            str(experiment),
            "--device-index",
            "0",
            "--dataset-root",
            str(dataset_root),
            "--motion-id",
            motion_id,
            "--clip-duration-seconds",
            str(clip_duration_seconds),
            "--output-dir",
            str(output_dir),
            "--disable-visualization",
        ],
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _collect_clip_results(assignments: Sequence[ClipAssignment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in assignments:
        output_dir = Path(item.output_dir)
        if item.action == "skip" or optimized_kinematics_is_complete(output_dir):
            status = "skipped" if item.action == "skip" else "success"
            rows.append(
                {
                    "motion_id": item.motion_id,
                    "status": status,
                    "output_dir": item.output_dir,
                    "optimized_kinematics": str(output_dir / OPTIMIZED_KINEMATICS_FILE),
                }
            )
            continue
        failure = output_dir / FAILURE_FILE
        rows.append(
            {
                "motion_id": item.motion_id,
                "status": "failed" if failure.is_file() else "pending",
                "output_dir": item.output_dir,
                "failure": str(failure) if failure.is_file() else None,
            }
        )
    return rows


def _continue_on_error_from_env() -> bool:
    raw = os.environ.get("CONTINUE_ON_ERROR", "0").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _write_failure(
    output_dir: Path,
    *,
    motion_id: str,
    gpu: int,
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
            "elapsed_seconds": elapsed_seconds,
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sil_tts.clip_queue")
    parser.add_argument("--worker-gpu", type=int, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _worker_parser().parse_args(argv)
    return gpu_worker(
        gpu=args.worker_gpu,
        queue_path=args.queue,
        lock_path=args.lock,
        spec_path=args.spec,
        stop_path=args.stop,
    )


if __name__ == "__main__":
    raise SystemExit(main())
