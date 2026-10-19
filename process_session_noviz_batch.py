#!/usr/bin/env python3
"""Batch scheduler for ``process_session_noviz.py``.

The scheduler discovers complete MCAP files, assigns them to tasks, and runs
one isolated processor subprocess per session. GPU workers are isolated with
``CUDA_VISIBLE_DEVICES`` so the hand backend (HaWoR or WiLoR) can safely use
multiple GPUs.

HaWoR (default) is the same tracker used by ``hawor_hand_tracker.py``:
``HandTracker(model="hawor").detect_sequence(..., intrinsics=rgb_K)``.
No LiDAR pin, no W0+index-MCP, no SLAM/infiller.

The current flat ``ar_sessions/session_*/`` layout has no task labels. Choose
one of ``--task``, ``--task-map``, or ``--session-as-task`` when using it.
Nested ``ar_sessions/<task>/<session>/`` layouts are inferred automatically.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_INPUT_ROOT = Path("data/origin_data")
DEFAULT_OUTPUT_ROOT = Path("data/egocentric_data")
DEFAULT_STERA_PYTHON = Path(sys.executable)
DEFAULT_HAWOR_DIR = Path(os.environ.get("KODA_HAWOR_DIR", str(Path(os.environ.get("KODA_ASSETS", "assets")) / "models/hawor")))
DEFAULT_HAWOR_YOLO_CONF = 0.2
EGO2HUMANOID_ROOT = Path(__file__).resolve().parent
MCAP_MAGIC = b"\x89MCAP0\r\n"
CORE_OUTPUTS = ("export_manifest.json", "annotation.hdf5", "rgb.mp4")
RESERVED_TASK_NAMES = {
    "_batch_failed",
    "_batch_logs",
    "_batch_previous",
    "_batch_work",
}


@dataclass(frozen=True)
class InputProbe:
    path: Path
    ready: bool
    reason: str
    size_bytes: int
    mtime: float


@dataclass(frozen=True)
class Job:
    index: int
    mcap: Path
    task: str
    session: str
    output_dir: Path
    log_path: Path
    mode: str = "full"


@dataclass(frozen=True)
class GPUInfo:
    selector: str
    index: str
    uuid: str
    name: str
    memory_free_mb: int
    utilization_percent: int


@dataclass(frozen=True)
class RunConfig:
    python: Path
    processor: Path
    processor_args: tuple[str, ...]
    output_root: Path
    cpu_threads_per_worker: int
    with_viz: bool
    keep_failed: bool
    fail_fast: bool
    job_timeout_s: float | None
    min_file_age_s: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_name(value: str, *, field: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."}:
        raise ValueError(f"{field} cannot be empty or '.'/'..'")
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be one directory name, got {value!r}")
    if field == "task" and value in RESERVED_TASK_NAMES:
        raise ValueError(f"task name {value!r} is reserved by the batch scheduler")
    return value


def load_task_map(path: Path) -> dict[str, str]:
    """Load ``session -> task`` from JSON or a headered CSV file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    mapping: dict[str, str] = {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if not {"session", "task"}.issubset(fields):
                raise ValueError(
                    f"{path}: CSV header must contain 'session' and 'task'"
                )
            for row_number, row in enumerate(reader, start=2):
                session = str(row.get("session", "")).strip()
                task = str(row.get("task", "")).strip()
                if not session or not task:
                    raise ValueError(f"{path}:{row_number}: empty session/task")
                if session in mapping and mapping[session] != task:
                    raise ValueError(f"{path}:{row_number}: duplicate {session!r}")
                mapping[session] = validate_name(task, field="task")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
            data = data["sessions"]
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: JSON must be {{\"session_name\": \"task_name\"}}"
            )
        for session, task in data.items():
            session = str(session).strip()
            if not session:
                raise ValueError(f"{path}: empty session key")
            mapping[session] = validate_name(str(task), field="task")

    if not mapping:
        raise ValueError(f"{path}: task map is empty")
    return mapping


def discover_mcaps(input_root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in input_root.rglob("*.mcap")
        if path.is_file()
    )


def discover_empty_session_dirs(input_root: Path) -> list[Path]:
    empty: list[Path] = []
    for path in sorted(input_root.rglob("session_*")):
        if path.is_dir() and not any(path.glob("*.mcap")):
            empty.append(path.resolve())
    return empty


def probe_mcap(path: Path, *, min_file_age_s: float) -> InputProbe:
    """Reject partial rsync files by checking age and MCAP header/footer magic."""
    try:
        stat = path.stat()
        if stat.st_size < 2 * len(MCAP_MAGIC):
            return InputProbe(path, False, "file_too_small", stat.st_size, stat.st_mtime)
        age = time.time() - stat.st_mtime
        if age < min_file_age_s:
            return InputProbe(
                path,
                False,
                f"modified_recently ({age:.1f}s < {min_file_age_s:.1f}s)",
                stat.st_size,
                stat.st_mtime,
            )
        with path.open("rb") as f:
            header = f.read(len(MCAP_MAGIC))
            f.seek(-len(MCAP_MAGIC), os.SEEK_END)
            footer = f.read(len(MCAP_MAGIC))
        if header != MCAP_MAGIC:
            return InputProbe(path, False, "invalid_header_magic", stat.st_size, stat.st_mtime)
        if footer != MCAP_MAGIC:
            return InputProbe(
                path,
                False,
                "missing_footer_magic (incomplete copy or corrupt MCAP)",
                stat.st_size,
                stat.st_mtime,
            )
        return InputProbe(path, True, "ready", stat.st_size, stat.st_mtime)
    except OSError as exc:
        return InputProbe(path, False, f"stat/read failed: {exc}", 0, 0.0)


def verify_with_mcap_reader(
    path: Path, *, python: Path, timeout_s: float
) -> tuple[bool, str]:
    code = (
        "import sys; from stera.data import MCAPReader; "
        "reader=MCAPReader(sys.argv[1]); "
        "n=int(reader.num_rgb_frames); "
        "assert n > 0, 'MCAP has no RGB frames'; print(n)"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"MCAPReader preflight failed: {exc}"
    if result.returncode != 0:
        detail = (result.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {result.returncode}"
        return False, f"MCAPReader rejected file: {tail}"
    return True, "ready"


def task_for_mcap(
    mcap: Path,
    *,
    input_root: Path,
    fixed_task: str | None,
    task_map: dict[str, str] | None,
    session_as_task: bool,
) -> tuple[str, str]:
    relative = mcap.relative_to(input_root)
    session = mcap.parent.name if mcap.parent != input_root else mcap.stem
    session = validate_name(session, field="session")

    if fixed_task is not None:
        return validate_name(fixed_task, field="task"), session

    if task_map is not None:
        candidates = (
            relative.as_posix(),
            relative.parent.as_posix(),
            mcap.name,
            mcap.stem,
            session,
        )
        for key in candidates:
            if key in task_map:
                return validate_name(task_map[key], field="task"), session
        raise ValueError(
            f"no task-map entry for {relative}; tried keys: {', '.join(candidates)}"
        )

    # A nested <task>/<session>/<file.mcap> layout carries an unambiguous task.
    if len(relative.parts) >= 3:
        return validate_name(relative.parts[0], field="task"), session
    if session_as_task:
        return validate_name(session, field="task"), session
    raise ValueError(
        f"cannot infer task for flat input {relative}; provide --task, "
        "--task-map, or --session-as-task"
    )


def build_jobs(
    mcaps: Sequence[Path],
    *,
    input_root: Path,
    output_root: Path,
    fixed_task: str | None,
    task_map: dict[str, str] | None,
    session_as_task: bool,
) -> list[Job]:
    assignments: list[tuple[Path, str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for mcap in mcaps:
        task, session = task_for_mcap(
            mcap,
            input_root=input_root,
            fixed_task=fixed_task,
            task_map=task_map,
            session_as_task=session_as_task,
        )
        assignments.append((mcap, task, session))
        counts[(task, session)] = counts.get((task, session), 0) + 1

    jobs: list[Job] = []
    outputs: set[Path] = set()
    for index, (mcap, task, session) in enumerate(assignments):
        base_session = session
        if counts[(task, base_session)] > 1:
            session = validate_name(f"{session}__{mcap.stem}", field="session")
        collapse_single_session_task = (
            session_as_task
            and fixed_task is None
            and task_map is None
            and counts[(task, base_session)] == 1
        )
        output_dir = (
            output_root / task
            if collapse_single_session_task
            else output_root / task / session
        )
        if output_dir in outputs:
            raise ValueError(f"multiple inputs resolve to output {output_dir}")
        outputs.add(output_dir)
        log_path = (
            output_root / "_batch_logs" / f"{task}.log"
            if collapse_single_session_task
            else output_root / "_batch_logs" / task / f"{session}.log"
        )
        jobs.append(
            Job(
                index=index,
                mcap=mcap,
                task=task,
                session=session,
                output_dir=output_dir,
                log_path=log_path,
            )
        )
    return jobs


def query_gpu_inventory() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    inventory: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=4)]
        if len(parts) != 5:
            continue
        index, gpu_uuid, name, memory_free, utilization = parts
        try:
            inventory.append(
                GPUInfo(
                    selector=index,
                    index=index,
                    uuid=gpu_uuid,
                    name=name,
                    memory_free_mb=int(memory_free),
                    utilization_percent=int(utilization),
                )
            )
        except ValueError:
            continue
    return inventory


def visible_inventory(inventory: Sequence[GPUInfo]) -> list[GPUInfo]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return list(inventory)
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or value.strip() in {"-1", "none", "None"}:
        return []

    visible: list[GPUInfo] = []
    for token in tokens:
        for gpu in inventory:
            if token == gpu.index or token == gpu.uuid or gpu.uuid.startswith(token):
                visible.append(
                    GPUInfo(
                        selector=token,
                        index=gpu.index,
                        uuid=gpu.uuid,
                        name=gpu.name,
                        memory_free_mb=gpu.memory_free_mb,
                        utilization_percent=gpu.utilization_percent,
                    )
                )
                break
    return visible


def select_gpus(
    spec: str,
    *,
    min_free_mb: int,
    max_utilization: int,
) -> list[GPUInfo]:
    spec = spec.strip()
    if spec.lower() in {"cpu", "none", "-1"}:
        return []

    inventory = query_gpu_inventory()
    visible = visible_inventory(inventory)
    if spec.lower() == "auto":
        return [
            gpu
            for gpu in visible
            if gpu.memory_free_mb >= min_free_mb
            and gpu.utilization_percent <= max_utilization
        ]
    if spec.lower() == "all":
        return visible

    selectors = [token.strip() for token in spec.split(",") if token.strip()]
    if not selectors:
        raise ValueError("--gpus did not contain any GPU identifiers")
    for selector in selectors:
        if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", selector):
            raise ValueError(f"invalid GPU identifier {selector!r}")

    selected: list[GPUInfo] = []
    for selector in selectors:
        match = next(
            (
                gpu
                for gpu in inventory
                if selector == gpu.index
                or selector == gpu.uuid
                or gpu.uuid.startswith(selector)
            ),
            None,
        )
        if match is None:
            selected.append(GPUInfo(selector, selector, "unknown", "unknown", -1, -1))
        else:
            selected.append(
                GPUInfo(
                    selector=selector,
                    index=match.index,
                    uuid=match.uuid,
                    name=match.name,
                    memory_free_mb=match.memory_free_mb,
                    utilization_percent=match.utilization_percent,
                )
            )
    if len({gpu.selector for gpu in selected}) != len(selected):
        raise ValueError("--gpus contains duplicate identifiers")
    return selected


def output_complete(path: Path, *, with_viz: bool) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "output directory missing"
    required = list(CORE_OUTPUTS)
    if with_viz:
        required.append("visualization.rrd")
    for name in required:
        candidate = path / name
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            return False, f"missing/empty {name}"
    try:
        manifest = json.loads((path / "export_manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid export_manifest.json: {exc}"
    if not isinstance(manifest, dict):
        return False, "export_manifest.json is not an object"
    return True, "complete"


def move_failed_stage(stage: Path, job: Job, output_root: Path) -> Path | None:
    if not stage.exists():
        return None
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    failed = output_root / "_batch_failed" / job.task / f"{job.session}.{suffix}"
    failed.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, failed)
    return failed


def promote_stage(stage: Path, job: Job, output_root: Path) -> Path | None:
    """Atomically publish stage; preserve any prior final output as a backup."""
    final = job.output_dir
    final.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if final.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        backup = output_root / "_batch_previous" / job.task / f"{job.session}.{suffix}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(final, backup)
    try:
        os.replace(stage, final)
    except BaseException:
        if backup is not None and backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    return backup


def run_job(
    job: Job,
    *,
    gpu_slots: queue.Queue[str | None],
    config: RunConfig,
    print_lock: threading.Lock,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {"status": "cancelled", "task": job.task, "session": job.session}

    gpu = gpu_slots.get()
    started_at = utc_now()
    start_clock = time.monotonic()
    rebuild = job.mode == "rebuild_viz"
    stage: Path | None = None
    if not rebuild:
        stage_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:10]}"
        stage = (
            config.output_root
            / "_batch_work"
            / job.task
            / f"{job.session}.{stage_suffix}"
        )
    command = [
        str(config.python),
        str(config.processor),
        "--mcap",
        str(job.mcap),
        "--out",
        str(job.output_dir if rebuild else stage),
        *config.processor_args,
    ]
    if rebuild:
        command.append("--rebuild-viz")
    result: dict[str, Any] = {
        "task": job.task,
        "session": job.session,
        "mcap": str(job.mcap),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "gpu": gpu if gpu is not None else "cpu",
        "started_at": started_at,
    }

    try:
        if stop_event.is_set():
            result["status"] = "cancelled"
            return result
        fresh_probe = probe_mcap(job.mcap, min_file_age_s=config.min_file_age_s)
        if not fresh_probe.ready:
            result["status"] = "input_not_ready"
            result["reason"] = fresh_probe.reason
            with print_lock:
                print(
                    f"[input_not_ready] {job.task}/{job.session}: "
                    f"{fresh_probe.reason}",
                    flush=True,
                )
            return result
        if stage is not None:
            stage.parent.mkdir(parents=True, exist_ok=True)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else gpu
        pythonpath = [str(EGO2HUMANOID_ROOT)]
        existing = env.get("PYTHONPATH", "")
        if existing:
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        thread_count = str(config.cpu_threads_per_worker)
        env["OMP_NUM_THREADS"] = thread_count
        env["MKL_NUM_THREADS"] = thread_count
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count

        with print_lock:
            print(
                f"[start] {job.task}/{job.session} gpu={result['gpu']} "
                f"log={job.log_path}",
                flush=True,
            )
        with job.log_path.open("w", encoding="utf-8") as log:
            log.write(f"started_at: {started_at}\n")
            log.write(f"gpu: {result['gpu']}\n")
            log.write(f"command: {shlex.join(command)}\n\n")
            log.flush()
            try:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=config.job_timeout_s,
                    check=False,
                )
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                return_code = 124
                log.write(f"\n[batch] timed out after {config.job_timeout_s}s\n")
            except OSError as exc:
                return_code = 127
                log.write(f"\n[batch] failed to start processor: {exc}\n")

        result["return_code"] = return_code
        if return_code != 0:
            result["status"] = "failed"
            result["reason"] = f"processor exited with code {return_code}"
        else:
            validated = job.output_dir if rebuild else stage
            complete, reason = output_complete(
                validated, with_viz=True if rebuild else config.with_viz
            )
            if not complete:
                result["status"] = "failed"
                result["reason"] = f"output validation failed: {reason}"
            elif rebuild:
                batch_job_path = job.output_dir / "batch_job.json"
                metadata: dict[str, Any] = {}
                if batch_job_path.is_file():
                    try:
                        loaded = json.loads(batch_job_path.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            metadata = loaded
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                metadata["visualization_rebuilt_at"] = utc_now()
                metadata["visualization_gpu"] = result["gpu"]
                batch_job_path.write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                result["status"] = "success"
            else:
                metadata = {
                    "source_mcap": str(job.mcap),
                    "task": job.task,
                    "session": job.session,
                    "gpu": result["gpu"],
                    "processor": str(config.processor),
                    "processor_args": list(config.processor_args),
                    "completed_at": utc_now(),
                }
                (stage / "batch_job.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                backup = promote_stage(stage, job, config.output_root)
                result["status"] = "success"
                if backup is not None:
                    result["previous_output"] = str(backup)

        if result["status"] == "failed" and stage is not None and stage.exists():
            if config.keep_failed:
                failed_path = move_failed_stage(stage, job, config.output_root)
                result["failed_output"] = str(failed_path) if failed_path else None
            else:
                shutil.rmtree(stage)
            if config.fail_fast:
                stop_event.set()
    except BaseException as exc:
        result["status"] = "failed"
        result["reason"] = f"batch worker error: {type(exc).__name__}: {exc}"
        if stage is not None and stage.exists():
            try:
                if config.keep_failed:
                    failed_path = move_failed_stage(stage, job, config.output_root)
                    result["failed_output"] = str(failed_path) if failed_path else None
                else:
                    shutil.rmtree(stage)
            except OSError as cleanup_exc:
                result["cleanup_error"] = str(cleanup_exc)
        if config.fail_fast:
            stop_event.set()
    finally:
        result["finished_at"] = utc_now()
        result["duration_s"] = round(time.monotonic() - start_clock, 3)
        gpu_slots.put(gpu)

    with print_lock:
        detail = f" reason={result.get('reason')}" if result.get("reason") else ""
        print(
            f"[{result['status']}] {job.task}/{job.session} "
            f"{result['duration_s']:.1f}s{detail}",
            flush=True,
        )
    return result


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return Path("/")
        current = parent
    return current


def absolute_without_resolving(path: Path) -> Path:
    """Make a path absolute while preserving virtualenv interpreter symlinks."""
    path = path.expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).absolute()


def option_present(arguments: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in arguments)


def normalize_processor_args(
    arguments: Sequence[str], *, with_viz: bool
) -> tuple[str, ...]:
    normalized = list(arguments)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    for managed in ("--mcap", "--out", "--trim-existing", "--rebuild-viz"):
        if option_present(normalized, managed):
            raise ValueError(f"{managed} is managed by the batch scheduler")
    if with_viz and option_present(normalized, "--no-viz"):
        raise ValueError("--with-viz/--rebuild-viz conflicts with processor argument --no-viz")
    if not with_viz:
        if not option_present(normalized, "--no-viz"):
            normalized.append("--no-viz")
        # Skeletons are only consumed by Rerun in the single-session processor.
        if not option_present(normalized, "--no-skeleton"):
            normalized.append("--no-skeleton")
    return tuple(normalized)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    default_python = DEFAULT_STERA_PYTHON if DEFAULT_STERA_PYTHON.is_file() else Path(sys.executable)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  # All flat sessions are demonstrations of one task:
  %(prog)s --task pick_up_box

  # Map individual sessions to semantic tasks:
  %(prog)s --task-map task_map.json --gpus auto

  # Treat every flat session as its own task and inspect without running:
  %(prog)s --session-as-task --dry-run

Unknown options are forwarded to process_session_noviz.py, for example:
  %(prog)s --task pick_up_box --blur --no-ground-align
""",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--processor",
        type=Path,
        default=Path(__file__).with_name("process_session_noviz.py"),
    )
    parser.add_argument("--python", type=Path, default=default_python)
    parser.add_argument(
        "--hand-model",
        choices=("wilor", "hawor"),
        default="hawor",
        help="Stera HandTracker backend (default: hawor, camera-space, no LiDAR pin)",
    )
    parser.add_argument(
        "--hawor-dir",
        type=Path,
        default=DEFAULT_HAWOR_DIR,
        help="Local HaWoR clone (https://github.com/ThunderVVV/HaWoR)",
    )
    parser.add_argument(
        "--hawor-yolo-conf",
        type=float,
        default=DEFAULT_HAWOR_YOLO_CONF,
        help="YOLO confidence forwarded to hawor_hand_tracker (default: 0.2)",
    )

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task", help="Assign every discovered session to this task")
    task_group.add_argument(
        "--task-map",
        type=Path,
        help="JSON mapping or CSV with session,task columns",
    )
    task_group.add_argument(
        "--session-as-task",
        action="store_true",
        help="Treat each session directory as a separate task",
    )

    parser.add_argument(
        "--gpus",
        default="auto",
        help="auto, all, cpu, or comma-separated physical GPU IDs (default: auto)",
    )
    parser.add_argument(
        "--min-free-gpu-mb",
        type=int,
        default=12000,
        help="Minimum free memory for --gpus auto (default: 12000)",
    )
    parser.add_argument(
        "--max-gpu-utilization",
        type=int,
        default=20,
        help="Maximum utilization percent for --gpus auto (default: 20)",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help="Concurrent sessions; defaults to one per selected GPU",
    )
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=positive_int,
        default=None,
        help="OMP/MKL/OpenBLAS threads per session (auto by default)",
    )
    parser.add_argument(
        "--min-file-age",
        type=nonnegative_float,
        default=5.0,
        help="Skip MCAPs modified more recently than this many seconds",
    )
    parser.add_argument(
        "--verify-reader",
        action="store_true",
        help="Preflight each complete-looking file with Stera MCAPReader",
    )
    parser.add_argument(
        "--reader-timeout",
        type=positive_int,
        default=60,
        help="Seconds allowed for each --verify-reader check",
    )
    parser.add_argument(
        "--with-viz",
        action="store_true",
        help=(
            "Rebuild visualization.rrd from an existing episode directory; "
            "never replace hdf5/rgb/mesh. rebuild-viz does not need HaWoR/WiLoR"
        ),
    )
    parser.add_argument(
        "--rebuild-viz",
        action="store_true",
        help=(
            "Only rebuild visualization.rrd in existing episode directories; "
            "never replace hdf5/rgb/mesh"
        ),
    )
    parser.add_argument("--force", action="store_true", help="Reprocess completed outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing")
    parser.add_argument("--fail-fast", action="store_true", help="Do not start new jobs after a failure")
    parser.add_argument(
        "--discard-failed",
        action="store_true",
        help="Delete this run's staging output after a failed session",
    )
    parser.add_argument(
        "--job-timeout",
        type=positive_int,
        default=None,
        help="Maximum seconds per session (unlimited by default)",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=nonnegative_float,
        default=10.0,
        help="Refuse to start below this free-space threshold (default: 10)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary JSON path (default: OUTPUT_ROOT/_batch_summary.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args, passthrough = parser.parse_known_args(argv)

    try:
        processor_args = normalize_processor_args(
            passthrough, with_viz=args.with_viz or args.rebuild_viz
        )
        if not option_present(processor_args, "--hand-model"):
            processor_args = ("--hand-model", args.hand_model, *processor_args)
        if args.hand_model == "hawor" and not option_present(
            processor_args, "--hawor-dir"
        ):
            processor_args = (
                "--hawor-dir",
                str(args.hawor_dir.expanduser().resolve()),
                *processor_args,
            )
        if args.hand_model == "hawor" and not option_present(
            processor_args, "--hawor-yolo-conf"
        ):
            processor_args = (
                "--hawor-yolo-conf",
                str(args.hawor_yolo_conf),
                *processor_args,
            )
        input_root = args.input_root.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        processor = args.processor.expanduser().resolve()
        # Resolving a venv's python symlink bypasses pyvenv.cfg and loses the
        # environment's site-packages (including stera).
        python = absolute_without_resolving(args.python)
        fixed_task = validate_name(args.task, field="task") if args.task else None
        task_map = load_task_map(args.task_map) if args.task_map else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not input_root.is_dir():
        parser.error(f"input root does not exist: {input_root}")
    if not processor.is_file():
        parser.error(f"single-session processor does not exist: {processor}")
    if not python.is_file():
        parser.error(f"Python interpreter does not exist: {python}")
    if args.min_free_gpu_mb < 0:
        parser.error("--min-free-gpu-mb must be >= 0")
    if not 0 <= args.max_gpu_utilization <= 100:
        parser.error("--max-gpu-utilization must be in [0, 100]")

    discovered = discover_mcaps(input_root)
    empty_session_dirs = discover_empty_session_dirs(input_root)
    probes = [probe_mcap(path, min_file_age_s=args.min_file_age) for path in discovered]

    if args.verify_reader:
        verified: list[InputProbe] = []
        for probe in probes:
            if not probe.ready:
                verified.append(probe)
                continue
            ok, reason = verify_with_mcap_reader(
                probe.path, python=python, timeout_s=float(args.reader_timeout)
            )
            verified.append(
                InputProbe(
                    probe.path,
                    ok,
                    reason,
                    probe.size_bytes,
                    probe.mtime,
                )
            )
        probes = verified

    ready_paths = [probe.path for probe in probes if probe.ready]
    pending_probes = [probe for probe in probes if not probe.ready]
    try:
        jobs = build_jobs(
            ready_paths,
            input_root=input_root,
            output_root=output_root,
            fixed_task=fixed_task,
            task_map=task_map,
            session_as_task=args.session_as_task,
        )
    except ValueError as exc:
        parser.error(str(exc))

    no_hands = option_present(processor_args, "--no-hands")
    try:
        selected_gpus = (
            []
            if no_hands and args.gpus == "auto"
            else select_gpus(
                args.gpus,
                min_free_mb=args.min_free_gpu_mb,
                max_utilization=args.max_gpu_utilization,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    skipped: list[dict[str, Any]] = []
    active_jobs: list[Job] = []
    want_viz = bool(args.with_viz or args.rebuild_viz)
    for job in jobs:
        core_ok, core_reason = output_complete(job.output_dir, with_viz=False)
        viz_path = job.output_dir / "visualization.rrd"
        viz_ok = viz_path.is_file() and viz_path.stat().st_size > 0
        skip_record = {
            "task": job.task,
            "session": job.session,
            "mcap": str(job.mcap),
            "output_dir": str(job.output_dir),
        }
        if args.rebuild_viz:
            if not core_ok:
                skipped.append(
                    {
                        **skip_record,
                        "status": "failed",
                        "reason": f"cannot rebuild viz: {core_reason}",
                    }
                )
            elif viz_ok and not args.force:
                skipped.append(
                    {**skip_record, "status": "skipped_complete", "reason": "complete"}
                )
            else:
                active_jobs.append(
                    replace(
                        job,
                        mode="rebuild_viz",
                        log_path=job.log_path.with_name(
                            f"{job.log_path.stem}.rebuild_viz.log"
                        ),
                    )
                )
            continue
        if core_ok and (not want_viz or viz_ok) and not args.force:
            skipped.append(
                {**skip_record, "status": "skipped_complete", "reason": "complete"}
            )
        elif want_viz and core_ok and not args.force:
            active_jobs.append(
                replace(
                    job,
                    mode="rebuild_viz",
                    log_path=job.log_path.with_name(
                        f"{job.log_path.stem}.rebuild_viz.log"
                    ),
                )
            )
        else:
            active_jobs.append(job)

    rebuild_only = bool(active_jobs) and all(
        job.mode == "rebuild_viz" for job in active_jobs
    )
    if rebuild_only and args.gpus == "auto" and selected_gpus:
        print("[plan] rebuild-viz does not need HaWoR/WiLoR; using CPU")
        selected_gpus = []

    needs_hand_gpu = any(job.mode != "rebuild_viz" for job in active_jobs)
    if (
        not selected_gpus
        and not no_hands
        and needs_hand_gpu
        and not args.dry_run
    ):
        parser.error(
            "no usable GPU selected, but HaWoR/WiLoR require CUDA; use --gpus IDS, "
            "free a GPU, or pass processor option --no-hands"
        )

    if selected_gpus:
        workers = args.workers or len(selected_gpus)
        if workers > len(selected_gpus):
            parser.error(
                f"--workers={workers} exceeds {len(selected_gpus)} selected GPUs; "
                "only one session per GPU is supported"
            )
    else:
        workers = args.workers or min(4, max(1, os.cpu_count() or 1))
    cpu_threads = args.cpu_threads_per_worker or min(
        8, max(1, (os.cpu_count() or 1) // workers)
    )

    gpu_description = (
        ", ".join(
            f"{gpu.selector}({gpu.memory_free_mb}MB,{gpu.utilization_percent}%)"
            for gpu in selected_gpus
        )
        if selected_gpus
        else "cpu"
    )
    print(
        f"[discover] mcap={len(discovered)} ready={len(ready_paths)} "
        f"pending={len(pending_probes)} empty_session_dirs={len(empty_session_dirs)}"
    )
    print(
        f"[plan] run={len(active_jobs)} skip_complete={len(skipped)} "
        f"workers={workers} devices={gpu_description}"
    )
    print(f"[processor-args] {shlex.join(processor_args)}")
    for probe in pending_probes:
        print(f"[pending] {probe.path}: {probe.reason}")
    for job in active_jobs:
        print(f"[job {job.mode}] {job.mcap} -> {job.output_dir}")

    if args.dry_run:
        print("[dry-run] no files were written")
        return 0

    disk_parent = nearest_existing_parent(output_root)
    free_disk_gb = shutil.disk_usage(disk_parent).free / (1024**3)
    if free_disk_gb < args.min_free_disk_gb:
        parser.error(
            f"only {free_disk_gb:.1f} GiB free at {disk_parent}; "
            f"requires --min-free-disk-gb={args.min_free_disk_gb:.1f}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary
        else output_root / "_batch_summary.json"
    )
    summary: dict[str, Any] = {
        "version": 1,
        "started_at": utc_now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "processor": str(processor),
        "python": str(python),
        "processor_args": list(processor_args),
        "with_viz": bool(args.with_viz or args.rebuild_viz),
        "rebuild_viz": bool(args.rebuild_viz),
        "workers": workers,
        "cpu_threads_per_worker": cpu_threads,
        "selected_gpus": [asdict(gpu) for gpu in selected_gpus],
        "free_disk_gb_at_start": round(free_disk_gb, 3),
        "discovered_mcaps": len(discovered),
        "empty_session_dirs": [str(path) for path in empty_session_dirs],
        "pending_inputs": [
            {
                "path": str(probe.path),
                "reason": probe.reason,
                "size_bytes": probe.size_bytes,
                "mtime": probe.mtime,
            }
            for probe in pending_probes
        ],
        "jobs": list(skipped),
    }
    write_json_atomic(summary_path, summary)

    results: list[dict[str, Any]] = []
    if active_jobs:
        gpu_slots: queue.Queue[str | None] = queue.Queue()
        if selected_gpus:
            for gpu in selected_gpus:
                gpu_slots.put(gpu.selector)
        else:
            for _ in range(workers):
                gpu_slots.put(None)

        config = RunConfig(
            python=python,
            processor=processor,
            processor_args=processor_args,
            output_root=output_root,
            cpu_threads_per_worker=cpu_threads,
            with_viz=args.with_viz,
            keep_failed=not args.discard_failed,
            fail_fast=args.fail_fast,
            job_timeout_s=float(args.job_timeout) if args.job_timeout else None,
            min_file_age_s=float(args.min_file_age),
        )
        print_lock = threading.Lock()
        stop_event = threading.Event()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="session") as pool:
            futures = {
                pool.submit(
                    run_job,
                    job,
                    gpu_slots=gpu_slots,
                    config=config,
                    print_lock=print_lock,
                    stop_event=stop_event,
                ): job
                for job in active_jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    result = {
                        "status": "failed",
                        "task": job.task,
                        "session": job.session,
                        "reason": f"unhandled worker exception: {exc}",
                    }
                    if args.fail_fast:
                        stop_event.set()
                results.append(result)
                summary["jobs"] = [*skipped, *results]
                write_json_atomic(summary_path, summary)

    all_results = [*skipped, *results]
    counts: dict[str, int] = {}
    for result in all_results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    summary["jobs"] = all_results
    summary["counts"] = counts
    summary["finished_at"] = utc_now()
    summary["status"] = "failed" if counts.get("failed", 0) else "success"
    write_json_atomic(summary_path, summary)

    print(f"[summary] {summary_path}")
    print(f"[counts] {json.dumps(counts, sort_keys=True)}")
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
