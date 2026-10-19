#!/usr/bin/env python3
"""Causally arrange complete 30 Hz processed bundles on the D435 timeline.

Every output stream consumes the exact same ``output_50hz.source_indices``
array from ``timeline_manifest.json``.  Frames are hard-linked when possible
and copied when requested (or when an automatic hard link crosses devices).
No rendering, compositing, inpainting, or target-pose update occurs here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    from frame_volume import create_volume, is_packed_stream, open_volume
    from timeline_alignment import previous_time_indices
except ImportError:  # package-style import
    from .frame_volume import create_volume, is_packed_stream, open_volume
    from .timeline_alignment import previous_time_indices


DEFAULT_STREAMS = (
    "rgb_geometry",
    "depth",
    "valid",
    "hole",
    "robot_mask",
    "paste_mask",
    "rgb_final",
    "synthetic",
    "provenance",
)


def _load_manifest(
    value: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"timeline manifest not found: {path}")
    return json.loads(path.read_text()), str(path.resolve())


def _timeline_contract(
    manifest: Mapping[str, Any], *, tolerance_s: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        source = np.asarray(
            manifest["source"]["timestamps_relative_s"], dtype=np.float64
        )
        target = np.asarray(
            manifest["target"]["timestamps_relative_s"], dtype=np.float64
        )
        mapping = np.asarray(manifest["output_50hz"]["source_indices"], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid timeline manifest structure: {exc}") from exc
    if source.ndim != 1 or target.ndim != 1 or mapping.ndim != 1:
        raise ValueError(
            "timeline timestamps and output mapping must be one-dimensional"
        )
    if len(mapping) != len(target):
        raise ValueError(
            f"output mapping length {len(mapping)} != target timeline length {len(target)}"
        )
    if not len(source) or not len(target):
        raise ValueError("timeline manifest contains an empty stream")
    if np.any((mapping < 0) | (mapping >= len(source))):
        raise ValueError("output mapping contains an out-of-range source index")

    expected, ages = previous_time_indices(target, source, tolerance_s=tolerance_s)
    if not np.array_equal(mapping, expected):
        mismatch = np.flatnonzero(mapping != expected)
        first = int(mismatch[0])
        raise ValueError(
            "manifest output mapping is not the measured past-only mapping: "
            f"output {first} stores {mapping[first]}, expected {expected[first]}"
        )
    if np.any(np.diff(mapping) < 0):
        raise ValueError("output mapping is not monotonic")
    if np.any(source[mapping] > target + tolerance_s):
        raise ValueError("output mapping contains a future source frame")
    return source, target, mapping, ages


def _indexed_files(directory: Path, expected_count: int) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"processed stream directory not found: {directory}")
    indexed: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or not path.stem.isdigit():
            continue
        index = int(path.stem)
        if index in indexed:
            raise ValueError(
                f"duplicate frame {index} in {directory}: "
                f"{indexed[index].name}, {path.name}"
            )
        indexed[index] = path
    required = set(range(expected_count))
    present = set(indexed)
    missing = sorted(required - present)
    extra = sorted(present - required)
    if missing or extra:
        raise ValueError(
            f"stream {directory.name} does not exactly cover 0..{expected_count - 1}; "
            f"missing={missing[:12]}, extra={extra[:12]}"
        )
    return indexed


def _materialize(
    source: Path,
    destination: Path,
    *,
    mode: str,
    force: bool,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stale = [
        p for p in destination.parent.glob(f"{destination.stem}.*") if p != destination
    ]
    existing = [p for p in [destination, *stale] if p.exists() or p.is_symlink()]
    if existing:
        if len(existing) == 1 and existing[0] == destination:
            try:
                if os.path.samefile(source, destination):
                    return "existing_hardlink"
            except OSError:
                pass
        if not force:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(f"output frame already exists (use --force): {names}")
        for path in existing:
            path.unlink()

    if mode in ("hardlink", "auto"):
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "copy"


def _read_frame_status(path: Path, expected_count: int) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"processed frame status JSONL not found: {path}")
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"frame status row {line_number} is not an object")
            default_index = len(rows)
            index = row.get(
                "source_frame_index",
                row.get("frame_index", row.get("frame", default_index)),
            )
            try:
                index = int(index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid frame index in {path}:{line_number}"
                ) from exc
            if index in rows:
                raise ValueError(f"duplicate frame status for source index {index}")
            rows[index] = row
    missing = sorted(set(range(expected_count)) - set(rows))
    extra = sorted(set(rows) - set(range(expected_count)))
    if missing or extra:
        raise ValueError(
            f"frame status does not exactly cover 0..{expected_count - 1}; "
            f"missing={missing[:12]}, extra={extra[:12]}"
        )
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_status_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.asarray(np.load(path, allow_pickle=False))
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise RuntimeError(f"failed to read frame: {path}")
    return array


def _count_fused_pixels(paths: Mapping[int, Path]) -> int:
    count = 0
    for path in paths.values():
        provenance = _read_array(path)
        if provenance.ndim == 3:
            provenance = np.max(provenance, axis=2)
        count += int(np.count_nonzero(provenance == 2))
    return count


def _count_fused_volume(volume) -> int:
    provenance = np.asarray(volume.array)
    if provenance.ndim == 4:
        provenance = provenance[..., 0]
    return int(np.count_nonzero(provenance == 2))


def _streams_are_packed(processed: Path, streams: Sequence[str]) -> bool:
    flags = [is_packed_stream(processed / stream) for stream in streams]
    if any(flags) and not all(flags):
        packed = [name for name, flag in zip(streams, flags) if flag]
        indexed = [name for name, flag in zip(streams, flags) if not flag]
        raise ValueError(
            "processed streams mix packed and indexed layouts: "
            f"packed={packed}, indexed={indexed}"
        )
    return bool(flags) and all(flags)






def _read_rgb(path: Path) -> np.ndarray:
    image = _read_array(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB frame must be HxWx3: {path} has {image.shape}")
    if image.dtype != np.uint8:
        if not np.all(np.isfinite(image)):
            raise ValueError(f"RGB frame contains non-finite values: {path}")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _write_video(
    rgb_paths: Sequence[Path],
    path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    codec: str,
    force: bool,
) -> tuple[dict[str, Any], list[int]]:
    if len(codec) != 4:
        raise ValueError("video codec must be a four-character code")
    if path.exists() and not force:
        raise FileExistsError(f"video already exists (use --force): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    if temporary.exists():
        temporary.unlink()
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"failed to open video writer for {temporary} with codec {codec}"
        )
    blank: list[int] = []
    try:
        for index, frame_path in enumerate(rgb_paths):
            frame = _read_rgb(frame_path)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"RGB frame {frame_path} is {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}; resizing is forbidden"
                )
            if not np.any(frame):
                blank.append(index)
            writer.write(frame)
    finally:
        writer.release()
    if blank:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"blank RGB output frames detected: {blank[:20]}")
    os.replace(temporary, path)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to verify output video: {path}")
    try:
        actual = {
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        }
    finally:
        capture.release()
    expected = {
        "frame_count": len(rgb_paths),
        "fps": float(fps),
        "width": width,
        "height": height,
    }
    if (
        actual["frame_count"] != expected["frame_count"]
        or actual["width"] != width
        or actual["height"] != height
        or abs(actual["fps"] - fps) > 0.1
    ):
        raise RuntimeError(
            f"output video verification failed: actual={actual}, expected={expected}"
        )
    return {
        "path": str(path.resolve()),
        "codec": codec,
        "actual": actual,
        "expected": expected,
    }, blank


def _write_video_from_rgb(
    frames: np.ndarray,
    path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    codec: str,
    force: bool,
    mapping: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[int]]:
    """Encode packed RGB uint8 frames. VideoWriter consumes BGR.

    ``mapping`` optionally gathers 30 Hz frames onto the 50 Hz timeline without
    materializing a 50 Hz volume.
    """
    if len(codec) != 4:
        raise ValueError("video codec must be a four-character code")
    if path.exists() and not force:
        raise FileExistsError(f"video already exists (use --force): {path}")
    rgb = np.asarray(frames)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"packed RGB must be TxHxWx3, got {rgb.shape}")
    if rgb.shape[1:3] != (height, width):
        raise ValueError(
            f"packed RGB is {rgb.shape[2]}x{rgb.shape[1]}, expected {width}x{height}; "
            "resizing is forbidden"
        )
    indices = (
        np.arange(rgb.shape[0], dtype=np.int64)
        if mapping is None
        else np.asarray(mapping, dtype=np.int64)
    )
    if np.any((indices < 0) | (indices >= rgb.shape[0])):
        raise ValueError("RGB mapping contains an out-of-range source index")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    if temporary.exists():
        temporary.unlink()
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"failed to open video writer for {temporary} with codec {codec}"
        )
    blank: list[int] = []
    try:
        for output_index, source_index in enumerate(indices.tolist()):
            frame = rgb[int(source_index)]
            if frame.dtype != np.uint8:
                if not np.all(np.isfinite(frame)):
                    raise ValueError(f"RGB frame {output_index} contains non-finite values")
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if not np.any(frame):
                blank.append(output_index)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if blank:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"blank RGB output frames detected: {blank[:20]}")
    os.replace(temporary, path)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to verify output video: {path}")
    try:
        actual = {
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        }
    finally:
        capture.release()
    expected = {
        "frame_count": int(len(indices)),
        "fps": float(fps),
        "width": width,
        "height": height,
    }
    if (
        actual["frame_count"] != expected["frame_count"]
        or actual["width"] != width
        or actual["height"] != height
        or abs(actual["fps"] - fps) > 0.1
    ):
        raise RuntimeError(
            f"output video verification failed: actual={actual}, expected={expected}"
        )
    return {
        "path": str(path.resolve()),
        "codec": codec,
        "actual": actual,
        "expected": expected,
    }, blank




def resample_processed_outputs(
    processed_dir: str | Path,
    timeline_manifest: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    *,
    streams: Sequence[str] = DEFAULT_STREAMS,
    frame_status_jsonl: str | Path | None = None,
    link_mode: str = "auto",
    expected_source_frames: int | None = None,
    expected_output_frames: int | None = 539,
    video_out: str | Path | None = None,
    fps: float = 50.0,
    width: int = 640,
    height: int = 480,
    codec: str = "mp4v",
    write_video: bool = True,
    force: bool = False,
    tolerance_s: float = 1e-9,
    skip_50hz_volumes: bool = False,
) -> dict[str, Any]:
    """Materialize a complete, auditable past-only output bundle."""
    if link_mode not in ("auto", "hardlink", "copy"):
        raise ValueError("link_mode must be auto, hardlink, or copy")
    processed = Path(processed_dir)
    output = Path(output_dir)
    if processed.resolve() == output.resolve():
        raise ValueError("processed_dir and output_dir must differ")
    manifest, manifest_path = _load_manifest(timeline_manifest)
    source_times, target_times, mapping, ages = _timeline_contract(
        manifest, tolerance_s=tolerance_s
    )
    if expected_source_frames is not None and len(source_times) != int(
        expected_source_frames
    ):
        raise ValueError(
            f"manifest source frame count {len(source_times)} != expected {expected_source_frames}"
        )
    if expected_output_frames is not None and len(mapping) != int(
        expected_output_frames
    ):
        raise ValueError(
            f"manifest output frame count {len(mapping)} != expected {expected_output_frames}"
        )
    if not streams or len(set(streams)) != len(streams):
        raise ValueError(
            "streams must be a non-empty sequence of unique directory names"
        )
    missing_required = sorted(set(DEFAULT_STREAMS) - set(streams))
    if missing_required:
        raise ValueError(f"required processed streams were omitted: {missing_required}")
    for stream in streams:
        if not stream or Path(stream).name != stream:
            raise ValueError(f"invalid stream directory name: {stream!r}")

    if "rgb_final" not in streams:
        raise ValueError("rgb_final must be included so the 50 Hz MP4 can be validated")
    if "provenance" not in streams:
        raise ValueError("provenance must be included so fused pixels can be rejected")

    packed_layout = _streams_are_packed(processed, streams)
    source_files: dict[str, dict[int, Path]] = {}
    packed_sources: dict[str, object] = {}
    if packed_layout:
        for stream in streams:
            packed_sources[stream] = open_volume(
                processed / stream, expected_count=len(source_times)
            )
        fused_pixels = _count_fused_volume(packed_sources["provenance"])
    else:
        source_files = {
            stream: _indexed_files(processed / stream, len(source_times))
            for stream in streams
        }
        fused_pixels = _count_fused_pixels(source_files["provenance"])
    if fused_pixels:
        raise ValueError(
            f"single-frame contract violated: found {fused_pixels} fused pixels"
        )

    status_path = (
        Path(frame_status_jsonl)
        if frame_status_jsonl is not None
        else processed / "frame_status.jsonl"
    )
    source_status = _read_frame_status(status_path, len(source_times))

    mode_counts: Counter[str] = Counter()
    output_paths: dict[str, list[Path]] = {stream: [] for stream in streams}
    packed_outputs: dict[str, object] = {}
    mapping_i64 = mapping.astype(np.int64, copy=False)
    if packed_layout and skip_50hz_volumes:
        mode_counts["skip_50hz_volumes"] += 1
        output_paths = {stream: [] for stream in streams}
    elif packed_layout:
        for stream in streams:
            source_vol = packed_sources[stream]
            dest_packed = (output / stream) / "frames.npy"
            if dest_packed.exists() and not force:
                raise FileExistsError(
                    f"output volume already exists (use --force): {dest_packed}"
                )
            destination = create_volume(
                output / stream,
                len(mapping),
                source_vol.frame_shape,
                source_vol.dtype,
                fill=None,
                color_space=source_vol.color_space,
            )
            destination.array[:] = source_vol.array[mapping_i64]
            destination.flush()
            packed_outputs[stream] = destination
            mode_counts["gather_packed"] += 1
            output_paths[stream] = [output / stream] * len(mapping)
    else:
        for stream in streams:
            destination_dir = output / stream
            for output_index, source_index in enumerate(mapping.tolist()):
                source_path = source_files[stream][source_index]
                destination = (
                    destination_dir / f"{output_index:06d}{source_path.suffix.lower()}"
                )
                materialized = _materialize(
                    source_path, destination, mode=link_mode, force=force
                )
                mode_counts[materialized] += 1
                output_paths[stream].append(destination)

    output_status: list[dict[str, Any]] = []
    previous_source: int | None = None
    for output_index, source_index in enumerate(mapping.tolist()):
        row = source_status[source_index]
        status_value = row.get("status", row.get("frame_status", "unknown"))
        output_status.append(
            {
                "output_frame_index": output_index,
                "target_frame_index": output_index,
                "output_timestamp_s": float(target_times[output_index]),
                "source_frame_index": source_index,
                "source_timestamp_s": float(source_times[source_index]),
                "source_age_ms": float(ages[output_index] * 1000.0),
                "repeated_from_previous": bool(source_index == previous_source),
                "status": status_value,
                "source_status": row,
            }
        )
        previous_source = source_index
    _write_status_jsonl(output / "frame_status.jsonl", output_status)

    mapping_sha256 = hashlib.sha256(
        mapping.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    resample_manifest = {
        "schema_version": 1,
        "timeline_manifest": manifest_path,
        "processed_dir": str(processed.resolve()),
        "output_dir": str(output.resolve()),
        "policy": "past_only_zero_order_hold",
        "output_to_source_indices": mapping.tolist(),
        "output_to_source_sha256": mapping_sha256,
        "streams": list(streams),
        "source_frame_count": len(source_times),
        "output_frame_count": len(mapping),
        "unique_source_frames": int(np.unique(mapping).size),
        "duplicate_frame_count": int(len(mapping) - np.unique(mapping).size),
        "materialization": dict(sorted(mode_counts.items())),
        "skipped_50hz_volumes": bool(skip_50hz_volumes),
    }
    _write_json(output / "resample_manifest.json", resample_manifest)

    video: dict[str, Any] | None = None
    blank_frames: list[int] = []
    if write_video:
        destination_video = (
            Path(video_out) if video_out is not None else output / "rgb_final_50hz.mp4"
        )
        if packed_layout:
            source_rgb = (
                packed_sources["rgb_final"].array
                if skip_50hz_volumes
                else packed_outputs["rgb_final"].array
            )
            video, blank_frames = _write_video_from_rgb(
                source_rgb,
                destination_video,
                fps=fps,
                width=width,
                height=height,
                codec=codec,
                force=force,
                mapping=mapping_i64 if skip_50hz_volumes else None,
            )
        else:
            video, blank_frames = _write_video(
                output_paths["rgb_final"],
                destination_video,
                fps=fps,
                width=width,
                height=height,
                codec=codec,
                force=force,
            )

    checks = {
        "source_frame_count": len(source_times)
        == (
            expected_source_frames
            if expected_source_frames is not None
            else len(source_times)
        ),
        "output_frame_count": len(mapping)
        == (
            expected_output_frames
            if expected_output_frames is not None
            else len(mapping)
        ),
        "all_stream_frame_counts": True
        if skip_50hz_volumes
        else all(
            (
                len(packed_outputs[stream]) == len(mapping)
                if packed_layout
                else len(paths) == len(mapping)
            )
            for stream, paths in output_paths.items()
        ),
        "same_mapping_all_streams": True,
        "mapping_monotonic": bool(np.all(np.diff(mapping) >= 0)),
        "mapping_no_future": bool(
            np.all(source_times[mapping] <= target_times + tolerance_s)
        ),
        "mapping_strict_no_future": bool(np.all(source_times[mapping] <= target_times)),
        "fused_pixel_count_zero": fused_pixels == 0,
        "blank_rgb_frame_count_zero": not blank_frames,
        "frame_status_count": len(output_status) == len(mapping),
        "video_spec": video is not None if write_video else True,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_frame_count": len(source_times),
        "output_frame_count": len(mapping),
        "mapping_sha256": mapping_sha256,
        "mapping_first_20": mapping[:20].tolist(),
        "mapping_last_20": mapping[-20:].tolist(),
        "unique_source_frames": int(np.unique(mapping).size),
        "duplicate_frame_count": int(len(mapping) - np.unique(mapping).size),
        "last_source_index_used": int(mapping[-1]),
        "source_age_ms": {
            "min": float(np.min(ages) * 1000.0),
            "p50": float(np.percentile(ages, 50) * 1000.0),
            "p95": float(np.percentile(ages, 95) * 1000.0),
            "max": float(np.max(ages) * 1000.0),
        },
        "fused_pixel_count": fused_pixels,
        "blank_rgb_frame_indices": blank_frames,
        "materialization": dict(sorted(mode_counts.items())),
        "video": video,
    }
    _write_json(output / "acceptance_report.json", report)
    if report["status"] != "pass":
        raise RuntimeError(f"50 Hz acceptance failed: {checks}")
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--timeline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-status-jsonl", type=Path, default=None)
    parser.add_argument("--streams", nargs="+", default=list(DEFAULT_STREAMS))
    parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "copy"), default="auto"
    )
    parser.add_argument("--expected-source-frames", type=_positive_int, default=324)
    parser.add_argument("--expected-output-frames", type=_positive_int, default=539)
    parser.add_argument("--video-out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--height", type=_positive_int, default=480)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="debug only; formal runs must encode video",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tolerance-s", type=float, default=1e-9)
    parser.add_argument(
        "--skip-50hz-volumes",
        action="store_true",
        help="Encode rgb_final_50hz.mp4 from 30 Hz packed rgb_final; do not write 50 Hz stream volumes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = resample_processed_outputs(
        args.processed_dir,
        args.timeline_manifest,
        args.output_dir,
        streams=args.streams,
        frame_status_jsonl=args.frame_status_jsonl,
        link_mode=args.link_mode,
        expected_source_frames=args.expected_source_frames,
        expected_output_frames=args.expected_output_frames,
        video_out=args.video_out,
        fps=args.fps,
        width=args.width,
        height=args.height,
        codec=args.codec,
        write_video=not args.skip_video,
        force=args.force,
        tolerance_s=args.tolerance_s,
        skip_50hz_volumes=bool(args.skip_50hz_volumes),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
