#!/usr/bin/env python3
"""Build the measured 30 Hz source to 50 Hz D435 timeline contract.

There are deliberately two mappings in the contract:

* each source frame uses the absolutely nearest D435 sample for rendering;
* each D435 output time holds the newest processed source frame that is not
  from the future.

The mapping helpers are NumPy-only pure functions.  HDF5 is imported lazily so
callers that already have timestamp arrays do not need an HDF5 runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_SOURCE_DATASET = "/cam-pose/timestamps"
DEFAULT_TARGET_KEY = "timestamps"


def _validated_timestamps(
    values: Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"{name} timestamps must be one-dimensional, got {array.shape}"
        )
    if array.size == 0:
        raise ValueError(f"{name} timestamps are empty")
    if not np.all(np.isfinite(array)):
        bad = np.flatnonzero(~np.isfinite(array))[:8].tolist()
        raise ValueError(f"{name} timestamps contain non-finite values at {bad}")
    decreasing = np.flatnonzero(np.diff(array) < 0.0)
    if decreasing.size:
        raise ValueError(
            f"{name} timestamps must be nondecreasing; decreases after indices "
            f"{decreasing[:8].tolist()}"
        )
    return array


def relative_timestamps(
    values: Sequence[float] | np.ndarray, *, name: str = "stream"
) -> np.ndarray:
    """Return a validated timestamp sequence relative to its first sample."""
    array = _validated_timestamps(values, name)
    return array - array[0]


def map_source_to_nearest_target(
    source_timestamps_s: Sequence[float] | np.ndarray,
    target_timestamps_s: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Map each source sample to the nearest target, breaking ties earlier.

    "Earlier" means the smaller target timestamp.  If the target contains
    duplicate timestamps, the first frame at the selected timestamp is used.
    """
    source = _validated_timestamps(source_timestamps_s, "source")
    target = _validated_timestamps(target_timestamps_s, "target")

    insertion = np.searchsorted(target, source, side="left")
    right = np.clip(insertion, 0, target.size - 1)
    left = np.clip(insertion - 1, 0, target.size - 1)
    left_distance = np.abs(source - target[left])
    right_distance = np.abs(target[right] - source)
    selected = np.where(left_distance <= right_distance, left, right).astype(np.int64)

    # Stable tie handling also applies when multiple target frames share one
    # timestamp: use the earliest frame index for that selected time.
    selected = np.searchsorted(target, target[selected], side="left").astype(np.int64)
    return selected


def map_target_to_past_source(
    source_timestamps_s: Sequence[float] | np.ndarray,
    target_timestamps_s: Sequence[float] | np.ndarray,
    *,
    tolerance_s: float = 1e-9,
) -> np.ndarray:
    """Map target times to ``max(i | source[i] <= target)``.

    A target time before the first source sample has no causal answer and is an
    error.  ``tolerance_s`` only absorbs floating-point roundoff at equality;
    the returned mapping is checked again against the strict no-future rule.
    """
    source = _validated_timestamps(source_timestamps_s, "source")
    target = _validated_timestamps(target_timestamps_s, "target")
    tolerance_s = float(tolerance_s)
    if not np.isfinite(tolerance_s) or tolerance_s < 0.0:
        raise ValueError("tolerance_s must be finite and non-negative")

    selected = np.searchsorted(source, target + tolerance_s, side="right") - 1
    missing = np.flatnonzero(selected < 0)
    if missing.size:
        first = int(missing[0])
        raise ValueError(
            "past-only mapping has no source frame for target index "
            f"{first} at {target[first]:.12g}s (first source {source[0]:.12g}s)"
        )
    selected = selected.astype(np.int64)
    future = source[selected] - target
    bad = np.flatnonzero(future > tolerance_s)
    if bad.size:
        first = int(bad[0])
        raise AssertionError(
            f"past-only mapping selected a future source at target {first}: "
            f"delta={future[first]:.12g}s"
        )
    return selected


def nearest_time_indices(
    query_timestamps_s: Sequence[float] | np.ndarray,
    sample_timestamps_s: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest sample indices and ``sample_time - query_time`` offsets."""
    query = _validated_timestamps(query_timestamps_s, "query")
    samples = _validated_timestamps(sample_timestamps_s, "samples")
    indices = map_source_to_nearest_target(query, samples)
    return indices, (samples[indices] - query).astype(np.float64)


def previous_time_indices(
    query_timestamps_s: Sequence[float] | np.ndarray,
    sample_timestamps_s: Sequence[float] | np.ndarray,
    *,
    tolerance_s: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return previous sample indices and non-negative ``query - sample`` ages."""
    query = _validated_timestamps(query_timestamps_s, "query")
    samples = _validated_timestamps(sample_timestamps_s, "samples")
    indices = map_target_to_past_source(samples, query, tolerance_s=tolerance_s)
    return indices, (query - samples[indices]).astype(np.float64)


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def build_timeline_manifest(
    source_timestamps_s: Sequence[float] | np.ndarray,
    target_timestamps_s: Sequence[float] | np.ndarray,
    *,
    source_path: str | Path | None = None,
    target_path: str | Path | None = None,
    source_dataset: str = DEFAULT_SOURCE_DATASET,
    target_key: str = DEFAULT_TARGET_KEY,
    expected_source_frames: int | None = None,
    expected_target_frames: int | None = None,
    tolerance_s: float = 1e-9,
) -> dict[str, Any]:
    """Create a JSON-serialisable timeline manifest from measured timestamps."""
    source_absolute = _validated_timestamps(source_timestamps_s, "source")
    target_absolute = _validated_timestamps(target_timestamps_s, "target")
    if expected_source_frames is not None and source_absolute.size != int(
        expected_source_frames
    ):
        raise ValueError(
            f"source frame count {source_absolute.size} != expected {expected_source_frames}"
        )
    if expected_target_frames is not None and target_absolute.size != int(
        expected_target_frames
    ):
        raise ValueError(
            f"target frame count {target_absolute.size} != expected {expected_target_frames}"
        )

    source_relative = source_absolute - source_absolute[0]
    target_relative = target_absolute - target_absolute[0]
    nearest = map_source_to_nearest_target(source_relative, target_relative)
    held = map_target_to_past_source(
        source_relative, target_relative, tolerance_s=tolerance_s
    )

    nearest_signed_s = target_relative[nearest] - source_relative
    nearest_abs_ms = np.abs(nearest_signed_s) * 1000.0
    source_age_s = target_relative - source_relative[held]
    source_age_ms = source_age_s * 1000.0
    future_lead_s = np.maximum(source_relative[held] - target_relative, 0.0)
    strict_no_future = bool(np.all(future_lead_s == 0.0))
    no_future = bool(np.all(future_lead_s <= tolerance_s))
    monotonic = bool(np.all(np.diff(held) >= 0))
    overlap = bool(
        max(float(source_relative[0]), float(target_relative[0]))
        <= min(float(source_relative[-1]), float(target_relative[-1])) + tolerance_s
    )
    if not overlap:
        raise ValueError("source and target relative timelines do not overlap")
    if not no_future or not monotonic:
        raise AssertionError("causal output mapping invariant failed")

    source_path_text = (
        str(Path(source_path).resolve()) if source_path is not None else None
    )
    target_path_text = (
        str(Path(target_path).resolve()) if target_path is not None else None
    )
    source_indices = held.tolist()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "time_unit": "seconds",
        "source": {
            "kind": "source_camera",
            "path": source_path_text,
            "timestamp_dataset": source_dataset,
            "frame_count": int(source_absolute.size),
            "time_origin_s": float(source_absolute[0]),
            "timestamps_absolute_s": source_absolute.tolist(),
            "timestamps_relative_s": source_relative.tolist(),
        },
        "target": {
            "kind": "d435",
            "path": target_path_text,
            "timestamp_key": target_key,
            "frame_count": int(target_absolute.size),
            "time_origin_s": float(target_absolute[0]),
            "timestamps_absolute_s": target_absolute.tolist(),
            "timestamps_relative_s": target_relative.tolist(),
        },
        "source_to_target": {
            "policy": "absolute_nearest_tie_earlier",
            "target_indices": nearest.tolist(),
            "signed_offset_s": nearest_signed_s.tolist(),
            "absolute_offset_ms": nearest_abs_ms.tolist(),
            "absolute_offset_ms_stats": _distribution(nearest_abs_ms),
        },
        "output_50hz": {
            "policy": "past_only_zero_order_hold",
            "timeline": "target",
            "target_indices": list(range(int(target_relative.size))),
            "source_indices": source_indices,
            "source_age_s": source_age_s.tolist(),
            "source_age_ms": source_age_ms.tolist(),
            "source_age_ms_stats": _distribution(source_age_ms),
            "frame_count": int(target_relative.size),
            "unique_source_frames": int(np.unique(held).size),
            "duplicate_frame_count": int(held.size - np.unique(held).size),
            "last_source_index_used": int(held[-1]),
            "no_future": no_future,
            "strict_no_future": strict_no_future,
            "max_numerical_future_lead_s": float(np.max(future_lead_s)),
            "monotonic": monotonic,
        },
        "checks": {
            "source_timestamps_nondecreasing": True,
            "target_timestamps_nondecreasing": True,
            "timelines_overlap": overlap,
            "source_to_target_in_range": bool(
                np.all((nearest >= 0) & (nearest < target_relative.size))
            ),
            "output_source_indices_in_range": bool(
                np.all((held >= 0) & (held < source_relative.size))
            ),
            "output_mapping_monotonic": monotonic,
            "output_mapping_no_future": no_future,
            "output_mapping_strict_no_future": strict_no_future,
        },
    }
    return manifest


def load_hdf5_timestamps(
    path: str | Path, dataset: str = DEFAULT_SOURCE_DATASET
) -> np.ndarray:
    """Load one timestamp vector from HDF5 with a clear runtime error."""
    hdf5_path = Path(path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"annotation HDF5 not found: {hdf5_path}")
    try:
        import h5py
    except Exception as exc:  # binary ABI errors are not ImportError
        raise RuntimeError(
            "h5py is unavailable or binary-incompatible; run with the episode "
            f"environment that can read HDF5 ({exc})"
        ) from exc
    with h5py.File(hdf5_path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"timestamp dataset {dataset!r} not found in {hdf5_path}")
        values = np.asarray(handle[dataset])
    return _validated_timestamps(values, "source")


def load_npz_timestamps(path: str | Path, key: str = DEFAULT_TARGET_KEY) -> np.ndarray:
    """Load one timestamp vector from a NumPy archive."""
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"target kinematics NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(
                f"timestamp key {key!r} not found in {npz_path}: {archive.files}"
            )
        values = np.asarray(archive[key])
    return _validated_timestamps(values, "target")


def create_timeline_manifest_from_files(
    annotation_hdf5: str | Path,
    target_kinematics_npz: str | Path,
    *,
    source_dataset: str = DEFAULT_SOURCE_DATASET,
    target_key: str = DEFAULT_TARGET_KEY,
    max_source_frames: int | None = None,
    max_target_frames: int | None = None,
    expected_source_frames: int | None = None,
    expected_target_frames: int | None = None,
    tolerance_s: float = 1e-9,
) -> dict[str, Any]:
    """Read the experiment files and build their shared timeline manifest."""
    source = load_hdf5_timestamps(annotation_hdf5, source_dataset)
    target = load_npz_timestamps(target_kinematics_npz, target_key)
    if max_source_frames is not None:
        if max_source_frames <= 0:
            raise ValueError("max_source_frames must be positive")
        source = source[: int(max_source_frames)]
    if max_target_frames is not None:
        if max_target_frames <= 0:
            raise ValueError("max_target_frames must be positive")
        target = target[: int(max_target_frames)]
    return build_timeline_manifest(
        source,
        target,
        source_path=annotation_hdf5,
        target_path=target_kinematics_npz,
        source_dataset=source_dataset,
        target_key=target_key,
        expected_source_frames=expected_source_frames,
        expected_target_frames=expected_target_frames,
        tolerance_s=tolerance_s,
    )


def write_timeline_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n")


def _positive_or_none(value: str | int) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("frame count must be non-negative")
    return parsed or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-hdf5", type=Path, required=True)
    parser.add_argument("--target-kinematics-npz", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="timeline_manifest.json"
    )
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--target-key", default=DEFAULT_TARGET_KEY)
    parser.add_argument("--max-source-frames", type=_positive_or_none, default=None)
    parser.add_argument("--max-target-frames", type=_positive_or_none, default=None)
    parser.add_argument(
        "--expected-source-frames",
        type=_positive_or_none,
        default=None,
        help="fail unless the selected source timeline has this many frames (0 disables)",
    )
    parser.add_argument(
        "--expected-target-frames",
        type=_positive_or_none,
        default=None,
        help="fail unless the selected target timeline has this many frames (0 disables)",
    )
    parser.add_argument("--tolerance-s", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_timeline_manifest_from_files(
        args.annotation_hdf5,
        args.target_kinematics_npz,
        source_dataset=args.source_dataset,
        target_key=args.target_key,
        max_source_frames=args.max_source_frames,
        max_target_frames=args.max_target_frames,
        expected_source_frames=args.expected_source_frames,
        expected_target_frames=args.expected_target_frames,
        tolerance_s=args.tolerance_s,
    )
    write_timeline_manifest(args.out, manifest)
    summary = {
        "timeline_manifest": str(args.out.resolve()),
        "source_frames": manifest["source"]["frame_count"],
        "target_frames": manifest["target"]["frame_count"],
        "nearest_offset_ms": manifest["source_to_target"]["absolute_offset_ms_stats"],
        "output_source_age_ms": manifest["output_50hz"]["source_age_ms_stats"],
        "duplicate_output_frames": manifest["output_50hz"]["duplicate_frame_count"],
        "no_future": manifest["output_50hz"]["no_future"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
