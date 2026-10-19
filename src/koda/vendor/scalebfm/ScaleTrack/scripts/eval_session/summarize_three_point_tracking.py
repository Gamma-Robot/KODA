#!/usr/bin/env python3
"""Summarize final three-point tracking without a baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_torso_umi3_tracking import (
    PRIMARY_METRICS,
    WRIST_POSITION_ACCEPTANCE_LIMIT_M,
    _atomic_json,
    _load_json,
    _load_npz,
    _run_errors,
    _sha256,
    _statistics,
    _validate_reference,
    _validate_run,
)


MINK_INVARIANT_GATES = (
    "all_output_values_finite",
    "non_arm_joint_positions_exact",
    "non_arm_joint_velocities_exact",
    "non_arm_body_pose_preserved",
    "base_velocity_reconstructed",
    "arm_joint_limits_satisfied",
    "arm_interframe_delta_bounded",
    "arm_velocity_bounded",
    "collision_clearance_at_margin",
)


def _optional_mink_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return _load_json(resolved)


def summarize_tracking(
    *,
    kinematics: Path,
    metadata: Path,
    reference: Path,
    output: Path,
    session_id: str,
    source_annotation: Path,
    source_calibration: Path,
    artifact_manifest: Path,
    pipeline_dependency_sha256: str,
    pipeline_config_id: str,
    mink_summary: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "kinematics": kinematics.expanduser().resolve(),
        "metadata": metadata.expanduser().resolve(),
        "reference": reference.expanduser().resolve(),
        "source_annotation": source_annotation.expanduser().resolve(),
        "source_calibration": source_calibration.expanduser().resolve(),
        "artifact_manifest": artifact_manifest.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    candidate = _load_npz(paths["kinematics"])
    candidate_metadata = _load_json(paths["metadata"])
    reference_arrays = _load_npz(paths["reference"])
    if "body_pos_w" not in reference_arrays or "fps" not in reference_arrays:
        raise KeyError("reference is missing body_pos_w or fps")
    frames = int(np.asarray(reference_arrays["body_pos_w"]).shape[0])
    fps = float(np.asarray(reference_arrays["fps"]).reshape(()))
    reference_names, frames, fps, timestamps = _validate_reference(
        reference_arrays,
        frames,
        fps,
    )
    candidate_names = _validate_run(
        "final Mink kinematics",
        candidate,
        candidate_metadata,
        frames,
        fps,
    )
    errors, d435_source = _run_errors(
        candidate,
        candidate_names,
        reference_arrays,
        reference_names,
    )
    metrics = {name: _statistics(errors[name]) for name in PRIMARY_METRICS}

    finite_metrics = all(
        values["finite_fraction"] == 1.0 for values in metrics.values()
    )
    wrist_position_mean = metrics["wrist_position_m"]["mean"]
    wrist_position_passed = (
        wrist_position_mean is not None
        and wrist_position_mean < WRIST_POSITION_ACCEPTANCE_LIMIT_M
    )

    mink = _optional_mink_summary(mink_summary)
    mink_gates = {} if mink is None else dict(mink.get("gates", {}))
    invariant_gates = {
        name: bool(mink_gates.get(name, False))
        for name in MINK_INVARIANT_GATES
    }
    invariant_passed = mink is None or all(invariant_gates.values())
    accepted = bool(finite_metrics and wrist_position_passed and invariant_passed)

    payload: dict[str, Any] = {
        "format": "final torso-plus-dual-wrist tracking error summary",
        "format_version": 2,
        "status": "passed" if accepted else "quality_failed",
        "accepted": accepted,
        "session_id": session_id,
        "sampling": {
            "frames": frames,
            "fps": fps,
            "duration_s": float(timestamps[-1]) if frames else 0.0,
            "sequence_policy": "complete source sequence resampled to 50 Hz",
        },
        "inputs": {
            "source_annotation": str(paths["source_annotation"]),
            "source_annotation_sha256": _sha256(paths["source_annotation"]),
            "source_calibration": str(paths["source_calibration"]),
            "source_calibration_sha256": _sha256(paths["source_calibration"]),
            "final_kinematics_sha256": _sha256(paths["kinematics"]),
            "final_metadata_sha256": _sha256(paths["metadata"]),
            "reference_sha256": _sha256(paths["reference"]),
            "artifact_manifest_sha256": _sha256(paths["artifact_manifest"]),
            "pipeline_dependency_sha256": pipeline_dependency_sha256,
            "pipeline_config_id": pipeline_config_id,
        },
        "pose_conventions": {
            "position": "world xyz in metres",
            "orientation_error": "shortest geodesic quaternion angle in degrees",
            "d435_pose": d435_source,
        },
        "metrics": metrics,
        "quality_gates": {
            "all_metrics_finite": finite_metrics,
            "wrist_position_mean_under_5cm": wrist_position_passed,
            "wrist_position_mean_m": wrist_position_mean,
            "wrist_position_mean_limit_m": WRIST_POSITION_ACCEPTANCE_LIMIT_M,
            "mink_invariants_passed": invariant_passed,
            "mink_invariants": invariant_gates,
        },
    }
    output = output.expanduser().resolve()
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-annotation", type=Path, required=True)
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--pipeline-dependency-sha256", required=True)
    parser.add_argument("--pipeline-config-id", required=True)
    parser.add_argument("--mink-summary", type=Path)
    args = parser.parse_args()
    payload = summarize_tracking(
        kinematics=args.kinematics,
        metadata=args.metadata,
        reference=args.reference,
        output=args.output,
        session_id=args.session_id,
        source_annotation=args.source_annotation,
        source_calibration=args.source_calibration,
        artifact_manifest=args.artifact_manifest,
        pipeline_dependency_sha256=args.pipeline_dependency_sha256,
        pipeline_config_id=args.pipeline_config_id,
        mink_summary=args.mink_summary,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
