#!/usr/bin/env python3
"""Verify that the render archive exactly preserves tracking state.

Dex3 renders every tracked joint one-for-one.  The BrainCo render model adds
mimic joints that no one tracks, so the archive carries their equality rules and
this check re-evaluates them instead of expecting an exact joint-set match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def strings(values: np.ndarray) -> list[str]:
    result: list[str] = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def check_mimic_joints(
    render: dict[str, np.ndarray],
    render_names: list[str],
    mimic_names: list[str],
) -> float:
    """Re-evaluate the render archive's mimic polynomials from its own qpos.

    The composer expands these joints because MuJoCo resolves ``mjEQ_JOINT``
    only in the dynamics solver.  Recomputing them here proves the expansion
    used the model's own coefficients and not a hand-written table.
    """

    reference_names = strings(render["mimic_reference_names"])
    polycoef = np.asarray(render["mimic_polycoef"], dtype=np.float64)
    qpos0 = np.asarray(render["mimic_qpos0"], dtype=np.float64)
    count = len(mimic_names)
    if len(reference_names) != count or polycoef.shape != (count, 5):
        raise ValueError("mimic provenance arrays disagree on the rule count")
    if qpos0.shape != (count, 2):
        raise ValueError(f"invalid mimic_qpos0 shape {qpos0.shape}")

    joint_pos = np.asarray(render["joint_pos"], dtype=np.float64)
    max_error = 0.0
    for index, (dependent, reference) in enumerate(zip(mimic_names, reference_names)):
        offset = joint_pos[:, render_names.index(reference)] - qpos0[index, 1]
        powers = np.stack([offset**exponent for exponent in range(5)], axis=-1)
        expected = qpos0[index, 0] + powers @ polycoef[index]
        actual = joint_pos[:, render_names.index(dependent)]
        max_error = max(max_error, float(np.max(np.abs(expected - actual))))
    # The composer saturates a mimic joint at its own XML range when the
    # polynomial and the range are rounded inconsistently, so allow a
    # milliradian of slack on top of the archive's float32 storage.
    if max_error > 2.0e-3:
        raise RuntimeError(
            f"render mimic joints deviate from their polynomial by {max_error:.9g} rad"
        )
    return max_error




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument(
        "--hand-dof",
        type=int,
        default=14,
        help="Tracked hand width: 14 for Dex3, 12 for the BrainCo hands.",
    )
    parser.add_argument(
        "--render-dof",
        type=int,
        default=43,
        help="Hinge count of the render model: 43 for Dex3, 51 for BrainCo.",
    )
    args = parser.parse_args()

    tracking_path = args.tracking.expanduser().resolve()
    render_path = args.render.expanduser().resolve()
    tracking = load(tracking_path)
    render = load(render_path)
    required_tracking = {
        "joint_pos",
        "joint_vel",
        "joint_names",
        "timestamps",
        "hand_qpos",
        "hand_qvel",
        "hand_joint_names",
    }
    required_render = required_tracking | {"body_pos_w", "body_quat_w"}
    missing_tracking = sorted(required_tracking - tracking.keys())
    missing_render = sorted(required_render - render.keys())
    if missing_tracking or missing_render:
        raise KeyError(
            f"missing tracking fields={missing_tracking}, render fields={missing_render}"
        )

    tracking_body_names = strings(tracking["joint_names"])
    tracking_hand_names = strings(tracking["hand_joint_names"])
    render_names = strings(render["joint_names"])
    render_hand_names = strings(render["hand_joint_names"])
    if len(tracking_body_names) != 29 or len(tracking_hand_names) != args.hand_dof:
        raise ValueError(
            f"expected tracking 29+{args.hand_dof} DoF, got "
            f"{len(tracking_body_names)}+{len(tracking_hand_names)}"
        )
    if len(render_names) != args.render_dof:
        raise ValueError(
            f"expected a {args.render_dof}-DoF render archive, got {len(render_names)}"
        )
    if set(tracking_body_names) & set(tracking_hand_names):
        raise ValueError("tracking body and hand joint schemas overlap")

    mimic_names = strings(render.get("mimic_joint_names", np.asarray([])))
    tracked_names = set(tracking_body_names) | set(tracking_hand_names)
    if set(render_names) != tracked_names | set(mimic_names):
        raise ValueError(
            "render archive is not the exact union of tracking joints and the "
            "render model's mimic joints"
        )
    if mimic_names:
        # The render archive lists hand joints in MuJoCo order, which interleaves
        # each mimic joint with the joint that drives it.
        if sorted(render_hand_names) != sorted(
            set(tracking_hand_names) | set(mimic_names)
        ):
            raise ValueError(
                "render hand schema is not the tracking hand schema plus mimics"
            )
    elif render_hand_names != tracking_hand_names:
        raise ValueError("render joint schema does not preserve tracking hand schema")

    body_indices = np.asarray(
        [render_names.index(name) for name in tracking_body_names], dtype=np.int64
    )
    hand_indices = np.asarray(
        [render_names.index(name) for name in tracking_hand_names], dtype=np.int64
    )
    render_hand_indices = np.asarray(
        [render_hand_names.index(name) for name in tracking_hand_names], dtype=np.int64
    )
    comparisons = {
        "body_joint_pos": (
            np.asarray(tracking["joint_pos"]),
            np.asarray(render["joint_pos"])[:, body_indices],
        ),
        "body_joint_vel": (
            np.asarray(tracking["joint_vel"]),
            np.asarray(render["joint_vel"])[:, body_indices],
        ),
        "hand_qpos": (
            np.asarray(tracking["hand_qpos"]),
            np.asarray(render["joint_pos"])[:, hand_indices],
        ),
        "hand_qvel": (
            np.asarray(tracking["hand_qvel"]),
            np.asarray(render["joint_vel"])[:, hand_indices],
        ),
        "render_hand_qpos": (
            np.asarray(tracking["hand_qpos"]),
            np.asarray(render["hand_qpos"])[:, render_hand_indices],
        ),
        "render_hand_qvel": (
            np.asarray(tracking["hand_qvel"]),
            np.asarray(render["hand_qvel"])[:, render_hand_indices],
        ),
        "timestamps": (
            np.asarray(tracking["timestamps"]),
            np.asarray(render["timestamps"]),
        ),
    }
    for label, (expected, actual) in comparisons.items():
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            max_error = None
            if expected.shape == actual.shape and np.issubdtype(expected.dtype, np.number):
                max_error = float(
                    np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))
                )
            raise RuntimeError(
                f"tracking/render mismatch for {label}: expected={expected.shape} "
                f"actual={actual.shape} max_error={max_error}"
            )

    mimic_max_error = 0.0
    if mimic_names:
        mimic_max_error = check_mimic_joints(render, render_names, mimic_names)

    report = {
        "tracking": str(tracking_path),
        "render": str(render_path),
        "frames": int(np.asarray(tracking["joint_pos"]).shape[0]),
        "body_dof": 29,
        "hand_dof": int(args.hand_dof),
        "render_dof": int(args.render_dof),
        "mimic_joint_count": len(mimic_names),
        "max_mimic_reconstruction_error_rad": mimic_max_error,
        "body_exact": True,
        "hand_exact": True,
        "timestamps_exact": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
