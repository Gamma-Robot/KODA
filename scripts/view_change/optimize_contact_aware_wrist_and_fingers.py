#!/usr/bin/env python3
"""Jointly refine a contact-aware wrist pose and Dex3 finger qpos.

It starts from the dense wrist-only
contact candidate and the validated ``position`` finger retargeting, then
optimizes the right wrist SE(3) residual and all seven right Dex3 hinge values
per contact frame.  The left hand and non-contact finger trajectory are kept
unchanged; short native-rate transitions avoid a qpos discontinuity at the
contact interval edges.  The resulting wrist candidate and 50 Hz hand archive
are consumed by the formal contact-aware tracking/view-change entry points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from cup_eval_fast import acquire_eval_buffers, evaluate_packed_configuration
from evaluate_hand_object_interaction import (
    HAND_MODE_BRAINCO,
    HAND_MODE_DEX3,
    CupModel,
    Dex3MeshSampler,
    finite_stats,
)
from optimize_contact_aware_wrist import load_cup_models


FINGER_GROUPS = ("thumb", "index", "middle")
RIGHT_SLICE = slice(7, 14)
RIGHT_LIMITS = np.asarray(
    [
        (-1.04719755, 1.04719755),
        (-1.04719755, 0.72431163),
        (-1.74532925, 0.0),
        (0.0, 1.57079632),
        (0.0, 1.74532925),
        (0.0, 1.57079632),
        (0.0, 1.74532925),
    ],
    dtype=np.float64,
)


def hand_side_slice(hand_mode: str, side: str) -> slice:
    if hand_mode == HAND_MODE_DEX3:
        if side != "right":
            raise ValueError("Dex3 joint contact supports --side right only")
        return RIGHT_SLICE
    if side == "left":
        return slice(0, 6)
    if side == "right":
        return slice(6, 12)
    raise ValueError(f"unsupported side {side}")


def hand_dof(hand_mode: str) -> int:
    return 14 if hand_mode == HAND_MODE_DEX3 else 12


def side_joint_count(hand_mode: str) -> int:
    return 7 if hand_mode == HAND_MODE_DEX3 else 6


def joint_contact_contract(hand_mode: str) -> str:
    if hand_mode == HAND_MODE_DEX3:
        return "right_wrist_yaw_link_SE3_plus_right_Dex3_7DoF"
    return "dual_wrist_yaw_link_SE3_plus_g1_brainco_6DoF_per_hand"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--interaction-dir", type=Path, required=True)
    parser.add_argument("--wrist-candidate", type=Path, required=True)
    parser.add_argument("--base-hand-50hz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--keyframe", type=int, default=None)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--hand-mode",
        choices=(HAND_MODE_DEX3, HAND_MODE_BRAINCO),
        default=HAND_MODE_DEX3,
    )
    parser.add_argument("--mesh-samples-per-geom", type=int, default=1600)
    parser.add_argument("--contact-threshold-m", type=float, default=0.008)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.005)
    parser.add_argument("--penetration-ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--robust-contact-m", type=float, default=0.007)
    parser.add_argument("--robust-penetration-m", type=float, default=0.0045)
    parser.add_argument("--robust-penetration-ratio", type=float, default=0.018)
    parser.add_argument("--penetration-target-m", type=float, default=0.0015)
    parser.add_argument("--penetration-ratio-target", type=float, default=0.010)
    parser.add_argument("--penetration-depth-weight", type=float, default=55.0)
    parser.add_argument("--penetration-ratio-weight", type=float, default=6.0)
    parser.add_argument("--fingertip-penetration-max-m", type=float, default=0.003)
    parser.add_argument("--fingertip-penetration-p95-m", type=float, default=0.0015)
    parser.add_argument("--fingertip-penetration-ratio", type=float, default=0.01)
    parser.add_argument("--fingertip-penetration-weight", type=float, default=0.0)
    parser.add_argument("--wrist-translation-bound-m", type=float, default=0.015)
    parser.add_argument("--wrist-rotation-bound-deg", type=float, default=12.0)
    parser.add_argument("--joint-delta-bound-rad", type=float, default=0.28)
    parser.add_argument("--transition-frames", type=int, default=15)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def apply_wrist_delta(
    position: np.ndarray, rotation: np.ndarray, delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(delta, dtype=np.float64)
    return (
        position + rotation @ value[:3],
        rotation @ Rotation.from_rotvec(value[3:]).as_matrix(),
    )


def annotate(metric: dict[str, Any]) -> dict[str, Any]:
    for group in FINGER_GROUPS:
        metric[f"{group}_surface_distance_m"] = float(
            metric["groups"][group]["surface_distance_min_m"]
        )
        metric[f"{group}_contact"] = bool(metric["groups"][group]["contact"])
        for name in ("penetration_max_m", "penetration_p95_m", "penetration_ratio"):
            metric[f"{group}_{name}"] = float(metric["groups"][group][name])
    metric["all_finger_contact"] = all(
        bool(metric[f"{group}_contact"]) for group in FINGER_GROUPS
    )
    metric["all_finger_valid_grasp"] = bool(
        metric["all_finger_contact"]
        and metric["bilateral_contact"]
        and metric["hand_penetration_p95_m"] <= 0.005
        and metric["hand_penetration_ratio"] <= 0.02
    )
    metric["robust_all_finger"] = bool(
        metric["all_finger_contact"]
        and metric["bilateral_contact"]
        and metric["hand_penetration_p95_m"] <= 0.0045
        and metric["hand_penetration_ratio"] <= 0.018
        and all(
            metric[f"{group}_surface_distance_m"] <= 0.007
            for group in FINGER_GROUPS
        )
    )
    return metric


def objective(
    metric: dict[str, Any],
    x: np.ndarray,
    previous: np.ndarray,
    args: argparse.Namespace,
) -> float:
    distances = [metric[f"{group}_surface_distance_m"] for group in FINGER_GROUPS]
    contact = 6.0 * sum(max(float(value) - 0.0025, 0.0) for value in distances)
    missing = 0.05 * sum(
        not bool(metric[f"{group}_contact"]) for group in FINGER_GROUPS
    )
    penetration = (
        args.penetration_depth_weight
        * max(float(metric["hand_penetration_p95_m"]) - args.penetration_target_m, 0.0)
        + args.penetration_ratio_weight
        * max(float(metric["hand_penetration_ratio"]) - args.penetration_ratio_target, 0.0)
    )
    fingertip_penetration = args.fingertip_penetration_weight * sum(
        max(float(metric[f"{group}_penetration_max_m"]) - args.fingertip_penetration_max_m, 0.0)
        + max(float(metric[f"{group}_penetration_p95_m"]) - args.fingertip_penetration_p95_m, 0.0)
        + 0.02 * max(float(metric[f"{group}_penetration_ratio"]) - args.fingertip_penetration_ratio, 0.0)
        for group in ("index", "middle")
    )
    wrist_anchor = 0.10 * np.linalg.norm(x[:3]) + 0.006 * np.linalg.norm(x[3:6])
    joint_anchor = 0.018 * np.linalg.norm(x[6:])
    temporal = 0.35 * np.linalg.norm(x[:6] - previous[:6]) + 0.025 * np.linalg.norm(
        x[6:] - previous[6:]
    )
    return float(contact + missing + penetration + fingertip_penetration + wrist_anchor + joint_anchor + temporal)


def robust(metric: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        metric["all_finger_contact"]
        and metric["bilateral_contact"]
        and metric["hand_penetration_p95_m"] <= args.robust_penetration_m
        and metric["hand_penetration_ratio"] <= args.robust_penetration_ratio
        and all(
            metric[f"{group}_penetration_max_m"] <= args.fingertip_penetration_max_m
            and metric[f"{group}_penetration_p95_m"] <= args.fingertip_penetration_p95_m
            and metric[f"{group}_penetration_ratio"] <= args.fingertip_penetration_ratio
            for group in ("index", "middle")
        )
        and all(
            metric[f"{group}_surface_distance_m"] <= args.robust_contact_m
            for group in FINGER_GROUPS
        )
    )


def evaluate(
    q_base: np.ndarray,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    x: np.ndarray,
    sampler: Dex3MeshSampler,
    cup: CupModel,
    args: argparse.Namespace,
    side_slice: slice,
    joint_limits: np.ndarray,
    packed: np.ndarray,
    offsets: np.ndarray,
    names: tuple[str, ...],
    world_buf: np.ndarray,
    stats_buf: np.ndarray,
    scratch_buf: np.ndarray,
) -> dict[str, Any]:
    q = q_base.copy()
    q[side_slice] = q[side_slice] + x[6:]
    q[side_slice] = np.clip(q[side_slice], joint_limits[:, 0], joint_limits[:, 1])
    sampler.fill_packed_local_samples(q, packed)
    position, rotation = apply_wrist_delta(wrist_position, wrist_rotation, x[:6])
    metric, _ = evaluate_packed_configuration(
        packed,
        offsets,
        names,
        position,
        rotation,
        cup,
        args.contact_threshold_m,
        args.penetration_tolerance_m,
        args.penetration_ratio_tolerance,
        return_world_samples=False,
        world_buf=world_buf,
        stats_buf=stats_buf,
        scratch_buf=scratch_buf,
    )
    metric = annotate(metric)
    metric["joint_delta_norm_rad"] = float(np.linalg.norm(x[6:]))
    metric["wrist_delta_translation_norm_m"] = float(np.linalg.norm(x[:3]))
    metric["wrist_delta_rotation_norm_deg"] = float(
        np.degrees(np.linalg.norm(x[3:6]))
    )
    metric["qpos"] = q
    return metric


def finite_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if len(values) > 1:
        result[0] = (values[1] - values[0]) / max(times[1] - times[0], 1e-6)
        result[-1] = (values[-1] - values[-2]) / max(times[-1] - times[-2], 1e-6)
        for index in range(1, len(values) - 1):
            result[index] = (values[index + 1] - values[index - 1]) / max(
                times[index + 1] - times[index - 1], 1e-6
            )
    return result.astype(np.float32)


def main() -> None:
    args = parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    interaction = args.interaction_dir.expanduser().resolve()
    wrist_path = args.wrist_candidate.expanduser().resolve()
    base_hand_path = args.base_hand_50hz.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    finger_path = experiment / "finger_candidates_native.npz"
    object_path = interaction / "object_models.jsonl"
    for path in (finger_path, object_path, wrist_path, base_hand_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(finger_path, allow_pickle=False) as archive:
        finger = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(wrist_path, allow_pickle=False) as archive:
        wrist = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(base_hand_path, allow_pickle=False) as archive:
        base_hand = {key: np.asarray(archive[key]) for key in archive.files}
    q_native = np.asarray(finger["position__qpos"], dtype=np.float64)
    native_time = np.asarray(finger["timestamps"], dtype=np.float64)
    base_wrist_position = np.asarray(wrist["causal_position"], dtype=np.float64)
    base_wrist_rotation = np.asarray(wrist["causal_rotation"], dtype=np.float64)
    frames = list(range(int(args.start_frame), int(args.end_frame) + 1))
    if not frames:
        raise ValueError("empty contact interval")
    cups = load_cup_models(object_path)
    if any(frame not in cups for frame in frames):
        raise KeyError("cup models do not cover the requested contact interval")
    hand_mode = str(args.hand_mode)
    side = str(args.side)
    side_slice = hand_side_slice(hand_mode, side)
    n_side = side_joint_count(hand_mode)
    n_hand = hand_dof(hand_mode)
    n_params = 6 + n_side
    contract = joint_contact_contract(hand_mode)
    if q_native.ndim != 2 or q_native.shape[1] != n_hand:
        raise ValueError(
            f"native finger qpos must be (*, {n_hand}) for {hand_mode}, "
            f"got {q_native.shape}"
        )
    sampler = Dex3MeshSampler(
        side,
        int(args.mesh_samples_per_geom),
        int(args.seed),
        hand_mode=hand_mode,
    )
    joint_limits = (
        RIGHT_LIMITS if hand_mode == HAND_MODE_DEX3 else sampler.side_joint_limits()
    )
    if joint_limits.shape != (n_side, 2):
        raise ValueError(f"unexpected joint limits {joint_limits.shape}")
    packed = np.empty((sampler.packed_count, 3), dtype=np.float64)
    packed_names = sampler.packed_names
    packed_offsets = sampler.packed_offsets
    world_buf, stats_buf, scratch_buf = acquire_eval_buffers(
        sampler.packed_count, len(packed_names)
    )
    bounds = (
        [(-args.wrist_translation_bound_m, args.wrist_translation_bound_m)] * 3
        + [(-np.deg2rad(args.wrist_rotation_bound_deg), np.deg2rad(args.wrist_rotation_bound_deg))] * 3
        + [(-args.joint_delta_bound_rad, args.joint_delta_bound_rad)] * n_side
    )
    deltas = np.zeros((len(frames), n_params), dtype=np.float64)
    records: list[dict[str, Any]] = []
    previous = np.zeros(n_params, dtype=np.float64)
    for output_index, frame in enumerate(frames):
        q_base = q_native[frame].copy()
        wrist_position = base_wrist_position[frame].copy()
        wrist_rotation = base_wrist_rotation[frame].copy()

        def evaluate_x(x: np.ndarray) -> dict[str, Any]:
            return evaluate(
                q_base, wrist_position, wrist_rotation, np.asarray(x), sampler,
                cups[frame], args, side_slice, joint_limits,
                packed, packed_offsets, packed_names,
                world_buf, stats_buf, scratch_buf,
            )

        initial_metric = evaluate_x(previous)
        initial_value = objective(initial_metric, previous, previous, args)

        def loss(x: np.ndarray) -> float:
            metric = evaluate_x(x)
            return objective(metric, np.asarray(x), previous, args)

        result = minimize(
            loss,
            previous.copy(),
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(args.max_iterations), "xtol": 2e-4, "ftol": 1e-6},
        )
        optimized = np.asarray(result.x, dtype=np.float64)
        optimized_metric = evaluate_x(optimized)
        choices = [("initial", previous.copy(), initial_metric, initial_value)]
        choices.append(("powell", optimized, optimized_metric, float(result.fun)))
        choices.sort(
            key=lambda item: (
                not robust(item[2], args),
                not bool(item[2]["all_finger_valid_grasp"]),
                float(item[3]),
            )
        )
        solver, selected, metric, value = choices[0]
        deltas[output_index] = selected
        previous = selected.copy()
        metric = {key: value for key, value in metric.items() if key != "qpos"}
        metric.update(
            {
                "frame": frame,
                "solver": solver,
                "objective": float(value),
                "function_evaluations": int(getattr(result, "nfev", 1)),
                "wrist_delta_local": selected[:6].tolist(),
                "joint_delta_side": selected[6:].tolist(),
                "joint_delta_right": selected[6:].tolist(),
                "side": side,
                "hand_mode": hand_mode,
            }
        )
        records.append(metric)
        print(
            f"[joint-contact] frame={frame} solver={solver} "
            f"all={int(metric['all_finger_contact'])} robust={int(metric['robust_all_finger'])} "
            f"pen={metric['hand_penetration_ratio']:.4f}",
            flush=True,
        )

    q_native_out = q_native.copy()
    first, last = frames[0], frames[-1]
    for index, frame in enumerate(frames):
        q_native_out[frame, side_slice] = np.clip(
            q_native[frame, side_slice] + deltas[index, 6:],
            joint_limits[:, 0], joint_limits[:, 1],
        )
    transition = max(0, int(args.transition_frames))
    for frame in range(max(0, first - transition), first):
        alpha = (frame - (first - transition) + 1) / (transition + 1)
        q_native_out[frame, side_slice] = q_native[frame, side_slice] + alpha * (
            q_native_out[first, side_slice] - q_native[first, side_slice]
        )
    for frame in range(last + 1, min(len(q_native), last + transition + 1)):
        alpha = 1.0 - (frame - last) / (transition + 1)
        q_native_out[frame, side_slice] = q_native[frame, side_slice] + alpha * (
            q_native_out[last, side_slice] - q_native[last, side_slice]
        )

    optimized_finger = {key: value.copy() for key, value in finger.items()}
    optimized_finger["position__qpos"] = q_native_out.astype(np.float32)
    optimized_finger["joint_contact_optimized_frames"] = np.asarray(frames, dtype=np.int64)
    optimized_finger["joint_contact_wrist_delta"] = deltas[:, :6]
    optimized_finger["joint_contact_right_qpos_delta"] = deltas[:, 6:]
    optimized_finger["joint_contact_qpos_delta"] = deltas[:, 6:]
    optimized_finger["joint_contact_contract"] = np.asarray(contract)
    np.savez_compressed(output / "finger_candidates_joint_contact_native.npz", **optimized_finger)

    optimized_wrist = {key: value.copy() for key, value in wrist.items()}
    optimized_causal_position = np.asarray(
        wrist["causal_position"], dtype=np.float64
    ).copy()
    optimized_causal_rotation = np.asarray(
        wrist["causal_rotation"], dtype=np.float64
    ).copy()
    for index, frame in enumerate(frames):
        optimized_causal_position[frame], optimized_causal_rotation[frame] = (
            apply_wrist_delta(
                optimized_causal_position[frame],
                optimized_causal_rotation[frame],
                deltas[index, :6],
            )
        )
    optimized_wrist["causal_position"] = optimized_causal_position
    optimized_wrist["causal_rotation"] = optimized_causal_rotation
    optimized_wrist["joint_contact_optimized_frames"] = np.asarray(
        frames, dtype=np.int64
    )
    optimized_wrist["joint_contact_contract"] = np.asarray(contract)
    np.savez_compressed(
        output / "contact_aware_wrist_candidates_joint_contact_native.npz",
        **optimized_wrist,
    )

    target_time = np.asarray(base_hand["timestamps"], dtype=np.float64)
    native_query = np.clip(target_time, native_time[0], native_time[-1])
    base_qpos_key = "qpos" if "qpos" in base_hand else "hand_qpos"
    q50 = np.asarray(base_hand[base_qpos_key], dtype=np.float64).copy()
    if q50.ndim != 2 or q50.shape[1] != n_hand:
        raise ValueError(
            f"base hand qpos must be (*, {n_hand}) for {hand_mode}, got {q50.shape}"
        )
    for joint in range(side_slice.start, side_slice.stop):
        q50[:, joint] = np.interp(native_query, native_time, q_native_out[:, joint])
    qvel = finite_difference(q50, target_time)
    hand50 = {key: value.copy() for key, value in base_hand.items()}
    hand50["qpos"] = q50.astype(np.float32)
    hand50["qvel"] = qvel
    if n_hand == 14:
        hand50["right_qpos"] = q50[:, 7:].astype(np.float32)
        hand50["right_qvel"] = qvel[:, 7:]
    else:
        hand50["left_qpos"] = q50[:, :6].astype(np.float32)
        hand50["right_qpos"] = q50[:, 6:].astype(np.float32)
        hand50["left_qvel"] = qvel[:, :6]
        hand50["right_qvel"] = qvel[:, 6:]
    # compose_dex3_kinematics.py treats `joint_names` as the hand schema when
    # it is present.  The base whole-body archive also carries a 43-joint
    # `joint_names`, so make this override unambiguously a hand archive while
    # retaining the auxiliary metadata above for provenance.
    hand_joint_names = np.asarray(
        base_hand.get("hand_joint_names", base_hand.get("joint_names")),
        dtype=str,
    )
    if hand_joint_names.shape != (n_hand,):
        raise ValueError(
            f"base hand archive must provide exactly {n_hand} hand_joint_names; "
            f"got {hand_joint_names.shape}"
        )
    hand50["joint_names"] = hand_joint_names
    hand50["hand_joint_names"] = hand_joint_names
    hand50["joint_contact_optimized"] = np.asarray(True)
    hand50["joint_contact_native_source"] = np.asarray(
        str((output / "finger_candidates_joint_contact_native.npz").resolve())
    )
    hand_output_name = (
        "hand_dex3_50hz_joint_contact.npz"
        if hand_mode == HAND_MODE_DEX3
        else "hand_brainco_50hz_joint_contact.npz"
    )
    np.savez_compressed(output / hand_output_name, **hand50)

    summary = {
        "schema_version": 1,
        "experiment_only": False,
        "pipeline_stage": "formal_contact_aware_precompute",
        "hand_mode": hand_mode,
        "side": side,
        "joint_and_wrist_joint_optimization": True,
        "frame_range_inclusive": [first, last],
        "frames": len(frames),
        "all_finger_contact_frames": int(sum(r["all_finger_contact"] for r in records)),
        "all_finger_contact_rate": float(np.mean([r["all_finger_contact"] for r in records])),
        "robust_all_finger_frames": int(sum(r["robust_all_finger"] for r in records)),
        "robust_all_finger_rate": float(np.mean([r["robust_all_finger"] for r in records])),
        "finger_contact_rate": {
            group: float(np.mean([r[f"{group}_contact"] for r in records]))
            for group in FINGER_GROUPS
        },
        "finger_distance_p95_mm": {
            group: 1000.0 * float(np.percentile([r[f"{group}_surface_distance_m"] for r in records], 95))
            for group in FINGER_GROUPS
        },
        "penetration_p95_mm": 1000.0 * float(np.percentile([r["hand_penetration_p95_m"] for r in records], 95)),
        "penetration_ratio_p95": float(np.percentile([r["hand_penetration_ratio"] for r in records], 95)),
        "joint_delta_norm_rad": finite_stats([r["joint_delta_norm_rad"] for r in records]),
        "wrist_delta_translation_norm_mm": {
            key: 1000.0 * value
            for key, value in finite_stats([r["wrist_delta_translation_norm_m"] for r in records]).items()
        },
        "wrist_delta_rotation_norm_deg": finite_stats(
            [r["wrist_delta_rotation_norm_deg"] for r in records]
        ),
        "outputs": [
            "finger_candidates_joint_contact_native.npz",
            "contact_aware_wrist_candidates_joint_contact_native.npz",
            "hand_dex3_50hz_joint_contact.npz"
            if hand_mode == HAND_MODE_DEX3
            else "hand_brainco_50hz_joint_contact.npz"
            if hand_mode == HAND_MODE_DEX3
            else "hand_brainco_50hz_joint_contact.npz",
            "joint_contact_frame_metrics.jsonl",
            "joint_contact_summary.json",
        ],
    }
    (output / "joint_contact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "joint_contact_frame_metrics.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
