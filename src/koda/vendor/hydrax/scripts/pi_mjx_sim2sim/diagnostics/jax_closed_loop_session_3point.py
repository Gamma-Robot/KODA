#!/usr/bin/env python3
"""Run the isolated torso + dual-wrist JAX CEM-MPC session pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DIAG_DIR = SCRIPT_DIR / "diagnostics"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DIAG_DIR))

from cem_three_point import (  # noqa: E402
    CEMConfig,
    LINK_COST_WEIGHTS,
    TRACKING_LINK_NAMES,
    adjusted_reference_from_knots,
    build_knot_reference_from_motion,
    candidate_feasible,
    cem_sample,
    cem_update_mean_cov,
    choose_candidate,
    make_sigma_vector,
)
from jax_closed_loop_session import (  # noqa: E402
    _save_scaletrack_kinematics,
    apply_initial_state,
)
from jax_closed_loop_three_point import (  # noqa: E402
    make_rollout,
    next_ref_pad_frames,
    rollout_arguments,
)
from mjx_pi_env_three_point import MjxPiEnvThreePoint  # noqa: E402

PRODUCTION_NUM_SAMPLES = 256
PRODUCTION_NUM_ELITES = 10
PRODUCTION_HORIZON_STEPS = 36
PRODUCTION_EXECUTE_STEPS = 12
PRODUCTION_SEED = 2
PRODUCTION_CEM_ITERS = 4


@dataclass
class CompiledRollouts:
    rollout_n: Any
    rollout_1: Any
    compile_timing: dict[str, Any]
    pad_frames: int | None


def default_xml_path() -> Path:
    return (
        REPO_ROOT
        / "scalebfm/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml"
    )


def require_production_cem(
    *,
    seed: int,
    num_samples: int,
    horizon_steps: int,
    execute_steps: int,
    cem_iters: int,
) -> None:
    expected = {
        "seed": PRODUCTION_SEED,
        "num_samples": PRODUCTION_NUM_SAMPLES,
        "horizon_steps": PRODUCTION_HORIZON_STEPS,
        "execute_steps": PRODUCTION_EXECUTE_STEPS,
        "cem_iters": PRODUCTION_CEM_ITERS,
    }
    actual = {
        "seed": int(seed),
        "num_samples": int(num_samples),
        "horizon_steps": int(horizon_steps),
        "execute_steps": int(execute_steps),
        "cem_iters": int(cem_iters),
    }
    mismatches = {
        key: {"expected": expected[key], "got": actual[key]}
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatches:
        raise ValueError(f"non-production CEM config rejected: {mismatches}")


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "median", "p95", "max")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _initial_error_row(env: MjxPiEnvThreePoint) -> dict[str, Any]:
    assert env.state is not None
    robot_pos, robot_quat = env._read_robot_tracking(env.state)
    ref_pos, ref_quat = env.true_tracking_at(0)
    from cem_wrist import quat_error_magnitude

    position_error = np.linalg.norm(robot_pos - ref_pos, axis=-1)
    rotation_error = quat_error_magnitude(ref_quat, robot_quat)
    return _error_row(0, "initial", position_error, rotation_error)


def _error_row(
    frame: int,
    candidate: str,
    position_error: np.ndarray,
    rotation_error: np.ndarray,
) -> dict[str, Any]:
    position_error = np.asarray(position_error, dtype=np.float64)
    rotation_error = np.asarray(rotation_error, dtype=np.float64)
    wrist_failure = bool(
        np.any(position_error[1:] > 0.5) or np.any(rotation_error[1:] > 1.5)
    )
    return {
        "frame": int(frame),
        "time_s": float(frame / 50.0),
        "selected_candidate": candidate,
        "torso_position_error_m": float(position_error[0]),
        "left_wrist_position_error_m": float(position_error[1]),
        "right_wrist_position_error_m": float(position_error[2]),
        "wrist_position_error_mean_m": float(np.mean(position_error[1:])),
        "torso_orientation_error_rad": float(rotation_error[0]),
        "left_wrist_orientation_error_rad": float(rotation_error[1]),
        "right_wrist_orientation_error_rad": float(rotation_error[2]),
        "wrist_orientation_error_mean_rad": float(np.mean(rotation_error[1:])),
        "wrist_failure": int(wrist_failure),
    }


def _compile_rollouts(
    env: MjxPiEnvThreePoint,
    cfg: CEMConfig,
    pad_frames: int | None = None,
) -> CompiledRollouts:
    """Compile the fixed production-N and N=1 batch shapes before measurement."""
    started = time.perf_counter()
    env.warm_compile_batch(cfg.num_samples)
    env.warm_compile_batch(1)
    pd_compile_s = time.perf_counter() - started

    reference, ref_pos, ref_quat = build_knot_reference_from_motion(
        env.ref_body_pos,
        env.ref_body_quat,
        0,
        env.track_motion_indices,
        cfg.knot_starts,
        cfg.num_frames,
    )
    samples = np.repeat(reference[None], cfg.num_samples, axis=0)
    adjusted_pos, adjusted_quat = adjusted_reference_from_knots(
        samples, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
    )
    rollout_n = make_rollout(env, cfg.horizon_steps, cfg.w_rot)
    rollout_1 = make_rollout(env, cfg.horizon_steps, cfg.w_rot)
    snapshot = env.snapshot()
    state_n, history_n = env.broadcast_snapshot(snapshot, cfg.num_samples)
    state_1, history_1 = env.broadcast_snapshot(snapshot, 1)

    compile_started = time.perf_counter()
    jax.block_until_ready(
        rollout_n(
            *rollout_arguments(
                env, state_n, history_n, adjusted_pos, adjusted_quat, 0, pad_frames
            )
        )
    )
    n_compile_s = time.perf_counter() - compile_started
    compile_started = time.perf_counter()
    jax.block_until_ready(
        rollout_1(
            *rollout_arguments(
                env,
                state_1,
                history_1,
                adjusted_pos[:1],
                adjusted_quat[:1],
                0,
                pad_frames,
            )
        )
    )
    one_compile_s = time.perf_counter() - compile_started
    return CompiledRollouts(
        rollout_n=rollout_n,
        rollout_1=rollout_1,
        compile_timing={
            "pd_batch_compile_s": float(pd_compile_s),
            "rollout_n_compile_s": float(n_compile_s),
            "rollout_1_compile_s": float(one_compile_s),
            "total_s": float(pd_compile_s + n_compile_s + one_compile_s),
            "reused_compiled_rollouts": False,
            "pad_frames": pad_frames,
        },
        pad_frames=pad_frames,
    )


def _run(
    env: MjxPiEnvThreePoint,
    cfg: CEMConfig,
    steps: int,
    seed: int,
    rollout_n,
    rollout_1,
    cem_iters: int = PRODUCTION_CEM_ITERS,
    pad_frames: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    assert env.state is not None
    rows = [_initial_error_row(env)]
    qpos0, qvel0 = jax.device_get((env.state.qpos, env.state.qvel))
    state_frames = [
        (
            np.asarray(qpos0, dtype=np.float64).copy(),
            np.asarray(qvel0, dtype=np.float64).copy(),
        )
    ]
    timing = {
        "planning_s": 0.0,
        "rollout_s": 0.0,
        "sampling_and_decode_s": 0.0,
        "cem_update_and_selection_s": 0.0,
        "execute_s": 0.0,
    }
    replan_count = 0
    t = 0
    wall_started = time.perf_counter()
    while t < steps:
        replan_started = time.perf_counter()
        reference, ref_pos, ref_quat = build_knot_reference_from_motion(
            env.ref_body_pos,
            env.ref_body_quat,
            t,
            env.track_motion_indices,
            cfg.knot_starts,
            cfg.num_frames,
        )
        snapshot = env.snapshot()
        if cem_iters < 1:
            raise ValueError("cem_iters must be >= 1")
        rng = np.random.default_rng(np.random.SeedSequence([seed, t]))
        mean = reference
        cov = make_sigma_vector(cfg)
        last_samples = None
        last_costs = None
        elite_mean = mean
        for _ in range(cem_iters):
            sample_started = time.perf_counter()
            samples = cem_sample(mean, cov, cfg, rng)
            feasible = np.asarray(candidate_feasible(samples, cfg), dtype=bool)
            safe_samples = samples.copy()
            safe_samples[~feasible] = reference
            adjusted_pos, adjusted_quat = adjusted_reference_from_knots(
                safe_samples, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
            )
            timing["sampling_and_decode_s"] += time.perf_counter() - sample_started

            state_n, history_n = env.broadcast_snapshot(snapshot, cfg.num_samples)
            rollout_started = time.perf_counter()
            output = rollout_n(
                *rollout_arguments(
                    env,
                    state_n,
                    history_n,
                    adjusted_pos,
                    adjusted_quat,
                    t,
                    pad_frames,
                )
            )
            jax.block_until_ready(output)
            timing["rollout_s"] += time.perf_counter() - rollout_started
            _, trace = output
            costs = np.asarray(trace[0], dtype=np.float64)
            costs[~feasible] = np.inf

            selection_started = time.perf_counter()
            elite_mean, cov = cem_update_mean_cov(samples, costs, cfg)
            timing["cem_update_and_selection_s"] += time.perf_counter() - selection_started
            mean = elite_mean
            last_samples = samples
            last_costs = costs

        if last_samples is None or last_costs is None:
            raise RuntimeError("CEM produced no samples")
        finite_costs = np.isfinite(last_costs)
        best_index = int(np.argmin(np.where(finite_costs, last_costs, np.inf)))
        best_sample = last_samples[best_index]
        feasible = np.asarray(candidate_feasible(last_samples, cfg), dtype=bool)
        reference_cost = float(last_costs[0])
        elite_feasible = bool(candidate_feasible(elite_mean, cfg))
        if elite_feasible:
            elite_pos, elite_quat = adjusted_reference_from_knots(
                elite_mean, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
            )
            state_1, history_1 = env.broadcast_snapshot(snapshot, 1)
            rollout_started = time.perf_counter()
            output_1 = rollout_1(
                *rollout_arguments(
                    env,
                    state_1,
                    history_1,
                    elite_pos[None],
                    elite_quat[None],
                    t,
                    pad_frames,
                )
            )
            jax.block_until_ready(output_1)
            timing["rollout_s"] += time.perf_counter() - rollout_started
            _, trace_1 = output_1
            elite_cost: float | None = float(np.asarray(trace_1[0])[0])
        else:
            elite_cost = None
        selection_started = time.perf_counter()
        selected, candidate, _ = choose_candidate(
            reference,
            best_sample,
            elite_mean,
            reference_cost,
            float(last_costs[best_index]),
            elite_cost,
            feasible_best_sample=bool(feasible[best_index] and finite_costs[best_index]),
            feasible_elite_mean=elite_feasible,
        )
        selected_pos, selected_quat = adjusted_reference_from_knots(
            selected, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
        )
        timing["cem_update_and_selection_s"] += time.perf_counter() - selection_started
        current_plan_s = time.perf_counter() - replan_started
        timing["planning_s"] += current_plan_s

        execute_started = time.perf_counter()
        execute_count = min(cfg.execute_steps, steps - t)
        for local_index in range(execute_count):
            info = env.step_with_window(
                selected_pos[local_index : local_index + cfg.num_frames],
                selected_quat[local_index : local_index + cfg.num_frames],
                advance_time=True,
            )
            position_error = np.asarray(
                [
                    info["err_pos_torso"],
                    info["err_pos_left"],
                    info["err_pos_right"],
                ],
                dtype=np.float64,
            )
            rotation_error = np.asarray(
                [
                    info["err_rot_torso"],
                    info["err_rot_left"],
                    info["err_rot_right"],
                ],
                dtype=np.float64,
            )
            rows.append(_error_row(t + local_index + 1, candidate, position_error, rotation_error))
            assert env.state is not None
            qpos, qvel = jax.device_get((env.state.qpos, env.state.qvel))
            state_frames.append(
                (
                    np.asarray(qpos, dtype=np.float64).copy(),
                    np.asarray(qvel, dtype=np.float64).copy(),
                )
            )
        timing["execute_s"] += time.perf_counter() - execute_started
        t += execute_count
        replan_count += 1
        if replan_count == 1 or replan_count % 10 == 0 or t == steps:
            print(
                f"[three-point] t={t}/{steps} replans={replan_count} "
                f"candidate={candidate} plan_ms={1e3 * current_plan_s:.2f}",
                flush=True,
            )
    wall_s = time.perf_counter() - wall_started
    timing["wall_s"] = float(wall_s)
    timing["replans"] = int(replan_count)
    timing["replan_mean_ms"] = float(1e3 * timing["planning_s"] / replan_count)
    timing["rollout_mean_ms_per_replan"] = float(1e3 * timing["rollout_s"] / replan_count)
    timing["execute_mean_ms_per_step"] = float(1e3 * timing["execute_s"] / steps)
    timing["non_rollout_planning_s"] = float(timing["planning_s"] - timing["rollout_s"])
    timing["non_rollout_planning_mean_ms_per_replan"] = float(
        1e3 * timing["non_rollout_planning_s"] / replan_count
    )
    timing["schema_version"] = 2

    transition_rows = rows[1:]
    wrist_position = np.asarray(
        [row["wrist_position_error_mean_m"] for row in transition_rows], dtype=np.float64
    )
    wrist_orientation = np.asarray(
        [row["wrist_orientation_error_mean_rad"] for row in transition_rows], dtype=np.float64
    )
    torso_position = np.asarray(
        [row["torso_position_error_m"] for row in transition_rows], dtype=np.float64
    )
    torso_orientation = np.asarray(
        [row["torso_orientation_error_rad"] for row in transition_rows], dtype=np.float64
    )
    candidates = Counter(row["selected_candidate"] for row in transition_rows)
    metrics = {
        "steps": int(steps),
        "wrist_position_error_m": _stats(wrist_position),
        "wrist_orientation_error_rad": _stats(wrist_orientation),
        "torso_position_error_m": _stats(torso_position),
        "torso_orientation_error_rad": _stats(torso_orientation),
        "failure_rate": float(
            np.mean([row["wrist_failure"] for row in transition_rows])
        ),
        "candidate_counts": {
            name: int(candidates.get(name, 0))
            for name in ("reference", "best_sample", "elite_mean")
        },
        "timing": timing,
    }
    return metrics, rows, state_frames


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _augment_kinematics_meta(
    meta_path: str | Path,
    *,
    reference: Path,
    init_state: Path,
    manifest: Path,
) -> None:
    path = Path(meta_path)
    metadata = json.loads(path.read_text())
    metadata["tracking_pipeline"] = {
        "name": "torso_umi3_model24600_seed02",
        "tracked_links": list(TRACKING_LINK_NAMES),
        "link_cost_weights": LINK_COST_WEIGHTS.tolist(),
        "reference_npz": str(reference),
        "init_state_npz": str(init_state),
        "artifact_manifest": str(manifest),
        "simulator_model": "g1_29dof_mjx.xml (29-DoF body only)",
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)


def reused_compile_timing(pad_frames: int | None) -> dict[str, Any]:
    return {
        "pd_batch_compile_s": 0.0,
        "rollout_n_compile_s": 0.0,
        "rollout_1_compile_s": 0.0,
        "total_s": 0.0,
        "reused_compiled_rollouts": True,
        "pad_frames": pad_frames,
    }


def run_three_point_session(
    *,
    motion_file: Path,
    init_state_npz: Path,
    artifact_manifest: Path,
    output_dir: Path,
    seed: int = PRODUCTION_SEED,
    num_samples: int = PRODUCTION_NUM_SAMPLES,
    horizon_steps: int = PRODUCTION_HORIZON_STEPS,
    execute_steps: int = PRODUCTION_EXECUTE_STEPS,
    cem_iters: int = PRODUCTION_CEM_ITERS,
    steps: int | None = None,
    xml_path: Path | None = None,
    torch_device: str = "cuda",
    env: MjxPiEnvThreePoint | None = None,
    compiled: CompiledRollouts | None = None,
    pad_frames: int | None = None,
    require_production: bool = False,
) -> tuple[dict[str, Any], MjxPiEnvThreePoint, CompiledRollouts]:
    """Run one isolated three-point JAX CEM-MPC session and write artifacts."""
    if require_production:
        require_production_cem(
            seed=seed,
            num_samples=num_samples,
            horizon_steps=horizon_steps,
            execute_steps=execute_steps,
            cem_iters=cem_iters,
        )
    elif seed != PRODUCTION_SEED:
        raise ValueError("the accepted three-point configuration fixes seed=2")

    reference = Path(motion_file).expanduser().resolve()
    init_state = Path(init_state_npz).expanduser().resolve()
    manifest = Path(artifact_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = manifest.parent
    xml_path = Path(xml_path).expanduser().resolve() if xml_path is not None else default_xml_path()
    cfg = CEMConfig(
        num_samples=int(num_samples),
        num_elites=PRODUCTION_NUM_ELITES,
        horizon_steps=int(horizon_steps),
        execute_steps=int(execute_steps),
    )

    startup_started = time.perf_counter()
    if env is None:
        env = MjxPiEnvThreePoint(
            artifacts,
            str(xml_path),
            str(reference),
            artifact_manifest=manifest,
            mode_index=0,
            torch_device=torch_device,
        )
    else:
        env.load_motion(str(reference))
    init_info = apply_initial_state(env, init_state)
    init_robot_pos, init_robot_quat = env._read_robot_tracking(env.state)
    init_ref_pos, init_ref_quat = env.true_tracking_at(0)
    from cem_wrist import quat_error_magnitude

    init_info["initial_tracking_links"] = list(TRACKING_LINK_NAMES)
    init_info["initial_three_point_position_error_m"] = np.linalg.norm(
        init_robot_pos - init_ref_pos, axis=-1
    ).tolist()
    init_info["initial_three_point_orientation_error_rad"] = quat_error_magnitude(
        init_ref_quat, init_robot_quat
    ).tolist()
    startup_s = time.perf_counter() - startup_started

    resolved_pad = pad_frames
    if pad_frames is not None:
        resolved_pad = next_ref_pad_frames(env.T, pad_frames)
        if compiled is not None and compiled.pad_frames is not None:
            resolved_pad = next_ref_pad_frames(env.T, compiled.pad_frames)
    reuse_compiled = (
        compiled is not None
        and compiled.pad_frames == resolved_pad
        and compiled.rollout_n is not None
        and compiled.rollout_1 is not None
    )
    if reuse_compiled:
        assert compiled is not None
        compile_timing = reused_compile_timing(resolved_pad)
    else:
        compiled = _compile_rollouts(env, cfg, pad_frames=resolved_pad)
        compile_timing = compiled.compile_timing

    steps = env.T - 1 if steps is None else min(int(steps), env.T - 1)
    if steps <= 0:
        raise ValueError(f"motion has no usable transitions: T={env.T}, steps={steps}")
    print(
        f"[three-point] frames={env.T} steps={steps} N={cfg.num_samples} "
        f"H={cfg.horizon_steps} elite={cfg.num_elites} execute={cfg.execute_steps} "
        f"cem_iters={cem_iters} pad_frames={resolved_pad} "
        f"reused_compiled={bool(reuse_compiled)}",
        flush=True,
    )
    metrics, rows, state_frames = _run(
        env,
        cfg,
        steps,
        seed,
        compiled.rollout_n,
        compiled.rollout_1,
        cem_iters,
        pad_frames=resolved_pad,
    )
    if len(rows) != steps + 1 or len(state_frames) != steps + 1:
        raise RuntimeError(
            f"expected {steps + 1} frames, got rows={len(rows)} states={len(state_frames)}"
        )

    run_name = (
        f"jax_mpc_3point_n{cfg.num_samples}_h{cfg.horizon_steps}_i{cem_iters}_"
        f"x{cfg.execute_steps}_seed{seed}"
    )
    error_csv = output_dir / f"{run_name}_errors.csv"
    _write_csv(error_csv, rows)
    kinematics_path = output_dir / "kinematics" / f"{run_name}_kinematics.npz"
    kinematics = _save_scaletrack_kinematics(
        env.mj_model,
        state_frames,
        env.body_names,
        env.policy_joints,
        kinematics_path,
    )
    _augment_kinematics_meta(
        kinematics["meta_path"],
        reference=reference,
        init_state=init_state,
        manifest=manifest,
    )
    result = {
        "status": "completed",
        "setting": {
            "motion_file": str(reference),
            "init_state_npz": str(init_state),
            "artifact_manifest": str(manifest),
            "checkpoint": Path(json.loads(manifest.read_text())["checkpoint"]["path"]).name,
            "checkpoint_identity": json.loads(manifest.read_text())["checkpoint"],
            "reference_frames": int(env.T),
            "recorded_frames": int(len(state_frames)),
            "control_steps": int(steps),
            "N": cfg.num_samples,
            "H": cfg.horizon_steps,
            "cem_iters": cem_iters,
            "elite": cfg.num_elites,
            "execute_steps": cfg.execute_steps,
            "mode_index": env.mode_index_val,
            "seed": seed,
            "tracking_links": list(TRACKING_LINK_NAMES),
            "selected_body_slots": env.track_selected_slots.tolist(),
            "motion_body_indices": env.track_motion_indices.tolist(),
            "mjx_body_ids": env.track_mjx_body_ids.tolist(),
            "decision_dimension": cfg.nu,
            "link_cost_weights": LINK_COST_WEIGHTS.tolist(),
            "w_rot": cfg.w_rot,
        },
        "initial_state": init_info,
        "jax_closed_loop": metrics,
        "startup_s": float(startup_s),
        "compile_timing": compile_timing,
        "error_csv": str(error_csv),
        "kinematics": kinematics,
        "note": (
            "isolated three-point path; legacy wrist runner, artifact, reference, "
            "initial state, and output directories are not modified"
        ),
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(summary_path)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[three-point] wrote {summary_path}", flush=True)
    return result, env, compiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--init-state-npz", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Smoke-test cap; default is all 538 transitions.")
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--num-samples", type=int, default=PRODUCTION_NUM_SAMPLES)
    parser.add_argument("--horizon-steps", type=int, default=PRODUCTION_HORIZON_STEPS)
    parser.add_argument("--execute-steps", type=int, default=PRODUCTION_EXECUTE_STEPS)
    parser.add_argument("--cem-iters", type=int, default=PRODUCTION_CEM_ITERS)
    args = parser.parse_args()
    run_three_point_session(
        motion_file=args.motion_file,
        init_state_npz=args.init_state_npz,
        artifact_manifest=args.artifact_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        horizon_steps=args.horizon_steps,
        execute_steps=args.execute_steps,
        cem_iters=args.cem_iters,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
