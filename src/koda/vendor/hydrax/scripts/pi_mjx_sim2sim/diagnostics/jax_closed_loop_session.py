#!/usr/bin/env python3
"""Run the isolated JAX CEM-MPC path on a ScaleTrack session motion.

This diagnostic is intentionally separate from the production runner.  The
reference motion is loaded from the session NPZ and is never modified.  An
external ``init_state.npz`` is applied only to the MuJoCo/MJX simulation state
and to the policy history, matching ``play_eval_session.py`` semantics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402
from hydrax.utils.video import VideoRecorder  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DIAG_DIR = SCRIPT_DIR / "diagnostics"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DIAG_DIR))

from cem_wrist import (  # noqa: E402
    CEMConfig,
    adjusted_reference_from_knots,
    build_knot_reference_from_motion,
)
from compare_jax_push1_100 import _run_jax, _summary  # noqa: E402
from jax_closed_loop_push1 import _make_rollout, _to_jax_history  # noqa: E402
from mjx_pi_env import MjxPiEnv, PiHistory  # noqa: E402


def _strings(value: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(value).reshape(-1).tolist()]


def _normalise_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(q)):
        raise ValueError("initial root quaternion contains non-finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-8:
        raise ValueError("initial root quaternion has near-zero norm")
    return q / norm


def _save_scaletrack_kinematics(
    model: mujoco.MjModel,
    state_frames: list[tuple[np.ndarray, np.ndarray]],
    body_names: list[str],
    joint_names: list[str],
    output_path: str | Path,
    *,
    fps: int = 50,
) -> dict:
    """Save executed MuJoCo states in the ScaleTrack training NPZ schema."""

    if not state_frames:
        raise ValueError("cannot save an empty kinematics trajectory")
    if fps != 50:
        raise ValueError(f"ScaleTrack motion data must be 50 Hz, got {fps}")
    if not body_names or body_names[0] != "pelvis":
        raise ValueError("ScaleTrack body order must start with the pelvis/root")
    if len(set(body_names)) != len(body_names):
        raise ValueError("ScaleTrack body names contain duplicates")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("ScaleTrack joint names contain duplicates")

    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names],
        dtype=np.int32,
    )
    missing_bodies = [name for name, body_id in zip(body_names, body_ids) if body_id < 0]
    if missing_bodies:
        raise ValueError(f"MuJoCo bodies not found: {missing_bodies}")

    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    missing_joints = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
    if missing_joints:
        raise ValueError(f"MuJoCo joints not found: {missing_joints}")
    invalid_joints = [
        name
        for name, joint_id in zip(joint_names, joint_ids)
        if int(model.jnt_type[joint_id])
        not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    ]
    if invalid_joints:
        raise ValueError(f"ScaleTrack joints must be 1-DoF: {invalid_joints}")

    qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
    qvel_addresses = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
    frame_count = len(state_frames)
    body_count = len(body_names)
    joint_count = len(joint_names)
    joint_pos = np.empty((frame_count, joint_count), dtype=np.float32)
    joint_vel = np.empty((frame_count, joint_count), dtype=np.float32)
    body_pos_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
    body_quat_w = np.empty((frame_count, body_count, 4), dtype=np.float32)
    body_lin_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
    body_ang_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)

    data = mujoco.MjData(model)
    object_velocity = np.empty(6, dtype=np.float64)
    for frame_index, (qpos, qvel) in enumerate(state_frames):
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        if qpos.shape != (model.nq,) or qvel.shape != (model.nv,):
            raise ValueError(
                f"state frame {frame_index} has qpos/qvel shapes "
                f"{qpos.shape}/{qvel.shape}, expected {(model.nq,)}/{(model.nv,)}"
            )
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
            raise ValueError(f"state frame {frame_index} contains non-finite values")

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        joint_pos[frame_index] = qpos[qpos_addresses]
        joint_vel[frame_index] = qvel[qvel_addresses]
        body_pos_w[frame_index] = data.xpos[body_ids]
        body_quat_w[frame_index] = data.xquat[body_ids]
        for body_index, body_id in enumerate(body_ids):
            # MuJoCo returns angular then linear velocity. With local=0 both
            # are expressed in world axes at the selected body-frame origin.
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                object_velocity,
                0,
            )
            body_ang_vel_w[frame_index, body_index] = object_velocity[:3]
            body_lin_vel_w[frame_index, body_index] = object_velocity[3:]

    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "fps": np.asarray(fps, dtype=np.int64),
    }
    for name, values in arrays.items():
        if name != "fps" and not np.all(np.isfinite(values)):
            raise RuntimeError(f"generated kinematics field {name} contains non-finite values")
    quat_norm_error = float(np.max(np.abs(np.linalg.norm(body_quat_w, axis=-1) - 1.0)))
    if quat_norm_error > 1.0e-4:
        raise RuntimeError(f"generated body quaternion norm error is {quat_norm_error}")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    metadata = {
        "format": "ScaleTrack motion NPZ",
        "format_version": 1,
        "source": "executed MuJoCo/MJX robot states",
        "sampling": {
            "fps": fps,
            "period_s": 1.0 / fps,
            "frame_0": "initial simulator state",
            "frame_t": "simulator state after control transition t",
        },
        "coordinate_conventions": {
            "body_frame": "MuJoCo body-frame origin",
            "world_suffix": "fields ending in _w are expressed in world axes",
            "quaternion_order": "wxyz",
        },
        "dimensions": {
            "T": "number of recorded frames in the NPZ",
            "J": joint_count,
            "B": body_count,
        },
        "joint_names": joint_names,
        "body_names": body_names,
        "joint_index_to_name": {str(i): name for i, name in enumerate(joint_names)},
        "body_index_to_name": {str(i): name for i, name in enumerate(body_names)},
        "root": {
            "body_index": 0,
            "body_name": body_names[0],
            "position": "body_pos_w[:, 0, :]",
            "orientation": "body_quat_w[:, 0, :]",
            "linear_velocity": "body_lin_vel_w[:, 0, :]",
            "angular_velocity": "body_ang_vel_w[:, 0, :]",
        },
        "fields": {
            "joint_pos": {
                "shape": ["T", "J"],
                "dtype": "float32",
                "unit": "rad",
                "axes": ["time", "joint_names"],
            },
            "joint_vel": {
                "shape": ["T", "J"],
                "dtype": "float32",
                "unit": "rad/s",
                "axes": ["time", "joint_names"],
            },
            "body_pos_w": {
                "shape": ["T", "B", 3],
                "dtype": "float32",
                "unit": "m",
                "axes": ["time", "body_names", ["x", "y", "z"]],
            },
            "body_quat_w": {
                "shape": ["T", "B", 4],
                "dtype": "float32",
                "unit": "unit quaternion",
                "axes": ["time", "body_names", ["w", "x", "y", "z"]],
            },
            "body_lin_vel_w": {
                "shape": ["T", "B", 3],
                "dtype": "float32",
                "unit": "m/s",
                "axes": ["time", "body_names", ["x", "y", "z"]],
            },
            "body_ang_vel_w": {
                "shape": ["T", "B", 3],
                "dtype": "float32",
                "unit": "rad/s",
                "axes": ["time", "body_names", ["x", "y", "z"]],
            },
            "fps": {
                "shape": [],
                "dtype": "int64",
                "unit": "Hz",
            },
        },
        "not_recorded": [
            "policy action",
            "policy history",
            "tracking target/reference",
            "actuator torque",
        ],
    }
    metadata_path = output_path.parent / "meta.json"
    temporary_metadata_path = output_path.parent / f".{output_path.stem}.meta.json.tmp"
    try:
        temporary_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_metadata_path.replace(metadata_path)
    finally:
        if temporary_metadata_path.exists():
            temporary_metadata_path.unlink()

    return {
        "path": str(output_path),
        "meta_path": str(metadata_path),
        "frames": frame_count,
        "fps": fps,
        "duration_s": float((frame_count - 1) / fps),
        "body_count": body_count,
        "joint_count": joint_count,
        "root_body_index": 0,
        "root_body_name": body_names[0],
        "quaternion_order": "wxyz",
        "max_quaternion_norm_error": quat_norm_error,
    }


def _render_qpos_video(
    model: mujoco.MjModel,
    qpos_frames: list[np.ndarray],
    output_path: str | Path,
    *,
    target_wrist_frames: np.ndarray | None = None,
    fps: float = 50.0,
    width: int = 720,
    height: int = 480,
) -> dict:
    """Render captured MuJoCo qpos frames to an MP4 without opening a window."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not qpos_frames:
        raise ValueError("cannot render an empty qpos trajectory")
    if target_wrist_frames is not None:
        target_wrist_frames = np.asarray(target_wrist_frames, dtype=np.float64)
        if target_wrist_frames.shape != (len(qpos_frames), 2, 3):
            raise ValueError(
                "target_wrist_frames must have shape "
                f"({len(qpos_frames)}, 2, 3), got {target_wrist_frames.shape}"
            )

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.5
    camera.elevation = -15.0
    camera.azimuth = 140.0
    first_q = np.asarray(qpos_frames[0], dtype=np.float64)
    camera.lookat = np.asarray([first_q[0], first_q[1], 0.8], dtype=np.float64)
    data = mujoco.MjData(model)
    recorder = VideoRecorder(
        output_dir=str(output_path.parent),
        width=width,
        height=height,
        fps=fps,
    )
    if not recorder.start():
        renderer.close()
        raise RuntimeError("ffmpeg video recorder could not be started")
    try:
        marker_colors = (
            np.asarray([0.95, 0.15, 0.15, 1.0], dtype=np.float32),
            np.asarray([0.15, 0.35, 0.95, 1.0], dtype=np.float32),
        )
        for frame_index, q in enumerate(qpos_frames):
            data.qpos[:] = np.asarray(q, dtype=np.float64)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            if target_wrist_frames is not None:
                for wrist_index, target in enumerate(target_wrist_frames[frame_index]):
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.asarray([0.035, 0.0, 0.0], dtype=np.float64),
                        pos=target,
                        mat=np.eye(3).flatten(),
                        rgba=marker_colors[wrist_index],
                    )
                    renderer.scene.ngeom += 1
            if not recorder.add_frame(renderer.render().tobytes()):
                raise RuntimeError("failed while writing a video frame")
        if not recorder.stop():
            raise RuntimeError("ffmpeg failed to finalize the video")
    finally:
        renderer.close()

    if recorder.video_path is None:
        raise RuntimeError("video recorder did not report an output path")
    generated = Path(recorder.video_path)
    if generated.resolve() != output_path:
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(generated), str(output_path))
    return {
        "path": str(output_path),
        "frames": len(qpos_frames),
        "fps": float(fps),
        "width": width,
        "height": height,
        "duration_s": float((len(qpos_frames) - 1) / fps),
    }


def apply_initial_state(env: MjxPiEnv, path: str | Path) -> dict:
    """Apply the session ``init_state.npz`` to the MJX environment.

    Joint values are mapped by name through MuJoCo joint addresses rather than
    relying on NPZ ordering.  The policy history is seeded with the applied
    state (and zero previous actions), then the reference clock starts at 0.
    """

    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as data:
        required = ("root_pos_w", "root_quat_wxyz", "joint_pos", "joint_names")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"init state missing keys {missing}: {path}")

        root_pos = np.asarray(data["root_pos_w"], dtype=np.float64).reshape(3)
        root_quat = _normalise_quat(data["root_quat_wxyz"])
        root_lin_vel = np.asarray(
            data["root_lin_vel_w"] if "root_lin_vel_w" in data.files else np.zeros(3),
            dtype=np.float64,
        ).reshape(3)
        root_ang_vel = np.asarray(
            data["root_ang_vel_w"] if "root_ang_vel_w" in data.files else np.zeros(3),
            dtype=np.float64,
        ).reshape(3)
        joint_names = _strings(data["joint_names"])
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64).reshape(-1)
        joint_vel = np.asarray(
            data["joint_vel"] if "joint_vel" in data.files else np.zeros_like(joint_pos),
            dtype=np.float64,
        ).reshape(-1)

    if len(joint_names) != joint_pos.size or joint_vel.size != joint_pos.size:
        raise ValueError(
            "joint_names, joint_pos and joint_vel have inconsistent lengths: "
            f"{len(joint_names)}, {joint_pos.size}, {joint_vel.size}"
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("init state contains duplicate joint names")
    if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_lin_vel)):
        raise ValueError("initial root position/velocity contains non-finite values")
    if not np.all(np.isfinite(root_ang_vel)) or not np.all(np.isfinite(joint_pos)):
        raise ValueError("initial angular/joint position contains non-finite values")
    if not np.all(np.isfinite(joint_vel)):
        raise ValueError("initial joint velocity contains non-finite values")

    model = env.mj_model
    qpos = np.zeros(model.nq, dtype=np.float64)
    qvel = np.zeros(model.nv, dtype=np.float64)
    qpos[:3] = root_pos
    qpos[3:7] = root_quat
    qvel[:3] = root_lin_vel
    qvel[3:6] = root_ang_vel

    values = dict(zip(joint_names, joint_pos))
    velocities = dict(zip(joint_names, joint_vel))
    expected = set(env.sim_joints)
    supplied = set(joint_names)
    missing_sim = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing_sim or extra:
        raise ValueError(
            "initial state joints do not match MJX actuators; "
            f"missing={missing_sim}, extra={extra}"
        )

    for name in env.sim_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        if int(model.jnt_type[jid]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(f"expected 1-DoF joint, got {name}")
        qpos[int(model.jnt_qposadr[jid])] = values[name]
        qvel[int(model.jnt_dofadr[jid])] = velocities[name]

    env.mj_data.qpos[:] = qpos
    env.mj_data.qvel[:] = qvel
    mujoco.mj_forward(model, env.mj_data)
    env.state = mjx.put_data(model, env.mj_data)

    # The policy uses its metadata joint order, while MuJoCo stores actuator
    # joints in XML order.  Seed all three history frames with the applied
    # state, exactly as the normal reset path does for frame 0.
    sim_q = np.asarray(
        [qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])]
         for n in env.sim_joints],
        dtype=np.float32,
    )
    sim_qd = np.asarray(
        [qvel[int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])]
         for n in env.sim_joints],
        dtype=np.float32,
    )
    dof_pol = sim_q[env.sim_idx_for_policy]
    dof_vel_pol = sim_qd[env.sim_idx_for_policy]
    context = int(env.context_len)
    env.hist = PiHistory(
        root_quat=np.broadcast_to(root_quat.astype(np.float32), (1, context, 4)).copy(),
        base_ang=np.broadcast_to(root_ang_vel.astype(np.float32), (1, context, 3)).copy(),
        dof_pos=np.broadcast_to(dof_pol, (1, context, 29)).copy(),
        dof_vel=np.broadcast_to(dof_vel_pol, (1, context, 29)).copy(),
        action=np.zeros((1, context, 29), dtype=np.float32),
        head=context - 1,
    )
    env.time_step = 0

    # A small preflight check makes coordinate-frame mistakes obvious before
    # an expensive CEM run.
    init_pos, init_quat = env._read_robot_wrists(env.state)
    ref_pos, ref_quat = env.true_wrist_at(0)
    pos_err = np.linalg.norm(init_pos - ref_pos, axis=-1)
    return {
        "path": str(path),
        "joint_count": len(joint_names),
        "root_pos_w": root_pos.tolist(),
        "root_quat_wxyz": root_quat.tolist(),
        "initial_wrist_error_m": pos_err.tolist(),
        "initial_wrist_error_mean_m": float(np.mean(pos_err)),
        "reference_time_step": 0,
    }


def _compile_rollouts(env: MjxPiEnv, cfg: CEMConfig):
    """Compile both fixed batch shapes before timing the session run."""

    u_ref, ref_pos, ref_quat = build_knot_reference_from_motion(
        env.ref_body_pos,
        env.ref_body_quat,
        0,
        env.lw_i,
        env.rw_i,
        cfg.knot_starts,
        cfg.num_frames,
    )
    warm_samples = np.repeat(u_ref[None], cfg.num_samples, axis=0)
    warm_pos, warm_quat = adjusted_reference_from_knots(
        warm_samples, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
    )
    rollout_n = _make_rollout(env, cfg.horizon_steps, cfg.w_rot)
    rollout_1 = _make_rollout(env, cfg.horizon_steps, cfg.w_rot)
    snap = env.snapshot()
    state_n, hist_n = env.broadcast_snapshot(snap, cfg.num_samples)
    state_1, hist_1 = env.broadcast_snapshot(snap, 1)
    args_n = (
        state_n,
        _to_jax_history(hist_n),
        jnp.asarray(hist_n.head, dtype=jnp.int32),
        jnp.asarray(warm_pos),
        jnp.asarray(warm_quat),
        jnp.asarray(0, dtype=jnp.int32),
    )
    args_1 = (
        state_1,
        _to_jax_history(hist_1),
        jnp.asarray(hist_1.head, dtype=jnp.int32),
        jnp.asarray(warm_pos[:1]),
        jnp.asarray(warm_quat[:1]),
        jnp.asarray(0, dtype=jnp.int32),
    )
    jax.block_until_ready(rollout_n(*args_n))
    jax.block_until_ready(rollout_1(*args_1))
    return rollout_n, rollout_1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--init-state-npz", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="Optional headless MP4 path; captures the initial state and executed MJX states.",
    )
    parser.add_argument(
        "--kinematics-out-dir",
        type=Path,
        default=None,
        help="Optional directory for executed 50 Hz ScaleTrack-format kinematics.",
    )
    parser.add_argument(
        "--kinematics-name",
        default=None,
        help="Optional NPZ filename inside --kinematics-out-dir.",
    )
    args = parser.parse_args()
    if args.kinematics_name is not None and args.kinematics_out_dir is None:
        parser.error("--kinematics-name requires --kinematics-out-dir")

    cfg = CEMConfig(num_samples=1024, num_elites=10, horizon_steps=36, execute_steps=3)
    xml = REPO_ROOT / "scalebfm/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml"
    artifacts = SCRIPT_DIR / "artifacts"
    env = MjxPiEnv(
        str(artifacts),
        str(xml),
        str(Path(args.motion_file).expanduser()),
        mode_index=1,
        torch_device="cuda",
    )
    env.warm_compile_batch(cfg.num_samples)
    env.warm_compile_batch(1)
    init_info = apply_initial_state(env, args.init_state_npz)
    steps = int(env.T - 1 if args.steps is None else min(args.steps, env.T - 1))
    if steps <= 0:
        raise ValueError(f"motion has no usable control steps: T={env.T}")
    state_frames: list[tuple[np.ndarray, np.ndarray]] | None = None
    if args.video_out is not None or args.kinematics_out_dir is not None:
        state_frames = [
            (
                np.asarray(env.mj_data.qpos, dtype=np.float64).copy(),
                np.asarray(env.mj_data.qvel, dtype=np.float64).copy(),
            )
        ]
        original_step = env.step_with_window

        def _step_and_capture(*step_args, **step_kwargs):
            info = original_step(*step_args, **step_kwargs)
            assert env.state is not None
            qpos, qvel = jax.device_get((env.state.qpos, env.state.qvel))
            state_frames.append(
                (
                    np.asarray(qpos, dtype=np.float64).copy(),
                    np.asarray(qvel, dtype=np.float64).copy(),
                )
            )
            return info

        # Capture only this isolated diagnostic run; production env code is
        # untouched and the wrapper does not alter the returned control data.
        env.step_with_window = _step_and_capture  # type: ignore[method-assign]
    print(
        f"[session] reference_frames={env.T} control_steps={steps} "
        f"N={cfg.num_samples} H={cfg.horizon_steps} execute={cfg.execute_steps}",
        flush=True,
    )
    rollout_n, rollout_1 = _compile_rollouts(env, cfg)
    run = _run_jax(env, cfg, steps, args.seed, rollout_n, rollout_1)
    if state_frames is not None and len(state_frames) != steps + 1:
        raise RuntimeError(
            f"captured {len(state_frames)} state frames for {steps} control steps"
        )
    kinematics_info = None
    if args.kinematics_out_dir is not None:
        filename = args.kinematics_name
        if filename is None:
            filename = (
                f"jax_mpc_n{cfg.num_samples}_h{cfg.horizon_steps}_i1_"
                f"x{cfg.execute_steps}_seed{args.seed}_kinematics.npz"
            )
        filename_path = Path(filename)
        if filename_path.is_absolute() or filename_path.name != filename:
            raise ValueError("--kinematics-name must be a filename, not a path")
        if filename_path.suffix.lower() != ".npz":
            filename_path = filename_path.with_suffix(filename_path.suffix + ".npz")
        assert state_frames is not None
        kinematics_info = _save_scaletrack_kinematics(
            env.mj_model,
            state_frames,
            env.body_names,
            env.policy_joints,
            args.kinematics_out_dir / filename_path,
        )
        print(f"[session] wrote {kinematics_info['path']}", flush=True)
    video_info = None
    if args.video_out is not None:
        assert state_frames is not None
        qpos_frames = [qpos for qpos, _ in state_frames]
        target_wrist = np.stack(
            [
                env.ref_body_pos[: steps + 1, env.lw_i],
                env.ref_body_pos[: steps + 1, env.rw_i],
            ],
            axis=1,
        )
        video_info = _render_qpos_video(
            env.mj_model,
            qpos_frames,
            args.video_out,
            target_wrist_frames=target_wrist,
        )
    result = {
        "setting": {
            "motion_file": str(Path(args.motion_file).expanduser().resolve()),
            "init_state_npz": str(Path(args.init_state_npz).expanduser().resolve()),
            "reference_frames": int(env.T),
            "control_steps": steps,
            "N": cfg.num_samples,
            "H": cfg.horizon_steps,
            "cem_iters": 1,
            "elite": cfg.num_elites,
            "execute_steps": cfg.execute_steps,
            "mode_index": env.mode_index_val,
            "seed": args.seed,
        },
        "initial_state": init_info,
        "jax_closed_loop": _summary(run),
        "kinematics": kinematics_info,
        "video": video_info,
        "note": (
            "isolated diagnostic; reference motion remains unchanged; "
            "JAX is used for batched MPC rollout while selected commands are "
            "executed through the existing single-env policy/MJX step"
        ),
    }
    payload = json.dumps(result, indent=2)
    print(payload, flush=True)
    if args.out is not None:
        out = args.out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
        print(f"[session] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
