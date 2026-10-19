#!/usr/bin/env python3
"""Enforce eager/TorchScript/JAX parity for the Torso-UMI-3 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax

jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from diagnostics.jax_closed_loop_push1 import _make_weights, _policy_forward  # noqa: E402
from load_eager_policy import load_eager_policy, resolve_policy_artifacts  # noqa: E402
from policy_inputs import build_policy_targets_single  # noqa: E402


TARGET_MAX_THRESHOLD = 1.0e-5
ACTION_MAX_THRESHOLD = 1.0e-3
ACTION_MEAN_THRESHOLD = 1.0e-4


def quat_apply_inverse(q_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    q_inv = np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)
    t = 2.0 * np.cross(q_inv[..., 1:], v)
    return v + q_inv[..., :1] * t + np.cross(q_inv[..., 1:], t)


def frozen_motion_inputs(
    motion_path: Path,
    meta: dict[str, Any],
    batch: int,
    time_step: int,
) -> tuple[np.ndarray, ...]:
    with np.load(motion_path, allow_pickle=False) as motion:
        required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
        missing = [key for key in required if key not in motion]
        if missing:
            raise KeyError(f"Reference is missing required arrays: {missing}")
        joint_pos = np.asarray(motion["joint_pos"])
        joint_vel = np.asarray(motion["joint_vel"])
        body_pos = np.asarray(motion["body_pos_w"])
        body_quat = np.asarray(motion["body_quat_w"])
        body_ang = np.asarray(motion["body_ang_vel_w"]) if "body_ang_vel_w" in motion else None

    body_names = list(meta["body_names"])
    selected_names = list(meta["selected_body_names"])
    if body_pos.shape[1] != len(body_names) or body_quat.shape[1] != len(body_names):
        raise ValueError(
            "Reference body dimension does not match artifact metadata: "
            f"{body_pos.shape[1]} vs {len(body_names)}"
        )
    if joint_pos.shape[1] != len(meta["joint_names"]):
        raise ValueError(
            "Reference joint dimension does not match artifact metadata: "
            f"{joint_pos.shape[1]} vs {len(meta['joint_names'])}"
        )

    pelvis = body_names.index("pelvis")
    selected_idx = np.asarray([body_names.index(name) for name in selected_names], dtype=np.int64)
    starts = np.clip(time_step + np.arange(batch), 0, joint_pos.shape[0] - 1)
    context = int(meta["history_buffer_size"])

    root_now = body_quat[starts, pelvis].astype(np.float32)
    root_hist = np.repeat(root_now[:, None, :], context, axis=1)
    dof_hist = np.repeat(joint_pos[starts, None, :], context, axis=1).astype(np.float32)
    vel_hist = np.repeat(joint_vel[starts, None, :], context, axis=1).astype(np.float32)
    action_hist = np.zeros_like(dof_hist, dtype=np.float32)
    if body_ang is None:
        base_ang_now = np.zeros((batch, 3), dtype=np.float32)
    else:
        base_ang_now = quat_apply_inverse(root_now, body_ang[starts, pelvis]).astype(np.float32)
    base_ang_hist = np.repeat(base_ang_now[:, None, :], context, axis=1)

    targets_pos = []
    targets_rot = []
    for row, start in enumerate(starts):
        pos, rot = build_policy_targets_single(
            body_pos,
            body_quat,
            int(start),
            meta["future_idx"],
            selected_idx,
            body_pos[start, pelvis],
            root_now[row],
            selected_names.index("left_wrist_yaw_link"),
            selected_names.index("right_wrist_yaw_link"),
        )
        targets_pos.append(pos)
        targets_rot.append(rot)

    return tuple(
        np.ascontiguousarray(value, dtype=np.float32)
        for value in (
            root_hist,
            base_ang_hist,
            dof_hist,
            vel_hist,
            action_hist,
            np.stack(targets_pos),
            np.stack(targets_rot),
        )
    )


def torch_call(
    policy: torch.nn.Module,
    inputs: tuple[np.ndarray, ...],
    mode: np.ndarray,
    offsets: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    args = tuple(torch.from_numpy(value).to(device) for value in inputs)
    mode_tensor = torch.from_numpy(mode).to(device)
    offset_tensor = torch.from_numpy(offsets).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        target, action = policy(*args, mode_tensor, offset_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = 1.0e3 * (time.perf_counter() - start)
    return target.detach().cpu().numpy(), action.detach().cpu().numpy(), elapsed_ms


def jax_call(
    policy: torch.nn.Module,
    inputs: tuple[np.ndarray, ...],
    mode: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    weights_np, constants_np, mask_np = _make_weights(SimpleNamespace(policy=policy))
    weights = jax.tree_util.tree_map(jnp.asarray, weights_np)
    constants = jax.tree_util.tree_map(jnp.asarray, constants_np)
    mask = jnp.asarray(mask_np, dtype=bool)
    parents = tuple(int(value) for value in np.asarray(constants_np["parent_indices"]).tolist())

    @jax.jit
    def forward(values, mode_index, time_offsets):
        return _policy_forward(
            values[:5],
            values[5],
            values[6],
            mode_index,
            time_offsets,
            weights,
            constants,
            mask,
            4,
            parents,
        )

    jax_inputs = tuple(jnp.asarray(value) for value in inputs)
    jax_mode = jnp.asarray(mode, dtype=jnp.int32)
    jax_offsets = jnp.asarray(offsets, dtype=jnp.float32)
    start = time.perf_counter()
    output = forward(jax_inputs, jax_mode, jax_offsets)
    jax.block_until_ready(output)
    compile_and_call_ms = 1.0e3 * (time.perf_counter() - start)
    start = time.perf_counter()
    output = forward(jax_inputs, jax_mode, jax_offsets)
    jax.block_until_ready(output)
    steady_ms = 1.0e3 * (time.perf_counter() - start)
    return np.asarray(output[0]), np.asarray(output[1]), compile_and_call_ms, steady_ms


def diff(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rms": float(np.sqrt(np.mean(delta * delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", "--reference", dest="motion", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=SCRIPT_DIR / "artifacts_torso_umi3")
    parser.add_argument("--artifact-manifest", default="artifact_manifest.json")
    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--time-step", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if args.torch_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--torch-device cuda requested, but Torch CUDA is unavailable")

    artifacts = args.artifacts.expanduser().resolve()
    spec = resolve_policy_artifacts(artifacts, args.artifact_manifest)
    meta = json.loads(spec.metadata_path.read_text())
    inputs = frozen_motion_inputs(args.motion.expanduser().resolve(), meta, args.batch, args.time_step)
    mode = np.full((args.batch,), spec.mode_index, dtype=np.int64)
    offsets = np.broadcast_to(
        np.asarray(meta["future_idx"], dtype=np.int64)[None, :, None],
        (args.batch, len(meta["future_idx"]), 1),
    ).copy()

    # Match the production environment's initialization order.  Initializing
    # Torch CUDA first can make JAX/cuDNN startup fail in a shared process.
    jax_device_names = [str(item) for item in jax.devices()]
    jnp.zeros((1,), dtype=jnp.float32).block_until_ready()
    device = torch.device(args.torch_device)
    eager, _ = load_eager_policy(
        artifacts,
        device=device,
        artifact_manifest=args.artifact_manifest,
        verify_checkpoint=False,
    )
    eager_target, eager_action, eager_ms = torch_call(eager, inputs, mode, offsets, device)

    if not spec.torchscript_path.is_file():
        raise FileNotFoundError(f"TorchScript parity artifact is missing: {spec.torchscript_path}")
    script_device = torch.device("cpu")
    scripted = torch.jit.load(str(spec.torchscript_path), map_location=script_device).eval()
    script_target, script_action, script_ms = torch_call(
        scripted, inputs, mode, offsets, script_device
    )

    # Extract the JAX weights from a CPU copy to avoid unnecessary cross-device
    # Torch synchronization before JAX owns the arrays.
    if device.type == "cpu":
        eager_cpu = eager
    else:
        eager_cpu, _ = load_eager_policy(
            artifacts,
            device="cpu",
            artifact_manifest=args.artifact_manifest,
            verify_checkpoint=False,
        )
    jax_target, jax_action, jax_first_ms, jax_steady_ms = jax_call(
        eager_cpu, inputs, mode, offsets
    )

    comparisons = {
        "torchscript_vs_eager": {
            "target": diff(script_target, eager_target),
            "action": diff(script_action, eager_action),
        },
        "jax_vs_eager": {
            "target": diff(jax_target, eager_target),
            "action": diff(jax_action, eager_action),
        },
        "jax_vs_torchscript": {
            "target": diff(jax_target, script_target),
            "action": diff(jax_action, script_action),
        },
    }
    enforced = tuple(comparisons)
    checks = {
        name: {
            "target_max": values["target"]["max_abs"] <= TARGET_MAX_THRESHOLD,
            "action_max": values["action"]["max_abs"] <= ACTION_MAX_THRESHOLD,
            "action_mean": values["action"]["mean_abs"] <= ACTION_MEAN_THRESHOLD,
        }
        for name, values in comparisons.items()
        if name in enforced
    }
    passed = all(all(item.values()) for item in checks.values())
    report = {
        "passed": passed,
        "thresholds": {
            "target_max_abs": TARGET_MAX_THRESHOLD,
            "action_max_abs": ACTION_MAX_THRESHOLD,
            "action_mean_abs": ACTION_MEAN_THRESHOLD,
        },
        "setting": {
            "reference": str(args.motion.expanduser().resolve()),
            "checkpoint": str(spec.checkpoint_path),
            "checkpoint_sha256": spec.checkpoint_sha256,
            "artifact_manifest": str(spec.manifest_path),
            "mode_index": spec.mode_index,
            "mode_name": meta["mode_names"][spec.mode_index],
            "batch": args.batch,
            "time_step": args.time_step,
            "torch_eager_device": str(device),
            "torchscript_device": str(script_device),
            "jax_devices": jax_device_names,
        },
        "timing_ms": {
            "torch_eager": eager_ms,
            "torchscript": script_ms,
            "jax_compile_and_first": jax_first_ms,
            "jax_steady": jax_steady_ms,
        },
        "comparisons": comparisons,
        "checks": checks,
    }
    output_text = json.dumps(report, indent=2)
    print(output_text, flush=True)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
