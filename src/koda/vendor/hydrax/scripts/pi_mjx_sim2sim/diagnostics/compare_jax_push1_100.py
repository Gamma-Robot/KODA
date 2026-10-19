#!/usr/bin/env python3
"""Compare the isolated JAX planner with the existing push1 MPC path.

The two runs use the same seed, N/H/CEM settings, motion, and execution
policy.  No production runner or output directory is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from cem_wrist import (  # noqa: E402
    CEMConfig,
    adjusted_reference_from_knots,
    build_knot_reference_from_motion,
    candidate_feasible,
    cem_sample,
    cem_update_mean_cov,
    choose_candidate,
    make_sigma_vector,
)
from mjx_pi_env import MjxPiEnv  # noqa: E402
from diagnostics.jax_closed_loop_push1 import (  # noqa: E402
    _make_rollout,
    _to_jax_history,
)


def _run_baseline(env: MjxPiEnv, cfg: CEMConfig, steps: int, seed: int) -> dict:
    rows: list[dict] = []
    plan_s = 0.0
    exec_s = 0.0
    t = 0
    while t < steps:
        u_ref, ref_pos, ref_quat = build_knot_reference_from_motion(
            env.ref_body_pos, env.ref_body_quat, t, env.lw_i, env.rw_i,
            cfg.knot_starts, cfg.num_frames,
        )
        snap = env.snapshot()
        rng = np.random.default_rng(np.random.SeedSequence([seed, t]))
        mean = u_ref.copy()
        cov = make_sigma_vector(cfg)
        t0 = time.perf_counter()
        samples = cem_sample(mean, cov, cfg, rng)
        feasible = np.asarray(candidate_feasible(samples, cfg), dtype=bool)
        safe = samples.copy()
        safe[~feasible] = u_ref
        adjusted_pos, adjusted_quat = adjusted_reference_from_knots(
            safe, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
        )
        costs = np.asarray(
            env.rollout_horizon(
                snap, adjusted_pos, adjusted_quat, cfg.horizon_steps, cfg.w_rot,
                initial_batch=env.broadcast_snapshot(snap, cfg.num_samples),
            ), dtype=np.float64,
        )
        costs[~feasible] = np.inf
        mean, cov = cem_update_mean_cov(samples, costs, cfg)
        best_idx = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
        best = samples[best_idx]
        ref_cost = float(costs[0])
        mean_feasible = bool(candidate_feasible(mean, cfg))
        if mean_feasible:
            mean_pos, mean_quat = adjusted_reference_from_knots(
                mean, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
            )
            mean_cost = float(
                np.asarray(
                    env.rollout_horizon(
                        snap, mean_pos[None], mean_quat[None], cfg.horizon_steps, cfg.w_rot,
                        initial_batch=env.broadcast_snapshot(snap, 1),
                    )
                )[0]
            )
        else:
            mean_cost = None
        selected, kind, _ = choose_candidate(
            u_ref, best, mean, ref_cost, float(costs[best_idx]), mean_cost,
            feasible_best_sample=True, feasible_elite_mean=mean_feasible,
        )
        selected_pos, selected_quat = adjusted_reference_from_knots(
            selected, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
        )
        plan_s += time.perf_counter() - t0

        t_exec = time.perf_counter()
        count = min(cfg.execute_steps, steps - t)
        for local_h in range(count):
            info = env.step_with_window(
                selected_pos[local_h : local_h + cfg.num_frames],
                selected_quat[local_h : local_h + cfg.num_frames],
                advance_time=True,
            )
            pos = np.asarray([info["err_pos_left"], info["err_pos_right"]])
            rot = np.asarray([info["err_rot_left"], info["err_rot_right"]])
            rows.append({
                "pos": float(np.mean(pos)),
                "rot": float(np.mean(rot)),
                "failure": int(np.any(pos > 0.5) or np.any(rot > 1.5)),
                "candidate": kind,
            })
        exec_s += time.perf_counter() - t_exec
        t += count
    return {"rows": rows, "plan_s": plan_s, "exec_s": exec_s}


def _run_jax(
    env: MjxPiEnv,
    cfg: CEMConfig,
    steps: int,
    seed: int,
    rollout_n,
    rollout_1,
) -> dict:
    rows: list[dict] = []
    plan_s = 0.0
    exec_s = 0.0
    t = 0
    while t < steps:
        u_ref, ref_pos, ref_quat = build_knot_reference_from_motion(
            env.ref_body_pos, env.ref_body_quat, t, env.lw_i, env.rw_i,
            cfg.knot_starts, cfg.num_frames,
        )
        snap = env.snapshot()
        rng = np.random.default_rng(np.random.SeedSequence([seed, t]))
        mean = u_ref.copy()
        cov = make_sigma_vector(cfg)
        t0 = time.perf_counter()
        samples = cem_sample(mean, cov, cfg, rng)
        feasible = np.asarray(candidate_feasible(samples, cfg), dtype=bool)
        safe = samples.copy()
        safe[~feasible] = u_ref
        adjusted_pos, adjusted_quat = adjusted_reference_from_knots(
            safe, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
        )
        state_b, hist_b = env.broadcast_snapshot(snap, cfg.num_samples)
        jout = rollout_n(
            state_b,
            _to_jax_history(hist_b),
            jax.numpy.asarray(hist_b.head, dtype=jax.numpy.int32),
            jax.numpy.asarray(adjusted_pos),
            jax.numpy.asarray(adjusted_quat),
            jax.numpy.asarray(t, dtype=jax.numpy.int32),
        )
        jax.block_until_ready(jout)
        _, trace = jout
        costs = np.asarray(trace[0], dtype=np.float64)
        costs[~feasible] = np.inf
        mean, cov = cem_update_mean_cov(samples, costs, cfg)
        best_idx = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
        best = samples[best_idx]
        ref_cost = float(costs[0])
        mean_feasible = bool(candidate_feasible(mean, cfg))
        if mean_feasible:
            mean_pos, mean_quat = adjusted_reference_from_knots(
                mean, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
            )
            state_1, hist_1 = env.broadcast_snapshot(snap, 1)
            jout_1 = rollout_1(
                state_1,
                _to_jax_history(hist_1),
                jax.numpy.asarray(hist_1.head, dtype=jax.numpy.int32),
                jax.numpy.asarray(mean_pos[None]),
                jax.numpy.asarray(mean_quat[None]),
                jax.numpy.asarray(t, dtype=jax.numpy.int32),
            )
            jax.block_until_ready(jout_1)
            _, trace_1 = jout_1
            mean_cost = float(np.asarray(trace_1[0])[0])
        else:
            mean_cost = None
        selected, kind, _ = choose_candidate(
            u_ref, best, mean, ref_cost, float(costs[best_idx]), mean_cost,
            feasible_best_sample=True, feasible_elite_mean=mean_feasible,
        )
        selected_pos, selected_quat = adjusted_reference_from_knots(
            selected, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
        )
        plan_s += time.perf_counter() - t0

        t_exec = time.perf_counter()
        count = min(cfg.execute_steps, steps - t)
        for local_h in range(count):
            info = env.step_with_window(
                selected_pos[local_h : local_h + cfg.num_frames],
                selected_quat[local_h : local_h + cfg.num_frames],
                advance_time=True,
            )
            pos = np.asarray([info["err_pos_left"], info["err_pos_right"]])
            rot = np.asarray([info["err_rot_left"], info["err_rot_right"]])
            rows.append({
                "pos": float(np.mean(pos)),
                "rot": float(np.mean(rot)),
                "failure": int(np.any(pos > 0.5) or np.any(rot > 1.5)),
                "candidate": kind,
            })
        exec_s += time.perf_counter() - t_exec
        t += count
    return {"rows": rows, "plan_s": plan_s, "exec_s": exec_s}


def _summary(run: dict) -> dict:
    rows = run["rows"]
    pos = np.asarray([x["pos"] for x in rows])
    rot = np.asarray([x["rot"] for x in rows])
    return {
        "steps": len(rows),
        "pos_mean": float(np.mean(pos)),
        "rot_mean": float(np.mean(rot)),
        "failure_rate": float(np.mean([x["failure"] for x in rows])),
        "plan_s": float(run["plan_s"]),
        "exec_s": float(run["exec_s"]),
        "wall_s": float(run["plan_s"] + run["exec_s"]),
        "candidate_counts": {
            k: int(sum(x["candidate"] == k for x in rows))
            for k in ("reference", "best_sample", "elite_mean")
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = CEMConfig(num_samples=1024, num_elites=10, horizon_steps=36, execute_steps=3)
    xml = REPO_ROOT / "scalebfm/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml"
    motion = REPO_ROOT / "motion_data/processed/lafan/lafan1_push1_subject2.npz"
    art = SCRIPT_DIR / "artifacts"

    env = MjxPiEnv(str(art), str(xml), str(motion), mode_index=1, torch_device="cuda")
    env.warm_compile_batch(1024)
    env.warm_compile_batch(1)
    env.reset_rsi()
    # Warm the reference path once before the measured 100-step run.
    u_ref, ref_pos, ref_quat = build_knot_reference_from_motion(
        env.ref_body_pos, env.ref_body_quat, 0, env.lw_i, env.rw_i,
        cfg.knot_starts, cfg.num_frames,
    )
    warm_samples = np.repeat(u_ref[None], cfg.num_samples, axis=0)
    warm_pos, warm_quat = adjusted_reference_from_knots(
        warm_samples, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
    )
    snap = env.snapshot()
    env.rollout_horizon(
        snap, warm_pos, warm_quat, cfg.horizon_steps, cfg.w_rot,
        initial_batch=env.broadcast_snapshot(snap, cfg.num_samples),
    )

    baseline = _run_baseline(env, cfg, args.steps, args.seed)

    env.reset_rsi()
    rollout_n = _make_rollout(env, cfg.horizon_steps, cfg.w_rot)
    rollout_1 = _make_rollout(env, cfg.horizon_steps, cfg.w_rot)
    # Compile both fixed batch shapes outside the measured run.
    snap = env.snapshot()
    state_n, hist_n = env.broadcast_snapshot(snap, 1024)
    state_1, hist_1 = env.broadcast_snapshot(snap, 1)
    dummy_pos = jax.numpy.asarray(warm_pos)
    dummy_quat = jax.numpy.asarray(warm_quat)
    jax.block_until_ready(rollout_n(
        state_n, _to_jax_history(hist_n), jax.numpy.asarray(hist_n.head, dtype=jax.numpy.int32),
        dummy_pos, dummy_quat, jax.numpy.asarray(0, dtype=jax.numpy.int32),
    ))
    jax.block_until_ready(rollout_1(
        state_1, _to_jax_history(hist_1), jax.numpy.asarray(hist_1.head, dtype=jax.numpy.int32),
        jax.numpy.asarray(warm_pos[:1]), jax.numpy.asarray(warm_quat[:1]), jax.numpy.asarray(0, dtype=jax.numpy.int32),
    ))
    optimized = _run_jax(env, cfg, args.steps, args.seed, rollout_n, rollout_1)
    result = {
        "setting": {"motion": "push1", "N": 1024, "H": 36, "cem_iters": 1, "seed": args.seed, "steps": args.steps},
        "baseline": _summary(baseline),
        "jax_closed_loop": _summary(optimized),
        "speedup_plan": float(baseline["plan_s"] / optimized["plan_s"]),
        "speedup_wall": float(baseline["plan_s"] + baseline["exec_s"]) / max(optimized["plan_s"] + optimized["exec_s"], 1.0e-9),
        "note": "isolated diagnostic; production pipeline and outputs were not modified",
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
