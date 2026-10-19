#!/usr/bin/env python3
"""Confidence-aware nearest-depth splat used by RGB-D registration.

Zero-distortion (pinhole) unproject/project runs on CUDA with the z-buffer.
Non-zero OpenCV Brown-Conrady distortion still uses CPU ``undistortPoints`` /
``projectPoints`` once per frame-chunk. The GPU z-buffer is used only when it
matches the CPU reference bit-for-bit.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def splat_offsets(radius: int) -> list[tuple[int, int, float]]:
    radius = max(0, int(radius))
    normalizer = max(float(radius), 1.0)
    offsets: list[tuple[int, int, float]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            falloff = max(0.0, 1.0 - 0.2 * (max(abs(dx), abs(dy)) / normalizer))
            offsets.append((int(dx), int(dy), float(falloff)))
    return offsets


def _splat_inputs(
    uv: np.ndarray,
    z_rgb: np.ndarray,
    src_c: np.ndarray,
    frame_id: np.ndarray | None,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n = max(1, int(n_frames))
    uv = np.asarray(uv, dtype=np.int64).reshape(-1, 2)
    z_rgb = np.asarray(z_rgb, dtype=np.float32).reshape(-1)
    src_c = np.clip(np.asarray(src_c, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if frame_id is None:
        frame_id = np.zeros(uv.shape[0], dtype=np.int64)
    else:
        frame_id = np.asarray(frame_id, dtype=np.int64).reshape(-1)
        if frame_id.shape[0] != uv.shape[0]:
            raise ValueError(
                f"frame_id length {frame_id.shape[0]} != uv length {uv.shape[0]}"
            )
    return uv, z_rgb, src_c, frame_id, n


def splat_zbuffer_cpu_batch(
    uv: np.ndarray,
    z_rgb: np.ndarray,
    src_c: np.ndarray,
    frame_id: np.ndarray | None,
    n_frames: int,
    height: int,
    width: int,
    offsets: Sequence[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass nearest-depth then max-confidence splat over ``n_frames``."""
    h, w = int(height), int(width)
    uv, z_rgb, src_c, frame_id, n = _splat_inputs(uv, z_rgb, src_c, frame_id, n_frames)
    hw = h * w
    flat_depth = np.full(n * hw, np.inf, dtype=np.float32)
    flat_conf = np.zeros(n * hw, dtype=np.float32)
    if uv.size == 0:
        valid = np.zeros((n, h, w), dtype=bool)
        return (
            np.zeros((n, h, w), np.float32),
            np.zeros((n, h, w), np.float32),
            ~valid,
        )
    center_u, center_v = uv[:, 0], uv[:, 1]
    base = frame_id * hw
    for dx, dy, _ in offsets:
        u, v = center_u + dx, center_v + dy
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.any(keep):
            idx = base[keep] + v[keep] * w + u[keep]
            np.minimum.at(flat_depth, idx, z_rgb[keep])
    for dx, dy, falloff in offsets:
        u, v = center_u + dx, center_v + dy
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(keep):
            continue
        idx = base[keep] + v[keep] * w + u[keep]
        z = z_rgb[keep]
        winners = np.abs(z - flat_depth[idx]) <= np.maximum(1e-5, 1e-4 * flat_depth[idx])
        if np.any(winners):
            np.maximum.at(flat_conf, idx[winners], src_c[keep][winners] * float(falloff))
    valid_out = np.isfinite(flat_depth)
    depth = np.where(valid_out, flat_depth, 0.0).reshape(n, h, w).astype(np.float32)
    conf = flat_conf.reshape(n, h, w).astype(np.float32)
    edge_invalid = ~valid_out.reshape(n, h, w)
    return depth, conf, edge_invalid


def splat_zbuffer_cpu(
    uv: np.ndarray,
    z_rgb: np.ndarray,
    src_c: np.ndarray,
    height: int,
    width: int,
    offsets: Sequence[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass nearest-depth then max-confidence splat. Independent of order."""
    depth, conf, edge = splat_zbuffer_cpu_batch(
        uv, z_rgb, src_c, None, 1, height, width, offsets
    )
    return depth[0], conf[0], edge[0]


def _splat_zbuffer_torch_tensors(
    uv: "torch.Tensor",
    z_rgb: "torch.Tensor",
    src_c: "torch.Tensor",
    frame_id: "torch.Tensor",
    n_frames: int,
    height: int,
    width: int,
    offsets: Sequence[tuple[int, int, float]],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Two-pass splat on-device. ``uv`` is (P, 2) int64; outputs are (N, H, W)."""
    if torch is None:
        raise RuntimeError("torch is required for CUDA splat")
    n, h, w = int(n_frames), int(height), int(width)
    hw = h * w
    device = uv.device
    if uv.numel() == 0:
        zeros = torch.zeros((n, h, w), dtype=torch.float32, device=device)
        return zeros, zeros.clone(), torch.ones((n, h, w), dtype=torch.bool, device=device)
    z_t = z_rgb.reshape(-1).to(dtype=torch.float32)
    c_t = src_c.reshape(-1).to(dtype=torch.float32).clamp(0.0, 1.0)
    frame_t = frame_id.reshape(-1).to(dtype=torch.int64)
    uv_t = uv.reshape(-1, 2).to(dtype=torch.int64)
    base = frame_t * hw
    flat_depth = torch.full((n * hw,), float("inf"), dtype=torch.float32, device=device)
    flat_conf = torch.zeros((n * hw,), dtype=torch.float32, device=device)
    center_u = uv_t[:, 0]
    center_v = uv_t[:, 1]
    for dx, dy, _ in offsets:
        u = center_u + int(dx)
        v = center_v + int(dy)
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not bool(keep.any()):
            continue
        idx = base[keep] + v[keep] * w + u[keep]
        flat_depth.scatter_reduce_(0, idx, z_t[keep], reduce="amin", include_self=True)
    for dx, dy, falloff in offsets:
        u = center_u + int(dx)
        v = center_v + int(dy)
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not bool(keep.any()):
            continue
        idx = base[keep] + v[keep] * w + u[keep]
        z = z_t[keep]
        winners = torch.abs(z - flat_depth[idx]) <= torch.maximum(
            torch.full((), 1e-5, dtype=torch.float32, device=device),
            1e-4 * flat_depth[idx],
        )
        if not bool(winners.any()):
            continue
        values = c_t[keep][winners] * float(falloff)
        flat_conf.scatter_reduce_(0, idx[winners], values, reduce="amax", include_self=True)
    valid = torch.isfinite(flat_depth)
    depth = torch.where(valid, flat_depth, torch.zeros_like(flat_depth)).reshape(n, h, w)
    conf = flat_conf.reshape(n, h, w)
    edge = ~valid.reshape(n, h, w)
    return depth, conf, edge


def splat_zbuffer_torch_batch(
    uv: np.ndarray,
    z_rgb: np.ndarray,
    src_c: np.ndarray,
    frame_id: np.ndarray | None,
    n_frames: int,
    height: int,
    width: int,
    offsets: Sequence[tuple[int, int, float]],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("torch is required for CUDA splat")
    h, w = int(height), int(width)
    uv_np, z_np, c_np, frame_np, n = _splat_inputs(uv, z_rgb, src_c, frame_id, n_frames)
    if uv_np.size == 0:
        return splat_zbuffer_cpu_batch(uv_np, z_np, c_np, frame_np, n, h, w, offsets)
    torch_device = torch.device(device)
    depth, conf, edge = _splat_zbuffer_torch_tensors(
        torch.from_numpy(np.ascontiguousarray(uv_np)).to(torch_device),
        torch.from_numpy(np.ascontiguousarray(z_np)).to(torch_device),
        torch.from_numpy(np.ascontiguousarray(c_np)).to(torch_device),
        torch.from_numpy(np.ascontiguousarray(frame_np)).to(torch_device),
        n,
        h,
        w,
        offsets,
    )
    return (
        depth.detach().cpu().numpy().astype(np.float32, copy=False),
        conf.detach().cpu().numpy().astype(np.float32, copy=False),
        edge.detach().cpu().numpy(),
    )


def distortion_is_zero(coeffs: np.ndarray | None) -> bool:
    if coeffs is None:
        return True
    values = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    return values.size == 0 or not np.any(np.abs(values) > 1e-12)


def _k_pinhole(K: np.ndarray, device: "torch.device") -> tuple["torch.Tensor", ...]:
    mat = torch.as_tensor(np.asarray(K, dtype=np.float32).reshape(3, 3), device=device)
    return mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2], mat[0, 1]


def register_depth_pinhole_torch(
    depth: np.ndarray,
    K_depth: np.ndarray,
    K_rgb: np.ndarray,
    T_rgb_from_depth: np.ndarray,
    out_hw: tuple[int, int],
    *,
    confidence: np.ndarray | None = None,
    source_valid: np.ndarray | None = None,
    source_edge_invalid: np.ndarray | None = None,
    splat_radius_px: int | None = None,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole unproject / SE3 / pinhole project / z-buffer, all on one device.

    Matches OpenCV ``undistortPoints`` / ``projectPoints`` with zero distortion
    (including skew). Used when both depth and RGB ``D`` coefficients are zero.
    """
    if torch is None:
        raise RuntimeError("torch is required for CUDA pinhole registration")
    depth_np = np.asarray(depth, dtype=np.float32)
    if depth_np.ndim == 2:
        depth_np = depth_np[None]
    n, h, w = (int(depth_np.shape[0]), int(depth_np.shape[1]), int(depth_np.shape[2]))
    H, W = int(out_hw[0]), int(out_hw[1])
    torch_device = torch.device(device)
    depth_t = torch.from_numpy(np.ascontiguousarray(depth_np)).to(torch_device)
    valid = torch.isfinite(depth_t) & (depth_t > 1e-6)
    if source_valid is not None:
        valid = valid & torch.from_numpy(
            np.ascontiguousarray(np.asarray(source_valid, dtype=bool))
        ).to(torch_device)
    if source_edge_invalid is not None:
        se = torch.from_numpy(
            np.ascontiguousarray(np.asarray(source_edge_invalid, dtype=np.uint8))
        ).to(torch_device)
        if se.ndim == 2:
            se = se[None]
        dilated = torch.nn.functional.max_pool2d(
            se.to(dtype=torch.float32).unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1) > 0
        valid = valid & ~dilated
    empty = (
        np.zeros((n, H, W), np.float32),
        np.zeros((n, H, W), np.float32),
        np.ones((n, H, W), bool),
    )
    if not bool(valid.any()):
        return empty
    if confidence is None:
        conf_t = torch.ones((n, h, w), dtype=torch.float32, device=torch_device)
    else:
        conf_t = torch.from_numpy(
            np.ascontiguousarray(np.asarray(confidence, dtype=np.float32))
        ).to(torch_device)
        if conf_t.ndim == 2:
            conf_t = conf_t[None]
    yy = torch.arange(h, device=torch_device, dtype=torch.float32).view(1, h, 1)
    xx = torch.arange(w, device=torch_device, dtype=torch.float32).view(1, 1, w)
    fx_d, fy_d, cx_d, cy_d, sk_d = _k_pinhole(K_depth, torch_device)
    y_n = (yy - cy_d) / fy_d
    x_n = (xx - cx_d - sk_d * y_n) / fx_d
    z = depth_t
    pts = torch.stack((x_n * z, y_n.expand_as(z) * z, z), dim=-1)
    T = torch.as_tensor(np.asarray(T_rgb_from_depth, dtype=np.float32).reshape(4, 4), device=torch_device)
    pts_rgb = torch.einsum("nhwj,ij->nhwi", pts, T[:3, :3]) + T[:3, 3]
    z_rgb = pts_rgb[..., 2]
    fx_r, fy_r, cx_r, cy_r, sk_r = _k_pinhole(K_rgb, torch_device)
    inv_z = torch.where(z_rgb.abs() > 1e-6, 1.0 / z_rgb, torch.zeros_like(z_rgb))
    u = torch.round(fx_r * pts_rgb[..., 0] * inv_z + sk_r * pts_rgb[..., 1] * inv_z + cx_r).to(torch.int64)
    v = torch.round(fy_r * pts_rgb[..., 1] * inv_z + cy_r).to(torch.int64)
    keep = (
        valid
        & torch.isfinite(pts_rgb).all(dim=-1)
        & (z_rgb > 1e-6)
        & (u >= 0)
        & (u < W)
        & (v >= 0)
        & (v < H)
        & (conf_t >= 0.2)
    )
    if not bool(keep.any()):
        return empty
    keep_f = keep.reshape(-1)
    uv = torch.stack((u.reshape(-1)[keep_f], v.reshape(-1)[keep_f]), dim=1)
    z_keep = z_rgb.reshape(-1)[keep_f]
    c_keep = conf_t.reshape(-1)[keep_f]
    frame_id = (
        torch.arange(n, device=torch_device, dtype=torch.int64)
        .view(n, 1, 1)
        .expand(n, h, w)
        .reshape(-1)[keep_f]
    )
    if splat_radius_px is None:
        splat_radius_px = max(0, int(np.ceil(0.5 * max(W / max(w, 1), H / max(h, 1)))))
    depth_o, conf_o, edge_o = _splat_zbuffer_torch_tensors(
        uv,
        z_keep,
        c_keep,
        frame_id,
        n,
        H,
        W,
        splat_offsets(max(0, int(splat_radius_px))),
    )
    return (
        depth_o.detach().cpu().numpy().astype(np.float32, copy=False),
        conf_o.detach().cpu().numpy().astype(np.float32, copy=False),
        edge_o.detach().cpu().numpy(),
    )


def splat_zbuffer_torch(
    uv: np.ndarray,
    z_rgb: np.ndarray,
    src_c: np.ndarray,
    height: int,
    width: int,
    offsets: Sequence[tuple[int, int, float]],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth, conf, edge = splat_zbuffer_torch_batch(
        uv, z_rgb, src_c, None, 1, height, width, offsets, device=device
    )
    return depth[0], conf[0], edge[0]


def splat_device_available(requested: str | None) -> str:
    """Return ``cuda`` only when torch CUDA is present; otherwise ``cpu``."""
    if requested is None or requested == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cpu"


def splat_zbuffer_batch(
    uv: np.ndarray,
    z_rgb: np.ndarray,
    src_c: np.ndarray,
    frame_id: np.ndarray | None,
    n_frames: int,
    height: int,
    width: int,
    offsets: Sequence[tuple[int, int, float]],
    *,
    device: str | None = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved = splat_device_available(device)
    if resolved.startswith("cuda"):
        try:
            return splat_zbuffer_torch_batch(
                uv,
                z_rgb,
                src_c,
                frame_id,
                n_frames,
                height,
                width,
                offsets,
                device=resolved,
            )
        except Exception:
            pass
    return splat_zbuffer_cpu_batch(
        uv, z_rgb, src_c, frame_id, n_frames, height, width, offsets
    )


def splat_zbuffer(
    uv: np.ndarray,
    z_rgb: np.ndarray,
    src_c: np.ndarray,
    height: int,
    width: int,
    offsets: Sequence[tuple[int, int, float]],
    *,
    device: str | None = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth, conf, edge = splat_zbuffer_batch(
        uv, z_rgb, src_c, None, 1, height, width, offsets, device=device
    )
    return depth[0], conf[0], edge[0]
