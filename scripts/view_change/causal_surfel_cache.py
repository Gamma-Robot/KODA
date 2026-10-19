#!/usr/bin/env python3
"""Causal multi-frame RGB-D background fusion.

This module is intentionally independent of the existing viewport warp code.  It
stores confidence-weighted world-coordinate surfels from the current frame and
the most recent ``history_size - 1`` frames, then renders them with a metric
z-buffer into a target camera.  Inputs are expected to be rectified RGB-D in
meters; calibration and dynamic masks are supplied by the caller.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import numpy as np

try:  # Optional: the cache stays usable in light-weight CPU-only tooling.
    import torch
except ImportError:  # pragma: no cover - depends on the deployment environment
    torch = None  # type: ignore[assignment]


PROVENANCE_INVALID = np.uint8(0)
PROVENANCE_OBSERVED = np.uint8(1)
PROVENANCE_FUSED = np.uint8(2)


def _numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert NumPy/Torch inputs without requiring Torch as a dependency."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.asarray(value)
    return out.astype(dtype, copy=False) if dtype is not None else out


@dataclass
class _FrameRecord:
    points_w: np.ndarray
    colors: np.ndarray
    normals_w: np.ndarray
    confidence: np.ndarray
    frame_index: int
    timestamp: float


class CausalSurfelBackgroundCache:
    """Voxel-hashed, confidence-weighted causal surfel cache.

    Parameters
    ----------
    voxel_size:
        World voxel edge in meters.  ``0.005`` (5 mm) is the default.
    history_size:
        Number of frames retained, including the frame most recently added.
    confidence_threshold:
        Samples below this normalized confidence are discarded.
    """

    def __init__(
        self,
        voxel_size: float = 0.005,
        history_size: int = 8,
        confidence_threshold: float = 0.05,
        min_depth: float = 1e-4,
        max_depth: float = 100.0,
        max_points_per_frame: int = 300_000,
        render_device: str | None = None,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        self.voxel_size = float(voxel_size)
        self.history_size = int(history_size)
        self.confidence_threshold = float(confidence_threshold)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.max_points_per_frame = int(max_points_per_frame)
        if self.max_points_per_frame < 1:
            raise ValueError("max_points_per_frame must be >= 1")
        self._frames: Deque[_FrameRecord] = deque(maxlen=self.history_size)
        self._surfels: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._surfel_arrays: Dict[str, np.ndarray] = {}
        # World voxels observed as dynamic are temporarily suppressed so an old
        # hand/arm surfel cannot reappear while the foreground moves away.
        self._dynamic_invalid_until: Dict[Tuple[int, int, int], int] = {}
        self._next_frame = 0
        self._active_frame_index: Optional[int] = None
        self._last_shape: Optional[Tuple[int, int]] = None
        # Rendering can use CUDA without making Torch a hard dependency for
        # cache construction, checkpoint inspection, or unit tests.
        self.render_device = render_device
        self.last_render_backend = "numpy"

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def surfel_count(self) -> int:
        return int(len(self._surfel_arrays.get("position", ())))

    @property
    def processed_frame_count(self) -> int:
        """Number of causally ingested frames, persisted in checkpoints."""
        return int(self._next_frame)

    @property
    def provenance_codes(self) -> Mapping[str, np.uint8]:
        return {"invalid": PROVENANCE_INVALID, "observed": PROVENANCE_OBSERVED, "fused": PROVENANCE_FUSED}

    @staticmethod
    def _estimate_normals_world(
        depth: np.ndarray,
        intrinsics: np.ndarray,
        c2w: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """Estimate oriented surfel normals from local RGB-D geometry.

        A forward-difference stencil is deliberate here: it only accepts a
        normal when the point and its right/down neighbours all passed the
        confidence, edge, and dynamic-mask tests.  This prevents a foreground
        discontinuity from being fused as a background normal.
        """
        H, W = depth.shape
        out = np.zeros((H, W, 3), dtype=np.float32)
        if H < 2 or W < 2 or not np.any(valid):
            return out
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        if abs(fx) < 1e-8 or abs(fy) < 1e-8:
            return out
        yy, xx = np.indices((H, W), dtype=np.float32)
        z = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
        points_c = np.stack(
            (
                (xx - float(intrinsics[0, 2])) * z / fx,
                (yy - float(intrinsics[1, 2])) * z / fy,
                z,
            ),
            axis=-1,
        )
        right = np.zeros_like(points_c)
        down = np.zeros_like(points_c)
        right[:, :-1] = points_c[:, 1:] - points_c[:, :-1]
        down[:-1, :] = points_c[1:, :] - points_c[:-1, :]
        support = valid.copy()
        support[:, :-1] &= valid[:, 1:]
        support[:, -1] = False
        support[:-1, :] &= valid[1:, :]
        support[-1, :] = False
        normal_c = np.cross(right, down)
        norm = np.linalg.norm(normal_c, axis=-1, keepdims=True)
        supported = support & np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
        normal_c = np.divide(normal_c, np.maximum(norm, 1e-8), where=supported[..., None])
        normal_c[~supported] = 0.0
        out = normal_c @ c2w[:3, :3].astype(np.float32, copy=False).T
        return np.nan_to_num(out, nan=0.0).astype(np.float32, copy=False)

    def add_frame(
        self,
        rgb: Any,
        depth_m: Any,
        intrinsics: Any,
        c2w: Any,
        *,
        confidence: Any = None,
        dynamic_mask: Any = None,
        edge_invalid: Any = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, float]:
        """Insert one frame and rebuild the bounded causal cache.

        ``c2w`` maps source camera coordinates to world coordinates.  Dynamic
        and edge-invalid pixels are excluded before voxel aggregation.
        Returns ingestion statistics useful for quality reports.
        """
        image = _numpy(rgb)
        depth = _numpy(depth_m, np.float32)
        K = _numpy(intrinsics, np.float64)
        pose = _numpy(c2w, np.float64)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("rgb must have shape HxWx3")
        H, W = image.shape[:2]
        if depth.shape != (H, W):
            raise ValueError(f"depth shape {depth.shape} does not match rgb {(H, W)}")
        if K.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("intrinsics must be 3x3 and c2w must be 4x4")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(pose)):
            raise ValueError("intrinsics/c2w contain non-finite values")

        if confidence is None:
            conf = np.ones((H, W), dtype=np.float32)
        else:
            conf = _numpy(confidence, np.float32)
            if conf.shape != (H, W):
                raise ValueError("confidence must match depth shape")
            finite_conf = conf[np.isfinite(conf)]
            if finite_conf.size and float(np.max(finite_conf)) > 1.5:
                conf = conf / 255.0
            conf = np.clip(np.nan_to_num(conf, nan=0.0), 0.0, 1.0)
        dyn = np.zeros((H, W), dtype=bool) if dynamic_mask is None else _numpy(dynamic_mask).astype(bool)
        edge = np.zeros((H, W), dtype=bool) if edge_invalid is None else _numpy(edge_invalid).astype(bool)
        if dyn.shape != (H, W) or edge.shape != (H, W):
            raise ValueError("dynamic_mask and edge_invalid must match depth shape")

        valid = np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)
        valid &= ~dyn & ~edge & (conf >= self.confidence_threshold)
        normals_w = self._estimate_normals_world(depth, K, pose, valid)
        if np.any(dyn):
            dyn_y, dyn_x = np.nonzero(
                np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth) & dyn
            )
            if len(dyn_x):
                dyn_z = depth[dyn_y, dyn_x].astype(np.float64)
                dyn_pix = np.stack(
                    (dyn_x.astype(np.float64), dyn_y.astype(np.float64), np.ones(len(dyn_x))), axis=1
                )
                dyn_cam = (dyn_pix @ np.linalg.inv(K).T) * dyn_z[:, None]
                dyn_h = np.concatenate((dyn_cam, np.ones((len(dyn_cam), 1))), axis=1)
                dyn_world = (dyn_h @ pose.T)[:, :3]
                dyn_keys = np.floor(dyn_world / self.voxel_size).astype(np.int64)
                expiry = self._next_frame + max(1, min(3, self.history_size))
                for key_arr in np.unique(dyn_keys, axis=0):
                    self._dynamic_invalid_until[tuple(int(x) for x in key_arr)] = int(expiry)
        ys, xs = np.nonzero(valid)
        if len(xs) > self.max_points_per_frame:
            # Deterministic even sampling avoids a per-frame RNG dependency and
            # bounds cache rebuild memory on high-resolution recordings.
            sample = np.linspace(0, len(xs) - 1, self.max_points_per_frame, dtype=np.int64)
            ys, xs = ys[sample], xs[sample]
        n_valid = int(len(xs))
        if n_valid:
            z = depth[ys, xs].astype(np.float64)
            pix = np.stack((xs.astype(np.float64), ys.astype(np.float64), np.ones(n_valid)), axis=1)
            rays = pix @ np.linalg.inv(K).T
            points_c = rays * z[:, None]
            points_h = np.concatenate((points_c, np.ones((n_valid, 1))), axis=1)
            points_w = (points_h @ pose.T)[:, :3].astype(np.float32)
            colors = np.clip(image[ys, xs, :3], 0, 255).astype(np.uint8)
            frame = _FrameRecord(
                points_w,
                colors,
                normals_w[ys, xs].astype(np.float32),
                conf[ys, xs].astype(np.float32),
                self._next_frame,
                float(timestamp if timestamp is not None else self._next_frame),
            )
        else:
            frame = _FrameRecord(
                np.empty((0, 3), np.float32),
                np.empty((0, 3), np.uint8),
                np.empty((0, 3), np.float32),
                np.empty((0,), np.float32),
                self._next_frame,
                float(timestamp if timestamp is not None else self._next_frame),
            )
        self._next_frame += 1
        self._frames.append(frame)
        self._active_frame_index = frame.frame_index
        self._dynamic_invalid_until = {
            key: expiry for key, expiry in self._dynamic_invalid_until.items() if expiry >= frame.frame_index
        }
        self._last_shape = (H, W)
        self._rebuild()
        return {"frame_index": float(frame.frame_index), "input_pixels": float(H * W), "valid_points": float(n_valid), "valid_ratio": float(n_valid / max(1, H * W)), "dynamic_ratio": float(np.mean(dyn)), "history_frames": float(len(self._frames)), "surfel_count": float(self.surfel_count)}

    def _rebuild(self) -> None:
        """Rebuild vectorized voxel surfels from the bounded frame window."""
        self._surfels.clear()  # retained only for backward-compatible internals
        populated = [f for f in self._frames if len(f.points_w)]
        if not populated:
            self._surfel_arrays = {}
            return
        points = np.concatenate([f.points_w for f in populated], axis=0).astype(np.float64, copy=False)
        colors = np.concatenate([f.colors for f in populated], axis=0).astype(np.float64, copy=False)
        normals = np.concatenate([f.normals_w for f in populated], axis=0).astype(np.float64, copy=False)
        confidence = np.concatenate([f.confidence for f in populated], axis=0).astype(np.float64, copy=False)
        frame_ids = np.concatenate(
            [np.full(len(f.points_w), f.frame_index, dtype=np.int64) for f in populated]
        )
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        if self._dynamic_invalid_until:
            invalid = np.asarray(
                [key for key, expiry in self._dynamic_invalid_until.items() if expiry >= int(self._active_frame_index or 0)],
                dtype=np.int64,
            )
            if len(invalid):
                key_dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
                key_view = np.ascontiguousarray(keys).view(key_dtype).reshape(-1)
                invalid_view = np.ascontiguousarray(invalid.reshape(-1, 3)).view(key_dtype).reshape(-1)
                keep = ~np.isin(key_view, invalid_view)
                points, colors, normals, confidence, frame_ids, keys = (
                    points[keep], colors[keep], normals[keep], confidence[keep], frame_ids[keep], keys[keep]
                )
        if not len(points):
            self._surfel_arrays = {}
            return
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        n_voxels = len(unique)
        weights = np.maximum(confidence, 1e-6)
        sums = np.bincount(inverse, weights=weights, minlength=n_voxels)
        sums = np.maximum(sums, 1e-12)
        counts = np.bincount(inverse, minlength=n_voxels).astype(np.int64)
        position = np.column_stack(
            [np.bincount(inverse, weights=points[:, j] * weights, minlength=n_voxels) / sums for j in range(3)]
        )
        color = np.column_stack(
            [np.bincount(inverse, weights=colors[:, j] * weights, minlength=n_voxels) / sums for j in range(3)]
        )
        conf_mean = np.bincount(inverse, weights=confidence, minlength=n_voxels) / np.maximum(counts, 1)
        normal = np.column_stack(
            [np.bincount(inverse, weights=normals[:, j] * weights, minlength=n_voxels) / sums for j in range(3)]
        )
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.divide(normal, np.maximum(normal_norm, 1e-8), where=normal_norm > 1e-8)
        normal[normal_norm[:, 0] <= 1e-8] = 0.0
        residual = points - position[inverse]
        variance = np.bincount(
            inverse, weights=np.sum(residual * residual, axis=1) * weights, minlength=n_voxels
        ) / sums
        # Count distinct source *frames*, not source pixels, in each voxel.
        ordering = np.lexsort((frame_ids, keys[:, 2], keys[:, 1], keys[:, 0]))
        k_order = keys[ordering]
        f_order = frame_ids[ordering]
        first_pair = np.r_[
            True,
            np.any(k_order[1:] != k_order[:-1], axis=1) | (f_order[1:] != f_order[:-1]),
        ]
        source_count = np.bincount(inverse[ordering[first_pair]], minlength=n_voxels).astype(np.uint32)
        active = int(self._active_frame_index or 0)
        active_count = np.bincount(inverse[frame_ids == active], minlength=n_voxels)
        last_frame = np.full(n_voxels, -1, dtype=np.int64)
        np.maximum.at(last_frame, inverse, frame_ids)
        self._surfel_arrays = {
            "key": unique,
            "position": position,
            "color": color,
            "normal": normal.astype(np.float32),
            "confidence": conf_mean.astype(np.float32),
            "source_count": source_count,
            "observed_only": ((source_count == 1) & (active_count > 0)),
            "last_frame": last_frame,
            "variance": variance.astype(np.float32),
        }

    def render(
        self,
        intrinsics: Any,
        c2w: Any,
        *,
        image_size: Optional[Tuple[int, int]] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        splat_radius_px: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Render the causal cache, preferring CUDA batched projection when set.

        ``render_device`` is intentionally advisory.  A checkpoint can be
        inspected on a CPU-only host, and a CUDA runtime failure falls back to
        the deterministic NumPy z-buffer instead of dropping an episode.
        ``last_render_backend`` records which path produced the frame.
        """
        requested = self.render_device
        if requested and torch is not None:
            wants_cuda = str(requested).startswith("cuda")
            if not wants_cuda or torch.cuda.is_available():
                try:
                    return self._render_torch(
                        intrinsics, c2w, image_size=image_size, near=near, far=far,
                        splat_radius_px=splat_radius_px,
                    )
                except Exception as exc:  # Keep long offline jobs resumable.
                    self.last_render_backend = f"numpy_fallback:{type(exc).__name__}"
                    return self._render_numpy(
                        intrinsics, c2w, image_size=image_size, near=near, far=far,
                        splat_radius_px=splat_radius_px,
                    )
        self.last_render_backend = "numpy"
        return self._render_numpy(
            intrinsics, c2w, image_size=image_size, near=near, far=far,
            splat_radius_px=splat_radius_px,
        )

    def _render_torch(
        self,
        intrinsics: Any,
        c2w: Any,
        *,
        image_size: Optional[Tuple[int, int]] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        splat_radius_px: int = 1,
    ) -> Dict[str, np.ndarray]:
        """CUDA/torch batched project-and-z-buffer path.

        Per-pixel nearest selection is performed on-device.  Maps are copied to
        CPU only once for the PNG/NPY writers used by the pipeline.
        """
        if torch is None:  # pragma: no cover - protected by ``render``
            raise RuntimeError("torch is unavailable")
        if not self._surfel_arrays:
            self.last_render_backend = "torch_empty"
            return self._render_numpy(intrinsics, c2w, image_size=image_size, near=near, far=far)
        K = _numpy(intrinsics, np.float64)
        pose = _numpy(c2w, np.float64)
        if K.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("intrinsics must be 3x3 and c2w must be 4x4")
        H, W = image_size or self._last_shape  # type: ignore[misc]
        H, W = int(H), int(W)
        device = torch.device(str(self.render_device))
        points = torch.as_tensor(self._surfel_arrays["position"], dtype=torch.float32, device=device)
        colors = torch.as_tensor(self._surfel_arrays["color"], dtype=torch.float32, device=device)
        confidences = torch.as_tensor(self._surfel_arrays["confidence"], dtype=torch.float32, device=device)
        source_counts = torch.as_tensor(self._surfel_arrays["source_count"], dtype=torch.int64, device=device)
        observed_only = torch.as_tensor(self._surfel_arrays["observed_only"], dtype=torch.bool, device=device)
        k = torch.as_tensor(K, dtype=torch.float32, device=device)
        world_to_cam = torch.linalg.inv(torch.as_tensor(pose, dtype=torch.float32, device=device))
        cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        z = cam[:, 2]
        n = float(near if near is not None else self.min_depth)
        f = float(far if far is not None else self.max_depth)
        keep = torch.isfinite(z) & (z > n) & (z < f)
        if not bool(keep.any()):
            self.last_render_backend = "torch"
            return self._empty_render(H, W)
        cam, z = cam[keep], z[keep]
        colors, confidences, source_counts, observed_only = (
            colors[keep], confidences[keep], source_counts[keep], observed_only[keep]
        )
        uvw = cam @ k.T
        u = torch.round(uvw[:, 0] / uvw[:, 2]).to(torch.int64)
        v = torch.round(uvw[:, 1] / uvw[:, 2]).to(torch.int64)
        keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not bool(keep.any()):
            self.last_render_backend = "torch"
            return self._empty_render(H, W)
        u, v, z = u[keep], v[keep], z[keep]
        colors, confidences, source_counts, observed_only = (
            colors[keep], confidences[keep], source_counts[keep], observed_only[keep]
        )
        radius = max(0, int(splat_radius_px))
        if radius:
            offsets = [
                (
                    dx,
                    dy,
                    max(
                        0.2,
                        1.0 - 0.2 * (max(abs(dx), abs(dy)) / max(float(radius), 1.0)),
                    ),
                )
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
            ]
            offset_x = torch.as_tensor([x for x, _, _ in offsets], dtype=torch.int64, device=device)
            offset_y = torch.as_tensor([y for _, y, _ in offsets], dtype=torch.int64, device=device)
            falloff = torch.as_tensor([w for _, _, w in offsets], dtype=torch.float32, device=device)
            n_offsets = len(offsets)
            u = (u[:, None] + offset_x[None, :]).reshape(-1)
            v = (v[:, None] + offset_y[None, :]).reshape(-1)
            z = z[:, None].expand(-1, n_offsets).reshape(-1)
            colors = colors[:, None, :].expand(-1, n_offsets, -1).reshape(-1, 3)
            confidences = (confidences[:, None] * falloff[None, :]).reshape(-1)
            source_counts = source_counts[:, None].expand(-1, n_offsets).reshape(-1)
            observed_only = observed_only[:, None].expand(-1, n_offsets).reshape(-1)
        pix = v * W + u
        # The multiplication keeps all samples for one pixel together, then
        # the additive depth term selects the nearest one. ``float64`` avoids
        # loss of integer pixel precision at 4K resolutions.
        order = torch.argsort(pix.to(torch.float64) * (f + 1.0) + z.to(torch.float64))
        pix_sorted = pix[order]
        first = torch.ones_like(pix_sorted, dtype=torch.bool)
        if len(first) > 1:
            first[1:] = pix_sorted[1:] != pix_sorted[:-1]
        sel = order[first]
        pix_cpu = pix[sel].detach().cpu().numpy()
        rgb = np.zeros((H * W, 3), dtype=np.uint8)
        depth = np.zeros(H * W, dtype=np.float32)
        valid = np.zeros(H * W, dtype=bool)
        conf_map = np.zeros(H * W, dtype=np.float32)
        counts = np.zeros(H * W, dtype=np.uint32)
        provenance = np.full(H * W, PROVENANCE_INVALID, dtype=np.uint8)
        rgb[pix_cpu] = np.clip(colors[sel].detach().cpu().numpy(), 0, 255).astype(np.uint8)
        depth[pix_cpu] = z[sel].detach().cpu().numpy().astype(np.float32)
        valid[pix_cpu] = True
        conf_map[pix_cpu] = np.clip(confidences[sel].detach().cpu().numpy(), 0, 1)
        counts[pix_cpu] = source_counts[sel].detach().cpu().numpy().astype(np.uint32)
        provenance[pix_cpu] = np.where(
            observed_only[sel].detach().cpu().numpy(), PROVENANCE_OBSERVED, PROVENANCE_FUSED,
        ).astype(np.uint8)
        self.last_render_backend = "torch"
        return {
            "rgb": rgb.reshape(H, W, 3),
            "depth": depth.reshape(H, W),
            "valid": valid.reshape(H, W),
            "confidence": conf_map.reshape(H, W),
            "source_count": counts.reshape(H, W),
            "provenance": provenance.reshape(H, W),
        }

    @staticmethod
    def _empty_render(H: int, W: int) -> Dict[str, np.ndarray]:
        return {
            "rgb": np.zeros((H, W, 3), dtype=np.uint8),
            "depth": np.zeros((H, W), dtype=np.float32),
            "valid": np.zeros((H, W), dtype=bool),
            "confidence": np.zeros((H, W), dtype=np.float32),
            "source_count": np.zeros((H, W), dtype=np.uint32),
            "provenance": np.full((H, W), PROVENANCE_INVALID, dtype=np.uint8),
        }

    def _render_numpy(
        self,
        intrinsics: Any,
        c2w: Any,
        *,
        image_size: Optional[Tuple[int, int]] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        splat_radius_px: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Render cache into a target camera using nearest-point z-buffering."""
        if not self._surfel_arrays:
            if image_size is None and self._last_shape is None:
                raise ValueError("cannot render an empty cache without image_size")
        K = _numpy(intrinsics, np.float64)
        pose = _numpy(c2w, np.float64)
        if K.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("intrinsics must be 3x3 and c2w must be 4x4")
        H, W = image_size or self._last_shape  # type: ignore[misc]
        H, W = int(H), int(W)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth = np.zeros((H, W), dtype=np.float32)
        valid = np.zeros((H, W), dtype=bool)
        conf_map = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.uint32)
        provenance = np.full((H, W), PROVENANCE_INVALID, dtype=np.uint8)
        if not self._surfel_arrays:
            return {"rgb": rgb, "depth": depth, "valid": valid, "confidence": conf_map, "source_count": counts, "provenance": provenance}
        points = self._surfel_arrays["position"].astype(np.float64, copy=False)
        colors = self._surfel_arrays["color"].astype(np.float64, copy=False)
        confidences = self._surfel_arrays["confidence"].astype(np.float32, copy=False)
        source_counts = self._surfel_arrays["source_count"].astype(np.uint32, copy=False)
        observed_only = self._surfel_arrays["observed_only"].astype(bool, copy=False)
        cam = (np.concatenate((points, np.ones((len(points), 1))), axis=1) @ np.linalg.inv(pose).T)[:, :3]
        z = cam[:, 2]
        n = near if near is not None else self.min_depth
        f = far if far is not None else self.max_depth
        keep = np.isfinite(z) & (z > n) & (z < f)
        if not np.any(keep):
            return {"rgb": rgb, "depth": depth, "valid": valid, "confidence": conf_map, "source_count": counts, "provenance": provenance}
        cam = cam[keep]
        z = z[keep]
        colors, confidences, source_counts, observed_only = (
            colors[keep], confidences[keep], source_counts[keep], observed_only[keep]
        )
        uvw = cam @ K.T
        u = np.rint(uvw[:, 0] / uvw[:, 2]).astype(np.int64)
        v = np.rint(uvw[:, 1] / uvw[:, 2]).astype(np.int64)
        keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(keep):
            return {"rgb": rgb, "depth": depth, "valid": valid, "confidence": conf_map, "source_count": counts, "provenance": provenance}
        u, v, z = u[keep], v[keep], z[keep]
        colors, confidences, source_counts, observed_only = (
            colors[keep], confidences[keep], source_counts[keep], observed_only[keep]
        )
        radius = max(0, int(splat_radius_px))
        if radius:
            offsets = [
                (
                    dx,
                    dy,
                    max(
                        0.2,
                        1.0 - 0.2 * (max(abs(dx), abs(dy)) / max(float(radius), 1.0)),
                    ),
                )
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
            ]
            u0, v0, z0 = u, v, z
            u = np.concatenate([u0 + dx for dx, _, _ in offsets])
            v = np.concatenate([v0 + dy for _, dy, _ in offsets])
            z = np.tile(z0, len(offsets))
            colors = np.concatenate([colors for _ in offsets], axis=0)
            confidences = np.concatenate(
                [confidences * falloff for _, _, falloff in offsets], axis=0
            )
            source_counts = np.tile(source_counts, len(offsets))
            observed_only = np.tile(observed_only, len(offsets))
        pix = v * W + u
        order = np.lexsort((z, pix))
        pix_sorted = pix[order]
        first = np.r_[True, pix_sorted[1:] != pix_sorted[:-1]]
        sel = order[first]
        uu, vv = u[sel], v[sel]
        rgb[vv, uu] = np.clip(colors[sel], 0, 255).astype(np.uint8)
        depth[vv, uu] = z[sel].astype(np.float32)
        valid[vv, uu] = True
        conf_map[vv, uu] = np.clip(confidences[sel], 0, 1)
        counts[vv, uu] = source_counts[sel]
        provenance[vv, uu] = np.where(
            observed_only[sel], PROVENANCE_OBSERVED, PROVENANCE_FUSED
        ).astype(np.uint8)
        return {"rgb": rgb, "depth": depth, "valid": valid, "confidence": conf_map, "source_count": counts, "provenance": provenance}

    def save(self, path: Path | str) -> None:
        """Checkpoint the bounded history and cache metadata."""
        records = list(self._frames)
        offsets = [0]
        pts, cols, normals, conf, ids, times = [], [], [], [], [], []
        for rec in records:
            pts.append(rec.points_w)
            cols.append(rec.colors)
            normals.append(rec.normals_w)
            conf.append(rec.confidence)
            ids.append(rec.frame_index)
            times.append(rec.timestamp)
            offsets.append(offsets[-1] + len(rec.points_w))
        total = offsets[-1]
        with open(Path(path), "wb") as fh:
            shape = self._last_shape or (0, 0)
            invalid_keys = np.asarray(list(self._dynamic_invalid_until.keys()), dtype=np.int64).reshape(-1, 3) if self._dynamic_invalid_until else np.empty((0, 3), dtype=np.int64)
            invalid_until = np.asarray([self._dynamic_invalid_until[tuple(k)] for k in invalid_keys], dtype=np.int64)
            np.savez_compressed(fh, voxel_size=self.voxel_size, history_size=self.history_size, confidence_threshold=self.confidence_threshold, min_depth=self.min_depth, max_depth=self.max_depth, max_points_per_frame=self.max_points_per_frame, next_frame=self._next_frame, shape=np.asarray(shape, dtype=np.int64), offsets=np.asarray(offsets, dtype=np.int64), points=np.concatenate(pts) if total else np.empty((0, 3), np.float32), colors=np.concatenate(cols) if total else np.empty((0, 3), np.uint8), normals=np.concatenate(normals) if total else np.empty((0, 3), np.float32), confidence=np.concatenate(conf) if total else np.empty((0,), np.float32), frame_ids=np.asarray(ids, dtype=np.int64), timestamps=np.asarray(times, dtype=np.float64), dynamic_keys=invalid_keys, dynamic_until=invalid_until)

    @classmethod
    def load(cls, path: Path | str) -> "CausalSurfelBackgroundCache":
        """Load a checkpoint produced by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            max_points = int(data["max_points_per_frame"]) if "max_points_per_frame" in data else 300_000
            obj = cls(float(data["voxel_size"]), int(data["history_size"]), float(data["confidence_threshold"]), float(data["min_depth"]), float(data["max_depth"]), max_points)
            offsets = data["offsets"].astype(np.int64)
            points, colors, conf = data["points"], data["colors"], data["confidence"]
            normals = data["normals"] if "normals" in data else np.zeros_like(points, dtype=np.float32)
            ids, times = data["frame_ids"], data["timestamps"]
            for i in range(len(ids)):
                a, b = int(offsets[i]), int(offsets[i + 1])
                obj._frames.append(
                    _FrameRecord(
                        points[a:b].astype(np.float32),
                        colors[a:b].astype(np.uint8),
                        normals[a:b].astype(np.float32),
                        conf[a:b].astype(np.float32),
                        int(ids[i]),
                        float(times[i]),
                    )
                )
            obj._next_frame = int(data["next_frame"])
            obj._active_frame_index = int(ids[-1]) if len(ids) else None
            if "dynamic_keys" in data and "dynamic_until" in data:
                keys = data["dynamic_keys"].astype(np.int64)
                vals = data["dynamic_until"].astype(np.int64)
                obj._dynamic_invalid_until = {
                    tuple(int(x) for x in key): int(value) for key, value in zip(keys, vals)
                }
            shape = data["shape"].astype(np.int64)
            obj._last_shape = (int(shape[0]), int(shape[1])) if len(shape) == 2 and int(shape[0]) > 0 else None
            obj._rebuild()
        return obj


# Short alias for integrations that prefer the plan's name.
SurfelBackgroundCache = CausalSurfelBackgroundCache
