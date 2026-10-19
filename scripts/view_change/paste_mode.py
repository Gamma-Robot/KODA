#!/usr/bin/env python3
"""Metric z-buffer paste for composite mode A/B.

The formal pipeline uses paste-mode B: non-hole pixels take the closer robot
sample; hole pixels take the robot if it is closer than the median of valid
non-hole background neighbors (and paste if no neighbor exists).

Neighbor median stays on CPU (existing ``nanquantile``). The boolean z-buffer
can run on CUDA when it matches the NumPy reference.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def paste_mask_mode_ab_numpy(
    d_st_raw: np.ndarray,
    d_bg: np.ndarray,
    robot: np.ndarray,
    hole: np.ndarray,
    *,
    paste_mode: str,
    neighbor_depth: np.ndarray | None,
    st_bg_depth: float,
) -> np.ndarray:
    """Return the boolean robot-wins mask. ``neighbor_depth`` is required for B."""
    if paste_mode not in ("A", "B"):
        raise ValueError(f"paste_mode must be A or B, got {paste_mode!r}")
    d_st_raw = np.asarray(d_st_raw, dtype=np.float32)
    d_bg = np.asarray(d_bg, dtype=np.float32)
    robot = np.asarray(robot, dtype=bool)
    hole = np.asarray(hole, dtype=bool)
    d_bg_cmp = np.where(
        np.isfinite(d_bg) & (d_bg > 1e-6),
        d_bg,
        np.float32(st_bg_depth),
    ).astype(np.float32)
    use_st = np.zeros(d_st_raw.shape, dtype=bool)
    non_hole = ~hole
    use_st[non_hole] = robot[non_hole] & (d_st_raw[non_hole] < d_bg_cmp[non_hole])
    in_hole_robot = hole & robot
    if paste_mode == "A":
        use_st[in_hole_robot] = True
        return use_st
    if neighbor_depth is None:
        raise ValueError("paste-mode B requires neighbor_depth")
    d_nb = np.asarray(neighbor_depth, dtype=np.float32)
    has_nb = np.isfinite(d_nb)
    use_st[in_hole_robot & ~has_nb] = True
    use_st[in_hole_robot & has_nb] = d_st_raw[in_hole_robot & has_nb] < d_nb[in_hole_robot & has_nb]
    return use_st


def paste_mask_mode_ab_torch(
    d_st_raw: np.ndarray,
    d_bg: np.ndarray,
    robot: np.ndarray,
    hole: np.ndarray,
    *,
    paste_mode: str,
    neighbor_depth: np.ndarray | None,
    st_bg_depth: float,
    device: str,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required for CUDA paste")
    if paste_mode not in ("A", "B"):
        raise ValueError(f"paste_mode must be A or B, got {paste_mode!r}")
    torch_device = torch.device(device)
    d_st = torch.from_numpy(np.array(d_st_raw, dtype=np.float32, copy=True)).to(torch_device)
    d_bg_t = torch.from_numpy(np.array(d_bg, dtype=np.float32, copy=True)).to(torch_device)
    robot_t = torch.from_numpy(np.array(robot, dtype=bool, copy=True)).to(torch_device)
    hole_t = torch.from_numpy(np.array(hole, dtype=bool, copy=True)).to(torch_device)
    valid_bg = torch.isfinite(d_bg_t) & (d_bg_t > 1e-6)
    d_bg_cmp = torch.where(
        valid_bg, d_bg_t, torch.full((), float(st_bg_depth), dtype=torch.float32, device=torch_device)
    )
    use_st = torch.zeros(d_st.shape, dtype=torch.bool, device=torch_device)
    non_hole = ~hole_t
    use_st = torch.where(non_hole, robot_t & (d_st < d_bg_cmp), use_st)
    in_hole_robot = hole_t & robot_t
    if paste_mode == "A":
        use_st = torch.where(in_hole_robot, torch.ones_like(use_st), use_st)
        return use_st.detach().cpu().numpy()
    if neighbor_depth is None:
        raise ValueError("paste-mode B requires neighbor_depth")
    d_nb = torch.from_numpy(np.array(neighbor_depth, dtype=np.float32, copy=True)).to(torch_device)
    has_nb = torch.isfinite(d_nb)
    hole_no_nb = in_hole_robot & ~has_nb
    hole_with_nb = in_hole_robot & has_nb
    use_st = torch.where(hole_no_nb, torch.ones_like(use_st), use_st)
    use_st = torch.where(hole_with_nb, d_st < d_nb, use_st)
    return use_st.detach().cpu().numpy()


_GPU_PASTE_ALIGNED: bool | None = None


def paste_mask_mode_ab(
    d_st_raw: np.ndarray,
    d_bg: np.ndarray,
    robot: np.ndarray,
    hole: np.ndarray,
    *,
    paste_mode: str,
    neighbor_depth: np.ndarray | None,
    st_bg_depth: float,
    device: str | None = "cpu",
) -> np.ndarray:
    global _GPU_PASTE_ALIGNED
    requested = "cpu" if device is None else str(device)
    if (
        requested.startswith("cuda")
        and torch is not None
        and torch.cuda.is_available()
        and _GPU_PASTE_ALIGNED is not False
    ):
        try:
            gpu = paste_mask_mode_ab_torch(
                d_st_raw,
                d_bg,
                robot,
                hole,
                paste_mode=paste_mode,
                neighbor_depth=neighbor_depth,
                st_bg_depth=st_bg_depth,
                device=requested,
            )
        except Exception:
            _GPU_PASTE_ALIGNED = False
            gpu = None
        if gpu is not None:
            if _GPU_PASTE_ALIGNED is None:
                cpu = paste_mask_mode_ab_numpy(
                    d_st_raw,
                    d_bg,
                    robot,
                    hole,
                    paste_mode=paste_mode,
                    neighbor_depth=neighbor_depth,
                    st_bg_depth=st_bg_depth,
                )
                _GPU_PASTE_ALIGNED = bool(np.array_equal(gpu, cpu))
                if not _GPU_PASTE_ALIGNED:
                    return cpu
            return gpu
    return paste_mask_mode_ab_numpy(
        d_st_raw,
        d_bg,
        robot,
        hole,
        paste_mode=paste_mode,
        neighbor_depth=neighbor_depth,
        st_bg_depth=st_bg_depth,
    )
