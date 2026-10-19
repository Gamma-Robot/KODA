#!/usr/bin/env python3
"""Contact evidence from EfficientSAM3 masks and the recorded RGB-D stream.

This is an analysis-only stage.  It intentionally does not import or invoke
ScaleTrack, Mink, MuJoCo rendering, view change, or any inpainting code.
Depth pixels are registered into RGB using the calibrated products, then both
hand and object points are transformed with the recorded source camera pose
into one Y-up world frame.  Finger regions are obtained from the recorded
2-D hand keypoints, constrained by the EfficientSAM3 human mask.

With ``--device cuda`` the registered projection, depth-pixel footprint, mask
sampling, world transform, finger support, and exact finger-to-object nearest
queries stay on Torch CUDA (fp32) until compact metrics are written.  CPU
NumPy/``cKDTree`` remains an explicit fallback for debugging and environments
without a GPU.  Stored object-cloud normals are computed after downsample and
do not drive interval flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

try:
    import cv2
    if not hasattr(cv2, "imread"):
        cv2 = None
except ImportError:  # pragma: no cover - BrainCo optimizer env has Pillow, not OpenCV
    cv2 = None

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only environments use the NumPy path
    torch = None

try:
    from frame_volume import is_packed_stream, load_mask_bool_volume
except ImportError:  # pragma: no cover - package-style execution
    from .frame_volume import is_packed_stream, load_mask_bool_volume


FINGER_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
PRIMARY_FINGERS = ("thumb", "index", "middle")
FINGER_NAMES = tuple(FINGER_INDICES)
SIDES = ("left", "right")
DEFAULT_NEIGHBORHOOD_M = 0.04
# Optional 3-D clearance from the hand cloud.  Off by default: true contact
# surfaces sit inside 20 mm of the finger, and mixed silhouette samples are
# rejected by the RGB footprint classifier instead.
DEFAULT_NEIGHBORHOOD_HAND_CLEARANCE_M = 0.0
VISIBILITY_VISIBLE = "visible"
VISIBILITY_OCCLUDED = "occluded"
VISIBILITY_GRAZING = "grazing"
VISIBILITY_UNKNOWN = "unknown"
YUP_TO_ZUP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--human-mask-dir", type=Path, required=True)
    p.add_argument(
        "--object-mask-root",
        type=Path,
        help="required when --object-source=mask",
    )
    p.add_argument(
        "--object-mask-dir",
        type=Path,
        help="single selected prompt mask directory; avoids mixing prompts per frame",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--side", choices=("left", "right"), default="right")
    p.add_argument(
        "--sides",
        default=None,
        help="comma-separated hands to test; overrides --side (for example left,right)",
    )
    p.add_argument(
        "--object-source",
        choices=("mask", "fingertip_neighborhood"),
        default="mask",
        help="mask uses object SAM pixels; fingertip_neighborhood uses 3-D fingertip "
        "balls for contact gates and a sparse object unprojection around the hand "
        "as the stored object cloud O",
    )
    p.add_argument(
        "--neighborhood-radius-m",
        type=float,
        default=DEFAULT_NEIGHBORHOOD_M,
    )
    p.add_argument(
        "--neighborhood-hand-clearance-m",
        type=float,
        default=DEFAULT_NEIGHBORHOOD_HAND_CLEARANCE_M,
        help="optional extra 3-D clearance from the hand cloud; 0 disables it. "
        "Mixed depth pixels are dropped by --depth-footprint, not by this gap.",
    )
    p.add_argument(
        "--neighborhood-mask-margin-px",
        type=float,
        default=0.0,
        help="deprecated; kept so older shells still parse. Footprint replaces it.",
    )
    p.add_argument(
        "--depth-footprint",
        dest="depth_footprint",
        action="store_true",
        default=True,
        help="classify each depth sample by its RGB footprint vs the human mask",
    )
    p.add_argument(
        "--no-depth-footprint",
        dest="depth_footprint",
        action="store_false",
        help="fall back to a single RGB center sample for the human-mask test",
    )
    p.add_argument(
        "--object-cloud-max-points",
        type=int,
        default=1024,
        help="downsample per-hand neighborhood object points stored in "
        "contact_anchors.npz; mesh pull-out uses this full stored cloud, "
        "while Laplacian SQP may keep a smaller prefix",
    )
    p.add_argument(
        "--object-cloud-radius-m",
        type=float,
        default=0.0,
        help="3-D radius around all MANO joints for stored O. 0 keeps the "
        "tight 2-D dilated fingertip neighborhood",
    )
    p.add_argument(
        "--object-cloud-voxel-m",
        type=float,
        default=0.0,
        help="voxel pitch for sparse stored O. 0 keeps random capping of the "
        "raw unprojection",
    )
    p.add_argument(
        "--neighborhood-dilation-px",
        type=float,
        default=16.0,
        help="dilate the 2-D MANO/finger-support region used to unproject stored O "
        "when --object-cloud-radius-m is 0",
    )
    p.add_argument(
        "--neighborhood-z-band-m",
        type=float,
        default=0.05,
        help="keep neighborhood object pixels within this optical-z band of the hand",
    )
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--finger-width-px", type=float, default=34.0)
    p.add_argument("--min-hand-points", type=int, default=3)
    p.add_argument("--min-object-points", type=int, default=30)
    p.add_argument("--max-object-gap", type=int, default=4)
    p.add_argument("--distance-threshold-m", type=float, default=0.020)
    p.add_argument("--proximity-envelope-m", type=float, default=0.035)
    # The depth agreement gate is intentionally the same 20 mm contract as
    # the 3-D proximity gate.  A looser value admits foreground/background
    # pairs across an occlusion boundary as contact.
    p.add_argument("--depth-gap-threshold-m", type=float, default=0.020)
    p.add_argument("--normal-dot-threshold", type=float, default=0.10)
    p.add_argument("--temporal-gap", type=int, default=3)
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument(
        "--min-fingers",
        type=int,
        default=2,
        help="minimum number of finger groups required per frame "
        "(thumb/index/middle in mask mode; all five in neighborhood mode)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="geometry/nearest-neighbour device: auto, cuda[:N], or cpu",
    )
    p.add_argument(
        "--nearest-chunk-size",
        type=int,
        default=4096,
        help="GPU finger-point chunk size for exact object nearest-neighbour queries",
    )
    return p.parse_args()


def runs(values: list[int], max_gap: int = 0) -> list[list[int]]:
    if not values:
        return []
    ordered = sorted(set(int(v) for v in values))
    out: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - out[-1][-1] - 1 <= int(max_gap):
            out[-1].extend(range(out[-1][-1] + 1, value + 1))
        else:
            out.append([value])
    return out


def fill_mask_gaps(masks: list[np.ndarray | None], max_gap: int) -> tuple[list[np.ndarray | None], list[str]]:
    """Hold a nearby object mask only across short segmentation gaps."""
    result = list(masks)
    source = ["observed" if m is not None and np.any(m) else "missing" for m in masks]
    valid = [i for i, m in enumerate(masks) if m is not None and np.any(m)]
    for left, right in zip(valid, valid[1:]):
        if right - left - 1 <= int(max_gap):
            # Prefer the nearer endpoint.  This is only a mask-support hold,
            # never a depth or pose interpolation.
            for frame in range(left + 1, right):
                result[frame] = masks[left] if frame - left <= right - frame else masks[right]
                source[frame] = "short_gap_hold"
    return result, source


def fill_mask_gaps_stack(
    stack: np.ndarray, max_gap: int
) -> tuple[np.ndarray, list[str]]:
    """In-place-style gap hold on a ``(T,H,W)`` bool stack."""
    filled = np.asarray(stack, dtype=bool).copy()
    valid = np.any(filled.reshape(len(filled), -1), axis=1)
    source = ["observed" if flag else "missing" for flag in valid]
    valid_idx = np.flatnonzero(valid)
    for left, right in zip(valid_idx, valid_idx[1:]):
        if int(right) - int(left) - 1 <= int(max_gap):
            for frame in range(int(left) + 1, int(right)):
                use_left = frame - int(left) <= int(right) - frame
                filled[frame] = stack[int(left)] if use_left else stack[int(right)]
                source[frame] = "short_gap_hold"
    return filled, source


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    image = _read_grayscale(path)
    if image is None:
        return None
    if image.shape != shape:
        image = _resize_nearest(image, shape)
    return image > 0


def _read_grayscale(path: Path) -> np.ndarray | None:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return None if image is None else np.asarray(image)
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def _resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    from PIL import Image

    return np.asarray(
        Image.fromarray(image).resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    )


def choose_object_masks(
    root: Path, n: int, shape: tuple[int, int], max_gap: int
) -> tuple[np.ndarray, list[str]]:
    """Choose the best available SAM3 object prompt, including non-blue cups.

    Historical runs used the explicit blue-cup names below.  New runs may use
    arbitrary prompts (for example ``a black cylindrical cup``), so remaining
    ``*/masks`` directories are appended as deterministic fallbacks.
    """
    preferred = [
        ("prompt_00_a_blue_cup", root / "prompt_00_a_blue_cup" / "masks"),
        ("prompt_01_a_blue_mug", root / "prompt_01_a_blue_mug" / "masks"),
        ("a_blue_cup_filtered", root / "_filtered" / "a_blue_cup" / "masks"),
        ("a_blue_cup", root / "a_blue_cup" / "masks"),
        ("a_blue_mug_filtered", root / "_filtered" / "a_blue_mug" / "masks"),
        ("a_blue_mug", root / "a_blue_mug" / "masks"),
    ]
    candidates = list(preferred)
    seen = {directory.resolve() for _, directory in candidates}
    for directory in sorted(root.glob("*/masks")):
        resolved = directory.resolve()
        if resolved not in seen:
            candidates.append((directory.parent.name, directory))
            seen.add(resolved)
    for directory in sorted(root.glob("_filtered/*/masks")):
        resolved = directory.resolve()
        if resolved not in seen:
            candidates.append((f"{directory.parent.name}_filtered", directory))
            seen.add(resolved)
    if not candidates:
        raise FileNotFoundError(f"no object mask directories under {root}")
    selected = np.zeros((n, int(shape[0]), int(shape[1])), dtype=bool)
    names = ["missing"] * n
    claimed = np.zeros(n, dtype=bool)
    opened = 0
    for name, directory in candidates:
        if bool(claimed.all()):
            break
        try:
            stack = load_mask_bool_volume(directory, count=n, shape=shape)
        except FileNotFoundError:
            continue
        opened += 1
        flat = stack.reshape(n, -1)
        fraction = flat.mean(axis=1)
        usable = (~claimed) & np.any(flat, axis=1) & (fraction <= 0.30)
        if not bool(usable.any()):
            continue
        selected[usable] = stack[usable]
        for frame in np.flatnonzero(usable):
            names[int(frame)] = name
        claimed |= usable
    if opened == 0:
        raise FileNotFoundError(f"no readable object mask streams under {root}")
    filled, sources = fill_mask_gaps_stack(selected, max_gap=max_gap)
    merged = [
        names[index] if source == "observed" else source
        for index, source in enumerate(sources)
    ]
    return filled, merged


def project_depth(
    depth_mm: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
    rgb_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return optical XYZ (m), projected UV, and valid source-pixel indices."""
    h, w = depth_mm.shape
    rows, cols = np.indices((h, w), dtype=np.float64)
    depth = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    valid = np.isfinite(depth) & (depth >= 0.10) & (depth <= 3.0)
    z = depth[valid]
    x = (cols[valid] - depth_k[0, 2]) * z / depth_k[0, 0]
    y = (rows[valid] - depth_k[1, 2]) * z / depth_k[1, 1]
    optical = np.column_stack((x, y, z))
    projected = (rgb_k @ optical.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    uv_round = np.rint(uv).astype(np.int32)
    inside = (
        (uv_round[:, 0] >= 0)
        & (uv_round[:, 0] < rgb_shape[1])
        & (uv_round[:, 1] >= 0)
        & (uv_round[:, 1] < rgb_shape[0])
    )
    return optical, uv_round, valid, inside


def depth_grid_rc(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column/row of each valid depth pixel, matching ``optical`` order."""
    rows, cols = np.indices(valid.shape, dtype=np.float64)
    return cols[valid], rows[valid]


def depth_footprint_uv(
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
    cols: np.ndarray,
    rows: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project depth-pixel corners ``(col±0.5, row±0.5)`` to an RGB box.

    Returns float ``(u0, v0, u1, v1)`` min/max over the four corners.  Boxes
    are not clipped to the image; ``classify_depth_against_mask`` clips.
    """
    cols = np.asarray(cols, dtype=np.float64).reshape(-1)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if cols.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    fx = float(depth_k[0, 0])
    fy = float(depth_k[1, 1])
    cx = float(depth_k[0, 2])
    cy = float(depth_k[1, 2])
    dc = np.asarray([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)
    dr = np.asarray([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    x = (cols[:, None] + dc[None, :] - cx) * z[:, None] / fx
    y = (rows[:, None] + dr[None, :] - cy) * z[:, None] / fy
    xyz = np.stack((x, y, np.broadcast_to(z[:, None], x.shape)), axis=-1)
    proj = xyz.reshape(-1, 3) @ np.asarray(rgb_k, dtype=np.float64).T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)
    uv = uv.reshape(-1, 4, 2)
    return (
        uv[:, :, 0].min(axis=1),
        uv[:, :, 1].min(axis=1),
        uv[:, :, 0].max(axis=1),
        uv[:, :, 1].max(axis=1),
    )


def classify_depth_against_mask(
    human: np.ndarray,
    boxes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label each depth sample as hand / mixed / object_ok from mask fraction.

    ``f = 0`` → object_ok, ``0 < f < 1`` → mixed, ``f = 1`` → hand.
    Mixed samples are never object and never finger depth.  Empty boxes
    (no overlap with the image) are none of the three.
    """
    u0, v0, u1, v1 = boxes
    n = int(np.asarray(u0).reshape(-1).shape[0])
    is_hand = np.zeros(n, dtype=bool)
    is_mixed = np.zeros(n, dtype=bool)
    is_object_ok = np.zeros(n, dtype=bool)
    if n == 0:
        return is_hand, is_mixed, is_object_ok
    height, width = int(human.shape[0]), int(human.shape[1])
    iu0 = np.clip(np.floor(u0).astype(np.int32), 0, width)
    iu1 = np.clip(np.ceil(u1).astype(np.int32), 0, width)
    iv0 = np.clip(np.floor(v0).astype(np.int32), 0, height)
    iv1 = np.clip(np.ceil(v1).astype(np.int32), 0, height)
    sat = np.zeros((height + 1, width + 1), dtype=np.int32)
    sat[1:, 1:] = np.cumsum(
        np.cumsum(np.asarray(human, dtype=np.int32), axis=0), axis=1
    )
    area = (iu1 - iu0) * (iv1 - iv0)
    count = sat[iv1, iu1] - sat[iv0, iu1] - sat[iv1, iu0] + sat[iv0, iu0]
    usable = area > 0
    is_hand[usable] = count[usable] == area[usable]
    is_object_ok[usable] = count[usable] == 0
    is_mixed[usable] = (count[usable] > 0) & (count[usable] < area[usable])
    return is_hand, is_mixed, is_object_ok


def depth_footprint_uv_torch(
    depth_k,
    rgb_k,
    cols,
    rows,
    z,
):
    """CUDA twin of :func:`depth_footprint_uv`."""
    if cols.numel() == 0:
        empty = cols.new_empty(0)
        return empty, empty, empty, empty
    fx = depth_k[0, 0]
    fy = depth_k[1, 1]
    cx = depth_k[0, 2]
    cy = depth_k[1, 2]
    dc = torch.tensor(
        [-0.5, 0.5, -0.5, 0.5], dtype=cols.dtype, device=cols.device
    )
    dr = torch.tensor(
        [-0.5, -0.5, 0.5, 0.5], dtype=cols.dtype, device=cols.device
    )
    x = (cols[:, None] + dc[None, :] - cx) * z[:, None] / fx
    y = (rows[:, None] + dr[None, :] - cy) * z[:, None] / fy
    xyz = torch.stack((x, y, z[:, None].expand_as(x)), dim=-1)
    proj = xyz.reshape(-1, 3) @ rgb_k.T
    uv = proj[:, :2] / proj[:, 2:3].clamp_min(1e-12)
    uv = uv.reshape(-1, 4, 2)
    return (
        uv[:, :, 0].amin(dim=1),
        uv[:, :, 1].amin(dim=1),
        uv[:, :, 0].amax(dim=1),
        uv[:, :, 1].amax(dim=1),
    )


def classify_depth_against_mask_torch(
    human,
    boxes: tuple,
) -> tuple:
    """CUDA twin of :func:`classify_depth_against_mask`."""
    u0, v0, u1, v1 = boxes
    n = int(u0.reshape(-1).shape[0])
    is_hand = torch.zeros(n, dtype=torch.bool, device=human.device)
    is_mixed = torch.zeros(n, dtype=torch.bool, device=human.device)
    is_object_ok = torch.zeros(n, dtype=torch.bool, device=human.device)
    if n == 0:
        return is_hand, is_mixed, is_object_ok
    height, width = int(human.shape[0]), int(human.shape[1])
    iu0 = torch.floor(u0).to(torch.int64).clamp(0, width)
    iu1 = torch.ceil(u1).to(torch.int64).clamp(0, width)
    iv0 = torch.floor(v0).to(torch.int64).clamp(0, height)
    iv1 = torch.ceil(v1).to(torch.int64).clamp(0, height)
    sat = human.new_zeros((height + 1, width + 1), dtype=torch.int32)
    sat[1:, 1:] = torch.cumsum(
        torch.cumsum(human.to(torch.int32), dim=0), dim=1
    )
    area = (iu1 - iu0) * (iv1 - iv0)
    count = sat[iv1, iu1] - sat[iv0, iu1] - sat[iv1, iu0] + sat[iv0, iu0]
    usable = area > 0
    is_hand[usable] = count[usable] == area[usable]
    is_object_ok[usable] = count[usable] == 0
    is_mixed[usable] = (count[usable] > 0) & (count[usable] < area[usable])
    return is_hand, is_mixed, is_object_ok


def _numpy_bool(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(bool, copy=False)


def _numpy_f64(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _len_points(value) -> int:
    if _is_torch(value):
        return int(value.shape[0]) if value.ndim > 0 else 0
    return int(len(value))


def _count_true(mask) -> int:
    if _is_torch(mask):
        return int(mask.sum().item()) if mask.numel() else 0
    return int(np.count_nonzero(mask))


def _as_numpy(value) -> np.ndarray:
    if _is_torch(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def pin_torch_threads() -> None:
    if torch is None:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def to_world_torch(optical, rotation, translation, r_optical_to_link):
    device = optical.device
    dtype = optical.dtype
    rot = torch.as_tensor(np.asarray(rotation), dtype=dtype, device=device)
    trans = torch.as_tensor(np.asarray(translation), dtype=dtype, device=device)
    rlink = torch.as_tensor(np.asarray(r_optical_to_link), dtype=dtype, device=device)
    yup_to_zup = torch.as_tensor(YUP_TO_ZUP, dtype=dtype, device=device)
    world_yup = (rot @ rlink @ optical.T).T + trans.reshape(1, 3)
    return (yup_to_zup @ world_yup.T).T


def finger_support_torch(uv, keypoints: np.ndarray, indices: tuple[int, ...], width: float):
    uv_f = uv.to(torch.float32)
    points = torch.as_tensor(
        np.asarray(keypoints, dtype=np.float32)[list(indices)],
        device=uv.device,
    )
    good = torch.isfinite(points).all(dim=1)
    points = points[good]
    n = int(uv_f.shape[0])
    if points.shape[0] < 2:
        return torch.zeros(n, dtype=torch.bool, device=uv.device)
    best = torch.full((n,), float("inf"), dtype=torch.float32, device=uv.device)
    for start, end in zip(points[:-1], points[1:]):
        vec = end - start
        denom = torch.dot(vec, vec)
        delta = uv_f - start.reshape(1, 2)
        if float(denom) < 1e-9:
            dist = torch.linalg.norm(delta, dim=1)
        else:
            t = torch.clamp((delta @ vec) / denom, 0.0, 1.0)
            closest = start.reshape(1, 2) + t.reshape(-1, 1) * vec.reshape(1, 2)
            dist = torch.linalg.norm(uv_f - closest, dim=1)
        best = torch.minimum(best, dist)
    return best <= float(width)


def estimate_normals_torch(points, indices, camera_position, k: int = 16):
    if points.shape[0] < 4 or int(indices.numel()) == 0:
        return torch.full((int(indices.shape[0]), 3), float("nan"), device=points.device)
    kk = min(int(k), int(points.shape[0]))
    query = points[indices]
    pairwise = torch.cdist(query.to(torch.float32), points.to(torch.float32))
    _, neighbours = torch.topk(pairwise, k=kk, largest=False, sorted=False)
    local = points[neighbours]
    centered = local - local.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered
    _evals, evecs = torch.linalg.eigh(cov)
    normal = evecs[:, :, 0]
    camera = torch.as_tensor(np.asarray(camera_position), dtype=points.dtype, device=points.device).reshape(1, 3)
    flip = (normal * (camera - points[indices])).sum(dim=1) < 0
    normal = torch.where(flip.unsqueeze(1), -normal, normal)
    normal = normal / torch.clamp(torch.linalg.norm(normal, dim=1, keepdim=True), min=1e-12)
    return normal


def lookup_depth_at_uv_torch(
    query_uv: np.ndarray,
    boxes: tuple,
    z,
    is_hand,
    is_object_ok,
    *,
    max_center_px: float = 12.0,
) -> tuple[float, str]:
    query = np.asarray(query_uv, dtype=np.float64).reshape(-1)
    if query.size < 2 or not np.all(np.isfinite(query[:2])) or int(z.numel()) == 0:
        return float("nan"), "none"
    u = float(query[0])
    v = float(query[1])
    u0, v0, u1, v1 = boxes
    contains = (u0 <= u) & (u < u1) & (v0 <= v) & (v < v1)

    def _frontmost(mask) -> int | None:
        selected = contains & mask
        if not bool(selected.any()):
            return None
        idx = torch.nonzero(selected, as_tuple=False).reshape(-1)
        return int(idx[torch.argmin(z[idx])].item())

    hit = _frontmost(is_hand)
    if hit is not None:
        return float(z[hit].item()), "hand"
    hit = _frontmost(is_object_ok)
    if hit is not None:
        return float(z[hit].item()), "object_ok"
    uc = 0.5 * (u0 + u1)
    vc = 0.5 * (v0 + v1)
    dist = torch.hypot(uc - u, vc - v)
    inf = torch.tensor(float("inf"), device=z.device, dtype=dist.dtype)
    hand_d = torch.where(is_hand, dist, inf)
    obj_d = torch.where(is_object_ok, dist, inf)
    j_hand = int(torch.argmin(hand_d).item())
    if float(hand_d[j_hand].item()) <= float(max_center_px):
        return float(z[j_hand].item()), "hand"
    j_obj = int(torch.argmin(obj_d).item())
    if float(obj_d[j_obj].item()) <= float(max_center_px):
        return float(z[j_obj].item()), "object_ok"
    return float("nan"), "none"


def build_rgb_occluder_z(
    depth_mm: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
    rgb_shape: tuple[int, int],
    human: np.ndarray,
    *,
    include_mixed: bool = True,
    return_hand_z: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Frontmost non-hand scene z (metres) on the RGB grid. ``inf`` = no occluder.

    Human-mask pixels are not occluders, so a robot overlay can sit on the
    recorded hand.  Object, mixed silhouette, and background depth still cut
    BrainCo wherever it goes behind the cup, table, or wall.
    """
    height, width = int(rgb_shape[0]), int(rgb_shape[1])
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    hand_z = np.full((height, width), np.inf, dtype=np.float32)
    optical, _uv, valid, inside = project_depth(
        depth_mm, depth_k, rgb_k, (height, width)
    )
    if len(optical) == 0:
        return (zbuf, hand_z) if return_hand_z else zbuf
    cols, rows = depth_grid_rc(valid)
    boxes = depth_footprint_uv(depth_k, rgb_k, cols, rows, optical[:, 2])
    is_hand, is_mixed, is_object_ok = classify_depth_against_mask(human, boxes)
    keep = is_object_ok | (is_mixed if include_mixed else False)
    keep = keep | (inside & ~is_hand & ~is_mixed & ~is_object_ok)
    u0, v0, u1, v1 = boxes
    z = np.asarray(optical[:, 2], dtype=np.float32)

    def _splat(mask: np.ndarray, target: np.ndarray) -> None:
        for index in np.flatnonzero(mask):
            x0 = int(np.clip(np.floor(u0[index]), 0, width))
            x1 = int(np.clip(np.ceil(u1[index]), 0, width))
            y0 = int(np.clip(np.floor(v0[index]), 0, height))
            y1 = int(np.clip(np.ceil(v1[index]), 0, height))
            if x1 <= x0 or y1 <= y0:
                continue
            patch = target[y0:y1, x0:x1]
            np.minimum(patch, z[index], out=patch)

    _splat(keep, zbuf)
    if return_hand_z:
        _splat(is_hand, hand_z)
        return zbuf, hand_z
    return zbuf




def lookup_depth_at_uv(
    query_uv: np.ndarray,
    boxes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    z: np.ndarray,
    is_hand: np.ndarray,
    is_object_ok: np.ndarray,
    is_mixed: np.ndarray,
    *,
    max_center_px: float = 12.0,
) -> tuple[float, str]:
    """Observed z and class at a 2-D keypoint. Mixed samples are never used.

    Prefer a footprint that contains the query.  If only mixed footprints
    cover it, fall back to the nearest hand, then the nearest object_ok,
    within ``max_center_px`` of the query in RGB.
    """
    query = np.asarray(query_uv, dtype=np.float64).reshape(-1)
    if query.size < 2 or not np.all(np.isfinite(query[:2])):
        return float("nan"), "none"
    u, v = float(query[0]), float(query[1])
    u0, v0, u1, v1 = boxes
    if len(z) == 0:
        return float("nan"), "none"
    contains = (u0 <= u) & (u < u1) & (v0 <= v) & (v < v1)
    z_arr = np.asarray(z, dtype=np.float64)

    def _frontmost(mask: np.ndarray) -> int | None:
        if not np.any(mask):
            return None
        idx = np.flatnonzero(mask)
        return int(idx[np.argmin(z_arr[idx])])

    hit = _frontmost(contains & is_hand)
    if hit is not None:
        return float(z_arr[hit]), "hand"
    hit = _frontmost(contains & is_object_ok)
    if hit is not None:
        return float(z_arr[hit]), "object_ok"
    uc = 0.5 * (u0 + u1)
    vc = 0.5 * (v0 + v1)
    dist = np.hypot(uc - u, vc - v)
    hand_d = np.where(is_hand, dist, np.inf)
    obj_d = np.where(is_object_ok, dist, np.inf)
    j_hand = int(np.argmin(hand_d)) if len(hand_d) else -1
    if j_hand >= 0 and hand_d[j_hand] <= float(max_center_px):
        return float(z_arr[j_hand]), "hand"
    j_obj = int(np.argmin(obj_d)) if len(obj_d) else -1
    if j_obj >= 0 and obj_d[j_obj] <= float(max_center_px):
        return float(z_arr[j_obj]), "object_ok"
    return float("nan"), "none"


def classify_fingertip_visibility(
    z_obs: float,
    z_tip: float,
    kind: str,
    margin_m: float,
) -> str:
    """z-buffer visibility: optical z larger is farther from the camera."""
    if kind not in {"hand", "object_ok"}:
        return VISIBILITY_UNKNOWN
    if not np.isfinite(z_obs) or not np.isfinite(z_tip):
        return VISIBILITY_UNKNOWN
    margin = float(margin_m)
    if kind == "hand":
        return VISIBILITY_VISIBLE
    if (z_obs + margin) < float(z_tip):
        return VISIBILITY_OCCLUDED
    if abs(float(z_obs) - float(z_tip)) <= margin:
        return VISIBILITY_GRAZING
    return VISIBILITY_UNKNOWN


def should_use_hand_depth(
    visibility: str,
    n_hand_points: int,
    min_hand_points: int,
) -> bool:
    """Visible / unknown fingers use hand depth; occluded fingers stay on MANO."""
    if str(visibility) == VISIBILITY_OCCLUDED:
        return False
    return int(n_hand_points) >= int(min_hand_points)


def resolve_device(requested: str):
    if torch is None:
        if requested not in {"auto", "cpu"}:
            raise RuntimeError("Torch is not installed; use --device cpu")
        return None
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


class TorchDepthProjector:
    """Registered RGB-D projection with the depth pixel grid cached on CUDA."""

    def __init__(
        self,
        depth_k: np.ndarray,
        rgb_k: np.ndarray,
        depth_shape: tuple[int, int],
        device,
    ) -> None:
        if torch is None or device is None or device.type != "cuda":
            raise ValueError("TorchDepthProjector requires a CUDA device")
        rows, cols = np.indices(depth_shape, dtype=np.float32)
        self.rows = torch.as_tensor(rows.reshape(-1), dtype=torch.float32, device=device)
        self.cols = torch.as_tensor(cols.reshape(-1), dtype=torch.float32, device=device)
        self.depth_k = torch.as_tensor(np.asarray(depth_k), dtype=torch.float32, device=device)
        self.rgb_k = torch.as_tensor(np.asarray(rgb_k), dtype=torch.float32, device=device)
        self.device = device
        self.shape = depth_shape

    def project_tensors(
        self, depth_mm: np.ndarray, rgb_shape: tuple[int, int]
    ):
        """Registered projection kept on CUDA. Valid pixels are C-order ravel."""
        with torch.inference_mode():
            depth = torch.as_tensor(
                np.asarray(depth_mm, dtype=np.float32), device=self.device
            ).reshape(-1)
            depth = depth / 1000.0
            valid = torch.isfinite(depth) & (depth >= 0.10) & (depth <= 3.0)
            z = depth[valid]
            cols = self.cols[valid]
            rows = self.rows[valid]
            x = (cols - self.depth_k[0, 2]) * z / self.depth_k[0, 0]
            y = (rows - self.depth_k[1, 2]) * z / self.depth_k[1, 1]
            optical = torch.stack((x, y, z), dim=1)
            projected = optical @ self.rgb_k.T
            uv_round = torch.round(projected[:, :2] / projected[:, 2:3]).to(torch.int32)
            height, width = int(rgb_shape[0]), int(rgb_shape[1])
            inside = (
                (uv_round[:, 0] >= 0)
                & (uv_round[:, 0] < width)
                & (uv_round[:, 1] >= 0)
                & (uv_round[:, 1] < height)
            )
            return optical, uv_round, valid.reshape(self.shape), inside, cols, rows

    def project(
        self, depth_mm: np.ndarray, rgb_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        optical, uv_round, valid, inside, _cols, _rows = self.project_tensors(
            depth_mm, rgb_shape
        )
        return (
            _numpy_f64(optical).reshape(-1, 3) if optical.numel() else np.zeros((0, 3), dtype=np.float64),
            uv_round.detach().cpu().numpy().astype(np.int32, copy=False),
            _numpy_bool(valid),
            _numpy_bool(inside),
        )

    def label_frame_tensors(
        self,
        depth_mm: np.ndarray,
        human: np.ndarray,
        object_mask: np.ndarray,
        rgb_shape: tuple[int, int],
        *,
        depth_footprint: bool,
    ) -> dict[str, Any]:
        """Project and classify one RGB-D frame, keeping compact arrays on CUDA."""
        with torch.inference_mode():
            optical_t, uv_t, valid_t, inside_t, cols_t, rows_t = self.project_tensors(
                depth_mm, rgb_shape
            )
            human_t = torch.as_tensor(np.asarray(human, dtype=bool), device=self.device)
            object_t = torch.as_tensor(
                np.asarray(object_mask, dtype=bool), device=self.device
            )
            n = int(optical_t.shape[0])
            pixel_object_t = torch.zeros(n, dtype=torch.bool, device=self.device)
            if n > 0 and bool(inside_t.any()):
                uv_long = uv_t.to(torch.int64)
                pixel_object_t[inside_t] = object_t[
                    uv_long[inside_t, 1], uv_long[inside_t, 0]
                ]
            if bool(depth_footprint) and n > 0:
                boxes_t = depth_footprint_uv_torch(
                    self.depth_k, self.rgb_k, cols_t, rows_t, optical_t[:, 2]
                )
                is_hand_t, is_mixed_t, is_object_ok_t = classify_depth_against_mask_torch(
                    human_t, boxes_t
                )
            else:
                is_hand_t = torch.zeros(n, dtype=torch.bool, device=self.device)
                if n > 0 and bool(inside_t.any()):
                    uv_long = uv_t.to(torch.int64)
                    is_hand_t[inside_t] = human_t[
                        uv_long[inside_t, 1], uv_long[inside_t, 0]
                    ]
                is_mixed_t = torch.zeros(n, dtype=torch.bool, device=self.device)
                is_object_ok_t = inside_t & ~is_hand_t
                if n == 0:
                    empty = optical_t.new_empty(0)
                    boxes_t = (empty, empty, empty, empty)
                else:
                    uv_f = uv_t.to(torch.float32)
                    boxes_t = (
                        uv_f[:, 0] - 0.5,
                        uv_f[:, 1] - 0.5,
                        uv_f[:, 0] + 0.5,
                        uv_f[:, 1] + 0.5,
                    )
            height, width = int(self.shape[0]), int(self.shape[1])
            return {
                "optical": optical_t,
                "uv": uv_t,
                "valid": valid_t,
                "inside": inside_t,
                "is_hand": is_hand_t,
                "is_mixed": is_mixed_t,
                "is_object_ok": is_object_ok_t,
                "pixel_object": pixel_object_t,
                "boxes": boxes_t,
                "human_mask_fraction": float(np.mean(human)),
                "object_mask_fraction": float(np.mean(object_mask)),
                "valid_depth_ratio": float(n / max(1, height * width)),
            }

    def label_frame(
        self,
        depth_mm: np.ndarray,
        human: np.ndarray,
        object_mask: np.ndarray,
        rgb_shape: tuple[int, int],
        *,
        depth_footprint: bool,
    ) -> dict[str, Any]:
        """NumPy copy of ``label_frame_tensors`` for CPU tests and the CPU fallback."""
        labeled = self.label_frame_tensors(
            depth_mm,
            human,
            object_mask,
            rgb_shape,
            depth_footprint=bool(depth_footprint),
        )
        optical_t = labeled["optical"]
        uv_t = labeled["uv"]
        if int(optical_t.shape[0]) == 0:
            optical = np.zeros((0, 3), dtype=np.float64)
            uv = np.zeros((0, 2), dtype=np.int32)
        else:
            optical = _numpy_f64(optical_t).reshape(-1, 3)
            uv = uv_t.detach().cpu().numpy().astype(np.int32, copy=False)
        return {
            "optical": optical,
            "uv": uv,
            "valid": _numpy_bool(labeled["valid"]),
            "inside": _numpy_bool(labeled["inside"]),
            "is_hand": _numpy_bool(labeled["is_hand"]),
            "is_mixed": _numpy_bool(labeled["is_mixed"]),
            "is_object_ok": _numpy_bool(labeled["is_object_ok"]),
            "pixel_object": _numpy_bool(labeled["pixel_object"]),
            "boxes": tuple(_numpy_f64(part) for part in labeled["boxes"]),
            "human_mask_fraction": float(labeled["human_mask_fraction"]),
            "object_mask_fraction": float(labeled["object_mask_fraction"]),
            "valid_depth_ratio": float(labeled["valid_depth_ratio"]),
        }

    def masked_optical(
        self, depth_mm: np.ndarray, mask: np.ndarray, rgb_shape: tuple[int, int]
    ) -> np.ndarray:
        """Unproject depth and keep samples that land on a true RGB mask pixel."""
        optical_t, uv_t, _valid_t, inside_t, _cols, _rows = self.project_tensors(
            depth_mm, rgb_shape
        )
        if optical_t.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        mask_t = torch.as_tensor(np.asarray(mask, dtype=bool), device=self.device)
        selected = torch.zeros(
            optical_t.shape[0], dtype=torch.bool, device=self.device
        )
        if bool(inside_t.any()):
            uv_long = uv_t.to(torch.int64)
            selected[inside_t] = mask_t[uv_long[inside_t, 1], uv_long[inside_t, 0]]
        kept = optical_t[selected]
        if kept.numel() == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return _numpy_f64(kept).reshape(-1, 3)


def label_registered_depth(
    depth_mm: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
    human: np.ndarray,
    object_mask: np.ndarray,
    rgb_shape: tuple[int, int],
    *,
    depth_footprint: bool,
    projector: TorchDepthProjector | None = None,
) -> dict[str, Any]:
    """Project one RGB-D frame and classify depth pixels against the human mask.

    The CUDA path keeps projection, footprint boxes, and the integral-image
    classifier on device.  Compact NumPy arrays are returned for the existing
    finger/world/nearest-neighbour loop.
    """
    if projector is not None:
        return projector.label_frame(
            depth_mm,
            human,
            object_mask,
            rgb_shape,
            depth_footprint=bool(depth_footprint),
        )
    optical, uv, valid_flat, inside = project_depth(
        depth_mm, depth_k, rgb_k, rgb_shape
    )
    pixel_object = np.zeros(len(optical), dtype=bool)
    inside_idx = np.flatnonzero(inside)
    if inside_idx.size:
        pixel_object[inside_idx] = np.asarray(object_mask, dtype=bool)[
            uv[inside_idx, 1], uv[inside_idx, 0]
        ]
    if bool(depth_footprint) and len(optical) > 0:
        cols, rows = depth_grid_rc(valid_flat)
        boxes = depth_footprint_uv(depth_k, rgb_k, cols, rows, optical[:, 2])
        is_hand, is_mixed, is_object_ok = classify_depth_against_mask(human, boxes)
    else:
        is_hand = np.zeros(len(optical), dtype=bool)
        if inside_idx.size:
            is_hand[inside_idx] = np.asarray(human, dtype=bool)[
                uv[inside_idx, 1], uv[inside_idx, 0]
            ]
        is_mixed = np.zeros(len(optical), dtype=bool)
        is_object_ok = inside & ~is_hand
        boxes = (
            uv[:, 0].astype(np.float64) - 0.5,
            uv[:, 1].astype(np.float64) - 0.5,
            uv[:, 0].astype(np.float64) + 0.5,
            uv[:, 1].astype(np.float64) + 0.5,
        )
    height, width = int(valid_flat.shape[0]), int(valid_flat.shape[1])
    return {
        "optical": optical,
        "uv": uv,
        "valid": valid_flat,
        "inside": inside,
        "is_hand": is_hand,
        "is_mixed": is_mixed,
        "is_object_ok": is_object_ok,
        "pixel_object": pixel_object,
        "boxes": boxes,
        "human_mask_fraction": float(np.mean(human)),
        "object_mask_fraction": float(np.mean(object_mask)),
        "valid_depth_ratio": float(len(optical) / max(1, height * width)),
    }


def unproject_masked_optical(
    depth_mm: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
    mask: np.ndarray,
    *,
    projector: TorchDepthProjector | None = None,
) -> np.ndarray:
    """Unproject metric depth and keep samples that land on a true RGB mask pixel."""
    rgb_shape = (int(mask.shape[0]), int(mask.shape[1]))
    if projector is not None:
        return projector.masked_optical(depth_mm, mask, rgb_shape)
    optical, uv, _valid, inside = project_depth(
        depth_mm, depth_k, rgb_k, rgb_shape
    )
    selected = np.zeros(len(optical), dtype=bool)
    inside_idx = np.flatnonzero(inside)
    if inside_idx.size:
        selected[inside_idx] = np.asarray(mask, dtype=bool)[
            uv[inside_idx, 1], uv[inside_idx, 0]
        ]
    return optical[selected]


def nearest_object_points(
    finger_points: np.ndarray,
    object_points: np.ndarray,
    tree: cKDTree | None,
    *,
    device,
    object_tensor=None,
    chunk_size: int = 4096,
    return_tensors: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Euclidean nearest neighbours, using CUDA when requested."""
    if device is None or getattr(device, "type", None) != "cuda":
        if tree is None:
            empty_d = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.int64)
            return empty_d, empty_i
        distances, nearest = tree.query(finger_points, k=1, workers=1)
        return np.asarray(distances, dtype=np.float64), np.asarray(nearest, dtype=np.int64)
    n_finger = _len_points(finger_points)
    n_object = _len_points(object_points if object_tensor is None else object_tensor)
    if n_finger == 0 or n_object == 0:
        if return_tensors:
            empty = torch.empty(0, dtype=torch.float32, device=device)
            return empty, torch.empty(0, dtype=torch.int64, device=device)
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    if object_tensor is None:
        object_tensor = torch.as_tensor(object_points, dtype=torch.float32, device=device)
    else:
        object_tensor = object_tensor.to(dtype=torch.float32)
    if _is_torch(finger_points):
        finger_tensor = finger_points.to(dtype=torch.float32)
    else:
        finger_tensor = torch.as_tensor(finger_points, dtype=torch.float32, device=device)
    chunk = max(1, int(chunk_size))
    max_pairwise_elements = 4_000_000
    distance_chunks: list = []
    nearest_chunks: list = []
    for start in range(0, n_finger, chunk):
        fingers = finger_tensor[start : start + chunk]
        object_chunk = max(1, max_pairwise_elements // max(1, int(fingers.shape[0])))
        values = torch.full(
            (int(fingers.shape[0]),), torch.inf, dtype=torch.float32, device=device
        )
        indices = torch.zeros((int(fingers.shape[0]),), dtype=torch.int64, device=device)
        for object_start in range(0, n_object, object_chunk):
            pairwise = torch.cdist(
                fingers, object_tensor[object_start : object_start + object_chunk]
            )
            local_values, local_indices = torch.min(pairwise, dim=1)
            update = local_values < values
            values = torch.where(update, local_values, values)
            indices = torch.where(
                update, local_indices + int(object_start), indices
            )
        distance_chunks.append(values)
        nearest_chunks.append(indices)
    distances_t = torch.cat(distance_chunks)
    nearest_t = torch.cat(nearest_chunks)
    if return_tensors:
        return distances_t, nearest_t
    return (
        distances_t.detach().cpu().numpy().astype(np.float64, copy=False),
        nearest_t.detach().cpu().numpy().astype(np.int64, copy=False),
    )


def to_world(optical: np.ndarray, rotation: np.ndarray, translation: np.ndarray, r_optical_to_link: np.ndarray) -> np.ndarray:
    world_yup = (np.asarray(rotation, dtype=np.float64) @ r_optical_to_link @ optical.T).T
    world_yup += np.asarray(translation, dtype=np.float64)
    return (YUP_TO_ZUP @ world_yup.T).T


def finger_support(uv: np.ndarray, keypoints: np.ndarray, indices: tuple[int, ...], width: float, shape: tuple[int, int]) -> np.ndarray:
    """Distance-to-polyline test in RGB pixels, with robust invalid handling."""
    points = np.asarray(keypoints, dtype=np.float64)[list(indices)]
    good = np.isfinite(points).all(axis=1)
    points = points[good]
    if len(points) < 2:
        return np.zeros(len(uv), dtype=bool)
    best = np.full(len(uv), np.inf, dtype=np.float64)
    for a, b in zip(points[:-1], points[1:]):
        v = b - a
        denom = float(np.dot(v, v))
        if denom < 1e-9:
            dist = np.linalg.norm(uv - a[None, :], axis=1)
        else:
            t = np.clip(((uv - a[None, :]) @ v) / denom, 0.0, 1.0)
            closest = a[None, :] + t[:, None] * v[None, :]
            dist = np.linalg.norm(uv - closest, axis=1)
        best = np.minimum(best, dist)
    return best <= float(width)


def project_optical_to_rgb(optical: np.ndarray, rgb_k: np.ndarray) -> np.ndarray:
    """Project optical-frame points with ``rgb_K`` into RGB pixels."""
    pts = np.asarray(optical, dtype=np.float64).reshape(-1, 3)
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    if len(pts) == 0:
        return uv
    z = pts[:, 2]
    good = np.isfinite(pts).all(axis=1) & (z > 1e-6)
    if not np.any(good):
        return uv
    projected = (np.asarray(rgb_k, dtype=np.float64) @ pts[good].T).T
    uv[good] = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
    return uv


def sample_mask_at_uv(uv: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if mask is None or len(uv_arr) == 0:
        return np.zeros(len(uv_arr), dtype=bool)
    height, width = np.asarray(mask).shape[:2]
    rounded = np.rint(uv_arr).astype(np.int64)
    inside = (
        np.isfinite(uv_arr).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < int(width))
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < int(height))
    )
    hit = np.zeros(len(uv_arr), dtype=bool)
    if np.any(inside):
        hit[inside] = np.asarray(mask, dtype=bool)[
            rounded[inside, 1], rounded[inside, 0]
        ]
    return hit


def _keypoint_disk_support(
    uv: np.ndarray, keypoints: np.ndarray, radius_px: float
) -> np.ndarray:
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    pts = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(uv_arr) == 0 or len(pts) == 0 or float(radius_px) < 0.0:
        return np.zeros(len(uv_arr), dtype=bool)
    delta = uv_arr[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    return np.min(dist, axis=1) <= float(radius_px)


def mano_neighborhood_support(
    uv: np.ndarray,
    keypoints: np.ndarray,
    *,
    finger_width_px: float,
    dilation_px: float,
    human: np.ndarray | None = None,
    shape: tuple[int, int] | None = None,
    projected_keypoints: np.ndarray | None = None,
) -> np.ndarray:
    """Dilated 2-D support from MANO/RGB keypoints, bone bands, and the human mask."""
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    support = np.zeros(len(uv_arr), dtype=bool)
    width = float(finger_width_px) + float(dilation_px)
    sources = [np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)]
    if projected_keypoints is not None:
        sources.append(np.asarray(projected_keypoints, dtype=np.float64).reshape(-1, 2))
    hw = shape
    if hw is None and human is not None:
        hw = tuple(int(v) for v in np.asarray(human).shape[:2])
    if hw is None:
        hw = (1080, 1920)
    for pts in sources:
        if len(pts) < 2:
            continue
        for inds in FINGER_INDICES.values():
            if max(inds) >= len(pts):
                continue
            support |= finger_support(uv_arr, pts, inds, width, hw)
        support |= _keypoint_disk_support(uv_arr, pts, dilation_px)
    if human is not None:
        support |= sample_mask_at_uv(uv_arr, human)
    return support


def neighborhood_z_gate(
    optical_z: np.ndarray,
    support: np.ndarray,
    is_hand: np.ndarray,
    mano_optical_z: np.ndarray,
    z_band_m: float,
) -> np.ndarray:
    """Keep pixels near the median hand/MANO optical depth."""
    z = np.asarray(optical_z, dtype=np.float64).reshape(-1)
    support_m = np.asarray(support, dtype=bool).reshape(-1)
    hand_m = np.asarray(is_hand, dtype=bool).reshape(-1)
    mano_z = np.asarray(mano_optical_z, dtype=np.float64).reshape(-1)
    hand_z = z[support_m & hand_m]
    hand_z = hand_z[np.isfinite(hand_z)]
    if len(hand_z) > 0:
        z_ref = float(np.median(hand_z))
    else:
        finite_mano = mano_z[np.isfinite(mano_z)]
        if len(finite_mano) == 0:
            return np.zeros(len(z), dtype=bool)
        z_ref = float(np.median(finite_mano))
    return np.isfinite(z) & (np.abs(z - z_ref) < float(z_band_m))


def nearest_mano_distance(points: np.ndarray, mano_points: np.ndarray) -> np.ndarray:
    """Minimum 3-D distance from each sample to a finite MANO joint."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mano = np.asarray(mano_points, dtype=np.float64).reshape(-1, 3)
    finite_pts = np.isfinite(pts).all(axis=1)
    finite_mano = np.isfinite(mano).all(axis=1)
    mano = mano[finite_mano]
    dist = np.full(len(pts), np.inf, dtype=np.float64)
    if len(pts) == 0 or len(mano) == 0 or not bool(np.any(finite_pts)):
        return dist
    delta = pts[finite_pts, None, :] - mano[None, :, :]
    dist[finite_pts] = np.linalg.norm(delta, axis=2).min(axis=1)
    return dist


def hand_neighborhood_object_sel(
    *,
    uv: np.ndarray,
    optical_z: np.ndarray,
    is_object_ok: np.ndarray,
    is_hand: np.ndarray,
    keypoints: np.ndarray,
    mano_optical: np.ndarray,
    rgb_k: np.ndarray,
    human: np.ndarray | None = None,
    finger_width_px: float = 34.0,
    dilation_px: float = 16.0,
    z_band_m: float = 0.05,
    shape: tuple[int, int] | None = None,
    points: np.ndarray | None = None,
    mano_points: np.ndarray | None = None,
    cloud_radius_m: float = 0.0,
) -> np.ndarray:
    """Unproject ``object_ok`` depth near the hand as stored Laplacian / pull-out O.

    A positive ``cloud_radius_m`` keeps every object sample within that 3-D
    distance of a MANO joint.  Radius 0 falls back to the tight 2-D dilated
    fingertip neighborhood.
    """
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    object_ok = np.asarray(is_object_ok, dtype=bool).reshape(-1)
    if len(uv_arr) == 0:
        return np.zeros(0, dtype=bool)
    mano_z = np.asarray(mano_optical, dtype=np.float64).reshape(-1, 3)[:, 2]
    if float(cloud_radius_m) > 0.0:
        if points is None:
            raise ValueError("cloud_radius_m > 0 requires 3-D sample points")
        mano_xyz = (
            np.asarray(mano_points, dtype=np.float64)
            if mano_points is not None
            else np.asarray(mano_optical, dtype=np.float64)
        )
        support = nearest_mano_distance(points, mano_xyz) <= float(cloud_radius_m)
        z_ok = neighborhood_z_gate(optical_z, support, is_hand, mano_z, z_band_m)
        return support & object_ok & z_ok
    projected = project_optical_to_rgb(mano_optical, rgb_k)
    support = mano_neighborhood_support(
        uv_arr,
        keypoints,
        finger_width_px=float(finger_width_px),
        dilation_px=float(dilation_px),
        human=human,
        shape=shape,
        projected_keypoints=projected,
    )
    z_ok = neighborhood_z_gate(optical_z, support, is_hand, mano_z, z_band_m)
    return support & object_ok & z_ok


def estimate_normals(points: np.ndarray, tree: cKDTree, indices: np.ndarray, camera_position: np.ndarray, k: int = 16) -> np.ndarray:
    if len(points) < 4 or len(indices) == 0:
        return np.full((len(indices), 3), np.nan, dtype=np.float64)
    kk = min(int(k), len(points))
    neighbours = tree.query(points[indices], k=kk, workers=1)[1]
    if kk == 1:
        neighbours = neighbours[:, None]
    normals = np.full((len(indices), 3), np.nan, dtype=np.float64)
    for i, ids in enumerate(np.asarray(neighbours)):
        local = points[np.asarray(ids, dtype=np.int64)]
        centered = local - local.mean(axis=0)
        cov = centered.T @ centered
        _, vecs = np.linalg.eigh(cov)
        normal = vecs[:, 0]
        if np.dot(normal, camera_position - points[indices[i]]) < 0.0:
            normal = -normal
        normals[i] = normal / max(np.linalg.norm(normal), 1e-12)
    return normals


def classify(values: list[bool], max_gap: int, min_run: int) -> list[list[int]]:
    candidates = [i for i, value in enumerate(values) if value]
    return [run for run in runs(candidates, max_gap=max_gap) if len(run) >= int(min_run)]


def parse_sides(side: str, sides: str | None) -> tuple[str, ...]:
    if sides is None or not str(sides).strip():
        if side not in SIDES:
            raise ValueError(f"invalid side: {side}")
        return (side,)
    parsed = tuple(item.strip() for item in str(sides).split(",") if item.strip())
    invalid = [item for item in parsed if item not in SIDES]
    if invalid:
        raise ValueError(f"invalid --sides entries: {invalid}")
    if not parsed:
        raise ValueError("--sides is empty")
    unique = tuple(dict.fromkeys(parsed))
    return unique


def flags_from_runs(n: int, run_list: list[list[int]]) -> np.ndarray:
    flags = np.zeros(int(n), dtype=np.int8)
    for run in run_list:
        for frame in run:
            if 0 <= int(frame) < len(flags):
                flags[int(frame)] = 1
    return flags


def neighborhood_object_sel(
    world: np.ndarray,
    center: np.ndarray,
    radius_m: float,
    not_human: np.ndarray,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    """Depth points in a 3-D ball around a fingertip, excluding the hand."""
    if _is_torch(world):
        n = int(world.shape[0])
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=world.device)
        if _is_torch(center):
            center_t = center.to(dtype=world.dtype).reshape(-1)
        else:
            center_t = torch.as_tensor(
                np.asarray(center), dtype=world.dtype, device=world.device
            )
        if not bool(torch.isfinite(center_t).all()):
            return torch.zeros(n, dtype=torch.bool, device=world.device)
        dist = torch.linalg.norm(world - center_t.reshape(1, 3), dim=1)
        selected = not_human & torch.isfinite(dist) & (dist <= float(radius_m))
        if extra is not None:
            selected = selected & extra
        return selected
    if len(world) == 0 or not np.all(np.isfinite(center)):
        return np.zeros(len(world), dtype=bool)
    delta = np.asarray(world, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    dist = np.linalg.norm(delta, axis=1)
    selected = (
        np.asarray(not_human, dtype=bool)
        & np.isfinite(dist)
        & (dist <= float(radius_m))
    )
    if extra is not None:
        selected = selected & np.asarray(extra, dtype=bool)
    return selected


def voxel_keep_indices(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """One index per occupied voxel, preferring the sample nearest the voxel center."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pitch = float(voxel_m)
    if len(pts) == 0 or pitch <= 0.0:
        return np.arange(len(pts), dtype=np.int64)
    finite = np.isfinite(pts).all(axis=1)
    if not bool(np.any(finite)):
        return np.zeros(0, dtype=np.int64)
    keys = np.zeros((len(pts), 3), dtype=np.int64)
    keys[finite] = np.floor(pts[finite] / pitch).astype(np.int64)
    centers = (keys.astype(np.float64) + 0.5) * pitch
    dist2 = np.sum((pts - centers) ** 2, axis=1)
    dist2[~finite] = np.inf
    order = np.lexsort((dist2, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    changed = np.ones(len(sorted_keys), dtype=bool)
    if len(sorted_keys) > 1:
        changed[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    keep = order[changed]
    keep = keep[finite[keep]]
    return np.sort(keep)


def farthest_point_indices(points: np.ndarray, count: int, seed: int = 17) -> np.ndarray:
    """Greedy farthest-point order so a prefix still covers the cloud."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    k = max(0, min(int(count), n))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = int(rng.integers(n))
    dist = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        last = pts[chosen[i - 1]]
        delta = pts - last
        dist = np.minimum(dist, np.einsum("ij,ij->i", delta, delta))
        chosen[i] = int(np.argmax(dist))
    return chosen


def downsample_object_cloud(
    points: np.ndarray,
    normals: np.ndarray,
    max_points: int = 1024,
    seed: int = 17,
    voxel_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a sparse neighborhood object cloud (and matching normals).

    A positive ``voxel_m`` collapses dense unprojections to one sample per
    voxel, then farthest-point sampling orders them so a Laplacian prefix
    still spans the object.  Voxel size 0 keeps the old random cap.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts, nrm[:0]
    if len(nrm) != len(pts):
        nrm = np.zeros_like(pts)
    if float(voxel_m) > 0.0:
        keep = voxel_keep_indices(pts, voxel_m)
        pts = pts[keep]
        nrm = nrm[keep]
    cap = max(1, int(max_points))
    if len(pts) == 0:
        return pts, nrm
    if float(voxel_m) > 0.0:
        order = farthest_point_indices(pts, min(len(pts), cap), seed=int(seed))
        return pts[order], nrm[order]
    if len(pts) <= cap:
        return pts, nrm
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(len(pts), size=cap, replace=False))
    return pts[keep], nrm[keep]


def away_from_hand_cloud(
    world: np.ndarray,
    hand_sel: np.ndarray,
    not_human: np.ndarray,
    clearance_m: float,
) -> np.ndarray:
    """True for non-hand samples at least ``clearance_m`` from the hand cloud."""
    if _is_torch(world):
        keep = torch.ones(int(world.shape[0]), dtype=torch.bool, device=world.device)
        if float(clearance_m) <= 0.0 or not bool(hand_sel.any()) or not bool(not_human.any()):
            return keep
        hand_pts = world[hand_sel]
        other_pts = world[not_human]
        pairwise = torch.cdist(other_pts.to(torch.float32), hand_pts.to(torch.float32))
        dist = pairwise.min(dim=1).values
        keep[not_human] = dist >= float(clearance_m)
        return keep
    keep = np.ones(len(world), dtype=bool)
    hand = np.asarray(hand_sel, dtype=bool)
    other = np.asarray(not_human, dtype=bool)
    if float(clearance_m) <= 0.0 or not np.any(hand) or not np.any(other):
        return keep
    tree = cKDTree(np.asarray(world[hand], dtype=np.float64))
    dist, _ = tree.query(np.asarray(world[other], dtype=np.float64), k=1, workers=1)
    keep[other] = np.asarray(dist, dtype=np.float64) >= float(clearance_m)
    return keep


def _empty_finger_metric() -> dict[str, Any]:
    return {
        "hand_points": 0,
        "valid": False,
        "raw_contact": False,
        "envelope_contact": False,
    }


def finger_contact_metric(
    finger_points: np.ndarray,
    finger_optical: np.ndarray,
    object_points: np.ndarray,
    object_optical: np.ndarray,
    tree: cKDTree | None,
    camera_world: np.ndarray,
    *,
    min_hand_points: int,
    distance_threshold_m: float,
    proximity_envelope_m: float,
    depth_gap_threshold_m: float,
    normal_dot_threshold: float,
    device,
    object_tensor=None,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    n_finger = _len_points(finger_points)
    n_object = _len_points(object_points if object_tensor is None else object_tensor)
    metric: dict[str, Any] = {
        "hand_points": int(n_finger),
        "valid": False,
        "raw_contact": False,
        "envelope_contact": False,
    }
    use_cuda = (
        device is not None
        and getattr(device, "type", None) == "cuda"
        and torch is not None
    )
    if n_finger < int(min_hand_points) or n_object == 0:
        return metric
    if not use_cuda and tree is None:
        return metric
    if use_cuda:
        distances_t, nearest_t = nearest_object_points(
            finger_points,
            object_points,
            tree,
            device=device,
            object_tensor=object_tensor,
            chunk_size=int(chunk_size),
            return_tensors=True,
        )
        keep_t = torch.isfinite(distances_t)
        distances_t = distances_t[keep_t]
        nearest_t = nearest_t[keep_t]
        if int(distances_t.numel()) == 0:
            return metric
        if _is_torch(finger_points):
            finger_w = finger_points.to(dtype=torch.float32)[keep_t]
            finger_z = finger_optical.to(dtype=torch.float32)[keep_t, 2]
        else:
            finger_w = torch.as_tensor(finger_points, dtype=torch.float32, device=device)[keep_t]
            finger_z = torch.as_tensor(finger_optical, dtype=torch.float32, device=device)[keep_t, 2]
        if object_tensor is None:
            object_w = torch.as_tensor(object_points, dtype=torch.float32, device=device)
            object_z = torch.as_tensor(object_optical, dtype=torch.float32, device=device)[:, 2]
        else:
            object_w = object_tensor.to(dtype=torch.float32)
            if _is_torch(object_optical):
                object_z = object_optical.to(dtype=torch.float32)[:, 2]
            else:
                object_z = torch.as_tensor(object_optical, dtype=torch.float32, device=device)[:, 2]
        pair = finger_w - object_w[nearest_t]
        pair_norm = torch.linalg.norm(pair, dim=1)
        normals_t = estimate_normals_torch(object_w, nearest_t, camera_world, k=16)
        pair_unit = pair / torch.clamp(pair_norm.unsqueeze(1), min=1e-9)
        dots_t = (pair_unit * normals_t).sum(dim=1)
        gaps_t = torch.abs(finger_z - object_z[nearest_t])
        distances = distances_t.detach().cpu().numpy().astype(np.float64, copy=False)
        nearest = nearest_t.detach().cpu().numpy().astype(np.int64, copy=False)
        dots = dots_t.detach().cpu().numpy().astype(np.float64, copy=False)
        gaps = gaps_t.detach().cpu().numpy().astype(np.float64, copy=False)
        normals = normals_t.detach().cpu().numpy().astype(np.float64, copy=False)
        object_points_np = object_w.detach().cpu().numpy().astype(np.float64, copy=False)
        object_z_np = object_z.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        distances, nearest = nearest_object_points(
            finger_points,
            object_points,
            tree,
            device=device,
            object_tensor=object_tensor,
            chunk_size=int(chunk_size),
        )
        keep = np.isfinite(distances)
        distances = distances[keep]
        nearest = nearest[keep]
        if len(distances) == 0:
            return metric
        pair = np.asarray(finger_points)[keep] - np.asarray(object_points)[nearest]
        pair_norm = np.linalg.norm(pair, axis=1)
        normals = estimate_normals(object_points, tree, nearest, camera_world, k=16)
        pair_unit = pair / np.maximum(pair_norm[:, None], 1e-9)
        dots = np.sum(pair_unit * normals, axis=1)
        gaps = np.abs(
            np.asarray(finger_optical)[keep, 2] - np.asarray(object_optical)[nearest, 2]
        )
        object_points_np = np.asarray(object_points)
        object_z_np = np.asarray(object_optical)[:, 2]
    order = np.argsort(distances)
    top = order[: min(12, len(order))]
    d_p10 = float(np.percentile(distances, 10))
    gap_p10 = float(np.percentile(gaps, 10))
    normal_p50 = (
        float(np.nanmedian(dots[top])) if np.any(np.isfinite(dots[top])) else float("nan")
    )
    normal_abs_p50 = (
        float(np.nanmedian(np.abs(dots[top])))
        if np.any(np.isfinite(dots[top]))
        else float("nan")
    )
    best = int(order[0])
    metric.update(
        {
            "valid": True,
            "distance_min_m": float(np.min(distances)),
            "distance_p10_m": d_p10,
            "distance_p50_m": float(np.percentile(distances, 50)),
            "depth_gap_p10_m": gap_p10,
            "normal_dot_p50": normal_p50,
            "normal_abs_dot_p50": normal_abs_p50,
            "nearest_pairs": int(len(distances)),
            "proximity": bool(d_p10 <= float(distance_threshold_m)),
            "proximity_envelope": bool(d_p10 <= float(proximity_envelope_m)),
            "depth_agreement": bool(gap_p10 <= float(depth_gap_threshold_m)),
            "normal_support": bool(
                np.isfinite(normal_abs_p50)
                and normal_abs_p50 >= float(normal_dot_threshold)
            ),
            "normal_outward_support": bool(
                np.isfinite(normal_p50) and normal_p50 >= -0.15
            ),
            "anchor_point": object_points_np[nearest[best]].tolist(),
            "anchor_normal": (
                normals[best].tolist()
                if np.all(np.isfinite(normals[best]))
                else [float("nan")] * 3
            ),
            "anchor_z_obs": float(object_z_np[nearest[best]]),
        }
    )
    metric["raw_contact"] = bool(
        metric["proximity"] and metric["depth_agreement"] and metric["normal_support"]
    )
    metric["envelope_contact"] = bool(
        metric["proximity_envelope"]
        and metric["depth_agreement"]
        and metric["normal_support"]
    )
    return metric


def _interval_payload(
    *,
    side: str,
    records: list[dict[str, Any]],
    aggregate_runs: list[list[int]],
    envelope_runs: list[list[int]],
    three_runs: list[list[int]],
    args: argparse.Namespace,
    object_source: str,
) -> dict[str, Any]:
    selected = [
        int(frame)
        for frame in (max(aggregate_runs, key=len) if aggregate_runs else [])
    ]
    all_frames = sorted({int(frame) for run in aggregate_runs for frame in run})
    aggregate_runs = [[int(frame) for frame in run] for run in aggregate_runs]
    envelope_selected = max(envelope_runs, key=len) if envelope_runs else []
    three_selected = max(three_runs, key=len) if three_runs else []
    return {
        "method": "pointcloud_distance_depthgap_normal_temporal",
        "side": side,
        "object_source": object_source,
        "first_contact": int(selected[0]) if selected else None,
        "last_contact": int(selected[-1]) if selected else None,
        "first_contact_time_s": (
            float(records[selected[0]]["time_s"]) if selected else None
        ),
        "last_contact_time_s": (
            float(records[selected[-1]]["time_s"]) if selected else None
        ),
        "proximity_envelope_first": (
            int(envelope_selected[0]) if envelope_selected else None
        ),
        "proximity_envelope_last": (
            int(envelope_selected[-1]) if envelope_selected else None
        ),
        "proximity_envelope_frame_count": len(envelope_selected),
        "frames": selected,
        "all_runs": aggregate_runs,
        "all_frames": all_frames,
        "frame_count": len(selected),
        "all_frame_count": len(all_frames),
        "three_finger_first": int(three_selected[0]) if three_selected else None,
        "three_finger_last": int(three_selected[-1]) if three_selected else None,
        "thresholds": {
            "distance_m": float(args.distance_threshold_m),
            "proximity_envelope_m": float(args.proximity_envelope_m),
            "depth_gap_m": float(args.depth_gap_threshold_m),
            "normal_dot": float(args.normal_dot_threshold),
            "temporal_gap_frames": int(args.temporal_gap),
            "min_run_frames": int(args.min_run),
            "primary_fingers": (
                list(FINGER_NAMES)
                if object_source == "fingertip_neighborhood"
                else list(PRIMARY_FINGERS)
            ),
            "min_fingers": int(args.min_fingers),
            "neighborhood_radius_m": float(args.neighborhood_radius_m),
            "neighborhood_hand_clearance_m": float(args.neighborhood_hand_clearance_m),
            "object_cloud_radius_m": float(args.object_cloud_radius_m),
            "object_cloud_voxel_m": float(args.object_cloud_voxel_m),
            "depth_footprint": bool(getattr(args, "depth_footprint", True)),
        },
    }


















def _sanitize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    """JSON cannot encode NaN; drop non-finite optional diagnostics."""
    cleaned: dict[str, Any] = {}
    for key, value in metric.items():
        if key.startswith("anchor_"):
            continue
        if isinstance(value, float) and not np.isfinite(value):
            continue
        cleaned[key] = value
    return cleaned


def main() -> None:
    pin_torch_threads()
    a = args()
    sides = parse_sides(a.side, a.sides)
    object_source = str(a.object_source)
    contact_groups = (
        FINGER_NAMES if object_source == "fingertip_neighborhood" else PRIMARY_FINGERS
    )
    if not 1 <= int(a.min_fingers) <= len(contact_groups):
        raise ValueError(
            f"--min-fingers must be in [1, {len(contact_groups)}]"
        )
    if int(a.temporal_gap) < 0 or int(a.min_run) < 1:
        raise ValueError("temporal-gap must be >= 0 and min-run must be >= 1")
    if float(a.neighborhood_radius_m) <= 0.0:
        raise ValueError("--neighborhood-radius-m must be positive")
    if float(a.neighborhood_mask_margin_px) < 0.0:
        raise ValueError("--neighborhood-mask-margin-px must be >= 0")
    if float(a.neighborhood_dilation_px) < 0.0:
        raise ValueError("--neighborhood-dilation-px must be >= 0")
    if float(a.neighborhood_z_band_m) <= 0.0:
        raise ValueError("--neighborhood-z-band-m must be positive")
    if float(a.object_cloud_radius_m) < 0.0:
        raise ValueError("--object-cloud-radius-m must be >= 0")
    if float(a.object_cloud_voxel_m) < 0.0:
        raise ValueError("--object-cloud-voxel-m must be >= 0")
    session = a.session_dir.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cal = session / "calibrations"
    rgb_k = np.asarray(np.load(cal / "rgb_K.npy"), dtype=np.float64)
    depth_k = np.asarray(np.load(cal / "depth_K.npy"), dtype=np.float64)
    r_optical_to_link = np.asarray(np.load(cal / "R_optical_to_link.npy"), dtype=np.float64)
    device = resolve_device(a.device)
    human_dir = a.human_mask_dir.expanduser().resolve() / "human_mask"
    with h5py.File(session / "annotation.hdf5", "r") as h5:
        depth_mm_all = np.asarray(h5["depth/frames"][:], dtype=np.float32)
        depth_t = np.asarray(h5["depth/timestamps"], dtype=np.float64)
        cam_t = np.asarray(h5["cam-pose/timestamps"], dtype=np.float64)
        cam_r = np.asarray(h5["cam-pose/rotations"], dtype=np.float64)
        cam_x = np.asarray(h5["cam-pose/translations"], dtype=np.float64)
        hand_t = np.asarray(h5["hand-pose/timestamps"], dtype=np.float64)
        n = len(hand_t) if int(a.max_frames) <= 0 else min(len(hand_t), int(a.max_frames))
        rgb_shape = (1080, 1920)
        keypoints = {
            side: np.asarray(h5[f"hand-pose/{side}_kpts_2d_rgb"], dtype=np.float64)
            for side in sides
        }
        joints = {
            side: np.asarray(h5[f"hand-pose/{side}_joints"], dtype=np.float64)
            for side in sides
        }
        hand_valid = {
            side: np.asarray(h5[f"hand-pose/{side}_valid"], dtype=bool)
            for side in sides
        }
        object_root = (
            a.object_mask_root.expanduser().resolve()
            if a.object_mask_root is not None
            else None
        )
        selected_mask_dir = (
            a.object_mask_dir.expanduser().resolve()
            if a.object_mask_dir is not None
            else None
        )
        if object_source == "mask":
            if object_root is None:
                raise ValueError("--object-mask-root is required for --object-source mask")
            if selected_mask_dir is not None:
                object_stack = load_mask_bool_volume(
                    selected_mask_dir, count=n, shape=rgb_shape
                )
                object_stack, object_sources = fill_mask_gaps_stack(
                    object_stack, max_gap=int(a.max_object_gap)
                )
            else:
                object_stack, object_sources = choose_object_masks(
                    object_root,
                    n,
                    rgb_shape,
                    int(a.max_object_gap),
                )
        else:
            object_stack = np.zeros((n, rgb_shape[0], rgb_shape[1]), dtype=bool)
            object_sources = ["fingertip_neighborhood"] * n
        human_stack = load_mask_bool_volume(human_dir, count=n, shape=rgb_shape)
        projector = None
        if device is not None and device.type == "cuda":
            projector = TorchDepthProjector(
                depth_k,
                rgb_k,
                (int(depth_mm_all.shape[1]), int(depth_mm_all.shape[2])),
                device,
            )
        print(
            f"[contact-pointcloud] device={device or 'cpu'} "
            f"object_source={object_source} sides={','.join(sides)} "
            f"projection_backend={'torch_cuda' if projector is not None else 'numpy_cpu'} "
            f"nearest_backend={'torch_cdist' if projector is not None else 'scipy_cKDTree'} "
            f"human_mask={'packed' if is_packed_stream(human_dir) else 'indexed'} "
            f"human_mask_mean={float(human_stack.mean()):.4f}",
            flush=True,
        )
        records_by_side: dict[str, list[dict[str, Any]]] = {side: [] for side in sides}
        raw_flags = {
            side: {name: [] for name in FINGER_INDICES} for side in sides
        }
        aggregate_raw = {side: [] for side in sides}
        three_raw = {side: [] for side in sides}
        envelope_aggregate_raw = {side: [] for side in sides}
        envelope_three_raw = {side: [] for side in sides}
        object_valid_frames: list[int] = []
        anchors = {
            side: {
                "point": np.full((n, len(FINGER_NAMES), 3), np.nan, dtype=np.float64),
                "normal": np.full((n, len(FINGER_NAMES), 3), np.nan, dtype=np.float64),
                "pixel": np.full((n, len(FINGER_NAMES), 2), np.nan, dtype=np.float64),
                "z_obs": np.full((n, len(FINGER_NAMES)), np.nan, dtype=np.float64),
                "valid": np.zeros((n, len(FINGER_NAMES)), dtype=bool),
                "raw_finger": np.zeros((n, len(FINGER_NAMES)), dtype=np.int8),
                "object_points": np.full(
                    (n, int(a.object_cloud_max_points), 3), np.nan, dtype=np.float64
                ),
                "object_normals": np.full(
                    (n, int(a.object_cloud_max_points), 3), np.nan, dtype=np.float64
                ),
                "object_count": np.zeros(n, dtype=np.int32),
            }
            for side in sides
        }
        for frame in range(n):
            depth_index = int(np.searchsorted(depth_t, hand_t[frame], side="left"))
            if depth_index <= 0:
                depth_index = 0
            elif depth_index >= len(depth_t):
                depth_index = len(depth_t) - 1
            elif hand_t[frame] - depth_t[depth_index - 1] <= depth_t[depth_index] - hand_t[frame]:
                depth_index -= 1
            if projector is not None:
                labeled = projector.label_frame_tensors(
                    depth_mm_all[depth_index],
                    human_stack[frame],
                    object_stack[frame],
                    rgb_shape,
                    depth_footprint=bool(a.depth_footprint),
                )
            else:
                labeled = label_registered_depth(
                    depth_mm_all[depth_index],
                    depth_k,
                    rgb_k,
                    human_stack[frame],
                    object_stack[frame],
                    rgb_shape,
                    depth_footprint=bool(a.depth_footprint),
                    projector=None,
                )
            optical = labeled["optical"]
            uv = labeled["uv"]
            pixel_object = labeled["pixel_object"]
            is_hand = labeled["is_hand"]
            is_mixed = labeled["is_mixed"]
            is_object_ok = labeled["is_object_ok"]
            boxes = labeled["boxes"]
            valid_depth_ratio = float(labeled["valid_depth_ratio"])
            hand_sel = is_hand & ~pixel_object
            mask_object_sel = pixel_object & is_object_ok
            camera_index = int(np.searchsorted(cam_t, hand_t[frame], side="left"))
            if camera_index <= 0:
                camera_index = 0
            elif camera_index >= len(cam_t):
                camera_index = len(cam_t) - 1
            elif hand_t[frame] - cam_t[camera_index - 1] <= cam_t[camera_index] - hand_t[frame]:
                camera_index -= 1
            rotation = cam_r[camera_index]
            translation = cam_x[camera_index]
            if projector is not None:
                world = to_world_torch(optical, rotation, translation, r_optical_to_link)
            else:
                world = to_world(optical, rotation, translation, r_optical_to_link)
            camera_world = YUP_TO_ZUP @ translation
            away_from_hand = None
            if (
                object_source == "fingertip_neighborhood"
                and float(a.neighborhood_hand_clearance_m) > 0.0
            ):
                away_from_hand = away_from_hand_cloud(
                    world,
                    is_hand,
                    is_object_ok,
                    float(a.neighborhood_hand_clearance_m),
                )
            mask_object_points = world[mask_object_sel]
            mask_object_optical = optical[mask_object_sel]
            mask_object_uv = uv[mask_object_sel]
            mask_valid = _len_points(mask_object_points) >= int(a.min_object_points)
            if object_source == "mask" and mask_valid:
                object_valid_frames.append(frame)
            mask_tree = (
                None
                if projector is not None
                else (cKDTree(_as_numpy(mask_object_points)) if mask_valid else None)
            )
            mask_tensor = (
                mask_object_points.to(dtype=torch.float32)
                if projector is not None and mask_valid
                else None
            )
            for side in sides:
                frame_row: dict[str, Any] = {
                    "frame": frame,
                    "side": side,
                    "time_s": float(hand_t[frame] - hand_t[0]),
                    "depth_frame": depth_index,
                    "depth_time_offset_ms": float(
                        1000.0 * (depth_t[depth_index] - hand_t[frame])
                    ),
                    "camera_frame": camera_index,
                    "camera_time_offset_ms": float(
                        1000.0 * (cam_t[camera_index] - hand_t[frame])
                    ),
                    "hand_valid": bool(hand_valid[side][frame]),
                    "human_mask_fraction": float(labeled["human_mask_fraction"]),
                    "object_mask_source": object_sources[frame],
                    "object_mask_fraction": float(labeled["object_mask_fraction"]),
                    "object_source": object_source,
                    "valid_depth_ratio": valid_depth_ratio,
                    "hand_points": _count_true(hand_sel),
                    "object_points": _len_points(mask_object_points),
                    "footprint_hand_points": _count_true(is_hand),
                    "footprint_mixed_points": _count_true(is_mixed),
                    "footprint_object_ok_points": _count_true(is_object_ok),
                    "depth_footprint": bool(a.depth_footprint),
                    "object_valid": bool(mask_valid) if object_source == "mask" else False,
                    "fingers": {},
                }
                finger_contacts: list[bool] = []
                if projector is not None:
                    side_object_sel = torch.zeros(
                        int(world.shape[0]), dtype=torch.bool, device=world.device
                    )
                else:
                    side_object_sel = np.zeros(len(world), dtype=bool)
                for finger_index, (name, inds) in enumerate(FINGER_INDICES.items()):
                    if object_source == "fingertip_neighborhood":
                        joint_optical = np.asarray(
                            joints[side][frame], dtype=np.float64
                        )[list(inds)]
                        finite = np.isfinite(joint_optical).all(axis=1)
                        mano_optical = joint_optical[finite]
                        if len(mano_optical) == 0:
                            metric = _empty_finger_metric()
                            object_sel = (
                                torch.zeros(
                                    int(world.shape[0]),
                                    dtype=torch.bool,
                                    device=world.device,
                                )
                                if projector is not None
                                else np.zeros(len(world), dtype=bool)
                            )
                        else:
                            mano_world = to_world(
                                mano_optical, rotation, translation, r_optical_to_link
                            )
                            center = mano_world[-1]
                            z_tip = float(mano_optical[-1, 2])
                            kpts = keypoints[side][frame]
                            tip_uv = np.asarray(kpts[inds[-1]], dtype=np.float64)
                            if projector is not None:
                                z_obs_q, z_kind = lookup_depth_at_uv_torch(
                                    tip_uv,
                                    boxes,
                                    optical[:, 2],
                                    is_hand,
                                    is_object_ok,
                                )
                                support = finger_support_torch(
                                    uv, kpts, inds, float(a.finger_width_px)
                                )
                            else:
                                z_obs_q, z_kind = lookup_depth_at_uv(
                                    tip_uv,
                                    boxes,
                                    optical[:, 2],
                                    is_hand,
                                    is_object_ok,
                                    is_mixed,
                                )
                                support = finger_support(
                                    uv.astype(np.float64),
                                    kpts,
                                    inds,
                                    float(a.finger_width_px),
                                    rgb_shape,
                                )
                            visibility = (
                                classify_fingertip_visibility(
                                    z_obs_q,
                                    z_tip,
                                    z_kind,
                                    float(a.depth_gap_threshold_m),
                                )
                                if bool(a.depth_footprint)
                                else VISIBILITY_UNKNOWN
                            )
                            hand_finger = is_hand & support
                            use_hand_depth = should_use_hand_depth(
                                visibility,
                                _count_true(hand_finger),
                                int(a.min_hand_points),
                            )
                            if use_hand_depth:
                                finger_points = world[hand_finger]
                                finger_optical = optical[hand_finger]
                                finger_source = "hand_depth"
                                if projector is not None:
                                    if (
                                        np.all(np.isfinite(tip_uv[:2]))
                                        and int(finger_points.shape[0]) > 0
                                    ):
                                        tip_t = torch.as_tensor(
                                            tip_uv[:2],
                                            dtype=torch.float32,
                                            device=uv.device,
                                        )
                                        delta_uv = uv[hand_finger].to(torch.float32) - tip_t
                                        center = finger_points[
                                            int(torch.argmin(torch.linalg.norm(delta_uv, dim=1)).item())
                                        ]
                                else:
                                    delta_uv = (
                                        uv[hand_finger].astype(np.float64)
                                        - tip_uv[None, :2]
                                    )
                                    if np.all(np.isfinite(tip_uv[:2])):
                                        center = finger_points[
                                            int(np.argmin(np.linalg.norm(delta_uv, axis=1)))
                                        ]
                            else:
                                finger_points = mano_world
                                finger_optical = mano_optical
                                finger_source = "mano_3d"
                            object_sel = neighborhood_object_sel(
                                world,
                                center,
                                float(a.neighborhood_radius_m),
                                is_object_ok,
                                extra=away_from_hand,
                            )
                            side_object_sel |= object_sel
                            object_points = world[object_sel]
                            object_optical = optical[object_sel]
                            object_uv = uv[object_sel]
                            object_valid = _len_points(object_points) >= int(a.min_object_points)
                            tree = (
                                None
                                if projector is not None
                                else (cKDTree(_as_numpy(object_points)) if object_valid else None)
                            )
                            object_tensor = (
                                object_points.to(dtype=torch.float32)
                                if projector is not None and object_valid
                                else None
                            )
                            metric = finger_contact_metric(
                                finger_points,
                                finger_optical,
                                object_points,
                                object_optical,
                                tree,
                                camera_world,
                                min_hand_points=int(a.min_hand_points),
                                distance_threshold_m=float(a.distance_threshold_m),
                                proximity_envelope_m=float(a.proximity_envelope_m),
                                depth_gap_threshold_m=float(a.depth_gap_threshold_m),
                                normal_dot_threshold=float(a.normal_dot_threshold),
                                device=device,
                                object_tensor=object_tensor,
                                chunk_size=int(a.nearest_chunk_size),
                            )
                            metric["object_points"] = _len_points(object_points)
                            metric["visibility"] = visibility
                            metric["finger_source"] = finger_source
                            metric["query_z_kind"] = z_kind
                            if np.isfinite(z_obs_q):
                                metric["query_z_obs_m"] = float(z_obs_q)
                            if np.isfinite(z_tip):
                                metric["query_z_tip_m"] = float(z_tip)
                            if object_valid:
                                object_valid_frames.append(frame)
                            if metric.get("valid") and "anchor_point" in metric:
                                nearest_world = np.asarray(
                                    metric["anchor_point"], dtype=np.float64
                                )
                                object_np = np.asarray(_as_numpy(object_points), dtype=np.float64)
                                delta = np.linalg.norm(
                                    object_np.reshape(-1, 3) - nearest_world[None, :], axis=1
                                )
                                pixel_index = int(np.argmin(delta)) if len(delta) else -1
                                if pixel_index >= 0:
                                    metric["anchor_pixel"] = np.asarray(
                                        _as_numpy(object_uv)
                                    )[pixel_index].tolist()
                    else:
                        if projector is not None:
                            support = finger_support_torch(
                                uv,
                                keypoints[side][frame],
                                inds,
                                float(a.finger_width_px),
                            )
                        else:
                            support = finger_support(
                                uv,
                                keypoints[side][frame],
                                inds,
                                float(a.finger_width_px),
                                rgb_shape,
                            )
                        selected = hand_sel & support
                        finger_points = world[selected]
                        finger_optical = optical[selected]
                        metric = finger_contact_metric(
                            finger_points,
                            finger_optical,
                            mask_object_points,
                            mask_object_optical,
                            mask_tree,
                            camera_world,
                            min_hand_points=int(a.min_hand_points),
                            distance_threshold_m=float(a.distance_threshold_m),
                            proximity_envelope_m=float(a.proximity_envelope_m),
                            depth_gap_threshold_m=float(a.depth_gap_threshold_m),
                            normal_dot_threshold=float(a.normal_dot_threshold),
                            device=device,
                            object_tensor=mask_tensor,
                            chunk_size=int(a.nearest_chunk_size),
                        )
                        if metric.get("valid") and "anchor_point" in metric:
                            nearest_world = np.asarray(
                                metric["anchor_point"], dtype=np.float64
                            )
                            mask_np = np.asarray(_as_numpy(mask_object_points), dtype=np.float64)
                            delta = np.linalg.norm(
                                mask_np.reshape(-1, 3) - nearest_world[None, :], axis=1
                            )
                            pixel_index = int(np.argmin(delta)) if len(delta) else -1
                            if pixel_index >= 0:
                                metric["anchor_pixel"] = np.asarray(
                                    _as_numpy(mask_object_uv)
                                )[pixel_index].tolist()
                    if metric.get("valid") and "anchor_point" in metric:
                        anchors[side]["point"][frame, finger_index] = np.asarray(
                            metric["anchor_point"], dtype=np.float64
                        )
                        anchors[side]["normal"][frame, finger_index] = np.asarray(
                            metric["anchor_normal"], dtype=np.float64
                        )
                        anchors[side]["z_obs"][frame, finger_index] = float(
                            metric["anchor_z_obs"]
                        )
                        if "anchor_pixel" in metric:
                            anchors[side]["pixel"][frame, finger_index] = np.asarray(
                                metric["anchor_pixel"], dtype=np.float64
                            )
                        anchors[side]["valid"][frame, finger_index] = True
                    raw_flags[side][name].append(bool(metric.get("raw_contact", False)))
                    finger_contacts.append(bool(metric.get("raw_contact", False)))
                    anchors[side]["raw_finger"][frame, finger_index] = np.int8(
                        metric.get("raw_contact", False)
                    )
                    frame_row["fingers"][name] = _sanitize_metric(metric)
                frame_row["raw_contact_fingers"] = [
                    name for name, ok in zip(FINGER_INDICES, finger_contacts) if ok
                ]
                envelope_contacts = [
                    bool(frame_row["fingers"][name].get("envelope_contact", False))
                    for name in FINGER_INDICES
                ]
                frame_row["envelope_contact_fingers"] = [
                    name for name, ok in zip(FINGER_INDICES, envelope_contacts) if ok
                ]
                frame_row["raw_contact_any"] = bool(any(finger_contacts))
                primary_contacts = [
                    bool(frame_row["fingers"][name].get("raw_contact", False))
                    for name in contact_groups
                ]
                primary_envelope_contacts = [
                    bool(frame_row["fingers"][name].get("envelope_contact", False))
                    for name in contact_groups
                ]
                frame_row["raw_contact_two_or_more"] = bool(
                    sum(primary_contacts) >= int(a.min_fingers)
                )
                frame_row["raw_contact_three_or_more"] = bool(
                    sum(primary_contacts) >= 3
                )
                frame_row["envelope_contact_two_or_more"] = bool(
                    sum(primary_envelope_contacts) >= int(a.min_fingers)
                )
                frame_row["envelope_contact_three_or_more"] = bool(
                    sum(primary_envelope_contacts) >= 3
                )
                aggregate_raw[side].append(
                    bool(sum(primary_contacts) >= int(a.min_fingers))
                )
                three_raw[side].append(bool(sum(primary_contacts) >= 3))
                envelope_aggregate_raw[side].append(
                    bool(sum(primary_envelope_contacts) >= int(a.min_fingers))
                )
                envelope_three_raw[side].append(
                    bool(sum(primary_envelope_contacts) >= 3)
                )
                if object_source == "fingertip_neighborhood":
                    cloud_sel_np = hand_neighborhood_object_sel(
                        uv=np.asarray(_as_numpy(uv), dtype=np.float64),
                        optical_z=np.asarray(_as_numpy(optical[:, 2]), dtype=np.float64),
                        is_object_ok=np.asarray(_as_numpy(is_object_ok), dtype=bool),
                        is_hand=np.asarray(_as_numpy(is_hand), dtype=bool),
                        keypoints=np.asarray(keypoints[side][frame], dtype=np.float64),
                        mano_optical=np.asarray(joints[side][frame], dtype=np.float64),
                        rgb_k=rgb_k,
                        human=np.asarray(human_stack[frame], dtype=bool),
                        finger_width_px=float(a.finger_width_px),
                        dilation_px=float(a.neighborhood_dilation_px),
                        z_band_m=float(a.neighborhood_z_band_m),
                        shape=rgb_shape,
                        points=np.asarray(_as_numpy(optical), dtype=np.float64),
                        mano_points=np.asarray(joints[side][frame], dtype=np.float64),
                        cloud_radius_m=float(a.object_cloud_radius_m),
                    )
                    if projector is not None:
                        cloud_sel = torch.as_tensor(
                            cloud_sel_np, dtype=torch.bool, device=world.device
                        )
                    else:
                        cloud_sel = cloud_sel_np
                else:
                    cloud_sel = mask_object_sel
                cloud_pts = world[cloud_sel]
                n_cloud = _len_points(cloud_pts)
                if n_cloud > 0:
                    packed_pts, _ = downsample_object_cloud(
                        np.asarray(_as_numpy(cloud_pts), dtype=np.float64).reshape(-1, 3),
                        np.zeros((n_cloud, 3), dtype=np.float64),
                        max_points=int(a.object_cloud_max_points),
                        seed=int(frame) + 17,
                        voxel_m=float(a.object_cloud_voxel_m),
                    )
                    packed_tree = cKDTree(packed_pts)
                    packed_nrm = estimate_normals(
                        packed_pts,
                        packed_tree,
                        np.arange(len(packed_pts)),
                        camera_world,
                        k=16,
                    )
                else:
                    packed_pts = np.zeros((0, 3), dtype=np.float64)
                    packed_nrm = np.zeros((0, 3), dtype=np.float64)
                count = int(len(packed_pts))
                anchors[side]["object_count"][frame] = count
                if count:
                    anchors[side]["object_points"][frame, :count] = packed_pts
                    anchors[side]["object_normals"][frame, :count] = packed_nrm
                frame_row["stored_object_points"] = count
                records_by_side[side].append(frame_row)

    metrics_path = out / "contact_frame_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as stream:
        if len(sides) == 1:
            for row in records_by_side[sides[0]]:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        else:
            for frame in range(n):
                payload = {
                    "frame": frame,
                    "time_s": records_by_side[sides[0]][frame]["time_s"],
                    "sides": {
                        side: records_by_side[side][frame] for side in sides
                    },
                }
                stream.write(json.dumps(payload, allow_nan=False) + "\n")

    per_side_runs: dict[str, dict[str, Any]] = {}
    per_side_interval: dict[str, Any] = {}
    contact_flags = {side: np.zeros(n, dtype=np.int8) for side in SIDES}
    finger_flags = {
        side: np.zeros((n, len(FINGER_NAMES)), dtype=np.int8) for side in SIDES
    }
    for side in sides:
        per_finger_runs = {
            name: classify(values, int(a.temporal_gap), int(a.min_run))
            for name, values in raw_flags[side].items()
        }
        aggregate_runs = classify(
            aggregate_raw[side], int(a.temporal_gap), int(a.min_run)
        )
        three_runs = classify(three_raw[side], int(a.temporal_gap), int(a.min_run))
        envelope_aggregate_runs = classify(
            envelope_aggregate_raw[side], int(a.temporal_gap), int(a.min_run)
        )
        envelope_three_runs = classify(
            envelope_three_raw[side], int(a.temporal_gap), int(a.min_run)
        )
        contact_flags[side] = flags_from_runs(n, aggregate_runs)
        for finger_index, name in enumerate(FINGER_NAMES):
            finger_flags[side][:, finger_index] = flags_from_runs(
                n, per_finger_runs[name]
            )
        per_side_runs[side] = {
            "per_finger": per_finger_runs,
            "two_or_more_fingers": aggregate_runs,
            "three_or_more_fingers": three_runs,
            "proximity_envelope_two_or_more_fingers": envelope_aggregate_runs,
            "proximity_envelope_three_or_more_fingers": envelope_three_runs,
        }
        per_side_interval[side] = _interval_payload(
            side=side,
            records=records_by_side[side],
            aggregate_runs=aggregate_runs,
            envelope_runs=envelope_aggregate_runs,
            three_runs=three_runs,
            args=a,
            object_source=object_source,
        )

    primary_side = sides[0]
    run_payload = per_side_runs[primary_side]
    interval = dict(per_side_interval[primary_side])
    interval["sides"] = list(sides)
    interval["per_side"] = per_side_interval
    union_frames = sorted(
        {
            int(frame)
            for side_payload in per_side_interval.values()
            for frame in side_payload.get("all_frames", [])
        }
    )
    interval["all_frames"] = union_frames
    interval["all_frame_count"] = len(union_frames)
    interval["all_runs"] = [
        run
        for side_payload in per_side_interval.values()
        for run in side_payload.get("all_runs", [])
    ]
    union_frames = sorted(
        {
            int(frame)
            for side_payload in per_side_interval.values()
            for frame in side_payload.get("all_frames", [])
        }
    )
    interval["all_frames"] = union_frames
    interval["all_frame_count"] = len(union_frames)
    interval["all_runs"] = [
        run
        for side_payload in per_side_interval.values()
        for run in side_payload.get("all_runs", [])
    ]
    (out / "contact_runs.json").write_text(
        json.dumps(
            {"sides": list(sides), "per_side": per_side_runs, **run_payload},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "contact_interval.json").write_text(
        json.dumps(interval, indent=2) + "\n", encoding="utf-8"
    )

    timestamps = np.asarray(hand_t[:n] - hand_t[0], dtype=np.float64)
    anchor_arrays: dict[str, Any] = {
        "sides": np.asarray(sides),
        "finger_names": np.asarray(FINGER_NAMES),
        "timestamps": timestamps,
        "object_source": np.asarray(object_source),
        "neighborhood_radius_m": np.asarray(
            float(a.neighborhood_radius_m), dtype=np.float64
        ),
        "neighborhood_dilation_px": np.asarray(
            float(a.neighborhood_dilation_px), dtype=np.float64
        ),
        "neighborhood_z_band_m": np.asarray(
            float(a.neighborhood_z_band_m), dtype=np.float64
        ),
        "object_cloud_radius_m": np.asarray(
            float(a.object_cloud_radius_m), dtype=np.float64
        ),
        "object_cloud_voxel_m": np.asarray(
            float(a.object_cloud_voxel_m), dtype=np.float64
        ),
        "schema_version": np.asarray("brainco_contact_anchors_v2"),
        "object_cloud_max_points": np.asarray(
            int(a.object_cloud_max_points), dtype=np.int32
        ),
    }
    cloud_max = int(a.object_cloud_max_points)
    for side in SIDES:
        if side in anchors:
            anchor_arrays[f"{side}_contact"] = contact_flags[side]
            anchor_arrays[f"{side}_finger_contact"] = finger_flags[side]
            anchor_arrays[f"{side}_anchor_point"] = anchors[side]["point"]
            anchor_arrays[f"{side}_anchor_normal"] = anchors[side]["normal"]
            anchor_arrays[f"{side}_anchor_pixel"] = anchors[side]["pixel"]
            anchor_arrays[f"{side}_anchor_z_obs"] = anchors[side]["z_obs"]
            anchor_arrays[f"{side}_anchor_valid"] = anchors[side]["valid"]
            anchor_arrays[f"{side}_object_points"] = anchors[side]["object_points"]
            anchor_arrays[f"{side}_object_normals"] = anchors[side]["object_normals"]
            anchor_arrays[f"{side}_object_count"] = anchors[side]["object_count"]
        else:
            anchor_arrays[f"{side}_contact"] = np.zeros(n, dtype=np.int8)
            anchor_arrays[f"{side}_finger_contact"] = np.zeros(
                (n, len(FINGER_NAMES)), dtype=np.int8
            )
            anchor_arrays[f"{side}_anchor_point"] = np.full(
                (n, len(FINGER_NAMES), 3), np.nan, dtype=np.float64
            )
            anchor_arrays[f"{side}_anchor_normal"] = np.full(
                (n, len(FINGER_NAMES), 3), np.nan, dtype=np.float64
            )
            anchor_arrays[f"{side}_anchor_pixel"] = np.full(
                (n, len(FINGER_NAMES), 2), np.nan, dtype=np.float64
            )
            anchor_arrays[f"{side}_anchor_z_obs"] = np.full(
                (n, len(FINGER_NAMES)), np.nan, dtype=np.float64
            )
            anchor_arrays[f"{side}_anchor_valid"] = np.zeros(
                (n, len(FINGER_NAMES)), dtype=bool
            )
            anchor_arrays[f"{side}_object_points"] = np.full(
                (n, cloud_max, 3), np.nan, dtype=np.float64
            )
            anchor_arrays[f"{side}_object_normals"] = np.full(
                (n, cloud_max, 3), np.nan, dtype=np.float64
            )
            anchor_arrays[f"{side}_object_count"] = np.zeros(n, dtype=np.int32)
    np.savez_compressed(out / "contact_anchors.npz", **anchor_arrays)

    summary = {
        "schema_version": 2,
        "session_dir": str(session),
        "object_source": object_source,
        "sides": list(sides),
        "object_mask_root": str(object_root) if object_root is not None else None,
        "object_mask_dir": str(selected_mask_dir) if selected_mask_dir is not None else None,
        "object_mask_manifest": (
            str(object_root / "sam3_prompt_manifest.json")
            if object_root is not None
            else None
        ),
        "human_mask_backend": "EfficientSAM3 + WiLoR keypoint component constraint",
        "device": str(device or "cpu"),
        "projection_backend": "torch_cuda" if projector is not None else "numpy_cpu",
        "nearest_backend": "torch_cdist" if projector is not None else "scipy_cKDTree",
        "contact_finger_policy": {
            "groups": list(contact_groups),
            "minimum": int(a.min_fingers),
            "keep_all_runs": True,
        },
        "human_mask_metadata": str(
            (a.human_mask_dir.expanduser().resolve() / "human_mask_metadata.json")
        ),
        "object_mask_policy": (
            "3-D fingertip balls for contact gates; stored O is object_ok "
            "unprojection in a "
            + (
                f"{float(a.object_cloud_radius_m)*100:.0f}cm MANO ball"
                if float(a.object_cloud_radius_m) > 0.0
                else (
                    f"2-D MANO/mask neighborhood (dilation "
                    f"{float(a.neighborhood_dilation_px):.0f}px)"
                )
            )
            + (
                f", voxel {float(a.object_cloud_voxel_m)*1000:.0f}mm, "
                f"z-band {float(a.neighborhood_z_band_m)*100:.0f}cm"
            )
            if object_source == "fingertip_neighborhood"
            else (
                "single selected prompt directory, <=4-frame mask hold, "
                "RGB-footprint mixed pixels excluded"
                if selected_mask_dir is not None
                else "preferred historical blue prompts plus arbitrary EfficientSAM3 prompt directories, <=4-frame mask hold"
            )
        ),
        "depth_footprint": bool(a.depth_footprint),
        "object_cloud_max_points": int(a.object_cloud_max_points),
        "object_cloud_radius_m": float(a.object_cloud_radius_m),
        "object_cloud_voxel_m": float(a.object_cloud_voxel_m),
        "num_frames": n,
        "object_valid_frames": len(set(object_valid_frames)),
        "pointcloud_frame_metrics": str(metrics_path),
        "contact_interval": str(out / "contact_interval.json"),
        "contact_runs": str(out / "contact_runs.json"),
        "contact_anchors": str(out / "contact_anchors.npz"),
        "per_side_contact_frames": {
            side: int(np.count_nonzero(contact_flags[side])) for side in sides
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"interval": interval, "runs": run_payload, "summary": summary}, indent=2
        )
    )


if __name__ == "__main__":
    main()
