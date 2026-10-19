#!/usr/bin/env python3
"""Select the longest RGB-D-supported window from EfficientSAM3 masks.

Registered depth projection runs through Torch CUDA when ``--device cuda``
is selected; ``--device cpu`` keeps the NumPy implementation for diagnostics.

The per-frame arithmetic is unchanged: ``torch.round`` (CUDA) or ``np.rint``
(CPU), the same depth validity gate, and the same blue-cup OpenCV HSV gate.
Frames are projected once, then every prompt mask gathers against that UV
grid so counts and the selected window match the old serial loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only environments keep the legacy path
    torch = None

try:
    from frame_volume import as_mask, create_volume, is_packed_stream, open_volume
except ImportError:  # pragma: no cover
    from .frame_volume import as_mask, create_volume, is_packed_stream, open_volume


def runs(values: list[int], max_gap: int) -> list[tuple[int, int]]:
    if not values:
        return []
    out: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value - previous - 1 > max_gap:
            out.append((start, previous))
            start = value
        previous = value
    out.append((start, previous))
    return out


def nearest_depth_indices(
    hand_t: np.ndarray, depth_t: np.ndarray, frame_count: int
) -> np.ndarray:
    """Same choice as ``int(np.argmin(np.abs(depth_t - hand_t[frame])))``."""
    queries = np.asarray(hand_t[:frame_count], dtype=np.float64).reshape(-1, 1)
    stamps = np.asarray(depth_t, dtype=np.float64).reshape(1, -1)
    return np.argmin(np.abs(stamps - queries), axis=1).astype(np.int64)


def load_depth_stack(depth_ds, indices: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
    stacked = np.stack(
        [np.asarray(depth_ds[int(index)], dtype=np.float64) for index in unique],
        axis=0,
    )
    return stacked[inverse]


def apply_blue_cup_filter(
    mask: np.ndarray, rgb: np.ndarray | None, bgr: np.ndarray | None
) -> np.ndarray:
    """Byte-identical to the previous per-frame OpenCV HSV gate."""
    hsv = None
    if rgb is not None:
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
    elif bgr is not None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if hsv is None:
        return mask
    hue, saturation, value = cv2.split(hsv)
    blue = (
        (hue >= 85) & (hue <= 130) & (saturation >= 35) & (value >= 20)
    )
    return np.where(blue & (mask > 0), 255, 0).astype(np.uint8)


def windows_from_counts(
    counts: np.ndarray, min_points: int, max_gap: int, min_frames: int
) -> list[tuple[float, int, int, int]]:
    valid = [i for i, count in enumerate(counts) if count >= min_points]
    selected: list[tuple[float, int, int, int]] = []
    for start, end in runs(valid, max_gap):
        length = end - start + 1
        if length >= min_frames:
            support = float(np.median(counts[start : end + 1]))
            selected.append((support, length, start, end))
    return selected


def point_count(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
    columns: np.ndarray,
    rows: np.ndarray,
) -> int:
    values = np.asarray(depth, dtype=np.float64).reshape(-1) / 1000.0
    valid = np.isfinite(values) & (values >= 0.10) & (values <= 3.0)
    z = values[valid]
    x = (columns[valid] - depth_k[0, 2]) * z / depth_k[0, 0]
    y = (rows[valid] - depth_k[1, 2]) * z / depth_k[1, 1]
    xyz = np.column_stack((x, y, z))
    projected = (rgb_k @ xyz.T).T
    uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int64)
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < mask.shape[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < mask.shape[0])
    )
    return int(np.count_nonzero(mask[uv[inside, 1], uv[inside, 0]] > 0))


def project_uv_batch(
    depths: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Registered depth→RGB UV for a depth stack. Independent of prompt masks."""
    depths = np.asarray(depths, dtype=np.float64)
    if depths.ndim != 3:
        raise ValueError(f"depths must be (T,H,W), got {depths.shape}")
    frame_count, height, width = depths.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    rows = rows.reshape(-1)
    columns = columns.reshape(-1)
    values = depths.reshape(frame_count, -1) / 1000.0
    valid = np.isfinite(values) & (values >= 0.10) & (values <= 3.0)
    z = np.where(valid, values, 1.0)
    x = (columns - depth_k[0, 2]) * z / depth_k[0, 0]
    y = (rows - depth_k[1, 2]) * z / depth_k[1, 1]
    xyz = np.stack((x, y, z), axis=-1)
    projected = xyz @ np.asarray(rgb_k, dtype=np.float64).T
    uv = np.rint(projected[..., :2] / projected[..., 2:3]).astype(np.int64)
    return uv, valid


def gather_counts_from_uv(
    masks: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Count in-mask projected points. ``uv`` / ``valid`` come from :func:`project_uv_batch`."""
    masks_u8 = np.asarray(masks)
    if masks_u8.ndim != 3 or masks_u8.shape[0] != uv.shape[0]:
        raise ValueError(
            f"masks must be (T,H,W) aligned with uv, got {masks_u8.shape} vs {uv.shape}"
        )
    mask_h, mask_w = int(masks_u8.shape[1]), int(masks_u8.shape[2])
    inside = (
        (uv[..., 0] >= 0)
        & (uv[..., 0] < mask_w)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < mask_h)
        & valid
    )
    uv_x = np.clip(uv[..., 0], 0, max(mask_w - 1, 0))
    uv_y = np.clip(uv[..., 1], 0, max(mask_h - 1, 0))
    time_index = np.arange(masks_u8.shape[0], dtype=np.int64)[:, None]
    picked = (masks_u8 > 0)[time_index, uv_y, uv_x]
    return np.count_nonzero(picked & inside, axis=1).astype(np.int64)


def point_count_batch(
    masks: np.ndarray,
    depths: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
) -> np.ndarray:
    """Vectorized CPU counts; valid-pixel UV matches :func:`point_count`."""
    uv, valid = project_uv_batch(depths, depth_k, rgb_k)
    return gather_counts_from_uv(masks, uv, valid)


def resolve_device(requested: str):
    """Resolve ``auto`` and fail clearly for an explicitly unavailable CUDA device."""
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


class TorchPointCounter:
    """GPU implementation of the registered RGB-D support count.

    The arithmetic mirrors :func:`point_count` and keeps the calibrated pixel
    grid resident on the selected device.  ``count`` is the original per-frame
    kernel; ``count_batch`` projects a depth stack once and gathers every mask.
    """

    def __init__(
        self,
        depth_k: np.ndarray,
        rgb_k: np.ndarray,
        depth_shape: tuple[int, int],
        device,
    ) -> None:
        if torch is None or device is None or device.type == "cpu":
            raise ValueError("TorchPointCounter requires a Torch CUDA device")
        height, width = depth_shape
        rows, columns = np.indices((height, width), dtype=np.float64)
        self.rows = torch.as_tensor(rows.reshape(-1), dtype=torch.float64, device=device)
        self.columns = torch.as_tensor(columns.reshape(-1), dtype=torch.float64, device=device)
        self.depth_k = torch.as_tensor(np.asarray(depth_k), dtype=torch.float64, device=device)
        self.rgb_k = torch.as_tensor(np.asarray(rgb_k), dtype=torch.float64, device=device)
        self.device = device
        self.pixel_count = int(height * width)

    def count(self, mask: np.ndarray, depth: np.ndarray) -> int:
        values = torch.as_tensor(np.asarray(depth), dtype=torch.float64, device=self.device).reshape(-1)
        values = values / 1000.0
        valid = torch.isfinite(values) & (values >= 0.10) & (values <= 3.0)
        z = values[valid]
        x = (self.columns[valid] - self.depth_k[0, 2]) * z / self.depth_k[0, 0]
        y = (self.rows[valid] - self.depth_k[1, 2]) * z / self.depth_k[1, 1]
        xyz = torch.stack((x, y, z), dim=1)
        projected = xyz @ self.rgb_k.T
        uv = torch.round(projected[:, :2] / projected[:, 2:3]).to(torch.int64)
        height, width = mask.shape[:2]
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        if not bool(torch.any(inside)):
            return 0
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
        return int(mask_t[uv[inside, 1], uv[inside, 0]].sum().item())

    def _chunk_size(self, frame_count: int) -> int:
        # Keep UV buffers around 64 MiB of int64 pairs per chunk.
        max_uv = 8_000_000
        return max(1, min(int(frame_count), max_uv // max(self.pixel_count, 1)))

    def project_uv(self, depths: np.ndarray):
        """Project a depth stack once with ``torch.round``. Keeps ``(uv, valid)`` on GPU."""
        depths = np.asarray(depths, dtype=np.float64)
        if depths.ndim != 3:
            raise ValueError(f"depths must be (T,H,W), got {depths.shape}")
        frame_count = int(depths.shape[0])
        if frame_count == 0:
            empty_uv = torch.empty((0, self.pixel_count, 2), dtype=torch.int64, device=self.device)
            empty_valid = torch.empty((0, self.pixel_count), dtype=torch.bool, device=self.device)
            return empty_uv, empty_valid
        chunk = self._chunk_size(frame_count)
        uv_parts = []
        valid_parts = []
        dummy_z = torch.ones((), dtype=torch.float64, device=self.device)
        for start in range(0, frame_count, chunk):
            end = min(start + chunk, frame_count)
            batch = end - start
            values = torch.as_tensor(
                depths[start:end], dtype=torch.float64, device=self.device
            ).reshape(batch, -1)
            values = values / 1000.0
            valid = torch.isfinite(values) & (values >= 0.10) & (values <= 3.0)
            z = torch.where(valid, values, dummy_z)
            x = (self.columns - self.depth_k[0, 2]) * z / self.depth_k[0, 0]
            y = (self.rows - self.depth_k[1, 2]) * z / self.depth_k[1, 1]
            xyz = torch.stack((x, y, z), dim=-1)
            projected = xyz @ self.rgb_k.T
            uv = torch.round(projected[..., :2] / projected[..., 2:3]).to(torch.int64)
            uv_parts.append(uv)
            valid_parts.append(valid)
        return torch.cat(uv_parts, dim=0), torch.cat(valid_parts, dim=0)

    def gather_counts(self, masks: np.ndarray, uv, valid) -> np.ndarray:
        """Gather prompt masks against a shared UV grid. Chunked to bound GPU RAM."""
        masks_u8 = np.asarray(masks)
        if torch.is_tensor(uv):
            frame_count = int(uv.shape[0])
        else:
            uv = np.asarray(uv)
            valid = np.asarray(valid)
            frame_count = int(uv.shape[0])
        if masks_u8.ndim != 3 or masks_u8.shape[0] != frame_count:
            raise ValueError(
                f"masks must be (T,H,W) aligned with uv, got {masks_u8.shape} vs {frame_count}"
            )
        if frame_count == 0:
            return np.zeros(0, dtype=np.int64)
        mask_h, mask_w = int(masks_u8.shape[1]), int(masks_u8.shape[2])
        chunk = self._chunk_size(frame_count)
        counts = np.empty(frame_count, dtype=np.int64)
        for start in range(0, frame_count, chunk):
            end = min(start + chunk, frame_count)
            batch = end - start
            if torch.is_tensor(uv):
                uv_chunk = uv[start:end]
                valid_chunk = valid[start:end]
            else:
                uv_chunk = torch.as_tensor(uv[start:end], dtype=torch.int64, device=self.device)
                valid_chunk = torch.as_tensor(valid[start:end], dtype=torch.bool, device=self.device)
            inside = (
                (uv_chunk[..., 0] >= 0)
                & (uv_chunk[..., 0] < mask_w)
                & (uv_chunk[..., 1] >= 0)
                & (uv_chunk[..., 1] < mask_h)
                & valid_chunk
            )
            uv_x = uv_chunk[..., 0].clamp(0, max(mask_w - 1, 0))
            uv_y = uv_chunk[..., 1].clamp(0, max(mask_h - 1, 0))
            # Same bool cast as :meth:`count` (any nonzero mask byte is True).
            mask_t = torch.as_tensor(
                np.asarray(masks_u8[start:end]), dtype=torch.bool, device=self.device
            )
            time_index = torch.arange(batch, device=self.device)[:, None].expand_as(uv_x)
            picked = mask_t[time_index, uv_y, uv_x]
            counts[start:end] = (picked & inside).sum(dim=1).to(torch.int64).cpu().numpy()
        return counts

    def count_batch(self, masks: np.ndarray, depths: np.ndarray) -> np.ndarray:
        uv, valid = self.project_uv(depths)
        return self.gather_counts(masks, uv, valid)


def _load_mask_stack(mask_dir: Path, frame_count: int) -> np.ndarray:
    """Return uint8 0/255 masks shaped (T, H, W). Missing frames stay 0."""
    if is_packed_stream(mask_dir):
        volume = open_volume(mask_dir)
        usable = min(frame_count, len(volume))
        packed_bool = np.asarray(
            as_mask(np.ascontiguousarray(volume.array[:usable])),
            dtype=bool,
        )
        if packed_bool.ndim != 3:
            raise ValueError(f"packed masks must be (T,H,W), got {packed_bool.shape}")
        stack = np.zeros(
            (frame_count, packed_bool.shape[1], packed_bool.shape[2]), dtype=np.uint8
        )
        stack[: packed_bool.shape[0]] = packed_bool.astype(np.uint8) * 255
        return stack
    first = None
    for frame in range(frame_count):
        path = mask_dir / f"{frame:06d}.png"
        if path.is_file():
            first = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if first is not None:
                break
    if first is None:
        return np.zeros((frame_count, 1, 1), dtype=np.uint8)
    stack = np.zeros((frame_count, first.shape[0], first.shape[1]), dtype=np.uint8)
    for frame in range(frame_count):
        path = mask_dir / f"{frame:06d}.png"
        if not path.is_file():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        stack[frame] = mask
    return stack


def _filter_blue_stack(
    masks: np.ndarray,
    *,
    packed_rgb,
    video,
) -> np.ndarray:
    filtered = masks.copy()
    for frame in range(len(filtered)):
        rgb = None
        bgr = None
        if packed_rgb is not None and frame < len(packed_rgb):
            rgb = np.asarray(packed_rgb[frame])
        elif video is not None:
            ok, bgr = video.read()
            if not ok:
                bgr = None
        filtered[frame] = apply_blue_cup_filter(filtered[frame], rgb, bgr)
    return filtered


def _write_filtered_volume(filtered_dir: Path, masks: np.ndarray) -> None:
    filtered_dir.mkdir(parents=True, exist_ok=True)
    volume = create_volume(
        filtered_dir,
        len(masks),
        masks.shape[1:],
        np.uint8,
        fill=0,
    )
    for frame, mask in enumerate(masks):
        if frame < len(volume):
            volume[frame] = mask
    volume.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--max-gap", type=int, default=6)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="projection device: auto (CUDA when available), cuda[:N], or cpu",
    )
    parser.add_argument(
        "--source-rgb-dir",
        type=Path,
        default=None,
        help="Optional packed OpenCV RGB volume for blue-cup HSV filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is not None:
        torch.set_num_threads(1)
    session = args.session_dir.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    rgb_k = np.asarray(np.load(session / "calibrations/rgb_K.npy"), dtype=np.float64)
    depth_k = np.asarray(np.load(session / "calibrations/depth_K.npy"), dtype=np.float64)
    device = resolve_device(args.device)
    gpu_counter = None
    with h5py.File(session / "annotation.hdf5", "r") as h5:
        depth = h5["depth/frames"]
        depth_t = np.asarray(h5["depth/timestamps"], dtype=np.float64)
        hand_t = np.asarray(h5["hand-pose/timestamps"], dtype=np.float64)
        if device is not None and device.type == "cuda":
            gpu_counter = TorchPointCounter(
                depth_k,
                rgb_k,
                (int(depth.shape[1]), int(depth.shape[2])),
                device,
            )
        print(
            f"[contact-mask-window] device={device or 'cpu'} "
            f"backend={'torch' if gpu_counter is not None else 'numpy'}",
            flush=True,
        )
        candidates = []
        frame_count = (
            len(hand_t)
            if int(args.max_frames) <= 0
            else min(len(hand_t), int(args.max_frames))
        )
        depth_indices = nearest_depth_indices(hand_t, depth_t, frame_count)
        depth_stack = load_depth_stack(depth, depth_indices)
        prompt_dirs = sorted(
            path
            for path in mask_root.glob("*/masks")
            if path.parent.name != "_filtered"
        )
        if gpu_counter is not None:
            projected_uv, projected_valid = gpu_counter.project_uv(depth_stack)
        else:
            projected_uv, projected_valid = project_uv_batch(depth_stack, depth_k, rgb_k)
        print(
            f"[contact-mask-window] projected {frame_count} frames "
            f"prompts={len(prompt_dirs)}",
            flush=True,
        )
        for mask_dir in prompt_dirs:
            prompt_name = mask_dir.parent.name.lower()
            blue_filter = "blue" in prompt_name
            filtered_dir = mask_root / "_filtered" / prompt_name / "masks"
            packed_rgb = None
            video = None
            source_rgb = args.source_rgb_dir
            if blue_filter and source_rgb is not None and is_packed_stream(source_rgb):
                packed_rgb = open_volume(source_rgb, color_space="rgb")
            elif blue_filter:
                video = cv2.VideoCapture(str(session / "rgb.mp4"))
            masks = _load_mask_stack(mask_dir, frame_count)
            if blue_filter:
                masks = _filter_blue_stack(masks, packed_rgb=packed_rgb, video=video)
                _write_filtered_volume(filtered_dir, masks)
            if gpu_counter is not None:
                counts = gpu_counter.gather_counts(masks, projected_uv, projected_valid)
            else:
                counts = gather_counts_from_uv(masks, projected_uv, projected_valid)
            chosen_dir = filtered_dir if blue_filter else mask_dir
            for support, length, start, end in windows_from_counts(
                counts,
                int(args.min_points),
                int(args.max_gap),
                int(args.min_frames),
            ):
                candidates.append((support, length, chosen_dir, start, end, counts))
            if video is not None:
                video.release()
        if not candidates:
            raise RuntimeError("no mask prompt has a stable RGB-D-supported window")
        candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
        support, length, mask_dir, start, end, counts = candidates[0]
        keyframe = start + int(np.argmax(np.asarray(counts[start : end + 1])))
    payload = {
        "schema_version": 1,
        "mask_dir": str(mask_dir),
        "window_start": int(start),
        "window_end": int(end),
        "keyframe": int(keyframe),
        "window_frames": int(length),
        "median_rgbd_mask_points": float(support),
        "min_points": int(args.min_points),
        "max_gap": int(args.max_gap),
        "device": str(device or "cpu"),
        "backend": "torch_cuda" if gpu_counter is not None else "numpy_cpu",
        "all_candidates": [
            {"mask_dir": str(item[2]), "start": int(item[3]), "end": int(item[4]),
             "length": int(item[1]), "median_points": float(item[0])}
            for item in candidates
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
