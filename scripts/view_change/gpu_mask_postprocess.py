#!/usr/bin/env python3
"""Fast mask postprocess matching CPU OpenCV ellipse semantics.

``morph``: elliptical close then dilate (``cv2.MORPH_ELLIPSE``, bit-exact).
``temporal_union``: max over ``±radius`` frames.

Default path (best wall-clock + quality):
  - morph on CPU with a thread pool (OpenCV is faster than GPU ellipse conv)
  - temporal_union on GPU in chunks (peak VRAM ~O(chunk), not O(T))

Pure-GPU ellipse dilate helpers remain for optional use / tests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=64)
def _ellipse_se(radius_px: int) -> torch.Tensor:
    k = 2 * radius_px + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return torch.from_numpy(se.astype(np.bool_))


def morph_cpu(mask: np.ndarray, close_px: int, dilation_px: int) -> np.ndarray:
    out = mask
    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if dilation_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        out = cv2.dilate(out, k)
    return out


def dilate_ellipse(masks01: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Binary dilate with OpenCV-matched elliptical SE. masks01: NCHW float {0,1}."""
    if radius_px <= 0:
        return masks01
    se = _ellipse_se(radius_px).to(device=masks01.device, dtype=masks01.dtype)
    k = 2 * radius_px + 1
    y = F.conv2d(masks01, se.view(1, 1, k, k), padding=radius_px)
    return (y > 0).to(dtype=masks01.dtype)


def erode_ellipse(masks01: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px <= 0:
        return masks01
    return 1.0 - dilate_ellipse(1.0 - masks01, radius_px)


def _to_nchw01(masks: torch.Tensor) -> tuple[torch.Tensor, bool, int]:
    orig_ndim = masks.ndim
    if masks.ndim == 2:
        masks = masks.unsqueeze(0).unsqueeze(0)
    elif masks.ndim == 3:
        masks = masks.unsqueeze(1)
    elif masks.ndim != 4:
        raise ValueError(f"expected HxW / NxHxW / Nx1xHxW, got {tuple(masks.shape)}")
    scale255 = bool(masks.dtype == torch.uint8 or float(masks.detach().amax()) > 1.5)
    x = (masks.float() > (127.0 if scale255 else 0.5)).float()
    return x, scale255, orig_ndim


def _from_nchw(x01: torch.Tensor, scale255: bool, orig_ndim: int) -> torch.Tensor:
    b = x01 > 0.5
    out = (b.to(torch.uint8) * 255) if scale255 else b.float()
    out = out.squeeze(1)
    if orig_ndim == 2:
        out = out[0]
    return out


def morph_close_dilate(
    masks: torch.Tensor,
    close_px: int = 5,
    dilation_px: int = 12,
) -> torch.Tensor:
    """Pure-GPU elliptical morph (exact, slower than threaded OpenCV on 1080p)."""
    x, scale255, orig_ndim = _to_nchw01(masks)
    if close_px > 0:
        x = dilate_ellipse(x, close_px)
        x = erode_ellipse(x, close_px)
    if dilation_px > 0:
        x = dilate_ellipse(x, dilation_px)
    return _from_nchw(x, scale255, orig_ndim)


def temporal_union(masks: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return masks
    if masks.ndim == 4:
        assert masks.shape[1] == 1
        x = masks[:, 0]
        squeeze_c = True
    elif masks.ndim == 3:
        x = masks
        squeeze_c = False
    else:
        raise ValueError(f"expected TxHxW or Tx1xHxW, got {tuple(masks.shape)}")
    scale255 = bool(x.dtype == torch.uint8 or float(x.detach().amax()) > 1.5)
    x01 = (x > (127 if scale255 else 0.5)).to(torch.uint8)
    out = x01.clone()
    for dt in range(1, int(radius) + 1):
        out[dt:] = torch.maximum(out[dt:], x01[:-dt])
        out[:-dt] = torch.maximum(out[:-dt], x01[dt:])
    out_t = out * 255 if scale255 else out.float()
    if squeeze_c:
        out_t = out_t.unsqueeze(1)
    return out_t


def temporal_union_chunked_numpy(
    arr: np.ndarray,
    radius: int,
    *,
    device: str | torch.device = "cuda",
    chunk: int = 512,
) -> np.ndarray:
    """GPU temporal_union without holding the full TxHxW clip in VRAM.

    Overlapping windows of ``chunk`` (±radius) → peak ~O(chunk), bit-identical
    to a single full-clip ``temporal_union``.
    """
    if radius <= 0:
        return arr
    t = int(arr.shape[0])
    r = int(radius)
    chunk = max(int(chunk), 2 * r + 1)
    device = torch.device(device)
    result = np.empty_like(arr)
    for i0 in range(0, t, chunk):
        i1 = min(t, i0 + chunk)
        a = max(0, i0 - r)
        b = min(t, i1 + r)
        window = torch.from_numpy(np.ascontiguousarray(arr[a:b])).to(
            device, non_blocking=True
        )
        window = temporal_union(window, r)
        result[i0:i1] = window[i0 - a : i1 - a].cpu().numpy()
        del window
    return result


def _as_list(masks) -> list[np.ndarray]:
    if isinstance(masks, list):
        return [np.asarray(m) for m in masks]
    arr = np.asarray(masks)
    if arr.ndim != 3:
        raise ValueError(f"expected TxHxW, got {arr.shape}")
    return [arr[i] for i in range(arr.shape[0])]


@torch.inference_mode()
def postprocess_masks_gpu(
    masks,
    *,
    close_px: int = 5,
    dilation_px: int = 12,
    temporal_radius: int = 3,
    device: str | torch.device = "cuda",
    morph_workers: int = 16,
    morph_mode: str = "cpu_threads",
    morph_batch: int = 16,
    temporal_chunk: int = 512,
) -> list[np.ndarray]:
    """Morph + temporal_union.

    ``morph_mode``:
      - ``cpu_threads`` (default): OpenCV ellipse morph via thread pool +
        chunked GPU temporal. Bit-exact vs serial CPU.
      - ``gpu``: pure GPU elliptical morph + chunked temporal.
    """
    frames = _as_list(masks)
    if not frames:
        return []

    if morph_mode == "cpu_threads":
        n_workers = max(1, min(int(morph_workers), len(frames)))

        def _one(m: np.ndarray) -> np.ndarray:
            return morph_cpu(m, close_px, dilation_px)

        if n_workers == 1:
            morphed = [_one(m) for m in frames]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                morphed = list(ex.map(_one, frames))
        if temporal_radius <= 0:
            return morphed
        arr = np.stack(morphed, axis=0)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            out = temporal_union_chunked_numpy(
                arr,
                int(temporal_radius),
                device=device,
                chunk=int(temporal_chunk),
            )
            return [out[i] for i in range(out.shape[0])]
        r = int(temporal_radius)
        bin_arr = arr > 127
        out_b = bin_arr.copy()
        for dt in range(1, r + 1):
            out_b[dt:] |= bin_arr[:-dt]
            out_b[:-dt] |= bin_arr[dt:]
        return [(out_b[i].astype(np.uint8) * 255) for i in range(out_b.shape[0])]

    if morph_mode != "gpu":
        raise ValueError(f"unknown morph_mode={morph_mode!r}")

    device = torch.device(device)
    arr = np.stack(frames, axis=0)
    t, h, w = arr.shape
    morphed_cpu = np.empty((t, h, w), dtype=np.uint8)
    for i0 in range(0, t, morph_batch):
        i1 = min(t, i0 + morph_batch)
        batch = torch.from_numpy(np.ascontiguousarray(arr[i0:i1])).to(
            device, non_blocking=True
        )
        morphed_cpu[i0:i1] = (
            morph_close_dilate(batch, close_px=close_px, dilation_px=dilation_px)
            .cpu()
            .numpy()
        )
    if temporal_radius > 0:
        morphed_cpu = temporal_union_chunked_numpy(
            morphed_cpu,
            int(temporal_radius),
            device=device,
            chunk=int(temporal_chunk),
        )
    return [morphed_cpu[i] for i in range(t)]
