#!/usr/bin/env python3
"""Uncompressed (T,H,W[,C]) frame volumes with a legacy %06d fallback.

Packed layout (formal path)::

    <stream_dir>/frames.npy
    <stream_dir>/volume_meta.json

``frames.npy`` is an uncompresssed memmap. RGB is stored as uint8 HxWx3 in
**RGB** channel order, matching ``cv2.cvtColor(..., COLOR_BGR2RGB)`` of a
PNG that the previous pipeline wrote with OpenCV. Masks are uint8 0/255.
Metric depth and confidence stay float32. Robot IDs stay uint16.

A directory of ``%06d.png`` / ``%06d.npy`` is still readable so tests and
older ``REUSE_BASE_ROOT`` trees keep working.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is a pipeline dependency
    cv2 = None  # type: ignore[assignment]


PACKED_FILENAME = "frames.npy"
META_FILENAME = "volume_meta.json"
INDEXED_SUFFIXES = (".npy", ".png")


def packed_path(stream_dir: str | Path) -> Path:
    return Path(stream_dir) / PACKED_FILENAME


def meta_path(stream_dir: str | Path) -> Path:
    return Path(stream_dir) / META_FILENAME


def is_packed_stream(stream_dir: str | Path) -> bool:
    path = Path(stream_dir)
    if path.is_file() and path.suffix.lower() == ".npy":
        return True
    return packed_path(path).is_file()


def _dtype_name(dtype: np.dtype | str) -> str:
    return np.dtype(dtype).str


def _write_meta(
    directory: Path,
    *,
    count: int,
    frame_shape: tuple[int, ...],
    dtype: np.dtype,
    color_space: str | None,
) -> None:
    payload = {
        "schema_version": 1,
        "layout": "packed_memmap",
        "filename": PACKED_FILENAME,
        "count": int(count),
        "dtype": _dtype_name(dtype),
        "frame_shape": [int(v) for v in frame_shape],
        "color_space": color_space,
        "compression": "none",
    }
    meta_path(directory).write_text(json.dumps(payload, indent=2) + "\n")


def _read_meta(directory: Path) -> dict[str, Any] | None:
    path = meta_path(directory)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _scan_indexed(directory: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in INDEXED_SUFFIXES:
            continue
        if path.name in {PACKED_FILENAME, META_FILENAME}:
            continue
        try:
            index = int(path.stem)
        except ValueError:
            continue
        previous = found.get(index)
        if previous is not None and previous.suffix.lower() == ".npy":
            continue
        if previous is not None and path.suffix.lower() != ".npy":
            continue
        found[index] = path
    return found


class FrameVolume:
    """Random-access (T, ...) frames, packed memmap or per-file fallback."""

    def __init__(
        self,
        *,
        directory: Path,
        mmap: np.memmap | np.ndarray | None,
        indexed: dict[int, Path] | None,
        dtype: np.dtype,
        frame_shape: tuple[int, ...],
        color_space: str | None,
        writable: bool,
    ) -> None:
        self.directory = Path(directory)
        self._mmap = mmap
        self._indexed = indexed
        self.dtype = np.dtype(dtype)
        self.frame_shape = tuple(int(v) for v in frame_shape)
        self.color_space = color_space
        self.writable = bool(writable)
        self._index_origin = 0
        if mmap is not None:
            self._count = int(mmap.shape[0])
        else:
            assert indexed is not None
            self._count = (max(indexed) + 1) if indexed else 0
            self._index_origin = min(indexed) if indexed else 0
            self._index_origin = min(indexed) if indexed else 0

    @property
    def packed(self) -> bool:
        return self._mmap is not None

    @property
    def array(self) -> np.ndarray:
        if self._mmap is None:
            raise TypeError(f"{self.directory} is a legacy indexed stream; no packed array")
        return np.asarray(self._mmap)

    def __len__(self) -> int:
        return self._count

    def __enter__(self) -> "FrameVolume":
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()

    def flush(self) -> None:
        if self._mmap is not None and hasattr(self._mmap, "flush"):
            self._mmap.flush()

    def frame_at(self, global_index: int, local_index: int | None = None) -> np.ndarray:
        """Resolve packed local frames or legacy global/local stems."""
        if self._mmap is not None:
            if local_index is not None and 0 <= int(local_index) < self._count:
                return np.asarray(self._mmap[int(local_index)])
            return np.asarray(self._mmap[int(global_index)])
        assert self._indexed is not None
        if int(global_index) in self._indexed:
            return self._legacy_read(int(global_index))
        if local_index is not None and int(local_index) in self._indexed:
            return self._legacy_read(int(local_index))
        raise FileNotFoundError(
            f"{self.directory} has no frame global={global_index} local={local_index}"
        )

    def frame_at(self, global_index: int, local_index: int | None = None) -> np.ndarray:
        """Resolve packed local frames or legacy global/local stems."""
        if self._mmap is not None:
            if local_index is not None and 0 <= int(local_index) < self._count:
                return np.asarray(self._mmap[int(local_index)])
            return np.asarray(self._mmap[int(global_index)])
        assert self._indexed is not None
        if int(global_index) in self._indexed:
            return self._legacy_read(int(global_index))
        if local_index is not None and int(local_index) in self._indexed:
            return self._legacy_read(int(local_index))
        raise FileNotFoundError(
            f"{self.directory} has no frame global={global_index} local={local_index}"
        )

    def _legacy_read(self, index: int) -> np.ndarray:
        assert self._indexed is not None
        path = self._indexed.get(int(index))
        if path is None:
            raise IndexError(f"{self.directory} has no frame {index}")
        if path.suffix.lower() == ".npy":
            frame = np.asarray(np.load(path, allow_pickle=False))
        else:
            if cv2 is None:
                raise RuntimeError("OpenCV is required to read legacy PNG volumes")
            flags = cv2.IMREAD_UNCHANGED
            if self.color_space == "rgb":
                flags = cv2.IMREAD_COLOR
            raw = cv2.imread(str(path), flags)
            if raw is None:
                raise RuntimeError(f"failed to decode {path}")
            if self.color_space == "rgb":
                frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            else:
                frame = raw
        return np.asarray(frame)

    def __getitem__(self, index: int | slice | np.ndarray) -> np.ndarray:
        if not isinstance(index, (int, np.integer)):
            if self._mmap is None:
                raise TypeError("legacy volumes only support integer indexing")
            return np.asarray(self._mmap[index])
        index = int(index)
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        if self._mmap is not None:
            return np.asarray(self._mmap[index])
        return self._legacy_read(index)

    def __setitem__(self, index: int, value: np.ndarray) -> None:
        if not self.writable:
            raise TypeError(f"{self.directory} is read-only")
        index = int(index)
        if index < 0 or index >= self._count:
            raise IndexError(index)
        frame = np.asarray(value)
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"frame shape {frame.shape} != volume frame shape {self.frame_shape}"
            )
        if self._mmap is not None:
            self._mmap[index] = np.asarray(frame, dtype=self.dtype)
            return
        assert self._indexed is not None
        path = self._indexed.get(index)
        if path is None:
            suffix = ".npy" if self.color_space != "rgb" and self.dtype != np.uint8 else ".png"
            if self.dtype.kind == "f" or self.dtype == np.uint32:
                suffix = ".npy"
            path = self.directory / f"{index:06d}{suffix}"
            self._indexed[index] = path
        if path.suffix.lower() == ".npy":
            np.save(path, np.asarray(frame, dtype=self.dtype), allow_pickle=False)
            return
        if cv2 is None:
            raise RuntimeError("OpenCV is required to write legacy PNG volumes")
        out = np.asarray(frame)
        if self.color_space == "rgb":
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), out)


def create_volume(
    stream_dir: str | Path,
    count: int,
    frame_shape: tuple[int, ...],
    dtype: np.dtype | str,
    *,
    fill: int | float | None = 0,
    color_space: str | None = None,
) -> FrameVolume:
    """Create a writable packed memmap under ``stream_dir/frames.npy``."""
    directory = Path(stream_dir)
    directory.mkdir(parents=True, exist_ok=True)
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    frame_shape = tuple(int(v) for v in frame_shape)
    dtype = np.dtype(dtype)
    path = packed_path(directory)
    if path.exists() or path.is_symlink():
        path.unlink()
    mmap = np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=(count, *frame_shape)
    )
    if fill is not None:
        mmap[:] = fill
        mmap.flush()
    _write_meta(
        directory,
        count=count,
        frame_shape=frame_shape,
        dtype=dtype,
        color_space=color_space,
    )
    return FrameVolume(
        directory=directory,
        mmap=mmap,
        indexed=None,
        dtype=dtype,
        frame_shape=frame_shape,
        color_space=color_space,
        writable=True,
    )


def open_volume(
    stream_dir: str | Path,
    *,
    writable: bool = False,
    color_space: str | None = None,
    expected_count: int | None = None,
) -> FrameVolume:
    """Open a packed volume or a legacy indexed directory / ``.npy`` file."""
    path = Path(stream_dir)
    if path.is_file() and path.suffix.lower() == ".npy":
        directory = path.parent
        packed = path
    else:
        directory = path
        packed = packed_path(directory)
    if packed.is_file():
        mode = "r+" if writable else "r"
        mmap = np.load(packed, mmap_mode=mode)
        if mmap.ndim < 2:
            raise ValueError(f"packed volume {packed} must be at least 2-D, got {mmap.shape}")
        meta = _read_meta(directory)
        frame_shape = tuple(int(v) for v in mmap.shape[1:])
        inferred_color = color_space
        if inferred_color is None and meta is not None:
            inferred_color = meta.get("color_space")
        if inferred_color is None and len(frame_shape) == 3 and frame_shape[2] == 3 and mmap.dtype == np.uint8:
            inferred_color = "rgb"
        if expected_count is not None and int(mmap.shape[0]) != int(expected_count):
            raise ValueError(
                f"{packed} has {mmap.shape[0]} frames, expected {expected_count}"
            )
        return FrameVolume(
            directory=directory,
            mmap=mmap,
            indexed=None,
            dtype=mmap.dtype,
            frame_shape=frame_shape,
            color_space=inferred_color,
            writable=writable,
        )
    indexed = _scan_indexed(directory)
    if not indexed:
        raise FileNotFoundError(f"no packed frames.npy or indexed frames in {directory}")
    origin = min(indexed)
    required = set(range(origin, max(indexed) + 1))
    missing = sorted(required - set(indexed))
    extra = sorted(set(indexed) - required)
    if missing or extra:
        raise ValueError(
            f"{directory} does not cover {origin}..{max(indexed)}; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    count = max(indexed) + 1
    if expected_count is not None and (count - origin) != int(expected_count) and count != int(expected_count):
        raise ValueError(
            f"{directory} has {count - origin} frames, expected {expected_count}"
        )
    sample = indexed[origin]
    if sample.suffix.lower() == ".npy":
        first = np.asarray(np.load(sample, allow_pickle=False))
        inferred = color_space
    else:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to read legacy PNG volumes")
        inferred = color_space
        flags = cv2.IMREAD_UNCHANGED
        if inferred == "rgb":
            flags = cv2.IMREAD_COLOR
        raw = cv2.imread(str(sample), flags)
        if raw is None:
            raise RuntimeError(f"failed to decode {sample}")
        if inferred is None and raw.ndim == 3 and raw.shape[2] >= 3:
            inferred = "rgb"
        if inferred == "rgb":
            first = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            first = raw
    return FrameVolume(
        directory=directory,
        mmap=None,
        indexed=indexed,
        dtype=first.dtype,
        frame_shape=tuple(int(v) for v in first.shape),
        color_space=inferred,
        writable=writable,
    )


def as_mask(frame: np.ndarray) -> np.ndarray:
    """Interpret uint8 0/255 or boolean frames as a boolean mask.

    Accepts ``(H,W)``, ``(H,W,C)``, ``(T,H,W)`` or ``(T,H,W,C)``. A trailing
    channel axis of size 1/3/4 is dropped; a packed ``(T,H,W)`` volume is not.
    """
    array = np.asarray(frame)
    if array.ndim >= 3 and array.shape[-1] in (1, 3, 4):
        array = array[..., 0]
    if array.dtype == np.bool_:
        return array
    return array > 127


def _resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape):
        return np.asarray(mask, dtype=bool)
    if cv2 is None:
        raise RuntimeError("OpenCV is required to resize mask volumes")
    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(shape[1]), int(shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def load_mask_bool_volume(
    stream_dir: str | Path,
    *,
    count: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Load a mask stream as a ``(count, H, W)`` bool array.

    Packed ``frames.npy`` is memory-mapped and thresholded in one sequential
    pass. Legacy ``%06d.png`` / ``%06d.npy`` directories are decoded into the
    same stack after a single directory scan. Missing frames stay False.
    """
    volume = open_volume(stream_dir)
    count = int(count)
    height, width = int(shape[0]), int(shape[1])
    target = (height, width)
    usable = min(count, len(volume))
    if volume.packed:
        # Materialize a C-contiguous prefix once so the threshold is a single
        # sequential memmap read, not T Python slice round-trips.
        raw = np.ascontiguousarray(volume.array[:usable])
        masked = np.asarray(as_mask(raw), dtype=bool)
        if masked.ndim != 3:
            raise ValueError(
                f"{stream_dir} packed mask must be (T,H,W), got {masked.shape}"
            )
        if tuple(masked.shape[1:]) == target and usable == count:
            return masked
        out = np.zeros((count, height, width), dtype=bool)
        if tuple(masked.shape[1:]) == target:
            out[:usable] = masked
        else:
            for index in range(usable):
                out[index] = _resize_bool_mask(masked[index], target)
        return out
    out = np.zeros((count, height, width), dtype=bool)
    for index in range(usable):
        try:
            frame = as_mask(volume[index])
        except (FileNotFoundError, IndexError, RuntimeError):
            continue
        out[index] = _resize_bool_mask(frame, target)
    return out


def as_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def hardlink_packed(source_dir: str | Path, destination_dir: str | Path, *, force: bool = False) -> str:
    """Hard-link a packed stream directory (frames.npy + meta). Falls back to copy."""
    import os
    import shutil

    source = Path(source_dir)
    destination = Path(destination_dir)
    src_packed = packed_path(source)
    if not src_packed.is_file():
        raise FileNotFoundError(f"packed volume not found: {src_packed}")
    destination.mkdir(parents=True, exist_ok=True)
    dst_packed = packed_path(destination)
    dst_meta = meta_path(destination)
    if dst_packed.exists() or dst_packed.is_symlink():
        if not force:
            try:
                if os.path.samefile(src_packed, dst_packed):
                    return "existing_hardlink"
            except OSError:
                pass
            raise FileExistsError(dst_packed)
        dst_packed.unlink()
    try:
        os.link(src_packed, dst_packed)
        mode = "hardlink"
    except OSError:
        shutil.copy2(src_packed, dst_packed)
        mode = "copy"
    src_meta = meta_path(source)
    if src_meta.is_file():
        if dst_meta.exists() and force:
            dst_meta.unlink()
        if not dst_meta.exists():
            shutil.copy2(src_meta, dst_meta)
    return mode
