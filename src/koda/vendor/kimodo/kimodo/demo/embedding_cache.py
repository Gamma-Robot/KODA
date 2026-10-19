# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import contextvars
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from kimodo.sanitize import sanitize_texts

_ACTIVE_SESSION = contextvars.ContextVar("kimodo_demo_active_session", default=None)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    disk_hits: int = 0


class EmbeddingCache:
    """Disk-backed text embedding cache with a small in-memory LRU."""

    def __init__(
        self,
        *,
        model_name: str,
        encoder_id: str,
        base_dir: Optional[str] = None,
        max_mem_entries: int = 128,
    ) -> None:
        cache_root = base_dir or os.environ.get(
            "kimodo_EMBED_CACHE_DIR",
            os.path.join("~", ".cache", "kimodo_demo", "embeddings"),
        )
        self.base_dir = os.path.expanduser(cache_root)
        self.model_name = model_name
        self.encoder_id = encoder_id
        self.max_mem_entries = max_mem_entries
        self.stats = CacheStats()

        self._lock = threading.Lock()
        self._mem_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._index = {}
        self._index_loaded = False

    def _model_dir(self) -> str:
        return os.path.join(self.base_dir, self.model_name)

    def _index_path(self) -> str:
        return os.path.join(self._model_dir(), "index.json")

    def _prewarm_marker_path(self, key: str) -> str:
        return os.path.join(self._model_dir(), f"prewarm_{key}.json")

    def has_prewarm_marker(self, key: str) -> bool:
        return os.path.exists(self._prewarm_marker_path(key))

    def write_prewarm_marker(self, key: str, *, prompt_count: int) -> None:
        os.makedirs(self._model_dir(), exist_ok=True)
        payload = {"prompt_count": prompt_count, "updated_at": time.time()}
        tmp_path = f"{self._prewarm_marker_path(key)}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, self._prewarm_marker_path(key))

    def _load_index(self) -> None:
        if self._index_loaded:
            return
        index_path = self._index_path()
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except json.JSONDecodeError:
                self._index = {}
        self._index_loaded = True

    def _save_index(self) -> None:
        os.makedirs(self._model_dir(), exist_ok=True)
        tmp_path = f"{self._index_path()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f)
        os.replace(tmp_path, self._index_path())

    def _make_key(self, text: str) -> str:
        key_src = f"{self.model_name}|{self.encoder_id}|{text}"
        return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    def _entry_path(self, key: str) -> str:
        return os.path.join(self._model_dir(), f"{key}.npy")

    def _mem_get(self, key: str) -> Optional[np.ndarray]:
        if key in self._mem_cache:
            self._mem_cache.move_to_end(key)
            return self._mem_cache[key]
        return None

    def _mem_put(self, key: str, value: np.ndarray) -> None:
        self._mem_cache[key] = value
        self._mem_cache.move_to_end(key)
        while len(self._mem_cache) > self.max_mem_entries:
            self._mem_cache.popitem(last=False)

    def _disk_load(self, key: str) -> Optional[np.ndarray]:
        path = self._entry_path(key)
        if not os.path.exists(path):
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    def _disk_save(self, key: str, value: np.ndarray) -> None:
        os.makedirs(self._model_dir(), exist_ok=True)
        np.save(self._entry_path(key), value)
        self._index[key] = {
            "length": int(value.shape[0]),
            "dtype": str(value.dtype),
            "updated_at": time.time(),
        }

    def _maybe_use_session_cache(self, texts: list[str]):
        session = _ACTIVE_SESSION.get()
        if session is None:
            return None
        if session.last_prompt_texts == texts and session.last_prompt_embeddings is not None:
            return session.last_prompt_embeddings, session.last_prompt_lengths
        return None

    def _update_session_cache(self, texts: list[str], tensor: torch.Tensor, lengths: list[int]) -> None:
        session = _ACTIVE_SESSION.get()
        if session is None:
            return
        session.last_prompt_texts = texts
        session.last_prompt_embeddings = tensor
        session.last_prompt_lengths = lengths

    def get_or_encode(self, texts: Iterable[str], encoder):
        if isinstance(texts, str):
            texts = [texts]
        texts = sanitize_texts(list(texts))
        if len(texts) == 0:
            empty = torch.empty()
            return empty, []

        session_cache = self._maybe_use_session_cache(texts)
        if session_cache is not None:
            return session_cache

        arrays: list[Optional[np.ndarray]] = [None] * len(texts)
        lengths: list[int] = [0] * len(texts)
        misses: list[tuple[int, str, str]] = []

        with self._lock:
            self._load_index()
            for idx, text in enumerate(texts):
                key = self._make_key(text)
                cached = self._mem_get(key)
                if cached is not None:
                    arrays[idx] = cached
                    lengths[idx] = cached.shape[0]
                    self.stats.hits += 1
                    continue

                cached = self._disk_load(key)
                if cached is not None:
                    arrays[idx] = cached
                    lengths[idx] = cached.shape[0]
                    self._mem_put(key, cached)
                    self.stats.disk_hits += 1
                    continue

                misses.append((idx, text, key))
                self.stats.misses += 1

        if misses:
            miss_texts = [text for _, text, _ in misses]
            miss_tensor, miss_lengths = encoder(miss_texts)
            miss_tensor = miss_tensor.detach().cpu()
            miss_tensor_np = miss_tensor.numpy()

            with self._lock:
                self._load_index()
                for miss_idx, length in enumerate(miss_lengths):
                    idx, _text, key = misses[miss_idx]
                    arr = miss_tensor_np[miss_idx, :length].copy()
                    arrays[idx] = arr
                    lengths[idx] = int(length)
                    self._mem_put(key, arr)
                    self._disk_save(key, arr)
                self._save_index()

        max_len = max(lengths) if lengths else 0
        feat_dim = arrays[0].shape[-1] if arrays[0] is not None else 0
        dtype = arrays[0].dtype if arrays[0] is not None else np.float32
        padded = np.zeros((len(texts), max_len, feat_dim), dtype=dtype)
        for idx, arr in enumerate(arrays):
            if arr is None:
                continue
            padded[idx, : arr.shape[0]] = arr

        result = torch.from_numpy(padded)
        self._update_session_cache(texts, result, lengths)
        return result, lengths


class CachedTextEncoder:
    """Wrapper around a text encoder to add disk-backed caching."""

    def __init__(self, encoder, *, model_name: str, base_dir: Optional[str] = None):
        self.encoder = encoder
        self.model_name = model_name
        encoder_id = f"{type(encoder).__name__}"
        self.cache = EmbeddingCache(model_name=model_name, encoder_id=encoder_id, base_dir=base_dir)

    def __call__(self, texts):
        return self.cache.get_or_encode(texts, self.encoder)

    def prewarm(self, texts) -> None:
        if isinstance(texts, str):
            texts = [texts]
        texts = sanitize_texts(list(texts))
        prewarm_key = hashlib.sha256("|".join(texts).encode("utf-8")).hexdigest()
        if self.cache.has_prewarm_marker(prewarm_key):
            return
        self.cache.get_or_encode(texts, self.encoder)
        self.cache.write_prewarm_marker(prewarm_key, prompt_count=len(texts))

    def to(self, device=None, dtype=None):
        if hasattr(self.encoder, "to"):
            self.encoder.to(device=device, dtype=dtype)
        return self

    @contextlib.contextmanager
    def session_context(self, session):
        token = _ACTIVE_SESSION.set(session)
        try:
            yield
        finally:
            _ACTIVE_SESSION.reset(token)

    def __getattr__(self, name):
        return getattr(self.encoder, name)
