# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse and normalize prompt text/duration data from meta dicts."""

import os
from typing import Any, Optional

from kimodo.tools import load_json

from .sanitize import sanitize_text, sanitize_texts


def load_prompts_from_meta(meta_path: str, **kwargs):
    """Load prompts from a meta dict or file. If fps is provided, the durations are converted to
    frames.

    Args:
        meta_path: Path to the meta file.
        **kwargs: Additional arguments to pass to parse_prompts_from_meta.

    Returns:
        texts: List of texts.
        durations: List of durations in seconds or frames.
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in input folder: {meta_path}")

    meta = load_json(meta_path)
    return parse_prompts_from_meta(meta, **kwargs)


def parse_prompts_from_meta(
    meta: dict[str, Any],
    fps: Optional[float] = None,
    sanitize: bool = False,
) -> tuple[list[str], list[float]]:
    """Parse prompt texts and durations from a meta dict into normalized lists. If fps is provided,
    the durations are converted to frames.

    Accepts either:
    - Single prompt: "text" (str) and "duration" (float) in seconds.
    - Multiple prompts: "texts" (list of str) and "durations" (list of float) in seconds.

    Returns:
        (texts, durations): texts as list of str, durations as list of float (seconds or frames).
        Lengths of both lists are equal.

    Raises:
        ValueError: If meta does not contain a recognized format.
    """
    # Single prompt
    if "text" in meta and "duration" in meta:
        text = meta["text"]
        duration = float(meta["duration"])
        if fps is not None:
            duration = int(duration * fps)
        if isinstance(text, list):
            raise ValueError("meta has 'text' but it is a list; use 'texts' for multiple prompts")

        if sanitize:
            text = sanitize_text(text)
        return ([text], [duration])

    # Multiple prompts
    if "texts" in meta and "durations" in meta:
        texts = meta["texts"]
        durations = meta["durations"]
        if not isinstance(texts, list) or not isinstance(durations, list):
            raise ValueError("meta 'texts' and 'durations' must be lists")
        if len(texts) != len(durations):
            raise ValueError(f"meta 'texts' and 'durations' length mismatch: {len(texts)} vs {len(durations)}")
        durations = [float(d) for d in durations]
        if fps is not None:
            durations = [int(d * fps) for d in durations]

        if sanitize:
            texts = sanitize_texts(texts)
        return texts, durations

    raise ValueError("meta must contain either 'text' and 'duration', or 'texts' and 'durations'.")
