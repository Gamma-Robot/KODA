"""Checkpoint-bound observation contracts for SONIC tracker instances."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SonicVariant:
    name: str
    query_offsets: tuple[int, ...]
    policy_input_names: tuple[str, str]
    guidance_input_dim: int


DEFAULT = SonicVariant(
    name="release",
    query_offsets=tuple(range(0, 50, 5)),
    policy_input_names=("g1_input", "proprioception"),
    guidance_input_dim=640,
)
LOW_LATENCY = SonicVariant(
    name="low_latency",
    query_offsets=tuple(range(10)),
    policy_input_names=("encoder_obs", "proprioception"),
    guidance_input_dim=1247,
)

VARIANTS = {variant.name: variant for variant in (DEFAULT, LOW_LATENCY)}


def get_variant(name: object) -> SonicVariant:
    if not isinstance(name, str):
        raise ValueError("sonic family_config.variant must be a string")
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown sonic variant {name!r}; available: {sorted(VARIANTS)}"
        ) from None


__all__ = ["DEFAULT", "LOW_LATENCY", "SonicVariant", "get_variant"]
