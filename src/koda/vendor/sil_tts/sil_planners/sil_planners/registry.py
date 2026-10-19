"""Name-keyed planner registry for config-driven selection.

Loading semantics: built-in algorithms register as a side effect of
importing `sil_planners.algs`; `sil_planners/__init__` imports it eagerly,
so `import sil_planners` alone guarantees built-ins are available. Optional
planners are registered as lazy import paths and load only when selected.
Custom planners register when their defining module is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec

from sil_planners.base import SamplingPlanner

_REGISTRY: dict[str, type[SamplingPlanner]] = {}


@dataclass(frozen=True)
class _OptionalPlanner:
    module: str
    attribute: str
    dependency: str
    extra: str


_OPTIONAL_REGISTRY: dict[str, _OptionalPlanner] = {}


def _validate_planner(name: str, cls: object) -> type[SamplingPlanner]:
    if not (isinstance(cls, type) and issubclass(cls, SamplingPlanner)):
        raise TypeError(
            f"planner {name!r} must be a SamplingPlanner subclass, got {cls!r}"
        )
    return cls


def register(name: str):
    def decorator(cls: type[SamplingPlanner]) -> type[SamplingPlanner]:
        cls = _validate_planner(name, cls)
        if name in _REGISTRY or name in _OPTIONAL_REGISTRY:
            raise ValueError(f"planner {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def _register_optional(
    name: str,
    *,
    module: str,
    attribute: str,
    dependency: str,
    extra: str,
) -> None:
    """Register an optional planner without importing its dependency stack."""
    if name in _REGISTRY or name in _OPTIONAL_REGISTRY:
        raise ValueError(f"planner {name!r} already registered")
    _OPTIONAL_REGISTRY[name] = _OptionalPlanner(
        module=module,
        attribute=attribute,
        dependency=dependency,
        extra=extra,
    )


def get_planner(name: str) -> type[SamplingPlanner]:
    try:
        return _REGISTRY[name]
    except KeyError:
        pass

    optional = _OPTIONAL_REGISTRY.get(name)
    if optional is None:
        raise KeyError(
            f"unknown planner {name!r}; available: {list(available())}"
        ) from None
    if find_spec(optional.dependency) is None:
        raise ModuleNotFoundError(
            f"planner {name!r} requires optional dependency {optional.dependency!r}; "
            f"install sil-planners[{optional.extra}]"
        ) from None

    module = import_module(optional.module)
    cls = _validate_planner(name, getattr(module, optional.attribute))
    _REGISTRY[name] = cls
    return cls


def available() -> tuple[str, ...]:
    optional = {
        name
        for name, entry in _OPTIONAL_REGISTRY.items()
        if find_spec(entry.dependency) is not None
    }
    return tuple(sorted(_REGISTRY.keys() | optional))
