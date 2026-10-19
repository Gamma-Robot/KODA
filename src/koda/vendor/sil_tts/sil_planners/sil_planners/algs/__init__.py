"""Algorithm implementations. Importing this package registers all built-ins."""

from sil_planners.registry import _register_optional

from sil_planners.algs import (  # noqa: F401
    cbo,
    cem,
    dial,
    mppi,
    mppi_cma,
    predictive_sampling,
    spider,
    weighted_elites,
)

_register_optional(
    "evosax",
    module="sil_planners.algs.evosax",
    attribute="Evosax",
    dependency="evosax",
    extra="evosax",
)
