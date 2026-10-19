# sil_planners

Tracker- and simulator-agnostic sampling-based black-box planners. The
package owns the optimizer interface and algorithms; `sil_tts` owns motion
decoding, tracker execution, MJWarp rollout evaluation, and Cost construction.
`sil_planners` never imports `sil_trackers` or `sil_tts`.

## Planner interface

Every planner sees only normalized latent candidates and finite scalar Costs:

```text
candidates (N, K, D) float64 ──▶ BatchEvaluator ──▶ costs (N,) float64
```

The public contract is deliberately small:

- Costs are lower-is-better and finite;
- public planner arrays use NumPy `float64`;
- randomness comes from the seeded `params.rng`;
- planner state belongs to one Planning Update and is not shared across
  receding updates;
- rollout failures are converted to finite penalties by the evaluator, not by
  the planner.

Physics, tracker observations, search-space decoding, terminal terms, domain
randomization, and wave pagination are outside this interface.

## Algorithms

All built-ins are registered by name and have defaults in
`configs/planners/<name>.yaml`.

| Name | Summary |
| --- | --- |
| `predictive_sampling` | Select the lowest-cost observed sample. |
| `mppi` | Softmax-weighted mean update. |
| `mppi_cma` | MPPI update with covariance adaptation. |
| `cem` | Fit a diagonal Gaussian to elite samples. |
| `dial` | MPPI with dual-loop annealed covariance. |
| `cbo` | Consensus-based stochastic particle update. |
| `weighted_elites` | Antithetic sampling and cost-weighted elite update. |
| `spider` | Horizon- and iteration-annealed structured sampling. |
| `evosax` *(optional)* | Adapter for EvoSax distribution strategies. |

The implementations are inspired by
[hydrax](https://github.com/vincekurtz/hydrax), the production sampler in
[sil_wbc](https://github.com/jason-wjs/sil_wbc), and
[SPIDER](https://github.com/facebookresearch/spider). Method references and
algorithm-specific details live in the source modules and tests.

## Optional EvoSax adapter

The base installation does not import JAX or EvoSax. To enable the adapter:

```bash
uv sync --package sil-planners --extra evosax
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --package sil-planners \
  --extra evosax python -c \
  "from sil_planners import get_planner; print(get_planner('evosax'))"
```

The adapter still returns the same NumPy `float64` interface.

## Adding a planner

1. Add `sil_planners/algs/<name>.py` and subclass `SamplingPlanner`.
2. Implement the sampling and parameter-update hooks; register the name.
3. Add `configs/planners/<name>.yaml`.
4. Add contract, determinism, and convergence tests under
   `tests/planners/`.
5. Extend the planner matrix if the algorithm is a built-in.

Optional runtime adapters belong behind a declared package extra and must not
be imported during base-package import. Keep planner changes local: adding an
algorithm must not require tracker, MJWarp, or objective changes.
