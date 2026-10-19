# sil_tts

Runtime for Simulator-in-the-Loop test-time scaling (SIL-TTS) experiments.

It composes a registered sampling planner, a pretrained tracker, an Any4HDMI
motion target, and one locked G1 MJWarp model into paired baseline and
optimized executions. The experiment path varies test-time compute and records
the resulting tracking metrics and planning evidence.

The package owns motion ingestion, search-space decoding, Cost construction,
candidate rollout evaluation, synchronous receding execution, experiment
records, and offline rendering. Planner and tracker packages remain independent
of this runtime.

## Run one experiment

Behavioral parameters belong in YAML under `configs/experiments/`:

```bash
uv run sil-tts run configs/experiments/wxy_smoke.yaml \
  --device-index 0
```

The command supports only device selection, output selection, and `--dry-run`.
Use `MUJOCO_GL=egl` on headless Linux machines. Batch manifests use the same
composition path:

```bash
uv run sil-tts batch \
  configs/experiments/numpy_planners_short_qualification.batch.yaml \
  --device-index 0
```

Outputs are written under `outputs/` and include the resolved experiment,
paired traces, planning ledger, metrics, provenance, and optional
`comparison.mp4`. An existing output can be rendered again with:

```bash
uv run sil-tts render outputs/<experiment>/<run>
```

## Experiment configuration

An experiment composes stable tracker, robot, planner, and motion-tracking
configuration by reference:

```yaml
schema_version: 2

tracker:
  ref: trackers/wxy_wbc.yaml
robot:
  ref: robots/g1_mjwarp.yaml
planner:
  ref: planners/weighted_elites.yaml
motion_tracking:
  ref: motion_tracking/default.yaml
```

See [`configs/experiments/wxy_smoke.yaml`](../../configs/experiments/wxy_smoke.yaml)
for a complete minimal experiment. Configuration references are resolved and
validated before artifacts or GPU execution are initialized.

## Robust SIL

Robust SIL is selected in experiment YAML. It evaluates each candidate under a
deterministic batch of physics scenarios and reduces scenario Costs before the
planner sees them:

```yaml
robustness:
  enabled: true
  num_scenarios: 4
  scenario_seed: 1313
  risk: mean
```

The randomization implementation and risk reduction belong to `sil_tts`; the
planner and tracker interfaces do not change.

## Runtime seams

- `motion_tracking/` — Any4HDMI targets, qpos-knot guidance, objective Costs,
  and tracking metrics;
- `planning/` — candidate evaluation and one finite-budget Planning Update;
- `simulation/` — Execution State, MJWarp worlds, and guided execution;
- `sil_loop.py` — synchronous receding execution;
- `run_sil.py` and `batch_run_sil.py` — console composition and batch entry
  points;
- `visualization/` — offline single/comparison rendering from persisted traces.

Canonical terminology and global dependency rules are in
[`CONTEXT.md`](../../CONTEXT.md) and
[`docs/architecture.md`](../../docs/architecture.md). This package exposes only
the current Interface v2 schema.
