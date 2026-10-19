# sil_trackers

Tracker-specific semantics for pretrained Unitree G1 whole-body policies. The
package owns observation construction, policy I/O, opaque tracker state,
action-to-command mapping, family registration, and the host/device inference
adapters. It does not import planners, MJWarp, or `sil_tts`.

## Integrated families

| Family / instance | Configuration |
| --- | --- |
| WXY-WBC | `configs/trackers/wxy_wbc.yaml` |
| HEFT-PMG | `configs/trackers/heft_pmg.yaml` |
| released native-G1 SONIC | `configs/trackers/sonic_release_g1.yaml` |
| low-latency G1 SONIC | `configs/trackers/sonic_low_latency_g1.yaml` |
| TWIST2 `twist2_1017_20k` | `configs/trackers/twist2_1017_20k.yaml` |
| Humanoid-GPT | `configs/trackers/humanoid_gpt.yaml` |
| Mimic-Lite `32x8192-huge` | `configs/trackers/mimic_lite_32x8192_huge.yaml` |

Each instance declares its family semantics, command joints, query offsets,
guidance-velocity stencil, policy I/O, and artifact identities. Large policies
and robot assets are resolved through the repository artifact lock rather than
committed to Git.

Mimic-Lite's legacy source graph is rank-one. Its family-owned, fail-closed
offline transform produces the batched artifact used by both runtime paths.
The checkpoint contract is `policy[535]`, `command[304]` to `action[29]`, with
17-step zero-filled history at lags `(0, 1, 2, 3, 4, 8, 16)`, three raw previous
actions, signed offsets `(-8, -4, -2, 0, 1, 2, 3, 4)`, pelvis anchoring with
absolute z, sticky `reference_yaw * conjugate(robot_yaw)` alignment, and no
action clipping before PD mapping. The prepared device implementation preserves
the same state semantics.

## Family interface

Each family implements `TrackerAdapter` as the authority for policy semantics:

- `initial_state`, `clone_state`, and `expand_state` own opaque state;
- `build_policy_inputs` constructs one Control Step of observations;
- `map_policy_outputs` returns the canonical `PDCommand` and next state.

`SerialReferenceTracker` is the required host bring-up and conformance path. A
family may additionally provide `PreparedDeviceTracker` to keep observation
history, policy I/O, and command mapping on the simulation device. The
application discovers this capability; it does not branch on tracker names.

## Adding a tracker family

1. Add the family adapter under `sil_trackers/families/<family>/`.
2. Add an instance YAML under `configs/trackers/`.
3. Add source and derived policy identities to `artifacts/lock.v1.json`.
4. Add serial semantics, policy I/O, and conformance tests under
   `tests/trackers/`.
5. Bring the family up through the host reference path first. Add a prepared
   device implementation only when profiling justifies it.

A conforming family changes its own directory, instance configuration, artifact
entries, and tests. Shared changes require a family-neutral interface and
cross-family tests.
