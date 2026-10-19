# KODA Pipelines

All public entry points, maintained pipeline code, logical data paths, and
runtime resources are rooted in this directory.

## Layout

```text
KODA/
  process_session_noviz_batch.py
  pipeline_smooth_motion.txt
  pipeline_initandtracking_3point_batch.txt
  run_batch_singleframe_view_change_30to50.sh
  run_full_pipeline_smoke.sh
  scripts/
  resources/
  data/
```

`resources/` contains stable links to large third-party repositories, Python
environments, checkpoints, and robot assets. `data/` physically contains raw
AR sessions, converted RGB-D sessions, kinematics, view-change results, smoke
artifacts, and retarget experiments. Compatibility links remain at the former
data locations so existing commands continue to work.

## 1. AR Session Conversion (MCAP → RGB-D episode)

Batch entry point: `process_session_noviz_batch.py`

Single-session worker: `process_session_noviz.py`

Python env: `$KODA_HOME/envs/stera/bin/python`

```text
raw AR session MCAP (session_*/session_data_*.mcap)
  -> discover complete MCAPs + assign task labels
  -> per-GPU isolated process_session_noviz.py
  -> optional RANSAC ground align (world Y → y=0)
  -> WiLoR hand tracking + both-hands trim
  -> optional upper-body estimate / face blur / Rerun viz
  -> annotation.hdf5 + rgb.mp4 + export_manifest.json
     (+ ground_align.json when ground align runs)
```

This stage turns raw Stera AR captures into the 30 Hz RGB-D episodes consumed
by Smooth Motion and Three-Point Tracking under
`data/egocentric_data/<task>/<session>/`.

**Input layouts.** Prefer
`data/origin_data/<task>/session_*/session_data_*.mcap`. If `--input-root` is
`data/origin_data` (parent of every task), the task name is inferred from the
first path component. If `--input-root` is already one task directory
(`data/origin_data/<task>`), the tree is flat and requires `--task`,
`--task-map`, or `--session-as-task`.

**Batch behavior.** The scheduler probes MCAP magic and file age, skips outputs
that already have `export_manifest.json` + `annotation.hdf5` + `rgb.mp4`
(unless `--force`), and runs one session per selected GPU via
`CUDA_VISIBLE_DEVICES`. Unless `--with-viz` is set, it always forwards
`--no-viz` and `--no-skeleton` to the worker. Unknown CLI flags after the batch
options are forwarded unchanged to `process_session_noviz.py` (for example
`--blur`, `--no-ground-align`, `--interpolate-hand-gaps`).

**Ground alignment (default on).** The worker fits the lowest near-horizontal
plane on the session mesh with RANSAC (`--ground-plane-thresh`, default
`0.015` m; `--ground-min-up`, default `0.9`) and subtracts that Y so the ground
plane sits at `y=0` (ARKit / Stera is Y-up). The shift is applied to the mesh,
camera poses (`annotation.hdf5:/cam-pose/translations`), and world point
clouds. It is **not** applied to hand poses (camera_3d), depth, IMU, or
calibrations. Provenance is written to `ground_align.json` and mirrored as
HDF5 attrs (`ground_y_offset`, `ground_axis=y`). Pass `--no-ground-align` to
leave the recorded world origin unchanged. This conversion-time Y lift is
separate from the later tracking-time
`WRIST_REFERENCE_WORLD_Z_OFFSET_M` / `--wrist-reference-world-z-offset-m`
(Z-up robot reference); do not conflate the two.

**Published episode fields.** A completed session directory contains at least
`annotation.hdf5`, `rgb.mp4`, and `export_manifest.json`. With ground align
enabled it also has `ground_align.json`. Batch metadata is written as
`batch_job.json`; a run summary defaults to
`<output-root>/_batch_summary.json`.

## 2. Smooth Motion

Entry point: `pipeline_smooth_motion.txt`

```text
30 Hz RGB-D session annotations
  -> bounded transient wrist repair
  -> timestamp-aware local Savitzky-Golay rigid hand smoothing
  -> pipeline_initandtracking_3point_batch.txt
```

The smoothing stage modifies only `hand-pose/{left,right}_joints` in the
selected working-copy sessions. Production quality review stops the pipeline.
`ALLOW_SMOOTHING_REVIEW=1` is reserved for functional smoke tests.

## 3. Three-Point Tracking

Entry point: `pipeline_initandtracking_3point_batch.txt`

```text
complete source sequence on the measured 50 Hz timeline
  -> bounded transient wrist repair
  -> template_palm_kabsch dual-wrist SE(3)
  -> position-retargeted 14-DoF Dex3 hands on the same timeline
  -> source RGB-D contact detection
  -> joint right wrist-link SE(3) + right Dex3 7-DoF contact optimization
  -> causal edge-held wrist residual injection
  -> torso + dual-wrist three-point reference
  -> bounded wrist excursion repair + timestamp-aware local SE(3) smoothing
  -> Kimodo initialization with IK/physics gate
  -> JAX CEM-MPC ScaleTrack
  -> offline Mink arm refinement with frame-0 initialization/clearance branch
  -> 29-DoF G1 + synchronized 14-DoF Dex3 + contact provenance
  -> kinematics.npz + meta.json + tracking_error.json
```

The wrist target uses the wrist plus index/middle/ring/little MCP anchors, a
sequence-median palm template, and per-frame Huber-IRLS Kabsch. Optional
per-session orientation calibration and a common world-Z offset are applied at
this point. Only the two wrist pose/velocity streams and matching Kimodo wrist
targets are replaced by template-palm retargeting.

Joint contact precomputation is enabled by default
(`JOINT_CONTACT_PRECOMPUTE=1`). It position-retargets both Dex3 hands, uses
EfficientSAM3 plus source RGB-D geometry to detect contact, and, when contact
exists, jointly optimizes the right wrist-yaw-link SE(3) and the right Dex3
7-DoF configuration. The first, per-contact-frame, and last optimized wrist
residuals are held before, during, and after contact respectively, then
composed with the moving template-palm wrist reference. A no-contact session
keeps the template-palm wrist reference unchanged and carries the base
position-retargeted hand trajectory. Automatic joint-contact precomputation is
mutually exclusive with the legacy `CONTACT_AWARE_ROOT` input.

The RGB-D contact geometry runs on the worker GPU by default
(`CONTACT_DEVICE=cuda`, bound through that worker's `GPU` and
`CUDA_VISIBLE_DEVICES`). `select_contact_mask_window.py` uses a cached Torch
CUDA depth projection, and `contact_pointcloud_analysis.py` uses Torch CUDA
projection plus chunked exact nearest-point queries; the output manifest records
the requested device and actual backends. Surface-normal estimation and the
subsequent SciPy/MuJoCo fitting/optimization remain CPU stages. Set
`CONTACT_DEVICE=cpu` only for an explicit CPU fallback.

After contact residual injection, the pipeline builds the torso plus dual-wrist
three-point reference. It always repairs bounded jump-out-and-return wrist
SE(3) excursions and applies timestamp-aware local Savitzky-Golay smoothing to
torso and both wrists (`window=5`, `polyorder=2`); there is no disable option.
The resulting reference and wrist targets are the sole inputs to Kimodo,
ScaleTrack, Mink, and final error evaluation; no post-Mink wrist correction is
permitted.

Kimodo Seed 02 initialization must pass its IK/physics gate. A physically
plausible source-state fallback exists only when explicitly enabled with
`ALLOW_KIMODO_BASELINE_FALLBACK=1`. ScaleTrack then runs JAX CEM-MPC, followed
by offline Mink arm refinement. The default Mink frame-0 branch removes arm
smoothness at frame 0 and enables sampled shoulder-clearance geometry with
cost 5; frames 1 onward retain the normal warm start and temporal smoothness.

Each published session contains only `kinematics.npz`, `meta.json`, and
`tracking_error.json`. `kinematics.npz` keeps the 29-DoF G1 trajectory and adds
timestamp-identical `(T, 14)` `hand_qpos`/`hand_qvel`, Dex3 joint names, 30 Hz
and 50 Hz contact flags, the joint-contact contract, configuration hashes, and
input/dependency provenance. Output completeness is checked against these
contracts before an existing result is reused.

## 4. View Change

Formal batch entry point:
`scripts/view_change/run_contact_aware_batch_view_change.sh`

Internal single-session renderer:
`scripts/view_change/run_singleframe_view_change_30to50.sh`

```text
raw source RGB-D session
  -> 50 Hz raw/template-palm reference bootstrap (no Kimodo/ScaleTrack/Mink)
  -> base Dex3 position retarget on the reference timeline
  -> EfficientSAM3 object and human hand/arm masks
  -> RGB-D point-cloud contact detection and wrist+Dex3 joint optimization
  -> causal residual injection and mandatory reference smoothing
  -> one final Kimodo/ScaleTrack/Mink tracking pass
  -> verified 29-DoF G1 + synchronized 14-DoF Dex3 tracking archive
  -> strict source RGB-D/calibration adapter
  -> EfficientSAM3 human hand/arm masks
  -> extract synchronized 29-DoF G1 + 14-DoF Dex3 tracking state
  -> compose and verify render kinematics
  -> fixed-camera MuJoCo RGB/metric-depth/object-ID render at 50 Hz
  -> explicit measured 30 Hz <-> 50 Hz timeline alignment
  -> masked single-frame RGB-D target-view warp with geometry gate
  -> metric-depth/object-ID robot compositing at source 30 Hz
  -> DiffuEraser visual-only completion of remaining synthetic holes
  -> monotonic past-only hold of all streams onto the measured 50 Hz timeline
  -> camera, timeline, pixel, provenance, and frame-count acceptance checks
```

The formal batch wrapper first prepares the reference directly from raw
`annotation.hdf5` in a temporary worker directory. This bootstrap runs only
the 50 Hz raw-reference preparation and template-palm Kabsch conversion; it
does not invoke Kimodo, ScaleTrack, Mink, and it does not read any prior
`data/kinematics/<session>/kinematics.npz`. Contact precompute consumes that
temporary reference/timeline to create the complete `contact_aware_auto`
bundle, including the wrist candidate, contact manifest, contact frames, and
base/optimized Dex3 hand. Only after that bundle exists does the worker call
the latest tracking driver, so each session has exactly one final
Kimodo/ScaleTrack/Mink pass.

Tracking injects the causal wrist residual, constructs and always smooths the
new torso/dual-wrist reference, and publishes a finite `(T, 29)` body plus
timestamp-identical `(T, 14)` hand with contact and smoothing provenance.
No-contact sessions retain the identity wrist residual and base
position-retargeted hand.

The resulting tracking archive is the sole robot-state source for View Change:
it performs no hand retargeting and applies no additional wrist offset. This
preserves the contact-aware fingers and template-palm/MPC/Mink body-wrist
solution as one synchronized state.

Source RGB-D registration uses each session's own RGB intrinsics and requires
depth aligned to RGB plus RGB frames locked to camera poses. All target-view
operations instead share the fixed calibration in
`configs/target_camera_672x376.json`; target dimensions cannot override that
contract. MuJoCo renders robot RGB, metric depth, object IDs, and target camera
poses from the composed G1+Dex3 trajectory, after which the rendered camera
contract is checked before warping or compositing.

The geometry path is strictly single-frame. Each 30 Hz source frame is matched
to a target pose by relative timestamp, warped once with source depth,
confidence, depth-edge invalidation, and the EfficientSAM3 human mask, and is
never temporally fused with historical backgrounds. A geometry gate records
per-frame quality and stops the formal pipeline on warp failure; only a clearly
marked diagnostic passthrough is written in that case. Robot compositing uses
metric depth and required MuJoCo object IDs without scale adjustment.

DiffuEraser runs in `visual_only` mode and may modify only the remaining
synthetic-hole mask; observed and robot pixels must remain unchanged. The
processed 30 Hz streams (`rgb_geometry`, depth, validity/hole/confidence,
source count, robot/paste masks, final RGB, synthetic mask, and provenance)
are then mapped to the measured 50 Hz robot clock with a monotonic past-only
hold. This is a timestamp mapping, not generic frame-rate interpolation.

The final verifier requires exact source/target frame counts, fixed target
camera intrinsics and dimensions, consistent warp/timeline indices, no
multi-frame fused pixels, nonblank RGB, verified native DiffuEraser execution,
and no changes outside synthetic holes. The main result is
`final_50hz/rgb_final_50hz.mp4`, with frame-level streams and quality reports
under the per-session output directory.

Runs are resumable through stage markers. `REUSE_BASE_ROOT` can reuse an
accepted mask/warp/timeline while always rebuilding robot kinematics, render,
composite, inpainting, and final outputs from the current tracking archive.
`PRUNE_INTERMEDIATES=1` removes large consumed intermediates only after their
downstream status contracts exist. The formal batch wrapper also defaults
`PRUNE_CONTACT_EVIDENCE=1`: after a session publishes its final contract, the
per-frame `human_masks/` and `cup_segmentation_auto/` trees are removed while
the compact contact candidate, hand bundle, manifest, and contact timeline are
retained. A failed precompute with no complete bundle discards the same partial
mask trees so a retry starts cleanly. Set `KEEP_INTERMEDIATE=1` (or
`PRUNE_CONTACT_EVIDENCE=0`) when those masks are needed for debugging or
post-hoc contact evaluation.

The formal batch runs one persistent worker per selected GPU from the
comma-separated `GPUS` list. Workers claim sessions from a shared atomic
dynamic queue, so a GPU that finishes early immediately receives the next
session instead of waiting for a static shard. Each worker uses an isolated
temporary directory, verifies the post-contact smoothing, 29+14 DoF shapes,
timestamps, contact flags, camera sidecar, and final video, then publishes only
`rgb_final_50hz.mp4`, `kinematics.npz`, and
`camera_kinematics_30hz.npz`. The temporary session directory is deleted after
successful or failed processing unless debugging retention is explicitly
requested inside the worker.

## Commands

The public pipeline covers session conversion, whole-body and hand kinematics, and Dex3/BrainCo view synthesis. See [`README.md`](README.md#run-the-kinematics-and-video-pipeline) for the supported CLI. Controller action labels, token encoding, dataset export, and training remain outside this release. Dry-run and layout checks:

```bash
# Validate paths and retarget contracts
uv run python scripts/validate_pipeline_layout.py

# AR session conversion (MCAP -> data/egocentric_data/<task>/<session>/)
# input-root is one flat task dir, so --task is required
$KODA_HOME/envs/stera/bin/python process_session_noviz_batch.py \
  --input-root data/origin_data/walk_pick_human \
  --output-root data/egocentric_data \
  --task walk_pick_human \
  --python $KODA_HOME/envs/stera/bin/python \
  --processor process_session_noviz.py \
  --gpus auto --workers 1 --dry-run

# Or point input-root at data/origin_data and let nested <task>/<session>/ infer tasks
# $KODA_HOME/envs/stera/bin/python process_session_noviz_batch.py \
#   --input-root data/origin_data --output-root data/egocentric_data ...

# Same conversion with ground align disabled (passthrough to the worker)
$KODA_HOME/envs/stera/bin/python process_session_noviz_batch.py \
  --input-root data/origin_data/walk_pick_human \
  --output-root data/egocentric_data \
  --task walk_pick_human \
  --python $KODA_HOME/envs/stera/bin/python \
  --processor process_session_noviz.py \
  --gpus 1 --workers 1 \
  --no-ground-align

# Optional ground RANSAC knobs (also forwarded to the worker)
#   --ground-plane-thresh 0.015
#   --ground-min-up 0.9

# Dry-run one tracking / View Change session
DRY_RUN=1 LIMIT=1 GPUS=1 WORKERS=1 ./pipeline_smooth_motion.txt
DRY_RUN=1 MAX_EPISODES=1 ./run_batch_singleframe_view_change_30to50.sh
DRY_RUN=1 MAX_EPISODES=1 scripts/view_change/run_contact_aware_batch_view_change.sh

# Multi-GPU formal tracking + View Change batch
GPUS=0,1,2,3 WORKERS=4 scripts/view_change/run_contact_aware_batch_view_change.sh

# Reproducible 30-frame full-chain smoke
GPU=1 FRAMES=30 ./run_full_pipeline_smoke.sh
```
