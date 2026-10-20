# KODA

**Kinematics–Observation Data Alignment for Humanoid Learning from Human Egocentric Demonstration**

KODA is a data conversion framework for turning human egocentric demonstrations into reusable, controller-specific whole-body training data for humanoid loco-manipulation.

The research framework aligns human observations, head and hand motion cues, robot states, compatible actions, and task instructions on a shared control clock. It targets simulation-feasible whole-body motion, motion-aligned robot onboard views, and controller-specific state–action records for vision–language–action (VLA) policies.

**This public release provides robot kinematics and motion-aligned onboard video.** It includes conversion, hand processing, whole-body motion generation, and view synthesis. Controller action labels, SONIC token encoding, GR00T dataset export, and policy training are outside the current release.

---

## Overview

The full research workflow has four steps. This repository currently provides capture conversion and robot kinematics/video generation; controller alignment and policy training are planned for future public releases.

| Step | Name | What happens |
| :---: | :--- | :--- |
| **01** | **Capture** — *Observe the task* | Record the human viewpoint, hand and head cues, and language goal with the demonstration. |
| **02** | **Convert** — *Recover whole-body motion* | Map visual motion into robot motion and robot views while preserving the original task context. |
| **03** | **Align** — *Instantiate the interface* | Future release: project states and actions onto the target controller's compatible action space and control clock. |
| **04** | **Train** — *Fine-tune the policy* | Future release: combine KODA records with robot demonstrations to train a controller-specific whole-body VLA. |

> The research dataset design keeps robot observations, whole-body states, compatible actions, and task instructions together. The current code produces the kinematics and video inputs to that later dataset construction stage.

---

## Pipeline

![KODA pipeline](docs/pipeline.png)

*Overview of the full KODA research pipeline. This release covers whole-body motion generation and robot onboard-view synthesis. Controller-aligned VLA records are planned for a future public release.*

### 1. Whole-Body Motion Generation

Human hand observations are turned into refined, contact-aware robot whole-body references:

1. **Hand Pose Estimation** – estimate hand poses from the egocentric view.
2. **Position Retargeting** – map human hand data into raw robot hand references.
3. **Contact-Aware Refinement** – refine the raw references into physically consistent hand motions.
4. **SCO + Arm Refinement** – resolve whole-body motion together with the arm trajectory.

### 2. Robot Onboard View Generation

Given the human RGB / depth images, camera pose, and the generated robot motion, KODA renders the motion-aligned robot onboard view:

- **Arm Segmentation** – segment the human arms from the egocentric frames.
- **Point Cloud Reprojection** – reproject the scene into the robot camera frame.
- **Robot Rendering** – render the simulated robot into the scene.
- **Depth Compositing** – compose the rendered robot with the observed scene by depth.
- **Inpainting** – inpaint the occluded regions to obtain the final onboard view.

### 3. Control Command Generation (future release)

The research pipeline converts whole-body motion into executable commands via **sampling-based command optimization**, producing controller-specific actions aligned to the shared control clock. Our action-label generation implementation is based on [bfm_ttc](https://github.com/Gamma-Robot/bfm_ttc). The KODA-specific action-label generation code is not included in this public release; its release is planned under **M3**.

### 4. Humanoid VLA Dataset (future release)

The research dataset design combines, on a shared control clock:

- 📷 **Robot Onboard View** — motion-aligned, simulation-rendered onboard images.
- 🤖 **Robot States & Actions** — controller-compatible state–action records.
- 💬 **Language Instructions** — the task goal recorded with the demonstration.

---

## Open-Source Timeline

The public portions of the internal KODA pipeline are included in this repository. The table below records what is available here and what remains outside this release:

| Phase | Milestone | Deliverables | Status |
| :---: | :--- | :--- | :---: |
| 🚀 **M0** | **Project launch** | Project page, paper, and this repository | ✅ Done |
| 🧩 **M1** | **Core conversion pipeline** | RGB-D/MCAP conversion, hand pose estimation, hand retargeting, contact-aware refinement, SCO tracking, and arm refinement | ✅ Done |
| 🎥 **M2** | **Onboard view generation** | Arm segmentation, point-cloud reprojection, robot rendering, depth compositing, and inpainting | ✅ Done |
| 🎛️ **M3** | **Command generation & alignment** | Controller action labels, motion-token encoding, and controller-aligned dataset export | 🔜 Planned |
| 🗃️ **M4** | **Data release** | Reproducible data-processing tools, two public example MCAPs, and a fine-tuned ScaleBFM tracking checkpoint through `TeleEmbodied/koda-assets` | 🚧 Partial |
| 🤖 **M5** | **Training & model details** | Controller-specific whole-body VLA training configs and model details | 🔜 Planned |

> **Status legend:** ✅ Done · 🚧 Partial or in progress · 🔜 Planned

This release includes the hand-processing code, including MANO/HaWoR hand estimation, BrainCo and Dex3 retargeting, contact-aware hand refinement, and hand–object geometric contact handling. The default runnable path uses G1 + Dex3 and ends with robot kinematics and synthesized onboard video. Controller replay, action/token generation, dataset export, and the training configurations used for the paper are planned for future public releases.

---

## Requirements

- Linux, NVIDIA GPU, and a CUDA-compatible driver
- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)
- Storage for the model files; DiffuEraser's Stable Diffusion base alone is several GB

## Install

```bash
git clone https://github.com/Gamma-Robot/KODA.git
cd KODA
uv sync --extra cuda
```

All paths are relative to this checkout. `KODA_HOME`, `KODA_ASSETS`, and `KODA_CACHE` are optional overrides for a user's own directories.

## Download the example and public checkpoints

Two example MCAP recordings and a fine-tuned ScaleBFM tracking checkpoint are available in the public [`TeleEmbodied/koda-assets`](https://huggingface.co/datasets/TeleEmbodied/koda-assets) dataset. These files can be downloaded without a Hugging Face token. To download and verify the default example:

```bash
uv run koda assets download --profile smoke
uv run koda assets verify --profile smoke
```

For all stages in the public kinematics and video pipeline, start the automatic downloads:

```bash
uv run koda assets download --profile full
```

This command downloads the public Hugging Face assets listed in [`assets/manifest.yaml`](assets/manifest.yaml). It prints the remaining manual steps instead of silently using a system or cluster path. Complete those steps and verify every file:

```bash
uv run koda assets verify --profile full
```

The manual assets are:

1. Register at [MANO](https://mano.is.tue.mpg.de/), accept its license, and copy `MANO_RIGHT.pkl` to `assets/models/mano/MANO_RIGHT.pkl`.
2. Clone [HaWoR](https://github.com/ThunderVVV/HaWoR) into `assets/models/hawor`. Download its `hawor.ckpt`, `infiller.pt`, `model_config.yaml`, and the WiLoR `detector.pt` listed in the manifest. The checkout must contain `lib/models/hawor.py`, `weights/hawor/...`, and `weights/external/detector.pt`.
3. Download the three [ProPainter v0.1.0 release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0) files (`ProPainter.pth`, `raft-things.pth`, and `recurrent_flow_completion.pth`) into `assets/models/diffueraser/propainter`.

All other model sources, revisions, destination paths, and required markers are recorded in the manifest. You may set `KODA_HAWOR_DIR` or `KODA_SCALEBFM_CHECKPOINT` to another user-owned checkout or checkpoint; no default points outside the repository.

### Fine-tuned ScaleBFM checkpoint

The full asset profile downloads the upstream ScaleBFM checkpoint. To use KODA's fine-tuned G1 torso/UMI3 checkpoint, download [`model_24600.pt`](https://huggingface.co/datasets/TeleEmbodied/koda-assets/resolve/main/models/scalebfm/model_24600.pt?download=true) and select it before running the pipeline:

```bash
uv run hf download TeleEmbodied/koda-assets models/scalebfm/model_24600.pt \
  --repo-type dataset \
  --local-dir assets

export KODA_SCALEBFM_CHECKPOINT=assets/models/scalebfm/model_24600.pt
```

Run the pipeline below in the same shell. The checkpoint file is 72,856,655 bytes; verify its SHA-256 on Linux with:

```bash
echo "2d40b7326c3435c9aaf01f46e1507f747e7a7371bfa504201aaea12c539f813d  assets/models/scalebfm/model_24600.pt" | sha256sum -c -
```

## Run the kinematics and video pipeline

```bash
uv run koda run \
  --input assets/samples/session_20260926_005030_iphone14_3 \
  --output outputs/example \
  --config configs/default.yaml \
  --gpu 0
```

Use `--frames 30` for a short smoke prefix. A successful run produces:

- `converted/`: MCAP-derived RGB-D and hand observations
- `tracking/`: 29-DoF G1 body plus 14-DoF Dex3 kinematics
- `view_change/`: fixed-camera RGB-D, robot render, and inpainted video
- `run_manifest.json`: every stage command, log path, and return code
- `pipeline_result.json`: final output locations

`koda run` finishes after view synthesis. The individual stages are also available as `koda convert`, `koda track`, and `koda view-change`. The public CLI has no action-label, token-encoding, dataset-export, or training stage.

## Checks

```bash
uv run koda validate --profile smoke
uv run python scripts/validate_pipeline_layout.py
```

Generated data, caches, model files, and releases are ignored by Git. See the [project page](https://koda-humanoid.github.io/koda/) for background and paper updates.

## Project Page

- 🌐 Website: <https://koda-humanoid.github.io/koda/>
- 💻 Code repository: <https://github.com/Gamma-Robot/KODA>

---

## Citation
```bibtex
@misc{zhou2026koda,
  title         = {KODA: Kinematics--Observation Data Alignment for Humanoid Learning from Human Egocentric Demonstration},
  author        = {Zibo Zhou and Yueru Chen and Junsong Wu and Weiji Xie and Qingyao Xu and Sheng Yin and Shiqing Wang and Jiyuan Shi and Siheng Chen and Chenjia Bai and Xuelong Li},
  year          = {2026},
  eprint        = {2610.21152},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.21152v1}
}
```

---

## License

KODA integration code and each vendored upstream component retain their own license. MANO, robot meshes, and model checkpoints may have additional terms; read [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) before redistribution.
