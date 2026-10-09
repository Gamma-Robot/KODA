# KODA

**Kinematics–Observation Data Alignment for Humanoid Learning from Human Egocentric Demonstration**

KODA is a data conversion framework for turning human egocentric demonstrations into reusable, controller-specific whole-body training data for humanoid loco-manipulation.

The project aligns human observations, head and hand motion cues, robot states, compatible actions, and task instructions on a shared control clock. Its conversion pipeline produces simulation-feasible whole-body motion, motion-aligned robot onboard views, and controller-specific state–action records that can be combined with real-robot demonstrations to train vision–language–action (VLA) policies.

---

## Overview

KODA converts a human egocentric demonstration into controller-specific whole-body training data in four steps:

| Step | Name | What happens |
| :---: | :--- | :--- |
| **01** | **Capture** — *Observe the task* | Record the human viewpoint, hand and head cues, and language goal with the demonstration. |
| **02** | **Convert** — *Recover whole-body motion* | Map visual motion into robot motion and robot views while preserving the original task context. |
| **03** | **Align** — *Instantiate the interface* | Project states and actions onto the target controller's compatible action space and control clock. |
| **04** | **Train** — *Fine-tune the policy* | Combine KODA records with robot demonstrations to train a controller-specific whole-body VLA. |

> Every training record keeps robot observations, whole-body states, compatible actions, and task instruction together, so later debugging can follow the same path as training.

---

## Pipeline

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

### 3. Control Command Generation

Whole-body motion is converted into executable commands via **sampling-based command optimization**, producing the controller-specific actions aligned to the shared control clock.

### 4. Humanoid VLA Dataset

The output dataset combines, on a shared control clock:

- 📷 **Robot Onboard View** — motion-aligned, simulation-rendered onboard images.
- 🤖 **Robot States & Actions** — controller-compatible state–action records.
- 💬 **Language Instructions** — the task goal recorded with the demonstration.

---

## Open-Source Timeline

Code, data-processing tools, model details, and release instructions will be added progressively. The planned release schedule is below (we will keep this table updated as items land):

| Phase | Milestone | Deliverables | Status |
| :---: | :--- | :--- | :---: |
| 🚀 **M0** | **Project launch** | Project page, paper, and this repository | ✅ Done |
| 🧩 **M1** | **Core conversion pipeline** | Whole-body motion generation (hand pose estimation → retargeting → contact-aware refinement → SCO + arm refinement) | 🚧 In progress |
| 🎥 **M2** | **Onboard view generation** | Robot onboard view generation tools (arm segmentation, point-cloud reprojection, robot rendering, depth compositing, inpainting) | 🚧 In progress |
| 🎛️ **M3** | **Command generation & alignment** | Sampling-based command optimization; projection onto target controllers' action space and control clock | 🔜 Planned |
| 🗃️ **M4** | **Data release** | Humanoid VLA dataset: robot onboard views, robot states & actions, and language instructions; data-processing tools and release instructions | 🔜 Planned |
| 🤖 **M5** | **Training & model details** | Controller-specific whole-body VLA training configs and model details (combined with real-robot demonstrations) | 🔜 Planned |

> **Status legend:** ✅ Done · 🚧 In progress · 🔜 Planned

---

## Getting Started

> Release instructions will be added progressively. Please refer to the [project page](https://koda-humanoid.github.io/koda/) for the latest updates.

```bash
# Clone the repository
git clone https://github.com/Gamma-Robot/KODA.git
cd KODA

# Setup instructions coming soon
```

---

## Project Page

- 🌐 Website: <https://koda-humanoid.github.io/koda/>
- 💻 Code repository: <https://github.com/Gamma-Robot/KODA>

---

## Citation

If you find KODA useful in your research, please consider citing:

```bibtex
@article{koda2026,
  title   = {KODA: Kinematics--Observation Data Alignment for Humanoid Learning},
  author  = {Zibo Zhou and Yueru Chen and Junsong Wu and Weiji Xie and Qingyao Xu and Sheng Yin and Jiyuan Shi and Siheng Chen and Chenjia Bai and Xuelong Li},
  year    = {2026},
  note    = {Project page}
}
```

---

## License

License information will be released together with the code.
