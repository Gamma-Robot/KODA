# Third-party components

KODA vendors the source needed to run its maintained pipeline adapters so a checkout does not need Git submodules. The vendor tree retains upstream notices where they were available.

Upstream code and assets include Stera SDK, Depth Anything 3, DiffuEraser, ProPainter components, ScaleBFM/ScaleTrack, SparseTrack, Hydrax, Kimodo, the SONIC/GR00T dataset format, and the HaWoR integration.

Code origins are recorded in the original upstream repositories and in `assets/manifest.yaml`. Checkpoint and model locations are intentionally left to the upstream releases because they can be large or separately licensed.

MANO files are distributed by the MANO project under its own terms. Do not redistribute `MANO_RIGHT.pkl`, HaWoR checkpoints, robot meshes, or SONIC/ScaleBFM weights without checking the upstream license.

Relevant upstream sources:

- [MANO](https://mano.is.tue.mpg.de/)
- [HaWoR](https://github.com/ThunderVVV/HaWoR)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [ScaleBFM](https://github.com/Gamma-Robot/ScaleBFM)
- [DiffuEraser](https://github.com/Gamma-Robot/DiffuEraser)
- [Dex-retargeting](https://github.com/dexsuite/dex-retargeting)
- [EfficientSAM3](https://github.com/Gamma-Robot/efficientsam3)
- [Kimodo](https://github.com/TeleHuman/kimodo)
