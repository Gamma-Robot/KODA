# Third-party components

KODA vendors the source needed to run its public kinematics and video pipeline so a checkout does not need Git submodules. The vendor tree retains upstream notices where they were available.

Upstream code and assets include Stera SDK, DiffuEraser, ProPainter components, ScaleBFM/ScaleTrack, utilities and robot assets from the SparseTrack adapter snapshot, Hydrax, Kimodo, and the HaWoR integration. DiffuEraser's internal ProPainter code and the ProPainter weight entries in the asset manifest remain part of the inpainting path.

The utilities previously stored in `src/koda/vendor/sparsetrack` are maintained directly in KODA:

| Function | Current location |
| --- | --- |
| Dex3 hand retargeting, configurations and fingertip evaluation | `src/koda/retargeting/` |
| Body/hand trajectory composition and MuJoCo model helpers | `src/koda/kinematics/` |
| Dual-wrist Mink IK and joint helpers | `src/koda/ik/` |
| Robot segmentation IDs | `src/koda/rendering/` |
| G1/Dex3 MJCF files and hand meshes | `resources/robots/g1_dex3/` |

This relocation removes the SparseTrack checkout dependency; it does not change the origin or licensing of the imported code and assets. The snapshot includes KODA integration helpers that replaced historical bytecode and untracked local helpers. No standalone SparseTrack `LICENSE` or `NOTICE`, or upstream revision record, was present in that snapshot. Moving these files into the `koda` namespace does not establish a new license grant. G1 body meshes continue to use the existing ScaleBFM asset tree through relative symlinks; Dex3-specific meshes and the MJCF files are retained unchanged.

Code origins are recorded in the original upstream repositories and in `assets/manifest.yaml`. Checkpoint and model locations are intentionally left to the upstream releases because they can be large or separately licensed.

MANO files are distributed by the MANO project under its own terms. Do not redistribute `MANO_RIGHT.pkl`, HaWoR checkpoints, robot meshes, or ScaleBFM weights without checking the upstream license.

Relevant upstream sources:

- [MANO](https://mano.is.tue.mpg.de/)
- [HaWoR](https://github.com/ThunderVVV/HaWoR)
- [ScaleBFM](https://github.com/Gamma-Robot/ScaleBFM)
- [DiffuEraser](https://github.com/Gamma-Robot/DiffuEraser)
- [ProPainter](https://github.com/sczhou/ProPainter)
- [Dex-retargeting](https://github.com/dexsuite/dex-retargeting)
- [EfficientSAM3](https://github.com/Gamma-Robot/efficientsam3)
- [Kimodo](https://github.com/TeleHuman/kimodo)
