# G1/Dex3 robot assets

These MJCF files and meshes were moved from the SparseTrack adapter snapshot previously stored in `src/koda/vendor/sparsetrack/assets`. KODA loads them directly; no SparseTrack checkout is required.

- `g1_with_visual_dex3_hands.xml`: full body and Dex3 visual model used for retarget evaluation, trajectory composition and rendering.
- `g1_29dof_with_hand_st_collision.xml`: collision model used by the optional offline FK diagnostic.
- Hand and merged-elbow STL files are retained byte-for-byte from the imported snapshot.
- Body meshes are relative symlinks to `resources/ScaleBFM/scalebridge/data/robot/g1_29dof/meshes`. Preserve those symlinks when copying a checkout. The explicit mesh directory supplied by the pipeline continues to take precedence when loading models.

Source and licensing information is recorded in [THIRD_PARTY.md](../../../docs/THIRD_PARTY.md). The move does not change the applicable upstream terms.
