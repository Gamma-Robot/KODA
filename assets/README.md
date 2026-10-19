# Assets

`manifest.yaml` is the source of truth for inputs and checkpoints.

- The only KODA-owned example input is the MCAP in `TeleEmbodied/koda-assets`.
- Licensed or large public checkpoints are not copied into Git or the KODA dataset.
- `source: manual` entries point to an upstream release and give the expected local destination.
- `koda assets verify --profile smoke` checks the example; `full` also checks the manually installed model assets.

Set `KODA_ASSETS` when the asset cache should live outside the repository. The downloaded sample keeps the relative path recorded in the manifest.
