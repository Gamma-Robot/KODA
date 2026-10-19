#!/usr/bin/env python3
"""Tests for EfficientSAM3 prompt slugs and persistent-worker orchestration."""

from __future__ import annotations

import unittest
from pathlib import Path

from efficientsam3_client import prompt_slug

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTE = PROJECT_ROOT / "scripts/view_change/run_contact_precompute.sh"
BATCH = PROJECT_ROOT / "scripts/view_change/run_contact_aware_batch_view_change.sh"
WORKERS = PROJECT_ROOT / "scripts/view_change/resident_gpu_workers.sh"
TRACKING = PROJECT_ROOT / "scripts/view_change/run_contact_aware_6d_reference_tracking.sh"
SINGLEFRAME = PROJECT_ROOT / "scripts/view_change/run_singleframe_view_change_30to50.sh"
ERASER = PROJECT_ROOT / "scripts/view_change/run_remaining_hole_eraser.py"
MASKS = PROJECT_ROOT / "scripts/view_change/make_human_arm_masks.py"


class PromptSlugTest(unittest.TestCase):
    def test_matches_precompute_tr(self) -> None:
        self.assertEqual(prompt_slug("a wooden cup"), "a_wooden_cup")
        self.assertEqual(prompt_slug("a light blue cup"), "a_light_blue_cup")
        self.assertEqual(prompt_slug("a blue cup"), "a_blue_cup")
        self.assertEqual(prompt_slug("Human Arms"), "human_arms")


class PersistentWorkerOrchestrationTest(unittest.TestCase):
    def test_precompute_uses_shared_client(self) -> None:
        text = PRECOMPUTE.read_text(encoding="utf-8")
        self.assertIn("efficientsam3_client.py", text)
        self.assertIn("--prompts-pipe", text)
        self.assertNotIn("load_efficientsam3_image_model(device=\"cuda\"", text)

    def test_batch_defaults_to_stage_barrier(self) -> None:
        text = BATCH.read_text(encoding="utf-8")
        workers = WORKERS.read_text(encoding="utf-8")
        self.assertIn('STAGE_BARRIER="${STAGE_BARRIER:-1}"', text)
        self.assertIn("start_efficientsam3_worker", text)
        self.assertIn("start_kimodo_worker", text)
        self.assertIn('if [[ -z "${KIMODO_PY:-}" ]]', workers)
        self.assertIn('"${KIMODO_PY}" "${KIMODO_WORKER}"', workers)
        self.assertNotIn('"${SCALER_PY}" "${KIMODO_WORKER}"', workers)
        self.assertIn("start_diffueraser_worker", workers)
        self.assertIn("start_da3_worker", workers)
        self.assertIn("ensure_view_change_workers", workers)
        self.assertIn("run_phase_pool precompute", text)
        self.assertIn("run_phase_pool tracking", text)
        self.assertIn("KIMODO_WORKER_SOCK", text)
        self.assertIn("DA3_WORKER_SOCK", text)

    def test_view_change_reuses_contact_human_masks(self) -> None:
        tracking = TRACKING.read_text(encoding="utf-8")
        single = SINGLEFRAME.read_text(encoding="utf-8")
        masks = MASKS.read_text(encoding="utf-8")
        batch = BATCH.read_text(encoding="utf-8")
        self.assertIn("REUSE_HUMAN_MASKS_DIR", tracking)
        self.assertIn("SOURCE_ADAPTER_DIR", tracking)
        self.assertIn("SOURCE_ADAPTER_DIR", single)
        self.assertIn("run_human_masks", single)
        self.assertIn("[human_masks] cache hit", single)
        self.assertIn("[adapter] cache hit", single)
        self.assertIn("--reuse-from", single)
        self.assertIn("def _reuse_packed_masks", masks)
        self.assertIn("EFFICIENTSAM3_WORKER_SOCK", masks)
        self.assertIn('"mask_kind": "human_body"', single)
        self.assertIn('"mask_kind": "human_body"', masks)
        self.assertIn("source_adapter", batch)

    def test_view_change_starts_deferred_da3_and_diffu(self) -> None:
        tracking = TRACKING.read_text(encoding="utf-8")
        single = SINGLEFRAME.read_text(encoding="utf-8")
        self.assertIn("LOG_ROOT", tracking)
        self.assertIn("DA3_WORKER_SOCK", tracking)
        self.assertIn("ensure_resident_view_workers", single)
        self.assertIn("DA3_WORKER_SOCK", single)
        sweep = (
            PROJECT_ROOT
            / "resources/ScaleBFM/ScaleTrack/scripts/eval_session/run_clean_kimodo_init_sweep.py"
        ).read_text(encoding="utf-8")
        self.assertIn("KIMODO_WORKER_SOCK", sweep)
        driver = (PROJECT_ROOT / "scripts/run_three_point_batch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("KIMODO_WORKER_SOCK", driver)

    def test_eraser_can_use_persistent_worker(self) -> None:
        text = ERASER.read_text(encoding="utf-8")
        self.assertIn("DIFFUERASER_WORKER_SOCK", text)
        self.assertIn("_diffueraser_worker_request", text)

    def test_pcm_lora_loads_from_local_checkout(self) -> None:
        de_py = PROJECT_ROOT / "resources/DiffuEraser/diffueraser/diffueraser.py"
        worker = PROJECT_ROOT / "resources/DiffuEraser/diffueraser_worker.py"
        manifest = (PROJECT_ROOT / "assets/manifest.yaml").read_text(
            encoding="utf-8"
        )
        source = de_py.read_text(encoding="utf-8")
        self.assertIn("resolve_pcm_lora_weights", source)
        self.assertIn("pcm_root, weight_name=PCM_ckpts", source)
        self.assertNotIn(
            'load_lora_weights(\n                "weights/PCM_Weights"',
            source,
        )
        self.assertIn("os.chdir(here)", worker.read_text(encoding="utf-8"))
        self.assertIn("repo_id: wangfuyun/PCM_Weights", manifest)
        self.assertIn("target: models/diffueraser/weights/PCM_Weights", manifest)

    def test_precompute_packs_source_rgb_and_shares_prompts(self) -> None:
        text = PRECOMPUTE.read_text(encoding="utf-8")
        self.assertIn("packed_source_rgb.py", text)
        self.assertIn("--packed-rgb", text)
        self.assertIn("--source-rgb-dir", text)
        self.assertIn("--skip-plots", text)

    def test_geometry_gate_defaults_off(self) -> None:
        single = SINGLEFRAME.read_text(encoding="utf-8")
        warp = (
            PROJECT_ROOT / "scripts/view_change/viewport_gt_depth_warp.py"
        ).read_text(encoding="utf-8")
        self.assertIn('--gate-mode "${GEOMETRY_GATE_MODE:-off}"', single)
        self.assertIn('default="off"', warp)
        self.assertIn("geometry gate skipped", warp)
        quality = (
            PROJECT_ROOT / "scripts/view_change/quality_report.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"bypassed"', quality)

    def test_orchestration_drops_dead_mp4s_and_50hz_volumes(self) -> None:
        single = SINGLEFRAME.read_text(encoding="utf-8")
        self.assertNotIn("robot_composite.mp4", single)
        self.assertNotIn("rgb_final_30hz.mp4", single)
        self.assertIn("--skip-50hz-volumes", single)
        self.assertIn("--source-rgb-dir", single)
        batch = BATCH.read_text(encoding="utf-8")
        self.assertIn("source_rgb", batch)
        self.assertIn("final_50hz contract missing", batch)


if __name__ == "__main__":
    unittest.main()
