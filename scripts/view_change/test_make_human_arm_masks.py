#!/usr/bin/env python3
"""Full-body human mask composition keeps limbs and drops a separate held object."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from make_human_arm_masks import compose_human_mask


class ComposeHumanMaskTest(unittest.TestCase):
    def test_legs_and_torso_are_kept_and_separate_object_is_not(self) -> None:
        shape = (200, 200)
        arm_raw = np.zeros(shape, dtype=np.uint8)
        arm_raw[20:50, 20:50] = 255
        arm_raw[80:110, 80:110] = 255
        support = np.zeros(shape, dtype=np.uint8)
        support[15:55, 15:55] = 255
        body = np.zeros(shape, dtype=np.uint8)
        body[140:190, 20:180] = 255
        body[80:110, 80:110] = 255
        legs = np.zeros(shape, dtype=np.uint8)
        legs[150:195, 30:90] = 255
        speck = np.zeros(shape, dtype=np.uint8)
        speck[5:8, 5:8] = 255

        mask = compose_human_mask(
            arm_raw,
            [body, legs, speck],
            support,
            min_area_fraction=0.002,
            object_protect_px=4,
            body_dilation_px=2,
        )

        self.assertTrue(mask[30, 30] > 127)
        self.assertTrue(mask[160, 40] > 127)
        self.assertTrue(mask[170, 100] > 127)
        self.assertFalse(mask[95, 95] > 127)
        self.assertFalse(mask[6, 6] > 127)

    def test_cache_contract_requires_full_body_kind(self) -> None:
        root = Path(__file__).resolve().parent
        single = (root / "run_singleframe_view_change_30to50.sh").read_text(encoding="utf-8")
        precompute = (root / "run_contact_precompute.sh").read_text(encoding="utf-8")
        masks = (root / "make_human_arm_masks.py").read_text(encoding="utf-8")
        self.assertIn('"mask_kind": "human_body"', single)
        self.assertIn('"mask_kind": "human_body"', precompute)
        self.assertIn('"mask_kind": "human_body"', masks)
        self.assertIn('default="person"', masks)
        self.assertIn('default="human legs"', masks)


if __name__ == "__main__":
    unittest.main()
