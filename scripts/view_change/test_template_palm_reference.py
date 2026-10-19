import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from retarget_template_palm_reference import (
    ANCHOR_INDICES,
    ROBOT_PALM_SITE,
    build_palm_template,
    apply_local_orientation_offset,
    repair_landmark_events,
    template_palm_kabsch,
)


class TemplatePalmReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        hand = np.zeros((21, 3), dtype=np.float64)
        hand[0] = [0.0, 0.0, 0.0]
        hand[5] = [0.08, 0.035, 0.0]
        hand[9] = [0.09, 0.012, 0.0]
        hand[13] = [0.087, -0.012, 0.0]
        hand[17] = [0.075, -0.035, 0.0]
        self.hand = hand

    def test_template_palm_kabsch_tracks_known_rigid_motion(self) -> None:
        base = np.repeat(self.hand[None], 5, axis=0)
        rotations = Rotation.from_euler(
            "xyz", np.column_stack((np.zeros(5), np.zeros(5), np.linspace(0.0, 0.4, 5)))
        ).as_matrix()
        translations = np.column_stack(
            (np.linspace(0.1, 0.2, 5), np.linspace(-0.1, 0.0, 5), np.full(5, 0.8))
        )
        moved = np.einsum("tij,tkj->tki", rotations, base) + translations[:, None]
        positions, fitted_rotations, _ = template_palm_kabsch(moved)
        template = build_palm_template(base)
        fitted_palm = positions + np.einsum(
            "tij,j->ti", fitted_rotations, ROBOT_PALM_SITE
        )
        reconstructed = (
            np.einsum("tij,pj->tpi", fitted_rotations, template)
            + fitted_palm[:, None]
        )
        np.testing.assert_allclose(
            reconstructed, moved[:, list(ANCHOR_INDICES)], atol=1.0e-8
        )

    def test_template_is_finite(self) -> None:
        template = build_palm_template(np.repeat(self.hand[None], 3, axis=0))
        self.assertEqual(template.shape, (5, 3))
        self.assertTrue(np.all(np.isfinite(template)))

    def test_repair_landmark_events_interpolates_complete_hand(self) -> None:
        hands = np.repeat(self.hand[None], 5, axis=0)
        hands[0] += [0.0, 0.0, 0.0]
        hands[1:4] += [1.0, -2.0, 3.0]
        hands[4] += [0.4, 0.8, -0.2]
        repaired, indices = repair_landmark_events(
            hands,
            np.arange(5, dtype=np.float64),
            [{"left_anchor_index": 0, "right_anchor_index": 4}],
            "right",
        )
        expected = np.stack(
            [
                self.hand + fraction * np.asarray([0.4, 0.8, -0.2])
                for fraction in np.linspace(0.0, 1.0, 5)
            ]
        )
        np.testing.assert_allclose(repaired, expected)
        self.assertEqual(indices, [1, 2, 3])

    def test_local_orientation_offset_preserves_requested_pivot(self) -> None:
        positions = np.asarray([[0.1, -0.2, 0.8], [0.2, 0.1, 0.9]])
        rotations = Rotation.from_euler("xyz", [[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]]).as_matrix()
        offset = Rotation.from_rotvec([0.2, -0.1, 0.05]).as_matrix()
        corrected_position, corrected_rotation = apply_local_orientation_offset(
            positions, rotations, offset, ROBOT_PALM_SITE
        )
        original_pivot = positions + np.einsum(
            "tij,j->ti", rotations, ROBOT_PALM_SITE
        )
        corrected_pivot = corrected_position + np.einsum(
            "tij,j->ti", corrected_rotation, ROBOT_PALM_SITE
        )
        np.testing.assert_allclose(corrected_pivot, original_pivot, atol=1.0e-12)
        np.testing.assert_allclose(
            corrected_rotation, np.einsum("tij,jk->tik", rotations, offset)
        )


if __name__ == "__main__":
    unittest.main()
