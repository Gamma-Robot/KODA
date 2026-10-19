import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from resample_processed_30to50 import DEFAULT_STREAMS, resample_processed_outputs
from frame_volume import create_volume, open_volume
from timeline_alignment import (
    build_timeline_manifest,
    create_timeline_manifest_from_files,
    map_source_to_nearest_target,
    map_target_to_past_source,
    nearest_time_indices,
    previous_time_indices,
)

try:
    import h5py
except Exception:
    h5py = None


class TimelineAlignmentTest(unittest.TestCase):
    def test_nearest_tie_uses_earlier_sample(self):
        query = np.array([0.5, 1.5, 2.0])
        samples = np.array([0.0, 1.0, 2.0])
        indices, offsets = nearest_time_indices(query, samples)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_allclose(offsets, [-0.5, -0.5, 0.0])
        np.testing.assert_array_equal(
            map_source_to_nearest_target(query, samples), indices
        )

    def test_nearest_duplicate_timestamp_uses_first_frame(self):
        indices = map_source_to_nearest_target([0.0, 0.25], [0.0, 0.0, 0.5])
        np.testing.assert_array_equal(indices, [0, 0])

    def test_past_only_mapping_repeats_without_future(self):
        source = np.array([0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0])
        target = np.arange(6, dtype=np.float64) / 50.0
        indices, ages = previous_time_indices(target, source)
        np.testing.assert_array_equal(indices, [0, 0, 1, 1, 2, 3])
        self.assertTrue(np.all(source[indices] <= target + 1e-9))
        self.assertTrue(np.all(ages >= -1e-12))
        np.testing.assert_array_equal(
            map_target_to_past_source(source, target), indices
        )

    def test_past_only_rejects_target_before_source(self):
        with self.assertRaisesRegex(ValueError, "no source frame"):
            map_target_to_past_source([0.1, 0.2], [0.0, 0.1])

    def test_manifest_has_independent_render_and_output_maps(self):
        source_absolute = np.array([100.0, 100.04, 100.08])
        target = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
        manifest = build_timeline_manifest(
            source_absolute,
            target,
            expected_source_frames=3,
            expected_target_frames=5,
        )
        self.assertEqual(manifest["source_to_target"]["target_indices"], [0, 2, 4])
        self.assertEqual(manifest["output_50hz"]["source_indices"], [0, 0, 1, 1, 2])
        self.assertTrue(manifest["output_50hz"]["no_future"])
        self.assertEqual(manifest["output_50hz"]["duplicate_frame_count"], 2)

    @unittest.skipIf(h5py is None, "working h5py is unavailable")
    def test_file_readers_build_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hdf5_path = root / "annotation.hdf5"
            npz_path = root / "kinematics.npz"
            with h5py.File(hdf5_path, "w") as handle:
                handle.create_dataset(
                    "cam-pose/timestamps", data=np.array([50.0, 50.04, 50.08])
                )
            np.savez(npz_path, timestamps=np.array([0.0, 0.02, 0.04, 0.06, 0.08]))
            manifest = create_timeline_manifest_from_files(
                hdf5_path,
                npz_path,
                expected_source_frames=3,
                expected_target_frames=5,
            )
        self.assertEqual(manifest["source"]["frame_count"], 3)
        self.assertEqual(manifest["target"]["frame_count"], 5)
        self.assertEqual(manifest["output_50hz"]["last_source_index_used"], 2)


class ResampleProcessedTest(unittest.TestCase):
    def _fixture(self, root: Path, *, fused: bool = False):
        processed = root / "processed_30hz"
        source_times = np.array([0.0, 0.04, 0.08])
        target_times = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
        manifest = build_timeline_manifest(source_times, target_times)
        for stream in DEFAULT_STREAMS:
            directory = processed / stream
            if stream in ("rgb_geometry", "rgb_final"):
                volume = create_volume(
                    directory, 3, (48, 64, 3), np.uint8, fill=None, color_space="rgb"
                )
                for index in range(3):
                    volume[index] = np.full((48, 64, 3), 20 + index, dtype=np.uint8)
            else:
                volume = create_volume(directory, 3, (4, 5), np.uint8, fill=0)
                for index in range(3):
                    if stream == "provenance":
                        value = 2 if fused and index == 1 else 1
                    else:
                        value = index + 1
                    volume[index] = np.full((4, 5), value, dtype=np.uint8)
            volume.flush()
        with (processed / "frame_status.jsonl").open("w") as handle:
            for index in range(3):
                handle.write(json.dumps({"frame": index, "status": "normal"}) + "\n")
        return processed, manifest

    def test_all_streams_use_identical_hold_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            processed, manifest = self._fixture(root)
            output = root / "final_50hz"
            report = resample_processed_outputs(
                processed,
                manifest,
                output,
                expected_source_frames=3,
                expected_output_frames=5,
                width=64,
                height=48,
                write_video=True,
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["mapping_first_20"], [0, 0, 1, 1, 2])
            for stream in DEFAULT_STREAMS:
                self.assertEqual(len(open_volume(output / stream)), 5)
            gathered = open_volume(output / "rgb_final")[3]
            source = open_volume(processed / "rgb_final")[1]
            np.testing.assert_array_equal(gathered, source)
            held = open_volume(output / "hole")[3]
            self.assertTrue(np.all(held == 2))
            statuses = [
                json.loads(line)
                for line in (output / "frame_status.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["source_frame_index"] for row in statuses], [0, 0, 1, 1, 2]
            )
            self.assertTrue(statuses[1]["repeated_from_previous"])
            self.assertEqual(report["video"]["actual"]["frame_count"], 5)
            self.assertEqual(report["video"]["actual"]["fps"], 50.0)

    def test_fused_provenance_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            processed, manifest = self._fixture(root, fused=True)
            with self.assertRaisesRegex(ValueError, "fused pixels"):
                resample_processed_outputs(
                    processed,
                    manifest,
                    root / "final_50hz",
                    expected_source_frames=3,
                    expected_output_frames=5,
                    write_video=False,
                )


if __name__ == "__main__":
    unittest.main()
