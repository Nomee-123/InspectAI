"""Tests for the progress hook, report contract and generated demo samples."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from inspectai import build_inspection_report, inspect_images

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
EXPECTED_STAGES = [
    "acquire",
    "orientation",
    "translation",
    "register",
    "photometric",
    "analyze",
    "verify",
]


def make_scene() -> np.ndarray:
    image = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(image, (30, 26), (190, 150), (196, 200, 206), -1)
    cv2.rectangle(image, (30, 26), (190, 150), (32, 36, 44), 3)
    cv2.circle(image, (246, 88), 38, (90, 140, 210), -1)
    cv2.line(image, (20, 205), (300, 176), (40, 120, 230), 7)
    cv2.putText(
        image,
        "A7",
        (58, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (28, 30, 36),
        3,
        cv2.LINE_AA,
    )
    for index in range(8):
        x = 42 + index * 24
        cv2.rectangle(image, (x, 168), (x + 5, 190), (0, 210, 240), -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


def make_grid() -> np.ndarray:
    grid = np.zeros((160, 200, 3), np.uint8)
    for y in range(0, 160, 20):
        for x in range(0, 200, 20):
            if ((x // 20) + (y // 20)) % 2 == 0:
                cv2.rectangle(grid, (x, y), (x + 19, y + 19), (220, 220, 220), -1)
    return grid


class ProgressTests(unittest.TestCase):
    def test_progress_events_cover_every_real_stage_once(self) -> None:
        reference = make_scene()
        test = reference.copy()
        matrix = cv2.getRotationMatrix2D((160, 120), 8, 1.0)
        matrix[0, 2] += 6
        matrix[1, 2] -= 4
        test = cv2.warpAffine(test, matrix, (320, 240))

        events: list[tuple[str, dict]] = []
        result = inspect_images(
            reference, test, progress_callback=lambda stage, detail: events.append((stage, detail))
        )

        starts = [stage for stage, detail in events if detail["phase"] == "start"]
        finishes = [stage for stage, detail in events if detail["phase"] == "finish"]
        self.assertEqual(starts, EXPECTED_STAGES)
        self.assertEqual(finishes, EXPECTED_STAGES)
        self.assertTrue(all(detail["phase"] in {"start", "finish", "update"} for _, detail in events))
        self.assertEqual(result.alignment_status, "applied")
        for stage, detail in events:
            if detail["phase"] == "finish":
                self.assertIn("status", detail)
                self.assertIsInstance(detail.get("functions", []), list)
        self.assertTrue(
            any(
                function.startswith(("cv2.", "np.", "detector."))
                for _, detail in events
                for function in detail.get("functions", [])
            )
        )

    def test_register_updates_match_the_models_that_were_tried(self) -> None:
        reference = make_scene()
        test = cv2.warpAffine(
            reference,
            cv2.getRotationMatrix2D((160, 120), 9, 1.0),
            (320, 240),
        )
        events: list[tuple[str, dict]] = []
        inspect_images(reference, test, progress_callback=lambda stage, detail: events.append((stage, detail)))

        updates = [
            detail
            for stage, detail in events
            if stage == "register" and detail["phase"] == "update"
        ]
        self.assertTrue(updates)
        for update in updates:
            self.assertIn(update["model"], {"similarity", "affine", "homography"})
            self.assertIn(update["status"], {"accepted", "rejected", "ambiguous"})
            self.assertTrue(update["reason"])
            self.assertIsInstance(update["functions"], list)

        finish = next(
            detail
            for stage, detail in events
            if stage == "register" and detail["phase"] == "finish"
        )
        self.assertEqual(
            [entry["model"] for entry in finish["models"]],
            [update["model"] for update in updates],
        )

    def test_progress_listener_failure_does_not_abort_the_inspection(self) -> None:
        reference = make_scene()

        def broken_listener(stage: str, detail: dict) -> None:
            raise RuntimeError("listener failed")

        result = inspect_images(reference, reference, progress_callback=broken_listener)

        self.assertEqual(result.alignment_status, "not_needed")
        self.assertTrue(
            any("progress listener" in warning for warning in result.uncertainty_warnings)
        )


class BackendGateTests(unittest.TestCase):
    def test_180_degree_rotation_is_an_applied_orientation_fix(self) -> None:
        reference = make_scene()
        test = cv2.rotate(reference, cv2.ROTATE_180)

        result = inspect_images(reference, test)

        self.assertEqual(result.alignment_status, "applied")
        self.assertEqual(result.alignment_method, "orientation")
        self.assertTrue(result.alignment_applied)
        self.assertLess(result.final_comparison.changed_percentage, 0.5)
        self.assertTrue(
            any(
                step.name == "Orientation fix" and step.status == "accepted"
                for step in result.alignment_steps
            )
        )

    def test_one_period_grid_shift_is_ambiguous_not_silently_corrected(self) -> None:
        grid = make_grid()
        shifted = np.roll(grid, 20, axis=1)

        result = inspect_images(grid, shifted)

        self.assertEqual(result.alignment_status, "ambiguous")
        self.assertFalse(result.alignment_applied)
        self.assertEqual(result.alignment_method, "none")
        self.assertTrue(
            any("ambiguous" in warning.lower() for warning in result.uncertainty_warnings)
        )


class ReportContractTests(unittest.TestCase):
    def test_alignment_steps_are_json_safe_and_contain_no_image_data(self) -> None:
        reference = make_scene()
        test = cv2.warpAffine(
            reference,
            cv2.getRotationMatrix2D((160, 120), 12, 1.0),
            (320, 240),
        )
        result = inspect_images(reference, test)
        report = build_inspection_report(reference, test, result)

        encoded = json.dumps(report, allow_nan=False)
        self.assertIn("alignment_steps", report)
        steps = report["alignment_steps"]
        self.assertEqual(len(steps), len(result.alignment_steps))
        self.assertTrue(steps)
        for step in steps:
            self.assertEqual(np.asarray(step["matrix"]).shape, (3, 3))
            self.assertIn(step["status"], {"accepted", "rejected", "skipped", "ambiguous"})
            self.assertNotIn("preview", step)
            self.assertNotIn("image", step)
            self.assertNotIn("keypoints_ref", step["metrics"])
        feature_steps = [step for step in steps if step["keypoints_test"]]
        self.assertTrue(feature_steps)
        for step in feature_steps:
            self.assertEqual(step["keypoint_format"], ["x", "y", "inlier"])
            first = step["keypoints_test"][0]
            self.assertEqual(len(first), 3)
        self.assertNotIn("data:image", encoded)
        self.assertNotIn("base64", encoded.lower())


class SampleManifestTests(unittest.TestCase):
    def test_manifest_lists_every_required_demo_pair(self) -> None:
        manifest_path = SAMPLES / "manifest.json"
        self.assertTrue(manifest_path.is_file(), "run samples/make_samples.py first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = {case["id"] for case in manifest["cases"]}
        self.assertTrue(
            {"shifted", "rotated_15", "rotated_180", "brighter", "grid_ambiguous"} <= ids
        )
        for case in manifest["cases"]:
            self.assertTrue((SAMPLES / case["reference"]).is_file())
            self.assertTrue((SAMPLES / case["test"]).is_file())
            self.assertTrue(case["label"])
            self.assertTrue(case["expectation"])


if __name__ == "__main__":
    unittest.main()
