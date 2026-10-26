"""Behavioural regression tests for the evidence-driven inspection agent.

These tests exercise the full pipeline rather than individual helpers.  They
encode the promises that matter most for InspectAI: geometric alignment is
attempted and inspectable, local changes survive it, ambiguous textures are
flagged rather than silently corrected, and the report never turns a
measurement into a defect claim.
"""

from __future__ import annotations

import json
import unittest
from typing import Iterator

import cv2
import numpy as np

from inspectai import build_inspection_report, inspect_images


def make_geometric_scene() -> np.ndarray:
    """Sparse but distinctive scene used for rotation and perspective checks."""
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (24, 20), (81, 76), (210, 210, 210), -1)
    cv2.circle(image, (116, 83), 19, (100, 170, 230), -1)
    cv2.line(image, (12, 108), (146, 96), (180, 70, 40), 5)
    return image


def make_textured_scene() -> np.ndarray:
    """Deterministic textured scene so feature alignment has enough support."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 60, (160, 200, 3), dtype=np.uint8)
    for _ in range(35):
        x, y = rng.integers(0, 200), rng.integers(0, 160)
        circle_radius = int(rng.integers(2, 10))
        colour = tuple(int(value) for value in rng.integers(80, 255, 3))
        cv2.circle(image, (int(x), int(y)), circle_radius, colour, -1)
    cv2.rectangle(image, (50, 40), (140, 120), (200, 200, 200), 3)
    return image


def _iter_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


class AlignmentBehaviourTests(unittest.TestCase):
    def test_quarter_turns_are_tried_but_arbitrary_rotations_are_aligned(self) -> None:
        reference = make_geometric_scene()

        for angle in (5, -5, 15, 45):
            with self.subTest(angle=angle):
                matrix = cv2.getRotationMatrix2D(
                    (reference.shape[1] / 2, reference.shape[0] / 2), angle, 1.0
                )
                rotated = cv2.warpAffine(
                    reference,
                    matrix,
                    (reference.shape[1], reference.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )

                result = inspect_images(reference, rotated)

                self.assertEqual(result.alignment_status, "applied")
                self.assertTrue(result.alignment_applied)
                self.assertIn("similarity", result.alignment_method)
                self.assertAlmostEqual(
                    result.estimated_rotation_degrees, angle, delta=1.0
                )
                self.assertLess(result.final_comparison.changed_percentage, 1.0)

    def test_combined_rotation_scale_and_shift_use_a_richer_model(self) -> None:
        reference = make_textured_scene()
        matrix = cv2.getRotationMatrix2D((100, 80), 7, 0.94)
        matrix[0, 2] += 9
        matrix[1, 2] -= 5
        transformed = cv2.warpAffine(
            reference, matrix, (reference.shape[1], reference.shape[0])
        )

        result = inspect_images(reference, transformed)

        self.assertEqual(result.alignment_status, "applied")
        self.assertTrue(result.alignment_applied)
        self.assertIn(result.alignment_method, {"affine", "similarity+affine"})
        self.assertAlmostEqual(result.estimated_rotation_degrees, 7.0, delta=1.0)
        # The descriptor maps test coordinates back to the reference canvas, so
        # the reported scale is the reciprocal of the applied scale.
        self.assertAlmostEqual(result.estimated_scale, 1.0 / 0.94, delta=0.02)
        self.assertLess(result.final_comparison.changed_percentage, 1.0)

    def test_perspective_warp_is_inspectable_and_aligned(self) -> None:
        reference = make_geometric_scene()
        cv2.putText(
            reference,
            "A7",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (30, 30, 30),
            2,
        )
        source = np.float32(
            [[0, 0], [159, 0], [159, 127], [0, 127]]
        )
        target = np.float32([[8, 6], [151, 11], [155, 119], [4, 122]])
        perspective = cv2.warpPerspective(
            reference,
            cv2.getPerspectiveTransform(source, target),
            (reference.shape[1], reference.shape[0]),
        )

        result = inspect_images(reference, perspective)

        self.assertEqual(result.alignment_status, "applied")
        self.assertTrue(result.alignment_applied)
        self.assertIn("homography", result.alignment_method)
        self.assertLess(result.final_comparison.changed_percentage, 1.0)

        steps = result.alignment_steps
        self.assertGreaterEqual(len(steps), 5)
        self.assertEqual({step.index for step in steps}, set(range(len(steps))))
        for step in steps:
            self.assertEqual(np.asarray(step.matrix).shape, (3, 3))
            self.assertIn(
                step.status, {"accepted", "rejected", "skipped", "ambiguous"}
            )
            self.assertTrue(step.reason)
            self.assertTrue(step.name)

    def test_patch_only_change_does_not_trigger_a_geometric_correction(self) -> None:
        reference = make_textured_scene()
        patched = reference.copy()
        cv2.rectangle(patched, (60, 50), (85, 75), (0, 0, 255), -1)

        result = inspect_images(reference, patched)

        self.assertEqual(result.alignment_status, "not_needed")
        self.assertFalse(result.alignment_applied)
        self.assertGreater(result.final_comparison.changed_percentage, 1.0)
        self.assertEqual(
            result.changed_regions["regions"],
            [{"x": 60, "y": 50, "width": 26, "height": 26, "area_pixels": 676}],
        )

    def test_smooth_lighting_change_is_normalized_without_misalignment(self) -> None:
        reference = make_textured_scene()
        gradient = np.linspace(0.55, 1.15, reference.shape[1], dtype=np.float32)
        lighting = np.clip(
            reference.astype(np.float32) * gradient[None, :, None] + 12, 0, 255
        ).astype(np.uint8)

        result = inspect_images(reference, lighting)

        self.assertEqual(result.alignment_status, "not_needed")
        self.assertFalse(result.alignment_applied)
        self.assertIsNotNone(result.photometric)
        self.assertTrue(result.photometric.applied)
        self.assertGreater(result.photometric.improvement_relative, 0.5)
        self.assertLess(result.final_comparison.changed_percentage, 1.0)
        self.assertFalse(result.method_disagreement)

    def test_colour_only_change_stays_visible_as_a_change_region(self) -> None:
        reference = make_textured_scene()
        recoloured = reference.copy()
        recoloured[:, :, 0] = np.clip(
            recoloured[:, :, 0].astype(np.int16) + 50, 0, 255
        )

        result = inspect_images(reference, recoloured)

        self.assertEqual(result.alignment_status, "not_needed")
        self.assertFalse(result.alignment_applied)
        self.assertGreater(result.final_comparison.changed_percentage, 1.0)
        self.assertGreater(result.changed_regions["region_count"], 0)
        self.assertTrue(result.pixel_change_indicated)
        self.assertEqual(result.indication_agreement, "pixel_only_indicates_change")

    def test_repetitive_grid_remains_ambiguous(self) -> None:
        grid = np.zeros((160, 200, 3), dtype=np.uint8)
        for y in range(0, 160, 20):
            for x in range(0, 200, 20):
                if ((x // 20) + (y // 20)) % 2 == 0:
                    cv2.rectangle(
                        grid, (x, y), (x + 19, y + 19), (220, 220, 220), -1
                    )
        shifted = np.roll(grid, 20, axis=1)

        result = inspect_images(grid, shifted)

        self.assertEqual(result.alignment_status, "ambiguous")
        self.assertFalse(result.alignment_applied)
        self.assertEqual(result.alignment_method, "none")
        self.assertTrue(
            any(
                "ambiguous" in warning.lower()
                for warning in result.uncertainty_warnings
            )
        )


class EvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reference = make_geometric_scene()
        rotated = cv2.warpAffine(
            reference,
            cv2.getRotationMatrix2D((80, 64), 15, 1.0),
            (reference.shape[1], reference.shape[0]),
        )
        cls.reference = reference
        cls.result = inspect_images(reference, rotated)
        cls.report = build_inspection_report(reference, rotated, cls.result)
        cls.serialized = json.dumps(cls.report, allow_nan=False)

    def test_report_uses_schema_two_and_keeps_measurements_separate(self) -> None:
        report = self.report

        self.assertEqual(report["schema_version"], 2)
        self.assertIn("alignment", report)
        self.assertIn("measurements", report)
        self.assertIn("agent", report)
        self.assertIn("function_trace", report)
        self.assertGreaterEqual(len(report["alignment"]["steps"]), 5)
        self.assertGreater(report["agent"]["iteration_count"], 0)
        self.assertGreater(len(report["function_trace"]), 0)
        self.assertEqual(
            report["measurements"]["lab_difference"]["threshold"],
            self.result.final_comparison.threshold,
        )
        self.assertIn("mean_ssim", report["measurements"]["ssim_comparison"])
        self.assertIn(
            "photometric_normalization",
            report["measurements"],
        )
        self.assertFalse(report["images"]["contents_included"])
        self.assertNotIn("difference_mask", self.serialized)
        self.assertNotIn("heatmap", self.serialized)
        self.assertNotIn("aligned_test_image", self.serialized)

    def test_every_agent_iteration_records_the_required_fields(self) -> None:
        iterations = self.report["agent"]["iterations"]

        self.assertGreater(len(iterations), 0)
        for iteration in iterations:
            self.assertIsInstance(iteration["index"], int)
            self.assertIsInstance(iteration["action"], str)
            self.assertTrue(iteration["action"])
            self.assertIsInstance(iteration["evidence"], list)
            self.assertTrue(iteration["evidence"])
            self.assertTrue(all(isinstance(item, str) for item in iteration["evidence"]))
            self.assertIsInstance(iteration["outcome"], str)
            self.assertTrue(iteration["outcome"])
            self.assertIsInstance(iteration["uncertainty"], str)
            self.assertTrue(iteration["uncertainty"])

        decided_actions = {
            iteration["action"]
            for iteration in iterations
            if iteration["action"] != "stop"
        }
        self.assertIn("fix_orientation", decided_actions)
        self.assertIn("compute_secondary_comparison", decided_actions)

    def test_function_trace_names_the_opencv_calls_used(self) -> None:
        stages = {
            entry["stage"]: entry["functions"] for entry in self.report["function_trace"]
        }

        self.assertIn("translation", stages)
        self.assertIn("cv2.phaseCorrelate", stages["translation"])
        self.assertIn("similarity_feature_matching", stages)
        self.assertIn("cv2.SIFT_create", stages["similarity_feature_matching"])
        self.assertIn("edge_comparison", stages)
        self.assertIn("cv2.Canny", stages["edge_comparison"])
        self.assertIn("ssim_comparison", stages)
        self.assertIn("region_detection", stages)

    def test_report_keeps_uncertainty_and_limitations_visible(self) -> None:
        self.assertTrue(self.report["warnings"])
        self.assertTrue(self.report["limitations"])
        self.assertTrue(self.report["selected_actions"])
        self.assertIn(
            "not calibrated defect probabilities",
            self.report["measurements"]["measurement_note"],
        )
        for step in self.report["alignment"]["steps"]:
            self.assertTrue(step["reason"])

    def test_report_never_claims_a_defect_or_a_confirmed_anomaly(self) -> None:
        forbidden = (
            "defect detected",
            "defect confirmed",
            "anomaly detected",
            "anomaly confirmed",
        )
        text_values = list(_iter_strings(self.report))
        text_values.extend(self.result.evidence)
        text_values.extend(self.result.decisions)
        text_values.extend(self.result.uncertainty_warnings)
        text_values.extend(self.result.limitations)

        combined = " ".join(text_values).lower()
        for phrase in forbidden:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, combined)


if __name__ == "__main__":
    unittest.main()
