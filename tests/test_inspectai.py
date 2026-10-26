import json
import unittest

import cv2
import numpy as np

from inspectai import (
    build_inspection_report,
    center_crop_pair,
    compare_edges,
    compare_images,
    decode_image,
    detect_changed_regions,
    inspect_images,
)


def make_reference() -> np.ndarray:
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (24, 20), (81, 76), (210, 210, 210), -1)
    cv2.circle(image, (116, 83), 19, (100, 170, 230), -1)
    cv2.line(image, (12, 108), (146, 96), (180, 70, 40), 5)
    return image


class ImageComparisonTests(unittest.TestCase):
    def test_empty_and_all_zero_masks_have_no_changed_regions(self) -> None:
        empty = detect_changed_regions(np.zeros((0, 0), dtype=np.uint8))
        no_changes = detect_changed_regions(np.zeros((40, 50), dtype=np.uint8))

        self.assertEqual(empty["region_count"], 0)
        self.assertEqual(empty["detected_region_count"], 0)
        self.assertEqual(no_changes["regions"], [])
        self.assertEqual(no_changes["filtered_region_count"], 0)

    def test_rectangular_change_returns_expected_bounding_box(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:45, 30:65] = 255

        result = detect_changed_regions(mask)

        self.assertEqual(result["region_count"], 1)
        self.assertEqual(
            result["regions"][0],
            {"x": 30, "y": 20, "width": 35, "height": 25, "area_pixels": 875},
        )

    def test_separated_changes_return_multiple_deterministically_sorted_regions(self) -> None:
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[10:20, 15:30] = 255
        mask[50:80, 70:100] = 255

        result = detect_changed_regions(mask)

        self.assertEqual(result["region_count"], 2)
        self.assertEqual([region["area_pixels"] for region in result["regions"]], [900, 150])
        self.assertEqual(
            [(region["x"], region["y"]) for region in result["regions"]],
            [(70, 50), (15, 10)],
        )

    def test_minimum_area_filters_tiny_components_and_reports_them(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        mask[10:15, 12:17] = 255

        result = detect_changed_regions(mask)

        self.assertEqual(result["minimum_area_pixels"], 25)
        self.assertEqual(result["detected_region_count"], 2)
        self.assertEqual(result["filtered_region_count"], 1)
        self.assertEqual(result["region_count"], 1)
        self.assertEqual(result["regions"][0]["area_pixels"], 25)

    def test_invalid_masks_and_minimum_area_raise_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            detect_changed_regions(None)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            detect_changed_regions(np.zeros((4, 4, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8 binary"):
            detect_changed_regions(np.zeros((4, 4), dtype=np.int16))
        invalid_values = np.zeros((4, 4), dtype=np.uint8)
        invalid_values[1, 1] = 1
        with self.assertRaisesRegex(ValueError, "0 or 255"):
            detect_changed_regions(invalid_values)
        with self.assertRaisesRegex(ValueError, "positive whole number"):
            detect_changed_regions(np.zeros((4, 4), dtype=np.uint8), minimum_area=0)

    def test_center_crop_pair_handles_landscape_and_portrait_without_mutation(self) -> None:
        landscape = np.full((120, 200, 3), 30, dtype=np.uint8)
        portrait = np.full((200, 120, 3), 90, dtype=np.uint8)
        landscape_before = landscape.copy()
        portrait_before = portrait.copy()

        cropped_reference, cropped_test = center_crop_pair(
            landscape, portrait, target_width=1, target_height=1
        )

        self.assertEqual(cropped_reference.shape, (120, 120, 3))
        self.assertEqual(cropped_test.shape, (120, 120, 3))
        self.assertFalse(np.shares_memory(cropped_reference, landscape))
        self.assertFalse(np.shares_memory(cropped_test, portrait))
        np.testing.assert_array_equal(landscape, landscape_before)
        np.testing.assert_array_equal(portrait, portrait_before)

    def test_center_crop_pair_rejects_invalid_target_dimensions(self) -> None:
        image = make_reference()

        with self.assertRaisesRegex(ValueError, "positive whole numbers"):
            center_crop_pair(image, image, target_width=0, target_height=1)
        with self.assertRaisesRegex(ValueError, "too large"):
            center_crop_pair(image, image, target_width=1000, target_height=1)

    def test_identical_images_have_no_changed_pixels(self) -> None:
        image = make_reference()

        result = compare_images(image, image.copy())

        self.assertEqual(result.changed_percentage, 0.0)
        self.assertEqual(cv2.countNonZero(result.difference_mask), 0)

    def test_identical_images_have_indicating_measurement_agreement(self) -> None:
        image = make_reference()

        result = inspect_images(image, image.copy())

        self.assertEqual(result.independent_comparison.edge_similarity_percentage, 100.0)
        self.assertEqual(result.independent_comparison.edge_disagreement_percentage, 0.0)
        self.assertEqual(result.indication_agreement, "neither_indicates_change")
        self.assertFalse(result.method_disagreement)

    def test_changed_region_is_reported(self) -> None:
        reference = make_reference()
        changed = reference.copy()
        cv2.rectangle(changed, (40, 40), (65, 65), (0, 0, 255), -1)

        result = compare_images(reference, changed)

        self.assertGreater(result.changed_percentage, 1.0)
        self.assertGreater(cv2.countNonZero(result.heatmap[:, :, 0]), 0)

    def test_localized_change_can_disagree_with_edge_measurement(self) -> None:
        reference = make_reference()
        changed = reference.copy()
        cv2.rectangle(changed, (40, 40), (65, 65), (0, 0, 255), -1)

        result = inspect_images(reference, changed)

        self.assertTrue(result.pixel_change_indicated)
        self.assertFalse(result.edge_change_indicated)
        self.assertEqual(result.indication_agreement, "pixel_only_indicates_change")
        self.assertTrue(result.method_disagreement)

    def test_shifted_image_is_aligned_and_recompared(self) -> None:
        reference = make_reference()
        shifted = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 5], [0, 1, -3]]),
            (reference.shape[1], reference.shape[0]),
        )

        result = inspect_images(reference, shifted)

        self.assertTrue(result.alignment_applied)
        self.assertEqual(result.alignment_status, "applied")
        self.assertGreater(result.initial_comparison.changed_percentage, 5.0)
        self.assertLess(result.final_comparison.changed_percentage, 1.0)

    def test_small_rotation_is_aligned_and_recompared(self) -> None:
        reference = make_reference()
        rotation = cv2.getRotationMatrix2D((80, 64), 3, 1)
        rotated = cv2.warpAffine(reference, rotation, (160, 128))

        result = inspect_images(reference, rotated)

        self.assertTrue(result.alignment_applied)
        self.assertEqual(result.alignment_status, "applied")
        self.assertAlmostEqual(result.estimated_rotation_degrees, 3.0, delta=0.5)
        self.assertLess(result.final_comparison.changed_percentage, 1.0)

    def test_blank_images_are_treated_as_ambiguous(self) -> None:
        blank = np.zeros((128, 160, 3), dtype=np.uint8)

        result = inspect_images(blank, blank.copy())

        self.assertFalse(result.alignment_applied)
        self.assertEqual(result.alignment_status, "ambiguous")
        self.assertEqual(result.alignment_response, 0.0)
        self.assertEqual(result.independent_comparison.status, "insufficient_evidence")
        self.assertIsNone(result.method_disagreement)
        self.assertTrue(
            any("alignment was ambiguous" in warning.lower() for warning in result.uncertainty_warnings)
        )

    def test_unrelated_images_can_both_indicate_change_with_low_edge_similarity(self) -> None:
        reference = make_reference()
        unrelated = np.zeros_like(reference)
        cv2.rectangle(unrelated, (5, 5), (150, 15), (255, 255, 255), -1)
        cv2.circle(unrelated, (82, 65), 32, (150, 150, 150), -1)
        cv2.line(unrelated, (0, 0), (159, 127), (230, 60, 190), 7)

        result = inspect_images(reference, unrelated)

        self.assertGreater(result.final_comparison.changed_percentage, 1.0)
        self.assertTrue(result.pixel_change_indicated)
        self.assertTrue(result.edge_change_indicated)
        self.assertLess(result.independent_comparison.edge_similarity_percentage, 50.0)
        self.assertEqual(result.indication_agreement, "both_indicate_change")
        self.assertFalse(result.method_disagreement)
        self.assertEqual(result.alignment_status, "ambiguous")
        self.assertTrue(
            any("alignment was ambiguous" in warning.lower() for warning in result.uncertainty_warnings)
        )
        report = build_inspection_report(reference, unrelated, result)
        self.assertFalse(report["measurements"]["methods_disagree"])
        self.assertEqual(
            report["measurements"]["indication_agreement"]["summary"],
            "both_indicate_change",
        )
        self.assertTrue(report["measurements"]["indication_agreement"]["indications_agree"])
        self.assertIn("Low edge similarity alone", report["measurements"]["measurement_note"])

    def test_brightness_change_is_normalized_instead_of_misaligned(self) -> None:
        reference = make_reference()
        brighter = cv2.convertScaleAbs(reference, alpha=1.0, beta=45)

        result = inspect_images(reference, brighter)

        self.assertFalse(result.alignment_applied)
        self.assertEqual(result.alignment_status, "not_needed")
        self.assertGreater(result.initial_comparison.changed_percentage, 50.0)
        self.assertLess(result.final_comparison.changed_percentage, 5.0)
        self.assertIsNotNone(result.photometric)
        self.assertTrue(result.photometric.applied)
        self.assertEqual(result.independent_comparison.status, "measured")
        self.assertGreater(result.independent_comparison.edge_similarity_percentage, 90.0)
        self.assertFalse(result.method_disagreement)

    def test_dimension_mismatch_is_resized_before_comparison(self) -> None:
        reference = make_reference()
        smaller = cv2.resize(reference, (80, 64))

        result = inspect_images(reference, smaller)

        self.assertTrue(result.resized_test_image)
        self.assertEqual(result.final_comparison.difference_mask.shape, reference.shape[:2])

    def test_different_aspect_ratios_are_rejected(self) -> None:
        reference = make_reference()
        different_ratio = np.zeros((128, 120, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "aspect ratios differ"):
            inspect_images(reference, different_ratio)

    def test_edge_comparison_rejects_mismatched_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching dimensions"):
            compare_edges(make_reference(), np.zeros((64, 80, 3), dtype=np.uint8))

    def test_report_is_json_safe_and_excludes_image_pixels(self) -> None:
        reference = make_reference()
        test = reference.copy()
        result = inspect_images(reference, test)

        report = build_inspection_report(reference, test, result)
        serialized = json.dumps(report, allow_nan=False)

        self.assertEqual(report["images"]["reference_dimensions"], {"width": 160, "height": 128})
        self.assertFalse(report["images"]["contents_included"])
        self.assertIn("thresholds", report)
        self.assertIn("measurements", report)
        self.assertIn("changed_regions", report["measurements"])
        self.assertIn("alignment", report)
        self.assertIn("selected_actions", report)
        self.assertIn("warnings", report)
        self.assertNotIn("difference_mask", serialized)
        self.assertNotIn("heatmap", serialized)
        self.assertNotIn("reference_pixels", serialized)
        self.assertNotIn("annotated_image", serialized)

    def test_report_builder_rejects_empty_image_data(self) -> None:
        reference = make_reference()
        result = inspect_images(reference, reference.copy())

        with self.assertRaisesRegex(ValueError, "pixel data"):
            build_inspection_report(np.empty((0, 0, 3), dtype=np.uint8), reference, result)

    def test_invalid_image_bytes_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            decode_image(b"not an image")


if __name__ == "__main__":
    unittest.main()