"""Image comparison and a small, evidence-reporting inspection workflow."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import gcd

import cv2
import numpy as np


@dataclass
class ComparisonResult:
    changed_percentage: float
    difference_mask: np.ndarray
    heatmap: np.ndarray


@dataclass
class EdgeComparisonResult:
    edge_similarity_percentage: float | None
    edge_disagreement_percentage: float | None
    reference_edge_pixels: int
    test_edge_pixels: int
    tolerance_pixels: int
    status: str


@dataclass
class InspectionResult:
    initial_comparison: ComparisonResult
    final_comparison: ComparisonResult
    changed_regions: dict[str, object]
    independent_comparison: EdgeComparisonResult
    alignment_applied: bool
    alignment_status: str
    alignment_method: str
    estimated_shift: tuple[float, float]
    alignment_response: float
    resized_test_image: bool
    pixel_change_indicated: bool
    edge_change_indicated: bool | None
    method_disagreement: bool | None
    indication_agreement: str
    evidence: list[str]
    decisions: list[str]
    uncertainty_warnings: list[str]
    limitations: list[str]


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR image or raise a useful error."""
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("This file could not be decoded as an image.")
    return image


def center_crop_pair(
    reference: np.ndarray,
    test: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop two images to their largest shared exact target aspect ratio."""
    if (
        reference is None
        or test is None
        or reference.size == 0
        or test.size == 0
        or reference.ndim < 2
        or test.ndim < 2
    ):
        raise ValueError("Both images must contain pixel data for center cropping.")
    if (
        not isinstance(target_width, (int, np.integer))
        or not isinstance(target_height, (int, np.integer))
        or target_width <= 0
        or target_height <= 0
    ):
        raise ValueError("Target crop width and height must be positive whole numbers.")

    divisor = gcd(int(target_width), int(target_height))
    unit_width = int(target_width) // divisor
    unit_height = int(target_height) // divisor
    reference_height, reference_width = reference.shape[:2]
    test_height, test_width = test.shape[:2]
    scale = min(
        reference_width // unit_width,
        reference_height // unit_height,
        test_width // unit_width,
        test_height // unit_height,
    )
    if scale < 1:
        raise ValueError("The selected crop ratio is too large for one or both images.")

    crop_width = unit_width * scale
    crop_height = unit_height * scale

    def crop_center(image: np.ndarray) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        return image[top : top + crop_height, left : left + crop_width].copy()

    return crop_center(reference), crop_center(test)


def compare_images(
    reference: np.ndarray,
    test: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> ComparisonResult:
    """Compare same-sized BGR images using a simple grayscale threshold."""
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        raise ValueError("Both images must contain pixel data.")
    if reference.shape[:2] != test.shape[:2]:
        raise ValueError("Images must have matching dimensions before comparison.")

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(reference_gray, test_gray)
    _, mask = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    if valid_mask is None:
        valid_mask = np.full(mask.shape, 255, dtype=np.uint8)
    elif valid_mask.shape != mask.shape:
        raise ValueError("The valid-area mask must match the image dimensions.")
    else:
        valid_mask = np.where(valid_mask > 0, 255, 0).astype(np.uint8)

    mask[valid_mask == 0] = 0
    valid_pixels = int(np.count_nonzero(valid_mask))
    changed_pixels = int(np.count_nonzero(mask))
    changed_percentage = 100.0 * changed_pixels / valid_pixels if valid_pixels else 0.0

    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
    heatmap[(difference <= 30) | (valid_mask == 0)] = 0
    return ComparisonResult(changed_percentage, mask, heatmap)


def detect_changed_regions(
    difference_mask: np.ndarray,
    minimum_area: int = 25,
) -> dict[str, object]:
    """Find 8-connected changed pixel regions in a binary uint8 mask.

    Regions smaller than ``minimum_area`` pixels are excluded from ``regions``
    and counted in ``filtered_region_count``.
    """
    if not isinstance(difference_mask, np.ndarray) or difference_mask.ndim != 2:
        raise ValueError("Difference mask must be a two-dimensional NumPy array.")
    if difference_mask.dtype != np.uint8:
        raise ValueError("Difference mask must use uint8 binary values of 0 or 255.")
    if (
        isinstance(minimum_area, (bool, np.bool_))
        or not isinstance(minimum_area, (int, np.integer))
        or minimum_area < 1
    ):
        raise ValueError("Minimum region area must be a positive whole number of pixels.")
    if difference_mask.size and np.any((difference_mask != 0) & (difference_mask != 255)):
        raise ValueError("Difference mask must use uint8 binary values of 0 or 255.")

    minimum_area = int(minimum_area)
    if difference_mask.size == 0:
        detected_count = 0
        regions: list[dict[str, int]] = []
    else:
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            difference_mask, connectivity=8
        )
        detected_count = component_count - 1
        regions = []
        for component_index in range(1, component_count):
            x, y, width, height, area = (
                int(value) for value in stats[component_index]
            )
            if area < minimum_area:
                continue
            regions.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area_pixels": area,
                }
            )

    regions.sort(key=lambda region: (-region["area_pixels"], region["y"], region["x"]))
    return {
        "regions": regions,
        "region_count": len(regions),
        "detected_region_count": detected_count,
        "filtered_region_count": detected_count - len(regions),
        "minimum_area_pixels": minimum_area,
    }


def compare_edges(
    reference: np.ndarray,
    test: np.ndarray,
    valid_mask: np.ndarray | None = None,
    tolerance_pixels: int = 2,
) -> EdgeComparisonResult:
    """Measure tolerant Canny-edge agreement independently of pixel intensity."""
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        raise ValueError("Both images must contain pixel data.")
    if reference.shape[:2] != test.shape[:2]:
        raise ValueError("Images must have matching dimensions before edge comparison.")
    if tolerance_pixels < 0:
        raise ValueError("Edge tolerance must be zero or greater.")

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    reference_edges = cv2.Canny(reference_gray, 50, 150)
    test_edges = cv2.Canny(test_gray, 50, 150)
    if valid_mask is None:
        valid_mask = np.full(reference_edges.shape, 255, dtype=np.uint8)
    elif valid_mask.shape != reference_edges.shape:
        raise ValueError("The valid-area mask must match the image dimensions.")
    else:
        valid_mask = np.where(valid_mask > 0, 255, 0).astype(np.uint8)

    reference_edges[valid_mask == 0] = 0
    test_edges[valid_mask == 0] = 0
    reference_count = int(cv2.countNonZero(reference_edges))
    test_count = int(cv2.countNonZero(test_edges))
    if reference_count < 4 or test_count < 4:
        return EdgeComparisonResult(
            edge_similarity_percentage=None,
            edge_disagreement_percentage=None,
            reference_edge_pixels=reference_count,
            test_edge_pixels=test_count,
            tolerance_pixels=tolerance_pixels,
            status="insufficient_evidence",
        )

    distance_to_reference = cv2.distanceTransform(
        cv2.bitwise_not(reference_edges), cv2.DIST_L2, 3
    )
    distance_to_test = cv2.distanceTransform(
        cv2.bitwise_not(test_edges), cv2.DIST_L2, 3
    )
    matched_reference = int(
        np.count_nonzero((reference_edges > 0) & (distance_to_test <= tolerance_pixels))
    )
    matched_test = int(
        np.count_nonzero((test_edges > 0) & (distance_to_reference <= tolerance_pixels))
    )
    precision = matched_test / test_count
    recall = matched_reference / reference_count
    edge_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    similarity = 100.0 * edge_f1
    return EdgeComparisonResult(
        edge_similarity_percentage=similarity,
        edge_disagreement_percentage=100.0 - similarity,
        reference_edge_pixels=reference_count,
        test_edge_pixels=test_count,
        tolerance_pixels=tolerance_pixels,
        status="measured",
    )


def inspect_images(reference: np.ndarray, test: np.ndarray) -> InspectionResult:
    """Compare images and apply translation only when evidence supports it."""
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        raise ValueError("Both images must contain pixel data.")

    reference_height, reference_width = reference.shape[:2]
    test_height, test_width = test.shape[:2]
    resized_test_image = (reference_height, reference_width) != (test_height, test_width)
    evidence: list[str] = []
    decisions: list[str] = []

    if resized_test_image:
        evidence.append(
            f"Image dimensions differ: reference {reference_width}x{reference_height}, "
            f"test {test_width}x{test_height}."
        )
        if test_width * reference_height != reference_width * test_height:
            raise ValueError(
                "Image aspect ratios differ. Crop or geometrically align the images before comparison."
            )
        interpolation = (
            cv2.INTER_AREA
            if test_width * test_height > reference_width * reference_height
            else cv2.INTER_LINEAR
        )
        test = cv2.resize(test, (reference_width, reference_height), interpolation=interpolation)
        decisions.append("Resized the test image to the reference dimensions before comparison.")

    initial = compare_images(reference, test)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32)
    minimum_dimension = min(reference_width, reference_height)
    maximum_shift = min(max(5.0, minimum_dimension * 0.2), minimum_dimension / 2)
    alignment_method = "phase_correlation_translation"
    alignment_status = "ambiguous"
    alignment_response = 0.0
    shift_x = 0.0
    shift_y = 0.0
    alignment_applied = False
    comparison_test = test
    final_valid_mask = None

    texture_level = min(float(reference_gray.std()), float(test_gray.std()))
    if minimum_dimension < 16 or texture_level < 1.0:
        decisions.append(
            "Translation alignment was skipped because the images are too small or have insufficient texture."
        )
        evidence.append(f"Minimum grayscale texture standard deviation: {texture_level:.3f}.")
    else:
        window = cv2.createHanningWindow((reference_width, reference_height), cv2.CV_32F)
        try:
            shift, response = cv2.phaseCorrelate(reference_gray, test_gray, window)
            shift_x, shift_y = float(shift[0]), float(shift[1])
            alignment_response = float(response)
        except cv2.error as error:
            decisions.append("Translation alignment was skipped because OpenCV could not estimate motion.")
            evidence.append(f"OpenCV phase-correlation error: {error}")

        shift_size = float(np.hypot(shift_x, shift_y))
        finite_estimate = np.isfinite((shift_x, shift_y, alignment_response)).all()
        evidence.append(
            f"Phase correlation estimated a shift of ({shift_x:.2f}, {shift_y:.2f}) pixels "
            f"with response {alignment_response:.3f}."
        )

        if not finite_estimate or alignment_response < 0.2:
            alignment_status = "ambiguous"
            decisions.append("No alignment correction was applied because the motion estimate was weak or invalid.")
        elif shift_size < 1.0:
            alignment_status = "not_needed"
            decisions.append("No alignment correction was applied because the estimated shift was under 1 pixel.")
        elif shift_size > maximum_shift:
            alignment_status = "ambiguous"
            decisions.append("No alignment correction was applied because the estimated shift was unusually large.")
        else:
            transform = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
            aligned_test = cv2.warpAffine(
                test,
                transform,
                (reference_width, reference_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            source_validity = np.full((reference_height, reference_width), 255, dtype=np.uint8)
            valid_area = cv2.warpAffine(
                source_validity,
                transform,
                (reference_width, reference_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            overlap_before = compare_images(reference, test, valid_area)
            overlap_after = compare_images(reference, aligned_test, valid_area)
            improvement = overlap_before.changed_percentage - overlap_after.changed_percentage
            required_improvement = max(0.5, overlap_before.changed_percentage * 0.1)
            evidence.append(
                f"On the same overlap, changed pixels were {overlap_before.changed_percentage:.2f}% "
                f"before and {overlap_after.changed_percentage:.2f}% after the candidate shift "
                f"(improvement {improvement:.2f} percentage points; required "
                f"{required_improvement:.2f})."
            )
            if improvement >= required_improvement:
                initial = overlap_before
                final = overlap_after
                comparison_test = aligned_test
                final_valid_mask = valid_area
                alignment_status = "applied"
                alignment_applied = True
                decisions.append(
                    "The candidate translation reduced the changed-pixel percentage enough on "
                    "the shared overlap, so it was applied."
                )
            else:
                final = initial
                alignment_status = "rejected"
                decisions.append(
                    "The candidate translation was rejected because it did not improve the "
                    "changed-pixel percentage enough on the shared overlap."
                )

    if not alignment_applied:
        final = initial

    changed_regions = detect_changed_regions(final.difference_mask)
    independent_comparison = compare_edges(
        reference, comparison_test, final_valid_mask
    )
    pixel_change_indicated = final.changed_percentage >= 1.0
    edge_change_indicated = None
    method_disagreement = None
    indication_agreement = "insufficient_edge_evidence"
    uncertainty_warnings = [
        "Pixel and edge measurements are heuristic comparison signals, not calibrated defect probabilities."
    ]
    if alignment_status == "ambiguous":
        uncertainty_warnings.append(
            "Image alignment was ambiguous, so no alignment correction was applied. Review the images before interpreting the comparison."
        )
    evidence.append(
        f"Canny edge comparison: {independent_comparison.reference_edge_pixels} reference and "
        f"{independent_comparison.test_edge_pixels} test edge pixels; "
        f"{independent_comparison.tolerance_pixels}-pixel matching tolerance."
    )
    if independent_comparison.status == "measured":
        edge_change_indicated = independent_comparison.edge_disagreement_percentage >= 10.0
        method_disagreement = pixel_change_indicated != edge_change_indicated
        if pixel_change_indicated and edge_change_indicated:
            indication_agreement = "both_indicate_change"
        elif pixel_change_indicated:
            indication_agreement = "pixel_only_indicates_change"
        elif edge_change_indicated:
            indication_agreement = "edge_only_indicates_change"
        else:
            indication_agreement = "neither_indicates_change"
        evidence.append(
            f"Edge similarity was {independent_comparison.edge_similarity_percentage:.2f}% "
            f"(edge disagreement {independent_comparison.edge_disagreement_percentage:.2f}%)."
        )
        if method_disagreement:
            uncertainty_warnings.append(
                "Pixel-difference and edge-based indications disagree; inspect the images and both measurements."
            )
    else:
        uncertainty_warnings.append(
            "The edge comparison had fewer than four edge pixels in at least one image, so agreement is not scored."
        )
        evidence.append("Edge comparison status: insufficient edges for a stable score.")

    if not resized_test_image:
        decisions.insert(0, "Image dimensions already matched; the initial pixel comparison ran directly.")

    limitations = [
        "Pixel differences are sensitive to lighting, camera position, focus, compression, and sensor noise.",
        "A changed-pixel percentage does not determine whether a real-world defect is present or important.",
        "This workflow checks translation only; it does not correct rotation, perspective, or non-rigid changes.",
        "The phase-correlation response is a matching signal, not a calibrated confidence or defect probability.",
        "The Canny edge score uses fixed thresholds and a pixel tolerance; it is not a calibrated defect score.",
    ]
    if resized_test_image:
        limitations.append(
            "Resizing can distort geometry when the source aspect ratios differ; review the images before trusting the comparison."
        )

    return InspectionResult(
        initial_comparison=initial,
        final_comparison=final,
        changed_regions=changed_regions,
        independent_comparison=independent_comparison,
        alignment_applied=alignment_applied,
        alignment_status=alignment_status,
        alignment_method=alignment_method,
        estimated_shift=(shift_x, shift_y),
        alignment_response=alignment_response,
        resized_test_image=resized_test_image,
        pixel_change_indicated=pixel_change_indicated,
        edge_change_indicated=edge_change_indicated,
        method_disagreement=method_disagreement,
        indication_agreement=indication_agreement,
        evidence=evidence,
        decisions=decisions,
        uncertainty_warnings=uncertainty_warnings,
        limitations=limitations,
    )


def build_inspection_report(
    reference: np.ndarray,
    test: np.ndarray,
    result: InspectionResult,
) -> dict[str, object]:
    """Build a JSON-safe report containing measurements and metadata, not image data."""
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        raise ValueError("Both images must contain pixel data to build a report.")

    reference_height, reference_width = reference.shape[:2]
    test_height, test_width = test.shape[:2]
    edge_result = result.independent_comparison

    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "images": {
            "reference_dimensions": {"width": reference_width, "height": reference_height},
            "test_dimensions": {"width": test_width, "height": test_height},
            "contents_included": False,
        },
        "thresholds": {
            "pixel_difference_intensity": 30,
            "pixel_change_indication_percentage": 1.0,
            "canny_low_threshold": 50,
            "canny_high_threshold": 150,
            "edge_matching_tolerance_pixels": edge_result.tolerance_pixels,
            "edge_disagreement_indication_percentage": 10.0,
            "minimum_edge_pixels_per_image": 4,
            "phase_correlation_minimum_response": 0.2,
            "minimum_alignment_dimension_pixels": 16,
            "minimum_grayscale_texture_stddev": 1.0,
            "minimum_translation_pixels": 1.0,
            "maximum_translation_rule": "min(max(5, 20% of shorter image dimension), half of shorter dimension)",
            "minimum_alignment_improvement_percentage_points": 0.5,
            "minimum_relative_alignment_improvement": 0.1,
        },
        "measurements": {
            "pixel_difference": {
                "initial_changed_percentage": result.initial_comparison.changed_percentage,
                "initial_changed_pixel_count": int(
                    cv2.countNonZero(result.initial_comparison.difference_mask)
                ),
                "final_changed_percentage": result.final_comparison.changed_percentage,
                "final_changed_pixel_count": int(
                    cv2.countNonZero(result.final_comparison.difference_mask)
                ),
                "change_indicated": result.pixel_change_indicated,
            },
            "edge_comparison": {
                "status": edge_result.status,
                "similarity_percentage": edge_result.edge_similarity_percentage,
                "disagreement_percentage": edge_result.edge_disagreement_percentage,
                "reference_edge_pixels": edge_result.reference_edge_pixels,
                "test_edge_pixels": edge_result.test_edge_pixels,
                "change_indicated": result.edge_change_indicated,
            },
            "methods_disagree": result.method_disagreement,
            "indication_agreement": {
                "summary": result.indication_agreement,
                "pixel_change_indicated": result.pixel_change_indicated,
                "edge_change_indicated": result.edge_change_indicated,
                "indications_agree": (
                    None if result.method_disagreement is None else not result.method_disagreement
                ),
            },
            "measurement_note": (
                "Pixel changed percentage and Canny edge similarity measure different properties "
                "and are not directly comparable. Low edge similarity alone does not establish a defect."
            ),
            "changed_regions": result.changed_regions,
        },
        "alignment": {
            "method": result.alignment_method,
            "status": result.alignment_status,
            "applied": result.alignment_applied,
            "estimated_shift_pixels": {
                "x": finite_or_none(result.estimated_shift[0]),
                "y": finite_or_none(result.estimated_shift[1]),
            },
            "response": finite_or_none(result.alignment_response),
            "test_image_resized": result.resized_test_image,
        },
        "selected_actions": result.decisions,
        "evidence": result.evidence,
        "warnings": result.uncertainty_warnings,
        "limitations": result.limitations,
    }