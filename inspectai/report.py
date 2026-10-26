"""JSON report builder for InspectAI (schema version 2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .types import InspectionResult


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return _json_safe(np.asarray(matrix, dtype=np.float64))


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
    ssim_result = result.ssim_comparison
    photometric = result.photometric

    steps = []
    for step in result.alignment_steps:
        steps.append(
            {
                "index": step.index,
                "name": step.name,
                "model": step.model,
                "matrix": _matrix(step.matrix),
                "status": step.status,
                "reason": step.reason,
                "metrics": _json_safe(step.metrics),
                "keypoints_ref": _json_safe(step.keypoints_ref),
                "keypoints_test": _json_safe(step.keypoints_test),
                "keypoint_format": ["x", "y", "inlier"],
            }
        )
    iterations = []
    for iteration in result.agent_trace:
        iterations.append(
            {
                "index": iteration.index,
                "action": iteration.action,
                "evidence": _json_safe(iteration.evidence),
                "outcome": iteration.outcome,
                "uncertainty": iteration.uncertainty,
                "metrics": _json_safe(iteration.metrics),
            }
        )

    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "images": {
            "reference_dimensions": {"width": reference_width, "height": reference_height},
            "test_dimensions": {"width": test_width, "height": test_height},
            "contents_included": False,
        },
        "thresholds": {
            "lab_delta_initial": result.initial_comparison.threshold,
            "lab_delta_final": result.final_comparison.threshold,
            "lab_noise_mad_final": result.final_comparison.noise_mad,
            "pixel_change_indication_percentage": 1.0,
            "region_change_indication": (
                "largest region >= adaptive minimum area and >= 0.05% of valid pixels "
                "with mean Lab contrast >= max(8, 2.5 * threshold)"
            ),
            "minimum_region_area_pixels": result.final_comparison.minimum_region_area,
            "minimum_component_contrast": result.final_comparison.minimum_component_contrast,
            "region_component_filter": result.final_comparison.region_filter,
            "morphological_cleanup": result.final_comparison.cleanup,
            "canny_low_threshold": edge_result.canny_low,
            "canny_high_threshold": edge_result.canny_high,
            "canny_rule": "low = max(10, 0.4 * median intensity); high = max(low + 20, 1.2 * median intensity)",
            "edge_matching_tolerance_pixels": edge_result.tolerance_pixels,
            "edge_disagreement_indication_percentage": 10.0,
            "minimum_edge_pixels_per_image": 4,
            "ssim_similarity_threshold": ssim_result.threshold if ssim_result else None,
            "ssim_change_indication_percentage": 0.5,
            "phase_correlation_minimum_response": 0.2,
            "phase_peak_ratio_minimum": 2.0,
            "minimum_alignment_dimension_pixels": 16,
            "minimum_grayscale_texture_stddev": 1.0,
            "minimum_translation_pixels": 0.5,
            "feature_minimum_inliers": 8,
            "feature_minimum_inlier_ratio": 0.35,
            "feature_minimum_hull_fraction": 0.12,
            "minimum_alignment_ncc": 0.80,
            "alignment_ncc_gain_rule": (
                "minimum gain = min(0.015 + 0.01 * model_complexity, "
                "max(0.0005, 0.25 * (1 - current_ncc)))"
            ),
            "minimum_overlap_fraction": 0.35,
            "photometric_minimum_relative_improvement": 0.08,
            "cross_check_agreement_pixels": 2.0,
            "cross_check_disagreement_pixels": 5.0,
            "action_budget": len([item for item in result.agent_trace if item.action != "stop"]),
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
                "global_percentage_indicated": result.final_comparison.global_change_indicated,
                "region_indicated": result.final_comparison.region_change_indicated,
                "largest_region_area_pixels": result.final_comparison.largest_region_area,
                "largest_region_mean_lab_contrast": result.final_comparison.largest_region_contrast,
                "valid_pixels": result.final_comparison.valid_pixels,
                "filtered_component_count": result.final_comparison.filtered_region_count,
                "filtered_pixel_count": result.final_comparison.filtered_pixel_count,
            },
            "lab_difference": {
                "description": "CIE-Lab Euclidean difference (ΔE-style), colour-aware.",
                "threshold": result.final_comparison.threshold,
                "noise_median": result.final_comparison.noise_median,
                "noise_mad": result.final_comparison.noise_mad,
            },
            "edge_comparison": {
                "status": edge_result.status,
                "similarity_percentage": edge_result.edge_similarity_percentage,
                "disagreement_percentage": edge_result.edge_disagreement_percentage,
                "reference_edge_pixels": edge_result.reference_edge_pixels,
                "test_edge_pixels": edge_result.test_edge_pixels,
                "change_indicated": edge_result.change_indicated,
            },
            "ssim_comparison": {
                "status": ssim_result.status if ssim_result else "unavailable",
                "mean_ssim": ssim_result.mean_ssim if ssim_result else None,
                "changed_percentage": ssim_result.changed_percentage if ssim_result else None,
                "threshold": ssim_result.threshold if ssim_result else None,
                "change_indicated": ssim_result.change_indicated if ssim_result else None,
            },
            "photometric_normalization": (
                {
                    "applied": photometric.applied,
                    "status": photometric.status,
                    "reason": photometric.reason,
                    "model": photometric.model,
                    "residual_before": photometric.residual_before,
                    "residual_after": photometric.residual_after,
                    "improvement_relative": photometric.improvement_relative,
                    "channel_gain": photometric.channel_gain,
                    "channel_offset": photometric.channel_offset,
                    "channel_gradient": photometric.channel_gradient,
                }
                if photometric is not None
                else None
            ),
            "method_states": {
                "lab_difference_change_indicated": result.pixel_change_indicated,
                "canny_change_indicated": result.edge_change_indicated,
                "ssim_change_indicated": (
                    ssim_result.change_indicated if ssim_result else None
                ),
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
                "Lab difference, Canny edge overlap and SSIM measure different visual properties. "
                "They are heuristic comparison signals, not calibrated defect probabilities; "
                "Low edge similarity alone or low SSIM alone does not establish a defect."
            ),
            "changed_regions": result.changed_regions,
        },
        "alignment": {
            "method": result.alignment_method,
            "status": result.alignment_status,
            "applied": result.alignment_applied,
            "confidence": result.alignment_confidence,
            "estimated_shift_pixels": {
                "x": result.estimated_shift[0],
                "y": result.estimated_shift[1],
            },
            "estimated_rotation_degrees": result.estimated_rotation_degrees,
            "estimated_scale": result.estimated_scale,
            "response": result.alignment_response,
            "overlap_fraction": result.overlap_fraction,
            "test_image_resized": result.resized_test_image,
            "final_matrix": _matrix(result.final_matrix),
            "steps": steps,
        },
        # Top-level alias required by the v2 report contract.  It is the same
        # sequence as alignment.steps, without previews or image pixels.
        "alignment_steps": steps,
        "agent": {
            "iterations": iterations,
            "iteration_count": len(iterations),
        },
        "function_trace": _json_safe(result.function_trace),
        "selected_actions": result.decisions,
        "evidence": result.evidence,
        "warnings": result.uncertainty_warnings,
        "limitations": result.limitations,
    }
