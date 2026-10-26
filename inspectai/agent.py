"""The InspectAI inspection loop.

The loop keeps explicit state, tries actions in order of increasing complexity,
records ``{action, evidence, outcome, uncertainty}`` for every iteration, and
stops when the evidence says it should.  It measures visual differences; it
does not identify real-world defects.
"""

from __future__ import annotations

import math
from math import gcd

import cv2
import numpy as np

from .align import (
    MODEL_ORDER,
    assess_orientation,
    candidate_acceptance,
    clockwise_quarter_matrix,
    cross_check,
    decompose_matrix,
    estimate_feature_transform,
    log_polar_candidate,
    mean_corner_displacement,
    phase_correlation,
    refine_with_ecc,
    rotate_clockwise_quarter,
    score_alignment,
    to_homography,
    validate_transform,
    warp_image,
    warp_mask,
)
from .compare import (
    compare_edges,
    compare_images,
    compare_ssim,
    fit_photometric_normalization,
)
from .regions import adaptive_minimum_area, detect_changed_regions
from .types import (
    AgentIteration,
    AlignmentStep,
    ComparisonResult,
    InspectionResult,
    TransformCandidate,
)

MIN_TEXTURE_STD = 1.0
MIN_ALIGNMENT_DIMENSION = 16
DEFAULT_MAX_ACTIONS = 10

LIMITATIONS = [
    "This tool measures visual differences. It does not identify real-world defects.",
    "Changed-pixel percentages, edge scores and SSIM values are heuristic comparison signals, not calibrated probabilities.",
    "Rotated, scaled, affine and perspective alignment are estimated heuristically; a transform can still be wrong in subtle ways.",
    "Photometric normalization can absorb smooth lighting changes, but it can also suppress a real change that looks like lighting.",
    "Repetitive textures and symmetric parts can make alignment ambiguous; InspectAI flags this instead of silently correcting it.",
    "Interpolation, focus, compression and sensor noise can create differences that have nothing to do with the scene content.",
    "The Canny edge comparison uses automatically derived thresholds and a pixel tolerance; it is not a calibrated defect score.",
    "This pass is local only. It does not include AWS deployment.",
]


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


def _scale_ref_to_source(reference_width: int, reference_height: int, source_width: int, source_height: int) -> np.ndarray:
    return np.array(
        [
            [source_width / reference_width, 0.0, 0.0],
            [0.0, source_height / reference_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _resize_interpolation(source_width: int, source_height: int, target_width: int, target_height: int) -> int:
    if source_width * source_height > target_width * target_height:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


def _classify_residual(
    reference: np.ndarray,
    test: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """Classify *why* a residual exists using visible, inspectable heuristics.

    The labels are heuristic categories, not calibrated classifiers.
    """
    comparison = compare_images(reference, test, valid_mask)
    changed = (comparison.difference_mask > 0) & (valid_mask > 0)
    changed_pixels = int(np.count_nonzero(changed))
    if changed_pixels == 0:
        return {
            "label": "no_clear_residual",
            "evidence": [
                "No changed pixels survived the adaptive Lab threshold and mask cleanup.",
            ],
            "metrics": {
                "changed_percentage": comparison.changed_percentage,
                "edge_adjacent_fraction": 0.0,
                "smooth_variance_explained": 0.0,
                "regions": comparison.minimum_region_area and 0,
            },
            "comparison": comparison,
        }

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    median = float(np.median(reference_gray))
    low = int(np.clip(max(10.0, 0.4 * median), 10, 160))
    high = int(np.clip(max(low + 20.0, 1.2 * median), low + 20, 255))
    reference_edges = cv2.Canny(reference_gray, low, high)
    distance = cv2.distanceTransform(cv2.bitwise_not(reference_edges), cv2.DIST_L2, 3)
    edge_fraction = float(np.mean(distance[changed] <= 3.0)) if changed_pixels else 0.0

    signed = reference_gray.astype(np.float32) - test_gray.astype(np.float32)
    kernel = min(51, max(15, (min(reference.shape[:2]) // 3) | 1))
    smooth = cv2.GaussianBlur(signed, (kernel, kernel), 0)
    valid_values = valid_mask > 0
    signed_variance = float(np.var(signed[valid_values]))
    smooth_variance = float(np.var(smooth[valid_values]))
    variance_explained = smooth_variance / signed_variance if signed_variance > 1e-6 else 0.0
    median_signed = float(np.median(signed[valid_values])) if np.any(valid_values) else 0.0

    if variance_explained >= 0.55 and abs(median_signed) >= 1.0 and edge_fraction < 0.75:
        label = "lighting_like"
        reason = (
            f"A large-kernel low-pass explains {variance_explained * 100:.0f}% of the signed "
            f"difference variance, and the difference is broadly smooth "
            f"(median signed difference {median_signed:.2f})."
        )
    elif edge_fraction >= 0.5 and comparison.changed_percentage > 0.5:
        label = "misalignment_like"
        reason = (
            f"{edge_fraction * 100:.0f}% of changed pixels lie within 3 px of reference edges, "
            "which is consistent with residual geometric misalignment."
        )
    elif comparison.region_change_indicated:
        label = "real_change_like"
        reason = (
            f"Compact changed regions survive away from edges "
            f"(largest region {comparison.largest_region_area} px, mean Lab contrast "
            f"{comparison.largest_region_contrast:.1f})."
        )
    else:
        label = "unclear"
        reason = "The residual does not clearly fit a misalignment, lighting or compact-change pattern."
    return {
        "label": label,
        "evidence": [
            reason,
            f"Changed pixels {comparison.changed_percentage:.2f}%; edge-adjacent fraction "
            f"{edge_fraction:.2f}; smooth variance explained {variance_explained:.2f}.",
        ],
        "metrics": {
            "changed_percentage": comparison.changed_percentage,
            "edge_adjacent_fraction": edge_fraction,
            "smooth_variance_explained": variance_explained,
            "median_signed_difference": median_signed,
            "largest_region_area": comparison.largest_region_area,
            "canny_low": low,
            "canny_high": high,
        },
        "comparison": comparison,
    }


def inspect_images(
    reference: np.ndarray,
    test: np.ndarray,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> InspectionResult:
    """Run the evidence-driven alignment and comparison loop."""
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        raise ValueError("Both images must contain pixel data.")
    if reference.ndim != 3 or test.ndim != 3:
        raise ValueError("Both images must be colour images with three channels.")
    if max_actions < 1:
        raise ValueError("The action budget must be at least one.")

    reference_height, reference_width = reference.shape[:2]
    test_height, test_width = test.shape[:2]
    evidence: list[str] = []
    decisions: list[str] = []
    uncertainty_warnings: list[str] = []
    trace: list[AgentIteration] = []
    function_trace: list[dict] = []
    steps: list[AlignmentStep] = []

    def record(
        action: str,
        action_evidence: list[str],
        outcome: str,
        uncertainty: str,
        metrics: dict | None = None,
    ) -> None:
        trace.append(
            AgentIteration(
                index=len(trace),
                action=action,
                evidence=list(action_evidence),
                outcome=outcome,
                uncertainty=uncertainty,
                metrics=metrics or {},
            )
        )

    def add_step(
        name: str,
        model: str,
        matrix: np.ndarray,
        status: str,
        reason: str,
        metrics: dict | None = None,
        preview: np.ndarray | None = None,
        keypoints_ref: list | None = None,
        keypoints_test: list | None = None,
    ) -> AlignmentStep:
        step = AlignmentStep(
            index=len(steps),
            name=name,
            model=model,
            matrix=to_homography(matrix),
            status=status,
            reason=reason,
            metrics=metrics or {},
            preview=preview,
            keypoints_ref=keypoints_ref,
            keypoints_test=keypoints_test,
        )
        steps.append(step)
        return step

    def add_functions(stage: str, functions: list[str]) -> None:
        function_trace.append({"stage": stage, "functions": list(dict.fromkeys(functions))})

    add_functions("decode", ["cv2.imdecode"])
    add_step(
        "Original",
        "none",
        np.eye(3),
        "accepted",
        f"Original test image ({test_width}x{test_height}).",
        {"width": test_width, "height": test_height},
        preview=test.copy(),
    )

    # ------------------------------------------------------------------
    # Orientation pre-step (multiples of 90 degrees only)
    # ------------------------------------------------------------------
    orientation = assess_orientation(reference, test)
    add_functions("orientation", orientation["opencv_functions"])
    current_test = test.copy()
    cumulative = np.eye(3, dtype=np.float64)
    orientation_applied = False
    orientation_ambiguous = False
    orientation_evidence = [orientation["reason"]]
    best = orientation["best"]
    runner_up = orientation["runner_up"]
    if orientation["status"] == "applied":
        orientation_applied = True
        k = int(best["clockwise_quarter_turns"])
        current_test = rotate_clockwise_quarter(test, k)
        cumulative = clockwise_quarter_matrix(k, test_height, test_width)
        add_step(
            "Orientation fix",
            "orientation",
            cumulative,
            "accepted",
            orientation["reason"],
            {
                "score": best["score"],
                "runner_up_score": runner_up["score"],
                "margin": orientation["margin"],
                "clockwise_degrees": best["clockwise_degrees"],
            },
            preview=current_test,
        )
        add_step(
            "Orientation runner-up",
            "orientation",
            clockwise_quarter_matrix(
                int(runner_up["clockwise_quarter_turns"]), test_height, test_width
            ),
            "rejected",
            (
                f"Runner-up orientation {runner_up['clockwise_degrees']} degrees scored "
                f"{runner_up['score']:.3f}, below the selected {best['score']:.3f}."
            ),
            {
                "score": runner_up["score"],
                "runner_up_score": best["score"],
                "margin": orientation["margin"],
            },
            preview=rotate_clockwise_quarter(
                test, int(runner_up["clockwise_quarter_turns"])
            ),
        )
        decisions.append(
            f"Applied a {best['clockwise_degrees']}-degree clockwise orientation fix before shape checks."
        )
    elif orientation["status"] == "not_needed":
        add_step(
            "Orientation check",
            "orientation",
            np.eye(3),
            "skipped",
            orientation["reason"],
            {
                "score": best["score"],
                "runner_up_score": runner_up["score"],
                "margin": orientation["margin"],
            },
            preview=test.copy(),
        )
        decisions.append("No orientation correction was needed.")
    else:
        orientation_ambiguous = True
        for candidate in orientation["ranked"][:2]:
            k = int(candidate["clockwise_quarter_turns"])
            add_step(
                f"Orientation candidate: {candidate['clockwise_degrees']} degrees",
                "orientation",
                clockwise_quarter_matrix(k, test_height, test_width),
                "ambiguous",
                orientation["reason"],
                {
                    "score": candidate["score"],
                    "margin": orientation["margin"],
                },
                preview=rotate_clockwise_quarter(test, k),
            )
        decisions.append(
            "No orientation correction was applied because orientation scoring was not decisive."
        )
        uncertainty_warnings.append(
            "Orientation scoring was ambiguous; the top two quarter-turn candidates are shown in the alignment sequence."
        )
    record(
        "fix_orientation",
        orientation_evidence,
        orientation["status"],
        (
            "Orientation scoring only tests multiples of 90 degrees; arbitrary rotation is handled later by the alignment ladder."
            if orientation["status"] != "ambiguous"
            else orientation["reason"]
        ),
        {
            "best_degrees": best["clockwise_degrees"],
            "best_score": best["score"],
            "margin": orientation["margin"],
        },
    )

    # ------------------------------------------------------------------
    # Shape normalization and initial comparison
    # ------------------------------------------------------------------
    resized_test_image = False
    current_height, current_width = current_test.shape[:2]
    if (current_height, current_width) != (reference_height, reference_width):
        if current_width * reference_height != reference_width * current_height:
            raise ValueError(
                "Image aspect ratios differ. Crop or geometrically align the images before comparison."
            )
        scale = _scale_ref_to_source(
            reference_width, reference_height, current_width, current_height
        )
        cumulative = cumulative @ scale
        interpolation = _resize_interpolation(
            current_width, current_height, reference_width, reference_height
        )
        current_test = cv2.resize(
            current_test, (reference_width, reference_height), interpolation=interpolation
        )
        resized_test_image = True
        add_step(
            "Shape normalization",
            "none",
            cumulative,
            "accepted",
            (
                f"Resized the oriented test image from {current_width}x{current_height} to "
                f"{reference_width}x{reference_height} with a preserved aspect ratio."
            ),
            {"scale_x": current_width / reference_width, "scale_y": current_height / reference_height},
            preview=current_test,
        )
        decisions.append("Resized the test image to the reference dimensions (aspect ratio preserved).")
        add_functions("shape_normalization", ["cv2.resize"])

    initial = compare_images(reference, current_test)
    add_functions("comparison_initial", initial.opencv_functions)
    evidence.append(
        f"Initial Lab comparison: {initial.changed_percentage:.2f}% changed pixels "
        f"(adaptive ΔE threshold {initial.threshold:.2f}, minimum region {initial.minimum_region_area} px)."
    )

    current_mask = np.full((reference_height, reference_width), 255, dtype=np.uint8)
    current_mask = cv2.erode(current_mask, np.ones((5, 5), np.uint8))
    _, _, current_metrics = score_alignment(
        reference, current_test, np.eye(3), erode=2
    )
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_test, cv2.COLOR_BGR2GRAY)

    texture_level = min(float(reference_gray.std()), float(current_gray.std()))
    too_small = min(reference_width, reference_height) < MIN_ALIGNMENT_DIMENSION
    insufficient_texture = too_small or texture_level < MIN_TEXTURE_STD
    if insufficient_texture:
        evidence.append(
            f"Alignment was skipped because the image is too small or has insufficient texture "
            f"(minimum standard deviation {texture_level:.3f})."
        )
        decisions.append(
            "No geometric alignment was attempted because the images are too small or nearly blank."
        )
        add_step(
            "Translation skipped",
            "translation",
            cumulative,
            "ambiguous",
            "Too little texture for a stable geometric estimate.",
            {"texture_std": texture_level},
            preview=current_test,
        )
        record(
            "estimate_translation",
            [f"Minimum grayscale standard deviation {texture_level:.3f} (< 1.0)."],
            "skipped",
            "No geometric correction can be justified from a nearly blank image.",
            {"texture_std": texture_level},
        )

    alignment_applied = orientation_applied
    alignment_confidence = "single_method"
    ambiguity_flag = False
    translation_ambiguous = False
    ladder_stopped = insufficient_texture
    accepted_models: list[str] = []
    termination_reason = "Image too small or texture insufficient." if insufficient_texture else ""
    photometric_done = False
    photometric_result = None
    secondary_done = False
    phase_response = 0.0
    phase_peak_ratio = 0.0
    last_residual = float(current_metrics["residual_normalized"])
    actions_used = 1  # orientation check

    def budget_left() -> bool:
        return actions_used < max_actions

    def apply_candidate(
        candidate_matrix: np.ndarray,
        candidate_aligned: np.ndarray,
        candidate_mask: np.ndarray,
        after_metrics: dict,
    ) -> None:
        nonlocal current_test, current_gray, current_mask, current_metrics, cumulative, last_residual
        current_test = candidate_aligned
        current_mask = candidate_mask
        current_metrics = after_metrics
        cumulative = cumulative @ to_homography(candidate_matrix)
        current_gray = cv2.cvtColor(current_test, cv2.COLOR_BGR2GRAY)
        last_residual = float(after_metrics["residual_normalized"])

    def residual_is_small() -> bool:
        return current_metrics["ncc"] >= 0.995 and current_metrics["residual_normalized"] <= 0.05

    def run_photometric(stage_note: str = "") -> None:
        nonlocal photometric_done, photometric_result, current_test, current_gray
        if photometric_done:
            return
        photometric_done = True
        result = fit_photometric_normalization(reference, current_test, current_mask)
        photometric_result = result
        add_functions("photometric_normalization", result.opencv_functions)
        if result.applied and result.normalized_image is not None:
            current_test = result.normalized_image
            current_gray = cv2.cvtColor(current_test, cv2.COLOR_BGR2GRAY)
            add_step(
                "Photometric normalization",
                "none",
                cumulative,
                "accepted",
                result.reason,
                {
                    "residual_before": result.residual_before,
                    "residual_after": result.residual_after,
                    "improvement_relative": result.improvement_relative,
                    "channel_gain": result.channel_gain,
                    "channel_offset": result.channel_offset,
                },
                preview=current_test,
            )
            decisions.append("Applied a robust per-channel photometric normalization before differencing.")
            evidence.append(result.reason)
        else:
            add_step(
                "Photometric normalization",
                "none",
                cumulative,
                "rejected" if result.status == "rejected" else "skipped",
                result.reason,
                {
                    "residual_before": result.residual_before,
                    "residual_after": result.residual_after,
                    "improvement_relative": result.improvement_relative,
                },
                preview=current_test,
            )
            decisions.append(f"Photometric normalization was {result.status}: {result.reason}")
        record(
            "normalize_photometry",
            [result.reason, stage_note] if stage_note else [result.reason],
            result.status,
            (
                "A smooth photometric fit can absorb a genuine low-contrast change; "
                "the limitation is reported with the result."
                if result.applied
                else "No lighting correction was applied."
            ),
            {
                "residual_before": result.residual_before,
                "residual_after": result.residual_after,
                "improvement_relative": result.improvement_relative,
                "channel_gain": result.channel_gain,
            },
        )

    # ------------------------------------------------------------------
    # Translation (cheapest candidate)
    # ------------------------------------------------------------------
    if not ladder_stopped:
        phase = phase_correlation(
            reference_gray.astype(np.float32), current_gray.astype(np.float32)
        )
        phase_response = float(phase.response)
        phase_peak_ratio = float(phase.peak_ratio)
        add_functions("translation", phase.functions)
        evidence.append(phase.reason)
        if phase.status in {"weak", "ambiguous", "failed", "skipped"}:
            phase_step_status = {
                "weak": "rejected",
                "ambiguous": "ambiguous",
                "failed": "skipped",
                "skipped": "skipped",
            }[phase.status]
            add_step(
                "Coarse: phase correlation",
                "translation",
                cumulative,
                phase_step_status,
                phase.reason,
                {
                    "response": phase.response,
                    "peak_ratio": phase.peak_ratio,
                    "peak": phase.peak,
                    "second_peak": phase.second_peak,
                },
                preview=current_test,
            )
            record(
                "estimate_translation",
                [phase.reason],
                phase.status,
                (
                    "Repetitive texture means the translation is not unique; no correction was applied."
                    if phase.status == "ambiguous"
                    else "Translation evidence was not usable."
                ),
                {
                    "response": phase.response,
                    "peak_ratio": phase.peak_ratio,
                },
            )
            if phase.status == "ambiguous":
                # A non-unique translation peak only rules out the cheapest
                # model.  Richer feature-based models can still resolve a real
                # rotation, scale or perspective change.
                translation_ambiguous = True
                uncertainty_warnings.append(phase.reason)
            actions_used += 1
        else:
            shift_matrix = np.eye(3, dtype=np.float64)
            shift_matrix[0, 2] = phase.shift[0]
            shift_matrix[1, 2] = phase.shift[1]
            shift_size = float(np.hypot(*phase.shift))
            plausibility = validate_transform(
                shift_matrix,
                "translation",
                reference_width,
                reference_height,
                reference_width,
                reference_height,
            )
            aligned, candidate_mask, after_metrics = score_alignment(
                reference, current_test, shift_matrix
            )
            accepted = False
            if shift_size < 0.5:
                reason = (
                    f"Estimated shift {shift_size:.2f} px is below the 0.5 px correction threshold."
                )
            elif not plausibility["valid"]:
                reason = plausibility["reason"]
            else:
                accepted, reason = candidate_acceptance(
                    current_metrics, after_metrics, "translation"
                )

            # Independent feature-based translation cross-check.
            feature_bundle = None
            feature_translation = None
            ambiguous_candidate = False
            skipped_candidate = False
            if shift_size < 0.5:
                skipped_candidate = True
                reason = (
                    f"Estimated shift {shift_size:.2f} px is below the 0.5 px correction threshold; no correction was needed."
                )
            if accepted:
                from .align import _detect_and_match

                feature_bundle = _detect_and_match(reference_gray, current_gray)
                add_functions("translation_feature_check", feature_bundle.functions)
                if feature_bundle.status == "estimated":
                    feature_translation = estimate_feature_transform(
                        feature_bundle,
                        "translation",
                        reference_width,
                        reference_height,
                    )
                    descriptor = decompose_matrix(feature_translation.matrix)
                    if (
                        feature_translation.status != "candidate"
                        or abs(descriptor["rotation_degrees"]) > 1.0
                        or abs(descriptor["scale"] - 1.0) > 0.03
                    ):
                        feature_translation = None
                if feature_translation is not None:
                    status, cross_reason = cross_check(
                        "translation",
                        feature_translation,
                        TransformCandidate(
                            "translation",
                            shift_matrix,
                            "candidate",
                            "Phase-correlation translation candidate.",
                            {},
                        ),
                        (reference_width, reference_height),
                    )
                    if status == "agree":
                        alignment_confidence = "cross_checked"
                        uncertainty_warnings.append(cross_reason)
                        evidence.append(cross_reason)
                    elif status == "disagree":
                        accepted = False
                        ambiguous_candidate = True
                        reason = cross_reason
                        ambiguity_flag = True
                        ladder_stopped = True
                        termination_reason = cross_reason
                        uncertainty_warnings.append(cross_reason)

            if accepted:
                step_status = "accepted"
            elif ambiguous_candidate:
                step_status = "ambiguous"
            elif skipped_candidate:
                step_status = "skipped"
            else:
                step_status = "rejected"
            metrics = {
                **after_metrics,
                "shift_x": phase.shift[0],
                "shift_y": phase.shift[1],
                "response": phase.response,
                "peak_ratio": phase.peak_ratio,
                "opencv_functions": phase.functions,
            }
            add_step(
                "Coarse: phase correlation",
                "translation",
                cumulative @ shift_matrix,
                step_status,
                reason,
                metrics,
                preview=aligned,
                keypoints_ref=feature_translation.keypoints_ref if feature_translation else None,
                keypoints_test=feature_translation.keypoints_test if feature_translation else None,
            )
            record(
                "estimate_translation",
                [phase.reason, reason],
                step_status,
                (
                    "Translation is accepted only when the shared-overlap residual improves and the peak is unique."
                    if accepted
                    else "No translation correction was applied."
                ),
                metrics,
            )
            if accepted:
                apply_candidate(shift_matrix, aligned, candidate_mask, after_metrics)
                accepted_models.append("translation")
                alignment_applied = True
                decisions.append("Applied the phase-correlation translation after the robust overlap gate passed.")
            decisions.append(reason)
            actions_used += 1

        # Let the residual classification choose the next action.
        if not ladder_stopped and budget_left():
            classification = _classify_residual(reference, current_test, current_mask)
            add_functions("residual_classification", ["cv2.Canny", "cv2.distanceTransform", "cv2.GaussianBlur"])
            if classification["label"] == "lighting_like" and current_metrics["ncc"] >= 0.90:
                evidence.extend(classification["evidence"])
                run_photometric(
                    "The residual classifier saw a smooth, image-wide difference and chose normalization before richer geometry."
                )
                actions_used += 1
                if photometric_result is not None and photometric_result.applied:
                    after_normalization = compare_images(
                        reference, current_test, current_mask
                    )
                    if after_normalization.changed_percentage < 1.0 and not after_normalization.region_change_indicated:
                        ladder_stopped = True
                        termination_reason = (
                            "Photometric normalization resolved the smooth residual; no geometric correction was needed."
                        )
                if not ladder_stopped:
                    current_metrics = score_alignment(
                        reference, current_test, np.eye(3), erode=2
                    )[2]
                    current_gray = cv2.cvtColor(current_test, cv2.COLOR_BGR2GRAY)
            elif residual_is_small():
                ladder_stopped = True
                termination_reason = (
                    "The robust overlap residual is already near zero after the simplest model."
                )

    # ------------------------------------------------------------------
    # Similarity, affine and homography ladder
    # ------------------------------------------------------------------
    for model in ("similarity", "affine", "homography"):
        if ladder_stopped or ambiguity_flag or not budget_left():
            break
        if residual_is_small():
            ladder_stopped = True
            termination_reason = "The residual is near zero; richer models are unnecessary."
            break

        from .align import _detect_and_match

        bundle = _detect_and_match(reference_gray, current_gray)
        add_functions(f"{model}_feature_matching", bundle.functions)
        feature_candidate = estimate_feature_transform(
            bundle, model, reference_width, reference_height
        )
        candidate = feature_candidate
        ecc_note = ""
        if feature_candidate.status == "candidate":
            refined = refine_with_ecc(
                reference_gray,
                current_gray,
                feature_candidate.matrix,
                model,
                current_mask,
            )
            if refined is not None:
                refined_matrix, ecc_cc, ecc_functions = refined
                add_functions(f"{model}_ecc_refinement", ecc_functions)
                plausibility = validate_transform(
                    refined_matrix,
                    model,
                    reference_width,
                    reference_height,
                    reference_width,
                    reference_height,
                )
                if plausibility["valid"]:
                    candidate = type(feature_candidate)(
                        model,
                        refined_matrix,
                        "candidate",
                        feature_candidate.reason + f" ECC refined the estimate (cc {ecc_cc:.3f}).",
                        {
                            **feature_candidate.metrics,
                            "ecc_cc": ecc_cc,
                            "ecc_functions": ecc_functions,
                        },
                        keypoints_ref=feature_candidate.keypoints_ref,
                        keypoints_test=feature_candidate.keypoints_test,
                        functions=feature_candidate.functions + ecc_functions,
                    )
                    ecc_note = f"cv2.findTransformECC refined the {model} estimate (cc {ecc_cc:.3f})."
                    actions_used += 1
                    record(
                        "refine_ecc",
                        [ecc_note],
                        "refined",
                        "ECC can lock onto the wrong local minimum when the overlap is small.",
                        {"ecc_cc": ecc_cc},
                    )
                else:
                    ecc_note = "ECC refinement was rejected: " + plausibility["reason"]
            else:
                ecc_note = "ECC refinement did not converge for this model."
        else:
            ecc_note = "The feature candidate failed its gates; trying an ECC-only estimate."

        # If features failed, or for a similarity scale check, try ECC from identity.
        independent = None
        if model in {"similarity", "affine"} and (
            candidate.status != "candidate" or model == "similarity"
        ):
            independent = refine_with_ecc(
                reference_gray,
                current_gray,
                np.eye(3, dtype=np.float64),
                model,
                current_mask,
            )
            if independent is not None:
                independent_matrix, independent_cc, independent_functions = independent
                add_functions(f"{model}_ecc_independent", independent_functions)
                plausibility = validate_transform(
                    independent_matrix,
                    model,
                    reference_width,
                    reference_height,
                    reference_width,
                    reference_height,
                )
                if plausibility["valid"]:
                    independent_candidate = type(feature_candidate)(
                        model,
                        independent_matrix,
                        "candidate",
                        f"ECC-only {model} estimate (cc {independent_cc:.3f}).",
                        {
                            "ecc_cc": independent_cc,
                            "inliers": 0,
                            "inlier_ratio": 0.0,
                            "opencv_functions": independent_functions,
                        },
                        functions=independent_functions,
                    )
                    if candidate.status != "candidate":
                        candidate = independent_candidate
                    else:
                        status, cross_reason = cross_check(
                            model,
                            candidate,
                            independent_candidate,
                            (reference_width, reference_height),
                        )
                        if status == "disagree":
                            add_step(
                                f"{model.title()}: feature + ECC",
                                model,
                                candidate.matrix,
                                "ambiguous",
                                cross_reason,
                                candidate.metrics,
                                preview=warp_image(
                                    current_test,
                                    candidate.matrix,
                                    reference_width,
                                    reference_height,
                                ),
                                keypoints_ref=candidate.keypoints_ref,
                                keypoints_test=candidate.keypoints_test,
                            )
                            add_step(
                                f"{model.title()}: independent ECC",
                                model,
                                independent_candidate.matrix,
                                "ambiguous",
                                cross_reason,
                                independent_candidate.metrics,
                                preview=warp_image(
                                    current_test,
                                    independent_candidate.matrix,
                                    reference_width,
                                    reference_height,
                                ),
                            )
                            record(
                                f"estimate_{model}",
                                [cross_reason],
                                "ambiguous",
                                "Independent estimators disagree; no transform was applied.",
                                {"corner_displacement": cross_reason},
                            )
                            ambiguity_flag = True
                            ladder_stopped = True
                            termination_reason = cross_reason
                            uncertainty_warnings.append(cross_reason)
                            actions_used += 1
                            break
                        if status == "agree":
                            alignment_confidence = "cross_checked"
                            evidence.append(cross_reason)
                            uncertainty_warnings.append(cross_reason)

        if model == "similarity" and candidate.status != "candidate" and budget_left():
            fallback = log_polar_candidate(
                reference_gray, current_gray, reference_width, reference_height
            )
            add_functions("log_polar_fallback", fallback.functions)
            if fallback.status == "candidate":
                candidate = fallback
                ecc_note += f" Log-polar fallback used: {fallback.reason}"
            else:
                ecc_note += f" Log-polar fallback unavailable: {fallback.reason}"

        if candidate.status != "candidate":
            add_step(
                f"{model.title()}: candidate",
                model,
                candidate.matrix,
                "rejected",
                candidate.reason,
                candidate.metrics,
                preview=current_test,
                keypoints_ref=candidate.keypoints_ref,
                keypoints_test=candidate.keypoints_test,
            )
            record(
                f"estimate_{model}",
                [candidate.reason, ecc_note] if ecc_note else [candidate.reason],
                "rejected",
                "No candidate cleared the inlier, spread and plausibility gates for this model.",
                candidate.metrics,
            )
            actions_used += 1
            continue

        aligned, candidate_mask, after_metrics = score_alignment(
            reference, current_test, candidate.matrix
        )
        accepted, reason = candidate_acceptance(current_metrics, after_metrics, model)
        descriptor = decompose_matrix(candidate.matrix)
        step_metrics = {
            **candidate.metrics,
            **after_metrics,
            "rotation_degrees": descriptor["rotation_degrees"],
            "scale": descriptor["scale"],
            "ecc_note": ecc_note,
            "opencv_functions": list(
                dict.fromkeys(
                    candidate.metrics.get("opencv_functions", []) + candidate.functions
                )
            ),
        }
        status = "accepted" if accepted else "rejected"
        add_step(
            f"{model.title()}: {'feature RANSAC + ECC' if 'ECC' in ecc_note else 'feature RANSAC'}",
            model,
            cumulative @ candidate.matrix if accepted else candidate.matrix,
            status,
            reason,
            step_metrics,
            preview=aligned,
            keypoints_ref=candidate.keypoints_ref,
            keypoints_test=candidate.keypoints_test,
        )
        record(
            f"estimate_{model}",
            [candidate.reason, reason, ecc_note] if ecc_note else [candidate.reason, reason],
            status,
            (
                "The candidate was accepted only after the complexity-penalized overlap gate."
                if accepted
                else "The more complex model did not clearly beat the current alignment."
            ),
            step_metrics,
        )
        actions_used += 1
        if accepted:
            apply_candidate(candidate.matrix, aligned, candidate_mask, after_metrics)
            accepted_models.append(model)
            alignment_applied = True
            decisions.append(f"Applied the {model} alignment after it clearly beat the simpler model.")
            classification = _classify_residual(reference, current_test, current_mask)
            add_functions("residual_classification", ["cv2.Canny", "cv2.distanceTransform"])
            evidence.extend(classification["evidence"])
            if residual_is_small():
                ladder_stopped = True
                termination_reason = f"The {model} alignment reduced the residual to near zero."
                break
            if classification["label"] == "real_change_like":
                ladder_stopped = True
                termination_reason = (
                    "Compact regions survived alignment away from edges; they are reported as change regions, not defects."
                )
                break
            if (
                classification["label"] == "lighting_like"
                and current_metrics["ncc"] >= 0.90
                and budget_left()
            ):
                run_photometric(
                    "The residual classifier saw a smooth image-wide difference after alignment."
                )
                actions_used += 1
                ladder_stopped = True
                termination_reason = (
                    "Photometric normalization was the better next action than a richer geometric model."
                )
                break
        else:
            decisions.append(reason)

    # ------------------------------------------------------------------
    # Photometric normalization (always considered before the final diff)
    # ------------------------------------------------------------------
    if not photometric_done and budget_left():
        classification = _classify_residual(reference, current_test, current_mask)
        run_photometric(
            f"Final photometric check after residual class '{classification['label']}'."
        )
        actions_used += 1
        if photometric_result is not None and photometric_result.applied:
            current_metrics = score_alignment(
                reference, current_test, np.eye(3), erode=2
            )[2]
            current_gray = cv2.cvtColor(current_test, cv2.COLOR_BGR2GRAY)

    final_valid_mask = current_mask
    final_test = current_test
    if accepted_models and not (
        photometric_result is not None and photometric_result.applied
    ):
        # Sequential warps each resample the previous result, so interpolation
        # blur accumulates.  Rebuild the final image from the original test
        # with the cumulative transform as one resampling pass.
        single_warp, single_mask, single_metrics = score_alignment(
            reference, test, cumulative, erode=2
        )
        current_test = single_warp
        current_gray = cv2.cvtColor(current_test, cv2.COLOR_BGR2GRAY)
        current_mask = single_mask
        current_metrics = single_metrics
        final_valid_mask = single_mask
        final_test = single_warp

    # Final comparison methods.
    final = compare_images(reference, final_test, final_valid_mask)
    add_functions("comparison_final", final.opencv_functions)
    independent_comparison = compare_edges(reference, final_test, final_valid_mask)
    add_functions("edge_comparison", independent_comparison.opencv_functions)
    ssim_comparison = compare_ssim(reference, final_test, final_valid_mask)
    add_functions("ssim_comparison", ssim_comparison.opencv_functions)
    changed_regions = detect_changed_regions(
        final.difference_mask, minimum_area=final.minimum_region_area
    )
    add_functions("region_detection", ["cv2.connectedComponentsWithStats"])

    edge_change_indicated = independent_comparison.change_indicated
    ssim_change_indicated = ssim_comparison.change_indicated
    pixel_change_indicated = bool(
        final.global_change_indicated or final.region_change_indicated
    )

    method_states = {
        "lab_difference": pixel_change_indicated,
        "canny_edges": edge_change_indicated,
        "ssim": ssim_change_indicated,
    }
    measured_states = {
        name: value for name, value in method_states.items() if value is not None
    }
    method_disagreement = (
        len(set(measured_states.values())) > 1
        if edge_change_indicated is not None and len(measured_states) >= 2
        else None
    )

    if independent_comparison.status == "measured":
        if pixel_change_indicated and edge_change_indicated:
            indication_agreement = "both_indicate_change"
        elif pixel_change_indicated:
            indication_agreement = "pixel_only_indicates_change"
        elif edge_change_indicated:
            indication_agreement = "edge_only_indicates_change"
        else:
            indication_agreement = "neither_indicates_change"
    else:
        indication_agreement = "insufficient_edge_evidence"

    if method_disagreement:
        uncertainty_warnings.append(
            "The Lab difference, Canny edge and SSIM methods do not all agree on whether a change is indicated. "
            "Inspect every method map before drawing a conclusion."
        )
    if edge_change_indicated is None:
        uncertainty_warnings.append(
            "The Canny edge comparison had fewer than four edge pixels in at least one image, so agreement is not scored."
        )
    if ssim_comparison.status != "measured":
        uncertainty_warnings.append("SSIM could not be scored on the valid overlap.")

    evidence.append(
        f"Canny edge comparison: {independent_comparison.reference_edge_pixels} reference and "
        f"{independent_comparison.test_edge_pixels} test edge pixels; "
        f"{independent_comparison.tolerance_pixels}-pixel matching tolerance."
    )
    if independent_comparison.edge_similarity_percentage is not None:
        evidence.append(
            f"Edge similarity {independent_comparison.edge_similarity_percentage:.2f}% "
            f"(disagreement {independent_comparison.edge_disagreement_percentage:.2f}%)."
        )
    if ssim_comparison.mean_ssim is not None:
        evidence.append(
            f"Mean SSIM {ssim_comparison.mean_ssim:.4f}; "
            f"{ssim_comparison.changed_percentage:.2f}% of the overlap is below the "
            f"SSIM threshold {ssim_comparison.threshold:.2f}."
        )
    evidence.append(
        f"Final Lab comparison: {final.changed_percentage:.2f}% changed pixels "
        f"(adaptive ΔE threshold {final.threshold:.2f})."
    )

    # ------------------------------------------------------------------
    # Terminate the loop with an explicit decision.
    # ------------------------------------------------------------------
    if translation_ambiguous and not alignment_applied:
        alignment_status = "ambiguous"
    elif ambiguity_flag:
        alignment_status = "ambiguous"
    elif alignment_applied:
        alignment_status = "applied"
    elif orientation_ambiguous or insufficient_texture:
        alignment_status = "ambiguous"
    elif current_metrics.get("ncc", 0.0) >= 0.995:
        alignment_status = "not_needed"
    elif (
        current_metrics.get("ncc", 0.0) >= 0.98
        and current_metrics.get("residual_normalized", 1.0) <= 0.25
    ):
        alignment_status = "not_needed"
    elif any(step.status == "rejected" and step.model in MODEL_ORDER for step in steps):
        alignment_status = "rejected"
    else:
        alignment_status = "not_needed"

    if not alignment_applied:
        alignment_confidence = "none" if insufficient_texture else alignment_confidence
    if alignment_status == "ambiguous":
        message = (
            "Alignment was ambiguous. No transform was silently chosen; review the candidate steps and reasons."
        )
        if not any("alignment was ambiguous" in warning.lower() for warning in uncertainty_warnings):
            uncertainty_warnings.append(message)
        decisions.append(
            "Alignment was left ambiguous; no correction was silently applied."
        )

    estimated_descriptor = decompose_matrix(cumulative)
    shift_x = estimated_descriptor["translation_x"]
    shift_y = estimated_descriptor["translation_y"]
    if not accepted_models and not orientation_applied and not resized_test_image:
        shift_x = 0.0
        shift_y = 0.0
    method_parts = []
    if orientation_applied:
        method_parts.append("orientation")
    method_parts.extend(accepted_models)
    if photometric_result is not None and photometric_result.applied:
        method_parts.append("photometric")
    alignment_method = "+".join(method_parts) if method_parts else "none"

    overlap_fraction = (
        float(np.count_nonzero(final_valid_mask)) / max(reference_width * reference_height, 1)
    )
    final_metrics = {
        "ncc": current_metrics.get("ncc"),
        "residual_normalized": current_metrics.get("residual_normalized"),
        "overlap_fraction": overlap_fraction,
        "changed_percentage": final.changed_percentage,
        "region_count": changed_regions["region_count"],
    }
    add_step(
        "Final",
        "none",
        cumulative,
        "accepted" if alignment_applied else "skipped",
        (
            "Final aligned test image used for the reported comparison."
            if alignment_applied
            else "No geometric correction was applied; the final comparison uses the shape-normalized test image."
        ),
        final_metrics,
        preview=final_test,
    )
    record(
        "compute_secondary_comparison",
        [
            f"Lab difference indicated: {pixel_change_indicated}.",
            f"Canny edges indicated: {edge_change_indicated}.",
            f"SSIM indicated: {ssim_change_indicated}.",
            f"Method disagreement: {method_disagreement}.",
        ],
        "measured",
        "The three methods measure different properties; disagreement is kept visible rather than averaged away.",
        {
            "lab_difference_indicated": pixel_change_indicated,
            "canny_indicated": edge_change_indicated,
            "ssim_indicated": ssim_change_indicated,
            "methods_disagree": method_disagreement,
        },
    )
    if not termination_reason:
        if alignment_status == "ambiguous":
            termination_reason = "The loop stopped because the evidence remained ambiguous."
        elif residual_is_small():
            termination_reason = "The loop stopped because the residual is near zero."
        elif not budget_left():
            termination_reason = "The loop stopped because the action budget was exhausted."
        else:
            termination_reason = "The loop stopped after the comparison methods were measured."
    record(
        "stop",
        [termination_reason],
        "stopped",
        (
            "Stopping does not make the measurement a defect decision; it only ends the investigation."
        ),
        final_metrics,
    )

    limitations = list(LIMITATIONS)
    if resized_test_image:
        limitations.append(
            "Resizing can alter detail even when the aspect ratio is preserved; review the images before trusting fine differences."
        )
    if final_valid_mask is not None and overlap_fraction < 0.999:
        limitations.append(
            f"The final comparison uses {overlap_fraction * 100:.1f}% of the canvas; the eroded border and invalid warp area are excluded."
        )

    uncertainty_warnings.insert(
        0,
        "Pixel, edge and SSIM measurements are heuristic comparison signals, not calibrated probabilities.",
    )

    return InspectionResult(
        initial_comparison=initial,
        final_comparison=final,
        changed_regions=changed_regions,
        independent_comparison=independent_comparison,
        alignment_applied=alignment_applied,
        alignment_status=alignment_status,
        alignment_method=alignment_method,
        estimated_shift=(float(shift_x), float(shift_y)),
        alignment_response=float(phase_response),
        resized_test_image=resized_test_image,
        pixel_change_indicated=pixel_change_indicated,
        edge_change_indicated=edge_change_indicated,
        method_disagreement=method_disagreement,
        indication_agreement=indication_agreement,
        evidence=evidence,
        decisions=decisions,
        uncertainty_warnings=uncertainty_warnings,
        limitations=limitations,
        alignment_steps=steps,
        agent_trace=trace,
        ssim_comparison=ssim_comparison,
        photometric=photometric_result,
        aligned_test_image=current_test,
        final_test_image=final_test,
        final_valid_mask=final_valid_mask,
        final_matrix=cumulative,
        estimated_rotation_degrees=estimated_descriptor["rotation_degrees"],
        estimated_scale=estimated_descriptor["scale"],
        alignment_confidence=alignment_confidence,
        overlap_fraction=overlap_fraction,
        function_trace=function_trace,
    )
