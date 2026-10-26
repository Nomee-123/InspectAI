"""Geometric alignment: orientation, phase correlation, features, ECC.

The ladder tries simple models first and only accepts a more complex model
when it clears a stricter evidence gate.  Every candidate returns metrics and
a human-readable reason so the UI and JSON report can show what was tried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import TransformCandidate

MODEL_ORDER = {"translation": 0, "similarity": 1, "affine": 2, "homography": 3}
MIN_TEXTURE_STD = 1.0
MIN_ALIGNMENT_DIMENSION = 16
PEAK_RATIO_MINIMUM = 2.0
PHASE_RESPONSE_MINIMUM = 0.2


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def to_homography(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape == (2, 3):
        result = np.eye(3, dtype=np.float64)
        result[:2, :] = array
        return result
    if array.shape == (3, 3):
        return array.copy()
    raise ValueError("Transform matrix must be 2x3 or 3x3.")


def is_perspective(matrix: np.ndarray) -> bool:
    array = to_homography(matrix)
    return bool(
        abs(array[2, 0]) > 1e-9
        or abs(array[2, 1]) > 1e-9
        or abs(array[2, 2] - 1.0) > 1e-9
    )


def normalized_homography(matrix: np.ndarray) -> np.ndarray:
    array = to_homography(matrix)
    if abs(array[2, 2]) < 1e-12:
        raise ValueError("Transform has a zero homogeneous scale.")
    return array / array[2, 2]


def decompose_matrix(matrix: np.ndarray) -> dict[str, float]:
    """Describe the transform that maps current/test coordinates to reference."""
    inverse = np.linalg.inv(to_homography(matrix))
    a = inverse[0, 0]
    b = inverse[0, 1]
    c = inverse[1, 0]
    d = inverse[1, 1]
    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    angle = math.degrees(math.atan2(c, a))
    shear = 0.0
    if scale_x > 1e-9 and scale_y > 1e-9:
        shear = abs(a * b + c * d) / (scale_x * scale_y)
    return {
        "rotation_degrees": angle,
        "scale": math.sqrt(abs(scale_x * scale_y)),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "translation_x": float(inverse[0, 2]),
        "translation_y": float(inverse[1, 2]),
        "shear": shear,
        "perspective": float(
            max(abs(inverse[2, 0]), abs(inverse[2, 1]))
        ),
    }


def mean_corner_displacement(matrix_a: np.ndarray, matrix_b: np.ndarray, width: int, height: int) -> float:
    corners = np.float64(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]
    ).T
    a = to_homography(matrix_a) @ corners
    b = to_homography(matrix_b) @ corners
    a = (a[:2] / np.maximum(np.abs(a[2:3]), 1e-9)).T
    b = (b[:2] / np.maximum(np.abs(b[2:3]), 1e-9)).T
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


# ---------------------------------------------------------------------------
# Orientation pre-step
# ---------------------------------------------------------------------------


def clockwise_quarter_matrix(k: int, height: int, width: int) -> np.ndarray:
    """Cumulative mapping from a k*90-degree clockwise canvas to the input."""
    matrix = np.eye(3, dtype=np.float64)
    if k == 1:
        matrix = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, height - 1.0], [0.0, 0.0, 1.0]])
    elif k == 2:
        matrix = np.array(
            [[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]]
        )
    elif k == 3:
        matrix = np.array([[0.0, -1.0, width - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    elif k != 0:
        raise ValueError("Quarter-turn index must be 0, 1, 2, or 3.")
    return matrix


def rotate_clockwise_quarter(image: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return image.copy()
    return np.ascontiguousarray(np.rot90(image, -k))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denominator)


def assess_orientation(reference: np.ndarray, test: np.ndarray, canvas: int = 192) -> dict:
    """Score the four 90-degree orientations of the test image.

    Mirrors are intentionally not tried: a mirrored image is a real change,
    not a harmless orientation difference.
    """
    reference_small = cv2.resize(reference, (canvas, canvas), interpolation=cv2.INTER_AREA)
    reference_gray = cv2.GaussianBlur(
        cv2.cvtColor(reference_small, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0
    )
    candidates = []
    for k in range(4):
        rotated = rotate_clockwise_quarter(test, k)
        small = cv2.resize(rotated, (canvas, canvas), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0
        )
        score = _correlation(reference_gray, gray)
        candidates.append(
            {
                "clockwise_quarter_turns": k,
                "clockwise_degrees": k * 90,
                "score": score,
            }
        )
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    runner_up = ranked[1]
    margin = best["score"] - runner_up["score"]
    if best["score"] < 0.35 or margin < 0.15:
        status = "ambiguous"
        reason = (
            f"Orientation scoring is not decisive: best {best['score']:.3f} at "
            f"{best['clockwise_degrees']} degrees, runner-up {runner_up['score']:.3f} "
            f"(margin {margin:.3f} < 0.150). No orientation correction was applied."
        )
    elif best["clockwise_quarter_turns"] == 0:
        status = "not_needed"
        reason = (
            f"The 0-degree orientation scored {best['score']:.3f}, ahead of the runner-up "
            f"by {margin:.3f}; no orientation correction was needed."
        )
    else:
        status = "applied"
        reason = (
            f"The {best['clockwise_degrees']}-degree clockwise orientation scored "
            f"{best['score']:.3f}, ahead of the runner-up by {margin:.3f}; "
            f"the test image was rotated to match the reference."
        )
    return {
        "candidates": candidates,
        "ranked": ranked,
        "best": best,
        "runner_up": runner_up,
        "margin": margin,
        "status": status,
        "reason": reason,
        "opencv_functions": ["cv2.resize", "cv2.cvtColor", "cv2.GaussianBlur", "np.corrcoef"],
    }


# ---------------------------------------------------------------------------
# Phase correlation and its ambiguity surface
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    shift: tuple[float, float]
    response: float
    peak_ratio: float
    peak: float
    second_peak: float
    status: str
    reason: str
    functions: list[str] = field(default_factory=list)
    ambiguous: bool = False


def correlation_peak_ratio(
    reference_gray: np.ndarray,
    test_gray: np.ndarray,
    neighborhood: int = 4,
) -> tuple[float, float, float]:
    """Return (peak, second_peak, ratio) from a phase-correlation surface.

    The best peak outside a small neighbourhood is compared with the global
    peak.  A repetitive texture produces several near-equal peaks, which
    lowers the ratio and makes translation ambiguous even when the response
    strength itself is high.
    """
    height, width = reference_gray.shape[:2]
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    a = (reference_gray.astype(np.float32) - float(np.mean(reference_gray))) * window
    b = (test_gray.astype(np.float32) - float(np.mean(test_gray))) * window
    f1 = np.fft.fft2(a)
    f2 = np.fft.fft2(b)
    cross = f1 * np.conj(f2)
    magnitude = np.abs(cross)
    cross = cross / np.maximum(magnitude, 1e-9)
    surface = np.abs(np.fft.ifft2(cross))
    peak_y, peak_x = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak = float(surface[peak_y, peak_x])
    yy, xx = np.mgrid[0:height, 0:width]
    dy = np.minimum(np.abs(yy - peak_y), height - np.abs(yy - peak_y))
    dx = np.minimum(np.abs(xx - peak_x), width - np.abs(xx - peak_x))
    outside = (dy > neighborhood) | (dx > neighborhood)
    second = float(surface[outside].max()) if bool(outside.any()) else 0.0
    ratio = peak / max(second, 1e-12)
    return peak, second, min(ratio, 1e9)


def phase_correlation(reference_gray: np.ndarray, test_gray: np.ndarray) -> PhaseResult:
    height, width = reference_gray.shape[:2]
    texture = min(float(np.std(reference_gray)), float(np.std(test_gray)))
    functions = ["cv2.createHanningWindow", "cv2.phaseCorrelate", "np.fft.fft2", "np.fft.ifft2"]
    if min(height, width) < MIN_ALIGNMENT_DIMENSION:
        return PhaseResult(
            (0.0, 0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            "skipped",
            "The image is too small for a stable phase-correlation estimate.",
            functions,
        )
    if texture < MIN_TEXTURE_STD:
        return PhaseResult(
            (0.0, 0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            "skipped",
            f"Insufficient grayscale texture (standard deviation {texture:.3f} < 1.0).",
            functions,
        )

    response = 0.0
    shift = (0.0, 0.0)
    try:
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        raw_shift, raw_response = cv2.phaseCorrelate(
            reference_gray.astype(np.float32), test_gray.astype(np.float32), window
        )
        shift = (float(raw_shift[0]), float(raw_shift[1]))
        response = float(raw_response)
    except cv2.error as error:
        return PhaseResult(
            (0.0, 0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            "failed",
            f"OpenCV could not estimate phase correlation: {error}",
            functions,
        )

    peak, second, ratio = correlation_peak_ratio(reference_gray, test_gray)
    finite = bool(np.isfinite((*shift, response, ratio)).all())
    if not finite:
        return PhaseResult(
            shift,
            response,
            ratio,
            peak,
            second,
            "failed",
            "Phase correlation returned a non-finite estimate.",
            functions,
        )
    if ratio < PEAK_RATIO_MINIMUM:
        return PhaseResult(
            shift,
            response,
            ratio,
            peak,
            second,
            "ambiguous",
            (
                f"The phase-correlation peak is not unique: peak-to-second-peak ratio "
                f"{ratio:.2f} < {PEAK_RATIO_MINIMUM:.2f}. The texture is likely repetitive, "
                "so no translation correction was applied."
            ),
            functions,
            True,
        )
    if response < PHASE_RESPONSE_MINIMUM:
        return PhaseResult(
            shift,
            response,
            ratio,
            peak,
            second,
            "weak",
            (
                f"Phase-correlation response {response:.3f} is below the "
                f"{PHASE_RESPONSE_MINIMUM:.2f} minimum; translation is not trusted, "
                "so the ladder moves to a richer model."
            ),
            functions,
        )
    return PhaseResult(
        shift,
        response,
        ratio,
        peak,
        second,
        "estimated",
        (
            f"Phase correlation estimated a translation of ({shift[0]:.2f}, {shift[1]:.2f}) px "
            f"with response {response:.3f} and peak-to-second-peak ratio {ratio:.2f}."
        ),
        functions,
        False,
    )


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------


@dataclass
class FeatureBundle:
    detector: str
    reference_points: np.ndarray
    test_points: np.ndarray
    match_count: int
    reference_size: tuple[int, int]
    test_size: tuple[int, int]
    status: str
    reason: str
    functions: list[str] = field(default_factory=list)


def _detect_and_match(reference_gray: np.ndarray, test_gray: np.ndarray) -> FeatureBundle:
    height, width = reference_gray.shape[:2]
    test_height, test_width = test_gray.shape[:2]
    attempts = []
    if hasattr(cv2, "SIFT_create"):
        attempts.append(("SIFT", cv2.SIFT_create(nfeatures=2000), cv2.NORM_L2, 0.75))
    attempts.append(("ORB", cv2.ORB_create(nfeatures=2500, fastThreshold=7), cv2.NORM_HAMMING, 0.78))
    if hasattr(cv2, "AKAZE_create"):
        attempts.append(("AKAZE", cv2.AKAZE_create(), cv2.NORM_HAMMING, 0.80))

    last_reason = "No feature detector produced usable matches."
    for name, detector, norm, ratio in attempts:
        keypoints_reference, descriptors_reference = detector.detectAndCompute(
            reference_gray, None
        )
        keypoints_test, descriptors_test = detector.detectAndCompute(test_gray, None)
        functions = [f"cv2.{name}_create", "detector.detectAndCompute"]
        if (
            descriptors_reference is None
            or descriptors_test is None
            or len(keypoints_reference) < 8
            or len(keypoints_test) < 8
        ):
            last_reason = f"{name} found too few keypoints."
            continue
        matcher = cv2.BFMatcher(norm)
        try:
            pairs = matcher.knnMatch(descriptors_reference, descriptors_test, k=2)
        except cv2.error as error:
            last_reason = f"{name} matching failed: {error}"
            continue
        functions.append("cv2.BFMatcher.knnMatch")
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair[0], pair[1]
            if first.distance < ratio * second.distance:
                good.append(first)
        if len(good) < 8:
            last_reason = f"{name} produced only {len(good)} ratio-test matches."
            continue
        # Keep the best match per test keypoint.
        best_for_test: dict[int, cv2.DMatch] = {}
        for match in good:
            existing = best_for_test.get(match.trainIdx)
            if existing is None or match.distance < existing.distance:
                best_for_test[match.trainIdx] = match
        unique_matches = sorted(best_for_test.values(), key=lambda item: item.distance)
        reference_points = np.float32(
            [keypoints_reference[item.queryIdx].pt for item in unique_matches]
        ).reshape(-1, 1, 2)
        test_points = np.float32(
            [keypoints_test[item.trainIdx].pt for item in unique_matches]
        ).reshape(-1, 1, 2)
        if len(unique_matches) < 8:
            last_reason = f"{name} produced only {len(unique_matches)} unique matches."
            continue
        return FeatureBundle(
            name,
            reference_points,
            test_points,
            len(unique_matches),
            (width, height),
            (test_width, test_height),
            "estimated",
            f"{name} produced {len(unique_matches)} ratio-tested matches.",
            functions,
        )
    return FeatureBundle(
        "none",
        np.empty((0, 1, 2), np.float32),
        np.empty((0, 1, 2), np.float32),
        0,
        (width, height),
        (test_width, test_height),
        "insufficient",
        last_reason,
        ["detector.detectAndCompute", "cv2.BFMatcher.knnMatch"],
    )


def _hull_fraction(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    area = abs(float(cv2.contourArea(hull)))
    return area / max(width * height, 1)


def validate_transform(
    matrix: np.ndarray,
    model: str,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> dict:
    """Check plausibility and shared-overlap coverage of a candidate."""
    result = {
        "valid": False,
        "reason": "",
        "scale": 0.0,
        "rotation_degrees": 0.0,
        "shear": 0.0,
        "overlap_fraction": 0.0,
    }
    try:
        homography = normalized_homography(matrix)
    except (ValueError, np.linalg.LinAlgError):
        result["reason"] = "The transform matrix is singular or non-finite."
        return result
    if not np.all(np.isfinite(homography)):
        result["reason"] = "The transform matrix contains non-finite values."
        return result

    descriptor = decompose_matrix(homography)
    scale = descriptor["scale"]
    shear = descriptor["shear"]
    result.update(
        scale=scale,
        rotation_degrees=descriptor["rotation_degrees"],
        shear=shear,
    )
    if not 0.35 <= scale <= 2.8:
        result["reason"] = f"Implausible scale {scale:.3f} (outside 0.35-2.8)."
        return result
    if model == "similarity" and shear > 0.08:
        result["reason"] = f"Similarity fit contains shear {shear:.3f}."
        return result
    if model in {"translation", "affine"} and shear > 0.75:
        result["reason"] = f"Extreme shear {shear:.3f} is not plausible."
        return result

    corners = np.float64(
        [
            [0, 0, 1],
            [output_width - 1, 0, 1],
            [output_width - 1, output_height - 1, 1],
            [0, output_height - 1, 1],
        ]
    ).T
    mapped = homography @ corners
    if np.any(np.abs(mapped[2]) < 1e-7):
        result["reason"] = "Perspective transform folds the image (zero denominator)."
        return result
    mapped = (mapped[:2] / mapped[2]).T
    if not np.all(np.isfinite(mapped)):
        result["reason"] = "Mapped corners are non-finite."
        return result
    margin_x = 0.75 * output_width
    margin_y = 0.75 * output_height
    if np.any(mapped[:, 0] < -margin_x) or np.any(mapped[:, 0] > source_width + margin_x):
        result["reason"] = "Mapped corners fall implausibly far outside the test image."
        return result
    if np.any(mapped[:, 1] < -margin_y) or np.any(mapped[:, 1] > source_height + margin_y):
        result["reason"] = "Mapped corners fall implausibly far outside the test image."
        return result

    ones = np.full((source_height, source_width), 255, dtype=np.uint8)
    valid = warp_mask(ones, homography, output_width, output_height, erode=0)
    overlap = float(np.count_nonzero(valid)) / max(output_width * output_height, 1)
    result["overlap_fraction"] = overlap
    if overlap < 0.35:
        result["reason"] = (
            f"Shared valid overlap {overlap * 100:.1f}% is below the 35% minimum."
        )
        return result
    result["valid"] = True
    result["reason"] = (
        f"Plausible {model} transform: scale {scale:.3f}, shear {shear:.3f}, "
        f"overlap {overlap * 100:.1f}%."
    )
    return result


def estimate_feature_transform(
    bundle: FeatureBundle, model: str, output_width: int, output_height: int
) -> TransformCandidate:
    points_reference = bundle.reference_points
    points_test = bundle.test_points
    functions = list(bundle.functions)
    if bundle.status != "estimated" or len(points_reference) < 8:
        return TransformCandidate(
            model,
            np.eye(3),
            "rejected",
            f"Not enough feature matches ({bundle.match_count}) for a {model} estimate.",
            {"matches": bundle.match_count, "opencv_functions": functions},
            functions=functions,
        )

    matrix = None
    inliers = None
    try:
        if model == "similarity":
            matrix, inliers = cv2.estimateAffinePartial2D(
                points_reference,
                points_test,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=3000,
                confidence=0.995,
                refineIters=10,
            )
            functions.append("cv2.estimateAffinePartial2D")
        elif model == "affine":
            matrix, inliers = cv2.estimateAffine2D(
                points_reference,
                points_test,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=3000,
                confidence=0.995,
                refineIters=10,
            )
            functions.append("cv2.estimateAffine2D")
        elif model == "homography":
            matrix, inliers = cv2.findHomography(
                points_reference,
                points_test,
                cv2.RANSAC,
                3.0,
                maxIters=3000,
                confidence=0.995,
            )
            functions.append("cv2.findHomography")
        elif model == "translation":
            affine, inlier_mask = cv2.estimateAffinePartial2D(
                points_reference,
                points_test,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=3000,
                confidence=0.995,
            )
            functions.append("cv2.estimateAffinePartial2D")
            if affine is not None:
                matrix = np.eye(3, dtype=np.float64)
                matrix[:2, :] = affine
            inliers = inlier_mask
        else:
            raise ValueError(f"Unknown feature model {model!r}.")
    except cv2.error as error:
        return TransformCandidate(
            model,
            np.eye(3),
            "rejected",
            f"OpenCV could not estimate the {model} transform: {error}",
            {"matches": bundle.match_count, "opencv_functions": functions},
            functions=functions,
        )

    if matrix is None:
        return TransformCandidate(
            model,
            np.eye(3),
            "rejected",
            f"RANSAC returned no {model} transform from {bundle.match_count} matches.",
            {"matches": bundle.match_count, "opencv_functions": functions},
            functions=functions,
        )

    homography = to_homography(matrix)
    inlier_mask = (
        np.asarray(inliers).ravel().astype(bool)
        if inliers is not None
        else np.zeros(len(points_reference), dtype=bool)
    )
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / max(len(points_reference), 1)
    inlier_reference = points_reference.reshape(-1, 2)[inlier_mask]
    inlier_test = points_test.reshape(-1, 2)[inlier_mask]
    spread = _hull_fraction(inlier_reference, bundle.reference_size[0], bundle.reference_size[1])
    metrics = {
        "matches": bundle.match_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "inlier_hull_fraction": spread,
        "detector": bundle.detector,
        "opencv_functions": functions,
    }
    keypoints_reference = [
        (float(x), float(y), bool(flag))
        for (x, y), flag in zip(points_reference.reshape(-1, 2), inlier_mask)
    ]
    keypoints_test = [
        (float(x), float(y), bool(flag))
        for (x, y), flag in zip(points_test.reshape(-1, 2), inlier_mask)
    ]

    def rejected(reason: str) -> TransformCandidate:
        metrics["reject_reason"] = reason
        return TransformCandidate(
            model,
            homography,
            "rejected",
            reason,
            metrics,
            keypoints_ref=keypoints_reference,
            keypoints_test=keypoints_test,
            functions=functions,
        )

    if inlier_count < 8:
        return rejected(f"Only {inlier_count} inliers; at least 8 are required.")
    if inlier_ratio < 0.35:
        return rejected(
            f"Inlier ratio {inlier_ratio:.2f} is below the 0.35 minimum."
        )
    if spread < 0.12:
        return rejected(
            f"Inlier spatial spread {spread:.2f} of the image area is below the 0.12 minimum."
        )
    plausibility = validate_transform(
        homography,
        model,
        bundle.test_size[0],
        bundle.test_size[1],
        output_width,
        output_height,
    )
    metrics.update(
        {
            "scale": plausibility["scale"],
            "rotation_degrees": plausibility["rotation_degrees"],
            "shear": plausibility["shear"],
            "overlap_fraction": plausibility["overlap_fraction"],
        }
    )
    if not plausibility["valid"]:
        return rejected(plausibility["reason"])
    return TransformCandidate(
        model,
        homography,
        "candidate",
        plausibility["reason"],
        metrics,
        keypoints_ref=keypoints_reference,
        keypoints_test=keypoints_test,
        functions=functions,
    )


# ---------------------------------------------------------------------------
# ECC refinement
# ---------------------------------------------------------------------------


def refine_with_ecc(
    reference_gray: np.ndarray,
    test_gray: np.ndarray,
    initial_matrix: np.ndarray,
    model: str,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[str]] | None:
    """Refine a candidate with ``cv2.findTransformECC`` (template -> input)."""
    motion = {
        "translation": cv2.MOTION_TRANSLATION,
        "similarity": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }.get(model)
    if motion is None:
        return None
    functions = ["cv2.GaussianBlur", "cv2.findTransformECC"]
    reference_blur = cv2.GaussianBlur(reference_gray.astype(np.float32), (5, 5), 1.0)
    test_blur = cv2.GaussianBlur(test_gray.astype(np.float32), (5, 5), 1.0)
    if model == "homography":
        warp = normalized_homography(initial_matrix).astype(np.float32)
    else:
        warp = to_homography(initial_matrix)[:2, :].astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        300,
        1e-5,
    )
    mask = None
    if valid_mask is not None:
        mask = np.where(valid_mask > 0, 255, 0).astype(np.uint8)
    try:
        correlation, warp = cv2.findTransformECC(
            reference_blur,
            test_blur,
            warp,
            motion,
            criteria,
            mask,
            5,
        )
    except cv2.error:
        return None
    return to_homography(warp), float(correlation), functions


# ---------------------------------------------------------------------------
# Log-polar fallback (Fourier-Mellin style)
# ---------------------------------------------------------------------------


def log_polar_candidate(
    reference_gray: np.ndarray,
    test_gray: np.ndarray,
    output_width: int,
    output_height: int,
) -> TransformCandidate:
    """Estimate rotation/scale from a log-polar phase correlation."""
    height, width = reference_gray.shape[:2]
    if min(height, width) < 32:
        return TransformCandidate(
            "similarity",
            np.eye(3),
            "rejected",
            "Image too small for a log-polar estimate.",
            {"opencv_functions": ["cv2.warpPolar"]},
            functions=["cv2.warpPolar"],
        )
    center = (width / 2.0, height / 2.0)
    max_radius = 0.5 * min(height, width)
    radius_bins = max(32, int(min(height, width) / 2))
    angle_bins = 360
    dsize = (angle_bins, radius_bins)
    flags = cv2.INTER_LINEAR | cv2.WARP_POLAR_LOG | cv2.WARP_FILL_OUTLIERS
    reference_window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    test_window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    reference_polar = cv2.warpPolar(
        (reference_gray.astype(np.float32) * reference_window), dsize, center, max_radius, flags
    )
    test_polar = cv2.warpPolar(
        (test_gray.astype(np.float32) * test_window), dsize, center, max_radius, flags
    )
    if reference_polar.std() < 1e-6 or test_polar.std() < 1e-6:
        return TransformCandidate(
            "similarity",
            np.eye(3),
            "rejected",
            "Log-polar representation has insufficient texture.",
            {"opencv_functions": ["cv2.warpPolar", "cv2.phaseCorrelate"]},
            functions=["cv2.warpPolar", "cv2.phaseCorrelate"],
        )
    reference_polar = reference_polar - reference_polar.mean()
    test_polar = test_polar - test_polar.mean()
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(reference_polar, test_polar)
    except cv2.error:
        return TransformCandidate(
            "similarity",
            np.eye(3),
            "rejected",
            "Log-polar phase correlation failed.",
            {"opencv_functions": ["cv2.warpPolar", "cv2.phaseCorrelate"]},
            functions=["cv2.warpPolar", "cv2.phaseCorrelate"],
        )
    angle = float(shift_x) * 360.0 / angle_bins
    if angle > 180.0:
        angle -= 360.0
    log_span = math.log(max(max_radius, 1.0))
    scale = math.exp(-float(shift_y) * log_span / max(radius_bins, 1))
    if not 0.5 <= scale <= 2.0:
        return TransformCandidate(
            "similarity",
            np.eye(3),
            "rejected",
            f"Log-polar estimate produced an implausible scale {scale:.3f}.",
            {"peak_response": float(response), "opencv_functions": ["cv2.warpPolar", "cv2.phaseCorrelate"]},
            functions=["cv2.warpPolar", "cv2.phaseCorrelate"],
        )
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = matrix
    plausibility = validate_transform(
        homography, "similarity", width, height, output_width, output_height
    )
    metrics = {
        "peak_response": float(response),
        "scale": scale,
        "rotation_degrees": angle,
        "overlap_fraction": plausibility["overlap_fraction"],
        "opencv_functions": ["cv2.warpPolar", "cv2.phaseCorrelate"],
    }
    if not plausibility["valid"]:
        return TransformCandidate(
            "similarity",
            homography,
            "rejected",
            "Log-polar estimate failed plausibility checks: " + plausibility["reason"],
            metrics,
            functions=["cv2.warpPolar", "cv2.phaseCorrelate"],
        )
    return TransformCandidate(
        "similarity",
        homography,
        "candidate",
        (
            f"Log-polar fallback estimated {angle:.2f} degrees rotation and "
            f"scale {scale:.3f} (response {response:.3f})."
        ),
        metrics,
        functions=["cv2.warpPolar", "cv2.phaseCorrelate"],
    )


# ---------------------------------------------------------------------------
# Warping and scoring
# ---------------------------------------------------------------------------


def warp_image(
    source: np.ndarray,
    matrix: np.ndarray,
    output_width: int,
    output_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    homography = to_homography(matrix)
    if is_perspective(homography):
        return cv2.warpPerspective(
            source,
            homography,
            (output_width, output_height),
            flags=interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return cv2.warpAffine(
        source,
        homography[:2, :].astype(np.float32),
        (output_width, output_height),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_mask(
    mask: np.ndarray,
    matrix: np.ndarray,
    output_width: int,
    output_height: int,
    erode: int = 2,
) -> np.ndarray:
    homography = to_homography(matrix)
    if is_perspective(homography):
        warped = cv2.warpPerspective(
            mask,
            homography,
            (output_width, output_height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = cv2.warpAffine(
            mask,
            homography[:2, :].astype(np.float32),
            (output_width, output_height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    warped = np.where(warped > 0, 255, 0).astype(np.uint8)
    if erode > 0:
        kernel_size = 2 * erode + 1
        warped = cv2.erode(warped, np.ones((kernel_size, kernel_size), np.uint8))
    return warped


def _standardized_residual(
    reference_gray: np.ndarray, test_gray: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    reference_values = reference_gray[mask > 0].astype(np.float64)
    test_values = test_gray[mask > 0].astype(np.float64)
    if reference_values.size < 8:
        return 0.0, 0.0, 0.0
    reference_median = float(np.median(reference_values))
    test_median = float(np.median(test_values))
    reference_mad = 1.4826 * float(np.median(np.abs(reference_values - reference_median)))
    test_mad = 1.4826 * float(np.median(np.abs(test_values - test_median)))
    reference_z = (reference_gray - reference_median) / max(reference_mad, 1.0)
    test_z = (test_gray - test_median) / max(test_mad, 1.0)
    difference = np.abs(reference_z - test_z)[mask > 0]
    if difference.size == 0:
        return 0.0, 0.0, 0.0
    trimmed = difference[difference <= np.percentile(difference, 90)]
    residual = float(np.mean(trimmed)) if trimmed.size else float(np.mean(difference))
    reference_centered = reference_values - reference_values.mean()
    test_centered = test_values - test_values.mean()
    denominator = float(np.linalg.norm(reference_centered) * np.linalg.norm(test_centered))
    ncc = (
        float(np.dot(reference_centered, test_centered) / denominator)
        if denominator > 1e-9
        else 0.0
    )
    return residual, ncc, reference_mad


def score_alignment(
    reference: np.ndarray,
    source: np.ndarray,
    matrix: np.ndarray,
    erode: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Warp ``source`` by ``matrix`` and measure robust overlap evidence."""
    output_height, output_width = reference.shape[:2]
    aligned = warp_image(source, matrix, output_width, output_height)
    mask = warp_mask(
        np.full(source.shape[:2], 255, dtype=np.uint8),
        matrix,
        output_width,
        output_height,
        erode=erode,
    )
    valid_pixels = int(np.count_nonzero(mask))
    overlap = valid_pixels / max(output_width * output_height, 1)
    reference_gray = cv2.GaussianBlur(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0
    )
    test_gray = cv2.GaussianBlur(
        cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0
    )
    residual, ncc, texture = _standardized_residual(reference_gray, test_gray, mask)
    raw = np.abs(reference_gray - test_gray)[mask > 0]
    raw_residual = (
        float(np.mean(raw[raw <= np.percentile(raw, 90)])) if raw.size else 0.0
    )
    metrics = {
        "residual_normalized": residual,
        "residual_raw": raw_residual,
        "ncc": ncc,
        "overlap_fraction": overlap,
        "valid_pixels": valid_pixels,
        "texture_mad": texture,
    }
    return aligned, mask, metrics


def candidate_acceptance(
    before: dict, after: dict, model: str
) -> tuple[bool, str]:
    """Decide whether a candidate is a clear enough improvement."""
    complexity = MODEL_ORDER.get(model, 3)
    # A fixed gain requirement is impossible once the current alignment is
    # already near-perfect.  Scale the requirement with the remaining
    # headroom so that a small but meaningful improvement can still count.
    headroom = max(0.0, 1.0 - float(before["ncc"]))
    required_ncc_gain = min(
        0.015 + 0.01 * complexity,
        max(0.0005, 0.25 * headroom),
    )
    ncc_gain = after["ncc"] - before["ncc"]
    before_residual = max(before["residual_normalized"], 1e-6)
    relative_residual_gain = (before_residual - after["residual_normalized"]) / before_residual
    if after["ncc"] < 0.80:
        return (
            False,
            f"Overlap correlation after the candidate is only {after['ncc']:.3f} (< 0.80).",
        )
    if ncc_gain < required_ncc_gain:
        return (
            False,
            (
                f"Correlation gain {ncc_gain:.3f} is below the {required_ncc_gain:.3f} "
                f"minimum for the {model} model."
            ),
        )
    if relative_residual_gain < 0.05 and after["ncc"] < 0.995:
        return (
            False,
            (
                f"Robust residual improved by only {relative_residual_gain * 100:.1f}% "
                "(< 5.0%) and the correlation is not near-perfect."
            ),
        )
    return (
        True,
        (
            f"Accepted: correlation {before['ncc']:.3f} -> {after['ncc']:.3f}, "
            f"robust residual {before['residual_normalized']:.3f} -> "
            f"{after['residual_normalized']:.3f} "
            f"({relative_residual_gain * 100:.1f}% relative improvement)."
        ),
    )


def cross_check(
    model: str,
    primary: TransformCandidate,
    secondary: TransformCandidate,
    reference_size: tuple[int, int],
) -> tuple[str, str]:
    """Cross-check independent estimates by mean corner displacement."""
    if primary.status != "candidate" or secondary.status != "candidate":
        return "unavailable", "An independent estimate was not available."
    width, height = reference_size
    if model == "similarity":
        primary_scale = float(primary.metrics.get("scale", 1.0))
        secondary_scale = float(secondary.metrics.get("scale", 1.0))
        if abs(primary_scale - secondary_scale) > 0.04:
            return (
                "skipped",
                (
                    "Independent ECC motion cannot represent the estimated scale "
                    f"({primary_scale:.3f} vs {secondary_scale:.3f}); cross-check skipped."
                ),
            )
    displacement = mean_corner_displacement(
        primary.matrix, secondary.matrix, width, height
    )
    if displacement <= 2.0:
        return (
            "agree",
            f"Independent estimates agree: mean corner displacement {displacement:.2f} px.",
        )
    if displacement > 5.0:
        return (
            "disagree",
            (
                f"Independent estimates disagree: mean corner displacement "
                f"{displacement:.2f} px (> 5.0 px). Neither candidate was applied."
            ),
        )
    return (
        "uncertain",
        f"Independent estimates are close but not decisive ({displacement:.2f} px).",
    )


# ---------------------------------------------------------------------------
# Scrubber interpolation
# ---------------------------------------------------------------------------


def interpolate_transform(
    final_matrix: np.ndarray,
    t: float,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Interpolate from identity to ``final_matrix`` for the sequence scrubber."""
    t = float(np.clip(t, 0.0, 1.0))
    target = to_homography(final_matrix)
    if t <= 0.0:
        return np.eye(3, dtype=np.float64)
    if t >= 1.0:
        return target.copy()

    if not is_perspective(target):
        a, b = target[0, 0], target[0, 1]
        c, d = target[1, 0], target[1, 1]
        scale_x = math.hypot(a, c)
        scale_y = math.hypot(b, d)
        scale = math.sqrt(max(scale_x * scale_y, 1e-12))
        angle = math.atan2(c, a)
        interpolated_scale = scale**t
        interpolated_angle = angle * t
        cos_angle = math.cos(interpolated_angle) * interpolated_scale
        sin_angle = math.sin(interpolated_angle) * interpolated_scale
        return np.array(
            [
                [cos_angle, -sin_angle, target[0, 2] * t],
                [sin_angle, cos_angle, target[1, 2] * t],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    corners = np.float64(
        [
            [0, 0, 1],
            [output_width - 1, 0, 1],
            [output_width - 1, output_height - 1, 1],
            [0, output_height - 1, 1],
        ]
    ).T
    destination = target @ corners
    destination = (destination[:2] / destination[2]).T
    source = corners[:2].T
    interpolated = source * (1.0 - t) + destination * t
    return to_homography(
        cv2.getPerspectiveTransform(
            source.astype(np.float32), interpolated.astype(np.float32)
        )
    )
