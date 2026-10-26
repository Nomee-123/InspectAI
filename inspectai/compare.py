"""Comparison layer: photometric fit, Lab difference, edges and SSIM.

Everything here is heuristic and uncalibrated.  The functions measure visual
differences; they do not establish whether a real-world defect exists.
"""

from __future__ import annotations

import cv2
import numpy as np

from .regions import adaptive_minimum_area, cleanup_mask, detect_changed_regions
from .types import (
    ComparisonResult,
    EdgeComparisonResult,
    PhotometricResult,
    SsimComparisonResult,
)

MIN_LAB_THRESHOLD = 2.5


def _validate_pair(reference: np.ndarray, test: np.ndarray, purpose: str) -> None:
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        raise ValueError("Both images must contain pixel data.")
    if reference.shape[:2] != test.shape[:2]:
        raise ValueError(f"Images must have matching dimensions before {purpose}.")


def _mask_or_full(valid_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if valid_mask is None:
        return np.full(shape, 255, dtype=np.uint8)
    if valid_mask.shape != shape:
        raise ValueError("The valid-area mask must match the image dimensions.")
    return np.where(valid_mask > 0, 255, 0).astype(np.uint8)


def _sampled_valid_pixels(valid_mask: np.ndarray, maximum: int = 30000) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(valid_mask)
    count = len(xs)
    if count > maximum:
        step = max(1, count // maximum)
        ys = ys[::step]
        xs = xs[::step]
    return ys, xs


def _fit_channel_plane(
    reference_channel: np.ndarray,
    test_channel: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Robustly fit ref = c0 + c1*x + c2*y + c3*test + c4*test*x + c5*test*y."""
    xn = (xs.astype(np.float64) / max(width - 1, 1)) * 2.0 - 1.0
    yn = (ys.astype(np.float64) / max(height - 1, 1)) * 2.0 - 1.0
    ref = reference_channel[ys, xs].astype(np.float64)
    value = test_channel[ys, xs].astype(np.float64)
    design = np.stack(
        [np.ones_like(value), xn, yn, value, value * xn, value * yn], axis=1
    )

    keep = np.ones(len(ref), dtype=bool)
    coefficients: np.ndarray | None = None
    for _ in range(3):
        if int(keep.sum()) < 12:
            return None
        try:
            coefficients, *_ = np.linalg.lstsq(design[keep], ref[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None
        residual = ref - design @ coefficients
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        limit = max(3.0, 3.5 * scale)
        new_keep = np.abs(residual - np.median(residual)) <= limit
        if int(new_keep.sum()) < 12:
            break
        keep = new_keep

    if coefficients is None or not np.all(np.isfinite(coefficients)):
        return None
    # Reject implausible photometric models instead of silently warping colours.
    gain = float(coefficients[3])
    if not 0.3 <= gain <= 3.0:
        return None
    gradient = np.hypot(float(coefficients[4]), float(coefficients[5]))
    if gradient > 1.5:
        return None
    return coefficients


def _apply_channel_plane(
    channel: np.ndarray,
    coefficients: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    yn, xn = np.mgrid[0:height, 0:width]
    xn = (xn.astype(np.float64) / max(width - 1, 1)) * 2.0 - 1.0
    yn = (yn.astype(np.float64) / max(height - 1, 1)) * 2.0 - 1.0
    value = channel.astype(np.float64)
    predicted = (
        coefficients[0]
        + coefficients[1] * xn
        + coefficients[2] * yn
        + coefficients[3] * value
        + coefficients[4] * value * xn
        + coefficients[5] * value * yn
    )
    return np.clip(predicted, 0, 255).astype(np.uint8)


def _trimmed_mean(values: np.ndarray, trim: float = 0.9) -> float:
    if values.size == 0:
        return 0.0
    limit = np.percentile(values, trim * 100.0)
    kept = values[values <= limit]
    return float(np.mean(kept)) if kept.size else float(np.mean(values))


def _gray_residual(reference: np.ndarray, test: np.ndarray, valid_mask: np.ndarray) -> float:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32)
    difference = np.abs(reference_gray - test_gray)
    values = difference[valid_mask > 0]
    return _trimmed_mean(values)


def fit_photometric_normalization(
    reference: np.ndarray,
    test: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> PhotometricResult:
    """Fit a robust per-channel gain/offset-plus-gradient map test -> reference.

    The fit is deliberately restricted to a low-order model.  It can absorb
    smooth lighting changes but cannot explain compact real changes, which
    therefore survive the normalization.
    """
    _validate_pair(reference, test, "photometric normalization")
    height, width = reference.shape[:2]
    mask = _mask_or_full(valid_mask, (height, width))
    ys, xs = _sampled_valid_pixels(mask)
    if len(xs) < 50:
        return PhotometricResult(
            applied=False,
            status="skipped",
            reason="Too few valid overlap pixels for a stable photometric fit.",
            opencv_functions=["np.linalg.lstsq"],
        )

    before = _gray_residual(reference, test, mask)
    if before < 0.5:
        return PhotometricResult(
            applied=False,
            status="skipped",
            reason="The overlap residual is already close to zero; photometric correction is unnecessary.",
            residual_before=before,
            residual_after=before,
            improvement_relative=0.0,
            opencv_functions=["np.linalg.lstsq"],
        )

    normalized = test.copy()
    gains: list[float] = []
    offsets: list[float] = []
    gradients: list[list[float]] = []
    for channel_index in range(3):
        reference_channel = reference[:, :, channel_index]
        test_channel = test[:, :, channel_index]
        coefficients = _fit_channel_plane(
            reference_channel, test_channel, ys, xs, height, width
        )
        if coefficients is None:
            return PhotometricResult(
                applied=False,
                status="rejected",
                reason=(
                    "The photometric fit was unstable or implausible for at least one channel; "
                    "no colour correction was applied."
                ),
                residual_before=before,
                residual_after=before,
                improvement_relative=0.0,
                opencv_functions=["np.linalg.lstsq"],
            )
        normalized[:, :, channel_index] = _apply_channel_plane(
            test_channel, coefficients, height, width
        )
        gains.append(float(coefficients[3]))
        offsets.append(float(coefficients[0]))
        gradients.append([float(coefficients[1]), float(coefficients[2])])

    after = _gray_residual(reference, normalized, mask)
    improvement = (before - after) / before if before > 1e-6 else 0.0
    applied = improvement >= 0.08 and after < before
    status = "applied" if applied else "not_needed"
    reason = (
        f"Smooth per-channel gain/offset/gradient fit reduced the robust overlap residual "
        f"from {before:.2f} to {after:.2f} intensity units "
        f"({improvement * 100:.1f}% relative)."
        if applied
        else (
            f"The photometric fit did not reduce the robust overlap residual enough "
            f"({before:.2f} -> {after:.2f}); it was not applied."
        )
    )
    return PhotometricResult(
        applied=applied,
        status=status,
        reason=reason,
        residual_before=before,
        residual_after=after,
        improvement_relative=improvement,
        channel_gain=gains,
        channel_offset=offsets,
        channel_gradient=gradients,
        opencv_functions=["cv2.cvtColor", "np.linalg.lstsq"],
        normalized_image=normalized if applied else None,
    )


def _lab_delta(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    reference_lab = cv2.cvtColor(
        reference.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab
    )
    test_lab = cv2.cvtColor(test.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    delta = np.linalg.norm(reference_lab - test_lab, axis=2)
    return delta.astype(np.float32)


def estimate_noise_mad(delta: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    """Estimate local difference noise with a median filter and a MAD."""
    if delta.size == 0:
        return 0.0, 0.0
    local_median = cv2.medianBlur(delta.astype(np.float32), 3)
    high_frequency = np.abs(delta - local_median)
    values = high_frequency[valid_mask > 0]
    if values.size == 0:
        return 0.0, 0.0
    mad = float(np.median(values))
    return float(np.median(delta[valid_mask > 0])), 1.4826 * mad


def compare_images(
    reference: np.ndarray,
    test: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> ComparisonResult:
    """Compare same-sized BGR images in Lab space with an adaptive threshold.

    The primary measurement is a CIE-Lab Euclidean difference (a ΔE-style
    metric), which keeps colour-only changes visible.  The threshold is
    derived from high-frequency difference noise instead of a fixed intensity
    value.  Change indication is region-aware, so a compact high-contrast
    change is not hidden by a global percentage rule.
    """
    _validate_pair(reference, test, "comparison")
    height, width = reference.shape[:2]
    mask = _mask_or_full(valid_mask, (height, width))
    valid_pixels = int(np.count_nonzero(mask))

    delta = _lab_delta(reference, test)
    delta[mask == 0] = 0.0
    delta_median, noise_mad = estimate_noise_mad(delta, mask)
    threshold = max(MIN_LAB_THRESHOLD, 3.5 * noise_mad)
    raw_mask = np.where((delta > threshold) & (mask > 0), 255, 0).astype(np.uint8)
    cleaned, cleanup_rule = cleanup_mask(raw_mask)
    cleaned[mask == 0] = 0

    # Reject components that are too small or too faint to report.  Geometric
    # resampling leaves 1-3 pixel edge slivers, and photometric normalization
    # leaves sub-visible rounding residue; neither is a reportable change.
    minimum_area = adaptive_minimum_area(valid_pixels)
    minimum_component_contrast = max(6.0, 1.5 * threshold)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    flat_labels = labels.ravel()
    component_pixels = np.bincount(flat_labels, minlength=component_count)
    component_totals = np.bincount(
        flat_labels, weights=delta.ravel(), minlength=component_count
    )
    component_means = np.divide(
        component_totals,
        component_pixels,
        out=np.zeros(component_count, dtype=np.float64),
        where=component_pixels > 0,
    )
    kept_mask = np.zeros_like(cleaned)
    regions: list[dict[str, int]] = []
    filtered_region_count = 0
    filtered_pixel_count = 0
    largest_area = 0
    largest_contrast = 0.0
    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[component_index]
        )
        mean_contrast = float(component_means[component_index])
        keep = area >= minimum_area and mean_contrast >= minimum_component_contrast
        if not keep:
            filtered_region_count += 1
            filtered_pixel_count += area
            continue
        kept_mask[labels == component_index] = 255
        regions.append(
            {
                "x": x,
                "y": y,
                "width": component_width,
                "height": component_height,
                "area_pixels": area,
            }
        )
        if area > largest_area:
            largest_area = area
            largest_contrast = mean_contrast
    regions.sort(key=lambda region: (-region["area_pixels"], region["y"], region["x"]))

    changed_pixels = int(np.count_nonzero(kept_mask))
    changed_percentage = (
        100.0 * changed_pixels / valid_pixels if valid_pixels else 0.0
    )
    global_indicated = changed_percentage >= 1.0
    minimum_contrast = max(8.0, 2.5 * threshold)
    region_filter = (
        f"components below {minimum_area} px or mean Lab difference below "
        f"{minimum_component_contrast:.2f} are excluded"
    )
    region_indicated = (
        largest_area >= minimum_area
        and valid_pixels > 0
        and (largest_area / valid_pixels) >= 0.0005
        and largest_contrast >= minimum_contrast
    )

    heatmap_scale = np.clip(delta / max(threshold * 6.0, 12.0), 0.0, 1.0)
    heatmap_gray = (heatmap_scale * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_TURBO)
    heatmap[(kept_mask == 0) | (mask == 0)] = 0

    return ComparisonResult(
        changed_percentage=changed_percentage,
        difference_mask=kept_mask,
        heatmap=heatmap,
        threshold=threshold,
        noise_median=delta_median,
        noise_mad=noise_mad,
        minimum_region_area=minimum_area,
        minimum_component_contrast=minimum_component_contrast,
        largest_region_area=largest_area,
        largest_region_contrast=largest_contrast,
        region_change_indicated=region_indicated,
        global_change_indicated=global_indicated,
        valid_pixels=valid_pixels,
        cleanup=cleanup_rule,
        region_filter=region_filter,
        filtered_region_count=filtered_region_count,
        filtered_pixel_count=filtered_pixel_count,
        opencv_functions=[
            "cv2.cvtColor",
            "cv2.medianBlur",
            "cv2.connectedComponentsWithStats",
            "cv2.morphologyEx",
            "cv2.applyColorMap",
        ],
    )


def _auto_canny(gray: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Canny with thresholds derived from the image median intensity."""
    median = float(np.median(gray))
    low = int(np.clip(max(10.0, 0.4 * median), 10, 160))
    high = int(np.clip(max(low + 20.0, 1.2 * median), low + 20, 255))
    return cv2.Canny(gray, low, high), low, high


def compare_edges(
    reference: np.ndarray,
    test: np.ndarray,
    valid_mask: np.ndarray | None = None,
    tolerance_pixels: int = 2,
) -> EdgeComparisonResult:
    """Measure tolerant Canny-edge agreement independently of pixel intensity."""
    _validate_pair(reference, test, "edge comparison")
    if tolerance_pixels < 0:
        raise ValueError("Edge tolerance must be zero or greater.")

    height, width = reference.shape[:2]
    mask = _mask_or_full(valid_mask, (height, width))
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    reference_edges, low, high = _auto_canny(reference_gray)
    test_edges, _, _ = _auto_canny(test_gray)
    reference_edges[mask == 0] = 0
    test_edges[mask == 0] = 0
    reference_count = int(cv2.countNonZero(reference_edges))
    test_count = int(cv2.countNonZero(test_edges))
    functions = ["cv2.cvtColor", "cv2.Canny", "cv2.distanceTransform"]
    if reference_count < 4 or test_count < 4:
        return EdgeComparisonResult(
            edge_similarity_percentage=None,
            edge_disagreement_percentage=None,
            reference_edge_pixels=reference_count,
            test_edge_pixels=test_count,
            tolerance_pixels=tolerance_pixels,
            status="insufficient_evidence",
            canny_low=low,
            canny_high=high,
            opencv_functions=functions,
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
    disagreement = 100.0 - similarity
    return EdgeComparisonResult(
        edge_similarity_percentage=similarity,
        edge_disagreement_percentage=disagreement,
        reference_edge_pixels=reference_count,
        test_edge_pixels=test_count,
        tolerance_pixels=tolerance_pixels,
        status="measured",
        canny_low=low,
        canny_high=high,
        change_indicated=disagreement >= 10.0,
        opencv_functions=functions,
    )


def compare_ssim(
    reference: np.ndarray,
    test: np.ndarray,
    valid_mask: np.ndarray | None = None,
    similarity_threshold: float = 0.7,
) -> SsimComparisonResult:
    """Compute a local SSIM map with Gaussian windows (no scikit-image)."""
    _validate_pair(reference, test, "SSIM comparison")
    height, width = reference.shape[:2]
    mask = _mask_or_full(valid_mask, (height, width))
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    window = (11, 11)
    sigma = 1.5

    mu_reference = cv2.GaussianBlur(reference_gray, window, sigma)
    mu_test = cv2.GaussianBlur(test_gray, window, sigma)
    mu_reference_sq = mu_reference * mu_reference
    mu_test_sq = mu_test * mu_test
    mu_product = mu_reference * mu_test
    sigma_reference_sq = cv2.GaussianBlur(reference_gray * reference_gray, window, sigma) - mu_reference_sq
    sigma_test_sq = cv2.GaussianBlur(test_gray * test_gray, window, sigma) - mu_test_sq
    sigma_product = cv2.GaussianBlur(reference_gray * test_gray, window, sigma) - mu_product

    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_product + c2)
    denominator = (mu_reference_sq + mu_test_sq + c1) * (sigma_reference_sq + sigma_test_sq + c2)
    ssim_map = np.where(denominator != 0, numerator / denominator, 1.0)
    ssim_map = np.nan_to_num(ssim_map, nan=1.0, posinf=1.0, neginf=1.0)
    ssim_map = np.clip(ssim_map, -1.0, 1.0)
    valid_values = ssim_map[mask > 0]
    if valid_values.size == 0:
        return SsimComparisonResult(
            mean_ssim=None,
            changed_percentage=None,
            changed_mask=None,
            ssim_map=None,
            threshold=similarity_threshold,
            status="insufficient_evidence",
            opencv_functions=["cv2.cvtColor", "cv2.GaussianBlur"],
        )

    changed_mask = np.where((ssim_map < similarity_threshold) & (mask > 0), 255, 0).astype(np.uint8)
    minimum_area = adaptive_minimum_area(int(valid_values.size))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        changed_mask, connectivity=8
    )
    changed_pixels = int(np.count_nonzero(changed_mask))
    changed_percentage = 100.0 * changed_pixels / max(int(valid_values.size), 1)
    largest_region = 0
    if component_count > 1:
        largest_region = int(stats[1:, cv2.CC_STAT_AREA].max())
    change_indicated = changed_percentage >= 0.5 or largest_region >= minimum_area
    return SsimComparisonResult(
        mean_ssim=float(np.mean(valid_values)),
        changed_percentage=changed_percentage,
        changed_mask=changed_mask,
        ssim_map=ssim_map,
        threshold=similarity_threshold,
        status="measured",
        change_indicated=change_indicated,
        opencv_functions=["cv2.cvtColor", "cv2.GaussianBlur", "cv2.connectedComponentsWithStats"],
    )
