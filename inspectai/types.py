"""Shared data model for InspectAI.

The core package deliberately avoids importing Streamlit so that the same
pipeline can run in a CLI, a test, or a future stateless AWS Lambda handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ComparisonResult:
    """Result of the primary, colour-aware visual difference measurement."""

    changed_percentage: float
    difference_mask: np.ndarray
    heatmap: np.ndarray
    threshold: float = 0.0
    noise_median: float = 0.0
    noise_mad: float = 0.0
    minimum_region_area: int = 16
    minimum_component_contrast: float = 0.0
    largest_region_area: int = 0
    largest_region_contrast: float = 0.0
    region_change_indicated: bool = False
    global_change_indicated: bool = False
    valid_pixels: int = 0
    cleanup: str = ""
    region_filter: str = ""
    filtered_region_count: int = 0
    filtered_pixel_count: int = 0
    opencv_functions: list[str] = field(default_factory=list)


@dataclass
class EdgeComparisonResult:
    edge_similarity_percentage: float | None
    edge_disagreement_percentage: float | None
    reference_edge_pixels: int
    test_edge_pixels: int
    tolerance_pixels: int
    status: str
    canny_low: int | None = None
    canny_high: int | None = None
    change_indicated: bool | None = None
    opencv_functions: list[str] = field(default_factory=list)


@dataclass
class SsimComparisonResult:
    mean_ssim: float | None
    changed_percentage: float | None
    changed_mask: np.ndarray | None
    ssim_map: np.ndarray | None
    threshold: float
    status: str
    change_indicated: bool | None = None
    opencv_functions: list[str] = field(default_factory=list)


@dataclass
class PhotometricResult:
    applied: bool
    status: str
    reason: str
    residual_before: float | None = None
    residual_after: float | None = None
    improvement_relative: float | None = None
    channel_gain: list[float] = field(default_factory=list)
    channel_offset: list[float] = field(default_factory=list)
    channel_gradient: list[list[float]] = field(default_factory=list)
    model: str = "gain_offset_plane"
    opencv_functions: list[str] = field(default_factory=list)
    normalized_image: np.ndarray | None = field(default=None, repr=False)


@dataclass
class AlignmentStep:
    """One inspectable step of the alignment process.

    ``matrix`` is a 3x3 transform from *reference-canvas* coordinates to the
    source image of this step.  For accepted steps it is cumulative and maps
    canvas coordinates back to the original test image; pixels are sampled
    from that source with ``WARP_INVERSE_MAP`` semantics, the convention used
    by ``cv2.findTransformECC`` and ``cv2.phaseCorrelate``.

    ``keypoints_ref`` / ``keypoints_test`` contain matched ``(x, y, inlier)``
    tuples for feature-based steps, or ``None`` for other steps.
    """

    index: int
    name: str
    model: str
    matrix: np.ndarray
    status: str
    reason: str
    metrics: dict = field(default_factory=dict)
    preview: np.ndarray | None = field(default=None, repr=False)
    keypoints_ref: list | None = None
    keypoints_test: list | None = None


@dataclass
class AgentIteration:
    """One loop iteration: action, evidence, outcome, uncertainty."""

    index: int
    action: str
    evidence: list[str]
    outcome: str
    uncertainty: str
    metrics: dict = field(default_factory=dict)


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
    alignment_steps: list[AlignmentStep] = field(default_factory=list)
    agent_trace: list[AgentIteration] = field(default_factory=list)
    ssim_comparison: SsimComparisonResult | None = None
    photometric: PhotometricResult | None = None
    aligned_test_image: np.ndarray | None = field(default=None, repr=False)
    final_test_image: np.ndarray | None = field(default=None, repr=False)
    final_valid_mask: np.ndarray | None = field(default=None, repr=False)
    final_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    estimated_rotation_degrees: float = 0.0
    estimated_scale: float = 1.0
    alignment_confidence: str = "none"
    overlap_fraction: float | None = None
    function_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TransformCandidate:
    """A geometric candidate produced by one estimator for internal use."""

    model: str
    matrix: np.ndarray
    status: str
    reason: str
    metrics: dict = field(default_factory=dict)
    aligned: np.ndarray | None = field(default=None, repr=False)
    valid_mask: np.ndarray | None = field(default=None, repr=False)
    keypoints_ref: list | None = None
    keypoints_test: list | None = None
    functions: list[str] = field(default_factory=list)
