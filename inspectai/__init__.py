"""InspectAI core: evidence-driven visual comparison.

The public names are re-exported here so existing ``from inspectai import ...``
imports keep working.  The core has no Streamlit dependency.
"""

from .agent import (
    DEFAULT_MAX_ACTIONS,
    LIMITATIONS,
    center_crop_pair,
    decode_image,
    inspect_images,
)
from .align import (
    assess_orientation,
    interpolate_transform,
    phase_correlation,
    score_alignment,
    warp_image,
    warp_mask,
)
from .compare import (
    compare_edges,
    compare_images,
    compare_ssim,
    fit_photometric_normalization,
)
from .regions import adaptive_minimum_area, cleanup_mask, detect_changed_regions
from .report import build_inspection_report
from .types import (
    AgentIteration,
    AlignmentStep,
    ComparisonResult,
    EdgeComparisonResult,
    InspectionResult,
    PhotometricResult,
    SsimComparisonResult,
    TransformCandidate,
)

__all__ = [
    "AgentIteration",
    "AlignmentStep",
    "ComparisonResult",
    "DEFAULT_MAX_ACTIONS",
    "EdgeComparisonResult",
    "InspectionResult",
    "LIMITATIONS",
    "PhotometricResult",
    "SsimComparisonResult",
    "TransformCandidate",
    "adaptive_minimum_area",
    "assess_orientation",
    "build_inspection_report",
    "center_crop_pair",
    "cleanup_mask",
    "compare_edges",
    "compare_images",
    "compare_ssim",
    "decode_image",
    "detect_changed_regions",
    "fit_photometric_normalization",
    "inspect_images",
    "interpolate_transform",
    "phase_correlation",
    "score_alignment",
    "warp_image",
    "warp_mask",
]
