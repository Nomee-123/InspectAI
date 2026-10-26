"""Real telemetry and display data for the VISION CORE interface.

The Streamlit layer never invents a number: everything here is derived from
``inspect_images`` and ``build_inspection_report``.  Images are downscaled and
encoded as data URIs only for display; the JSON report keeps its no-pixels
contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import streamlit as st

from inspectai import build_inspection_report, center_crop_pair, decode_image, inspect_images
from inspectai.align import to_homography

BASE_DIRECTORY = Path(__file__).resolve().parent.parent
SAMPLES_DIRECTORY = BASE_DIRECTORY / "samples"
LOGO_PATH = BASE_DIRECTORY / "assets" / "opencv_logo.png"
LOGO_SVG_PATH = BASE_DIRECTORY / "assets" / "opencv_logo.svg"

STAGE_ORDER = [
    "acquire",
    "orientation",
    "translation",
    "register",
    "photometric",
    "analyze",
    "verify",
]

STAGE_LABELS = {
    "acquire": "ACQUIRE",
    "orientation": "ORIENT",
    "translation": "TRANSLATE",
    "register": "REGISTER",
    "photometric": "PHOTOMETRIC",
    "analyze": "ANALYZE",
    "verify": "VERIFY",
}


def capabilities() -> dict[str, object]:
    """Real local build details used by the header and boot sequence."""
    return {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "sift": hasattr(cv2, "SIFT_create"),
        "akaze": hasattr(cv2, "AKAZE_create"),
        "orb": hasattr(cv2, "ORB_create"),
        "ecc": hasattr(cv2, "findTransformECC"),
        "phase": hasattr(cv2, "phaseCorrelate"),
        "connected_components": hasattr(cv2, "connectedComponentsWithStats"),
        "threads": cv2.getNumThreads(),
    }


def opencv_logo_uri() -> tuple[str | None, str]:
    """Return the user-supplied logo, or a text fallback and its warning."""
    for path in (LOGO_PATH, LOGO_SVG_PATH):
        if path.is_file():
            mime = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}", ""
    return None, (
        "[InspectAI] assets/opencv_logo.png was not found; using the text badge "
        "'BUILT WITH OpenCV'. Place the official OpenCV logo at assets/opencv_logo.png "
        "to show it (the image is never generated or hotlinked)."
    )


def image_data_uri(image: np.ndarray | None, max_side: int = 1400, quality: int = 82) -> str:
    """Encode a BGR image as a JPEG data URI for display only."""
    if image is None or image.size == 0:
        return ""
    display = image
    height, width = display.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        display = cv2.resize(
            display,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _edge_image(image: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 40, 110)
    canvas = np.zeros((*edges.shape, 3), dtype=np.uint8)
    canvas[edges > 0] = colour
    return canvas


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
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


def _display_matrix(matrix: np.ndarray) -> list[list[float]]:
    """The library matrix samples ref->test; the UI needs test->ref."""
    homography = to_homography(matrix)
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        inverse = np.eye(3)
    return _json_safe(inverse)


def _sample_keypoints(
    reference_points: list | None, test_points: list | None
) -> tuple[list[list[float]], list[list[float]], list[bool]]:
    if not reference_points or not test_points:
        return [], [], []
    pairs = list(zip(reference_points, test_points))
    inliers = [pair for pair in pairs if bool(pair[0][2]) and bool(pair[1][2])]
    outliers = [pair for pair in pairs if not (bool(pair[0][2]) and bool(pair[1][2]))]
    chosen = inliers[:34] + outliers[:6]
    reference = [[float(pair[0][0]), float(pair[0][1])] for pair in chosen]
    test = [[float(pair[1][0]), float(pair[1][1])] for pair in chosen]
    flags = [bool(pair[0][2]) and bool(pair[1][2]) for pair in chosen]
    return reference, test, flags


def _step_payload(step: Any) -> dict[str, object]:
    reference_points, test_points, keypoint_flags = _sample_keypoints(
        step.keypoints_ref, step.keypoints_test
    )
    preview = step.preview
    metrics = _json_safe(step.metrics)
    return {
        "index": int(step.index),
        "name": step.name,
        "model": step.model,
        "status": step.status,
        "reason": step.reason,
        "matrix": _json_safe(step.matrix),
        "display_matrix": _display_matrix(step.matrix),
        "metrics": metrics,
        "preview_uri": image_data_uri(preview),
        "edge_uri": image_data_uri(_edge_image(preview, (0, 176, 255))) if preview is not None else "",
        "keypoints_ref": reference_points,
        "keypoints_test": test_points,
        "keypoint_flags": keypoint_flags,
        "has_keypoints": bool(reference_points),
        "residual_before": _first_float(
            metrics, "residual_before", "residual_raw", "response"
        ),
        "residual_after": _first_float(
            metrics, "residual_after", "residual_normalized", "ncc"
        ),
        "inlier_ratio": _first_float(metrics, "inlier_ratio"),
        "ecc_cc": _first_float(metrics, "ecc_cc"),
        "functions": _json_safe(step.metrics.get("opencv_functions", [])),
    }


def _first_float(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


GEOMETRIC_MODELS = {"orientation", "translation", "similarity", "affine", "homography", "log_polar"}


def _default_step_index(step_payloads: list[dict], alignment_status: str) -> int:
    """Pick the step that explains the final alignment result.

    An ambiguous result should open on an ambiguous candidate so the ghost
    comparison is visible.  Otherwise prefer the last accepted geometric
    transform, then another accepted step (for example photometric
    normalization), and finally the final step.
    """
    if not step_payloads:
        return 0
    if alignment_status == "ambiguous":
        ambiguous = [
            step
            for step in step_payloads
            if step["status"] == "ambiguous" and step.get("preview_uri")
        ]
        if ambiguous:
            return int(ambiguous[-1]["index"])
    geometric = [
        step
        for step in step_payloads
        if step["status"] == "accepted" and step["model"] in GEOMETRIC_MODELS
    ]
    if geometric:
        return int(geometric[-1]["index"])
    accepted = [
        step
        for step in step_payloads
        if step["status"] == "accepted" and step["name"] != "Original"
    ]
    if accepted:
        return int(accepted[-1]["index"])
    return int(step_payloads[-1]["index"])


def stage_telemetry(events: list[dict]) -> dict[str, dict]:
    """Collapse real progress events into per-stage display state."""
    telemetry: dict[str, dict] = {}
    for event in events:
        stage = event.get("stage")
        if stage not in STAGE_ORDER:
            continue
        entry = telemetry.setdefault(
            stage,
            {"state": "pending", "status": "", "functions": [], "detail": {}},
        )
        phase = event.get("phase")
        if phase == "start":
            entry["state"] = "running"
        elif phase == "finish":
            status = str(event.get("status", "done"))
            entry["state"] = "ambiguous" if status == "ambiguous" else (
                "rejected" if status == "rejected" else "done"
            )
            entry["status"] = status
            entry["detail"] = _json_safe(
                {key: value for key, value in event.items() if key not in {"functions", "phase"}}
            )
        if event.get("functions"):
            entry["functions"] = list(dict.fromkeys(
                entry.get("functions", []) + list(event["functions"])
            ))
    return telemetry


def _gauge_payload(result: Any) -> list[dict[str, object]]:
    final = result.final_comparison
    edge = result.independent_comparison
    changed = float(final.changed_percentage)
    edge_value = float(edge.edge_similarity_percentage) if edge.edge_similarity_percentage is not None else 0.0
    overlap = float(result.overlap_fraction or 0.0) * 100.0
    agreed = not bool(result.method_disagreement)

    def method_word(value: Any) -> str:
        if value is None:
            return "n/a"
        return "change" if bool(value) else "no change"

    lab_state = "change" if result.pixel_change_indicated else "no change"
    canny_state = method_word(result.edge_change_indicated)
    ssim_state = method_word(
        result.ssim_comparison.change_indicated if result.ssim_comparison else None
    )
    method_summary = f"Lab {lab_state} · Canny {canny_state} · SSIM {ssim_state}"
    alignment_tone = (
        "red"
        if result.alignment_status == "rejected"
        else "amber"
        if result.alignment_status == "ambiguous"
        else ""
    )
    changed_tone = "amber" if changed >= 5.0 else ("amber" if changed >= 1.0 else "")
    return [
        {
            "key": "changed",
            "label": "Changed pixels",
            "value": changed,
            "unit": "%",
            "numeric": True,
            "display": f"{changed:.2f}",
            "fraction": min(1.0, changed / 100.0),
            "tone": changed_tone,
            "sub": "Lab ΔE after alignment",
            "aria": (
                f"Changed pixels {changed:.2f} percent. "
                "Uncalibrated heuristic, not a defect probability."
            ),
        },
        {
            "key": "alignment",
            "label": "Alignment overlap",
            "value": overlap,
            "unit": "%",
            "numeric": True,
            "display": f"{overlap:.1f}",
            "fraction": min(1.0, overlap / 100.0),
            "tone": alignment_tone,
            "sub": f"{result.alignment_status} / {result.alignment_confidence}",
            "aria": (
                f"Alignment overlap {overlap:.1f} percent. "
                f"Status {result.alignment_status}. Uncalibrated heuristic."
            ),
        },
        {
            "key": "edge",
            "label": "Edge similarity",
            "value": edge_value,
            "unit": "%",
            "numeric": edge.edge_similarity_percentage is not None,
            "display": f"{edge_value:.1f}" if edge.edge_similarity_percentage is not None else "n/a",
            "fraction": min(1.0, edge_value / 100.0),
            "tone": "" if edge.edge_similarity_percentage is not None else "amber",
            "sub": edge.status,
            "aria": (
                f"Edge similarity {edge_value:.1f} percent. "
                "Uncalibrated heuristic, not a defect probability."
            ),
        },
        {
            "key": "agreement",
            "label": "Method agreement",
            "value": 100.0 if agreed else 0.0,
            "unit": "",
            "numeric": False,
            "display": "AGREE" if agreed else "SPLIT",
            "fraction": 1.0 if agreed else 0.0,
            "tone": "" if agreed else "amber",
            "sub": method_summary,
            "aria": (
                "Method agreement: "
                + method_summary
                + ". This is a heuristic comparison signal, not a verdict."
            ),
        },
    ]


def _construct_payload(
    reference_bytes: bytes,
    test_bytes: bytes,
    crop_enabled: bool,
    aspect_source: str,
    listener: Callable[[str, dict], None] | None = None,
) -> dict[str, object]:
    reference_image = decode_image(reference_bytes)
    test_image = decode_image(test_bytes)
    if crop_enabled:
        target = reference_image if aspect_source == "Reference image" else test_image
        comparison_reference, comparison_test = center_crop_pair(
            reference_image,
            test_image,
            target_width=target.shape[1],
            target_height=target.shape[0],
        )
    else:
        comparison_reference, comparison_test = reference_image, test_image

    events: list[dict] = []

    def on_progress(stage: str, detail: dict) -> None:
        event = {"stage": stage, **_json_safe(detail)}
        events.append(event)
        if listener is not None:
            listener(stage, detail)

    result = inspect_images(
        comparison_reference, comparison_test, progress_callback=on_progress
    )
    report = build_inspection_report(comparison_reference, comparison_test, result)

    regions = list(result.changed_regions.get("regions", []))
    final = result.final_comparison
    difference_mask = final.difference_mask
    if difference_mask.ndim == 2:
        mask_visual = np.zeros((*difference_mask.shape, 3), dtype=np.uint8)
        mask_visual[difference_mask > 0] = (255, 229, 0)
    else:
        mask_visual = difference_mask

    step_payloads = [_step_payload(step) for step in result.alignment_steps]
    default_index = _default_step_index(step_payloads, result.alignment_status)

    telemetry = stage_telemetry(events)
    final_test = result.final_test_image
    aligned_test = result.aligned_test_image
    if aligned_test is None:
        aligned_test = final_test

    payload: dict[str, object] = {
        "signature": hashlib.sha256(
            reference_bytes + b"::" + test_bytes + f"::{crop_enabled}::{aspect_source}".encode()
        ).hexdigest(),
        "center_cropped": bool(crop_enabled),
        "aspect_source": aspect_source,
        "dimensions": {
            "reference": [int(comparison_reference.shape[1]), int(comparison_reference.shape[0])],
            "test": [int(comparison_test.shape[1]), int(comparison_test.shape[0])],
        },
        "reference_uri": image_data_uri(comparison_reference),
        "test_uri": image_data_uri(comparison_test),
        "aligned_uri": image_data_uri(aligned_test),
        "heatmap_uri": image_data_uri(final.heatmap),
        "mask_uri": image_data_uri(mask_visual),
        "ref_edges_uri": image_data_uri(_edge_image(comparison_reference, (255, 229, 0))),
        "regions": _json_safe(regions),
        "region_count": int(result.changed_regions.get("region_count", 0)),
        "detected_region_count": int(result.changed_regions.get("detected_region_count", 0)),
        "filtered_region_count": int(result.changed_regions.get("filtered_region_count", 0)),
        "minimum_region_area": int(result.changed_regions.get("minimum_area_pixels", 0)),
        "gauges": _json_safe(_gauge_payload(result)),
        "alignment": {
            "status": result.alignment_status,
            "method": result.alignment_method,
            "applied": bool(result.alignment_applied),
            "confidence": result.alignment_confidence,
            "rotation": float(result.estimated_rotation_degrees),
            "scale": float(result.estimated_scale),
            "shift": [float(result.estimated_shift[0]), float(result.estimated_shift[1])],
            "overlap": float(result.overlap_fraction or 0.0),
            "response": float(result.alignment_response),
            "final_matrix": _json_safe(result.final_matrix),
        },
        "measurements": {
            "initial_changed": float(result.initial_comparison.changed_percentage),
            "final_changed": float(final.changed_percentage),
            "threshold": float(final.threshold),
            "noise_mad": float(final.noise_mad),
            "valid_pixels": int(final.valid_pixels),
            "edge_similarity": _json_safe(result.independent_comparison.edge_similarity_percentage),
            "edge_disagreement": _json_safe(result.independent_comparison.edge_disagreement_percentage),
            "edge_status": result.independent_comparison.status,
            "ssim": _json_safe(
                result.ssim_comparison.mean_ssim if result.ssim_comparison else None
            ),
            "ssim_status": result.ssim_comparison.status if result.ssim_comparison else "unavailable",
            "photometric_applied": bool(result.photometric.applied) if result.photometric else False,
            "photometric_reason": result.photometric.reason if result.photometric else "",
            "photometric_improvement": _json_safe(
                result.photometric.improvement_relative if result.photometric else None
            ),
            "method_disagreement": bool(result.method_disagreement),
            "indication_agreement": result.indication_agreement,
            "pixel_change_indicated": bool(result.pixel_change_indicated),
            "edge_change_indicated": _json_safe(result.edge_change_indicated),
            "ssim_change_indicated": _json_safe(
                result.ssim_comparison.change_indicated if result.ssim_comparison else None
            ),
            "resized_test_image": bool(result.resized_test_image),
        },
        "evidence": list(result.evidence),
        "decisions": list(result.decisions),
        "warnings": list(result.uncertainty_warnings),
        "limitations": list(result.limitations),
        "agent_trace": [
            {
                "index": int(iteration.index),
                "action": iteration.action,
                "evidence": list(iteration.evidence),
                "outcome": iteration.outcome,
                "uncertainty": iteration.uncertainty,
                "metrics": _json_safe(iteration.metrics),
            }
            for iteration in result.agent_trace
        ],
        "function_trace": _json_safe(result.function_trace),
        "events": events,
        "telemetry": telemetry,
        "steps": step_payloads,
        "register": {
            "width": int(comparison_reference.shape[1]),
            "height": int(comparison_reference.shape[0]),
            "reference_uri": image_data_uri(comparison_reference),
            "test_uri": image_data_uri(comparison_test),
            "ref_edges_uri": image_data_uri(_edge_image(comparison_reference, (255, 229, 0))),
            "steps": step_payloads,
            "default_index": default_index,
        },
        "report": report,
        "report_json": json.dumps(report, indent=2, allow_nan=False),
    }
    return payload


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_payload(
    reference_bytes: bytes,
    test_bytes: bytes,
    crop_enabled: bool,
    aspect_source: str,
) -> dict[str, object]:
    """Cache the finished display payload, never the live UI callback."""
    return _construct_payload(reference_bytes, test_bytes, crop_enabled, aspect_source)


def run_inspection(
    reference_bytes: bytes,
    test_bytes: bytes,
    crop_enabled: bool,
    aspect_source: str,
    listener: Callable[[str, dict], None] | None = None,
) -> tuple[dict[str, object], bool]:
    """Return a payload and whether this call emitted real progress events.

    A listener can only be honoured on a real pipeline run.  Streamlit cached
    functions must not create UI elements, so a call that carries a listener
    runs uncached; the finished payload is still stored in session state by the
    app, and the listener-free path uses ``st.cache_data``.
    """
    if listener is not None:
        payload = _construct_payload(
            reference_bytes, test_bytes, crop_enabled, aspect_source, listener
        )
        return payload, True

    payload = _cached_payload(reference_bytes, test_bytes, crop_enabled, aspect_source)
    return payload, False


def load_manifest() -> dict[str, object]:
    path = SAMPLES_DIRECTORY / "manifest.json"
    if not path.is_file():
        return {"cases": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cases": []}


def load_demo_bytes(case: dict[str, object]) -> tuple[bytes, bytes, str, str]:
    reference_path = SAMPLES_DIRECTORY / str(case["reference"])
    test_path = SAMPLES_DIRECTORY / str(case["test"])
    return (
        reference_path.read_bytes(),
        test_path.read_bytes(),
        reference_path.name,
        test_path.name,
    )
