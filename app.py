"""Streamlit dashboard for the InspectAI evidence-driven inspection workflow.

The dashboard is intentionally explicit about uncertainty.  It shows the
measurements, the alignment ladder, the rejected candidates and the agent
trace; it never converts them into a defect verdict or a calibrated
probability.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from inspectai import (
    build_inspection_report,
    center_crop_pair,
    decode_image,
    inspect_images,
    interpolate_transform,
    warp_image,
)

BASE_DIRECTORY = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIRECTORY / "assets" / "opencv_logo.png"
LOGO_FALLBACK_TEXT = "BUILT WITH OpenCV"

st.set_page_config(
    page_title="InspectAI | Evidence-Driven Visual Inspection",
    page_icon="🔎",
    layout="wide",
)

STATUS_LABELS = {
    "applied": "Alignment applied",
    "not_needed": "No alignment needed",
    "ambiguous": "Alignment ambiguous",
    "rejected": "Alignment rejected",
}
STATUS_CLASSES = {
    "applied": "ok",
    "not_needed": "ok",
    "ambiguous": "warn",
    "rejected": "bad",
}
AGREEMENT_LABELS = {
    "both_indicate_change": "Both methods indicate change",
    "pixel_only_indicates_change": "Pixel difference only",
    "edge_only_indicates_change": "Edges only",
    "neither_indicates_change": "Neither indicates change",
    "insufficient_edge_evidence": "Not scored",
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def cached_decode(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes once per distinct payload."""
    return decode_image(image_bytes)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_inspection(
    reference_bytes: bytes,
    test_bytes: bytes,
    crop_enabled: bool,
    aspect_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """Run the full inspection for a byte-identical input pair."""
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
    result = inspect_images(comparison_reference, comparison_test)
    return (
        reference_image,
        test_image,
        comparison_reference,
        comparison_test,
        result,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def format_aspect_ratio(width: int, height: int) -> str:
    ratio = Fraction(width, height)
    return f"{ratio.numerator}:{ratio.denominator} ({width / height:.3f})"


def format_number(value: object, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "Not scored"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not np.isfinite(number):
        return "Not scored"
    return f"{number:.{digits}f}{suffix}"


def format_metric_value(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return "not finite"
        if abs(number) >= 1000:
            return f"{number:,.1f}"
        if abs(number) >= 10:
            return f"{number:.2f}"
        return f"{number:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_metric_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return html.escape(str(value))


def step_label(step: object) -> str:
    return f"{step.index + 1:02d} · {step.name} · {step.status}"


def status_pill(label: str, state: str) -> str:
    return f'<span class="status-pill {html.escape(state)}">{html.escape(label)}</span>'


def _fraction(value: object, maximum: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number) or maximum <= 0:
        return 0.0
    return float(np.clip(number / maximum, 0.0, 1.0))


def _gauge_color(fraction: float, inverted: bool = False) -> str:
    if inverted:
        return "#e8b45b" if fraction >= 0.2 else "#48c8e8"
    return "#68d6a0" if fraction >= 0.75 else "#e8b45b"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --surface: #08111f;
        --surface-raised: #0e1b2d;
        --surface-soft: #0b1726;
        --line: #223950;
        --text: #e8f1fa;
        --muted: #93a8bd;
        --cyan: #4cc9e9;
        --green: #68d6a0;
        --amber: #e8b45b;
        --red: #f07c86;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% -5%, rgba(76, 201, 233, 0.18), transparent 34%),
            radial-gradient(circle at 92% 0%, rgba(92, 150, 242, 0.13), transparent 30%),
            linear-gradient(155deg, #07101d 0%, #0a1524 52%, #0b1829 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(7, 16, 29, 0.78); }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1480px;
        padding: 0.6rem 2.1rem 2.2rem;
    }
    html, body, .stApp {
        font-family: "Bahnschrift", "Aptos", "Segoe UI", sans-serif;
        letter-spacing: 0;
    }
    h1, h2, h3 { color: var(--text); letter-spacing: 0; }
    h2 { font-size: 1.2rem; }
    h3 { font-size: 1rem; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    .inspect-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 4px 0 13px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 13px;
        flex-wrap: wrap;
    }
    .brand-lockup { display: flex; align-items: center; gap: 13px; min-width: 0; }
    .brand-logo { max-height: 46px; max-width: 190px; object-fit: contain; }
    .brand-textbadge {
        display: inline-flex;
        align-items: center;
        min-height: 34px;
        padding: 5px 11px;
        border: 1px solid #35637b;
        border-radius: 5px;
        color: #bdeffb;
        background: rgba(21, 55, 75, 0.55);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .brand-title { color: var(--text); font-size: 1.16rem; font-weight: 750; line-height: 1.15; }
    .brand-tagline { color: #c3d3e2; font-size: 0.78rem; margin-top: 3px; }
    .brand-subtitle { color: var(--muted); font-size: 0.71rem; margin-top: 2px; }
    .header-right { margin-left: auto; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .header-status, .status-pill {
        border: 1px solid #2a4f66;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        white-space: nowrap;
        color: #a9d9e7;
        background: rgba(13, 37, 53, 0.7);
    }
    .status-pill.ok { color: #bcf3d5; border-color: #2c6b50; background: rgba(25, 74, 55, 0.55); }
    .status-pill.warn { color: #f5d69a; border-color: #7b5d28; background: rgba(83, 62, 22, 0.55); }
    .status-pill.bad { color: #ffc1c6; border-color: #7e3940; background: rgba(85, 35, 41, 0.55); }
    .section-kicker {
        color: var(--cyan);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 2px 0 7px;
    }
    .workflow-bar {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--muted);
        margin: 5px 0 18px;
        font-size: 0.82rem;
        flex-wrap: wrap;
    }
    .workflow-step { display: flex; align-items: center; gap: 8px; }
    .workflow-number {
        display: grid;
        place-items: center;
        width: 23px;
        height: 23px;
        border: 1px solid #3b5068;
        border-radius: 50%;
        color: #b9c8d8;
        font-size: 0.7rem;
    }
    .workflow-step.done .workflow-number {
        border-color: #2f91aa;
        color: #b5eff9;
        background: #103040;
    }
    .workflow-step.active { color: var(--text); }
    .workflow-step.active .workflow-number {
        border-color: var(--cyan);
        color: var(--cyan);
        box-shadow: 0 0 0 3px rgba(76, 201, 233, 0.1);
    }
    .workflow-connector { width: 34px; height: 1px; background: #34475e; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 8px;
        background: rgba(13, 27, 45, 0.72);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #38536d;
        border-radius: 7px;
        background: #0a1727;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--cyan); }
    [data-testid="stFileUploader"] section button {
        border: 1px solid #39728f;
        border-radius: 5px;
        background: #17364d;
        color: #e2f4fb;
    }
    [data-testid="stFileUploader"] section button:hover {
        border-color: var(--cyan);
        background: #204b66;
        color: #ffffff;
    }
    [data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #c5d3e1;
    }
    [data-testid="stMetric"] {
        min-height: 104px;
        padding: 13px 14px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface-raised);
    }
    [data-testid="stMetricLabel"] { color: #a8b8ca; }
    [data-testid="stMetricValue"] { color: var(--text); font-size: 1.45rem; }
    div.stButton > button[kind="primary"] {
        min-height: 45px;
        border: 1px solid #42b8da;
        border-radius: 6px;
        background: #147493;
        color: white;
        font-weight: 750;
    }
    div.stButton > button[kind="primary"]:hover {
        border-color: #78e1f5;
        background: #1884a6;
    }
    [data-testid="stDownloadButton"] button {
        border: 1px solid #36779a;
        border-radius: 6px;
        background: #102b40;
        color: #c9eff8;
        font-weight: 650;
    }
    [data-testid="stTabs"] [role="tab"] { color: var(--muted); }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--cyan);
        border-bottom-color: var(--cyan);
    }
    [data-testid="stAlert"] { border-radius: 7px; }
    .empty-state {
        padding: 25px;
        border: 1px dashed #344b64;
        border-radius: 8px;
        background: rgba(10, 23, 39, 0.65);
        color: var(--muted);
        text-align: center;
    }
    .empty-title { color: var(--text); font-size: 1.02rem; font-weight: 650; margin-bottom: 5px; }
    .hud-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 4px 0 14px;
    }
    .hud-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 13px 14px;
        border: 1px solid var(--line);
        border-radius: 9px;
        background: linear-gradient(150deg, rgba(17, 36, 57, 0.96), rgba(11, 24, 40, 0.94));
        min-width: 0;
    }
    .hud-ring {
        position: relative;
        flex: 0 0 76px;
        width: 76px;
        height: 76px;
        border-radius: 50%;
        background: conic-gradient(var(--gauge) calc(var(--pct) * 1%), #1b2d42 0);
        display: grid;
        place-items: center;
    }
    .hud-ring:after {
        content: "";
        position: absolute;
        inset: 7px;
        border-radius: 50%;
        background: #0d1a2b;
        border: 1px solid rgba(80, 118, 150, 0.25);
    }
    .hud-ring span {
        position: relative;
        z-index: 1;
        color: var(--text);
        font-size: 0.78rem;
        font-weight: 800;
        text-align: center;
        line-height: 1.1;
    }
    .hud-copy { min-width: 0; }
    .hud-label { color: #b9cfe0; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .hud-sub { color: var(--muted); font-size: 0.7rem; margin-top: 4px; line-height: 1.35; }
    .timeline { border-left: 1px solid #35546b; margin: 5px 0 4px 8px; padding-left: 19px; }
    .timeline-item { position: relative; padding: 0 0 16px; }
    .timeline-item:last-child { padding-bottom: 2px; }
    .timeline-item:before {
        content: "";
        position: absolute;
        left: -24px;
        top: 4px;
        width: 9px;
        height: 9px;
        border: 2px solid var(--cyan);
        border-radius: 50%;
        background: #0a1524;
    }
    .timeline-label { color: #b5ddec; font-size: 0.78rem; font-weight: 750; margin-bottom: 4px; }
    .timeline-copy { color: #a9b9ca; font-size: 0.83rem; line-height: 1.55; white-space: pre-wrap; }
    .step-card {
        border: 1px solid var(--line);
        border-left: 3px solid #3b5068;
        border-radius: 6px;
        padding: 9px 11px;
        margin-bottom: 7px;
        background: rgba(12, 25, 42, 0.7);
    }
    .step-card.accepted { border-left-color: var(--green); }
    .step-card.rejected { border-left-color: var(--red); }
    .step-card.ambiguous { border-left-color: var(--amber); }
    .step-name { color: var(--text); font-size: 0.82rem; font-weight: 700; }
    .step-reason { color: var(--muted); font-size: 0.77rem; margin-top: 3px; line-height: 1.45; }
    .disclaimer { color: #c4a976; font-size: 0.8rem; line-height: 1.5; }
    .mono { font-family: "Cascadia Mono", "Consolas", monospace; }
    @media (max-width: 980px) {
        .hud-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
        [data-testid="stAppViewContainer"] > .main .block-container { padding: 1rem 0.85rem 2rem; }
        .header-right { margin-left: 0; }
        .workflow-bar { gap: 6px; font-size: 0.72rem; }
        .workflow-connector { width: 14px; }
        .hud-card { gap: 8px; padding: 10px; }
        .hud-ring { flex-basis: 66px; width: 66px; height: 66px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header and workflow chrome
# ---------------------------------------------------------------------------


def logo_markup() -> str:
    """Use a user-supplied logo; never draw or regenerate the OpenCV mark."""
    try:
        if LOGO_PATH.is_file():
            encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
            return (
                '<img class="brand-logo" alt="OpenCV logo" '
                f'src="data:image/png;base64,{encoded}" />'
            )
    except OSError:
        pass
    if not st.session_state.get("_logo_fallback_warned"):
        print(
            "[InspectAI] assets/opencv_logo.png was not found; "
            f"using the text badge '{LOGO_FALLBACK_TEXT}'."
        )
        st.session_state["_logo_fallback_warned"] = True
    return f'<div class="brand-textbadge">{html.escape(LOGO_FALLBACK_TEXT)}</div>'


def render_header(analysis_payload: object | None) -> None:
    if analysis_payload is None:
        state_label = "AWAITING INPUTS"
        state_class = "warn"
    else:
        result = analysis_payload["result"]
        state_label = STATUS_LABELS.get(result.alignment_status, result.alignment_status)
        state_class = STATUS_CLASSES.get(result.alignment_status, "warn")
    header = f"""
    <div class="inspect-header">
      <div class="brand-lockup">
        {logo_markup()}
        <div>
          <div class="brand-title">InspectAI</div>
          <div class="brand-tagline">Evidence-driven visual inspection, powered by OpenCV {html.escape(cv2.__version__)}.</div>
          <div class="brand-subtitle">Align. Measure. Investigate. Verify.</div>
        </div>
      </div>
      <div class="header-right">
        {status_pill(state_label, state_class)}
        <span class="header-status">Local · No cloud call</span>
      </div>
    </div>
    """
    st.markdown(header, unsafe_allow_html=True)


def render_workflow(reference_ready: bool, test_ready: bool, analyzed: bool) -> None:
    uploads_ready = reference_ready and test_ready
    steps = [
        ("1", "Upload", uploads_ready, not uploads_ready),
        ("2", "Analyze", analyzed, uploads_ready and not analyzed),
        ("3", "Review evidence", analyzed, analyzed),
    ]
    parts = []
    for number, label, done, active in steps:
        state = "done" if done else "active" if active else ""
        parts.append(
            f'<div class="workflow-step {state}"><span class="workflow-number">{number}</span>'
            f"<span>{label}</span></div>"
        )
    st.markdown(
        '<div class="workflow-bar">'
        + '<div class="workflow-connector"></div>'.join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_hud(result: object) -> None:
    final = result.final_comparison
    edge_similarity = result.independent_comparison.edge_similarity_percentage
    ssim_value = result.ssim_comparison.mean_ssim if result.ssim_comparison else None
    overlap = result.overlap_fraction

    changed_fraction = _fraction(final.changed_percentage, 5.0)
    edge_fraction = _fraction(edge_similarity, 100.0)
    ssim_fraction = _fraction(ssim_value, 1.0)
    overlap_fraction = _fraction(overlap, 1.0)

    cards = [
        (
            "Changed area",
            format_number(final.changed_percentage, 2, "%"),
            changed_fraction,
            _gauge_color(changed_fraction, inverted=True)
            if final.changed_percentage >= 1.0
            else "#4cc9e9",
            f"Initial {format_number(result.initial_comparison.changed_percentage, 2, '%')} · "
            f"{result.changed_regions['region_count']} reported region(s)",
        ),
        (
            "Edge agreement",
            format_number(edge_similarity, 1, "%"),
            edge_fraction,
            _gauge_color(edge_fraction),
            f"Canny F1 at {result.independent_comparison.tolerance_pixels} px tolerance",
        ),
        (
            "Mean SSIM",
            format_number(ssim_value, 4),
            ssim_fraction,
            _gauge_color(ssim_fraction),
            f"Threshold {format_number(result.ssim_comparison.threshold if result.ssim_comparison else None, 2)}",
        ),
        (
            "Comparison overlap",
            format_number(overlap * 100.0 if overlap is not None else None, 1, "%"),
            overlap_fraction,
            _gauge_color(overlap_fraction),
            f"{result.alignment_method} · {result.alignment_status.replace('_', ' ')}",
        ),
    ]
    html_cards = []
    for label, value, fraction, color, sub in cards:
        html_cards.append(
            f'<div class="hud-card">'
            f'<div class="hud-ring" style="--pct:{fraction * 100.0:.1f};--gauge:{color};">'
            f"<span>{html.escape(value)}</span></div>"
            f'<div class="hud-copy"><div class="hud-label">{html.escape(label)}</div>'
            f'<div class="hud-sub">{html.escape(sub)}</div></div>'
            "</div>"
        )
    st.markdown('<div class="hud-grid">' + "".join(html_cards) + "</div>", unsafe_allow_html=True)
    st.caption(
        "Gauges are heuristic comparison signals, not calibrated probabilities. "
        "Full rings mean more of the dial range, not a higher chance of a defect."
    )


# ---------------------------------------------------------------------------
# Image and evidence widgets
# ---------------------------------------------------------------------------


def annotated_regions_image(image: np.ndarray, regions: list[dict[str, int]]) -> np.ndarray:
    annotated = image.copy()
    for region_index, region in enumerate(regions, start=1):
        x = int(region["x"])
        y = int(region["y"])
        right = x + int(region["width"]) - 1
        bottom = y + int(region["height"]) - 1
        cv2.rectangle(annotated, (x, y), (right, bottom), (64, 200, 232), 2)
        cv2.putText(
            annotated,
            f"R{region_index}",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (64, 200, 232),
            1,
            cv2.LINE_AA,
        )
    return annotated


def render_region_table(result: object, comparison_test: np.ndarray) -> None:
    region_summary = result.changed_regions
    final = result.final_comparison
    columns = st.columns(4)
    columns[0].metric("Reported regions", int(region_summary["region_count"]))
    columns[1].metric("Filtered components", int(final.filtered_region_count))
    columns[2].metric("Minimum area", f"{int(region_summary['minimum_area_pixels'])} px")
    columns[3].metric(
        "Minimum mean ΔE",
        format_number(final.minimum_component_contrast, 2),
    )
    st.caption(
        "Boxes localize connected changed pixels in the final difference mask. "
        "They are image-difference regions, not confirmed defects. "
        f"Filter rule: {final.region_filter}."
    )
    if region_summary["regions"]:
        st.image(
            bgr_to_rgb(annotated_regions_image(comparison_test, region_summary["regions"])),
            width="stretch",
        )
        st.dataframe(region_summary["regions"], hide_index=True, width="stretch")
    elif region_summary["detected_region_count"]:
        st.info(
            f"No region met the {region_summary['minimum_area_pixels']}-pixel minimum. "
            f"{region_summary['filtered_region_count']} smaller connected component(s) were filtered."
        )
    else:
        st.info("No changed region survived the adaptive threshold and component filter.")


def render_feature_matches(step: object, reference_image: np.ndarray) -> None:
    reference_points = step.keypoints_ref
    test_points = step.keypoints_test
    if not reference_points or not test_points:
        st.info("This step did not produce feature correspondences.")
        return
    canvas = reference_image.copy()
    inliers = 0
    for reference_point, test_point in zip(reference_points, test_points):
        x_reference, y_reference, reference_inlier = reference_point
        x_test, y_test, test_inlier = test_point
        is_inlier = bool(reference_inlier and test_inlier)
        inliers += int(is_inlier)
        color = (104, 214, 160) if is_inlier else (240, 124, 134)
        start = (int(round(x_reference)), int(round(y_reference)))
        end = (int(round(x_test)), int(round(y_test)))
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 2, color, -1, cv2.LINE_AA)
    st.image(bgr_to_rgb(canvas), width="stretch")
    st.caption(
        f"{inliers} inlier and {len(reference_points) - inliers} outlier correspondence(s). "
        "Both point sets live in the common reference canvas, so a good match is a short line. "
        "Green = inlier, red = outlier."
    )


def render_step_cards(result: object) -> None:
    cards = []
    for step in result.alignment_steps:
        state = html.escape(step.status)
        cards.append(
            f'<div class="step-card {state}">'
            f'<div class="step-name">{step.index + 1:02d} · {html.escape(step.name)} '
            f'<span class="step-reason">[{state}]</span></div>'
            f'<div class="step-reason">{html.escape(step.reason)}</div>'
            "</div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_overlay(
    reference_image: np.ndarray,
    comparison_test: np.ndarray,
    matrix: np.ndarray,
    alpha: float,
    show_difference: bool,
    difference_mask: np.ndarray | None,
) -> None:
    height, width = reference_image.shape[:2]
    warped = warp_image(comparison_test, matrix, width, height)
    blend = cv2.addWeighted(reference_image, 1.0 - alpha, warped, alpha, 0.0)
    if show_difference and difference_mask is not None:
        contours, _ = cv2.findContours(
            difference_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blend, contours, -1, (80, 220, 240), 1, cv2.LINE_AA)
    left, middle, right = st.columns(3)
    with left:
        st.caption("Reference canvas")
        st.image(bgr_to_rgb(reference_image), width="stretch")
    with middle:
        st.caption("Scrubbed test warp")
        st.image(bgr_to_rgb(warped), width="stretch")
    with right:
        st.caption(f"Blend at α={alpha:.2f}")
        st.image(bgr_to_rgb(blend), width="stretch")


@st.fragment
def render_alignment_inspector(
    result: object,
    comparison_reference: np.ndarray,
    comparison_test: np.ndarray,
) -> None:
    st.markdown('<div class="section-kicker">Alignment inspector</div>', unsafe_allow_html=True)
    st.caption(
        "Every candidate the agent considered is listed. Matrices use the "
        "canvas-to-original-test convention and are applied with inverse-map sampling."
    )
    steps = result.alignment_steps
    if not steps:
        st.info("No alignment steps were recorded.")
        return
    if st.session_state.get("alignment_step_select") not in list(range(len(steps))):
        st.session_state["alignment_step_select"] = len(steps) - 1
    selected = st.selectbox(
        "Alignment step",
        options=list(range(len(steps))),
        format_func=lambda index: step_label(steps[index]),
        key="alignment_step_select",
    )
    step = steps[selected]
    state = STATUS_CLASSES.get(step.status, "warn")
    detail_left, detail_right = st.columns([2, 1])
    with detail_left:
        st.markdown(
            f"{status_pill(step.name, state)} {status_pill(f'{step.status} · {step.model}', state)}",
            unsafe_allow_html=True,
        )
        st.write(step.reason)
    with detail_right:
        st.metric("Step index", f"{step.index + 1} / {len(steps)}")
        st.metric("Model", step.model)

    transform_fraction = st.slider(
        "Scrub from identity to this step",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        key=f"alignment_scrub_{step.index}",
        help="Interpolates the cumulative transform for this step.",
    )
    alpha = st.slider(
        "Overlay opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key=f"alignment_alpha_{step.index}",
    )
    show_difference = st.checkbox(
        "Outline final change regions on the blend",
        value=True,
        key=f"alignment_outline_{step.index}",
    )
    height, width = comparison_reference.shape[:2]
    interpolated = interpolate_transform(step.matrix, transform_fraction, width, height)
    render_overlay(
        comparison_reference,
        comparison_test,
        interpolated,
        alpha,
        show_difference,
        result.final_comparison.difference_mask,
    )

    metrics_tab, matrix_tab, matches_tab = st.tabs(
        ["Step metrics", "Transform matrix", "Feature matches"]
    )
    with metrics_tab:
        if step.metrics:
            metrics_rows = [
                {"metric": key, "value": format_metric_value(value)}
                for key, value in step.metrics.items()
            ]
            st.dataframe(metrics_rows, hide_index=True, width="stretch")
        else:
            st.info("This step recorded no metrics.")
        if step.preview is not None:
            st.caption("Step preview (the aligned result at this point in the ladder)")
            st.image(bgr_to_rgb(step.preview), width="stretch")
    with matrix_tab:
        st.code(
            "\n".join(
                "[" + ", ".join(f"{value: .6f}" for value in row) + "]"
                for row in np.asarray(step.matrix, dtype=np.float64)
            ),
            language="text",
        )
        st.caption(
            "The cumulative matrix at this step maps reference-canvas coordinates to "
            "the original test image. Rejected rows are candidate transforms, not applied results."
        )
    with matches_tab:
        render_feature_matches(step, comparison_reference)

    st.markdown("### Candidate review")
    rejected_rows = [
        {
            "step": candidate.index + 1,
            "name": candidate.name,
            "model": candidate.model,
            "status": candidate.status,
            "reason": candidate.reason,
        }
        for candidate in steps
        if candidate.status in {"rejected", "ambiguous", "skipped"}
    ]
    if rejected_rows:
        st.dataframe(rejected_rows, hide_index=True, width="stretch")
    else:
        st.info("No candidate was rejected or left ambiguous.")
    with st.expander("Full step sequence", expanded=False):
        render_step_cards(result)


# ---------------------------------------------------------------------------
# Results tabs
# ---------------------------------------------------------------------------


def render_overview(payload: dict[str, object]) -> None:
    result = payload["result"]
    reference_image = payload["reference_image"]
    test_image = payload["test_image"]
    comparison_test = payload["comparison_test"]
    final = result.final_comparison

    if result.method_disagreement:
        st.warning(
            "The Lab difference, Canny edge and SSIM methods do not all agree on whether "
            "a change is indicated. Inspect each method map before drawing a conclusion."
        )
    elif result.method_disagreement is None:
        st.info("Edge evidence is insufficient for a method-agreement check.")
    if result.alignment_status == "ambiguous":
        st.warning(
            "Alignment was ambiguous. No transform was silently chosen; review the candidate "
            "steps, reasons and rejected alternatives in the Alignment tab."
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Reference (baseline)**")
        st.caption(
            f"{reference_image.shape[1]} × {reference_image.shape[0]} px · "
            f"aspect {format_aspect_ratio(reference_image.shape[1], reference_image.shape[0])}"
        )
        st.image(bgr_to_rgb(reference_image), width="stretch")
    with right:
        st.markdown("**Final test image used for comparison**")
        st.caption(
            f"Original {test_image.shape[1]} × {test_image.shape[0]} px · "
            f"comparison {comparison_test.shape[1]} × {comparison_test.shape[0]} px · "
            f"method {result.alignment_method}"
        )
        st.image(bgr_to_rgb(comparison_test), width="stretch")

    st.markdown('<div class="section-kicker">Change regions</div>', unsafe_allow_html=True)
    render_region_table(result, comparison_test)

    image_tab, heatmap_tab = st.tabs(["Final aligned test", "Difference heatmap"])
    with image_tab:
        st.image(bgr_to_rgb(comparison_test), width="stretch")
    with heatmap_tab:
        st.caption(
            "The heatmap scales the adaptive Lab difference. It is not a defect severity scale."
        )
        st.image(bgr_to_rgb(final.heatmap), width="stretch")

    with st.expander("Known limitations", expanded=True):
        st.markdown(
            "\n".join(f"- {html.escape(item)}" for item in result.limitations)
        )
        st.markdown(
            '<div class="disclaimer">Changed regions are connected groups of pixels in the '
            "final difference mask, not confirmed defects. Lighting, camera position, focus, "
            "compression and sensor noise can produce the same signals.</div>",
            unsafe_allow_html=True,
        )


def render_methods(payload: dict[str, object]) -> None:
    result = payload["result"]
    final = result.final_comparison
    edge = result.independent_comparison
    ssim = result.ssim_comparison

    lab_tab, edge_tab, ssim_tab, photometric_tab = st.tabs(
        ["Lab difference", "Canny edges", "SSIM", "Photometric fit"]
    )
    with lab_tab:
        columns = st.columns(4)
        columns[0].metric("Changed pixels", format_number(final.changed_percentage, 2, "%"))
        columns[1].metric("Adaptive ΔE threshold", format_number(final.threshold, 2))
        columns[2].metric("Noise MAD", format_number(final.noise_mad, 2))
        columns[3].metric("Regions", int(result.changed_regions["region_count"]))
        st.write(
            "A CIE-Lab Euclidean difference is computed per pixel. The threshold is "
            f"{format_number(final.threshold, 2)} and the component filter is: {final.region_filter}."
        )
        st.image(bgr_to_rgb(final.heatmap), width="stretch")
        st.caption(
            "Colour-only changes remain visible because the metric operates in Lab space. "
            "The heatmap is a difference visualization, not a defect probability."
        )
    with edge_tab:
        columns = st.columns(4)
        columns[0].metric("Edge similarity", format_number(edge.edge_similarity_percentage, 2, "%"))
        columns[1].metric(
            "Edge disagreement", format_number(edge.edge_disagreement_percentage, 2, "%")
        )
        columns[2].metric("Reference edges", f"{edge.reference_edge_pixels:,}")
        columns[3].metric("Test edges", f"{edge.test_edge_pixels:,}")
        st.write(
            f"Status: **{edge.status}**. Canny thresholds were "
            f"{edge.canny_low} / {edge.canny_high} with a {edge.tolerance_pixels} px matching tolerance. "
            "Similarity and disagreement describe the same overlap from opposite directions."
        )
        st.markdown(
            '<div class="disclaimer">Low edge similarity alone does not establish a defect. '
            "Sparse textures and lighting boundaries can dominate this measurement.</div>",
            unsafe_allow_html=True,
        )
    with ssim_tab:
        if ssim is None or ssim.status != "measured" or ssim.ssim_map is None:
            st.info("SSIM was not scored on the valid overlap.")
        else:
            columns = st.columns(3)
            columns[0].metric("Mean SSIM", format_number(ssim.mean_ssim, 4))
            columns[1].metric("Below threshold", format_number(ssim.changed_percentage, 2, "%"))
            columns[2].metric("Threshold", format_number(ssim.threshold, 2))
            ssim_visual = np.clip(ssim.ssim_map, 0.0, 1.0)
            ssim_visual = (ssim_visual * 255.0).astype(np.uint8)
            ssim_colour = cv2.applyColorMap(255 - ssim_visual, cv2.COLORMAP_TURBO)
            st.image(bgr_to_rgb(ssim_colour), width="stretch")
            st.caption(
                "Brighter/warmer pixels indicate lower local SSIM. "
                "SSIM is a structural similarity signal, not a calibrated defect probability."
            )
    with photometric_tab:
        photometric = result.photometric
        if photometric is None:
            st.info("No photometric fit was recorded.")
        else:
            columns = st.columns(4)
            columns[0].metric("Status", photometric.status)
            columns[1].metric("Applied", "Yes" if photometric.applied else "No")
            columns[2].metric(
                "Residual before", format_number(photometric.residual_before, 2)
            )
            columns[3].metric(
                "Residual after", format_number(photometric.residual_after, 2)
            )
            st.write(photometric.reason)
            st.caption(
                f"Model: {photometric.model}. Channel gains: "
                f"{format_metric_value(photometric.channel_gain)}. "
                "A smooth photometric fit can absorb a genuine low-contrast change."
            )

    st.markdown("### Method indications")
    method_rows = [
        {
            "method": "Lab difference",
            "indicated": result.pixel_change_indicated,
            "summary": f"{final.changed_percentage:.2f}% changed pixels",
        },
        {
            "method": "Canny edges",
            "indicated": result.edge_change_indicated,
            "summary": f"{format_number(edge.edge_similarity_percentage, 2, '%')} similarity",
        },
        {
            "method": "SSIM",
            "indicated": ssim.change_indicated if ssim else None,
            "summary": f"{format_number(ssim.mean_ssim if ssim else None, 4)} mean",
        },
    ]
    st.dataframe(method_rows, hide_index=True, width="stretch")
    st.info(
        f"Agreement: {AGREEMENT_LABELS.get(result.indication_agreement, result.indication_agreement)}. "
        "The methods measure different visual properties and are deliberately not averaged into "
        "a single defect score."
    )


def render_trace(result: object) -> None:
    items = []
    for iteration in result.agent_trace:
        evidence = "\n".join(f"- {item}" for item in iteration.evidence)
        items.append(
            '<div class="timeline-item">'
            f'<div class="timeline-label">{iteration.index + 1:02d} · '
            f"{html.escape(iteration.action)} → {html.escape(iteration.outcome)}</div>"
            f'<div class="timeline-copy">{html.escape(evidence)}\n'
            f"Uncertainty: {html.escape(iteration.uncertainty)}</div>"
            "</div>"
        )
    st.markdown('<div class="timeline">' + "".join(items) + "</div>", unsafe_allow_html=True)


def render_report(payload: dict[str, object]) -> None:
    result = payload["result"]
    report = build_inspection_report(
        payload["comparison_reference"],
        payload["comparison_test"],
        result,
    )
    download_column, note_column = st.columns([1, 3])
    with download_column:
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2, allow_nan=False),
            file_name="inspectai-report.json",
            mime="application/json",
            type="primary",
            width="stretch",
        )
    with note_column:
        note = (
            f"Schema version {report['schema_version']} · measurements and evidence metadata only. "
            "Uploaded image contents are excluded."
        )
        if payload["center_cropped"]:
            note += " Report dimensions refer to the center-cropped comparison images."
        st.caption(note)

    trace_tab, evidence_tab, raw_tab = st.tabs(
        ["Agent trace", "Evidence & warnings", "Raw JSON"]
    )
    with trace_tab:
        render_trace(result)
        st.caption(
            "Every automatic action records the action, supporting evidence, outcome and uncertainty. "
            "Stopping the loop is not a defect decision."
        )
        with st.expander("Iteration metrics"):
            st.dataframe(
                [
                    {
                        "iteration": iteration.index + 1,
                        "action": iteration.action,
                        "outcome": iteration.outcome,
                        "metrics": json.dumps(iteration.metrics, default=str),
                    }
                    for iteration in result.agent_trace
                ],
                hide_index=True,
                width="stretch",
            )
    with evidence_tab:
        st.markdown("**Selected actions and reasons**")
        st.markdown("\n".join(f"- {item}" for item in result.decisions))
        st.markdown("**Evidence**")
        st.markdown("\n".join(f"- {item}" for item in result.evidence))
        if result.uncertainty_warnings:
            st.markdown("**Uncertainty warnings**")
            for warning in result.uncertainty_warnings:
                st.warning(warning)
        if result.limitations:
            st.markdown("**Limitations**")
            st.markdown("\n".join(f"- {item}" for item in result.limitations))
        with st.expander("Function trace"):
            st.dataframe(
                [
                    {
                        "stage": item["stage"],
                        "functions": ", ".join(item["functions"]),
                    }
                    for item in result.function_trace
                ],
                hide_index=True,
                width="stretch",
            )
    with raw_tab:
        st.caption(
            "The report is JSON-safe (no NaN or Infinity) and contains no image pixels."
        )
        st.json(report)


def render_results(payload: dict[str, object]) -> None:
    result = payload["result"]
    st.markdown('<div class="section-kicker">Inspection HUD</div>', unsafe_allow_html=True)
    render_hud(result)
    overview_tab, alignment_tab, methods_tab, report_tab = st.tabs(
        ["Overview", "Alignment", "Comparison methods", "Evidence & report"]
    )
    with overview_tab:
        render_overview(payload)
    with alignment_tab:
        render_alignment_inspector(
            result,
            payload["comparison_reference"],
            payload["comparison_test"],
        )
    with methods_tab:
        render_methods(payload)
    with report_tab:
        render_report(payload)


# ---------------------------------------------------------------------------
# Upload, geometry and analysis workflow
# ---------------------------------------------------------------------------

analysis_payload = st.session_state.get("analysis_payload")
render_header(analysis_payload)
st.markdown('<div class="section-kicker">Inspection workflow</div>', unsafe_allow_html=True)

reference_upload = None
test_upload = None
reference_bytes = None
test_bytes = None
reference_column, test_column = st.columns(2)
with reference_column:
    with st.container(border=True):
        st.markdown("### Reference image")
        st.caption("Baseline or expected image")
        reference_upload = st.file_uploader(
            "Reference file",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            key="reference",
            help="Choose the expected or baseline image.",
        )
        if reference_upload is not None:
            reference_bytes = reference_upload.getvalue()
            st.caption(f"Loaded: {reference_upload.name} · {len(reference_bytes) / 1024:.1f} KB")
with test_column:
    with st.container(border=True):
        st.markdown("### Test image")
        st.caption("Image captured for inspection")
        test_upload = st.file_uploader(
            "Test file",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            key="test",
            help="Choose the image to inspect against the baseline.",
        )
        if test_upload is not None:
            test_bytes = test_upload.getvalue()
            st.caption(f"Loaded: {test_upload.name} · {len(test_bytes) / 1024:.1f} KB")

uploads_ready = reference_bytes is not None and test_bytes is not None
upload_signature = None
if uploads_ready:
    signature = hashlib.sha256()
    for image_bytes in (reference_bytes, test_bytes):
        signature.update(len(image_bytes).to_bytes(8, "big"))
        signature.update(image_bytes)
    upload_signature = signature.hexdigest()

if st.session_state.get("upload_signature") != upload_signature:
    st.session_state["upload_signature"] = upload_signature
    st.session_state["center_crop_enabled"] = False
    st.session_state["crop_aspect_source"] = "Reference image"
    st.session_state["analysis_payload"] = None
    st.session_state["analysis_signature"] = None

reference_image = None
test_image = None
decode_error = None
if uploads_ready:
    try:
        reference_image = cached_decode(reference_bytes)
        test_image = cached_decode(test_bytes)
    except ValueError as error:
        decode_error = str(error)
        st.error(f"Image could not be opened: {decode_error}")

aspect_mismatch = False
crop_enabled = False
crop_error = None
cropped_reference = None
cropped_test = None
target_aspect_source = "Reference image"
if reference_image is not None and test_image is not None:
    reference_height, reference_width = reference_image.shape[:2]
    test_height, test_width = test_image.shape[:2]
    aspect_mismatch = reference_width * test_height != test_width * reference_height

    st.markdown("### Image geometry")
    geometry_reference, geometry_test = st.columns(2)
    with geometry_reference:
        st.metric(
            "Reference dimensions",
            f"{reference_width} × {reference_height} px",
            delta=f"Aspect ratio {format_aspect_ratio(reference_width, reference_height)}",
            delta_color="off",
        )
    with geometry_test:
        st.metric(
            "Test dimensions",
            f"{test_width} × {test_height} px",
            delta=f"Aspect ratio {format_aspect_ratio(test_width, test_height)}",
            delta_color="off",
        )

    if aspect_mismatch:
        st.warning(
            "These images have different shapes. Comparing them directly would stretch one "
            "image. Use center-cropped preview copies for this analysis, or upload images "
            "with matching aspect ratios."
        )
        crop_enabled = st.checkbox(
            "Preview and use center-cropped copies for this analysis",
            key="center_crop_enabled",
        )
        if crop_enabled:
            target_aspect_source = st.selectbox(
                "Match the aspect ratio of",
                ["Reference image", "Test image"],
                key="crop_aspect_source",
                help="The selected image's aspect ratio is used for both temporary center-cropped copies.",
            )
            target_image = (
                reference_image if target_aspect_source == "Reference image" else test_image
            )
            target_height, target_width = target_image.shape[:2]
            try:
                cropped_reference, cropped_test = center_crop_pair(
                    reference_image,
                    test_image,
                    target_width=target_width,
                    target_height=target_height,
                )
            except ValueError as error:
                crop_error = str(error)
                st.error(f"These images cannot be cropped to that common ratio: {crop_error}")
            else:
                st.info(
                    f"Preview only: analysis will use {cropped_reference.shape[1]} × "
                    f"{cropped_reference.shape[0]} pixel copies. The original uploads remain unchanged."
                )
                preview_reference, preview_test = st.columns(2)
                with preview_reference:
                    st.markdown("**Reference crop preview**")
                    st.image(bgr_to_rgb(cropped_reference), width="stretch")
                with preview_test:
                    st.markdown("**Test crop preview**")
                    st.image(bgr_to_rgb(cropped_test), width="stretch")

if uploads_ready and reference_image is not None and test_image is not None:
    analysis_signature = f"{upload_signature}:{crop_enabled}:{target_aspect_source}"
else:
    analysis_signature = upload_signature
if st.session_state.get("analysis_signature") != analysis_signature:
    st.session_state["analysis_signature"] = analysis_signature
    st.session_state["analysis_payload"] = None

images_decoded = reference_image is not None and test_image is not None
crop_ready = not crop_enabled or (cropped_reference is not None and cropped_test is not None)
analysis_blocked = aspect_mismatch and not crop_enabled
render_workflow(
    reference_upload is not None,
    test_upload is not None,
    st.session_state.get("analysis_payload") is not None,
)
analyze_clicked = st.button(
    "Run inspection",
    type="primary",
    width="stretch",
    disabled=not images_decoded or not crop_ready or analysis_blocked,
    help="Run alignment, Lab difference, Canny edge, SSIM and photometric evidence checks.",
)

if analyze_clicked and images_decoded and crop_ready and not analysis_blocked:
    try:
        with st.spinner("Running the alignment ladder and comparison methods…"):
            (
                reference_image,
                test_image,
                comparison_reference,
                comparison_test,
                result,
            ) = cached_inspection(
                reference_bytes,
                test_bytes,
                crop_enabled,
                target_aspect_source,
            )
    except (ValueError, cv2.error) as error:
        if "aspect ratios differ" in str(error).lower():
            st.error(
                "Analysis was not run because the image shapes do not match. Enable the "
                "center-crop option or upload images with matching aspect ratios."
            )
        else:
            st.error(f"Analysis could not be completed: {error}")
        st.session_state["analysis_payload"] = None
    else:
        st.session_state["analysis_payload"] = {
            "reference_image": reference_image,
            "test_image": test_image,
            "comparison_reference": comparison_reference,
            "comparison_test": comparison_test,
            "center_cropped": crop_enabled,
            "result": result,
        }

if not uploads_ready:
    st.markdown(
        '<div class="empty-state"><div class="empty-title">Ready when you are</div>'
        "Add a reference image and a test image to enable inspection.</div>",
        unsafe_allow_html=True,
    )
elif analysis_blocked:
    st.markdown(
        '<div class="empty-state"><div class="empty-title">Shape review required</div>'
        "Enable center-cropped copies above to compare images with different aspect ratios.</div>",
        unsafe_allow_html=True,
    )
elif st.session_state.get("analysis_payload") is None and not analyze_clicked:
    st.markdown(
        '<div class="empty-state"><div class="empty-title">Inputs ready for review</div>'
        "Run the inspection to build the alignment sequence and evidence report.</div>",
        unsafe_allow_html=True,
    )

analysis_payload = st.session_state.get("analysis_payload")
if analysis_payload is not None:
    render_results(analysis_payload)

st.caption(
    "InspectAI reports visual differences, alignment evidence and uncertainty. "
    "It does not state that a defect was detected or confirmed. "
    "Pixel, edge and SSIM measurements are heuristic comparison signals, not calibrated probabilities."
)
