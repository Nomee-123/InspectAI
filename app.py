"""Streamlit dashboard for the local InspectAI inspection workflow."""

import hashlib
import html
import json
from fractions import Fraction

import cv2
import streamlit as st

from inspectai import build_inspection_report, center_crop_pair, decode_image, inspect_images


st.set_page_config(page_title="InspectAI | Vision Inspection", layout="wide")
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --surface: #0b1423;
        --surface-raised: #101d30;
        --surface-soft: #0e1929;
        --line: #24364c;
        --text: #e6eef7;
        --muted: #96a8bc;
        --cyan: #48c8e8;
        --blue: #5c96f2;
        --amber: #e8b45b;
    }
    .stApp {
        background: linear-gradient(155deg, #091321 0%, #0b1423 48%, #0c1728 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(9, 19, 33, 0.72); }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1440px;
        padding: 0.7rem 2.3rem 2.2rem;
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
        gap: 12px;
        padding: 3px 0 12px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 12px;
    }
    .brand-mark {
        position: relative;
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        flex: 0 0 38px;
        border: 1px solid #2d6382;
        border-radius: 9px;
        color: var(--cyan);
        background: #102638;
    }
    .brand-mark:before {
        content: "";
        width: 13px;
        height: 13px;
        border: 2px solid var(--cyan);
        border-radius: 50%;
    }
    .brand-mark:after {
        content: "";
        position: absolute;
        left: 22px;
        top: 22px;
        width: 7px;
        height: 2px;
        border-radius: 2px;
        background: var(--cyan);
        transform: rotate(45deg);
    }
    .brand-title { color: var(--text); font-size: 1.12rem; font-weight: 700; line-height: 1.2; }
    .brand-tagline { color: #c3d1df; font-size: 0.78rem; margin-top: 3px; }
    .brand-subtitle { color: var(--muted); font-size: 0.72rem; margin-top: 2px; }
    .header-status {
        margin-left: auto;
        color: #a5d9e9;
        border: 1px solid #28516a;
        border-radius: 5px;
        padding: 4px 8px;
        font-size: 0.7rem;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .section-kicker {
        color: var(--cyan);
        font-size: 0.69rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .workflow-bar {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--muted);
        margin: 5px 0 19px;
        font-size: 0.83rem;
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
        box-shadow: 0 0 0 3px rgba(72, 200, 232, 0.09);
    }
    .workflow-connector { width: 34px; height: 1px; background: #34475e; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 8px;
        background: rgba(15, 28, 45, 0.72);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #38536d;
        border-radius: 7px;
        background: #0c1828;
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
    [data-testid="stFileUploader"] section button:focus-visible {
        outline: 2px solid var(--cyan);
        outline-offset: 2px;
    }
    [data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #c5d3e1;
    }
    [data-testid="stMetric"] {
        min-height: 110px;
        padding: 14px 15px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface-raised);
    }
    [data-testid="stMetricLabel"] { color: #a8b8ca; }
    [data-testid="stMetricValue"] { color: var(--text); font-size: 1.5rem; }
    div.stButton > button[kind="primary"] {
        min-height: 45px;
        border: 1px solid #42b8da;
        border-radius: 6px;
        background: #147493;
        color: white;
        font-weight: 700;
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
        background: rgba(12, 24, 40, 0.62);
        color: var(--muted);
        text-align: center;
    }
    .empty-title { color: var(--text); font-size: 1.02rem; font-weight: 650; margin-bottom: 5px; }
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
        background: #0b1423;
    }
    .timeline-label { color: #b5ddec; font-size: 0.78rem; font-weight: 700; margin-bottom: 4px; }
    .timeline-copy { color: #a9b9ca; font-size: 0.84rem; line-height: 1.55; white-space: pre-wrap; }
    .disclaimer { color: #c4a976; font-size: 0.8rem; }
    @media (max-width: 760px) {
        [data-testid="stAppViewContainer"] > .main .block-container { padding: 1rem 0.85rem 2rem; }
        .header-status { display: none; }
        .workflow-bar { gap: 6px; font-size: 0.72rem; }
        .workflow-connector { width: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_workflow(reference_ready: bool, test_ready: bool, analyzed: bool) -> None:
    uploads_ready = reference_ready and test_ready
    steps = [
        ("1", "Upload", uploads_ready, not uploads_ready),
        ("2", "Analyze", analyzed, uploads_ready and not analyzed),
        ("3", "Review", analyzed, analyzed),
    ]
    parts = []
    for number, label, done, active in steps:
        state = "done" if done else "active" if active else ""
        parts.append(
            f'<div class="workflow-step {state}"><span class="workflow-number">{number}</span>'
            f"<span>{label}</span></div>"
        )
    workflow_html = '<div class="workflow-bar">' + '<div class="workflow-connector"></div>'.join(parts) + "</div>"
    st.markdown(workflow_html, unsafe_allow_html=True)


def format_aspect_ratio(width: int, height: int) -> str:
    ratio = Fraction(width, height)
    return f"{ratio.numerator}:{ratio.denominator} ({width / height:.3f})"


def render_activity_timeline(result: object) -> None:
    edge_result = result.independent_comparison
    edge_summary = (
        f"{edge_result.edge_similarity_percentage:.2f}% edge similarity; "
        f"{edge_result.reference_edge_pixels} reference and {edge_result.test_edge_pixels} test edge pixels."
        if edge_result.edge_similarity_percentage is not None
        else "Edge comparison was not scored because there were too few edge pixels."
    )
    action_summary = "\n".join(f"- {decision}" for decision in result.decisions)
    observation_summary = "\n".join(f"- {item}" for item in result.evidence)
    outcome_summary = (
        f"Alignment status: {result.alignment_status}. "
        f"Changed pixels: {result.initial_comparison.changed_percentage:.2f}% before, "
        f"{result.final_comparison.changed_percentage:.2f}% after the selected action. "
        f"Independent check: {edge_summary}"
    )
    events = [
        ("Recorded observations", observation_summary),
        ("Selected actions and reasons", action_summary),
        ("Analysis outcome", outcome_summary),
    ]
    timeline_html = ['<div class="timeline">']
    for label, content in events:
        timeline_html.append(
            '<div class="timeline-item">'
            f'<div class="timeline-label">{html.escape(label)}</div>'
            f'<div class="timeline-copy">{html.escape(content)}</div>'
            "</div>"
        )
    timeline_html.append("</div>")
    st.markdown("".join(timeline_html), unsafe_allow_html=True)


def render_results(payload: dict[str, object]) -> None:
    reference_image = payload["reference_image"]
    test_image = payload["test_image"]
    comparison_reference = payload["comparison_reference"]
    comparison_test = payload["comparison_test"]
    result = payload["result"]

    st.markdown('<div class="section-kicker">Inspection results</div>', unsafe_allow_html=True)
    if result.method_disagreement:
        st.warning(result.uncertainty_warnings[-1])
    elif result.method_disagreement is None:
        st.info("Independent edge evidence is insufficient for a method-agreement check.")
    for warning in result.uncertainty_warnings:
        if warning not in {result.uncertainty_warnings[-1]} or not result.method_disagreement:
            st.warning(warning)

    changed_column, edge_column, alignment_column, agreement_column = st.columns(4)
    changed_column.metric(
        "Changed pixels",
        f"{result.final_comparison.changed_percentage:.2f}%",
        delta=f"Initial {result.initial_comparison.changed_percentage:.2f}%",
        delta_color="off",
    )
    edge_similarity = result.independent_comparison.edge_similarity_percentage
    edge_column.metric(
        "Edge similarity",
        f"{edge_similarity:.2f}%" if edge_similarity is not None else "Not scored",
    )
    alignment_column.metric(
        "Alignment",
        result.alignment_status.replace("_", " ").title(),
        delta=f"Shift {result.estimated_shift[0]:.2f}, {result.estimated_shift[1]:.2f} px",
        delta_color="off",
    )
    agreement_labels = {
        "both_indicate_change": "Both indicate change",
        "pixel_only_indicates_change": "Pixel only",
        "edge_only_indicates_change": "Edge only",
        "neither_indicates_change": "Neither indicates change",
        "insufficient_edge_evidence": "Not scored",
    }
    agreement_column.metric(
        "Change-indication agreement",
        agreement_labels[result.indication_agreement],
    )
    st.caption(
        f"Phase-correlation response: {result.alignment_response:.3f} (matching signal, not confidence). "
        "The agreement card compares thresholded change/no-change indications, not raw percentages. "
        "Thresholds: pixel change >= 1.00%; edge disagreement >= 10.00%."
    )

    original_tab, mask_tab, heatmap_tab, edge_tab = st.tabs(
        ["Original Images", "Difference Mask", "Heatmap", "Edge Comparison"]
    )
    with original_tab:
        reference_column, test_column = st.columns(2)
        with reference_column:
            st.markdown("**Reference**")
            st.caption(f"{reference_image.shape[1]} x {reference_image.shape[0]} px")
            st.image(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB), width="stretch")
        with test_column:
            st.markdown("**Test**")
            st.caption(f"{test_image.shape[1]} x {test_image.shape[0]} px")
            st.image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), width="stretch")
    with mask_tab:
        st.caption("White pixels exceed the fixed grayscale intensity-difference threshold.")
        st.image(result.final_comparison.difference_mask, clamp=True, width="stretch")
    with heatmap_tab:
        st.caption("Color intensity visualizes grayscale pixel differences; it is not a defect severity scale.")
        heatmap_rgb = cv2.cvtColor(result.final_comparison.heatmap, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, width="stretch")
    with edge_tab:
        edge_left, edge_right = st.columns(2)
        edge_left.metric(
            "Canny edge similarity",
            f"{edge_similarity:.2f}%" if edge_similarity is not None else "Insufficient edge evidence",
        )
        edge_right.metric(
            "Canny edge disagreement",
            f"{result.independent_comparison.edge_disagreement_percentage:.2f}%"
            if result.independent_comparison.edge_disagreement_percentage is not None
            else "Not scored",
        )
        st.write(
            f"Reference edge pixels: {result.independent_comparison.reference_edge_pixels}  "
            f"| Test edge pixels: {result.independent_comparison.test_edge_pixels}  "
            f"| Matching tolerance: {result.independent_comparison.tolerance_pixels} px"
        )
        st.markdown(
            '<div class="disclaimer">Similarity and disagreement describe the same Canny edge overlap from opposite directions. They are not directly comparable to the pixel changed percentage; low similarity alone does not establish a defect.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Agent activity")
    with st.container(border=True):
        render_activity_timeline(result)

    st.markdown("### Evidence report")
    report = build_inspection_report(comparison_reference, comparison_test, result)
    report_column, note_column = st.columns([1, 3])
    with report_column:
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2, allow_nan=False),
            file_name="inspectai-report.json",
            mime="application/json",
            type="primary",
            width="stretch",
        )
    with note_column:
        report_note = "Contains measurements and evidence metadata only. Uploaded image contents are excluded."
        if payload["center_cropped"]:
            report_note += " Report dimensions refer to the center-cropped images used for analysis."
        st.caption(report_note)


st.markdown(
    f"""
    <div class="inspect-header">
    <div class="brand-mark" aria-hidden="true"></div>
      <div>
        <div class="brand-title">InspectAI</div>
                <div class="brand-tagline">Evidence-driven visual inspection, powered by OpenCV 5.</div>
                <div class="brand-subtitle">Compare. Investigate. Verify.</div>
      </div>
            <div class="header-status">OpenCV {html.escape(cv2.__version__)} | Local</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="section-kicker">Inspection workflow</div>', unsafe_allow_html=True)

reference_upload = None
test_upload = None
reference_bytes = None
test_bytes = None
reference_column, test_column = st.columns(2)
with reference_column:
    with st.container(border=True):
        st.markdown("### Reference image")
        st.caption("Baseline image for comparison")
        reference_upload = st.file_uploader(
            "Reference file",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            key="reference",
            help="Choose the expected or baseline image.",
        )
        if reference_upload is not None:
            reference_bytes = reference_upload.getvalue()
            st.caption(f"Loaded: {reference_upload.name} | {len(reference_bytes) / 1024:.1f} KB")
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
            st.caption(f"Loaded: {test_upload.name} | {len(test_bytes) / 1024:.1f} KB")

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
        reference_image = decode_image(reference_bytes)
        test_image = decode_image(test_bytes)
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
            f"{reference_width} x {reference_height} px",
            delta=f"Aspect ratio {format_aspect_ratio(reference_width, reference_height)}",
            delta_color="off",
        )
    with geometry_test:
        st.metric(
            "Test dimensions",
            f"{test_width} x {test_height} px",
            delta=f"Aspect ratio {format_aspect_ratio(test_width, test_height)}",
            delta_color="off",
        )

    if aspect_mismatch:
        st.warning(
            "These images have different shapes. Comparing them directly would stretch one image and could make objects look wider or taller. InspectAI will not do that."
        )
        st.warning(
            "Center cropping removes content outside the preview frames for this comparison. It does not change your uploaded originals. Uncheck the option to cancel."
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
            target_image = reference_image if target_aspect_source == "Reference image" else test_image
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
                    f"Preview only: analysis will use {cropped_reference.shape[1]} x "
                    f"{cropped_reference.shape[0]} pixel copies. The original uploads remain unchanged."
                )
                preview_reference, preview_test = st.columns(2)
                with preview_reference:
                    st.markdown("**Reference crop preview**")
                    st.image(cv2.cvtColor(cropped_reference, cv2.COLOR_BGR2RGB), width="stretch")
                with preview_test:
                    st.markdown("**Test crop preview**")
                    st.image(cv2.cvtColor(cropped_test, cv2.COLOR_BGR2RGB), width="stretch")

if uploads_ready and reference_image is not None and test_image is not None:
    analysis_signature = f"{upload_signature}:{crop_enabled}:{target_aspect_source}"
else:
    analysis_signature = upload_signature
if st.session_state.get("analysis_signature") != analysis_signature:
    st.session_state["analysis_signature"] = analysis_signature
    st.session_state["analysis_payload"] = None

images_decoded = reference_image is not None and test_image is not None
crop_ready = not crop_enabled or (cropped_reference is not None and cropped_test is not None)

render_workflow(
    reference_upload is not None,
    test_upload is not None,
    st.session_state.get("analysis_payload") is not None,
)
analyze_clicked = st.button(
    "Analyze Images",
    type="primary",
    width="stretch",
    disabled=not images_decoded or not crop_ready,
    help="Run pixel comparison, translation review, and the independent edge check.",
)

if analyze_clicked and images_decoded and crop_ready:
    try:
        comparison_reference = cropped_reference if crop_enabled else reference_image
        comparison_test = cropped_test if crop_enabled else test_image
        result = inspect_images(comparison_reference, comparison_test)
    except (ValueError, cv2.error) as error:
        if "aspect ratios differ" in str(error).lower():
            st.error(
                "Analysis was not run because the image shapes do not match. Select the center-crop option above to preview compatible copies, or upload images with matching aspect ratios."
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
        "Add a reference image and a test image to enable analysis.</div>",
        unsafe_allow_html=True,
    )
elif st.session_state.get("analysis_payload") is None and not analyze_clicked:
    st.markdown(
        '<div class="empty-state"><div class="empty-title">Inputs ready for review</div>'
        "Select Analyze Images to run the inspection and generate evidence.</div>",
        unsafe_allow_html=True,
    )

analysis_payload = st.session_state.get("analysis_payload")
if analysis_payload is not None:
    render_results(analysis_payload)

st.caption(
    "Pixel and edge measurements are heuristic comparison signals. They do not establish whether a real-world defect is present."
)