"""Reusable VISION CORE widgets.

Everything displayed here is rendered from the real payload produced by
``ui.pipeline``: stage telemetry, OpenCV function names, step matrices,
keypoint inliers, regions, measurements and the JSON report.  The module only
decides how to present those values.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import streamlit as st

from .pipeline import STAGE_LABELS, STAGE_ORDER
from .theme import CSS

TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "templates"

REDUCED_MOTION_CSS = """
html, body, .stApp, .stApp * , .stApp *::before, .stApp *::after {
  animation-duration: 0.001ms !important;
  animation-iteration-count: 1 !important;
  transition-duration: 0.001ms !important;
  scroll-behavior: auto !important;
}
"""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def inject_theme(reduced: bool = False) -> None:
    """Attach the global VISION CORE stylesheet to the Streamlit page."""
    css = CSS + (REDUCED_MOTION_CSS if reduced else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _template(name: str) -> str:
    return (TEMPLATE_DIRECTORY / name).read_text(encoding="utf-8")


def _json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return encoded.replace("</", "<\\/")


def _fill(
    source: str,
    payload: Any,
    reduced: bool,
    *,
    aspect: str | None = None,
) -> str:
    result = (
        source.replace("__DATA__", _json(payload))
        .replace("__REDUCED__", "true" if reduced else "false")
    )
    if aspect is not None:
        result = result.replace("__ASPECT__", aspect)
    return result


def _component(source: str, *, alt: str | None = None) -> None:
    st.iframe(source, width="stretch", height="content", alt=alt)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _compact(markup: str) -> str:
    """Collapse template indentation so Markdown never treats HTML as a code block."""
    return "".join(part.strip() for part in markup.strip().splitlines())


def _functions(entry: dict[str, Any] | None) -> list[str]:
    if not entry:
        return []
    return [str(item) for item in entry.get("functions", []) if item]


# ---------------------------------------------------------------------------
# Hero, boot and ambient field
# ---------------------------------------------------------------------------


def render_hero(
    capabilities: dict[str, Any],
    logo_uri: str | None,
    *,
    boot: bool,
    reduced: bool,
) -> None:
    """Animated header: particles, reticle, brand and the first-load boot."""
    detectors = [
        name
        for name in ("SIFT", "AKAZE", "ORB")
        if capabilities.get(name.lower())
    ]
    payload = {
        "opencv": capabilities.get("opencv", "?"),
        "numpy": capabilities.get("numpy", "?"),
        "python": capabilities.get("python", "?"),
        "threads": capabilities.get("threads", 0),
        "detectors": detectors,
        "logo_uri": logo_uri or "",
        "boot": bool(boot),
    }
    _component(
        _fill(_template("hero.html"), payload, reduced),
        alt="InspectAI VISION CORE header: runtime versions and an animated particle field.",
    )


# ---------------------------------------------------------------------------
# Pipeline strip and live radar
# ---------------------------------------------------------------------------


def render_pipeline(telemetry: dict[str, dict[str, Any]]) -> None:
    nodes: list[str] = []
    for position, stage in enumerate(STAGE_ORDER, start=1):
        entry = telemetry.get(stage, {})
        state = str(entry.get("state", "pending"))
        status = str(entry.get("status", "") or state)
        functions = _functions(entry)
        if functions:
            function_text = functions[0]
            extra = len(functions) - 1
            if extra > 0:
                function_text = f"{function_text} +{extra}"
            tooltip = f"{status} · {' '.join(functions)}"
        elif state == "pending":
            function_text = "awaiting"
            tooltip = "not run yet"
        elif state == "running":
            function_text = "running"
            tooltip = "stage in progress; no OpenCV call recorded yet"
        else:
            # VERIFY ends with a decision rather than a new OpenCV call, so it
            # has no function list.  Show the real status instead of "awaiting".
            function_text = status
            tooltip = f"{status} · no OpenCV call recorded for this stage"
        nodes.append(
            f"""
            <div class="vc-node {state}" title="{_esc(tooltip)}">
              <div class="vc-node-ring">{position:02d}</div>
              <div class="vc-node-name">{_esc(STAGE_LABELS[stage])}</div>
              <span class="vc-node-fn">{_esc(function_text)}</span>
            </div>
            """
        )
    st.markdown(
        _compact(
            f'<div class="vc-pipeline" role="list" aria-label="Real pipeline stages">{"".join(nodes)}</div>'
        ),
        unsafe_allow_html=True,
    )


def render_radar(
    stage: str | None,
    detail: dict[str, Any] | None,
    *,
    running: bool,
) -> None:
    """A radar panel whose text is driven by the actual progress callback."""
    detail = detail or {}
    stage_label = STAGE_LABELS.get(stage or "", "IDLE").upper()
    functions = [str(item) for item in detail.get("functions", [])]
    phase = str(detail.get("phase", "waiting"))
    if not functions:
        if phase == "finish":
            # VERIFY is a decision stage: it records no new OpenCV call.
            functions = ["no OpenCV call recorded in this stage"]
        elif phase == "cache":
            functions = ["cached measurements replayed"]
        elif running:
            functions = ["stage in progress"]
        else:
            functions = ["waiting for a real progress event"]
    title = "PIPELINE RUNNING" if running else "PIPELINE COMPLETE"
    reason = str(detail.get("reason", "") or detail.get("message", "") or "")
    status = str(detail.get("status", "") or "")
    meta = " · ".join(part for part in (phase, status, reason[:160]) if part)
    st.markdown(
        _compact(
            f"""
        <div class="hud-panel vc-radar-panel" aria-live="polite">
          <div class="vc-radar" aria-hidden="true"></div>
          <div class="vc-radar-text">
            <div class="vc-radar-title">{_esc(title)}</div>
            <div class="vc-radar-stage">{_esc(stage_label)}</div>
            <div class="vc-radar-fn">{_esc(' · '.join(functions[:3]))}</div>
            <div class="vc-radar-fn">{_esc(meta)}</div>
          </div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------


def render_gauges(gauges: Iterable[dict[str, Any]], *, reduced: bool) -> None:
    payload = list(gauges)
    if not payload:
        st.markdown(
            '<div class="vc-empty"><div class="vc-empty-title">no measurements yet</div>'
            '<div class="vc-empty-body">Run an inspection to populate the real gauges.</div></div>',
            unsafe_allow_html=True,
        )
        return
    _component(
        _fill(_template("gauges.html"), payload, reduced),
        alt="Measured changed pixels, alignment overlap, edge similarity and method agreement.",
    )


# ---------------------------------------------------------------------------
# Image panels
# ---------------------------------------------------------------------------


def image_panel(
    title: str,
    uri: str,
    *,
    meta: str = "",
    caption: str = "",
    aspect: float = 4 / 3,
    scan: bool = True,
) -> None:
    if not uri:
        return
    ratio = f"{max(0.2, aspect):.5f}"
    scan_html = '<span class="vc-scan" aria-hidden="true"></span>' if scan else ""
    caption_html = f'<div class="vc-image-caption">{_esc(caption)}</div>' if caption else ""
    st.markdown(
        _compact(
            f"""
        <div class="vc-image-panel">
          <div class="vc-image-head">
            <span class="vc-image-title">{_esc(title)}</span>
            <span class="vc-image-meta">{_esc(meta)}</span>
          </div>
          <div class="vc-img-wrap" style="--vc-aspect:{ratio}">
            <img src="{_esc(uri)}" alt="{_esc(title)}" loading="lazy" />
            {scan_html}
          </div>
          {caption_html}
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Swipe comparison + change regions
# ---------------------------------------------------------------------------


def render_swipe(
    *,
    title: str,
    meta: str,
    top_uri: str,
    bottom_uri: str,
    width: int,
    height: int,
    left_label: str = "reference",
    right_label: str = "aligned test",
    reduced: bool,
) -> None:
    payload = {
        "title": title,
        "meta": meta,
        "top_uri": top_uri,
        "bottom_uri": bottom_uri,
        "left_label": left_label,
        "right_label": right_label,
    }
    aspect = f"{max(1, width)} / {max(1, height)}"
    _component(
        _fill(_template("swipe.html"), payload, reduced, aspect=aspect),
        alt="Interactive wipe comparison between the reference and the aligned test image.",
    )


def render_regions(
    *,
    image_uri: str,
    regions: list[dict[str, Any]],
    width: int,
    height: int,
    minimum_area: int,
    reduced: bool,
) -> None:
    payload = {
        "image_uri": image_uri,
        "width": width,
        "height": height,
        "minimum_area": int(minimum_area),
        "regions": [
            {
                "x": int(region.get("x", 0)),
                "y": int(region.get("y", 0)),
                "width": int(region.get("width", 0)),
                "height": int(region.get("height", 0)),
                "area_pixels": int(region.get("area_pixels", 0)),
            }
            for region in regions
        ],
    }
    aspect = f"{max(1, width)} / {max(1, height)}"
    _component(
        _fill(_template("regions.html"), payload, reduced, aspect=aspect),
        alt="Change-region overlay; select a box or row to read its measured bounds.",
    )


# ---------------------------------------------------------------------------
# REGISTER component
# ---------------------------------------------------------------------------


def render_register(register: dict[str, Any], *, reduced: bool) -> None:
    if not register or not register.get("steps"):
        st.markdown(
            '<div class="vc-empty"><div class="vc-empty-title">no alignment sequence</div>'
            '<div class="vc-empty-body">The pipeline did not record any alignment steps for this pair.</div></div>',
            unsafe_allow_html=True,
        )
        return
    payload = {
        "width": int(register["width"]),
        "height": int(register["height"]),
        "reference_uri": register["reference_uri"],
        "test_uri": register["test_uri"],
        "ref_edges_uri": register["ref_edges_uri"],
        "steps": register["steps"],
        "default_index": int(register.get("default_index", 0)),
    }
    _component(
        _fill(_template("register.html"), payload, reduced),
        alt="Alignment step timeline with markers, matrices and residual gauges.",
    )


# ---------------------------------------------------------------------------
# Agent feed, chips, readouts and report
# ---------------------------------------------------------------------------


def render_feed(trace: Iterable[dict[str, Any]]) -> None:
    entries: list[str] = []
    for position, iteration in enumerate(trace):
        outcome = str(iteration.get("outcome", "note"))
        tone = (
            "rejected"
            if "reject" in outcome.lower()
            else "ambiguous"
            if "ambig" in outcome.lower() or "uncertain" in outcome.lower()
            else ""
        )
        evidence = [str(item) for item in iteration.get("evidence", [])]
        body = " ".join(evidence[:2]) if evidence else "No additional evidence recorded."
        uncertainty = str(iteration.get("uncertainty", "") or "")
        entries.append(
            f"""
            <div class="vc-feed-entry {tone}" style="--i:{position}">
              <div class="vc-feed-head">
                <span class="vc-feed-action">{int(iteration.get('index', position)):02d} · {_esc(iteration.get('action', 'action'))}</span>
                <span class="vc-feed-outcome {tone}">{_esc(outcome)}</span>
              </div>
              <div class="vc-feed-body">{_esc(body)}</div>
              <div class="vc-feed-note">{_esc(uncertainty)}</div>
            </div>
            """
        )
    if not entries:
        st.markdown('<div class="vc-note">No agent iterations were recorded.</div>', unsafe_allow_html=True)
        return
    st.markdown(_compact(f'<div class="vc-feed">{"".join(entries)}</div>'), unsafe_allow_html=True)


def chips(items: Iterable[tuple[str, str]]) -> None:
    rendered = "".join(
        f'<span class="vc-chip {_esc(tone)}">{_esc(text)}</span>'
        for text, tone in items
    )
    st.markdown(_compact(f'<div class="vc-chip-row">{rendered}</div>'), unsafe_allow_html=True)


def readouts(items: Iterable[tuple[str, str]]) -> None:
    rendered = "".join(
        f'<div class="vc-readout"><div class="vc-readout-label">{_esc(label)}</div>'
        f'<div class="vc-readout-value">{_esc(value)}</div></div>'
        for label, value in items
    )
    st.markdown(_compact(f'<div class="vc-readout-grid">{rendered}</div>'), unsafe_allow_html=True)


def report_preview(report_json: str, *, limit: int = 9000) -> None:
    preview = report_json
    suffix = ""
    if len(preview) > limit:
        preview = preview[:limit]
        suffix = "\n\n... preview truncated; download the JSON for the full report."
    st.markdown(
        f'<pre class="vc-report-preview" aria-label="JSON report preview">{_esc(preview)}{_esc(suffix)}</pre>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="vc-empty"><div class="vc-empty-title">{_esc(title)}</div>'
        f'<div class="vc-empty-body">{_esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def kicker(text: str) -> None:
    st.markdown(f'<div class="vc-kicker">{_esc(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="vc-note">{_esc(text)}</div>', unsafe_allow_html=True)


def legend() -> None:
    st.markdown(
        _compact(
            """
        <div class="vc-legend">
          <span class="nice">accepted / locked</span>
          <span class="warn">ambiguous</span>
          <span class="bad">rejected</span>
          <span>heuristics are uncalibrated</span>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )
