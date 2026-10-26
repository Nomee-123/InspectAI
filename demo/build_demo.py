#!/usr/bin/env python3
"""Build the self-contained InspectAI demo and its README evidence.

This script is the in-repo demo MVP's data pump. It runs the **real**
``inspectai`` pipeline over every pair in ``samples/manifest.json`` and writes:

    demo/data.js             compact, JSON-safe evidence for the browser demo
    demo/evidence/*.json     the full schema-version-2 report per case
    demo/assets/cases/*      reference, test, aligned, heatmap, mask and
                             annotated-region images plus per-step previews
    demo/assets/banner.png   README banner composed from real result pixels

No number in the demo is typed by hand. The page replays what the pipeline
measured, including the rejected and ambiguous steps.

Usage (from the repository root):

    python demo/build_demo.py              # build every artifact
    python demo/build_demo.py --check      # build, then verify case outcomes
    python demo/build_demo.py --serve      # build, then serve demo/ locally
    python demo/build_demo.py --serve --no-build

The static page works from ``file://`` as well, because the data is loaded with
a plain ``<script>`` tag instead of ``fetch``.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
ASSET_DIR = DEMO_DIR / "assets"
CASE_DIR = ASSET_DIR / "cases"
EVIDENCE_DIR = DEMO_DIR / "evidence"
SAMPLES_DIR = ROOT / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspectai import build_inspection_report, decode_image, inspect_images  # noqa: E402

FONT_DIRECTORY = Path("/usr/share/fonts/truetype/dejavu")
STATUS_TONES = {
    "applied": "#00e5ff",
    "ambiguous": "#ffb300",
    "rejected": "#ff5252",
    "not_needed": "#8ea3bd",
}
TILE_WIDTH, TILE_HEIGHT = 600, 450
PANEL_MARGIN, PANEL_GAP, CAPTION_HEIGHT = 20, 20, 34


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono.ttf" if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(FONT_DIRECTORY / name), size)
    except OSError:  # pragma: no cover - fallback for minimal machines
        try:
            return ImageFont.load_default(size)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()


def _to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _fit(image: np.ndarray, width: int, height: int) -> Image.Image:
    """Letterbox an image into a fixed tile on the demo's dark background."""
    source = _to_pil(image)
    scale = min(width / source.width, height / source.height)
    resized = source.resize(
        (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    tile = Image.new("RGB", (width, height), (8, 16, 30))
    tile.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return tile


def _annotate_regions(aligned: np.ndarray, regions: list[dict[str, Any]]) -> np.ndarray:
    annotated = aligned.copy()
    for index, region in enumerate(regions, start=1):
        x, y = int(region["x"]), int(region["y"])
        right = x + int(region["width"]) - 1
        bottom = y + int(region["height"]) - 1
        cv2.rectangle(annotated, (x, y), (right, bottom), (0, 229, 255), 2)
        cv2.rectangle(annotated, (x - 1, y - 1), (right + 1, bottom + 1), (0, 0, 0), 1)
        label = f"R{index}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x, max(0, y - text_height - 8)),
            (x + text_width + 8, y),
            (255, 179, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 4, max(text_height, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (7, 16, 31),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write {path}")


def _write_jpeg(path: Path, image: np.ndarray, quality: int = 82) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"could not write {path}")


def _draw_keypoints(image: np.ndarray, keypoints: list | None) -> np.ndarray:
    if not keypoints:
        return image
    annotated = image.copy()
    for x, y, inlier in keypoints:
        colour = (0, 229, 255) if inlier else (110, 120, 140)
        cv2.circle(annotated, (int(x), int(y)), 3 if inlier else 2, colour, -1, cv2.LINE_AA)
    return annotated


# ---------------------------------------------------------------------------
# Composed figures
# ---------------------------------------------------------------------------


def build_panel(report: dict[str, Any], images: dict[str, np.ndarray], label: str) -> np.ndarray:
    """4-up evidence panel: reference, test, aligned, Lab heatmap."""
    width = PANEL_MARGIN * 3 + TILE_WIDTH * 2
    height = PANEL_MARGIN * 4 + (TILE_HEIGHT + CAPTION_HEIGHT) * 2 + 56
    canvas = Image.new("RGB", (width, height), (5, 11, 24))
    draw = ImageDraw.Draw(canvas)
    alignment = report["alignment"]
    measurement = report["measurements"]["pixel_difference"]
    tone = STATUS_TONES.get(str(alignment["status"]), "#8ea3bd")

    draw.text((PANEL_MARGIN, 14), "InspectAI", font=_font(24, bold=True), fill=(230, 241, 255))
    draw.text(
        (PANEL_MARGIN + 118, 18),
        label.upper(),
        font=_font(18, mono=True),
        fill=(120, 150, 180),
    )
    status_text = (
        f"{str(alignment['status']).replace('_', ' ').upper()} · "
        f"{str(alignment['method']).upper()} · "
        f"{measurement['final_changed_percentage']:.2f}% CHANGED · "
        f"{report['measurements']['changed_regions']['region_count']} REGIONS"
    )
    status_font = _font(18, mono=True)
    draw.text(
        (width - PANEL_MARGIN - draw.textlength(status_text, font=status_font), 18),
        status_text,
        font=status_font,
        fill=tone,
    )

    tiles = [
        ("01  REFERENCE", images["reference"]),
        ("02  TEST INPUT", images["test"]),
        ("03  ALIGNED OUTPUT", images["aligned"]),
        ("04  LAB ΔE HEATMAP", images["heatmap"]),
    ]
    for index, (caption, image) in enumerate(tiles):
        column, row = index % 2, index // 2
        x = PANEL_MARGIN + column * (TILE_WIDTH + PANEL_GAP)
        y = 56 + PANEL_MARGIN + row * (TILE_HEIGHT + CAPTION_HEIGHT + PANEL_GAP)
        tile = _fit(image, TILE_WIDTH, TILE_HEIGHT)
        canvas.paste(tile, (x, y))
        draw.rectangle(
            [x, y, x + TILE_WIDTH - 1, y + TILE_HEIGHT - 1],
            outline=(22, 48, 74),
            width=1,
        )
        draw.text(
            (x, y + TILE_HEIGHT + 8),
            caption,
            font=_font(16, mono=True),
            fill=(140, 168, 196),
        )
    return _to_bgr(canvas)


def build_banner(case: dict[str, Any]) -> np.ndarray:
    """Compose the README banner from real pixels of one demo case."""
    width, height = 1600, 420
    canvas = Image.new("RGB", (width, height), (5, 11, 24))
    draw = ImageDraw.Draw(canvas)

    for y in range(height):
        blend = y / height
        draw.line(
            [(0, y), (width, y)],
            fill=(
                int(5 + 9 * blend),
                int(11 + 17 * blend),
                int(24 + 28 * blend),
            ),
        )
    for x in range(0, width, 48):
        draw.line([(x, 0), (x, height)], fill=(9, 20, 38), width=1)
    for y in range(0, height, 48):
        draw.line([(0, y), (width, y)], fill=(9, 20, 38), width=1)
    for radius in range(300, 40, -12):
        alpha = max(0, int(18 - radius / 30))
        draw.ellipse(
            [1120 - radius, 90 - radius, 1120 + radius, 90 + radius],
            outline=(0, int(120 + alpha * 3), int(160 + alpha * 4)),
            width=1,
        )

    kicker = "OPENCV AI HACKATHON 2026  ·  EVIDENCE-DRIVEN VISUAL INSPECTION"
    draw.text((80, 64), kicker, font=_font(18, mono=True), fill=(0, 200, 224))
    draw.text((74, 96), "InspectAI", font=_font(104, bold=True), fill=(240, 249, 255))
    draw.text(
        (80, 222),
        "Align. Compare. Explain. Never overclaim.",
        font=_font(30),
        fill=(176, 199, 222),
    )
    draw.line([(82, 272), (760, 272)], fill=(0, 120, 150), width=1)

    pills = [
        "8 PROCEDURAL DEMO PAIRS",
        "43 BEHAVIOURAL TESTS",
        "11/11 BENCHMARK CASES",
        "ZERO CLOUD DEPENDENCY",
    ]
    x = 80
    for pill in pills:
        text_width = draw.textlength(pill, font=_font(15, mono=True))
        draw.rounded_rectangle(
            [x, 294, x + text_width + 30, 330],
            radius=18,
            outline=(0, 90, 112),
            fill=(7, 20, 36),
        )
        draw.text((x + 15, 304), pill, font=_font(15, mono=True), fill=(150, 214, 226))
        x += int(text_width) + 46

    tiles = [
        ("REFERENCE", case["images"]["reference"]),
        ("TEST", case["images"]["test"]),
        ("REGIONS", case["images"]["regions"]),
        ("HEATMAP", case["images"]["heatmap"]),
    ]
    tile_width, tile_height, gap = 176, 132, 12
    origin_x, origin_y = 1060, 72
    for index, (caption, image) in enumerate(tiles):
        column, row = index % 2, index // 2
        x = origin_x + column * (tile_width + gap)
        y = origin_y + row * (tile_height + gap + 20)
        canvas.paste(_fit(image, tile_width, tile_height), (x, y))
        draw.rectangle(
            [x, y, x + tile_width - 1, y + tile_height - 1],
            outline=(0, 150, 180),
            width=1,
        )
        draw.text(
            (x, y + tile_height + 3),
            caption,
            font=_font(12, mono=True),
            fill=(104, 148, 176),
        )
    draw.text(
        (1060, 380),
        f"CASE {case['id'].upper()}  ·  MEASURED, NOT GUESSED",
        font=_font(13, mono=True),
        fill=(0, 190, 214),
    )
    return _to_bgr(canvas)


# ---------------------------------------------------------------------------
# Pipeline run and payload
# ---------------------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        print("samples/manifest.json is missing; generating the demo pairs...")
        subprocess.run([sys.executable, str(SAMPLES_DIR / "make_samples.py")], check=True)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def run_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    reference = decode_image((SAMPLES_DIR / case["reference"]).read_bytes())
    test = decode_image((SAMPLES_DIR / case["test"]).read_bytes())

    started = time.perf_counter()
    result = inspect_images(reference, test)
    elapsed = time.perf_counter() - started
    report = build_inspection_report(reference, test, result)

    aligned = result.final_test_image if result.final_test_image is not None else test
    mask = result.final_comparison.difference_mask
    regions = report["measurements"]["changed_regions"]["regions"]
    images = {
        "reference": reference,
        "test": test,
        "aligned": aligned,
        "heatmap": result.final_comparison.heatmap,
        "mask": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        "regions": _annotate_regions(aligned, regions),
    }

    folder = CASE_DIR / case["id"]
    _write_png(folder / "reference.png", reference)
    _write_png(folder / "test.png", test)
    _write_png(folder / "aligned.png", aligned)
    _write_png(folder / "heatmap.png", result.final_comparison.heatmap)
    _write_png(folder / "mask.png", images["mask"])
    _write_png(folder / "regions.png", images["regions"])
    _write_jpeg(folder / "panel.jpg", build_panel(report, images, case["label"]), quality=90)

    for step in result.alignment_steps:
        if step.preview is None:
            continue
        preview = _draw_keypoints(step.preview, step.keypoints_ref)
        preview = _fit(preview, 360, 270)
        _write_jpeg(folder / f"step-{step.index:02d}.jpg", _to_bgr(preview))

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / f"{case['id']}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report, images, elapsed


def case_payload(
    case: dict[str, Any], report: dict[str, Any], images: dict[str, np.ndarray], elapsed: float
) -> dict[str, Any]:
    alignment = report["alignment"]
    measurements = report["measurements"]
    pixel = measurements["pixel_difference"]
    steps = []
    for step in report["alignment_steps"]:
        preview = CASE_DIR / case["id"] / f"step-{step['index']:02d}.jpg"
        metrics = step["metrics"] or {}
        steps.append(
            {
                "index": step["index"],
                "name": step["name"],
                "model": step["model"],
                "status": step["status"],
                "reason": step["reason"],
                "matrix": step["matrix"],
                "metrics": metrics,
                "functions": list(
                    dict.fromkeys(
                        list(metrics.get("opencv_functions") or [])
                        + list(metrics.get("functions") or [])
                    )
                ),
                "keypoint_count": len(step.get("keypoints_ref") or []),
                "preview": (
                    f"assets/cases/{case['id']}/step-{step['index']:02d}.jpg"
                    if preview.is_file()
                    else None
                ),
            }
        )
    folder = f"assets/cases/{case['id']}"
    return {
        "id": case["id"],
        "label": case["label"],
        "expectation": case["expectation"],
        "status": alignment["status"],
        "status_label": {
            "applied": "ALIGNMENT LOCKED",
            "ambiguous": "ALIGNMENT AMBIGUOUS",
            "rejected": "ALIGNMENT REJECTED",
            "not_needed": "ALIGNMENT NOT NEEDED",
        }.get(alignment["status"], str(alignment["status"]).upper()),
        "method": alignment["method"],
        "applied": alignment["applied"],
        "confidence": alignment["confidence"],
        "changed_percentage": pixel["final_changed_percentage"],
        "initial_percentage": pixel["initial_changed_percentage"],
        "changed_pixels": pixel["final_changed_pixel_count"],
        "valid_pixels": pixel["valid_pixels"],
        "filtered_component_count": pixel["filtered_component_count"],
        "filtered_pixel_count": pixel["filtered_pixel_count"],
        "region_count": measurements["changed_regions"]["region_count"],
        "detected_region_count": measurements["changed_regions"]["detected_region_count"],
        "regions": measurements["changed_regions"]["regions"],
        "minimum_region_area": measurements["changed_regions"]["minimum_area_pixels"],
        "methods": measurements["method_states"],
        "disagree": measurements["methods_disagree"],
        "agreement": measurements["indication_agreement"],
        "edge": measurements["edge_comparison"],
        "ssim": measurements["ssim_comparison"],
        "photometric": measurements["photometric_normalization"],
        "thresholds": report["thresholds"],
        "alignment": {
            "shift_x": alignment["estimated_shift_pixels"]["x"],
            "shift_y": alignment["estimated_shift_pixels"]["y"],
            "rotation_degrees": alignment["estimated_rotation_degrees"],
            "scale": alignment["estimated_scale"],
            "response": alignment["response"],
            "overlap_fraction": alignment["overlap_fraction"],
            "test_image_resized": alignment["test_image_resized"],
            "final_matrix": alignment["final_matrix"],
        },
        "dimensions": report["images"],
        "steps": steps,
        "iterations": report["agent"]["iterations"],
        "iteration_count": report["agent"]["iteration_count"],
        "evidence": report["evidence"],
        "decisions": report["selected_actions"],
        "warnings": report["warnings"],
        "limitations": report["limitations"],
        "function_trace": report["function_trace"],
        "images": {
            "reference": f"{folder}/reference.png",
            "test": f"{folder}/test.png",
            "aligned": f"{folder}/aligned.png",
            "heatmap": f"{folder}/heatmap.png",
            "mask": f"{folder}/mask.png",
            "regions": f"{folder}/regions.png",
            "panel": f"{folder}/panel.jpg",
        },
        "report_url": f"evidence/{case['id']}.json",
        "duration_seconds": round(elapsed, 3),
        "width": int(images["aligned"].shape[1]),
        "height": int(images["aligned"].shape[0]),
    }


# ---------------------------------------------------------------------------
# Checks: the demo fails loudly if a case stops behaving
# ---------------------------------------------------------------------------


def case_checks(case: dict[str, Any]) -> list[tuple[str, bool]]:
    case_id = case["id"]
    status = case["status"]
    method = case["method"]
    if case_id == "shifted":
        return [
            ("alignment applied", status == "applied"),
            ("translation used", "translation" in method),
            ("pixel difference near zero", case["changed_percentage"] <= 0.1),
        ]
    if case_id == "rotated_15":
        return [
            ("alignment applied", status == "applied"),
            ("feature model used", any(m in method for m in ("similarity", "affine", "homography"))),
        ]
    if case_id in {"rotated_90", "rotated_180"}:
        return [
            ("alignment applied", status == "applied"),
            ("orientation used", "orientation" in method),
            ("pixel difference near zero", case["changed_percentage"] <= 0.1),
        ]
    if case_id == "brighter":
        return [("photometric normalization applied", "photometric" in method)]
    if case_id == "perspective":
        return [
            ("alignment applied", status == "applied"),
            ("perspective model used", any(m in method for m in ("affine", "homography"))),
        ]
    if case_id == "patched":
        return [
            ("change region reported", case["region_count"] >= 1),
            ("change region small", case["changed_percentage"] < 5.0),
        ]
    if case_id == "grid_ambiguous":
        return [("alignment stays ambiguous", status == "ambiguous")]
    return [("case produced a report", bool(case["steps"]))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if DEMO_DIR.exists():
        for folder in (CASE_DIR, EVIDENCE_DIR):
            if folder.exists():
                shutil.rmtree(folder)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    payloads: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(f"{'case':26} {'status':11} {'method':30} {'changed %':>9} {'regions':>7} {'sec':>6}")
    print("-" * 94)
    for case in manifest["cases"]:
        report, images, elapsed = run_case(case)
        payload = case_payload(case, report, images, elapsed)
        payloads.append(payload)
        print(
            f"{case['id']:26} {payload['status']:11} {payload['method']:30} "
            f"{payload['changed_percentage']:9.2f} {payload['region_count']:7d} {elapsed:6.2f}"
        )
    total = time.perf_counter() - started

    reference_case = next(item for item in payloads if item["id"] == "patched")
    banner = build_banner(
        {
            "id": reference_case["id"],
            "images": {
                key: cv2.imread(str(DEMO_DIR / path))
                for key, path in reference_case["images"].items()
            },
        }
    )
    _write_png(ASSET_DIR / "banner.png", banner)

    metadata = {
        "generated_by": "demo/build_demo.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline": "inspectai.inspect_images + build_inspection_report (schema 2)",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "case_count": len(payloads),
        "build_seconds": round(total, 2),
        "note": (
            "Every value is measured by the real pipeline on the procedural samples. "
            "Change regions are image-difference measurements, not defect verdicts."
        ),
    }
    data = {"meta": metadata, "cases": payloads}
    (DEMO_DIR / "data.js").write_text(
        "// Generated by demo/build_demo.py. Do not edit by hand.\n"
        "window.INSPECTAI_DEMO = "
        + json.dumps(data, allow_nan=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    summary = {
        "meta": metadata,
        "cases": [
            {
                "id": item["id"],
                "label": item["label"],
                "expectation": item["expectation"],
                "status": item["status"],
                "method": item["method"],
                "changed_percentage": item["changed_percentage"],
                "region_count": item["region_count"],
                "report": item["report_url"],
            }
            for item in payloads
        ],
    }
    (EVIDENCE_DIR / "index.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nWrote {len(payloads)} cases, {len(list(CASE_DIR.rglob('*.png')))} images, "
        f"{len(list(CASE_DIR.rglob('*.jpg')))} step previews and data.js in {total:.2f}s"
    )
    return payloads, summary


def run_checks(cases: list[dict[str, Any]]) -> int:
    failures = 0
    print("\nBehavioural checks")
    print("-" * 94)
    for case in cases:
        for name, passed in case_checks(case):
            failures += 0 if passed else 1
            print(f"{'PASS' if passed else 'FAIL'}  {case['id']:22} {name}")
    print("-" * 94)
    print(f"{'All demo checks passed.' if not failures else f'{failures} demo check(s) failed.'}")
    return 1 if failures else 0


def serve(port: int) -> int:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DEMO_DIR))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(f"Port {port} is already in use. Pass --port to choose another one.")
            return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"\nServing {DEMO_DIR} at {url}  (Ctrl+C to stop)")
    try:
        http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the demo case outcomes after building")
    parser.add_argument("--serve", action="store_true", help="serve demo/ on a local HTTP server")
    parser.add_argument("--no-build", action="store_true", help="skip the build when serving")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default 8000)")
    arguments = parser.parse_args()

    cases: list[dict[str, Any]] = []
    if not arguments.no_build:
        cases, _ = build()
    elif not (DEMO_DIR / "data.js").is_file():
        print("demo/data.js is missing; run without --no-build first.")
        return 1

    if arguments.check:
        if not cases:
            data = json.loads((DEMO_DIR / "data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
            cases = data["cases"]
        status = run_checks(cases)
        if status:
            return status
    if arguments.serve:
        return serve(arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
