#!/usr/bin/env python3
"""Run a reproducible synthetic benchmark for the InspectAI core.

The benchmark is intentionally small and deterministic.  It does not claim to
measure real-world inspection accuracy; it records whether the alignment ladder
and comparison stage behave as designed on known synthetic transforms, and how
long each case takes.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --json /tmp/inspectai-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inspectai import build_inspection_report, inspect_images  # noqa: E402


def make_geometric_scene() -> np.ndarray:
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (24, 20), (81, 76), (210, 210, 210), -1)
    cv2.circle(image, (116, 83), 19, (100, 170, 230), -1)
    cv2.line(image, (12, 108), (146, 96), (180, 70, 40), 5)
    return image


def make_perspective_scene() -> np.ndarray:
    image = make_geometric_scene()
    cv2.putText(
        image, "A7", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2
    )
    return image


def make_textured_scene() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 60, (160, 200, 3), dtype=np.uint8)
    for _ in range(35):
        x, y = rng.integers(0, 200), rng.integers(0, 160)
        radius = int(rng.integers(2, 10))
        colour = tuple(int(value) for value in rng.integers(80, 255, 3))
        cv2.circle(image, (int(x), int(y)), radius, colour, -1)
    cv2.rectangle(image, (50, 40), (140, 120), (200, 200, 200), 3)
    return image


def rotate(image: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
    centre = (image.shape[1] / 2, image.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
    )


def build_cases() -> list[dict[str, object]]:
    geometric = make_geometric_scene()
    textured = make_textured_scene()

    shifted = cv2.warpAffine(
        geometric, np.float32([[1, 0, 5], [0, 1, -3]]), (160, 128)
    )
    rotated_5 = rotate(geometric, 5)
    rotated_45 = rotate(geometric, 45)

    combined_matrix = cv2.getRotationMatrix2D((100, 80), 7, 0.94)
    combined_matrix[0, 2] += 9
    combined_matrix[1, 2] -= 5
    combined = cv2.warpAffine(
        textured, combined_matrix, (textured.shape[1], textured.shape[0])
    )

    perspective_scene = make_perspective_scene()
    source = np.float32([[0, 0], [159, 0], [159, 127], [0, 127]])
    target = np.float32([[8, 6], [151, 11], [155, 119], [4, 122]])
    perspective = cv2.warpPerspective(
        perspective_scene, cv2.getPerspectiveTransform(source, target), (160, 128)
    )

    patched = textured.copy()
    cv2.rectangle(patched, (60, 50), (85, 75), (0, 0, 255), -1)

    gradient = np.linspace(0.55, 1.15, textured.shape[1], dtype=np.float32)
    lighting = np.clip(
        textured.astype(np.float32) * gradient[None, :, None] + 12, 0, 255
    ).astype(np.uint8)

    recoloured = textured.copy()
    recoloured[:, :, 0] = np.clip(recoloured[:, :, 0].astype(np.int16) + 50, 0, 255)

    grid = np.zeros((160, 200, 3), dtype=np.uint8)
    for y in range(0, 160, 20):
        for x in range(0, 200, 20):
            if ((x // 20) + (y // 20)) % 2 == 0:
                cv2.rectangle(grid, (x, y), (x + 19, y + 19), (220, 220, 220), -1)
    grid_shift = np.roll(grid, 20, axis=1)

    blank = np.zeros((128, 160, 3), dtype=np.uint8)

    return [
        {
            "name": "identical",
            "description": "Same image; no correction or change expected.",
            "reference": geometric,
            "test": geometric.copy(),
            "expected_status": {"not_needed"},
            "maximum_final_change": 0.05,
        },
        {
            "name": "translation",
            "description": "Five-pixel horizontal and three-pixel vertical shift.",
            "reference": geometric,
            "test": shifted,
            "expected_status": {"applied"},
            "maximum_final_change": 0.5,
        },
        {
            "name": "rotation_5deg",
            "description": "Five-degree rotation about the image centre.",
            "reference": geometric,
            "test": rotated_5,
            "expected_status": {"applied"},
            "maximum_final_change": 0.5,
        },
        {
            "name": "rotation_45deg",
            "description": "Forty-five-degree rotation; tests the richer similarity path.",
            "reference": geometric,
            "test": rotated_45,
            "expected_status": {"applied"},
            "maximum_final_change": 1.0,
        },
        {
            "name": "combined_rotation_scale_shift",
            "description": "Seven-degree rotation, 0.94 scale and translation.",
            "reference": textured,
            "test": combined,
            "expected_status": {"applied"},
            "maximum_final_change": 1.0,
        },
        {
            "name": "perspective",
            "description": "Mild perspective warp with text and shape features.",
            "reference": perspective_scene,
            "test": perspective,
            "expected_status": {"applied"},
            "maximum_final_change": 1.0,
            "method_contains": "homography",
        },
        {
            "name": "patch_only",
            "description": "One local red patch; no geometric correction needed.",
            "reference": textured,
            "test": patched,
            "expected_status": {"not_needed"},
            "maximum_final_change": 3.0,
            "minimum_regions": 1,
        },
        {
            "name": "lighting_gradient",
            "description": "Smooth illumination change without scene motion.",
            "reference": textured,
            "test": lighting,
            "expected_status": {"not_needed"},
            "maximum_final_change": 1.0,
            "expect_photometric": True,
        },
        {
            "name": "colour_only",
            "description": "Blue-channel shift; the change should remain visible.",
            "reference": textured,
            "test": recoloured,
            "expected_status": {"not_needed"},
            "minimum_final_change": 1.0,
            "maximum_final_change": 5.0,
        },
        {
            "name": "repetitive_grid",
            "description": "Phase-ambiguous checkerboard; alignment should stay ambiguous.",
            "reference": grid,
            "test": grid_shift,
            "expected_status": {"ambiguous"},
        },
        {
            "name": "blank",
            "description": "Blank pair; no evidence for alignment or change.",
            "reference": blank,
            "test": blank.copy(),
            "expected_status": {"ambiguous"},
        },
    ]


def run_case(case: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    result = inspect_images(case["reference"], case["test"])
    elapsed = time.perf_counter() - started

    final_change = result.final_comparison.changed_percentage
    region_count = result.changed_regions["region_count"]
    photometric_applied = bool(result.photometric and result.photometric.applied)

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    expected_status = case["expected_status"]
    check(
        "alignment_status",
        result.alignment_status in expected_status,
        {"expected": sorted(expected_status), "actual": result.alignment_status},
    )
    if "maximum_final_change" in case:
        maximum = float(case["maximum_final_change"])
        check(
            "final_change_at_most",
            final_change <= maximum,
            {"maximum": maximum, "actual": final_change},
        )
    if "minimum_final_change" in case:
        minimum = float(case["minimum_final_change"])
        check(
            "final_change_at_least",
            final_change >= minimum,
            {"minimum": minimum, "actual": final_change},
        )
    if "minimum_regions" in case:
        minimum = int(case["minimum_regions"])
        check(
            "region_count_at_least",
            region_count >= minimum,
            {"minimum": minimum, "actual": region_count},
        )
    if "method_contains" in case:
        token = str(case["method_contains"])
        check(
            "alignment_method_contains",
            token in result.alignment_method,
            {"token": token, "actual": result.alignment_method},
        )
    if case.get("expect_photometric"):
        check(
            "photometric_applied",
            photometric_applied,
            photometric_applied,
        )
    check(
        "not_applied_when_not_applied",
        not (result.alignment_status != "applied" and result.alignment_applied),
        {"status": result.alignment_status, "applied": result.alignment_applied},
    )

    report = build_inspection_report(
        case["reference"], case["test"], result
    )
    json.dumps(report, allow_nan=False)

    return {
        "name": case["name"],
        "description": case["description"],
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "alignment_status": result.alignment_status,
        "alignment_method": result.alignment_method,
        "alignment_applied": result.alignment_applied,
        "initial_changed_percentage": result.initial_comparison.changed_percentage,
        "final_changed_percentage": final_change,
        "region_count": region_count,
        "estimated_rotation_degrees": result.estimated_rotation_degrees,
        "estimated_scale": result.estimated_scale,
        "photometric_applied": photometric_applied,
        "elapsed_seconds": elapsed,
    }


def render_table(rows: list[dict[str, object]]) -> str:
    header = (
        f"{'case':30} {'result':6} {'status':10} {'method':28} "
        f"{'initial %':>9} {'final %':>8} {'regions':>7} {'seconds':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name']:30} {'PASS' if row['passed'] else 'FAIL':6} "
            f"{row['alignment_status']:10} {row['alignment_method'][:28]:28} "
            f"{row['initial_changed_percentage']:9.3f} "
            f"{row['final_changed_percentage']:8.3f} "
            f"{row['region_count']:7d} {row['elapsed_seconds']:8.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the full benchmark record to this JSON path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show every individual check, including passing ones.",
    )
    arguments = parser.parse_args()

    rows = [run_case(case) for case in build_cases()]
    print(render_table(rows))

    failures = [row for row in rows if not row["passed"]]
    if failures:
        print("\nFailed checks:")
        for row in failures:
            for check in row["checks"]:
                if not check["passed"] or arguments.verbose:
                    print(
                        f"  {row['name']}: {check['check']} "
                        f"-> {check['detail']}"
                    )
    elif arguments.verbose:
        print("\nAll checks:")
        for row in rows:
            for check in row["checks"]:
                print(f"  {row['name']}: {check['check']} -> {check['detail']}")

    elapsed = sum(float(row["elapsed_seconds"]) for row in rows)
    passed = len(rows) - len(failures)
    print(
        f"\nSummary: {passed}/{len(rows)} cases passed in {elapsed:.2f}s "
        f"(numpy {np.__version__}, OpenCV {cv2.__version__})"
    )

    record = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Synthetic behavioural benchmark for alignment and comparison. "
            "It is not a real-world accuracy measurement or a defect classifier."
        ),
        "versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "cases": rows,
        "summary": {
            "total": len(rows),
            "passed": passed,
            "failed": len(failures),
            "elapsed_seconds": elapsed,
        },
    }
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {arguments.json}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
