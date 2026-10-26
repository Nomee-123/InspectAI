#!/usr/bin/env python3
"""Generate the procedural InspectAI demo pair set.

Everything is drawn with NumPy and OpenCV; no copyrighted imagery is used.
Run from the repository root:

    python samples/make_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIRECTORY = Path(__file__).resolve().parent
WIDTH, HEIGHT = 640, 480


def make_reference() -> np.ndarray:
    """A textured, procedural registration target with distinctive features."""
    x_gradient = np.linspace(0, 1, WIDTH, dtype=np.float32)[None, :, None]
    y_gradient = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    base = (
        np.array([16, 22, 30], dtype=np.float32)
        + x_gradient * np.array([24, 34, 42], dtype=np.float32)
        + y_gradient * np.array([18, 12, 26], dtype=np.float32)
    )
    image = np.clip(base, 0, 255).astype(np.uint8)

    # Panel with a chamfered silhouette.
    panel = np.array(
        [[86, 52], [552, 52], [594, 94], [594, 398], [548, 432], [86, 432], [48, 394], [48, 92]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [panel], (36, 49, 65))
    cv2.polylines(image, [panel], True, (0, 229, 255), 3, cv2.LINE_AA)

    # Structural rectangles, circles and a diagonal rail.
    cv2.rectangle(image, (96, 108), (286, 264), (92, 112, 132), -1)
    cv2.rectangle(image, (96, 108), (286, 264), (160, 184, 204), 3, cv2.LINE_AA)
    cv2.circle(image, (400, 182), 62, (62, 92, 120), -1)
    cv2.circle(image, (400, 182), 62, (255, 179, 0), 3, cv2.LINE_AA)
    cv2.circle(image, (400, 182), 23, (0, 229, 255), -1)
    cv2.line(image, (104, 342), (548, 300), (41, 121, 255), 9, cv2.LINE_AA)
    cv2.line(image, (104, 342), (548, 300), (230, 241, 255), 2, cv2.LINE_AA)

    # Registration-friendly details: bolts, ticks, text and small rectangles.
    for centre in ((120, 128), (262, 128), (120, 244), (262, 244), (520, 360)):
        cv2.circle(image, centre, 9, (225, 235, 245), -1)
        cv2.circle(image, centre, 4, (18, 28, 42), -1)
    for index in range(12):
        x = 92 + index * 38
        height = 12 if index % 3 else 22
        cv2.rectangle(image, (x, 294), (x + 5, 294 + height), (0, 229, 255), -1)
        cv2.putText(
            image,
            f"{index:02d}",
            (x - 2, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (195, 214, 234),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "VISION CORE",
        (104, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (230, 241, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "A7 // REGISTRATION",
        (332, 396),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        (0, 229, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (456, 80), (566, 134), (72, 44, 38), -1)
    cv2.rectangle(image, (456, 80), (566, 134), (255, 82, 82), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        "CAL",
        (480, 113),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 230, 190),
        2,
        cv2.LINE_AA,
    )
    # A slight blur keeps the demo free of sensor-noise resampling artefacts;
    # the structural detail still provides plenty of stable keypoints.
    return cv2.GaussianBlur(image, (3, 3), 0)


def shift(image: np.ndarray, x: float, y: float) -> np.ndarray:
    return cv2.warpAffine(
        image,
        np.float32([[1, 0, x], [0, 1, y]]),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
    )


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    centre = (image.shape[1] / 2, image.shape[0] / 2)
    return cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D(centre, angle, 1.0),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
    )


def make_grid() -> np.ndarray:
    grid = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(0, HEIGHT, 40):
        for x in range(0, WIDTH, 40):
            if ((x // 40) + (y // 40)) % 2 == 0:
                cv2.rectangle(grid, (x, y), (x + 39, y + 39), (212, 222, 232), -1)
    # Deliberately perfectly periodic (80 px period, no unique landmark) so a
    # one-period shift cannot be resolved without an assumption.
    return grid


def main() -> int:
    reference = make_reference()
    grid = make_grid()

    brighter = cv2.convertScaleAbs(reference, alpha=1.0, beta=35)

    source = np.float32(
        [[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]]
    )
    target = np.float32([[12, 8], [WIDTH - 11, 6], [WIDTH - 13, HEIGHT - 9], [10, HEIGHT - 8]])
    perspective = cv2.warpPerspective(
        reference,
        cv2.getPerspectiveTransform(source, target),
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
    )

    patched = reference.copy()
    cv2.rectangle(patched, (300, 300), (344, 344), (28, 30, 238), -1)
    cv2.rectangle(patched, (300, 300), (344, 344), (255, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(
        patched,
        "NEW",
        (307, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    files = {
        "reference.png": reference,
        "shifted.png": shift(reference, 11, -7),
        "rotated_15.png": rotate(reference, 15),
        "rotated_90.png": cv2.rotate(reference, cv2.ROTATE_90_CLOCKWISE),
        "rotated_180.png": cv2.rotate(reference, cv2.ROTATE_180),
        "brighter.png": brighter,
        "perspective.png": perspective,
        "patched.png": patched,
        "grid.png": grid,
        "grid_shifted.png": np.roll(grid, 80, axis=1),
    }
    for name, image in files.items():
        path = OUTPUT_DIRECTORY / name
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")

    manifest = {
        "generated_by": "samples/make_samples.py",
        "description": "Procedural demo pairs for InspectAI. No copyrighted imagery.",
        "cases": [
            {
                "id": "shifted",
                "label": "Shifted +11 / -7 px",
                "reference": "reference.png",
                "test": "shifted.png",
                "expectation": "translation alignment should lock",
            },
            {
                "id": "rotated_15",
                "label": "Rotated 15 degrees",
                "reference": "reference.png",
                "test": "rotated_15.png",
                "expectation": "arbitrary rotation should lock",
            },
            {
                "id": "rotated_90",
                "label": "Rotated 90 degrees",
                "reference": "reference.png",
                "test": "rotated_90.png",
                "expectation": "orientation fix should lock",
            },
            {
                "id": "rotated_180",
                "label": "Rotated 180 degrees",
                "reference": "reference.png",
                "test": "rotated_180.png",
                "expectation": "orientation fix should lock",
            },
            {
                "id": "brighter",
                "label": "Brighter exposure",
                "reference": "reference.png",
                "test": "brighter.png",
                "expectation": "photometric normalization should reduce the residual",
            },
            {
                "id": "perspective",
                "label": "Perspective warp",
                "reference": "reference.png",
                "test": "perspective.png",
                "expectation": "affine or homography alignment should lock",
            },
            {
                "id": "patched",
                "label": "Pasted patch change",
                "reference": "reference.png",
                "test": "patched.png",
                "expectation": "change region should remain visible",
            },
            {
                "id": "grid_ambiguous",
                "label": "Periodic grid shifted one period (80 px)",
                "reference": "grid.png",
                "test": "grid_shifted.png",
                "expectation": "alignment should remain ambiguous",
            },
        ],
    }
    (OUTPUT_DIRECTORY / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(files)} images and manifest.json to {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
