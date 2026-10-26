# InspectAI

**Evidence-driven visual inspection — OpenCV AI Hackathon 2026.**

InspectAI compares a reference image with a test image. It estimates and
inspects geometric alignment, measures visual differences with more than one
method, records what the agent tried and why, and exports a JSON evidence
report. It does **not** decide that a real-world defect exists, and it does not
produce calibrated defect probabilities.

The project is a local prototype. It has no AWS deployment, database, or
external model dependency in this pass.

## What it does

1. **Decodes and shape-checks the pair.** Matching aspect ratios are required;
   optional center-cropped copies handle different frame shapes without
   stretching the images.
2. **Checks quarter-turn orientation** (0°, 90°, 180°, 270° clockwise) before
   the main ladder. Unless one orientation is clearly better, the candidates
   are reported as ambiguous and no rotation is silently chosen.
3. **Runs an inspectable alignment ladder.**
   - phase correlation for coarse translation;
   - SIFT feature matching with RANSAC for similarity, affine, and homography;
   - `cv2.findTransformECC` refinement;
   - a log-polar fallback when the similarity estimate needs another attempt.
   Every candidate is kept as a step with its matrix, status, reason, metrics,
   and preview. Rejected and ambiguous candidates remain visible in the UI and
   the report.
4. **Normalizes smooth photometric differences** with a robust gain/offset/
   gradient fit. The fit is rejected when it is implausible, and the change can
   still be inspected if normalization is applied.
5. **Compares the aligned pair in three ways.**
   - CIE-Lab ΔE-style colour difference with an adaptive noise threshold.
   - Canny edge agreement with automatically derived thresholds and a pixel
     tolerance.
   - SSIM, implemented locally with OpenCV and NumPy (no scikit-image).
   Method disagreement is reported, not averaged away.
6. **Localizes change regions.** Small and low-contrast components are
   filtered, and surviving components are reported as **change regions** with
   bounding boxes — not as defects.
7. **Records an agent trace.** Every automatic action has `{action, evidence,
   outcome, uncertainty}`, plus a function trace naming the OpenCV/NumPy calls
   used at each stage.
8. **Exports a JSON report** (schema version 2) with dimensions, thresholds,
   measurements, alignment steps, agent iterations, function trace, warnings,
   and limitations. Image pixels are never embedded.

## Honest language

InspectAI uses terms such as **change region**, **alignment applied**,
**alignment ambiguous**, **alignment rejected**, and **no alignment needed**.
It never says "defect detected" or "anomaly confirmed". Percentage differences,
edge similarity, and SSIM are heuristic comparison signals, not calibrated
probabilities. Limitations and uncertainty warnings appear in both the UI and
the JSON report.

## Requirements

- Python 3.10 or newer
- `numpy`
- `opencv-python-headless`
- `streamlit`

The inspection core uses only NumPy and OpenCV. Streamlit is the UI layer; the
core is importable without it.

## Install

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked, call the environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run the app

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

1. Upload a reference image and a test image.
2. Enable center-cropped copies if the images have different aspect ratios.
3. Select **Run inspection**.
4. Review the four result tabs:
   - **Overview** — HUD measurements, final aligned test, difference heatmap,
     and change-region table.
   - **Alignment** — every candidate step, originals vs. scrubbed warp,
     overlay opacity, transform matrix, feature matches, and candidate review.
   - **Comparison methods** — Lab difference, Canny edges, SSIM, and
     photometric normalization.
   - **Evidence & report** — agent trace, function trace, warnings,
     limitations, and a downloadable JSON report.

### OpenCV logo

The header expects a **user-supplied** logo at `assets/opencv_logo.png`.
InspectAI does not draw, trace, or regenerate the OpenCV logo. If the file is
absent, the header shows the text badge **BUILT WITH OpenCV** and prints a
one-time warning in the Streamlit console. See `assets/README.md`.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The suite covers legacy compatibility plus the evidence-agent behaviours:
rotation and 45° alignment, combined rotation/scale/shift, perspective warps,
patch-only change localization, lighting normalization, colour-only change,
repetitive-texture ambiguity, blank inputs, method disagreement, report schema
2, required agent-trace fields, function tracing, and a forbidden-claim guard.

## Run the benchmark

```bash
python benchmarks/run_benchmark.py
python benchmarks/run_benchmark.py --verbose
python benchmarks/run_benchmark.py --json /tmp/inspectai-benchmark.json
```

The benchmark builds deterministic synthetic cases: identical images,
translation, rotations, combined rotation/scale/shift, perspective, a local
patch, a lighting gradient, a colour-only change, a repetitive grid, and blank
inputs. It prints alignment status, method, initial/final changed percentage,
region count, and elapsed time, then exits non-zero if a designed check fails.

The benchmark is a behavioural regression tool, not a real-world accuracy
measurement. Its synthetic scores must not be presented as inspection accuracy.

## Core API

```python
from inspectai import build_inspection_report, decode_image, inspect_images

reference = decode_image(reference_bytes)
test = decode_image(test_bytes)
result = inspect_images(reference, test)
report = build_inspection_report(reference, test, result)
```

`result` contains the comparison measurements, alignment steps, agent trace,
function trace, final matrix, and the aligned images needed by the UI. The JSON
report is safe to serialize with `json.dumps(..., allow_nan=False)` and excludes
image contents.

## Limitations

- **No defect verdict.** A change region is an image-difference measurement.
  Real-world validation, domain thresholds, and a qualified reviewer are still
  required before any maintenance or quality decision.
- **No calibrated confidence.** Phase-correlation response, NCC gains, and
  percentage differences are decision evidence, not probabilities.
- **Repetitive textures and symmetric parts** can make alignment ambiguous.
  InspectAI flags this instead of silently correcting it.
- **Photometric normalization can absorb a real change** that looks like a
  smooth lighting gradient. Review the photometric residual and the normalized
  image, not only the final percentage.
- **Global transforms only.** The ladder handles translation, similarity,
  affine, and perspective changes. It does not model non-rigid deformation.
- **Resizing, cropping, interpolation, focus, compression, and sensor noise**
  can create or hide differences.
- **The report excludes image pixels.** It is evidence metadata, not a
  self-contained image archive.
- **Local only.** No AWS deployment, authentication, persistence, or
  multi-tenant service is included in this pass.

## Project layout

```text
app.py                       Streamlit HUD and inspection inspectors
inspectai/                   Streamlit-free inspection core
  agent.py                   Loop, status decisions, report evidence
  align.py                   Orientation and alignment ladder
  compare.py                 Lab difference, edges, SSIM, photometric fit
  regions.py                 Mask cleanup and component localization
  report.py                  JSON schema version 2
  types.py                   Dataclasses and transform convention
tests/                       Unit and behavioural regression tests
benchmarks/run_benchmark.py  Deterministic synthetic benchmark
assets/README.md             User-supplied logo instructions
```
