# InspectAI

InspectAI is a local Streamlit prototype for comparing a reference image with a test image. It uses OpenCV 5 to calculate a thresholded pixel-difference mask and heatmap, and it reports the percentage of pixels above the comparison threshold.

## Run locally on Windows

Open PowerShell in the project folder and activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies if needed, then start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser and upload a reference image and a test image.

If PowerShell blocks virtual-environment activation, run the environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover identical and changed images, translation alignment, rotation rejection, blank/ambiguous inputs, brightness-driven method disagreement, dimension and aspect-ratio handling, invalid image bytes, edge-comparison failures, and JSON report serialization/privacy.

## What the agent does

1. Decodes both uploaded files and safely rejects empty or undecodable data.
2. Resizes the test image only when dimensions differ but aspect ratios match; mismatched aspect ratios are rejected for manual review.
3. Runs an initial thresholded pixel comparison.
4. Uses OpenCV phase correlation to estimate translation and its response strength.
5. If the estimate is strong, at least one pixel, and within a conservative range, tests the candidate translation on the overlapping area and applies it only if the changed-pixel percentage improves by a minimum amount.
6. Independently measures tolerant Canny-edge agreement, reports when its change indication disagrees with the pixel-difference indication, and marks edge results as insufficient when too few edges are present.
7. Localizes 8-connected components in the final binary difference mask. The default minimum region area is 25 changed pixels; smaller components are excluded from the displayed region list and reported in the filtered-region count.
8. Displays the initial and final changed-pixel percentages, final mask and heatmap, localized regions, decisions, evidence, and limitations.
9. Offers a downloadable JSON report with dimensions, thresholds, measurements, alignment details, selected actions, evidence, warnings, and localized region metadata. The report excludes image contents.

Changed regions are connected groups of pixels in the final difference mask, not confirmed defects. The minimum-area filter suppresses small components but does not classify the remaining regions. The changed-pixel percentage and edge similarity are simple comparison signals, not defect classifiers or calibrated probabilities. The edge score uses fixed Canny thresholds and a 2-pixel matching tolerance. Lighting, camera position, focus, compression, and sensor noise can affect the measurements. The agent only checks translation; it does not correct rotation, perspective, or non-rigid changes. Phase-correlation response is a matching signal, not a calibrated confidence value. Even same-aspect-ratio resizing can alter image detail, so review the input images and reported evidence. This project currently targets local use and does not include AWS deployment.