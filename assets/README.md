# Brand assets

InspectAI expects a **user-supplied** OpenCV logo at:

```text
assets/opencv_logo.png
```

The application does **not** draw, trace, or regenerate the OpenCV logo. If the
file is absent, the header falls back to the text badge **BUILT WITH OpenCV** and
prints a one-time warning in the Streamlit console.

Only add the logo file if you have the right to use it. The logo is not included
in this repository.
