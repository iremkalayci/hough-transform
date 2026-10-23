# Hough Transform

NovaVision capsule for detecting line segments with OpenCV `cv2.HoughLinesP()` or circles with `cv2.HoughCircles()`. Connect an image input, choose a task, and read the annotated image and detections outputs.

## Inputs and outputs

| Name | Description |
|---|---|
| `inputImage` | One image or a list of images (grayscale, BGR, or BGRA; `uint8`) |
| `outputImage` | Input image annotated with the detected lines or circles |
| `outputDetections` | List of NovaVision `Detection` objects with the SDK `boundingBox`, `confidence`, `classId`, `classLabel`, and `imgUID` fields |

Each standard NovaVision bounding box contains `left`, `top`, `width`, and `height` in pixels. The `imgUID` comes from the processed image (or the input image model if the decoded frame wrapper omits it); the capsule instance UID is not used as an image UID. Shape-specific endpoints and radii are drawn on `outputImage`; the output schema stays the SDK's `Detection` schema and does not add custom geometry fields.

## Configuration

`Task` is a dropdown with `LineDetection` and `CircleDetection`. All numeric parameters are available in the request configuration; only the selected task's parameters are used.

| Parameter | Used by | Default | Meaning |
|---|---|---:|---|
| `CannyLow` | Lines | 50 | Lower Canny edge threshold (0–255) |
| `CannyHigh` | Lines | 150 | Upper Canny edge threshold (1–255; must exceed `CannyLow`) |
| `HoughThreshold` | Lines | 50 | Minimum Hough votes |
| `MinLineLength` | Lines | 50 | Minimum accepted segment length in pixels |
| `MaxLineGap` | Lines | 10 | Maximum gap between joined segment points |
| `DP` | Circles | 1.2 | Inverse accumulator resolution ratio |
| `MinDist` | Circles | 0 (automatic) | Minimum center-to-center distance in pixels; zero uses one eighth of image height |
| `Param1` | Circles | 100 | Upper Canny threshold used by the circle detector |
| `Param2` | Circles | 30 | Circle accumulator threshold; lower values detect more candidates |
| `MinRadius` | Circles | 0 | Minimum radius; zero leaves the lower bound unspecified |
| `MaxRadius` | Circles | 0 | Maximum radius; zero leaves the upper bound unspecified |

The capsule validates that images are non-empty `uint8` arrays and that the parameters are in valid ranges before calling OpenCV. A small Gaussian blur is applied before line edge detection and near-collinear overlapping line segments are merged. Circle detections use a size-aware minimum distance when `MinDist` is zero and remove strongly overlapping candidates.

NovaVision supplies its private `sdks.novavision` package in the host runtime. The PyPI distribution named `sdk` is unrelated and is not installed by this capsule.

## Development checks

Run `python -m compileall -q src setup.py` for syntax validation. The NovaVision SDK is supplied by the host environment and is not vendored in this repository, so full capsule loading and image/response integration must be checked in a NovaVision runtime.
