# Hough Transform

NovaVision capsule for detecting line segments with OpenCV `cv2.HoughLinesP()` or circles with `cv2.HoughCircles()`. Connect an image input, choose a task, and read the annotated image and detections outputs.

## Inputs and outputs

| Name | Description |
|---|---|
| `inputImage` | One image or a list of images (grayscale, BGR, or BGRA; `uint8`) |
| `outputImage` | Input image annotated with the detected lines or circles |
| `outputDetections` | List of NovaVision `Detection` objects with `boundingBox`, `confidence`, `classId`, `classLabel`, `imgUID`, and Hough details in `metadata.geometry` |

Line detections include endpoints (`x1`, `y1`, `x2`, `y2`) in `metadata.geometry`; circle detections include center (`centerX`, `centerY`) and `radius`. Each standard NovaVision bounding box contains `left`, `top`, `width`, and `height` in pixels.

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
| `MinDist` | Circles | 20 | Minimum center-to-center distance in pixels |
| `Param1` | Circles | 100 | Upper Canny threshold used by the circle detector |
| `Param2` | Circles | 30 | Circle accumulator threshold; lower values detect more candidates |
| `MinRadius` | Circles | 0 | Minimum radius; zero leaves the lower bound unspecified |
| `MaxRadius` | Circles | 0 | Maximum radius; zero leaves the upper bound unspecified |

The capsule validates that images are non-empty `uint8` arrays and that the parameters are in valid ranges before calling OpenCV.

## Development checks

Run `python -m compileall -q src setup.py` for syntax validation. The NovaVision SDK is supplied by the host environment and is not vendored in this repository, so full capsule loading and image/response integration must be checked in a NovaVision runtime.
