# Hough Transform

NovaVision capsule for detecting line segments with OpenCV `cv2.HoughLinesP()` or circles with `cv2.HoughCircles()`. Connect an image input, choose a task, and read the annotated image and detections outputs.

## Inputs and outputs

| Name | Description |
|---|---|
| `inputImage` | One image or a list of grayscale, BGR, or BGRA images. Accepts `uint8` and floating-point pixel values in `[0, 255]` or normalized `[0, 1]`. |
| `outputImage` | Input image annotated with the detected lines or circles |
| `outputDetections` | List of NovaVision `Detection` objects with the SDK `boundingBox`, `confidence`, `classId`, `classLabel`, and `imgUID` fields |

Each standard NovaVision bounding box contains `left`, `top`, `width`, and `height` in pixels. The `imgUID` comes from the processed image (or the input image model if the decoded frame wrapper omits it); the capsule instance UID is not used as an image UID. Shape-specific endpoints and radii are drawn on `outputImage`; the output schema stays the SDK's `Detection` schema and does not add custom geometry fields.

## Configuration

`Task` is the NovaVision executor selector. It offers two executor branches: `LineDetection` and `CircleDetection`. Each branch has its own request, response, inputs, outputs, and configuration schema, matching the `ConfigExecutor.value` union used by the working TemplateMatching capsule. Line-only and circle-only settings are shown under their respective task branch.

| Parameter | Used by | Default | Meaning |
|---|---|---:|---|
| `CannyLow` | `LineDetection` | 50 | Lower Canny edge threshold (0–255) |
| `CannyHigh` | `LineDetection` | 150 | Upper Canny edge threshold (1–255; must exceed `CannyLow`) |
| `HoughThreshold` | `LineDetection` | 50 | Minimum Hough votes |
| `MinLineLength` | `LineDetection` | 50 | Minimum accepted segment length in pixels |
| `MaxLineGap` | `LineDetection` | 10 | Maximum gap between joined segment points |
| `MergeDistance` | `LineDetection` | 10 | Maximum perpendicular distance in pixels for compatible Hough segments to merge; `0` disables merging |
| `MinAngle` | `LineDetection` | 0 | Inclusive minimum unsigned angle from horizontal (0–90 degrees) |
| `MaxAngle` | `LineDetection` | 90 | Inclusive maximum unsigned angle from horizontal (0–90 degrees) |
| `RoiLeft` | `LineDetection` | 0 | Left edge of the image area kept for line detection, in percent |
| `RoiTop` | `LineDetection` | 0 | Top edge of the image area kept for line detection, in percent |
| `RoiRight` | `LineDetection` | 100 | Right edge of the image area kept for line detection, in percent |
| `RoiBottom` | `LineDetection` | 100 | Bottom edge of the image area kept for line detection, in percent |
| `RoiTopInset` | `LineDetection` | 0 | Inset each top corner toward the center by this percentage of the ROI width (0–50), producing a trapezoid |
| `DP` | `CircleDetection` | 1.2 | Inverse accumulator resolution ratio |
| `MinDist` | `CircleDetection` | 0 (automatic) | Minimum center-to-center distance in pixels; zero uses one eighth of image height |
| `Param1` | `CircleDetection` | 100 | Upper Canny threshold used by the circle detector |
| `Param2` | `CircleDetection` | 30 | Circle accumulator threshold; lower values detect more candidates |
| `MinRadius` | `CircleDetection` | 0 | Minimum radius; zero leaves the lower bound unspecified |
| `MaxRadius` | `CircleDetection` | 0 | Maximum radius; zero leaves the upper bound unspecified |

For line detection, processing order is Gaussian blur, Canny edges, optional ROI mask, `HoughLinesP`, angle filtering, segment merging, angle filtering again, clipping to image bounds, then annotation. The ROI percentages describe the area kept; the top inset narrows its upper edge into a trapezoid. The angle is unsigned from horizontal, so both rising and falling lines have the same angle. Merging groups compatible segments transitively when their angle difference is at most 3 degrees, endpoint distance is within `MergeDistance`, and the gap along the line is within `MaxLineGap`. Setting `MergeDistance` to zero leaves raw Hough segments unmerged. Horizontal parallel features that are within the configured merge distance can become one segment, so lower this value when separate nearby edges must remain distinct.

For a first lane-oriented experiment on the road sample image, try `RoiTop=55`, `RoiTopInset=30`, `MinAngle=20`, `MaxAngle=80`, `MinLineLength=25`, `HoughThreshold=25`, `MaxLineGap=40`, and Canny thresholds 40/120. Place the ROI top at the road horizon or below it; placing it above the horizon includes foliage and can create long false lines. These are starting values, not a guarantee of lane-only detection: Hough detects geometric lines and does not classify road markings. Validate and adjust against the target camera and image size.

The capsule converts supported floating-point frames to `uint8` for OpenCV processing, scaling normalized `[0, 1]` pixels to `[0, 255]`. Annotated output is restored to the input frame's dtype and range. It validates that images are non-empty and that the selected task's parameters are in valid ranges before calling OpenCV. Circle detections use a size-aware minimum distance when `MinDist` is zero and remove strongly overlapping candidates. Detection confidence is left unset because the Hough accumulator is not a calibrated confidence score.

NovaVision supplies its private `sdks.novavision` package in the host runtime. The PyPI distribution named `sdk` is unrelated and is not installed by this capsule.

The NovaVision SDK is not vendored in this repository, so migration behavior for saved Flow nodes created before the new line settings has not been verified. If an existing Flow reports missing configuration values, refresh or recreate the Hough Transform node and select `LineDetection` again so the current configuration schema is loaded.

## Development checks

Run `python -m compileall -q src setup.py` for syntax validation. The NovaVision SDK is supplied by the host environment and is not vendored in this repository, so full capsule loading and image/response integration must be checked in a NovaVision runtime.
