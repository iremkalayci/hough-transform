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

Choose the `Task` first, then select a profile. Each profile supplies practical defaults, while `Custom` exposes every relevant OpenCV control. This keeps the first use simple without hardcoding a single configuration for every image source.

| Task | Preset profiles | Custom controls |
|---|---|---|
| Line Detection | `General Lines`, `Broken / Dashed Lines`, `Long, Strong Lines`, `Road Lane Markings` | Canny low/high thresholds, Hough vote threshold, minimum line length, maximum line gap, segment merge distance, and maximum output lines |
| Circle Detection | `General Circles`, `Coins / Circular Parts` | `dp`, minimum circle distance, edge threshold (`param1`), accumulator threshold (`param2`), minimum/maximum radius, and maximum output circles |

Every configuration field has a short and long description in the capsule schema. The descriptions explain both the visual effect and the trade-off—for example, lowering a Hough vote threshold can recover faint segments but also raises false positives. General line profiles process the complete image using Gaussian denoising, Canny edges, and OpenCV's probabilistic Hough transform (`cv2.HoughLinesP()`). A primary Hough pass finds stable segments; a second lower-threshold pass recovers short or interrupted segments such as dashed markings. Compatible collinear segments are merged before the configured output limit is applied. `Road Lane Markings` is a deliberately separate preset: it limits evidence to white/yellow marking pixels in a lower-road trapezoid and retains only perspective-consistent left, right, and near-vertical centre segments.

The complete input/configuration/output contract is documented in [Configuration Design](docs/CONFIGURATION_DESIGN.md).

`outputDetections` is the line/no-line signal: a non-empty list means the detector found one or more line candidates; an empty list means it found none. The annotated image shows those candidates. The three general line profiles do not assume a road, a region of interest, a vanishing point, a line colour, or an orientation; they may therefore detect real straight texture edges in a scene. Choose `Road Lane Markings` for a forward-facing road image. Semantic lane classification in arbitrary viewpoints, weather, or marking colours still requires a dedicated learned model rather than Hough geometry alone.

Circle detection uses fixed defaults (`dp=1.2`, `param1=100`, `param2=30`, automatic minimum center distance, and unrestricted radius bounds). Candidate circles must also have edge pixels supporting enough of their circumference, and strongly overlapping candidates are suppressed. This reduces circles caused by shadows and texture, but it cannot guarantee that every candidate is a real coin; Hough is a geometric detector, not a coin classifier. The package keeps both detection tasks under the single Task dropdown.

The capsule converts supported floating-point frames to `uint8` for OpenCV processing, scaling normalized `[0, 1]` pixels to `[0, 255]`. Annotated output is restored to the input frame's dtype and range. It validates image data and internal detector defaults before calling OpenCV. Detection confidence is left unset because the Hough accumulator is not a calibrated confidence score.

NovaVision supplies its private `sdks.novavision` package in the host runtime. The PyPI distribution named `sdk` is unrelated and is not installed by this capsule.

The NovaVision SDK is not vendored in this repository, so migration behavior for saved Flow nodes created before the configuration was simplified has not been verified. If an existing Flow reports missing configuration values, refresh or recreate the Hough Transform node and select its task again so the current schema is loaded.

`grpcio` is declared as a capsule dependency. If NovaVision still logs `No module named 'grpc'` after deployment, the running environment did not install that dependency (or is reusing an older build); it is a deployment/runtime image issue separate from Hough image processing. Confirm the rebuilt capsule environment includes `grpcio` before treating that warning as resolved.

## Development checks

Run `python -m compileall -q src setup.py` for syntax validation. The NovaVision SDK is supplied by the host environment and is not vendored in this repository, so full capsule loading and image/response integration must be checked in a NovaVision runtime.
