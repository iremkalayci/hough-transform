# Hough Transform Capsule — Configuration Design

## Purpose

The capsule exposes OpenCV Hough Transform with one NovaVision input and two task branches:

```text
inputImage → Task
               ├─ Line Detection   → Line Detection Profile
               └─ Circle Detection → Circle Detection Profile
```

The profile dropdown supplies safe starting points for normal users. Selecting `Custom` reveals the corresponding OpenCV controls so advanced users and package-aware LLMs can tune the algorithm without code changes.

## Input and outputs

| Port | SDK type | Meaning |
|---|---|---|
| `inputImage` | `Image` or `List[Image]` | Grayscale, BGR, or BGRA image data. `uint8`, `[0, 255]` floats, and normalized `[0, 1]` floats are accepted. |
| `outputImage` | `Image` or `List[Image]` | Input frame annotated with detected Hough segments or circles. |
| `outputDetections` | `List[Detection]` | SDK detections with `boundingBox`, `classId`, `classLabel`, optional `confidence`, and source-image `imgUID`. |

`confidence` is unset because neither Hough accumulator score is a calibrated probability.

## Task: Line Detection

Implementation: Gaussian denoising → Canny edge detection → two `cv2.HoughLinesP()` passes → collinear segment merging.

| Profile | Intent |
|---|---|
| `General Lines` | Balanced default for ordinary straight segments. |
| `Broken / Dashed Lines` | More sensitive profile for interrupted paint, dashes, and partially occluded lines. |
| `Long, Strong Lines` | Conservative general-line profile for noisy or highly textured images. It does not semantically identify lanes. |
| `Road Lane Markings` | Forward-facing road preset: white/yellow marking mask, lower-road trapezoid, and perspective-consistent lane filtering; retains left/right boundaries and near-vertical dashed centre markings. |
| `Custom` | Shows all parameters below. |

| Custom setting | Validation | Effect |
|---|---|---|
| Canny Low Threshold | integer 0–254 | Weak-edge threshold; lower values retain faint edges and more noise. |
| Canny High Threshold | integer 1–255 | Strong-edge threshold; must be larger than the low threshold. |
| Hough Vote Threshold | integer 1–1000 | Minimum accumulator votes to accept a line. |
| Minimum Line Length | integer ≥1 px | Rejects shorter line segments. |
| Maximum Line Gap | integer ≥0 px | Links interrupted pieces of a line. |
| Segment Merge Distance | number ≥0 px | Merges nearby collinear Hough results; 0 disables merging. |
| Maximum Output Lines | integer 1–1000 | Bounds the response size for dense images. |

## Task: Circle Detection

Implementation: median blur → `cv2.HoughCircles()` → circumference edge-support validation → overlapping-circle suppression.

| Profile | Intent |
|---|---|
| `General Circles` | Balanced default for unknown radii and spacing. |
| `Coins / Circular Parts` | More conservative acceptance for separated round objects. |
| `Custom` | Shows all parameters below. |

| Custom setting | Validation | Effect |
|---|---|---|
| Accumulator Resolution (`dp`) | number >0 | Image-to-accumulator resolution ratio. |
| Minimum Circle Distance | number ≥0 px | Minimum centre distance; 0 uses automatic image-size-aware spacing. |
| Circle Edge Threshold (`param1`) | number >0 | Internal Canny high threshold. |
| Circle Accumulator Threshold (`param2`) | number >0 | Candidate acceptance threshold. |
| Minimum / Maximum Radius | integer ≥0 px | Restricts accepted circle sizes; maximum 0 means unrestricted. |
| Maximum Output Circles | integer 1–1000 | Bounds response size in dense images. |

## Schema and execution contract

- `PackageConfigs.executor` remains the SDK-required `ConfigExecutor` dependent dropdown.
- `LineDetectionExecutor` and `CircleDetectionExecutor` retain independent request/response models and outputs.
- A `Profile` is a nested `dropdownlist` inside the selected task branch.
- The executor reads the selected profile from `request.model.configs.executor.value.value.configs`.
- Presets map to documented parameter sets; `Custom` maps each user-entered validated value directly to OpenCV.
- Every UI control includes `shortDescription` and `longDescription` schema metadata.

## Limits

Hough Transform detects geometric lines and circles, not semantic objects. A general line configuration can correctly return straight texture edges in addition to lane markings. `Road Lane Markings` applies geometric and colour assumptions that are appropriate for a conventional forward-facing road view; it is not selected for arbitrary line images. Selecting a stricter profile or adjusting Custom settings controls the general-line trade-off; semantic lane classification across arbitrary viewpoints, weather, and marking colours requires a separate model.
