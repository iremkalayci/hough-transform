# HoughTransform

Hough-based detection for NovaVision. Every detection is checked against the
edge map, so results stay reliable on noisy, broken and partial edges.

## Executors

| Executor | Finds |
| --- | --- |
| Line Detection | Straight lines |
| Circle Detection | Circles |
| Noise-Robust Detection | Lines and circles in noisy images |
| Broken Line Detection | Dashed lines and dashed circle rims |
| Partial Edge Detection | Lines and circle arcs hidden behind other objects |
| Ellipse Detection | Ellipses |
| Shape Detection | Triangle, square, rectangle, pentagon, hexagon, octagon, circle, ellipse |

Each executor opens with its own settings pre-filled (defaults are in
`src/models/PackageModel.py`). Sizes are percentages of the shorter image side.

## Output

An annotated image and a list of detections with `classLabel`
(`line`, `circle`, `ellipse` or the shape name).

## Layout

```
src/executors/   one small file per executor
src/models/      PackageModel.py
src/utils/       vision_ops.py (algorithms), response.py
tests/           unit tests
```