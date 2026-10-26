"""SDK-independent helpers for merging and clipping Hough line segments."""

import math

import numpy as np


_EPSILON = 1e-9
_PAIR_BLOCK_SIZE = 256
_MAX_MERGE_SEGMENTS = 4000


def road_roi_polygon(shape):
    """Return the internal trapezoid used to focus line detection on the road."""
    if shape is None or len(shape) < 2:
        raise ValueError("Image shape must contain height and width.")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be positive.")

    last_x, last_y = width - 1, height - 1
    top_y = int(round(last_y * 0.36))
    top_left = int(round(last_x * 0.40))
    top_right = int(round(last_x * 0.60))
    return np.asarray(
        [(top_left, top_y), (top_right, top_y), (last_x, last_y), (0, last_y)],
        dtype=np.int32,
    )


def filter_by_angle(segments, min_angle=18.0, max_angle=58.0):
    """Keep line segments in the internal road-line angle range."""
    minimum, maximum = float(min_angle), float(max_angle)
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError("Line angle bounds must be finite.")
    if not 0 <= minimum <= maximum <= 90:
        raise ValueError("Line angles must satisfy 0 <= min <= max <= 90.")
    values = _as_segments(segments)
    if not len(values):
        return values
    dx = values[:, 2] - values[:, 0]
    dy = values[:, 3] - values[:, 1]
    angles = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))
    return values[(angles >= minimum) & (angles <= maximum)]


def filter_toward_vanishing_point(
    segments, width, height, x_fraction=0.5, y_fraction=0.36, tolerance_fraction=0.07
):
    """Keep segments whose supporting line passes near a road vanishing point."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if not (0 <= x_fraction <= 1 and 0 <= y_fraction <= 1):
        raise ValueError("Vanishing point fractions must be between 0 and 1.")
    if not math.isfinite(tolerance_fraction) or tolerance_fraction < 0:
        raise ValueError("tolerance_fraction must be finite and non-negative.")

    values = _as_segments(segments)
    if not len(values):
        return values
    dx = values[:, 2] - values[:, 0]
    dy = values[:, 3] - values[:, 1]
    lengths = np.hypot(dx, dy)
    vp_x = x_fraction * (width - 1)
    vp_y = y_fraction * (height - 1)
    distance = np.abs(dx * (vp_y - values[:, 1]) - dy * (vp_x - values[:, 0]))
    distance /= np.maximum(lengths, _EPSILON)
    return values[distance <= tolerance_fraction * min(width, height)]


def estimate_vanishing_point(
    segments,
    width,
    height,
    x_bounds=(0.30, 0.70),
    y_bounds=(0.20, 0.58),
    tolerance_fraction=0.035,
    max_candidates=64,
):
    """Estimate a road vanishing point from intersections of long segments.

    Only points in the central upper portion of the road ROI are considered.
    A candidate wins when the greatest total segment length points towards it;
    this makes isolated asphalt cracks less influential than coherent lane edges.
    Returns ``(point, support_mask)`` or ``(None, all_false_mask)``.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if not (0 <= x_bounds[0] < x_bounds[1] <= 1):
        raise ValueError("x_bounds must be increasing image fractions.")
    if not (0 <= y_bounds[0] < y_bounds[1] <= 1):
        raise ValueError("y_bounds must be increasing image fractions.")
    if not math.isfinite(tolerance_fraction) or tolerance_fraction <= 0:
        raise ValueError("tolerance_fraction must be a positive finite number.")

    values = _as_segments(segments)
    empty_mask = np.zeros(len(values), dtype=bool)
    if len(values) < 2:
        return None, empty_mask

    vectors = values[:, 2:4] - values[:, :2]
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    usable = np.flatnonzero(lengths > _EPSILON)
    if len(usable) < 2:
        return None, empty_mask
    longest = usable[np.argsort(-lengths[usable], kind="stable")[:max_candidates]]

    # ax + by + c = 0 for every supporting line.
    starts = values[longest, :2]
    ends = values[longest, 2:4]
    homogeneous_starts = np.column_stack((starts, np.ones(len(longest))))
    homogeneous_ends = np.column_stack((ends, np.ones(len(longest))))
    lines = np.cross(homogeneous_starts, homogeneous_ends)
    first, second = np.triu_indices(len(longest), k=1)
    intersections = np.cross(lines[first], lines[second])
    finite = np.abs(intersections[:, 2]) > _EPSILON
    intersections = intersections[finite]
    if not len(intersections):
        return None, empty_mask
    points = intersections[:, :2] / intersections[:, 2:3]

    minimum_x, maximum_x = x_bounds[0] * (width - 1), x_bounds[1] * (width - 1)
    minimum_y, maximum_y = y_bounds[0] * (height - 1), y_bounds[1] * (height - 1)
    inside = (
        (points[:, 0] >= minimum_x) & (points[:, 0] <= maximum_x)
        & (points[:, 1] >= minimum_y) & (points[:, 1] <= maximum_y)
    )
    points = points[inside]
    if not len(points):
        return None, empty_mask

    midpoints = (values[:, :2] + values[:, 2:4]) * 0.5
    tolerance = tolerance_fraction * min(width, height)
    best_score, best_point, best_mask = 0.0, None, empty_mask
    for point in points:
        distance = np.abs(
            vectors[:, 0] * (point[1] - values[:, 1])
            - vectors[:, 1] * (point[0] - values[:, 0])
        ) / np.maximum(lengths, _EPSILON)
        support = (distance <= tolerance) & (midpoints[:, 1] >= point[1])
        if np.count_nonzero(support) < 2:
            continue
        score = float(np.sum(lengths[support]))
        if score > best_score:
            best_score, best_point, best_mask = score, point, support

    if best_point is None:
        return None, empty_mask
    return (float(best_point[0]), float(best_point[1])), best_mask


def clip_above_horizon(segments, horizon_y):
    """Remove portions above a detected road horizon from line segments."""
    values = _as_segments(segments)
    if not len(values):
        return values
    horizon_y = float(horizon_y)
    result = []
    for x1, y1, x2, y2 in values:
        if y1 < horizon_y and y2 < horizon_y:
            continue
        if (y1 < horizon_y) != (y2 < horizon_y):
            ratio = (horizon_y - y1) / (y2 - y1)
            crossing_x = x1 + ratio * (x2 - x1)
            if y1 < horizon_y:
                x1, y1 = crossing_x, horizon_y
            else:
                x2, y2 = crossing_x, horizon_y
        result.append((x1, y1, x2, y2))
    return _as_segments(result)


def line_marking_scores(gray, segments, half_width=3, max_samples=96):
    """Score how strongly each segment is supported by bright road marking pixels.

    Hough detects both painted lane markings and dark asphalt seams.  Sampling
    a narrow strip around a segment lets the road-specific path prefer painted
    markings without changing the generic line detector's output schema.
    Scores are in the inclusive ``[0, 1]`` range.
    """
    image = np.asarray(gray)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("gray must be a non-empty 2D image.")
    if half_width < 0 or max_samples < 2:
        raise ValueError("half_width must be non-negative and max_samples >= 2.")
    values = _as_segments(segments)
    scores = np.zeros(len(values), dtype=np.float64)
    if not len(values):
        return scores

    height, width = image.shape
    offsets = np.arange(-int(half_width), int(half_width) + 1)
    for index, (x1, y1, x2, y2) in enumerate(values):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= _EPSILON:
            continue
        samples = min(int(max_samples), max(16, int(round(length / 4.0))))
        position = np.linspace(0.0, 1.0, samples)
        normal_x, normal_y = -dy / length, dx / length
        xs = np.rint(x1 + position[:, None] * dx + offsets * normal_x).astype(int)
        ys = np.rint(y1 + position[:, None] * dy + offsets * normal_y).astype(int)
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)
        # A Hough segment often lies on one edge of a white painted stripe,
        # so use the brightest value across its small perpendicular strip.
        scores[index] = float(np.mean(np.max(image[ys, xs], axis=1))) / 255.0
    return scores


def _as_segments(segments):
    values = np.asarray(segments)
    if values.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    try:
        values = values.reshape(-1, 4).astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Segments must be an array with four coordinates each.") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("Segment coordinates must be finite.")
    return values


def _canonicalize(segments):
    values = _as_segments(segments)
    if not len(values):
        return values, np.empty((0,), dtype=np.int64)
    reverse = (values[:, 0] > values[:, 2]) | (
        (values[:, 0] == values[:, 2]) & (values[:, 1] > values[:, 3])
    )
    result = values.copy()
    result[reverse, :2] = values[reverse, 2:4]
    result[reverse, 2:4] = values[reverse, :2]
    order = np.lexsort((result[:, 3], result[:, 2], result[:, 1], result[:, 0]))
    return result[order], order


def _ordered_int_segments(segments):
    if not segments:
        return np.empty((0, 4), dtype=np.int32)
    rounded = np.rint(np.asarray(segments, dtype=np.float64)).astype(np.int32)
    lengths = np.hypot(rounded[:, 2] - rounded[:, 0], rounded[:, 3] - rounded[:, 1])
    order = np.lexsort(
        (rounded[:, 3], rounded[:, 2], rounded[:, 1], rounded[:, 0], -lengths)
    )
    return rounded[order]


def _find(parent, item):
    root = item
    while parent[root] != root:
        root = parent[root]
    while parent[item] != item:
        next_item = parent[item]
        parent[item] = root
        item = next_item
    return root


def _union(parent, rank, first, second):
    root_first = _find(parent, first)
    root_second = _find(parent, second)
    if root_first == root_second:
        return
    if rank[root_first] < rank[root_second]:
        root_first, root_second = root_second, root_first
    parent[root_second] = root_first
    if rank[root_first] == rank[root_second]:
        rank[root_first] += 1


def _merge_group(group):
    if len(group) == 1:
        return group[0]

    vectors = group[:, 2:4] - group[:, :2]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) % np.pi
    weights = np.hypot(vectors[:, 0], vectors[:, 1])
    total_weight = float(np.sum(weights))
    sin_sum = float(np.sum(weights * np.sin(2.0 * angles)))
    cos_sum = float(np.sum(weights * np.cos(2.0 * angles)))
    theta = 0.5 * math.atan2(sin_sum, cos_sum)
    direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    midpoints = (group[:, :2] + group[:, 2:4]) * 0.5
    offset = float(np.sum(weights * (midpoints @ normal)) / total_weight)
    origin = offset * normal
    projections = np.concatenate(
        ((group[:, :2] - origin) @ direction, (group[:, 2:4] - origin) @ direction)
    )
    first = origin + float(np.min(projections)) * direction
    second = origin + float(np.max(projections)) * direction
    return np.concatenate((first, second))


def merge_segments(segments, max_gap, merge_distance, angle_tolerance_deg=3):
    """Merge compatible line segments with bounded, order-independent grouping."""
    max_gap = float(max_gap)
    merge_distance = float(merge_distance)
    angle_tolerance_deg = float(angle_tolerance_deg)
    if not math.isfinite(max_gap) or max_gap < 0:
        raise ValueError("max_gap must be a finite non-negative number.")
    if not math.isfinite(merge_distance):
        raise ValueError("merge_distance must be finite.")
    if not math.isfinite(angle_tolerance_deg) or not 0 <= angle_tolerance_deg <= 90:
        raise ValueError("angle_tolerance_deg must be between 0 and 90 degrees.")

    raw = _as_segments(segments)
    if not len(raw):
        return np.empty((0, 4), dtype=np.int32)
    lengths_raw = np.hypot(raw[:, 2] - raw[:, 0], raw[:, 3] - raw[:, 1])
    raw = raw[lengths_raw > _EPSILON]
    if not len(raw):
        return np.empty((0, 4), dtype=np.int32)
    canonical, canonical_order = _canonicalize(raw)
    vectors = canonical[:, 2:4] - canonical[:, :2]
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    if merge_distance <= 0:
        return _ordered_int_segments(list(raw))

    # Bound pairwise working memory; untouched excess segments remain outputs.
    longest_order = np.argsort(-lengths, kind="stable")
    process_order = longest_order[:_MAX_MERGE_SEGMENTS]
    passthrough_order = longest_order[_MAX_MERGE_SEGMENTS:]
    work = canonical[process_order]
    work_original = raw[canonical_order[process_order]]
    work_lengths = lengths[process_order]
    work_vectors = work[:, 2:4] - work[:, :2]
    unit = work_vectors / work_lengths[:, None]
    theta = np.arctan2(work_vectors[:, 1], work_vectors[:, 0]) % np.pi
    count = len(work)

    parent = np.arange(count, dtype=np.int32)
    rank = np.zeros(count, dtype=np.uint8)
    angle_limit = math.radians(float(angle_tolerance_deg)) + _EPSILON
    gap_limit = float(max_gap) + _EPSILON
    distance_limit = float(merge_distance) + _EPSILON
    all_indices = np.arange(count, dtype=np.int32)

    for block_start in range(0, count, _PAIR_BLOCK_SIZE):
        block_end = min(block_start + _PAIR_BLOCK_SIZE, count)
        row_indices = all_indices[block_start:block_end]
        angle_delta = np.abs(theta[row_indices, None] - theta[None, :])
        angle_delta = np.minimum(angle_delta, np.pi - angle_delta)
        upper_triangle = all_indices[None, :] > row_indices[:, None]
        row_offsets, columns = np.nonzero((angle_delta <= angle_limit) & upper_triangle)
        if not len(columns):
            continue

        first_indices = row_indices[row_offsets]
        second_indices = columns
        first_is_longer = work_lengths[first_indices] >= work_lengths[second_indices]
        reference_indices = np.where(first_is_longer, first_indices, second_indices)
        short_indices = np.where(first_is_longer, second_indices, first_indices)

        reference_start = work[reference_indices, :2]
        reference_unit = unit[reference_indices]
        reference_length = work_lengths[reference_indices]
        short_start = work[short_indices, :2]
        short_end = work[short_indices, 2:4]
        start_delta = short_start - reference_start
        end_delta = short_end - reference_start

        start_distance = np.abs(
            reference_unit[:, 0] * start_delta[:, 1]
            - reference_unit[:, 1] * start_delta[:, 0]
        )
        end_distance = np.abs(
            reference_unit[:, 0] * end_delta[:, 1]
            - reference_unit[:, 1] * end_delta[:, 0]
        )
        start_projection = np.sum(start_delta * reference_unit, axis=1)
        end_projection = np.sum(end_delta * reference_unit, axis=1)
        low = np.minimum(start_projection, end_projection)
        high = np.maximum(start_projection, end_projection)
        gap = np.maximum(np.maximum(low - reference_length, -high), 0.0)
        linked = (
            (start_distance <= distance_limit)
            & (end_distance <= distance_limit)
            & (gap <= gap_limit)
        )

        for first_index, second_index in zip(
            first_indices[linked], second_indices[linked]
        ):
            _union(parent, rank, int(first_index), int(second_index))

    roots = np.asarray([_find(parent, index) for index in range(count)])
    groups = {}
    for index, root in enumerate(roots):
        groups.setdefault(int(root), []).append(index)

    merged = []
    for member_indices in groups.values():
        members = work[member_indices]
        if len(member_indices) == 1:
            merged.append(work_original[member_indices[0]])
        else:
            merged.append(_merge_group(members))
    if len(passthrough_order):
        merged.extend(raw[canonical_order[passthrough_order]])
    return _ordered_int_segments(merged)


def clip_segments(segments, width, height):
    """Clamp segment endpoints to valid pixel coordinates in an image."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    values = _as_segments(segments).copy()
    if not len(values):
        return np.empty((0, 4), dtype=np.int32)
    values[:, (0, 2)] = np.clip(values[:, (0, 2)], 0, width - 1)
    values[:, (1, 3)] = np.clip(values[:, (1, 3)], 0, height - 1)
    return np.rint(values).astype(np.int32)


def filter_circles_by_edge_support(circles, edge_map, min_support=0.30, samples=72,
                                   radial_tolerance=2):
    """Keep circle candidates whose circumference is supported by image edges.

    ``circles`` is an ``(N, 3)`` array of ``x, y, radius`` candidates and
    ``edge_map`` is a binary edge image. A sample counts as supported when an
    edge lies within ``radial_tolerance`` pixels of the candidate circumference.
    """
    candidates = np.asarray(circles, dtype=np.float64)
    edges = np.asarray(edge_map)
    if candidates.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("circles must have shape (N, 3).")
    if edges.ndim != 2 or not edges.size:
        raise ValueError("edge_map must be a non-empty 2D array.")
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be between 0 and 1.")
    if samples < 8 or radial_tolerance < 0:
        raise ValueError("samples must be >= 8 and radial_tolerance non-negative.")

    angles = np.linspace(0.0, 2.0 * math.pi, int(samples), endpoint=False)
    offsets = np.arange(-int(radial_tolerance), int(radial_tolerance) + 1)
    height, width = edges.shape
    supported = []
    for x, y, radius in candidates:
        if not np.all(np.isfinite((x, y, radius))) or radius <= 0:
            continue
        xs = np.rint(x + np.cos(angles)[:, None] * (radius + offsets)).astype(int)
        ys = np.rint(y + np.sin(angles)[:, None] * (radius + offsets)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        hits = np.zeros((len(angles), len(offsets)), dtype=bool)
        hits[valid] = edges[ys[valid], xs[valid]] != 0
        ratio = float(np.mean(np.any(hits, axis=1)))
        if ratio >= min_support:
            supported.append((x, y, radius, ratio))

    supported.sort(key=lambda item: (-item[3], -item[2], item[1], item[0]))
    return np.asarray([item[:3] for item in supported], dtype=np.float64).reshape(-1, 3)
