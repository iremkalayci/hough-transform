"""SDK-independent helpers for merging and clipping Hough line segments."""

import math

import numpy as np


_EPSILON = 1e-9
_PAIR_BLOCK_SIZE = 256
_MAX_MERGE_SEGMENTS = 4000


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


def filter_road_lane_segments(
    segments,
    width,
    height,
    horizon_ratio=0.35,
    min_bottom_ratio=0.48,
    center_tolerance_ratio=0.10,
    vanishing_left_ratio=0.38,
    vanishing_right_ratio=0.62,
    max_slope_x_per_y=2.8,
):
    """Keep segments compatible with a forward-facing road perspective.

    The function deliberately works only on Hough geometry; the caller is
    responsible for supplying road/marking edges.  A valid lane candidate must
    reach the lower road area, extrapolate towards the central horizon band,
    and have the expected left, right, or near-vertical centre-line direction.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if not 0 < horizon_ratio < min_bottom_ratio < 1:
        raise ValueError("Road horizon and lower-road ratios are invalid.")
    if not 0 < center_tolerance_ratio < 0.5:
        raise ValueError("Center tolerance ratio must be between 0 and 0.5.")
    if not 0 <= vanishing_left_ratio < vanishing_right_ratio <= 1:
        raise ValueError("Vanishing-point band ratios are invalid.")
    if not math.isfinite(max_slope_x_per_y) or max_slope_x_per_y <= 0:
        raise ValueError("Maximum road-line slope must be positive and finite.")

    values = _as_segments(segments)
    if not len(values):
        return np.empty((0, 4), dtype=np.int32)

    centre_x = width * 0.5
    horizon_y = height * float(horizon_ratio)
    lower_road_y = height * float(min_bottom_ratio)
    centre_band = width * float(center_tolerance_ratio)
    vanishing_left = width * float(vanishing_left_ratio)
    vanishing_right = width * float(vanishing_right_ratio)
    accepted = []

    for segment in values:
        x1, y1, x2, y2 = segment
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dy = y2 - y1
        if dy <= _EPSILON or y2 < lower_road_y:
            continue

        slope_x_per_y = (x2 - x1) / dy
        # Horizontal texture edges and near-horizontal cracks are not lanes.
        # Perspective lane boundaries can have a large horizontal change
        # per vertical pixel. The old 1.5 limit rejected real road edges.
        if abs(slope_x_per_y) > max_slope_x_per_y:
            continue

        projected_x = x1 + slope_x_per_y * (horizon_y - y1)
        if not vanishing_left <= projected_x <= vanishing_right:
            continue

        midpoint_x = (x1 + x2) * 0.5
        if midpoint_x < centre_x - centre_band:
            direction_ok = slope_x_per_y < -0.04
        elif midpoint_x > centre_x + centre_band:
            direction_ok = slope_x_per_y > 0.04
        else:
            # Broken centre markings stay close to vertical in a rectified
            # forward-facing view, unlike diagonal road cracks.
            direction_ok = abs(slope_x_per_y) <= 0.30
        if direction_ok:
            accepted.append((x1, y1, x2, y2))

    return _ordered_int_segments(accepted)


def fit_road_lane_boundaries(segments, width, height, horizon_ratio=0.35):
    """Fit one representative left and right boundary from accepted Hough segments.

    Uses length-weighted least squares in x(y) space, then extrapolates each
    side across the visible road region. Near-vertical centre/dashed markings
    are left as individual segments so dashed paint is not drawn as a solid line.
    """
    values = _as_segments(segments)
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if not 0.0 < horizon_ratio < 1.0:
        raise ValueError("Horizon ratio must be between 0 and 1.")
    if not len(values):
        return np.empty((0, 4), dtype=np.int32)

    left_points, left_weights = [], []
    right_points, right_weights = [], []
    centre_segments = []
    centre_x = width * 0.5
    for x1, y1, x2, y2 in values:
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= _EPSILON:
            continue
        midpoint_x = (x1 + x2) * 0.5
        dx_dy = (x2 - x1) / (y2 - y1) if abs(y2 - y1) > _EPSILON else math.copysign(float("inf"), x2 - x1)
        points = [(x1, y1), (x2, y2)]
        if midpoint_x < centre_x and dx_dy < -0.04:
            left_points.extend(points)
            left_weights.extend([length, length])
        elif midpoint_x >= centre_x and dx_dy > 0.04:
            right_points.extend(points)
            right_weights.extend([length, length])
        else:
            centre_segments.append((x1, y1, x2, y2))

    fitted = []
    y_top = height * float(horizon_ratio)
    y_bottom = height - 1
    for points, weights in ((left_points, left_weights), (right_points, right_weights)):
        if len(points) < 2:
            continue
        points = np.asarray(points, dtype=np.float64)
        weights = np.sqrt(np.asarray(weights, dtype=np.float64))
        # Fit x = a*y + b; weighted least squares tolerates multiple short segments.
        design = np.column_stack((points[:, 1], np.ones(len(points))))
        try:
            a, b = np.linalg.lstsq(design * weights[:, None], points[:, 0] * weights, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite((a, b))):
            continue
        x_top = float(a * y_top + b)
        x_bottom = float(a * y_bottom + b)
        # Do not draw extrapolated lines outside the image or backwards across the road.
        if not (-width <= x_top <= 2 * width and -width <= x_bottom <= 2 * width):
            continue
        fitted.append((x_top, y_top, x_bottom, y_bottom))

    return _ordered_int_segments(fitted + centre_segments)


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
