"""SDK-independent helpers for Hough line filtering and segment merging."""

import math

import numpy as np


_EPSILON = 1e-9
_PAIR_BLOCK_SIZE = 256
_MAX_MERGE_SEGMENTS = 4000


def _image_size(shape):
    if shape is None or len(shape) < 2:
        raise ValueError("Image shape must contain height and width.")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be positive.")
    return height, width


def _validate_roi(left, top, right, bottom, top_inset):
    values = (left, top, right, bottom, top_inset)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("ROI percentages must be finite numbers.")
    left, top, right, bottom, top_inset = map(float, values)
    if not (0 <= left < right <= 100):
        raise ValueError("ROI horizontal bounds must satisfy 0 <= left < right <= 100.")
    if not (0 <= top < bottom <= 100):
        raise ValueError("ROI vertical bounds must satisfy 0 <= top < bottom <= 100.")
    if not 0 <= top_inset <= 50:
        raise ValueError("ROI top inset must be between 0 and 50 percent.")
    return left, top, right, bottom, top_inset


def is_full_roi(left, top, right, bottom, top_inset=0):
    """Return whether the ROI settings keep the complete image rectangle."""
    return (
        float(left) == 0.0
        and float(top) == 0.0
        and float(right) == 100.0
        and float(bottom) == 100.0
        and float(top_inset) == 0.0
    )


def roi_polygon(shape, left, top, right, bottom, top_inset=0):
    """Return an int32 ROI polygon; percentages describe the kept image area."""
    height, width = _image_size(shape)
    left, top, right, bottom, top_inset = _validate_roi(
        left, top, right, bottom, top_inset
    )

    x_left = int(round(left * (width - 1) / 100.0))
    x_right = int(round(right * (width - 1) / 100.0))
    y_top = int(round(top * (height - 1) / 100.0))
    y_bottom = int(round(bottom * (height - 1) / 100.0))
    if x_left >= x_right or y_top >= y_bottom:
        raise ValueError("ROI is empty at this image size.")

    inset = int(round((x_right - x_left) * top_inset / 100.0))
    polygon = np.asarray(
        [
            (x_left + inset, y_top),
            (x_right - inset, y_top),
            (x_right, y_bottom),
            (x_left, y_bottom),
        ],
        dtype=np.int32,
    )
    x = polygon[:, 0].astype(np.float64)
    y = polygon[:, 1].astype(np.float64)
    area_twice = float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if abs(area_twice) <= _EPSILON:
        raise ValueError("ROI is empty at this image size.")
    return polygon


def roi_mask(shape, left, top, right, bottom, top_inset=0):
    """Rasterize the kept ROI as a uint8 mask containing 0 and 255 values."""
    height, width = _image_size(shape)
    polygon = roi_polygon(shape, left, top, right, bottom, top_inset)
    if is_full_roi(left, top, right, bottom, top_inset):
        return np.full((height, width), 255, dtype=np.uint8)

    yy, xx = np.ogrid[:height, :width]
    inside = np.ones((height, width), dtype=bool)
    px = polygon[:, 0]
    py = polygon[:, 1]
    signed_area = np.sum(px * np.roll(py, -1) - py * np.roll(px, -1))
    orientation = 1.0 if signed_area >= 0 else -1.0
    for index in range(len(polygon)):
        x1, y1 = float(px[index]), float(py[index])
        x2, y2 = float(px[(index + 1) % len(polygon)]), float(
            py[(index + 1) % len(polygon)]
        )
        cross = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
        inside &= orientation * cross >= -_EPSILON
    return inside.astype(np.uint8) * 255


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


def segment_angles_deg(segments):
    """Return unsigned line angles from the horizontal, in the range [0, 90]."""
    values = _as_segments(segments)
    if not len(values):
        return np.empty((0,), dtype=np.float64)
    dx = values[:, 2] - values[:, 0]
    dy = values[:, 3] - values[:, 1]
    return np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))


def filter_by_angle(segments, min_angle, max_angle):
    """Keep segments whose unsigned angle from horizontal is in the inclusive range."""
    minimum, maximum = float(min_angle), float(max_angle)
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError("Line angle bounds must be finite.")
    if not 0 <= minimum <= maximum <= 90:
        raise ValueError("Line angles must satisfy 0 <= min_angle <= max_angle <= 90.")
    values = _as_segments(segments)
    if not len(values):
        return values
    angles = segment_angles_deg(values)
    keep = (angles >= minimum - _EPSILON) & (angles <= maximum + _EPSILON)
    return values[keep]


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
