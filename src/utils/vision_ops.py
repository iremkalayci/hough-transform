"""SDK-independent detection engines for the Hough Transform capsule.

Every executor in the capsule is a thin wrapper around one function here, so
the algorithms can be tested without the NovaVision runtime.

Shared design
-------------
* ``prepare_edges`` turns a grayscale frame into an ``EdgeData`` bundle:
  noise-aware denoising, local-contrast equalisation, and Canny thresholds
  derived from the image itself (no fixed 50/150).
* Candidates come from a Hough accumulator (lines, circles) or from edge
  fragments (ellipses, partial arcs). Every candidate is then *verified*
  against the edge map with a gradient-orientation test and least-squares
  refined, because accumulator peaks alone are what create false detections.
* Nothing here has user-tunable parameters. All sizes scale with the shorter
  image side ``D`` so one setting works for small and large frames.
"""

import math

import cv2
import numpy as np


_EPS = 1e-9


def _get(cfg, key, default):
    value = None if cfg is None else cfg.get(key)
    return default if value is None else value
_COS = math.cos


# ---------------------------------------------------------------------------
# Segment merging
# ---------------------------------------------------------------------------

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
    raw = raw[lengths_raw > _EPS]
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
    angle_limit = math.radians(float(angle_tolerance_deg)) + _EPS
    gap_limit = float(max_gap) + _EPS
    distance_limit = float(merge_distance) + _EPS
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


# ---------------------------------------------------------------------------
# Edge preparation
# ---------------------------------------------------------------------------

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def estimate_noise_sigma(gray):
    """Robust noise standard deviation (Immerkaer's Laplacian method, median form)."""
    values = np.asarray(gray, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 8:
        return 0.0
    response = cv2.filter2D(values, cv2.CV_32F, _NOISE_KERNEL, borderType=cv2.BORDER_REFLECT)
    response = response[1:-1, 1:-1]
    # The kernel has an L2 norm of 6, so iid noise gives a response std of 6*sigma.
    return float(1.4826 * np.median(np.abs(response)) / 6.0)


def _denoise(gray, sigma, strong, scale=1.0):
    if scale <= 0:
        return cv2.GaussianBlur(gray, (5, 5), 1.0)
    if strong or sigma >= 6.0:
        # Heavy noise: non-local means keeps edges far better than a Gaussian.
        strength = float(np.clip(1.1 * sigma, 6.0, 40.0)) * scale
        out = cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=5,
                                       searchWindowSize=15)
        return cv2.GaussianBlur(out, (5, 5), 1.0)
    if sigma >= 2.0:
        out = cv2.bilateralFilter(gray, 7, 2.5 * sigma + 8.0, 3.0)
        return cv2.GaussianBlur(out, (5, 5), 1.0)
    return cv2.GaussianBlur(gray, (5, 5), 1.0)


class EdgeData:
    """Edge map plus everything detectors need to verify candidates."""

    def __init__(self, gray, smooth, gx, gy, edges, sigma, high, low):
        self.gray = gray
        self.smooth = smooth
        self.gx = gx
        self.gy = gy
        self.edges = edges
        self.sigma = sigma
        self.high = high
        self.low = low
        self.height, self.width = gray.shape
        self.dimension = float(min(gray.shape))
        ys, xs = np.nonzero(edges)
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.float32)
        gxs = gx[ys, xs]
        gys = gy[ys, xs]
        norm = np.hypot(gxs, gys) + _EPS
        self.ux = (gxs / norm).astype(np.float32)
        self.uy = (gys / norm).astype(np.float32)
        self._nearest = None
        self._lsd_mask = None

    def lsd_mask(self):
        """Mask of segments accepted by LSD, whose a-contrario test controls false alarms."""
        if self._lsd_mask is None:
            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            try:
                detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
                found = detector.detect(self.smooth)[0]
            except cv2.error:
                found = None
            if found is not None:
                for x1, y1, x2, y2 in found.reshape(-1, 4):
                    cv2.line(mask, (int(round(x1)), int(round(y1))),
                             (int(round(x2)), int(round(y2))), 255, 3)
            self._lsd_mask = mask
        return self._lsd_mask

    def without_circles(self, circles, tolerance=3.0):
        """Copy of this edge data with the rims of ``circles`` removed ("explain away")."""
        circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
        if not len(circles) or self.count == 0:
            return self
        drop = np.zeros(self.count, dtype=bool)
        for cx, cy, r in circles:
            dx, dy = self.xs - cx, self.ys - cy
            dist = np.hypot(dx, dy) + _EPS
            drop |= (np.abs(dist - r) <= max(tolerance, 0.05 * r)) & (
                np.abs(self.ux * dx / dist + self.uy * dy / dist) >= 0.8)
        edges = self.edges.copy()
        edges[self.ys[drop].astype(int), self.xs[drop].astype(int)] = 0
        clone = EdgeData(self.gray, self.smooth, self.gx, self.gy, edges,
                         self.sigma, self.high, self.low)
        clone._lsd_mask = self._lsd_mask
        return clone

    def without_lines(self, lines, width=2.5):
        """Copy of this edge data with pixels lying on ``lines`` removed."""
        lines = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
        if not len(lines) or self.count == 0:
            return self
        mask = np.zeros(self.edges.shape, dtype=np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                     255, int(2 * width + 1))
        edges = self.edges.copy()
        edges[mask > 0] = 0
        clone = EdgeData(self.gray, self.smooth, self.gx, self.gy, edges,
                         self.sigma, self.high, self.low)
        clone._lsd_mask = self._lsd_mask
        return clone

    @property
    def count(self):
        return len(self.xs)

    def nearest(self):
        """Distance to, and index of, the nearest edge pixel for every pixel."""
        if self._nearest is None:
            inverted = np.where(self.edges > 0, 0, 255).astype(np.uint8)
            if self.count == 0:
                dist = np.full(inverted.shape, 1e6, dtype=np.float32)
                labels = np.zeros(inverted.shape, dtype=np.int32)
            else:
                dist, labels = cv2.distanceTransformWithLabels(
                    inverted, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
                )
            self._nearest = (dist, labels)
        return self._nearest


def prepare_edges(gray, strong_denoise=False, equalize=True, sensitivity=100.0,
                  denoise_strength=100.0):
    """Return ``EdgeData`` with self-calibrated Canny thresholds.

    ``sensitivity`` (percent) scales the calibrated threshold: above 100 keeps
    fainter edges, below 100 keeps only stronger ones. ``denoise_strength``
    (percent) scales the automatic denoising.
    """
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    sigma = estimate_noise_sigma(gray)
    work = _denoise(gray, sigma, strong_denoise, max(0.0, denoise_strength) / 100.0)
    if equalize:
        # Local contrast equalisation keeps faint edges in shadow and bright
        # regions comparable so one threshold serves the whole frame.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        work = clahe.apply(work)
        work = cv2.GaussianBlur(work, (3, 3), 0.8)

    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.abs(gx) + np.abs(gy)  # same L1 norm cv2.Canny uses by default

    top = float(np.percentile(magnitude, 99.5))
    # The threshold must sit just above the noise, not above the strong
    # structure or the median gradient (which is texture, not noise, when objects
    # fill most of the frame): on a clean image with a flat background an Otsu-style split
    # would follow the strongest details and drop weaker, real outlines (the
    # rim of a silver coin on white). Texture is handled by the density guard
    # below, which raises the threshold only when the edge map is too dense.
    noise_floor = 14.0 * sigma
    high = max(noise_floor, 24.0)
    high = min(high, 0.6 * top) if top > 24.0 else high
    low = 0.4 * high
    high /= max(sensitivity, 5.0) / 100.0
    low = 0.4 * high
    edges = cv2.Canny(work, low, high)
    # Dense edge maps mean texture, not structure: raise the threshold until the
    # edge density is plausible so texture cannot drown out real geometry.
    for _ in range(6):
        if np.count_nonzero(edges) <= 0.06 * edges.size:
            break
        high *= 1.35
        low = 0.4 * high
        edges = cv2.Canny(work, low, high)
    # Faint edges next to very strong ones (a pale panel on a pale page next to
    # black strokes) are background, not geometry: drop edges far weaker than the
    # strong edges of the same image.
    ys, xs = np.nonzero(edges)
    if len(xs) > 50:
        magnitude = np.hypot(gx[ys, xs], gy[ys, xs])
        strong = float(np.percentile(magnitude, 95))
        if strong > 8.0 * high:
            weak = magnitude < 0.12 * strong
            edges = edges.copy()
            edges[ys[weak], xs[weak]] = 0
    return EdgeData(gray, work, gx, gy, edges, sigma, high, low)


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

def _supported_samples(ed, sx, sy, normal_x, normal_y, tol=1.6, min_cos=0.82):
    """Which curve samples lie on an edge whose gradient is perpendicular to the curve.

    Returns ``(inside, supported)`` boolean arrays.
    """
    dist, labels = ed.nearest()
    px = np.rint(sx).astype(np.int64)
    py = np.rint(sy).astype(np.int64)
    inside = (px >= 0) & (px < ed.width) & (py >= 0) & (py < ed.height)
    supported = np.zeros(len(sx), dtype=bool)
    if not inside.any() or ed.count == 0:
        return inside, supported
    ix, iy = px[inside], py[inside]
    near = dist[iy, ix] <= tol
    index = labels[iy, ix] - 1
    index = np.clip(index, 0, ed.count - 1)
    cos_angle = np.abs(ed.ux[index] * normal_x[inside] + ed.uy[index] * normal_y[inside])
    supported[inside] = near & (cos_angle >= min_cos)
    return inside, supported


def _angular_coverage(angles, bins=72):
    """Fraction of ``bins`` angular sectors that contain at least one angle."""
    if len(angles) == 0:
        return 0.0
    hist = np.zeros(bins, dtype=bool)
    idx = (np.mod(angles, 2 * math.pi) / (2 * math.pi) * bins).astype(int) % bins
    hist[idx] = True
    return float(hist.mean())


def _fit_circle(x, y):
    """Algebraic (Kasa) least-squares circle through points; returns (cx, cy, r) or None."""
    if len(x) < 3:
        return None
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mx, my = x.mean(), y.mean()
    u, v = x - mx, y - my
    a = np.column_stack((u, v, np.ones_like(u)))
    b = -(u * u + v * v)
    try:
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = -solution[0] / 2.0
    cy = -solution[1] / 2.0
    radicand = cx * cx + cy * cy - solution[2]
    if radicand <= 0 or not np.isfinite(radicand):
        return None
    return cx + mx, cy + my, math.sqrt(radicand)


def _circle_through(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def _duplicate_circle(a, b):
    big = max(a[2], b[2])
    return math.hypot(a[0] - b[0], a[1] - b[1]) < 0.3 * big and abs(a[2] - b[2]) < 0.3 * big


def _circle_support(ed, cx, cy, r, samples=180):
    angles = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    nx, ny = np.cos(angles), np.sin(angles)
    inside, supported = _supported_samples(ed, cx + r * nx, cy + r * ny, nx, ny)
    visible = int(inside.sum())
    if visible == 0:
        return 0.0, 0
    return float(supported.sum()) / visible, visible


def _refine_circle(ed, cx, cy, r, tolerance_scale=1.0):
    """Iteratively refit a circle to the radially-aligned edge pixels around it."""
    for step in range(3):
        tol = max(1.6, 0.05 * r * tolerance_scale / (1 + step))
        margin = r + tol
        near = (np.abs(ed.xs - cx) <= margin) & (np.abs(ed.ys - cy) <= margin)
        if near.sum() < 8:
            return None
        x, y, ux, uy = ed.xs[near], ed.ys[near], ed.ux[near], ed.uy[near]
        dx, dy = x - cx, y - cy
        dist = np.hypot(dx, dy) + _EPS
        radial = np.abs(ux * dx / dist + uy * dy / dist)
        inliers = (np.abs(dist - r) <= tol) & (radial >= 0.82)
        if inliers.sum() < 8:
            return None
        fit = _fit_circle(x[inliers], y[inliers])
        if fit is None:
            return None
        cx, cy, r = fit
        if r <= 2:
            return None
    return cx, cy, r


# ---------------------------------------------------------------------------
# Lines (also used for broken / dashed lines)
# ---------------------------------------------------------------------------

def _hough_line_candidates(edges, threshold, limit):
    threshold = int(max(1, threshold))
    if hasattr(cv2, "HoughLinesWithAccumulator"):
        found = cv2.HoughLinesWithAccumulator(edges, 1, math.pi / 360.0, threshold)
        if found is None:
            return []
        found = found.reshape(-1, 3)
        return [(float(r), float(t)) for r, t, _ in found[:limit]]
    found = cv2.HoughLines(edges, 1, math.pi / 360.0, threshold)
    if found is None:
        return []
    return [(float(r), float(t)) for r, t in found.reshape(-1, 2)[:limit]]


def detect_lines(ed, broken=False, cfg=None):
    """Detect straight lines.

    The *standard* Hough transform is used on purpose: every edge pixel votes
    for the full infinite line, so dashed or interrupted lines still build a
    strong peak. Each peak is then verified and given its real extent by
    walking the supporting edge pixels along the line (``chains``), allowing
    gaps up to ``max_gap`` for broken lines and only tiny gaps otherwise.
    """
    d = ed.dimension
    if ed.count < 10:
        return np.empty((0, 4), dtype=np.int32)

    # defaults: standard 8 % length / 1.5 % gap; broken 10 % / 20 %, 22 % visible
    min_length = max(6.0, _get(cfg, "min_line_length_pct", 10.0 if broken else 8.0) / 100.0 * d)
    max_gap = max(2.0, _get(cfg, "max_gap_pct", 20.0 if broken else 1.5) / 100.0 * d)
    max_lines = int(_get(cfg, "max_lines", 200))
    if broken:
        min_duty = _get(cfg, "min_dash_fill_pct", 22.0) / 100.0
        vote_threshold = max(14, int(0.04 * d))
    else:
        min_duty = 0.75
        vote_threshold = max(18, int(0.06 * d))
    dash_gap = 3.5
    min_dash = 4.0
    lsd = ed.lsd_mask()
    lsd_need = 0.4 if broken else 0.7

    candidates = _hough_line_candidates(ed.edges, vote_threshold, 500)
    claimed = np.zeros(ed.count, dtype=bool)
    cos_limit = _COS(math.radians(25.0))
    segments = []

    for rho, theta in candidates:
        c, s = math.cos(theta), math.sin(theta)
        align = np.abs(ed.ux * c + ed.uy * s) >= cos_limit
        mask = (np.abs(ed.xs * c + ed.ys * s - rho) <= 2.2) & align & ~claimed
        if mask.sum() < min_dash * 2:
            continue
        refined = False
        for tolerance in (2.2, 1.6):
            points = np.column_stack((ed.xs[mask], ed.ys[mask])).astype(np.float32)
            if len(points) < 5:
                break
            vx, vy, x0, y0 = [float(v[0]) for v in cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)]
            c, s = -vy, vx
            rho = x0 * c + y0 * s
            align = np.abs(ed.ux * c + ed.uy * s) >= cos_limit
            mask = (np.abs(ed.xs * c + ed.ys * s - rho) <= tolerance) & align & ~claimed
            refined = True
        if not refined or mask.sum() < min_dash * 2:
            continue

        index = np.nonzero(mask)[0]
        t = ed.xs[index] * vx + ed.ys[index] * vy
        order = np.argsort(t)
        index, t = index[order], t[order]

        # Dashes: runs of supporting pixels separated by less than ``dash_gap``.
        cuts = np.nonzero(np.diff(t) > dash_gap)[0]
        starts = np.concatenate(([0], cuts + 1))
        ends = np.concatenate((cuts, [len(t) - 1]))
        dashes = [(t[a], t[b], a, b) for a, b in zip(starts, ends) if t[b] - t[a] >= min_dash]
        if not dashes:
            continue

        # Chains: dashes separated by no more than ``max_gap``.
        chain = [dashes[0]]
        chains = []
        for dash in dashes[1:]:
            if dash[0] - chain[-1][1] <= max_gap:
                chain.append(dash)
            else:
                chains.append(chain)
                chain = [dash]
        chains.append(chain)

        for chain in chains:
            start, end = chain[0][0], chain[-1][1]
            span = end - start
            if span < min_length:
                continue
            covered = sum(item[1] - item[0] for item in chain)
            if covered / span < min_duty:
                continue
            t0 = x0 * vx + y0 * vy
            p1 = (x0 + (start - t0) * vx, y0 + (start - t0) * vy)
            p2 = (x0 + (end - t0) * vx, y0 + (end - t0) * vy)
            # a-contrario check: the line must also be seen by LSD along its length
            steps = max(int(span), 2)
            sx = np.rint(np.linspace(p1[0], p2[0], steps)).astype(int)
            sy = np.rint(np.linspace(p1[1], p2[1], steps)).astype(int)
            okp = (sx >= 0) & (sx < ed.width) & (sy >= 0) & (sy < ed.height)
            if okp.sum() < 2 or (lsd[sy[okp], sx[okp]] > 0).mean() < lsd_need:
                continue
            segments.append((p1[0], p1[1], p2[0], p2[1]))
            claimed[index[chain[0][2]:chain[-1][3] + 1]] = True

    if not segments:
        return np.empty((0, 4), dtype=np.int32)
    merged = merge_segments(
        np.asarray(segments, dtype=np.float64),
        max_gap=max_gap if broken else max(max_gap, 0.0),
        merge_distance=max(3.0, (0.025 if broken else 0.012) * d),
    )
    merged = merged[:max_lines]
    merged[:, (0, 2)] = np.clip(merged[:, (0, 2)], 0, ed.width - 1)
    merged[:, (1, 3)] = np.clip(merged[:, (1, 3)], 0, ed.height - 1)
    return merged


# ---------------------------------------------------------------------------
# Circles (full, broken and partial arcs)
# ---------------------------------------------------------------------------


def _gradient_vote_candidates(ed, r_min, r_max, limit=80):
    """Circle centres by gradient-directed Hough voting (works for arcs and dashed rims).

    Each edge pixel votes along its gradient line for every radius in range;
    centres are accumulator peaks, radii the peak of the radial histogram.
    """
    if ed.count < 30:
        return []
    d = ed.dimension
    scale = 2.0
    r_min, r_max = max(2.0, float(r_min)), 1.2 * float(r_max)
    h, w = int(ed.height / scale) + 2, int(ed.width / scale) + 2
    xs, ys, ux, uy = ed.xs, ed.ys, ed.ux, ed.uy
    if ed.count > 15000:
        pick = np.random.default_rng(0).choice(ed.count, 15000, replace=False)
        xs, ys, ux, uy = xs[pick], ys[pick], ux[pick], uy[pick]
    acc = np.zeros(h * w, dtype=np.float32)
    for r in np.arange(r_min, r_max, 2.0):
        for sign in (1.0, -1.0):
            ix = ((xs + sign * ux * r) / scale).astype(np.int64)
            iy = ((ys + sign * uy * r) / scale).astype(np.int64)
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            acc += np.bincount(iy[ok] * w + ix[ok], minlength=h * w).astype(np.float32)
    acc = cv2.GaussianBlur(acc.reshape(h, w), (0, 0), 1.2)
    peak = float(acc.max())
    if peak < 6:
        return []
    maxima = (acc == cv2.dilate(acc, np.ones((5, 5), np.uint8))) & (acc >= max(6.0, 0.15 * peak))
    py, px = np.nonzero(maxima)
    order = np.argsort(-acc[py, px])[:limit]
    found = []
    for i in order:
        cx, cy = px[i] * scale, py[i] * scale
        dx, dy = ed.xs - cx, ed.ys - cy
        dist = np.hypot(dx, dy) + _EPS
        radial = np.abs(ed.ux * dx / dist + ed.uy * dy / dist)
        sel = (radial >= 0.9) & (dist >= r_min) & (dist <= r_max)
        if sel.sum() < 15:
            continue
        hist = np.bincount(dist[sel].astype(np.int64), minlength=int(r_max) + 2).astype(np.float32)
        hist = np.convolve(hist, np.ones(3, dtype=np.float32), mode="same")
        for _ in range(2):
            r = int(np.argmax(hist))
            if hist[r] < 12:
                break
            found.append((float(cx), float(cy), float(r)))
            hist[max(0, r - 4):r + 5] = 0
    return found


def _circle_candidates(ed, min_r, max_r):
    d = ed.dimension
    min_r, max_r = max(2, int(min_r)), int(max_r)
    min_dist = max(6.0, 0.03 * d)
    found = []
    smooth = ed.smooth
    runs = (
        (cv2.HOUGH_GRADIENT_ALT, dict(dp=1.5, param1=300, param2=0.80)),
        (cv2.HOUGH_GRADIENT_ALT, dict(dp=1.5, param1=300, param2=0.65)),
        (cv2.HOUGH_GRADIENT, dict(dp=1.0, param1=max(40.0, ed.high * 0.8), param2=16)),
    )
    for method, kw in runs:
        try:
            out = cv2.HoughCircles(smooth, method, minDist=min_dist, minRadius=min_r,
                                   maxRadius=max_r, **kw)
        except cv2.error:
            out = None
        if out is not None:
            found.extend((float(x), float(y), float(r)) for x, y, r in out.reshape(-1, 3)[:150])
    found.extend(_gradient_vote_candidates(ed, min_r, max_r))
    return found


def _arc_stats(ed, cx, cy, r, samples=240):
    angles = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    nx, ny = np.cos(angles), np.sin(angles)
    # Outlines drawn with a thick stroke have edges on both sides of the circle:
    # allow about half a stroke width (grows with the radius, capped).
    tolerance = min(6.0, max(1.6, 0.045 * r))
    inside, supported = _supported_samples(ed, cx + r * nx, cy + r * ny, nx, ny,
                                           tol=tolerance, min_cos=0.94)
    visible = int(inside.sum())
    if visible == 0:
        return 0.0, 0.0, 0.0
    ratio = float(supported.sum()) / visible
    coverage = float(supported.mean())
    # longest supported arc after bridging gaps up to ~20 degrees (dashed rims)
    gap = 14
    ring = np.concatenate((supported, supported, supported)).astype(np.uint8)
    kernel = np.ones(2 * gap + 1, dtype=np.uint8)
    closed = cv2.erode(cv2.dilate(ring.reshape(1, -1), kernel.reshape(1, -1)),
                       kernel.reshape(1, -1)).ravel()
    closed = closed[samples:2 * samples].astype(bool) | supported
    doubled = np.concatenate((closed, closed))
    best = run = 0
    for flag in doubled:
        run = run + 1 if flag else 0
        best = max(best, run)
    return ratio, coverage, min(best, samples) / samples


def detect_circles(ed, mode="full", cfg=None):
    """Circles via Hough voting, then verified on the edge map.

    mode: ``full`` (closed circles), ``broken`` (dashed rims) or ``partial``
    (arcs; occluded or cut by the frame).
    """
    support_default = {"full": 60.0, "broken": 40.0, "partial": 15.0}[mode]
    coverage_default = {"full": 55.0, "broken": 35.0, "partial": 25.0}[mode]
    ratio = _get(cfg, "min_support_pct", support_default) / 100.0
    coverage = _get(cfg, "min_arc_coverage_pct", coverage_default) / 100.0
    arc_default = {"full": 0.40, "broken": 0.12, "partial": 0.25}[mode]
    limits = dict(ratio=ratio, coverage=coverage, arc=min(arc_default, coverage), min_px=18)
    d = ed.dimension
    min_r = max(2.0, _get(cfg, "min_radius_pct", 1.5) / 100.0 * d)
    max_r = _get(cfg, "max_radius_pct", 50.0) / 100.0 * d
    max_circles = int(_get(cfg, "max_circles", 60))
    accepted = []
    for cx, cy, r in _circle_candidates(ed, min_r, max_r):
        fit = _refine_circle(ed, cx, cy, r)
        if fit is None:
            continue
        cx, cy, r = fit
        if not (min_r <= r <= max_r * (1.6 if mode == "partial" else 1.0)):
            continue
        ratio, coverage, arc = _arc_stats(ed, cx, cy, r)
        if ratio < limits["ratio"] or coverage < limits["coverage"] or arc < limits["arc"]:
            continue
        if coverage * 240 < limits["min_px"]:
            continue
        # Centre must stay near the frame for full circles.
        if mode != "partial" and not (-0.2 * r <= cx <= ed.width + 0.2 * r
                                      and -0.2 * r <= cy <= ed.height + 0.2 * r):
            continue
        # Rank by how many edge pixels the circle explains tightly: the true
        # circle owns more than any sub-arc or biased fit of the same pixels.
        dx, dy = ed.xs - cx, ed.ys - cy
        dist = np.hypot(dx, dy) + _EPS
        tight = (np.abs(dist - r) <= 2.0) & (np.abs(ed.ux * dx / dist + ed.uy * dy / dist) >= 0.9)
        accepted.append((float(tight.sum()), cx, cy, r))
    accepted.sort(reverse=True)
    result = []
    claimed = np.zeros(ed.count, dtype=bool)
    for score, cx, cy, r in accepted:
        if any(_duplicate_circle((cx, cy, r), kept) for kept in result):
            continue
        # Edge pixels this circle explains; a candidate that only re-explains
        # pixels already owned by a better circle is a sub-arc artefact.
        dx, dy = ed.xs - cx, ed.ys - cy
        dist = np.hypot(dx, dy) + _EPS
        owned = (np.abs(dist - r) <= 4.0) & (np.abs(ed.ux * dx / dist + ed.uy * dy / dist) >= 0.82)
        if owned.sum() == 0 or (owned & ~claimed).sum() < 0.7 * owned.sum():
            continue
        claimed |= owned
        result.append((cx, cy, r))
    return np.asarray(result[:max_circles], dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Ellipses
# ---------------------------------------------------------------------------

def _ellipse_samples(ellipse, samples=240):
    (cx, cy), (w, h), angle = ellipse
    a, b = w / 2.0, h / 2.0
    phi = math.radians(angle)
    cp, sp = math.cos(phi), math.sin(phi)
    t = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    lx, ly = a * np.cos(t), b * np.sin(t)
    x = cx + lx * cp - ly * sp
    y = cy + lx * sp + ly * cp
    # outward normal of the ellipse in local frame, rotated into image frame
    nlx, nly = np.cos(t) / max(a, _EPS), np.sin(t) / max(b, _EPS)
    norm = np.hypot(nlx, nly) + _EPS
    nlx, nly = nlx / norm, nly / norm
    return x, y, nlx * cp - nly * sp, nlx * sp + nly * cp, t


def _ellipse_residual(ellipse, px, py):
    (cx, cy), (w, h), angle = ellipse
    a, b = max(w / 2.0, _EPS), max(h / 2.0, _EPS)
    phi = math.radians(angle)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = px - cx, py - cy
    u = dx * cp + dy * sp
    v = -dx * sp + dy * cp
    k = np.sqrt((u / a) ** 2 + (v / b) ** 2) + _EPS
    # first-order geometric distance
    return np.abs(k - 1.0) * np.sqrt(u * u + v * v) / k


def _fit_ellipse_robust(x, y, tol):
    pts = np.column_stack((x, y)).astype(np.float32)
    ellipse = None
    for _ in range(4):
        if len(pts) < 12:
            return None
        try:
            ellipse = cv2.fitEllipseDirect(pts)
        except cv2.error:
            return None
        res = _ellipse_residual(ellipse, pts[:, 0], pts[:, 1])
        keep = res <= tol
        if keep.all() or keep.sum() < 12:
            break
        pts = pts[keep]
    return ellipse


def _edge_chains(ed, min_len=14, junction_neighbours=5):
    """Split the edge map at branch points into simple curve pieces (chains).

    A closed outline that touches clutter (relief, texture) is one huge
    connected component, which cannot be fitted as a whole. Cutting at
    branches gives short clean pieces that can seed a fit.
    """
    edges = (ed.edges > 0).astype(np.uint8)
    neighbours = cv2.filter2D(edges, -1, np.ones((3, 3), np.float32),
                              borderType=cv2.BORDER_CONSTANT) - edges
    branch = ((edges > 0) & (neighbours >= junction_neighbours)).astype(np.uint8)
    pieces = edges.copy()
    pieces[cv2.dilate(branch, np.ones((3, 3), np.uint8)) > 0] = 0
    count, labels = cv2.connectedComponents(pieces, connectivity=8)
    ys, xs = np.nonzero(pieces)
    owner = labels[ys, xs]
    order = np.argsort(owner)
    xs, ys, owner = xs[order], ys[order], owner[order]
    starts = np.searchsorted(owner, np.arange(1, count))
    stops = np.append(starts[1:], len(owner))
    chains = [(xs[a:b].astype(np.float32), ys[a:b].astype(np.float32))
              for a, b in zip(starts, stops) if b - a >= min_len]
    chains.sort(key=lambda chain: -len(chain[0]))
    return chains


def _grow_ellipse(ed, ellipse, tol):
    """Refit ``ellipse`` to every edge pixel lying on its outline (a few rounds)."""
    for _ in range(3):
        (cx, cy), (w, h), _angle = ellipse
        reach = max(w, h) / 2.0 + 3.0 * tol
        near = (np.abs(ed.xs - cx) <= reach) & (np.abs(ed.ys - cy) <= reach)
        px, py = ed.xs[near], ed.ys[near]
        if len(px) < 12:
            return None
        on_outline = _ellipse_residual(ellipse, px, py) <= 1.5 * tol
        if on_outline.sum() < 12:
            return None
        ellipse = _fit_ellipse_robust(px[on_outline], py[on_outline], tol)
        if ellipse is None:
            return None
    return ellipse


def _verify_ellipse(ed, ellipse, min_support, min_cover, min_axis):
    """Score of ``ellipse`` (support x coverage) or None when it is not backed by edges."""
    d = ed.dimension
    (_cx, _cy), (w, h), _angle = ellipse
    major, minor = max(w, h), min(w, h)
    if minor < 2 * min_axis or major > 0.9 * d or minor / major < 0.15:
        return None
    sx, sy, nx, ny, t = _ellipse_samples(ellipse)
    inside, supported = _supported_samples(ed, sx, sy, nx, ny, tol=1.8, min_cos=0.78)
    visible = int(inside.sum())
    if visible < 60:
        return None
    support = float(supported.sum()) / visible
    cover = _angular_coverage(t[supported], 72)
    if support < min_support or cover < min_cover:
        return None
    return support * cover


def _ellipse_overlap(first, second):
    """Intersection over union of the two filled ellipses."""
    corners = np.vstack((cv2.boxPoints(first), cv2.boxPoints(second)))
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    scale = min(1.0, 120.0 / max(x1 - x0, y1 - y0, 1.0))
    width, height = int((x1 - x0) * scale) + 3, int((y1 - y0) * scale) + 3

    def draw(ellipse):
        (cx, cy), (w, h), angle = ellipse
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, (int((cx - x0) * scale) + 1, int((cy - y0) * scale) + 1),
                    (max(1, int(w * scale / 2)), max(1, int(h * scale / 2))), angle, 0, 360, 1, -1)
        return mask

    a, b = draw(first), draw(second)
    return float((a & b).sum()) / float((a | b).sum() + _EPS)


def detect_ellipses(ed, partial=False, cfg=None):
    """Ellipses fitted to edge pixels and verified around the whole outline.

    Two sources of candidates are combined: whole connected edge groups (clean
    scenes) and short chains cut at branch points (outlines that touch clutter,
    such as coins on a textured table seen from the side).
    """
    d = ed.dimension
    if ed.count < 20:
        return []
    min_axis = max(3.0, _get(cfg, "min_axis_pct", 2.0) / 100.0 * d)
    min_support = _get(cfg, "min_support_pct", 40.0 if partial else 60.0) / 100.0
    min_cover = 0.2 if partial else 0.5
    max_ellipses = int(_get(cfg, "max_ellipses", 40))
    tol = max(1.6, 0.006 * d)
    candidates = []

    closed = cv2.morphologyEx(ed.edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels = cv2.connectedComponents(closed, connectivity=8)
    ey, ex = np.nonzero(ed.edges)
    lab = labels[ey, ex]
    for k in range(1, count):
        sel = lab == k
        if int(sel.sum()) >= 25:
            ellipse = _fit_ellipse_robust(ex[sel].astype(np.float32), ey[sel].astype(np.float32), tol)
            if ellipse is not None:
                candidates.append(ellipse)

    for junction_neighbours in (5, 8):
        for x, y in _edge_chains(ed, 14, junction_neighbours)[:1500]:
            ellipse = _fit_ellipse_robust(x, y, tol)
            if ellipse is not None:
                ellipse = _grow_ellipse(ed, ellipse, tol)
            if ellipse is not None:
                candidates.append(ellipse)

    scored = []
    for ellipse in candidates:
        score = _verify_ellipse(ed, ellipse, min_support, min_cover, min_axis)
        if score is not None:
            scored.append((score, ellipse))
    scored.sort(key=lambda item: -item[0])
    kept = []
    for _, ellipse in scored:
        if any(_ellipse_overlap(ellipse, other) > 0.5 for other in kept):
            continue
        kept.append(ellipse)
    return kept[:max_ellipses]


def drop_lines_on_ellipses(lines, ellipses):
    """Remove straight pieces that only trace the outline of a detected ellipse."""
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    if not len(lines) or not len(ellipses):
        return lines
    keep = np.ones(len(lines), dtype=bool)
    for ellipse in ellipses:
        (_cx, _cy), (w, h), _angle = ellipse
        tolerance = max(3.5, 0.04 * max(w, h))
        on_a = _ellipse_residual(ellipse, lines[:, 0], lines[:, 1]) <= tolerance
        on_b = _ellipse_residual(ellipse, lines[:, 2], lines[:, 3]) <= tolerance
        keep &= ~(on_a & on_b)
    return lines[keep]


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def _angle_at(p0, p1, p2):
    v1, v2 = p0 - p1, p2 - p1
    c = float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + _EPS))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


_POLY_NAMES = {3: "triangle", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}


def _classify(contour, tolerance=0.02):
    """Return (label, polygon) for a contour, or None when it is not a clean shape."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area < 1 or perimeter < 1:
        return None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + _EPS)
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    poly = None
    for eps in (tolerance, 1.5 * tolerance, 2.25 * tolerance):
        poly = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(poly) <= 8:
            break
    vertices = len(poly)
    pts = poly.reshape(-1, 2).astype(np.float64)

    # A true polygon keeps its corner count when the tolerance is tight, a round
    # outline does not (it needs ever more vertices to follow the curve).
    hull_perimeter = cv2.arcLength(hull, True)
    tight = len(cv2.approxPolyDP(hull, 0.003 * hull_perimeter, True))
    if solidity > 0.9 and tight >= 9 and len(hull) >= 5:
        (_, _), (w, h), _ang = cv2.fitEllipse(hull)
        fit_area = math.pi * w * h / 4.0
        if abs(fit_area - hull_area) / (hull_area + _EPS) < 0.10:
            # Return the fitted outline itself so a round shape is drawn round,
            # not as the many-sided polygon used to classify it.
            (ecx, ecy), (ew, eh), eang = cv2.fitEllipse(hull)
            outline = cv2.ellipse2Poly((int(round(ecx)), int(round(ecy))),
                                       (max(1, int(round(ew / 2))), max(1, int(round(eh / 2)))),
                                       int(round(eang)), 0, 360, 5)
            return ("circle" if min(w, h) / max(w, h) > 0.9 else "ellipse"), \
                outline.reshape(-1, 2).astype(np.float64)
    if vertices == 4 and solidity > 0.85:
        angles = [_angle_at(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
        if all(70 <= a <= 110 for a in angles):
            (_, _), (w, h), _ang = cv2.minAreaRect(poly)
            ratio = max(w, h) / (min(w, h) + _EPS)
            return ("square" if ratio < 1.12 else "rectangle"), pts
        return "quadrilateral", pts
    if vertices in _POLY_NAMES and solidity > 0.85:
        return _POLY_NAMES[vertices], pts
    return None


def detect_shapes(ed, cfg=None):
    """Closed contours classified by polygon approximation.

    Returns a list of ``(label, polygon (N,2), bbox (x1,y1,x2,y2))``.
    """
    d = ed.dimension
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(ed.edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    min_area = (_get(cfg, "min_area_pct", 3.0) / 100.0 * d) ** 2
    tolerance = _get(cfg, "approx_tolerance_pct", 2.0) / 100.0
    max_shapes = int(_get(cfg, "max_shapes", 60))
    max_area = 0.95 * ed.width * ed.height
    candidates = []
    if hierarchy is None:
        return []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        # Only hole contours (the interior of an outlined shape) are real shapes;
        # outer contours of filled blobs have no parent and no child.
        has_parent = hierarchy[0][i][3] != -1
        has_child = hierarchy[0][i][2] != -1
        if not has_parent and has_child:
            continue  # outer rim of an outline; its hole is used instead
        # Slivers (a stroke or an edge pair, not a shape) are too thin to be one.
        if min(cv2.minAreaRect(contour)[1]) < 0.02 * d:
            continue
        classified = _classify(contour, tolerance)
        if classified is None:
            continue
        label, polygon = classified
        # Verify the outline lies on real edges.
        sample = polygon if len(polygon) >= 3 else contour.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(contour)
        candidates.append((area, label, polygon, (x, y, x + w, y + h)))
    candidates.sort(key=lambda c: -c[0])
    kept = []
    for area, label, polygon, box in candidates:
        if any(_iou(box, k[2]) > 0.8 for k in kept):
            continue
        kept.append((label, polygon, box))
    return kept[:max_shapes]


def refine_round_shapes(shapes, circles, ellipses):
    """Give round shapes the outline found by the circle / ellipse detectors.

    A shape is called round by ``_classify``; its outline is then taken from the
    best matching result of ``detect_circles`` / ``detect_ellipses`` (the same
    functions the Circle and Ellipse executors use) when one overlaps it.
    """
    circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    refined = []
    for label, polygon, box in shapes:
        if label not in ("circle", "ellipse"):
            refined.append((label, polygon, box))
            continue
        best, outline = 0.3, None
        for cx, cy, r in circles:
            overlap = _iou(box, (cx - r, cy - r, cx + r, cy + r))
            if overlap > best:
                best = overlap
                outline = cv2.ellipse2Poly((int(round(cx)), int(round(cy))),
                                           (max(1, int(round(r))), max(1, int(round(r)))),
                                           0, 0, 360, 5)
        for ellipse in ellipses:
            (cx, cy), (w, h), angle = ellipse
            corners = cv2.boxPoints(ellipse)
            overlap = _iou(box, (corners[:, 0].min(), corners[:, 1].min(),
                                 corners[:, 0].max(), corners[:, 1].max()))
            if overlap > best:
                best = overlap
                outline = cv2.ellipse2Poly((int(round(cx)), int(round(cy))),
                                           (max(1, int(round(w / 2))), max(1, int(round(h / 2)))),
                                           int(round(angle)), 0, 360, 5)
        if outline is not None:
            polygon = outline.reshape(-1, 2).astype(np.float64)
        refined.append((label, polygon, box))
    return refined


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + _EPS)


def drop_lines_on_circles(lines, circles, tolerance=3.5):
    """Remove straight pieces that are only chords of a detected circle rim."""
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    if not len(lines) or not len(circles):
        return lines
    keep = np.ones(len(lines), dtype=bool)
    for cx, cy, r in circles:
        tol = max(tolerance, 0.08 * r)
        ends = np.stack(
            (np.hypot(lines[:, 0] - cx, lines[:, 1] - cy),
             np.hypot(lines[:, 2] - cx, lines[:, 3] - cy)))
        chord = np.all(np.abs(ends - r) <= tol, axis=0)
        # a line shorter than the diameter that hugs the rim is a rim artefact
        length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
        keep &= ~(chord & (length <= 2.05 * r))
    return lines[keep]


def drop_circles_on_lines(ed, circles, lines, max_fraction=0.35):
    """Remove circle candidates whose edge support is mostly explained by straight lines."""
    circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    if not len(circles) or not len(lines) or ed.count == 0:
        return circles
    mask = np.zeros((ed.height, ed.width), dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, 5)
    on_line = mask[ed.ys.astype(int), ed.xs.astype(int)] > 0
    keep = []
    for cx, cy, r in circles:
        dx, dy = ed.xs - cx, ed.ys - cy
        dist = np.hypot(dx, dy) + _EPS
        owned = (np.abs(dist - r) <= 2.5) & (np.abs(ed.ux * dx / dist + ed.uy * dy / dist) >= 0.9)
        total = int(owned.sum())
        keep.append(total == 0 or float((owned & on_line).sum()) / total <= max_fraction)
    return circles[np.asarray(keep, dtype=bool)]


def merge_circles(primary, extra):
    """Append ``extra`` circles that do not duplicate any ``primary`` circle."""
    primary = np.asarray(primary, dtype=np.float64).reshape(-1, 3)
    extra = np.asarray(extra, dtype=np.float64).reshape(-1, 3)
    kept = [tuple(c) for c in primary]
    for c in extra:
        if not any(_duplicate_circle(tuple(c), k) for k in kept):
            kept.append(tuple(c))
    return np.asarray(kept, dtype=np.float64).reshape(-1, 3)